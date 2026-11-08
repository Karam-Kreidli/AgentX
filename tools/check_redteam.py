"""Red-team subset for CI (SAFE-6): hostile vessel names must not change any action.

    python tools/check_redteam.py                          # all 5 practice scenarios
    python tools/check_redteam.py --scenario practice_02   # one (repeatable)
    python tools/check_redteam.py --data X.xlsx --out runs/check/redteam

Every scenario runs twice with the LLM off: once as it is, and once with every ship renamed
to a string from tests/redteam/injections.jsonl (override, fake authority, JSON breaking,
bidi and control characters, Arabic, tool-call mimicry, ...). The vessel name is the free
text agents can read: get_vessel_schedule shows it and simulator error messages quote it.
With no LLM nothing may depend on it, so the hostile run must pass these checks:

  - actions.jsonl is identical to the clean run's once the names are mapped back
  - 0 violations, 0 refused calls, 0 invariant breaches, 0 loop exceptions
  - every state-changing action has a reason
  - our own log files carry no raw control, bidi or line-separator characters

The names are changed on the simulator *instance* only (its in-memory vessel table), from
this harness. vendor/ and the workbook are never touched, and targets are every ship, not
chosen by ID. Each run gets its own process, like eval_all.py. Exit code 1 on any failure.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import zip_longest
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "redteam" / "injections.jsonl"
DEFAULT_DATA = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
PRACTICE = [f"practice_0{i}" for i in range(1, 6)]
FIELD = "vessel_name"
OWN_LOGS = ("decisions.jsonl", "negotiation.jsonl", "negotiation.md", "supervisor_feed.txt",
            "ticks.jsonl", "run_meta.json", "summary.json", "llm_calls.jsonl")
# Characters core/log.py must never write raw: C0/C1 controls (except \t \n \r), zero-width,
# bidi controls, and line/paragraph separators (str.splitlines() would split a JSONL line on them).
RAW_BAD = re.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f​-‏"
                     "‪-‮  ⁠-⁤⁦-⁩﻿]")


# --------------------------------------------------------------------------
# Corpus and injection (pure functions, unit-tested in tests/test_check.py)
# --------------------------------------------------------------------------

def load_corpus(path: Path = CORPUS) -> list[str]:
    """The injection strings meant for vessel names."""
    texts = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            if FIELD in row.get("target_fields", []):
                texts.append(row["text"])
    if not texts:
        raise ValueError(f"no {FIELD} injections in {path}")
    return texts


def inject(calls: dict, texts: list[str], offset: int = 0) -> dict[str, str]:
    """Rename every ship in the simulator's vessel table. Returns {hostile name: real name}.
    Names are unique within a scenario, so they map back without ambiguity."""
    back: dict[str, str] = {}
    for i, cid in enumerate(sorted(calls)):
        text = texts[(offset + i) % len(texts)]
        if text in back:
            text = f"{text} #{i}"
        back[text] = calls[cid]["vessel_name"]
        calls[cid]["vessel_name"] = text
    return back


def unmap(obj: Any, back: dict[str, str]) -> Any:
    """obj with every hostile name replaced by the real one (longest first)."""
    if isinstance(obj, str):
        for hostile in sorted(back, key=len, reverse=True):
            if hostile in obj:
                obj = obj.replace(hostile, back[hostile])
        return obj
    if isinstance(obj, dict):
        return {k: unmap(v, back) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unmap(v, back) for v in obj]
    return obj


def _jsonl(path: Path) -> list[Any]:
    # split on "\n" only: the simulator writes names raw (ensure_ascii=False), and a name may
    # hold U+2028, which str.splitlines() would treat as a line break.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _short(rec: Any) -> str:
    if isinstance(rec, dict):
        rec = {k: rec.get(k) for k in ("action", "sim_time", "tool", "arguments") if k in rec}
    s = json.dumps(rec, ensure_ascii=True, default=str)
    return s if len(s) <= 160 else s[:157] + "..."


def first_difference(clean: list[Any], hostile: list[Any], back: dict[str, str]) -> Optional[str]:
    """None if the hostile log equals the clean one once names are mapped back."""
    for i, (a, b) in enumerate(zip_longest(clean, hostile), 1):
        if a != (unmap(b, back) if b is not None else None):
            return f"line {i}: clean {_short(a)} vs hostile {_short(b)}"
    return None


def raw_characters(run_dir: Path) -> list[str]:
    """Our own log files that contain characters core/log.py must strip."""
    out = []
    for name in OWN_LOGS:
        p = run_dir / name
        if p.exists():
            m = RAW_BAD.search(p.read_text(encoding="utf-8", errors="replace"))
            if m:
                out.append(f"{name} has a raw U+{ord(m.group()):04X}")
    return out


# --------------------------------------------------------------------------
# Runs (one process each: core/log.py keeps one run log per process)
# --------------------------------------------------------------------------

def _run_job(job: dict) -> dict:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    t0 = time.perf_counter()
    back: dict[str, str] = {}
    try:
        import run as runner                       # the same agents and wiring as `python run.py`
        from core import log
        runner.llm_client.configure(enabled=False)
        if job["texts"]:
            real = runner.TerminalAPI

            def hostile_terminal(*a, **kw):
                api = real(*a, **kw)
                back.update(inject(api.calls, job["texts"], job["offset"]))
                return api

            runner.TerminalAPI = hostile_terminal
        run_dir = Path(job["runs"]) / job["scenario"]
        summary = runner.run_one(job["scenario"], Path(job["data"]), None, Path(job["runs"]), "multi")
        return {"summary": summary, "back": back, "sim": log.simulator_log_counts(run_dir),
                "missing": log.count_missing_reasons(run_dir), "wall_s": round(time.perf_counter() - t0, 1)}
    except Exception as e:
        return {"crashed": f"{type(e).__name__}: {e}"[:300], "back": back}


def problems(res: dict, run_dir: Path, what: str) -> list[str]:
    if res.get("crashed"):
        return [f"{what} run could not finish: {res['crashed']}"]
    s, sim, loop = res["summary"], res["sim"], res["summary"].get("loop", {})
    out = []
    viol = max(len(s.get("violations") or []), sim.get("VIOLATION", 0))
    if viol or sim.get("tool_error") or loop.get("breaches"):
        out.append(f"{what}: {viol} violations, {sim.get('tool_error')} refused calls, "
                   f"{loop.get('breaches')} breaches")
    if loop.get("exceptions"):
        out.append(f"{what}: {loop['exceptions']} loop exceptions")
    if res["missing"]:
        out.append(f"{what}: {res['missing']} actions without a reason")
    out += [f"{what}: {r}" for r in raw_characters(run_dir)]
    return out


def check(scenarios: list[str], data: Path, out: Path, texts: list[str], jobs: int) -> tuple[bool, list[str]]:
    specs = []
    for i, sc in enumerate(scenarios):
        for kind in ("clean", "hostile"):
            specs.append({"scenario": sc, "data": str(data.resolve()), "runs": str((out / kind).resolve()),
                          "texts": texts if kind == "hostile" else [], "offset": 3 * i})
    with ProcessPoolExecutor(max_workers=max(1, jobs), max_tasks_per_child=1) as pool:
        results = list(pool.map(_run_job, specs))
    ok, lines = True, []
    for i, sc in enumerate(scenarios):
        clean, hostile = results[2 * i], results[2 * i + 1]
        fails = problems(clean, out / "clean" / sc, "clean") + problems(hostile, out / "hostile" / sc, "hostile")
        n = 0
        if not clean.get("crashed") and not hostile.get("crashed"):
            try:
                a, b = _jsonl(out / "clean" / sc / "actions.jsonl"), _jsonl(out / "hostile" / sc / "actions.jsonl")
                n = len(a)
                diff = first_difference(a, b, hostile["back"])
                if diff:
                    fails.append(f"a hostile vessel name changed what we did: {diff}")
            except (OSError, ValueError) as e:
                fails.append(f"could not compare actions.jsonl: {type(e).__name__}: {e}"[:200])
        ok = ok and not fails
        head = f"{sc:12} {len(hostile.get('back', {}))} hostile names · {n} log lines"
        lines.append(f"{'ok  ' if not fails else 'FAIL'} {head}" + ("" if fails else " · identical"))
        lines += [f"       {f}" for f in fails]
    return ok, lines


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", action="append", help="one scenario (repeatable); default: all 5 practice")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "check" / "redteam")
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = p.parse_args(argv)
    if not a.data.exists():
        p.error(f"data workbook not found: {a.data}")
    texts = load_corpus(a.corpus)
    t0 = time.perf_counter()
    ok, lines = check(a.scenario or PRACTICE, a.data, a.out, texts, a.jobs)
    print("\n".join(lines))
    print(f"red team: {len(texts)} hostile vessel names from {a.corpus.name}, LLM off · "
          f"{'PASS' if ok else 'FAIL'} · {time.perf_counter() - t0:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
