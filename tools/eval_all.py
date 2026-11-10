"""Scorecard (CORE-6): one command that tells the whole team whether we're on track.

    python tools/eval_all.py --all --no-llm                 # 5 practice scenarios, rules only
    python tools/eval_all.py --scenario practice_02         # one scenario
    python tools/eval_all.py --all --no-llm --against origin/main   # fresh before/after vs main
    python tools/eval_all.py --all --no-llm --md            # PR-template table to paste
    python tools/eval_all.py --all --variants 99 --no-llm   # + workbooks in data/variants/ (CORE-7: tools/perturb.py)
    python tools/eval_all.py --all --variants 99 --fcfs --no-llm    # + never worse than FCFS on no-shortage scenarios
    python tools/eval_all.py --all --variants-dir data/variants/limits --variants 9 --limits --no-llm   # toolkit-limit set

Columns are the scorecard from the brief's "Goals & KPI contract" page:

    must pass   violations (simulator VIOLATION + refused calls + live-state breaches) = 0
                crashes (loop exceptions, or a scenario that couldn't run) = 0
                ships departed >= main
                honest turnaround (unfinished ships counted to the window end) <= main + 3%
                actions without a reason = 0
                runtime < 5 min per scenario; LLM calls <= budget
    report      simulator's avg turnaround vs FCFS (departed ships only), crane moves/h,
                truck idle h, placements into blocks over 85%, boxes delivered,
                guard refusals, escalations, watchdog / rules-only ticks

Variants (CORE-7): data/variants/index.json says which family each scenario is and what must hold
afterwards ("expect": ships never berthed or placed, anomalies escalated once, same KPIs as the original
for the renamed family, ...). --fcfs also runs the rules-only FCFS agent (no coordinator) on practice and
no-shortage scenarios: multi must never be worse there (ships departed >= FCFS, honest turnaround <= FCFS + 3%).
--limits is the pass rule for the toolkit-limit set, where the organizers' toolkit itself fails on the data:
no crash, no VIOLATION, no unhandled exception, every action has a reason; refused calls are only reported.

A refused call (tool_error) counts as a violation: the toolkit README says "attempting a
forbidden action counts as a violation". The FCFS baseline sheet ignores yard capacity
(every ship departs), so ships departed are compared with main, and FCFS only with the
simulator's turnaround of departed ships.

"Main" is, in order: --baseline FILE, a fresh run of --against REF (a temporary git worktree
of that commit, same workbook), or runs/eval/main-<agent>-<llm|no-llm>.json, which is saved
automatically whenever --all runs on a clean checkout of origin/main with the same settings. Without any of these, nothing is
compared and only the absolute checks apply.

Results go to runs/eval/scorecard.json (latest) and runs/eval/<commit>.json. The exit code
is 1 if any check fails, so CI (SAFE-6) can use it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
PRACTICE = [f"practice_0{i}" for i in range(1, 6)]
DEFAULT_DATA = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
VARIANTS_DIR = ROOT / "data" / "variants"
EVAL_DIR = ROOT / "runs" / "eval"

TURNAROUND_SLACK = 0.03         # GROUND_RULES §6: honest turnaround not more than 3% worse
MAX_RUNTIME_S = 300             # GROUND_RULES §1.9: under 5 minutes per scenario


# --------------------------------------------------------------------------
# One scenario, in its own process (core/log.py keeps one run log per process)
# --------------------------------------------------------------------------

def _run_job(job: dict) -> dict:
    """Run one scenario with the code in job['root'] and return its scorecard row."""
    root = Path(job["root"])
    sys.path.insert(0, str(root))
    os.chdir(root)
    label, run_dir = job["label"], Path(job["run_dir"])
    t0 = time.perf_counter()
    try:
        import run as runner                       # the tree's own run.py: same agents as `python run.py`
        from core import log
        if job["no_llm"]:
            runner.llm_client.configure(enabled=False)
        summary = runner.run_one(job["scenario"], Path(job["data"]), None, run_dir.parent, job["agent"])
        sim = log.simulator_log_counts(run_dir)
        missing = log.count_missing_reasons(run_dir)
        escalations = _count_choice(run_dir, "escalate")
        row = make_row(label, summary, sim, missing, escalations)
        row["family"] = job.get("family") or "practice"
        row["expect_fails"] = expect_checks(run_dir, row, job.get("expect") or {})
        return row
    except Exception as e:                          # the scenario couldn't run: that is a crash
        return {"label": label, "crashed": f"{type(e).__name__}: {e}"[:300], "family": job.get("family") or "practice",
                "wall_s": round(time.perf_counter() - t0, 1)}


def _count_choice(run_dir: Path, choice: str) -> int:
    path = run_dir / "decisions.jsonl"
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("choice") == choice:
            n += 1
    return n


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def expect_checks(run_dir: Path, row: dict, expect: dict) -> list[str]:
    """What the generator said must hold for this scenario (CORE-7), read from the run's own logs.
    Also for every run: the same 'Bad data:' sentence must not be logged twice (YARD-5: each anomaly once)."""
    fails = []
    calls = [r for r in _jsonl(run_dir / "actions.jsonl") if r.get("action") == "tool_call"]
    decisions = _jsonl(run_dir / "decisions.jsonl")
    escalations = [str(d.get("reason", "")) for d in decisions if d.get("choice") == "escalate"]

    berthed = {(c.get("arguments") or {}).get("call_id") for c in calls if c.get("tool") == "assign_berth"}
    placed = set()
    for c in calls:
        if c.get("tool") == "place_container" and isinstance(c.get("arguments"), dict):
            placed |= set(c["arguments"].get("container_ids") or [])
    for cid in expect.get("never_berthed", []):
        if cid in berthed:
            fails.append(f"expect: {cid} has bad data but was berthed")
    for box in expect.get("never_placed", []):
        if box in placed:
            fails.append(f"expect: box {box} has bad data but was placed in the yard")
    for cid in expect.get("not_departed", []):
        if cid not in (row.get("unfinished") or []):
            fails.append(f"expect: {cid} cannot finish but is not listed as unfinished")
    if expect.get("event") == "ship_impossible":
        for cid in expect.get("not_departed", []):
            if not any(cid in e and "can never finish" in e for e in escalations):
                fails.append(f"expect: no 'can never finish' escalation for {cid}")
    for key in expect.get("anomaly", []):
        if not any(key in e for e in escalations):
            fails.append(f"expect: no escalation mentions {key}")
    seen: dict[str, int] = {}
    for e in escalations:
        if e.startswith("Bad data:"):
            seen[e] = seen.get(e, 0) + 1
    twice = [e for e, n in seen.items() if n > 1]
    if twice:
        fails.append(f"{len(twice)} anomaly sentences logged more than once, e.g. {twice[0][:80]}")
    return fails


SAME_AS_KEYS = ("departed", "total", "honest_h", "sim_h", "moves_h", "truck_idle_h", "over85", "delivered")


def same_as_checks(rows: list[dict], expects: dict) -> dict[str, list[str]]:
    """The renamed family: every KPI equals the original practice scenario's, or some code reads an ID."""
    by = {r["label"]: r for r in rows}
    out = {}
    for label, ex in expects.items():
        if not ex.get("same_as") or label not in by or by[label].get("crashed"):
            continue
        orig = by.get(label.split("/")[-1])
        if orig is None or orig.get("crashed"):
            print(f"note: {label}: no original {label.split('/')[-1]} row in this run, 'same as original' not checked")
            continue
        diff = [f"{k} {orig[k]} -> {by[label][k]}" for k in SAME_AS_KEYS if orig.get(k) != by[label].get(k)]
        if diff:
            out[label] = ["renamed IDs changed the result (some code reads the content of an ID): " + "; ".join(diff)]
    return out


def fcfs_checks(rows: list[dict], fcfs_rows: list[dict], noshort: set) -> dict[str, list[str]]:
    """On no-shortage scenarios multi must never be worse than FCFS: ships departed >= FCFS and
    honest turnaround <= FCFS + 3 % (GROUND_RULES §6 slack)."""
    ref = {r["label"]: r for r in fcfs_rows}
    out = {}
    for r in rows:
        f = ref.get(r["label"])
        if r["label"] not in noshort or r.get("crashed") or f is None or f.get("crashed"):
            continue
        bad = []
        if r["departed"] is not None and f["departed"] is not None and r["departed"] < f["departed"]:
            bad.append(f"ships departed {r['departed']} < FCFS {f['departed']}")
        if r["honest_h"] is not None and f["honest_h"] is not None and r["honest_h"] > f["honest_h"] * (1 + TURNAROUND_SLACK) + 1e-9:
            bad.append(f"honest turnaround {r['honest_h']} h > FCFS {f['honest_h']} h + 3%")
        if bad:
            out[r["label"]] = ["worse than FCFS on a no-shortage scenario: " + "; ".join(bad)]
    return out


def limits_checks(r: dict) -> list[str]:
    """Pass rule for the toolkit-limit set: the toolkit itself fails on this data, so refused calls are
    only reported. What must hold: no crash, no simulator VIOLATION, no unhandled exception, a reason on every action."""
    if r.get("crashed"):
        return [f"crashed: {r['crashed']}"]
    fails = []
    if r["violations"]:
        fails.append(f"{r['violations']} simulator VIOLATIONs")
    if r["exceptions"]:
        fails.append(f"{r['exceptions']} unhandled exceptions")
    if r["missing_reasons"]:
        fails.append(f"{r['missing_reasons']} actions without a reason")
    if r["wall_s"] is not None and r["wall_s"] >= MAX_RUNTIME_S:
        fails.append(f"runtime {r['wall_s']} s >= {MAX_RUNTIME_S} s")
    return fails


def make_row(label: str, s: dict, sim: dict, missing: int, escalations: int) -> dict:
    """One scenario's scorecard row from run_one()'s summary and the run's log files."""
    loop = s.get("loop", {})
    llm = s.get("llm") or {}
    return {
        "label": label,
        "crashed": None,
        # run_summary()'s list and the VIOLATION lines in actions.jsonl should agree; trust the larger.
        "violations": max(len(s.get("violations") or []), sim.get("VIOLATION", 0)),
        "refused_calls": sim.get("tool_error", 0),
        "breaches": loop.get("breaches", 0),
        "exceptions": loop.get("exceptions", 0),
        "departed": s.get("vessels_departed"),
        "total": s.get("vessels_total"),
        "honest_h": s.get("honest_turnaround_h"),
        "sim_h": s.get("avg_turnaround_h"),
        "fcfs_h": s.get("fcfs_baseline_avg_turnaround_h_same_calls"),
        "moves_h": s.get("crane_moves_per_hour"),
        "truck_idle_h": s.get("truck_idle_hours"),
        "over85": s.get("placements_into_blocks_over_85pct"),
        "delivered": s.get("import_boxes_delivered"),
        "missing_reasons": missing,
        "guard_refusals": loop.get("actions_refused", 0),
        "mismatches": loop.get("mismatches", 0),
        "watchdog": loop.get("watchdog", 0),
        "rules_only_ticks": loop.get("rules_only_ticks", 0),
        "escalations": escalations,
        "llm_calls": llm.get("requests_used", 0) or 0,
        "llm_budget": llm.get("budget"),
        "wall_s": s.get("wall_s"),
        "unfinished": s.get("unfinished_calls", []),
    }


# --------------------------------------------------------------------------
# Checks and comparison
# --------------------------------------------------------------------------

def violations_total(r: dict) -> int:
    """Everything that counts against Rule 1: simulator VIOLATIONs, refused calls, breaches."""
    return r.get("violations", 0) + r.get("refused_calls", 0) + r.get("breaches", 0)


def checks(r: dict, base: Optional[dict]) -> list[str]:
    """Failed checks for one row ([] = pass). `base` is main's row for the same label, if any."""
    if r.get("crashed"):
        return [f"crashed: {r['crashed']}"]
    fails = []
    if violations_total(r):
        fails.append(f"violations {r['violations']} + refused calls {r['refused_calls']}"
                     f" + breaches {r['breaches']}")
    if r["exceptions"]:
        fails.append(f"{r['exceptions']} unhandled exceptions")
    if r["missing_reasons"]:
        fails.append(f"{r['missing_reasons']} actions without a reason")
    if r["wall_s"] is not None and r["wall_s"] >= MAX_RUNTIME_S:
        fails.append(f"runtime {r['wall_s']} s >= {MAX_RUNTIME_S} s")
    if r["llm_budget"] is not None and r["llm_calls"] > r["llm_budget"]:
        fails.append(f"LLM calls {r['llm_calls']} > budget {r['llm_budget']}")
    fails.extend(r.get("expect_fails") or [])
    if base and not base.get("crashed"):
        if r["departed"] is not None and base["departed"] is not None and r["departed"] < base["departed"]:
            fails.append(f"ships departed {r['departed']} < main {base['departed']}")
        if r["honest_h"] is not None and base["honest_h"] is not None \
                and r["honest_h"] > base["honest_h"] * (1 + TURNAROUND_SLACK) + 1e-9:
            fails.append(f"honest turnaround {r['honest_h']} h > main {base['honest_h']} h + 3%")
    return fails


def compare(rows: list[dict], baseline: Optional[dict]) -> dict:
    """{label: [failed checks]} for every row."""
    base = {b["label"]: b for b in (baseline or {}).get("rows", [])}
    return {r["label"]: checks(r, base.get(r["label"])) for r in rows}


def totals(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("crashed")]
    hon = [r["honest_h"] for r in ok if r["honest_h"] is not None]
    return {
        "departed": sum(r["departed"] or 0 for r in ok),
        "total": sum(r["total"] or 0 for r in ok),
        "honest_h_mean": round(sum(hon) / len(hon), 2) if hon else None,
        "violations": sum(violations_total(r) for r in ok),
        "crashes": sum(1 for r in rows if r.get("crashed")) + sum(r["exceptions"] for r in ok),
        "missing_reasons": sum(r["missing_reasons"] for r in ok),
        "wall_s_max": max((r["wall_s"] or 0 for r in ok), default=0),
    }


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def load_index(vdir: Path) -> Optional[dict]:
    """index.json that tools/perturb.py wrote next to the variant books, or None."""
    try:
        return json.loads((Path(vdir) / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def variant_books(vdir: Path, index: Optional[dict]) -> list[Path]:
    """Variant workbooks, perturbed (v*) first, then malformed (m*). With an index only the books it lists are
    taken, so the mock answer key and half-written files in the folder are never run as variants."""
    vdir = Path(vdir)
    names = sorted(index["books"]) if index and index.get("books") else \
        sorted(p.stem for p in vdir.glob("*.xlsx") if not p.name.startswith("MOCK") and ".tmp" not in p.name)
    names.sort(key=lambda n: (not n.startswith("v"), n))
    return [vdir / f"{n}.xlsx" for n in names if (vdir / f"{n}.xlsx").exists()]


def plan(scenarios: list[str], variants: int, data: Path, vdir: Optional[Path] = None,
         families: Optional[list[str]] = None) -> list[tuple[str, str, Path]]:
    """(label, scenario, workbook) for every run: practice first, then variant workbooks.
    families: only the variant scenarios of these families (needs index.json)."""
    out = [(s, s, data) for s in scenarios]
    if variants:
        vdir = Path(vdir) if vdir else VARIANTS_DIR
        index = load_index(vdir)
        for b in variant_books(vdir, index)[:variants]:
            for s in scenarios:
                if families:
                    fam = (((index or {}).get("books") or {}).get(b.stem) or {}).get(s, {}).get("family")
                    if fam not in families:
                        continue
                out.append((f"{b.stem}/{s}", s, b))
    return out


def meta_for(spec: list[tuple[str, str, Path]], vdir: Optional[Path]) -> dict:
    """{label: {"family", "expect"}} for the variant runs, from index.json (practice runs have none)."""
    index = load_index(Path(vdir) if vdir else VARIANTS_DIR) or {}
    books = index.get("books") or {}
    out = {}
    for label, sc, book in spec:
        m = (books.get(Path(book).stem) or {}).get(sc) if "/" in label else None
        out[label] = {"family": (m or {}).get("family") or ("practice" if "/" not in label else "variant"),
                      "expect": (m or {}).get("expect") or {}}
    return out


def run_all(root: Path, jobs_spec: list[tuple[str, str, Path]], out_dir: Path, agent: str,
            no_llm: bool, jobs: int, meta: Optional[dict] = None) -> list[dict]:
    """Run every job with the code in `root`, each scenario in its own process."""
    meta = meta or {}
    jobs_list = [{"root": str(root), "label": label, "scenario": sc, "data": str(Path(book).resolve()),
                  "run_dir": str(out_dir / label.replace("/", "__") / sc), "agent": agent, "no_llm": no_llm,
                  "family": (meta.get(label) or {}).get("family"), "expect": (meta.get(label) or {}).get("expect")}
                 for label, sc, book in jobs_spec]
    # max_tasks_per_child=1: every scenario gets a fresh process, so module state never leaks.
    with ProcessPoolExecutor(max_workers=max(1, jobs), max_tasks_per_child=1) as pool:
        return list(pool.map(_run_job, jobs_list))


def _git(*args: str, cwd: Path = ROOT) -> Optional[str]:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def head_info() -> dict:
    sha = _git("rev-parse", "HEAD")
    main = _git("rev-parse", "origin/main")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"commit": sha, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "is_main": bool(sha and sha == main and not dirty), "dirty": dirty}


def run_against(ref: str, jobs_spec, agent, no_llm, jobs, tmp_runs: Path, meta: Optional[dict] = None) -> dict:
    """Fresh scorecard of `ref` from a temporary git worktree (same workbook, same settings)."""
    sha = _git("rev-parse", ref)
    if not sha:
        raise SystemExit(f"--against: unknown git ref {ref!r}")
    tree = Path(tempfile.mkdtemp(prefix="eval_main_"))
    shutil.rmtree(tree)
    if _git("worktree", "add", "--detach", str(tree), sha) is None:
        raise SystemExit(f"--against: could not check out {ref} into a temporary worktree")
    try:
        rows = run_all(tree, jobs_spec, tmp_runs, agent, no_llm, jobs, meta)
    finally:
        _git("worktree", "remove", "--force", str(tree))
        _git("worktree", "prune")
    return {"commit": sha, "ref": ref, "rows": rows}


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _f(v) -> str:
    return "-" if v is None else str(v)


def _delta(new, old) -> str:
    if old is None or new is None or old == new:
        return _f(new)
    return f"{_f(old)}→{_f(new)}"


def print_table(rows: list[dict], fails: dict, baseline: Optional[dict]) -> None:
    base = {b["label"]: b for b in (baseline or {}).get("rows", [])}
    cols = [("scenario", 22), ("pass", 5), ("departed", 10), ("honest h", 14), ("sim h", 7), ("FCFS h", 7),
            ("viol", 5), ("refused", 8), ("crash", 6), ("no reason", 10), ("moves/h", 8), ("truck idle", 11),
            (">85%", 6), ("escal.", 7), ("LLM", 5), ("wall s", 7)]
    print("".join(h.ljust(w) for h, w in cols))
    for r in rows:
        b = base.get(r["label"]) or {}
        if r.get("crashed"):
            print(r["label"].ljust(22) + "FAIL ".ljust(5) + f"could not run: {r['crashed']}")
            continue
        dep = f"{r['departed']}/{r['total']}"
        bdep = f"{b['departed']}/{b['total']}" if b.get("departed") is not None else None
        cells = [r["label"], "ok" if not fails[r["label"]] else "FAIL", _delta(dep, bdep),
                 _delta(r["honest_h"], b.get("honest_h")), _f(r["sim_h"]), _f(r["fcfs_h"]),
                 r["violations"] + r["breaches"], r["refused_calls"], r["exceptions"], r["missing_reasons"],
                 _f(r["moves_h"]), _f(r["truck_idle_h"]), _f(r["over85"]), r["escalations"],
                 r["llm_calls"], _f(r["wall_s"])]
        print("".join(str(c).ljust(w) for c, (_, w) in zip(cells, cols)))
    t = totals(rows)
    print(f"\ntotal: {t['departed']}/{t['total']} ships departed · mean honest turnaround {_f(t['honest_h_mean'])} h"
          f" · violations {t['violations']} · crashes {t['crashes']} · actions without a reason"
          f" {t['missing_reasons']} · slowest scenario {t['wall_s_max']} s")
    if baseline:
        print(f"compared with main @ {str(baseline.get('commit'))[:8]} ({baseline.get('source', '')})")
    else:
        print("no main scorecard to compare with (run once on a clean origin/main, or use --against origin/main)")
    for label, fl in fails.items():
        for f in fl:
            print(f"  FAIL {label}: {f}")


def by_family(rows: list[dict], fails: dict) -> list[dict]:
    """One line per family over the variant rows (labels with a '/'): how many, ships departed, honest turnaround, failures."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if "/" in r["label"]:
            groups.setdefault(r.get("family") or "variant", []).append(r)
    out = []
    for fam, rs in sorted(groups.items()):
        ok = [r for r in rs if not r.get("crashed")]
        hon = [r["honest_h"] for r in ok if r["honest_h"] is not None]
        out.append({"family": fam, "n": len(rs), "departed": sum(r["departed"] or 0 for r in ok),
                    "total": sum(r["total"] or 0 for r in ok),
                    "honest_h_mean": round(sum(hon) / len(hon), 2) if hon else None,
                    "violations": sum(violations_total(r) for r in ok),
                    "crashes": sum(1 for r in rs if r.get("crashed")) + sum(r["exceptions"] for r in ok),
                    "failed": sum(1 for r in rs if fails.get(r["label"]))})
    return out


def print_families(rows: list[dict], fails: dict, fcfs_rows: Optional[list[dict]] = None) -> None:
    fam = by_family(rows, fails)
    if not fam:
        return
    print("\nvariants by family")
    print("".join(h.ljust(w) for h, w in [("family", 16), ("n", 4), ("departed", 10), ("honest h", 10),
                                          ("viol", 6), ("crash", 7), ("failed", 7)]))
    for f in fam:
        cells = [f["family"], f["n"], f"{f['departed']}/{f['total']}", _f(f["honest_h_mean"]), f["violations"],
                 f["crashes"], f["failed"]]
        print("".join(str(c).ljust(w) for c, w in zip(cells, (16, 4, 10, 10, 6, 7, 7))))
    if fcfs_rows:
        ref = {r["label"]: r for r in fcfs_rows if not r.get("crashed")}
        pairs = [(r, ref[r["label"]]) for r in rows if r["label"] in ref and not r.get("crashed")
                 and (r.get("family") or "").startswith("noshort")]
        if pairs:
            d = sum(r["departed"] or 0 for r, _ in pairs)
            fd = sum(f["departed"] or 0 for _, f in pairs)
            h = [r["honest_h"] for r, _ in pairs if r["honest_h"] is not None]
            fh = [f["honest_h"] for _, f in pairs if f["honest_h"] is not None]
            print(f"\nno-shortage vs FCFS ({len(pairs)} scenarios): ships departed {d} vs {fd} · mean honest turnaround "
                  f"{round(sum(h) / len(h), 2) if h else '-'} h vs {round(sum(fh) / len(fh), 2) if fh else '-'} h")


def markdown(rows: list[dict], baseline: Optional[dict], fails: Optional[dict] = None) -> str:
    """The PR template's scorecard table (before → after). Variant scenarios are one line per family."""
    base = {b["label"]: b for b in (baseline or {}).get("rows", [])}
    out = ["| Scenario | Ships departed (before → after) | Honest turnaround h | Violations | Crashes |",
           "|---|---|---|---|---|"]
    for r in [x for x in rows if "/" not in x["label"]]:
        b = base.get(r["label"]) or {}
        if r.get("crashed"):
            out.append(f"| {r['label']} | – | – | – | crashed |")
            continue
        bdep = f"{b['departed']}/{b['total']}" if b.get("departed") is not None else "?"
        bv = violations_total(b) if b and not b.get("crashed") else "?"
        bc = b.get("exceptions", "?") if b else "?"
        out.append(f"| {r['label']} | {bdep} → {r['departed']}/{r['total']} | {_f(b.get('honest_h', '?'))} → "
                   f"{_f(r['honest_h'])} | {bv} → {violations_total(r)} | {bc} → {r['exceptions']} |")
    fam = by_family(rows, fails or {})
    if fam:
        out += ["", "| Variant family | Scenarios | Ships departed | Mean honest h | Violations | Crashes | Failed checks |",
                "|---|---|---|---|---|---|---|"]
        out += [f"| {f['family']} | {f['n']} | {f['departed']}/{f['total']} | {_f(f['honest_h_mean'])} | "
                f"{f['violations']} | {f['crashes']} | {f['failed']} |" for f in fam]
    return "\n".join(out)


def main_file(out_dir: Path, agent: str, no_llm: bool) -> Path:
    """main's saved scorecard for these settings (all practice scenarios, no variants)."""
    return out_dir / f"main-{agent}-{'no-llm' if no_llm else 'llm'}.json"


def save(card: dict, out_dir: Path, main_path: Optional[Path]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(card, indent=2, default=str)
    latest = out_dir / "scorecard.json"
    latest.write_text(text, encoding="utf-8")
    if card.get("commit"):
        (out_dir / f"{card['commit'][:12]}.json").write_text(text, encoding="utf-8")
    if main_path is not None:
        main_path.write_text(text, encoding="utf-8")
    return latest


def load_baseline(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------

def _utf8_stdout() -> None:
    """Windows writes redirected output in cp1252, which has no "→": switch to UTF-8 then."""
    try:
        "→·".encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def main(argv: Optional[list[str]] = None) -> int:
    _utf8_stdout()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="all 5 practice scenarios")
    g.add_argument("--scenario", action="append", help="one scenario (repeatable), e.g. practice_02")
    p.add_argument("--variants", type=int, default=0, metavar="N",
                   help="also run the first N workbooks in --variants-dir (CORE-7; perturbed first, then malformed)")
    p.add_argument("--variants-dir", type=Path, default=None, metavar="DIR", help="default data/variants")
    p.add_argument("--family", metavar="F[,F]", help="only these variant families (implies all books)")
    p.add_argument("--fcfs", action="store_true",
                   help="also run the FCFS rules-only agent on practice and no-shortage scenarios; multi must not be worse")
    p.add_argument("--limits", action="store_true", help="pass rule of the toolkit-limit set (see the module docstring)")
    p.add_argument("--no-llm", action="store_true", help="rules only: never call a language model")
    p.add_argument("--agent", choices=["multi", "skeleton"], default="multi")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--against", metavar="REF", help="fresh before/after: also run git REF (e.g. origin/main)")
    p.add_argument("--baseline", type=Path, metavar="FILE", help="compare with a saved scorecard JSON")
    p.add_argument("--jobs", type=int, default=None,
                   help="scenarios in parallel (default: all CPUs with --no-llm, 1 with the LLM)")
    p.add_argument("--out", type=Path, default=EVAL_DIR, help="where to save (default: runs/eval/)")
    p.add_argument("--md", action="store_true", help="also print the PR-template table")
    a = p.parse_args(argv)

    if not a.data.exists():
        p.error(f"data workbook not found: {a.data} (put the official workbook in data/)")
    scenarios = PRACTICE if a.all else a.scenario
    vdir = a.variants_dir or VARIANTS_DIR
    families = [x for x in (a.family or "").split(",") if x] or None
    n_books = a.variants or (len(variant_books(vdir, load_index(vdir))) if families else 0)
    spec = plan(scenarios, n_books, a.data, vdir, families)
    if n_books and len(spec) == len(scenarios):
        print(f"note: no variant workbooks in {vdir} (python tools/perturb.py makes them); running practice only")
    meta = meta_for(spec, vdir)
    # With the LLM on, scenarios run one at a time so the measured runtime is honest.
    jobs = a.jobs or (os.cpu_count() or 1 if a.no_llm else 1)

    t0 = time.perf_counter()
    rows = run_all(ROOT, spec, a.out / "runs", a.agent, a.no_llm, jobs, meta)
    info = head_info()

    baseline = None
    if a.baseline:
        baseline = load_baseline(a.baseline)
        if baseline is None:
            p.error(f"could not read baseline {a.baseline}")
        baseline["source"] = str(a.baseline)
    elif a.against:
        baseline = run_against(a.against, spec, a.agent, a.no_llm, jobs, a.out / "against", meta)
        baseline["source"] = f"fresh run of {a.against}"
    elif not info["is_main"]:
        mf = main_file(a.out, a.agent, a.no_llm)
        baseline = load_baseline(mf)
        if baseline:
            baseline["source"] = f"runs/eval/{mf.name}"

    fails = compare(rows, baseline)
    if a.limits:
        fails = {r["label"]: limits_checks(r) for r in rows}
    else:
        for extra in (same_as_checks(rows, {k: v["expect"] for k, v in meta.items()}),):
            for label, fl in extra.items():
                fails[label] = fails.get(label, []) + fl
    fcfs_rows = None
    if a.fcfs and not a.limits:
        ref_spec = [e for e in spec if "/" not in e[0] or (meta[e[0]]["family"] or "").startswith("noshort")]
        fcfs_rows = run_all(ROOT, ref_spec, a.out / "fcfs", "skeleton", a.no_llm, jobs, meta)
        noshort = {l for l, m in meta.items() if (m["family"] or "").startswith("noshort")}
        for label, fl in fcfs_checks(rows, fcfs_rows, noshort).items():
            fails[label] = fails.get(label, []) + fl
    card = {"commit": info["commit"], "branch": info["branch"], "dirty": info["dirty"],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": {"no_llm": a.no_llm, "agent": a.agent, "variants": a.variants, "data": a.data.name},
            "rows": rows, "totals": totals(rows), "failures": fails, "fcfs_rows": fcfs_rows,
            "families": by_family(rows, fails),
            "baseline": {"commit": baseline.get("commit"), "source": baseline.get("source")} if baseline else None,
            "eval_wall_s": round(time.perf_counter() - t0, 1)}
    # Only a clean origin/main run of all practice scenarios (no variants) becomes "main".
    saves_main = info["is_main"] and a.all and not a.variants
    path = save(card, a.out, main_file(a.out, a.agent, a.no_llm) if saves_main else None)

    print_table(rows, fails, baseline)
    print_families(rows, fails, fcfs_rows)
    if a.md:
        print("\n" + markdown(rows, baseline, fails))
    print(f"\nsaved {path}" + (f" and {main_file(a.out, a.agent, a.no_llm).name} (this is origin/main)"
                                    if saves_main else "")
          + f" · {card['eval_wall_s']} s")
    return 1 if any(fails.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
