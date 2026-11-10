"""BC-4 tuning sweep: run one parameter at a time over a fixed tuning set and show what it does per set.

    python tools/tune.py --param berth.hold_h --values 0,1,2,4,6
    python tools/tune.py --param crane.tick_h --values 0.25,0.5 --variants-dir data/variants --books v03,v04

Tuning set: the 5 practice scenarios + the variant books given (default v03-v06 of data/variants/,
made by `python tools/perturb.py --n 20 --seed 1`), grouped into three sets: practice, perturbed and
no-shortage (from index.json). Each value runs with every other parameter at its config/params.yaml
value (AGENTX_PARAMS points the child processes at a temporary copy). A value is "holds" only if,
on every set, ships departed are not lower and mean honest turnaround is at most 3% worse than the
current value; the BC-4 PR then checks the chosen values on a held-out variant set (another seed).

A test tool (tools/ is out of scope for the honest-play check); agent code never reads variants.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import params                                          # noqa: E402
from tools import eval_all                                       # noqa: E402

SLACK = 0.03


def set_of(family: str) -> str:
    if family == "practice":
        return "practice"
    return "no-shortage" if family.startswith("noshort") else "perturbed"


def summarise(rows: list[dict]) -> dict:
    """{set: {"n", "departed", "total", "honest_h", "violations", "crashes"}}."""
    out: dict = {}
    for r in rows:
        g = out.setdefault(set_of(r.get("family") or "practice"),
                           {"n": 0, "departed": 0, "total": 0, "_h": [], "violations": 0, "crashes": 0,
                            "over85": 0, "_blk": []})
        g["n"] += 1
        if r.get("crashed"):
            g["crashes"] += 1
            continue
        g["departed"] += r["departed"] or 0
        g["total"] += r["total"] or 0
        if r["honest_h"] is not None:
            g["_h"].append(r["honest_h"])
        g["violations"] += eval_all.violations_total(r)
        g["over85"] += r.get("over85") or 0
        if r.get("blocks_ship") is not None:
            g["_blk"].append(r["blocks_ship"])
        g["crashes"] += r["exceptions"]
    for g in out.values():
        h = g.pop("_h")
        g["honest_h"] = round(sum(h) / len(h), 2) if h else None
        blk = g.pop("_blk")
        g["blocks_ship"] = round(sum(blk) / len(blk), 2) if blk else None
    return out


def holds(new: dict, base: dict) -> list[str]:
    """Why `new` is worse than `base` on some set (empty list = holds or improves everywhere)."""
    bad = []
    for name, b in base.items():
        n = new.get(name)
        if n is None:
            continue
        if n["violations"] or n["crashes"]:
            bad.append(f"{name}: {n['violations']} violations, {n['crashes']} crashes")
        if n["departed"] < b["departed"]:
            bad.append(f"{name}: departed {n['departed']} < {b['departed']}")
        if b["honest_h"] and n["honest_h"] and n["honest_h"] > b["honest_h"] * (1 + SLACK) + 1e-9:
            bad.append(f"{name}: honest {n['honest_h']} h > {b['honest_h']} h + 3%")
    return bad


def run_value(key: str, value, spec, meta, out: Path, jobs: int) -> dict:
    vals = params.load()
    vals[key] = value
    sections: dict = {}
    for k, v in vals.items():
        s, name = k.split(".", 1)
        sections.setdefault(s, []).append(f"  {name}: {v}")
    text = "\n".join(f"{s}:\n" + "\n".join(lines) for s, lines in sections.items()) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(text)
    os.environ["AGENTX_PARAMS"] = f.name
    try:
        rows = eval_all.run_all(ROOT, spec, out / f"{key}={value}", "multi", True, jobs, meta)
    finally:
        os.environ.pop("AGENTX_PARAMS", None)
        os.unlink(f.name)
    return {"value": value, "sets": summarise(rows), "rows": rows}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--param", required=True, choices=sorted(params.SPEC))
    p.add_argument("--values", required=True, help="comma-separated; the current value is always run first")
    p.add_argument("--variants-dir", type=Path, default=ROOT / "data" / "variants")
    p.add_argument("--books", default="v03,v04,v05,v06")
    p.add_argument("--data", type=Path, default=eval_all.DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "tune")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = p.parse_args(argv)

    index = eval_all.load_index(a.variants_dir)
    books = [b for b in a.books.split(",") if b]
    spec = [(s, s, a.data) for s in eval_all.PRACTICE]
    spec += [(f"{b}/{s}", s, a.variants_dir / f"{b}.xlsx") for b in books for s in eval_all.PRACTICE]
    meta = eval_all.meta_for(spec, a.variants_dir)
    if not index:
        raise SystemExit(f"no index.json in {a.variants_dir}: run python tools/perturb.py first")

    current = params.get(a.param)
    conv = (lambda x: x) if isinstance(current, str) else float
    values = [current] + [conv(v) for v in a.values.split(",") if conv(v) != current]
    results = []
    for v in values:
        t0 = time.perf_counter()
        r = run_value(a.param, v, spec, meta, a.out, a.jobs)
        r["wall_s"] = round(time.perf_counter() - t0)
        results.append(r)
        base = results[0]["sets"]
        why = holds(r["sets"], base) if r is not results[0] else []
        cells = " · ".join(f"{k} {g['departed']}/{g['total']} {g['honest_h']} h >85% {g['over85']} blk {g['blocks_ship']}"
                           for k, g in sorted(r["sets"].items()))
        print(f"{a.param}={v!s:<14} {cells}  [{'current' if r is results[0] else ('holds' if not why else 'worse: ' + '; '.join(why))}]"
              f"  ({r['wall_s']} s)", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.param}.json").write_text(json.dumps(
        [{"value": r["value"], "sets": r["sets"]} for r in results], indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
