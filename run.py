"""Run the P2 terminal agent on one or more scenarios.

    python run.py --scenario practice_01
    python run.py --all
    python run.py --scenario hidden_01 --key path/to/P2_Container_Terminal_ANSWER_KEY.xlsx

Output per scenario in runs/<scenario>/:
    actions.jsonl    the toolkit's own action log (what the organizers score)
    decisions.jsonl  our decision log: what we did and why
    summary.json     run_summary() plus our honest turnaround
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.skeleton import Skeleton  # noqa: E402
from vendor import TerminalAPI  # noqa: E402

PRACTICE = [f"practice_0{i}" for i in range(1, 6)]
DEFAULT_DATA = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"


def run_one(scenario: str, data: Path, key: Path | None, runs: Path) -> dict:
    run_dir = runs / scenario
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    api = TerminalAPI(scenario, data_xlsx=str(data), key_xlsx=str(key) if key else None,
                      log_path=str(run_dir / "actions.jsonl"))
    summary = Skeleton(api, run_dir).run()
    summary["wall_s"] = round(time.perf_counter() - t0, 1)
    return summary


def print_table(rows: list[dict]) -> None:
    cols = [("scenario", "scenario", 12), ("departed", None, 9), ("avg TA h", "avg_turnaround_h", 9),
            ("FCFS h", "fcfs_baseline_avg_turnaround_h_same_calls", 8), ("honest h", "honest_turnaround_h", 9),
            ("moves/h", "crane_moves_per_hour", 8), ("truck idle", "truck_idle_hours", 11),
            (">85%", "placements_into_blocks_over_85pct", 6), ("delivered", "import_boxes_delivered", 10),
            ("viol", None, 5), ("errors", "skeleton_exceptions", 7), ("wall s", "wall_s", 7)]
    print("".join(h.ljust(w) for h, _, w in cols))
    for r in rows:
        cells = []
        for h, k, w in cols:
            if h == "departed":
                v = f"{r['vessels_departed']}/{r['vessels_total']}"
            elif h == "viol":
                v = len(r["violations"])
            else:
                v = r.get(k)
            cells.append(str("-" if v is None else v).ljust(w))
        print("".join(cells))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scenario", help="e.g. practice_01 or hidden_01")
    g.add_argument("--all", action="store_true", help="run all 5 practice scenarios")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="data workbook (default: data/...)")
    p.add_argument("--key", type=Path, help="answer-key workbook, needed for hidden_* scenarios")
    p.add_argument("--runs", type=Path, default=ROOT / "runs", help="output folder (default: runs/)")
    p.add_argument("--no-llm", action="store_true", help="rules only (the skeleton never uses an LLM)")
    p.add_argument("--agent", choices=["skeleton"], default="skeleton")
    a = p.parse_args(argv)

    if not a.data.exists():
        p.error(f"data workbook not found: {a.data} (put the official workbook in data/)")
    scenarios = PRACTICE if a.all else [a.scenario]
    if any(s.startswith("hidden") for s in scenarios) and not a.key:
        p.error("hidden scenarios need --key")

    rows = []
    for s in scenarios:
        try:
            rows.append(run_one(s, a.data, a.key, a.runs))
        except Exception as e:  # a broken scenario must not stop the others
            print(f"[{s}] could not run: {type(e).__name__}: {e}", file=sys.stderr)
    if rows:
        print_table(rows)
    return 0 if len(rows) == len(scenarios) and all(not r["violations"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
