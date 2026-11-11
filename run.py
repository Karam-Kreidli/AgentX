"""Run the P2 terminal agent on one or more scenarios.

    python run.py --scenario practice_01
    python run.py --all
    python run.py --scenario hidden_01 --key path/to/P2_Container_Terminal_ANSWER_KEY.xlsx
    python run.py --all-hidden --key path/to/P2_Container_Terminal_ANSWER_KEY.xlsx

From Python (organizers who build the TerminalAPI themselves):
    from run import run_agent
    summary = run_agent(api, run_dir)

Output per scenario in runs/<scenario>/:
    actions.jsonl    the toolkit's own action log (what the organizers score)
    decisions.jsonl  our decision log: what we did and why
    summary.json     run_summary() plus our honest turnaround
The command line also leaves a copy of actions.jsonl where the toolkit writes it by default,
runs/P2/<scenario>/actions.jsonl (relative to the current folder).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents.berth import BerthAgent  # noqa: E402
from agents.crane import CraneAgent  # noqa: E402
from agents.trucks import TruckAgent  # noqa: E402
from agents.yard import YardAgent  # noqa: E402
from baselines.single_agent import SingleAgent  # noqa: E402
from core.loop import Loop  # noqa: E402
from core.skeleton import SkeletonAgent  # noqa: E402
from llm import client as llm_client  # noqa: E402
from vendor import TerminalAPI  # noqa: E402

PRACTICE = [f"practice_0{i}" for i in range(1, 6)]
AGENTS = ["multi", "skeleton", "single", "multi-nocoord"]
DATA_NAME = "P2_Container_Terminal_Data.xlsx"
DEFAULT_DATA = ROOT / "data" / DATA_NAME
TOOLKIT_LOG_DIR = Path("runs") / "P2"    # where TerminalAPI writes actions.jsonl when given no log_path


def find_data(explicit: Path | None, key: Path | None = None, cwd: Path | None = None) -> Path | None:
    """CORE-9 (D16): the data workbook. --data wins; else the first that exists of data/, the folder
    of the key file, the current folder and toolskit/data/. None if nothing is found."""
    if explicit:
        return explicit
    cwd = cwd or Path.cwd()
    folders = [ROOT / "data"] + ([key.resolve().parent] if key else []) + [cwd, ROOT / "toolskit" / "data"]
    for f in folders:
        if (f / DATA_NAME).is_file():
            return f / DATA_NAME
    return None


def hidden_scenarios(key: Path) -> list[str]:
    """CORE-9 (D11): the scenario names in the answer key. Reads the `scenario` column of
    Hidden_Vessels and nothing else; the names go to the loader, not to the agents."""
    import pandas as pd
    col = pd.read_excel(key, sheet_name="Hidden_Vessels", usecols=["scenario"])["scenario"]  # honest-play: ok launcher lists scenario names (D11); no agent sees the key
    return sorted({str(x) for x in col.dropna()})


def make_agents(kind: str) -> list:
    """The agents the loop runs. 'multi' uses every real agent built so far; the skeleton
    covers the jobs no real agent does yet. 'skeleton' is the rules-only baseline."""
    if kind == "skeleton":
        return [SkeletonAgent()]
    if kind == "single":                    # CORE-8: one planner for berth, crane, truck and yard
        return [SingleAgent()]
    return [BerthAgent(), CraneAgent(), TruckAgent(), YardAgent()]


def run_agent(api, run_dir: Path, agent: str = "multi") -> dict:
    """CORE-9 (D12): play one scenario on a TerminalAPI the caller built, and return the summary.
    Writes decisions.jsonl and summary.json into run_dir. The actions log goes wherever the api was
    told to write it (the toolkit default, if the caller gave no log_path)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    llm_client.new_scenario(run_dir / "llm_calls.jsonl")         # fresh call budget per scenario (LLM-2)
    return Loop(api, run_dir, agents=make_agents(agent), coordinate=agent == "multi").run()


def run_one(scenario: str, data: Path, key: Path | None, runs: Path, agent: str = "multi") -> dict:
    run_dir = runs / scenario
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        api = TerminalAPI(scenario, data_xlsx=str(data), key_xlsx=str(key) if key else None,
                          log_path=str(run_dir / "actions.jsonl"))
        summary = run_agent(api, run_dir, agent)
    except Exception as e:      # SAFE-4: a scenario that can't even start still gets a summary.json
        _write_failed_summary(run_dir, scenario, e, time.perf_counter() - t0)
        raise
    summary["wall_s"] = round(time.perf_counter() - t0, 1)
    try:
        summary["llm"] = llm_client.get_client().stats()
    except Exception as e:
        summary["llm"] = {"error": type(e).__name__}
    return summary


def _write_failed_summary(run_dir: Path, scenario: str, e: Exception, wall_s: float) -> None:
    """Overwrites any summary.json from an earlier run, so a stale one can't pass for this one."""
    stub = {"scenario": scenario, "crashed": True, "error": f"{type(e).__name__}: {e}"[:300],
            "violations": [], "vessels_departed": 0, "vessels_total": None, "wall_s": round(wall_s, 1)}
    try:
        (run_dir / "summary.json").write_text(json.dumps(stub, indent=2), encoding="utf-8")
    except Exception as w:
        print(f"[{scenario}] could not write summary.json: {type(w).__name__}: {w}", file=sys.stderr)


def copy_toolkit_log(run_dir: Path, scenario: str) -> Path | None:
    """CORE-9 (D10): leave a copy of actions.jsonl at the toolkit's default path. Never raises."""
    src = run_dir / "actions.jsonl"
    dst = TOOLKIT_LOG_DIR / scenario / "actions.jsonl"
    try:
        if not src.is_file() or (dst.exists() and dst.resolve() == src.resolve()):
            return None
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return dst
    except Exception as e:
        print(f"[{scenario}] could not copy actions.jsonl to {dst}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


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
    g.add_argument("--all-hidden", action="store_true", help="run every scenario named in the answer key (needs --key)")
    p.add_argument("--data", type=Path, default=None,
                   help=f"data workbook (default: {DATA_NAME} in data/, next to the key, the current folder or toolskit/data/)")
    p.add_argument("--key", type=Path, help="answer-key workbook, needed for hidden_* scenarios")
    p.add_argument("--runs", type=Path, default=ROOT / "runs", help="output folder (default: runs/)")
    p.add_argument("--no-llm", action="store_true", help="rules only: never call a language model")
    p.add_argument("--agent", choices=AGENTS, default="multi",
                   help="multi: berth, crane, truck and yard agents with the coordinator; skeleton: rules-only (FCFS); "
                        "single: one planner (CORE-8); multi-nocoord: the four agents without negotiation (CORE-8)")
    a = p.parse_args(argv)

    if a.all_hidden and not a.key:
        p.error("--all-hidden needs --key")
    if a.key and not a.key.is_file():
        p.error(f"answer key not found: {a.key}")
    a.data = find_data(a.data, a.key)
    if a.data is None or not a.data.is_file():
        p.error(f"data workbook not found: {a.data or DATA_NAME} (put the official workbook in data/)")
    print(f"data workbook: {a.data}")
    if a.all_hidden:
        try:
            scenarios = hidden_scenarios(a.key)
        except Exception as e:
            p.error(f"could not read the scenario names from {a.key}: {type(e).__name__}: {e}")
        if not scenarios:
            p.error(f"no scenarios in Hidden_Vessels of {a.key}")
    else:
        scenarios = PRACTICE if a.all else [a.scenario]
    if any(s.startswith("hidden") for s in scenarios) and not a.key:
        p.error("hidden scenarios need --key")

    if a.no_llm:
        llm_client.configure(enabled=False)
    else:
        llm_client.warmup()     # pay the first-connection cost once, not inside a real call
    rows = []
    for s in scenarios:
        try:
            rows.append(run_one(s, a.data, a.key, a.runs, a.agent))
            copy_toolkit_log(a.runs / s, s)
        except Exception as e:  # a broken scenario must not stop the others
            print(f"[{s}] could not run: {type(e).__name__}: {e}", file=sys.stderr)
    if rows:
        print_table(rows)
    return 0 if len(rows) == len(scenarios) and all(not r["violations"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
