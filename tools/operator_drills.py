"""LLM-6 drills against the real model (needs the LLM in .env and the workbook in data/).

    python tools/operator_drills.py

Builds practice_03 at t=0 and sends every message in llm/operator_drills.json through
llm.operator translate + validate. This is a harness like run.py: it builds the simulator
itself, so it lives in tools/, not in agent code. Exit code 1 if any drill fails.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.state import build_state          # noqa: E402
from llm import operator                    # noqa: E402
from llm import client as llm_client        # noqa: E402
from llm.client import load_dotenv          # noqa: E402
from vendor import TerminalAPI              # noqa: E402

SCENARIO = "practice_03"
DATA = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"


def main() -> int:
    if not DATA.exists():
        print(f"workbook not found: {DATA}")
        return 2
    load_dotenv()
    llm_client.warmup()          # first connection to the VM, outside the drills' 8 s calls
    with tempfile.TemporaryDirectory() as tmp:
        api = TerminalAPI(SCENARIO, data_xlsx=str(DATA), log_path=str(Path(tmp) / "actions.jsonl"))
        rows = operator.run_drills(build_state(api))
    for r in rows:
        print(("PASS " if r["pass"] else "FAIL ") + f"{r['expected']:8} {str(r['command']):20} {r['message'][:70]}")
        if not r["pass"]:
            print(f"       why: {r['why']}")
    n = sum(r["pass"] for r in rows)
    print(f"\n{n}/{len(rows)} drills passed")
    return 0 if n == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
