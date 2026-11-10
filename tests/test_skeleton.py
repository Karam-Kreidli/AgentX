"""CORE-2 tests: the fixed crane split is rail-safe, and a full scenario runs clean."""
import itertools
from pathlib import Path

import pytest

from core.skeleton import BERTHS, CRANE_SPLIT

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
HOME = {"QC1": 1, "QC2": 14, "QC3": 27, "QC4": 40, "QC5": 53, "QC6": 66}   # simulator start bays


def test_crane_split_safe_in_any_state():
    """Every crane is either at home or at its working bay, in any combination:
    rail order must always hold, and working bays must be >= 2 apart and inside the berth."""
    work = {c: bay for split in CRANE_SPLIT.values() for c, bay in split}
    cranes = sorted(HOME)
    assert set(work) == set(cranes)
    for combo in itertools.product(*[(HOME[c], work[c]) for c in cranes]):
        assert all(a < b for a, b in zip(combo, combo[1:])), combo
    bays = sorted(work.values())
    assert all(b - a >= 2 for a, b in zip(bays, bays[1:]))
    for b in BERTHS:
        for _, bay in CRANE_SPLIT[b.berth_id]:
            assert b.start_bay <= bay <= b.end_bay


@pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
def test_practice_01_runs_clean(tmp_path):
    from run import run_one
    s = run_one("practice_01", WORKBOOK, None, tmp_path)
    assert s["violations"] == []
    assert s["skeleton_exceptions"] == 0
    assert (tmp_path / "practice_01" / "actions.jsonl").exists()
    assert (tmp_path / "practice_01" / "decisions.jsonl").stat().st_size > 0


def test_rules_only_yard_escalates_unknown_cargo_once_per_ship():
    """Rules-only fallback: 20 boxes of unknown cargo give one escalation for their ship, not 20."""
    from core.skeleton import SkeletonAgent
    from tests.test_yard import boxes, ship, world
    s = world([ship()], boxes(20, "banana") + boxes(2))
    agent = SkeletonAgent(parts=("yard",))
    actions, _ = agent.propose(s, [])
    placed = {c for a in actions for c in a.args["container_ids"]}
    assert len(placed) == 2 and not any(c.startswith("B") for c in placed)
    unknown = [e for e in agent.escalations if "unknown cargo" in e]
    assert len(unknown) == 1 and "20 import box(es) of C1" in unknown[0]
    agent.propose(s, [])
    assert sum("unknown cargo" in e for e in agent.escalations) == 1
