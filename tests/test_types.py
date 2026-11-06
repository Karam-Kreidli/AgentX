"""CORE-1 smoke tests: shared types import, are frozen, and the toolkit loads."""
import dataclasses
from datetime import datetime
from pathlib import Path

import pytest

from core.types import (Action, Berth, Block, SHIP_WORKING, Ship, WorldState, Crane)

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"


def _ship():
    return Ship("C1", "V", 250.0, 12.0, datetime(2026, 1, 1), datetime(2026, 1, 1), SHIP_WORKING,
                "B1", ("QC1",), 3, 10, 5, None)


def test_types_are_frozen():
    s = _ship()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.status = "departed"
    a = Action("assign_berth", {"call_id": "C1", "berth_id": "B1"}, "berth", "arrived and B1 is free")
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.reason = ""


def test_helpers():
    assert _ship().remaining_moves == 15
    assert Block("A", "standard", 187, 220).free == 33
    b1 = Berth("B1", 320, 14.5, 1, 30)
    assert b1.fits(300, 14.5) and not b1.fits(301, 14.0)
    st = WorldState(datetime(2026, 1, 1), datetime(2026, 1, 6), (_ship(),),
                    (Crane("QC1", "working", "C1", 5),), (), (), ())
    assert st.ship("C1").vessel == "V" and st.ship("nope") is None


@pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
def test_vendor_toolkit_loads(tmp_path):
    from vendor import TerminalAPI
    api = TerminalAPI("practice_01", data_xlsx=str(WORKBOOK), log_path=str(tmp_path / "a.jsonl"))
    assert api.get_current_time()["finished"] is False
    assert len(api.get_vessel_schedule()) > 0
