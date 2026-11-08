"""YARD-4 tests: stalled and impossible ship detector (agents/stall.py).

Impossible: more discharge left than import boxes left in the manifest, a general check (no
IDs), reported once. Stalled: at berth, nothing to discharge, no legal yard slot for its next
boxes. Events on change, never raises, wired into core/events.py and escalated by the loop;
P03-004 flagged within an hour of running out of boxes on the real practice_03.
"""
import ast
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import pytest

from agents import stall as S
from core.skeleton import BERTHS
from core.types import Block, Container, Ship, WorldState

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
T0 = datetime(2026, 9, 1, 6, 0)


@pytest.fixture(autouse=True)
def fresh():
    S.reset()
    yield
    S.reset()


def ship(cid, status="working", berth="B1", discharge=100, load=50):
    return Ship(cid, "MV " + cid, 200.0, 11.0, T0, T0, status, berth if status in ("berthed", "working") else None,
                ("QC1",) if status == "working" else (), 3, discharge, load, None)


def boxes(cid, n, cargo="dry", placed=None, start=0):
    return tuple(Container(f"{cid}-{cargo}-{i}", cid, "import", 40, cargo, 3.0 if cargo == "IMDG" else None, 20.0, placed)
                 for i in range(start, start + n))


def world(ships, manifests, free=None, now=T0):
    free = free or {}
    types = {"A": "standard", "G": "empty", "H": "IMDG", "I": "reefer"}
    blocks = tuple(Block(b, t, 100 - free.get(b, 0), 100) for b, t in types.items())
    return WorldState(now, T0 + timedelta(days=5), tuple(ships), (), (), blocks, (),
                      MappingProxyType(manifests), BERTHS)


# ---------------------------------------------------------------- impossible

def test_more_discharge_than_imports_is_impossible_reported_once_with_a_reason():
    st = world([ship("Z9", "expected", discharge=1600)], {"Z9": boxes("Z9", 551)})
    ev = S.detect(st)
    assert [(e.kind, e.subject) for e in ev] == [("ship_impossible", "Z9")]
    assert "1600" in ev[0].detail["message"] and "551" in ev[0].detail["message"]
    assert "never finish" in ev[0].detail["message"]
    assert S.detect(st) == []                                        # once
    assert S.impossible_calls(st) == {"Z9"}


def test_discharged_boxes_are_counted_so_a_normal_ship_is_never_impossible():
    # 300 imports; seen with 300 to discharge, later 100 left after 200 discharged (200 placed).
    before = world([ship("OK", "berthed", discharge=300)], {"OK": boxes("OK", 300)}, free={"A": 50})
    assert S.detect(before) == []
    after = world([ship("OK", discharge=100)], {"OK": boxes("OK", 200, placed="A") + boxes("OK", 100, start=200)},
                  free={"A": 50})
    assert S.detect(after) == [] and S.impossible_calls(after) == set()


def test_ship_first_seen_mid_discharge_is_still_caught_when_it_can_never_finish():
    st = world([ship("LATE", discharge=900)], {"LATE": boxes("LATE", 400, placed="A")}, free={"A": 50})
    assert [e.kind for e in S.detect(st)] == ["ship_impossible"]


def test_a_ship_with_no_manifest_rows_is_not_judged():
    st = world([ship("NOROWS", "expected", discharge=500)], {})
    assert S.detect(st) == [] and S.impossible_calls(st) == set()     # unknown data: YARD-5's job


def test_departed_or_finished_ships_are_not_flagged():
    st = world([ship("GONE", "departed", discharge=0)], {"GONE": boxes("GONE", 10, placed="A")})
    assert S.detect(st) == [] and S.impossible_calls(st) == set()
    left_unfinished = world([ship("GONE", "departed", discharge=50)], {"GONE": boxes("GONE", 10, placed="A")})
    assert S.detect(left_unfinished) == []                            # gone: nothing to flag


def test_impossible_stays_impossible_while_it_discharges_what_it_has():
    S.detect(world([ship("Z9", "berthed", discharge=600)], {"Z9": boxes("Z9", 551)}, free={"A": 50}))
    later = world([ship("Z9", discharge=500)], {"Z9": boxes("Z9", 100, placed="A") + boxes("Z9", 451, start=100)})
    assert S.impossible_calls(later) == {"Z9"}                        # 500 left > 551 - 100 discharged = 451


def test_no_ids_in_the_detector():
    src = (ROOT / "agents" / "stall.py").read_text(encoding="utf-8")
    strings = [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not [s for s in strings if s[:1] == "P" and s[1:3].isdigit()]


# ---------------------------------------------------------------- stalled

def test_stalled_when_nothing_to_discharge_and_no_legal_slot():
    # 100 imports, 40 placed and 40 discharged (60 left), 60 waiting, yard full.
    st = world([ship("S1", "berthed", discharge=100)], {"S1": boxes("S1", 100)}, free={"A": 50})
    S.detect(st)                                                      # first seen before discharging
    full = world([ship("S1", discharge=60)], {"S1": boxes("S1", 40, placed="A") + boxes("S1", 60, start=40)})
    ev = S.detect(full)
    assert [(e.kind, e.subject) for e in ev] == [("ship_stalled", "S1")]
    assert ev[0].detail["imports_without_slot"] == 60 and S.stalled_calls(full) == {"S1"}


def test_not_stalled_while_placed_boxes_wait_at_the_quay_or_there_is_room():
    S.detect(world([ship("S1", "berthed", discharge=100)], {"S1": boxes("S1", 100)}, free={"A": 50}))
    waiting = world([ship("S1", discharge=80)], {"S1": boxes("S1", 40, placed="A") + boxes("S1", 60, start=40)})
    assert S.stalled_calls(waiting) == set()                          # 20 placed boxes still to discharge
    room = world([ship("S1", discharge=60)], {"S1": boxes("S1", 40, placed="A") + boxes("S1", 60, start=40)},
                 free={"G": 5})
    assert S.stalled_calls(room) == set()                             # dry may go to G


def test_reefer_waits_even_with_standard_room_and_imdg_needs_h():
    S.detect(world([ship("R", "berthed", discharge=10)], {"R": boxes("R", 10, "reefer")}, free={"I": 50}))
    st = world([ship("R", discharge=10)], {"R": boxes("R", 10, "reefer")}, free={"A": 80, "G": 80})
    assert S.stalled_calls(st) == {"R"}
    S.reset()
    S.detect(world([ship("X", "berthed", discharge=10)], {"X": boxes("X", 10, "IMDG")}, free={"H": 50}))
    assert S.stalled_calls(world([ship("X", discharge=10)], {"X": boxes("X", 10, "IMDG")}, free={"I": 9})) == {"X"}
    assert S.stalled_calls(world([ship("X", discharge=10)], {"X": boxes("X", 10, "IMDG")}, free={"H": 1})) == set()


def test_a_ship_not_at_a_berth_is_never_stalled():
    st = world([ship("W", "arrived", discharge=50)], {"W": boxes("W", 50)})
    assert S.stalled_calls(st) == set()


def test_impossible_ships_are_not_reported_as_stalled():
    st = world([ship("Z9", discharge=1600)], {"Z9": boxes("Z9", 551)})            # no room either
    assert S.stalled_calls(st) == set() and [e.kind for e in S.detect(st)] == ["ship_impossible"]


def test_stalled_event_on_each_new_stall_not_every_tick():
    S.detect(world([ship("S1", "berthed", discharge=100)], {"S1": boxes("S1", 100)}, free={"A": 50}))
    full = world([ship("S1", discharge=60)], {"S1": boxes("S1", 40, placed="A") + boxes("S1", 60, start=40)})
    assert len(S.detect(full)) == 1 and S.detect(full) == []
    room = world([ship("S1", discharge=60)], {"S1": boxes("S1", 40, placed="A") + boxes("S1", 60, start=40)},
                 free={"A": 3})
    assert S.detect(room) == [] and len(S.detect(full)) == 1         # recovered, then stalled again


def test_never_raises_and_new_scenario_resets():
    assert S.detect(None) == [] and S.stalled_calls(None) == set() and S.impossible_calls(None) == set()
    st = world([ship("Z9", "expected", discharge=1600)], {"Z9": boxes("Z9", 551)})
    assert len(S.detect(st)) == 1
    other = world([ship("Z9", "expected", discharge=1600), ship("NEW", "expected")],
                  {"Z9": boxes("Z9", 551), "NEW": boxes("NEW", 100)})
    assert len(S.detect(other)) == 1                                  # different scenario: reported again


# ---------------------------------------------------------------- wiring

def test_events_hook_returns_the_detectors_events():
    from core.events import detect_events
    st = world([ship("Z9", "expected", discharge=1600)], {"Z9": boxes("Z9", 551)})
    assert "ship_impossible" in [e.kind for e in detect_events(None, st)]


@needs_workbook
def test_p03_004_flagged_by_the_general_check_and_escalated_once(tmp_path):
    import run
    run.run_one("practice_03", WORKBOOK, None, tmp_path, "multi")
    d = [json.loads(x) for x in (tmp_path / "practice_03" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    esc = [x for x in d if x["agent"] == "stall" and x["choice"] == "escalate"]
    assert len(esc) == 1 and esc[0]["reason"].startswith("P03-004 ") and "never finish" in esc[0]["reason"]
    berthed = min(x["t"] for x in d if x["choice"] == "assign_berth" and "P03-004" in json.dumps(x["saw"]) + x["reason"])
    assert esc[0]["t"] <= berthed + 1.0                               # well before it runs out of boxes
