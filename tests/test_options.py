"""BC-3 tests: option generator for the supervisor (agents/options.py).

The task's test: every option passes the guard (step by step, on the state as it would be).
Plus: option 0 is the agents' own plan, options only when plans genuinely compete, no
metric-only alternatives, crane options off until CRANE-2 but rail-safe when built, the
coordinator returns them, and the loop runs the chosen option first and drops the agents'
conflicting actions.
"""
import sys
import types
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import pytest

import guard
from agents import options as O
from core.skeleton import BERTHS
from core.types import (Action, Container, Crane, Option, Ship, WorldState)

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
T0 = datetime(2026, 9, 3, 6, 0)


def ship(cid, status="arrived", berth=None, loa=300.0, draft=14.0, arrived_h=-5.0, moves=400, cranes=(), load=0):
    arr = T0 + timedelta(hours=arrived_h)
    return Ship(cid, "MV " + cid, loa, draft, arr, arr if status != "expected" else None, status, berth,
                tuple(cranes), 0, moves, load, None)


def cranes(on=None):
    """QC1-QC6 at the skeleton's fixed bays; `on` = {crane: call_id} for working ones."""
    on = on or {}
    bays = {"QC1": 5, "QC2": 20, "QC3": 35, "QC4": 50, "QC5": 62, "QC6": 75}
    return tuple(Crane(c, "working" if c in on else "available", on.get(c), b) for c, b in bays.items())


def world(ships, crs=None, manifests=None, days_left=3.0):
    return WorldState(T0, T0 + timedelta(days=days_left), tuple(ships), crs or cranes(), (), (), (),
                      MappingProxyType(manifests or {}), BERTHS)


def b1_busy_b2_b3_free():
    # Big ships (LOA 300 m) fit only B1. B1 is free; B2/B3 are free but too short.
    return [ship("OLD", arrived_h=-8, moves=900), ship("SHORT", arrived_h=-2, moves=150)]


# ---------------------------------------------------------------- berth choice

def test_berth_agents_plan_is_option_0_and_oldest_first_is_offered_when_it_changes_the_outcome():
    st = world(b1_busy_b2_b3_free())
    opts = O.build(st)
    assert [o.id for o in opts] == ["berth_agent_plan", "oldest_first"]
    assert opts[0].actions[0].args == {"call_id": "SHORT", "berth_id": "B1"}      # BERTH-2: least work first
    assert opts[1].actions[0].args == {"call_id": "OLD", "berth_id": "B1"}
    a, b = opts[0].predicted, opts[1].predicted
    assert b["avg_finish_h"] - a["avg_finish_h"] >= O.MIN_GAIN_H
    assert set(a["finish_h"]) == {"OLD", "SHORT"} and b["waited_h"] == 8.0
    for o in opts:
        assert o.template and "yard space not modelled" in o.template


def test_every_option_passes_the_guard():
    st = world(b1_busy_b2_b3_free())
    for o in O.build(st):
        assert O.rule_safe(o, st) == (True, "ok")
        for act in o.actions:
            assert guard.check(act, st)[0]


def test_no_options_when_every_waiting_ship_has_a_free_berth():
    ships = [ship("A", loa=180, draft=11, arrived_h=-3, moves=800), ship("B", loa=180, draft=11, arrived_h=-1, moves=100)]
    assert O.build(world(ships)) == []                   # B1, B2, B3 all free and both fit several


def test_no_options_when_a_berth_is_taken_and_the_rest_is_too_short():
    ships = b1_busy_b2_b3_free() + [ship("ONB1", "working", "B1", moves=500, cranes=("QC1", "QC2"))]
    assert O.build(world(ships, cranes({"QC1": "ONB1", "QC2": "ONB1"}))) == []


def test_no_metric_only_alternative():
    # Same arrival order effect within MIN_GAIN_H: the alternative changes nothing that matters.
    ships = [ship("OLD", arrived_h=-8, moves=200), ship("NEW", arrived_h=-2, moves=190)]
    assert O.build(world(ships)) == []


def only_fits_world():
    ships = [ship("SMALL", loa=180, draft=11, arrived_h=-9, moves=300),   # fits all berths, oldest
             ship("MID", loa=250, draft=12.5, arrived_h=-6, moves=100),   # fits B1/B2, least work
             ship("BIG", loa=300, draft=14, arrived_h=-4, moves=1500)]    # fits only B1
    on = {"QC3": "X2", "QC4": "X2", "QC5": "X3", "QC6": "X3"}
    busy = [ship("X2", "working", "B2", moves=2000, cranes=("QC3", "QC4")),
            ship("X3", "working", "B3", moves=2000, cranes=("QC5", "QC6"))]
    return world(ships + busy, cranes(on), days_left=2.0)


def test_ship_that_fits_only_this_berth_is_offered():
    st = only_fits_world()
    opts = O.build(st)
    assert opts and opts[0].predicted["ship"] == "MID"
    assert [o.id for o in opts] == ["berth_agent_plan", "oldest_first", "only_fits_here"]
    assert "only_fits_here" in [o.id for o in opts] and all(O.rule_safe(o, st)[0] for o in opts)


def test_an_option_the_guard_refuses_is_dropped_and_option_0_refused_means_none(monkeypatch):
    st = world(b1_busy_b2_b3_free())
    real = guard.check
    monkeypatch.setattr(guard, "check", lambda a, s: (False, "test") if a.args["call_id"] == "OLD" else real(a, s))
    assert O.build(st) == []
    monkeypatch.setattr(guard, "check", lambda a, s: (False, "test") if a.args["call_id"] == "SHORT" else real(a, s))
    assert O.build(st) == []                             # only one safe option left: no choice to make
    three = only_fits_world()
    assert len(O.build(three)) >= 2
    monkeypatch.setattr(guard, "check", lambda a, s: (False, "test") if a.args["call_id"] == "MID" else real(a, s))
    assert O.build(three) == []                          # option 0 refused: never offer only the others


def test_never_raises():
    assert O.build(world([ship("A", loa=None)])) == []
    assert O.build(None) == []


# ---------------------------------------------------------------- crane choice (off until CRANE-2)

def stall_world():
    # STUCK on B2 has QC3/QC4 but no placed boxes to discharge; BUSY on B1 has placed boxes.
    # STUCK: 600 imports, 100 placed and already discharged (500 moves left), the rest wait for slots.
    stuck = ship("STUCK", "working", "B2", loa=250, draft=12, moves=500, cranes=("QC3", "QC4"))
    busy = ship("BUSY", "working", "B1", moves=600, cranes=("QC1", "QC2"))
    man = {"STUCK": tuple(Container(f"s{i}", "STUCK", "import", 40, "dry", None, 20.0, "A" if i < 100 else None)
                          for i in range(600)),
           "BUSY": tuple(Container(f"b{i}", "BUSY", "import", 40, "dry", None, 20.0, "A") for i in range(600))}
    return world([stuck, busy], cranes({"QC1": "BUSY", "QC2": "BUSY", "QC3": "STUCK", "QC4": "STUCK"}), man)


def test_crane_options_are_off_until_crane_2():
    assert O.CRANE_OPTIONS is False and O.build(stall_world()) == []


def test_crane_options_when_on_keep_is_option_0_and_the_move_is_rail_safe_step_by_step(monkeypatch):
    st = stall_world()
    opts = O.crane_options(st)
    assert [o.id for o in opts] == ["keep_cranes", "move_cranes"] and opts[0].actions == ()
    move = opts[1]
    assert move.predicted["extra_moves_per_h"] == 46 and move.actions
    assert O.rule_safe(move, st) == (True, "ok")
    after = st
    for a in move.actions:
        after = O.apply(a, after)
    assert sorted(after.ship("BUSY").cranes) == ["QC1", "QC2", "QC3", "QC4"] and after.ship("STUCK").cranes == ()
    bays = [after.crane(c).bay for c in ("QC1", "QC2", "QC3", "QC4", "QC5", "QC6")]
    assert bays == sorted(bays) and len(set(bays)) == 6                  # rail order kept
    monkeypatch.setattr(O, "CRANE_OPTIONS", True)
    assert [o.id for o in O.build(st)] == ["keep_cranes", "move_cranes"]


def test_no_crane_option_when_no_rail_safe_plan_exists():
    st = stall_world()
    crs = tuple(replace(c, status="out_of_service", call_id=None) if c.crane_id == "QC3" else c for c in st.cranes)
    ships = tuple(replace(s, cranes=("QC4",)) if s.call_id == "STUCK" else s for s in st.ships)
    assert O.crane_options(replace(st, cranes=crs, ships=ships)) == []    # QC4 can't pass broken QC3


def test_apply_releases_cranes_a_ship_no_longer_lists():
    st = stall_world()
    after = O.apply(Action("plan_crane_sequence", {"call_id": "STUCK", "crane_ids": [], "bay_positions": []},
                           "t", "r"), st)
    assert after.crane("QC3").call_id is None and after.crane("QC3").status == "available"
    assert after.crane("QC3").bay == 35 and after.ship("STUCK").cranes == ()


def test_no_crane_option_when_nothing_is_stalled():
    st = stall_world()
    man = dict(st.manifests)
    man["STUCK"] = tuple(replace(c, yard_block="A") for c in man["STUCK"])
    assert O.crane_options(replace(st, manifests=MappingProxyType(man))) == []


# ---------------------------------------------------------------- coordinator and loop

def test_coordinator_returns_berth_options_and_still_writes_the_transcript(tmp_path, monkeypatch):
    import coordinator as C
    from core import log as runlog
    from core.types import Request
    C.reset()
    monkeypatch.setattr(C, "_forecast", lambda st: (None, None))
    st = world(b1_busy_b2_b3_free())
    runlog.start(tmp_path)
    try:
        grants, opts = C.decide([Request("trucks", "trucks", 1, "OLD", 1.0, "x")], st)
    finally:
        runlog.stop()
    assert [o.id for o in opts] == ["berth_agent_plan", "oldest_first"]
    assert (tmp_path / "negotiation.jsonl").read_text(encoding="utf-8").strip()


def test_loop_runs_the_chosen_option_first_and_drops_conflicting_agent_actions(tmp_path, monkeypatch):
    from core.loop import Loop, _conflicts
    from tests.test_executor import FakeAPI
    pick = Option("oldest_first", (Action("assign_berth", {"call_id": "C1", "berth_id": "B2"}, "options", "pick"),),
                  {}, "t")
    coord = types.ModuleType("coordinator")
    coord.decide = lambda reqs, st: ([], [Option("berth_agent_plan", (), {}, "t0"), pick])
    monkeypatch.setitem(sys.modules, "coordinator", coord)
    sup = types.ModuleType("llm.supervisor")
    sup.decide = lambda ev, opts, state=None: ("oldest_first", "because")
    monkeypatch.setitem(sys.modules, "llm.supervisor", sup)

    class Agent:
        name = "berth"

        def propose(self, state, events):
            return [Action("assign_berth", {"call_id": "C1", "berth_id": "B1"}, "berth", "oldest first")], []

    api = FakeAPI()
    Loop(api, tmp_path, agents=[Agent()]).step()
    berths = [a for t, a in api.calls if t == "assign_berth"]
    assert berths == [{"call_id": "C1", "berth_id": "B2"}]              # the pick, once; the agent's dropped
    assert _conflicts(pick.actions[0], Action("assign_berth", {"call_id": "X", "berth_id": "B2"}, "a", "r"))
    assert not _conflicts(pick.actions[0], Action("place_container", {"call_id": "C1"}, "a", "r"))


@needs_workbook
def test_practice_run_is_violation_free(tmp_path):
    import run
    s = run.run_one("practice_03", WORKBOOK, None, tmp_path, "multi")
    assert s["violations"] == [] and len(s["unfinished_calls"]) == 2
