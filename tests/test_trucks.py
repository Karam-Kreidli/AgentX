"""YARD-2 tests: 3 trucks per crane, broken trucks replaced the same tick, extras
released, shortages sent to the coordinator as Requests, working ships served
first, every action passes the guard, and across the 5 practice scenarios no
working crane is left below 3 trucks while trucks sit idle."""
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agents.trucks import PER_CRANE, TruckAgent
from core.skeleton import BERTHS
from core.types import RES_TRUCKS, Ship, Truck, WorldState
from guard import check

T0 = datetime(2026, 9, 1, 6, 0)


def ship(call_id="C1", status="working", cranes=("QC1",), berth="B1"):
    return Ship(call_id, "MV", 200.0, 11.0, T0, T0, status, berth, tuple(cranes), 3, 100, 10, None)


def trucks(n, call=None, status=None, start=1):
    st = status or ("working" if call else "available")
    return [Truck(f"T{i:02d}", st, call) for i in range(start, start + n)]


def world(ships, fleet):
    return WorldState(T0, T0 + timedelta(days=5), tuple(ships), (), tuple(fleet), (), (), {}, BERTHS)


def run(state):
    actions, requests = TruckAgent().propose(state, [])
    final = {a.args["call_id"]: a.args["truck_ids"] for a in actions}
    return final, actions, requests


def apply(action, state):
    """What the simulator does with dispatch_truck: the list replaces the ship's trucks."""
    call, ids = action.args["call_id"], set(action.args["truck_ids"])
    fleet = []
    for t in state.trucks:
        if t.truck_id in ids:
            fleet.append(Truck(t.truck_id, "working", call))
        elif t.call_id == call:
            fleet.append(Truck(t.truck_id, "out_of_service" if t.status == "out_of_service" else "available", None))
        else:
            fleet.append(t)
    return replace(state, trucks=tuple(fleet))


def guard_ok(actions, state):
    # The executor re-reads the state after every call, so each action is checked
    # against the state left by the actions before it (releases go first).
    for a in actions:
        assert check(a, state) == (True, "ok"), a
        assert a.reason and a.agent == "trucks"
        state = apply(a, state)


# ---------------------------------------------------------------- the target

def test_new_ship_gets_three_trucks_per_crane():
    s = world([ship(cranes=("QC1", "QC2"))], trucks(10))
    final, actions, requests = run(s)
    assert len(final["C1"]) == 2 * PER_CRANE and requests == []
    guard_ok(actions, s)


def test_ship_at_target_gets_no_action():
    s = world([ship()], trucks(3, "C1") + trucks(5, start=4))
    final, actions, requests = run(s)
    assert actions == [] and requests == []


def test_ship_with_no_cranes_releases_all_its_trucks():
    s = world([ship(cranes=())], trucks(4, "C1"))
    final, actions, _ = run(s)
    assert final == {"C1": []}
    guard_ok(actions, s)


def test_ships_not_at_berth_are_ignored():
    s = world([ship(status="waiting", berth=None), ship("C2", status="departed")], trucks(6))
    assert run(s) == ({}, [], [])


# ---------------------------------------------------------------- breakdowns

def test_broken_truck_is_replaced_the_same_tick():
    fleet = trucks(2, "C1") + [Truck("T03", "out_of_service", "C1")] + trucks(3, start=4)
    s = world([ship()], fleet)
    final, actions, requests = run(s)
    assert "T03" not in final["C1"] and len(final["C1"]) == 3
    assert {"T01", "T02"} <= set(final["C1"]) and requests == []
    assert len(actions) == 1                       # one final list, not release + add
    guard_ok(actions, s)


def test_broken_truck_dropped_even_when_no_replacement():
    fleet = trucks(2, "C1") + [Truck("T03", "out_of_service", "C1")]
    s = world([ship()], fleet)
    final, actions, requests = run(s)
    assert final["C1"] == ["T01", "T02"]
    assert [(r.resource, r.amount, r.call_id) for r in requests] == [(RES_TRUCKS, 1, "C1")]
    guard_ok(actions, s)


def test_broken_and_busy_trucks_are_never_assigned():
    fleet = [Truck("T01", "out_of_service", None), Truck("T02", "working", "C9")] + trucks(3, start=3)
    s = world([ship()], fleet)
    final, actions, _ = run(s)
    assert set(final["C1"]) == {"T03", "T04", "T05"}
    guard_ok(actions, s)


# ---------------------------------------------------------------- extras

def test_extras_released_when_cranes_drop_and_reused_same_tick():
    # C1 dropped from 2 cranes to 1: 3 extras; C2 has 1 crane and no trucks, none free.
    s = world([ship(), ship("C2", cranes=("QC4",), berth="B3")], trucks(6, "C1"))
    final, actions, requests = run(s)
    assert len(final["C1"]) == 3 and len(final["C2"]) == 3
    assert not set(final["C1"]) & set(final["C2"]) and requests == []
    guard_ok(actions, s)


# ---------------------------------------------------------------- shortages

def test_shortage_becomes_a_request_with_the_gap():
    s = world([ship(cranes=("QC1", "QC2"))], trucks(4))
    final, _, requests = run(s)
    assert len(final["C1"]) == 4
    assert len(requests) == 1
    r = requests[0]
    assert (r.agent, r.resource, r.amount, r.call_id) == ("trucks", RES_TRUCKS, 2, "C1") and r.reason


def test_working_ships_served_before_berthed_ones():
    s = world([ship("C1", status="berthed", berth="B1"), ship("C2", berth="B3", cranes=("QC4",))],
              trucks(3))
    final, _, requests = run(s)
    assert len(final["C2"]) == 3 and "C1" not in final
    assert [(r.call_id, r.priority) for r in requests] == [("C1", 0.5)]


def test_more_cranes_served_first_among_working_ships():
    s = world([ship("C1", cranes=("QC1",)), ship("C2", berth="B3", cranes=("QC4", "QC5"))], trucks(6))
    final, _, requests = run(s)
    assert len(final["C2"]) == 6 and "C1" not in final
    assert [(r.call_id, r.amount, r.priority) for r in requests] == [("C1", 3, 1.0)]


def test_never_assigns_beyond_three_per_crane():
    s = world([ship()], trucks(20))
    final, _, _ = run(s)
    assert len(final["C1"]) == PER_CRANE          # no spare trucks to game idle hours


# ---------------------------------------------------------------- 5 scenarios

@pytest.fixture(scope="module")
def workbook():
    path = Path(__file__).resolve().parents[1] / "data" / "P2_Container_Terminal_Data.xlsx"
    if not path.exists():
        pytest.skip("workbook not in data/")
    return str(path)


@pytest.mark.parametrize("scenario", [f"practice_0{i}" for i in range(1, 6)])
def test_no_working_crane_short_while_trucks_idle(scenario, tmp_path, workbook):
    from core.loop import Loop
    from core.state import build_state
    from run import make_agents
    from vendor import TerminalAPI

    api = TerminalAPI(scenario, data_xlsx=workbook, log_path=str(tmp_path / "actions.jsonl"))
    loop = Loop(api, Path(tmp_path), agents=make_agents("multi"))
    while True:
        loop.step()
        s = build_state(api)
        idle = [t for t in s.trucks if t.status == "available"]
        for sh in s.ships:
            if sh.status == "working" and sh.cranes:
                healthy = [t for t in s.trucks if t.call_id == sh.call_id and t.status != "out_of_service"]
                assert len(healthy) >= PER_CRANE * len(sh.cranes) or not idle, \
                    (s.now, sh.call_id, len(sh.cranes), len(healthy), len(idle))
        if s.now >= s.scenario_end:
            break
        api.call("advance_time", {"minutes": loop._tick_minutes(s)})
        loop.tick += 1
    assert api.violations == []
