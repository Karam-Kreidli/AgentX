"""BERTH-1 tests: fit rules, occupied berth, not-yet-arrived ship, and no refused
berth calls across the 5 practice scenarios."""
import json
from datetime import datetime, timedelta

from agents.berth import BerthAgent
from core.skeleton import BERTHS
from core.types import Ship, WorldState
from guard import check

T0 = datetime(2026, 9, 1, 6, 0)


def ship(call_id, status="arrived", loa=200.0, draft=11.0, berth=None, arrived=T0):
    return Ship(call_id, "MV " + call_id, loa, draft, T0, None if status == "expected" else arrived,
                status, berth, (), 0, 100, 50, None)


def world(*ships):
    return WorldState(T0, T0 + timedelta(days=5), tuple(ships), (), (), (), (), {}, BERTHS)


def plan(state):
    agent = BerthAgent()
    actions, requests = agent.propose(state, [])
    return {a.args["call_id"]: a.args["berth_id"] for a in actions}, actions, agent


# ---------------------------------------------------------------- fit rules

def test_big_ship_only_fits_b1():
    got, actions, _ = plan(world(ship("BIG", loa=290.0, draft=14.0)))
    assert got == {"BIG": "B1"}
    assert "fits no other berth" in actions[0].reason


def test_small_ship_takes_first_free_berth_it_fits():
    got, _, _ = plan(world(ship("S", loa=180.0, draft=11.0)))
    assert got == {"S": "B1"}
    got, _, _ = plan(world(ship("S", loa=180.0, draft=11.0), ship("X", status="working", berth="B1")))
    assert got == {"S": "B2"}


def test_fit_edges_length_plus_20_and_draft():
    # B3 is 220 m / 12.0 m: LOA 200 fits exactly, 200.1 does not; draft 12.0 fits, 12.1 does not.
    busy = [ship("X1", status="working", berth="B1"), ship("X2", status="working", berth="B2")]
    assert plan(world(ship("OK", loa=200.0, draft=12.0), *busy))[0] == {"OK": "B3"}
    assert plan(world(ship("LONG", loa=200.1, draft=12.0), *busy))[0] == {}
    assert plan(world(ship("DEEP", loa=150.0, draft=12.1), *busy))[0] == {}


def test_ship_that_fits_no_berth_is_escalated_once():
    s = world(ship("HUGE", loa=310.0, draft=14.0))
    agent = BerthAgent()
    assert agent.propose(s, [])[0] == [] and len(agent.escalations) == 1
    agent.propose(s, [])
    assert len(agent.escalations) == 1
    assert "fits no berth" in agent.escalations[0]


def test_bad_size_data_is_escalated_not_berthed():
    _, actions, agent = plan(world(ship("NAN", loa=None), ship("NEG", draft=-1.0)))
    assert actions == [] and len(agent.escalations) == 2


# ---------------------------------------------------------------- occupied berths

def test_berth_busy_while_berthed_working_or_sailing():
    for status in ("berthed", "working", "sailing"):
        got, _, _ = plan(world(ship("BIG", loa=290.0, draft=14.0), ship("X", status=status, berth="B1")))
        assert got == {}, status                       # BIG waits for B1
    got, _, _ = plan(world(ship("BIG", loa=290.0, draft=14.0), ship("X", status="departed", berth="B1")))
    assert got == {"BIG": "B1"}


def test_two_ships_same_tick_never_share_a_berth_oldest_first():
    early, late = ship("LATE", arrived=T0), ship("EARLY", arrived=T0 - timedelta(hours=3))
    got, actions, _ = plan(world(early, late))
    assert got == {"EARLY": "B1", "LATE": "B2"}
    assert [a.args["call_id"] for a in actions] == ["EARLY", "LATE"]


# ---------------------------------------------------------------- not arrived

def test_not_yet_arrived_ship_is_never_berthed():
    got, _, _ = plan(world(ship("SOON", status="expected")))
    assert got == {}


def test_every_proposal_passes_the_guard_and_has_a_reason():
    s = world(ship("A", loa=290.0, draft=14.0), ship("B", loa=250.0), ship("C", loa=150.0),
              ship("D", status="expected"))
    _, actions, _ = plan(s)
    assert len(actions) == 3
    for a in actions:
        assert check(a, s) == (True, "ok") and a.reason


# ---------------------------------------------------------------- the 5 scenarios

def test_no_refused_berth_calls_across_the_5_scenarios(multi_runs):
    for scenario, (summary, run_dir) in multi_runs.items():
        decisions = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines()]
        berth = [d for d in decisions if d["choice"] == "assign_berth"]
        assert berth and all(d["agent"] == "berth" for d in berth), scenario
        bad = [d for d in berth if not d["guard_ok"] or "error" in (d["tool_result"] or {})]
        assert bad == [], (scenario, bad[:1])
        assert summary["violations"] == []
