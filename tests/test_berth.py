"""BERTH-1/BERTH-2 tests: fit rules, occupied berth, not-yet-arrived ship, best fit, least work
first, impossible ships last, keeping a berth for a ship that only fits there, negotiation with
the coordinator, no refused berth calls and lower waits than BERTH-1 on the practice scenarios."""
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


def test_small_ship_takes_the_smallest_free_berth_it_fits():
    got, _, _ = plan(world(ship("S", loa=180.0, draft=11.0)))
    assert got == {"S": "B3"}
    got, _, _ = plan(world(ship("M", loa=220.0, draft=12.2)))          # the practice_02 case: B2, not B1
    assert got == {"M": "B2"}
    got, _, _ = plan(world(ship("S", loa=180.0, draft=11.0), ship("X", status="working", berth="B3")))
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


def test_two_ships_same_tick_never_share_a_berth_equal_work_oldest_first():
    early, late = ship("LATE", arrived=T0), ship("EARLY", arrived=T0 - timedelta(hours=3))
    got, actions, _ = plan(world(early, late))
    assert got == {"EARLY": "B3", "LATE": "B2"}
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


# ---------------------------------------------------------------- BERTH-2 priorities

def work(call_id, moves, **kw):
    s = ship(call_id, **kw)
    from dataclasses import replace
    return replace(s, discharge_remaining=moves, load_remaining=0)


def test_least_work_first_when_ships_compete_for_one_berth():
    busy = [ship("X2", status="working", berth="B2"), ship("X3", status="working", berth="B3")]
    old_big = work("OLDBIG", 900, arrived=T0 - timedelta(hours=5))
    new_small = work("NEWSMALL", 150, arrived=T0 - timedelta(hours=1))
    assert plan(world(old_big, new_small, *busy))[0] == {"NEWSMALL": "B1"}


def test_impossible_ships_go_last(monkeypatch):
    import agents.berth as B
    monkeypatch.setattr(B, "_impossible", lambda st: {"HOPELESS"})
    busy = [ship("X2", status="working", berth="B2"), ship("X3", status="working", berth="B3")]
    got = plan(world(work("HOPELESS", 10), work("FINE", 800), *busy))[0]
    assert got == {"FINE": "B1"}


def test_berth_kept_for_a_ship_that_only_fits_there_when_it_is_due_soon():
    from dataclasses import replace
    busy = [ship("X2", status="working", berth="B2"), ship("X3", status="working", berth="B3")]
    flexible = ship("FLEX", loa=200.0)                                     # fits B1/B2/B3
    big_soon = replace(ship("BIG", status="expected", loa=290.0, draft=14.0), scheduled_eta=T0 + timedelta(hours=1))
    assert plan(world(flexible, big_soon, *busy))[0] == {}                 # B1 kept for BIG
    big_later = replace(big_soon, scheduled_eta=T0 + timedelta(hours=10))
    assert plan(world(flexible, big_later, *busy))[0] == {"FLEX": "B1"}
    flex_soon = replace(ship("FLEX2", status="expected", loa=200.0), scheduled_eta=T0 + timedelta(hours=1))
    assert plan(world(flexible, flex_soon, *busy))[0] == {"FLEX": "B1"}    # a ship that fits anywhere holds nothing
    big_arrived_only = ship("BIG2", loa=290.0, draft=14.0)                # a B1-only ship itself still gets B1
    assert plan(world(big_arrived_only, big_soon, *busy))[0] == {"BIG2": "B1"}


def test_requests_for_the_coordinator_and_grants_are_binding():
    busy = [ship("X2", status="working", berth="B2"), ship("X3", status="working", berth="B3")]
    s = world(ship("BIG", loa=290.0, draft=14.0, arrived=T0 - timedelta(hours=2)), ship("BIG2", loa=290.0, draft=14.0), *busy)
    agent = BerthAgent()
    acts, reqs = agent.propose(s, [])
    assert [(r.resource, r.call_id) for r in reqs] == [("berth", "BIG"), ("berth", "BIG2")]
    assert reqs[0].priority > reqs[1].priority and len(acts) == 1
    agent.grants = {}
    assert agent.propose(s, [])[0] == []                                    # nothing granted: nothing berthed
    agent.grants = {("berth", "BIG"): 1}
    assert [a.args["call_id"] for a in agent.propose(s, [])[0]] == ["BIG"]


def test_coordinator_only_grants_a_berth_the_ship_fits():
    import coordinator as C
    from core.types import Request
    C.reset()
    s = world(ship("BIG", loa=290.0, draft=14.0), ship("X1", status="working", berth="B1"))
    g = C.decide([Request("berth", "berth", 1, "BIG", 1.0, "x")], s)[0][0]
    assert g.status == "refused" and "No free berth fits it" in g.reason     # B2, B3 free but too short


def _avg_wait(run_dir):
    first = {}
    for line in (run_dir / "ticks.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for x in r["ships"]:
            first.setdefault((x["id"], x["status"]), r["t"])
    w = [first[(i, "berthed")] - first[(i, "arrived")] for (i, st) in first if st == "arrived" and (i, "berthed") in first]
    return sum(w) / len(w)


def test_lower_average_wait_than_berth_1_on_practice_02_and_05(multi_runs):
    berth_1 = {"practice_02": 6.47, "practice_05": 5.83}          # main on 7 Nov (first free berth, oldest first)
    for scenario, before in berth_1.items():
        summary, run_dir = multi_runs[scenario]
        assert _avg_wait(run_dir) < before, scenario
