"""CRANE-1 tests: safe_crane_moves() never produces a step that breaks a crane rule.

- Property test: 10,000 random targets; every intermediate step is checked with SAFE-1's
  guard (guard.check) against the state at that moment, then applied. Zero violations allowed.
- Completeness: whenever the target is itself a safe crane layout, the plan reaches it.
- Hand cases: broken crane in the middle, moving across an empty berth, releasing cranes,
  impossible targets (partial plan + reason).
- Real simulator: plans applied with plan_crane_sequence record no violation.
"""
import random
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agents.crane_moves import CallPlan, MovePlan, plans_from_state, safe_crane_moves, unsafe
from core.skeleton import BERTHS
from core.types import Crane, Ship, WorldState
from guard import check

CRANES = ["QC1", "QC2", "QC3", "QC4", "QC5", "QC6"]
HOME = {"QC1": 1, "QC2": 14, "QC3": 27, "QC4": 40, "QC5": 53, "QC6": 66}
BAYS = {b.berth_id: (b.start_bay, b.end_bay) for b in BERTHS}
T0 = datetime(2026, 9, 1, 6)
ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"


# --------------------------------------------------------------------------- a tiny crane world

def world(pos, broken, ships):
    """WorldState for the guard: ships = {call_id: (berth_id, [cranes])}."""
    on = {c: call for call, (_, cs) in ships.items() for c in cs}
    cranes = tuple(Crane(c, "out_of_service" if c in broken else ("working" if c in on else "available"),
                         on.get(c), pos[c]) for c in CRANES)
    ship_objs = tuple(Ship(call, "MV", 150.0, 10.0, T0, T0, "working", berth, tuple(cs), 0, 10, 10, None)
                      for call, (berth, cs) in ships.items())
    return WorldState(T0, T0 + timedelta(days=5), ship_objs, cranes, (), (), (), {}, BERTHS)


def apply(pos, ships, action):
    """What the simulator does with plan_crane_sequence."""
    call = action.args["call_id"]
    pos = {**pos, **dict(zip(action.args["crane_ids"], action.args["bay_positions"]))}
    ships = {k: (b, list(action.args["crane_ids"]) if k == call else list(cs)) for k, (b, cs) in ships.items()}
    return pos, ships


def run_plan(pos, broken, ships, want):
    """Make a plan, check every step with the guard, apply it. Returns (plan, final pos, final ships)."""
    target = {call: CallPlan(*BAYS[b], tuple(cs), want.get(call, {c: pos[c] for c in cs if c not in broken}))
              for call, (b, cs) in ships.items()}
    plan = safe_crane_moves(pos, broken, target)
    assert isinstance(plan, MovePlan) and isinstance(plan, list)
    for i, a in enumerate(plan):
        ok, why = check(a, world(pos, broken, ships))
        assert ok, f"step {i} refused by the guard: {why}\n{a.args}"
        assert a.reason
        pos, ships = apply(pos, ships, a)
        assert unsafe(pos, {k: [c for c in cs if c not in broken] for k, (_, cs) in ships.items()},
                      broken, target) is None
    return plan, pos, ships, target


# --------------------------------------------------------------------------- random situations

def random_case(rng):
    # Random strictly increasing positions over bays 1..80.
    bays = sorted(rng.sample(range(1, 81), 6))
    pos = dict(zip(CRANES, bays))
    broken = set(rng.sample(CRANES, rng.choice([0, 0, 1, 1, 2])))
    berths = rng.sample(["B1", "B2", "B3"], rng.randint(1, 3))
    ships, used = {}, set()
    for i, b in enumerate(berths):
        lo, hi = BAYS[b]
        # current cranes: some in-service cranes already inside this berth, spaced >= 2 apart
        inside = [c for c in CRANES if lo <= pos[c] <= hi and c not in used and c not in broken]
        cur = []
        for c in inside:
            if rng.random() < 0.6 and all(abs(pos[c] - pos[x]) >= 2 for s in ships.values() for x in s[1]) \
                    and all(abs(pos[c] - pos[x]) >= 2 for x in cur):
                cur.append(c)
        used |= set(cur)
        ships[f"C{i + 1}"] = (b, cur)
    # Random target: each ship wants a random set of cranes at random bays inside its berth.
    free = [c for c in CRANES if c not in broken]
    rng.shuffle(free)
    want = {}
    for call, (b, _) in ships.items():
        lo, hi = BAYS[b]
        k = rng.randint(0, min(3, len(free)))
        pick, free = free[:k], free[k:]
        want[call] = {c: rng.randint(lo, hi) for c in pick}
    if rng.random() < 0.5:     # make it more often feasible: sort wanted bays to match rail order
        allw = sorted(((c, call) for call, w in want.items() for c in w), key=lambda x: int(x[0][2:]))
        bays = sorted(want[call][c] for c, call in allw)
        for (c, call), bay in zip(allw, bays):
            want[call][c] = bay
    return pos, broken, ships, want


def test_10000_random_targets_zero_intermediate_violations():
    rng = random.Random(2026)
    complete = feasible = 0
    for n in range(10_000):
        pos, broken, ships, want = random_case(rng)
        plan, end_pos, end_ships, target = run_plan(pos, broken, ships, want)
        final_pos = {**pos, **{c: b for w in want.values() for c, b in w.items()}}
        reachable = unsafe(final_pos, {k: list(w) for k, w in want.items()} |
                           {k: [c for c in cs if c not in broken] for k, (_, cs) in ships.items() if k not in want},
                           broken, target) is None
        if plan.complete:
            complete += 1
            for call, w in want.items():                     # target reached exactly
                assert sorted(end_ships[call][1]) == sorted(w), (n, call)
                assert all(end_pos[c] == b for c, b in w.items()), (n, call)
        else:
            assert plan.reason, n
        if reachable:
            feasible += 1
            assert plan.complete, f"case {n}: target is safe but the plan stopped: {plan.reason}"
    assert complete == feasible and 1_000 < feasible < 9_000      # both kinds were exercised (~1,760 reachable)


# --------------------------------------------------------------------------- hand cases

def test_broken_crane_in_the_middle_is_never_passed():
    # QC3 broken at bay 27 between B1 and B2. B2's ship wants QC2 on the far side of it: impossible.
    pos = dict(HOME)
    plan, *_ = run_plan(pos, {"QC3"}, {"C1": ("B2", [])}, {"C1": {"QC2": 35}})
    assert not plan.complete and plan == [] and "QC3" in plan.reason
    # The same ship can use QC4 and QC5 to the right of the broken crane.
    plan, end_pos, end_ships, _ = run_plan(pos, {"QC3"}, {"C1": ("B2", [])}, {"C1": {"QC4": 40, "QC5": 50}})
    assert plan.complete and sorted(end_ships["C1"][1]) == ["QC4", "QC5"] and end_pos["QC3"] == 27


def test_moving_across_an_empty_berth_in_a_safe_order():
    # B2 is empty. The ship on B3 wants QC4 (now at 40, in B2's range) plus QC5 and QC6 further right.
    # QC6 and QC5 must move right first, otherwise QC4 would pass them.
    pos = dict(HOME)
    plan, end_pos, end_ships, _ = run_plan(pos, set(), {"C3": ("B3", [])},
                                           {"C3": {"QC4": 60, "QC5": 70, "QC6": 78}})
    assert plan.complete
    order = [a.args["crane_ids"] for a in plan]
    moved = [next(c for c in ids if end_pos[c] == a.args["bay_positions"][ids.index(c)]) for a, ids in zip(plan, order)]
    assert [a.args["crane_ids"] for a in plan][-1] == ["QC4", "QC5", "QC6"]
    assert end_pos == {**HOME, "QC4": 60, "QC5": 70, "QC6": 78}


def test_swap_cranes_between_ships_releases_first():
    pos = {**HOME, "QC2": 20, "QC3": 35}
    ships = {"C1": ("B1", ["QC1", "QC2"]), "C2": ("B2", ["QC3"])}
    plan, end_pos, end_ships, _ = run_plan(pos, set(), ships, {"C1": {"QC1": 5}, "C2": {"QC2": 31, "QC3": 38}})
    assert plan.complete
    assert plan[0].args == {"call_id": "C1", "crane_ids": ["QC1"], "bay_positions": [1]}   # release QC2 first
    assert end_ships["C2"][1] == ["QC2", "QC3"]


def test_spacing_respected_with_other_ships_working():
    # C1 keeps QC2 at bay 30 (B1's last bay); C2 on B2 wants QC3 (idle at 33) at 31: 1 bay apart, never allowed.
    pos = {**HOME, "QC2": 30, "QC3": 33}
    plan, *_ = run_plan(pos, set(), {"C1": ("B1", ["QC2"]), "C2": ("B2", [])}, {"C2": {"QC3": 31}})
    assert not plan.complete and "2 bays" in plan.reason
    plan, *_ = run_plan(pos, set(), {"C1": ("B1", ["QC2"]), "C2": ("B2", [])}, {"C2": {"QC3": 32}})
    assert plan.complete


def test_nothing_to_do_gives_an_empty_complete_plan():
    pos = {**HOME, "QC1": 5}
    plan, *_ = run_plan(pos, set(), {"C1": ("B1", ["QC1"])}, {"C1": {"QC1": 5}})
    assert plan == [] and plan.complete


@pytest.mark.parametrize("want, words", [
    ({"C1": {"QC9": 5}}, "unknown crane"),
    ({"C1": {"QC1": 5}, "C2": {"QC1": 40}}, "wanted by both"),
    ({"C2": {"QC3": 30}}, "outside"),                    # bay 30 is outside B2 (31-58), rail order fine
])
def test_impossible_targets_return_no_steps_and_a_reason(want, words):
    plan, *_ = run_plan(dict(HOME), set(), {"C1": ("B1", []), "C2": ("B2", [])}, want)
    assert plan == [] and not plan.complete and words in plan.reason


def test_broken_crane_wanted_is_refused():
    plan, *_ = run_plan(dict(HOME), {"QC1"}, {"C1": ("B1", [])}, {"C1": {"QC1": 5}})
    assert plan == [] and "out of service" in plan.reason


# --------------------------------------------------------------------------- real simulator

@pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
def test_plans_on_the_real_simulator_record_no_violations(tmp_path):
    """Random crane targets on real states; every plan step sent to the simulator through the guard."""
    from core.loop import Loop
    from core.state import build_state
    from vendor import TerminalAPI
    rng = random.Random(7)
    sent = complete = 0
    for scenario in ("practice_02", "practice_05"):
        api = TerminalAPI(scenario, data_xlsx=str(WORKBOOK), log_path=str(tmp_path / f"{scenario}.jsonl"))
        loop = Loop(api, tmp_path / scenario)
        for tick in range(300):
            loop.step()
            if tick % 15 == 0:
                state = build_state(api)
                active = [s for s in state.ships if s.status in ("berthed", "working")]
                broken = {c.crane_id for c in state.cranes if c.status == "out_of_service"}
                for _ in range(10 if active else 0):          # several random targets per check
                    state = build_state(api)
                    free = [c for c in CRANES if c not in broken]
                    rng.shuffle(free)
                    want = {}
                    for s in active:
                        lo, hi = BAYS[s.berth]
                        k = rng.randint(1, 2)
                        want[s.call_id] = {c: rng.randint(lo, hi) for c in free[:k]}
                        free = free[k:]
                    plan = safe_crane_moves(*plans_from_state(state, want))
                    complete += plan.complete
                    for a in plan:
                        ok, why = check(a, build_state(api))
                        assert ok, why
                        res = api.call(a.tool, dict(a.args))
                        assert "error" not in res, res
                        sent += 1
                    assert api.violations == [], api.violations
            api.call("advance_time", {"minutes": 15})
            loop.tick += 1
    print(f"\nsimulator: {sent} crane steps sent, {complete} complete plans, 0 violations")
    assert sent > 50 and complete > 5


# --------------------------------------------------------------------------- the safety check itself

@pytest.mark.parametrize("pos, assign, broken, words", [
    ({**HOME, "QC3": 13}, {}, set(), "rail order"),                                # idle cranes crossed
    ({**HOME, "QC1": 5, "QC2": 6}, {"C1": ["QC1", "QC2"]}, set(), "2 bays"),
    ({**HOME, "QC1": 5}, {"C1": ["QC1"]}, {"QC1"}, "out of service"),
    ({**HOME, "QC2": 20}, {"C1": ["QC2"], "C2": ["QC2"]}, set(), "on C1 and C2"),
    ({**HOME, "QC3": 30}, {"C2": ["QC3"]}, set(), "outside"),                      # bay 30 not in B2
])
def test_unsafe_names_each_rule(pos, assign, broken, words):
    target = {"C1": CallPlan(1, 30), "C2": CallPlan(31, 58)}
    why = unsafe(pos, assign, broken, target)
    assert why is not None and (words in why or words.replace("rail order", "left of") in why)


def test_unsafe_accepts_a_safe_layout():
    target = {"C1": CallPlan(1, 30), "C2": CallPlan(31, 58)}
    assert unsafe({**HOME, "QC1": 5, "QC2": 20, "QC3": 35}, {"C1": ["QC1", "QC2"], "C2": ["QC3"]}, set(), target) is None
