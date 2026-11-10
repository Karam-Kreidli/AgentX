"""CRANE-2 tests: crane agent v1 (agents/crane.py).

- demand: loading ships use every crane, discharging ships only what the yard lets them reach,
  stalled and impossible ships none, a ship showing 0/0 that has not sailed keeps one.
- allocate: every ship with work gets a crane, idle cranes are never put on ships that can't
  use them, extra cranes go to the ship with the least work left, broken cranes are never used.
- Property test: 3,000 random crane layouts; every step of the plan is checked with the guard.
- Grants are binding; the coordinator counts every in-service crane; it never raises.
- Real scenarios: more crane moves per crane-hour than the skeleton's fixed split, and no
  stalled ship keeps a crane for more than one tick.
"""
import ast
import random
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import pytest

import coordinator
from agents import crane as C
from agents import stall
from agents.crane_moves import unsafe, CallPlan
from core.skeleton import BERTHS
from core.types import (GRANTED, RES_CRANES, Block, Container, Crane, Request, Ship, Truck,
                        WorldState)
from guard import check

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
T0 = datetime(2026, 9, 1, 6, 0)
CRANES = ["QC1", "QC2", "QC3", "QC4", "QC5", "QC6"]
HOME = {"QC1": 1, "QC2": 14, "QC3": 27, "QC4": 40, "QC5": 53, "QC6": 66}
BAYS = {b.berth_id: (b.start_bay, b.end_bay) for b in BERTHS}


@pytest.fixture(autouse=True)
def fresh():
    stall.reset()
    coordinator.reset()
    yield
    stall.reset()


def ship(cid, berth="B1", status="working", discharge=0, load=500, cranes=()):
    return Ship(cid, "MV " + cid, 200.0, 11.0, T0, T0, status, berth, tuple(cranes), 3 * len(cranes),
                discharge, load, None)


def boxes(cid, n, placed=None, start=0, cargo="dry"):
    return tuple(Container(f"{cid}-{i}", cid, "import", 40, cargo, None, 20.0, placed) for i in range(start, start + n))


@pytest.fixture(autouse=True)
def _pre_bc4_horizon(monkeypatch):
    """These tests pin the demand arithmetic at the pre-BC-4 horizon (0.25 h); the tuned value lives in
    config/params.yaml and is checked by tests/test_params.py and the scorecard."""
    monkeypatch.setattr(C, "TICK_H", 0.25)


def world(ships, pos=None, broken=(), manifests=None, free=50):
    pos = pos or HOME
    on = {c: s.call_id for s in ships for c in s.cranes}
    cranes = tuple(Crane(c, "out_of_service" if c in broken else ("working" if c in on else "available"),
                         on.get(c), pos[c]) for c in CRANES)
    blocks = (Block("A", "standard", 100 - free, 100), Block("H", "IMDG", 100, 100), Block("I", "reefer", 100, 100))
    trucks = tuple(Truck(f"T{i:02d}", "available", None) for i in range(20))
    return WorldState(T0, T0 + timedelta(days=5), tuple(ships), cranes, trucks, blocks, (),
                      MappingProxyType(manifests or {}), BERTHS)


def first_seen(*states):
    for st in states:
        stall.stalled_calls(st)


# ---------------------------------------------------------------- demand

def test_a_loading_ship_can_use_every_crane_and_a_nearly_done_one_fewer():
    st = world([ship("L", load=500)])
    assert C.demand(st, st.ships[0]) == 6
    st = world([ship("L", load=10)])
    assert C.demand(st, st.ships[0]) == 2                     # 10 moves / (23 x 0.25 h) -> 2 cranes


def test_a_ship_showing_no_moves_left_that_has_not_sailed_keeps_one_crane():
    st = world([ship("Z", discharge=0, load=0)])
    assert C.demand(st, st.ships[0]) == 1


def test_discharging_ship_is_limited_by_boxes_at_the_quay_and_yard_room():
    first_seen(world([ship("D", discharge=100, load=50)], manifests={"D": boxes("D", 100)}))
    # 20 placed, 0 discharged yet, 80 waiting and no room: 20 / 5.75 -> 4 cranes
    st = world([ship("D", discharge=100, load=50)],
               manifests={"D": boxes("D", 20, placed="A") + boxes("D", 80, start=20)}, free=0)
    assert C.demand(st, st.ships[0]) == 4
    # with room the yard agent can place the rest this tick: all six
    st = world([ship("D", discharge=100, load=50)],
               manifests={"D": boxes("D", 20, placed="A") + boxes("D", 80, start=20)}, free=50)
    assert C.demand(st, st.ships[0]) == 6


def test_a_ship_finishing_discharge_counts_its_load_too():
    first_seen(world([ship("F", discharge=100, load=500)], manifests={"F": boxes("F", 100)}))
    # 10 boxes left to discharge, all at the quay; the 500 load moves follow straight after
    st = world([ship("F", discharge=10, load=500)], manifests={"F": boxes("F", 100, placed="A")}, free=0)
    assert C.demand(st, st.ships[0]) == 6


def test_stalled_and_impossible_ships_get_no_demand():
    first_seen(world([ship("S", discharge=100)], manifests={"S": boxes("S", 100)}))
    st = world([ship("S", discharge=60)], manifests={"S": boxes("S", 40, placed="A") + boxes("S", 60, start=40)},
               free=0)
    assert "S" in stall.stalled_calls(st) and C.demand(st, st.ships[0]) == 0
    stall.reset()
    st = world([ship("X", discharge=1600)], manifests={"X": boxes("X", 551)})
    assert "X" in stall.impossible_calls(st) and C.demand(st, st.ships[0]) == 0


def test_ships_not_at_berth_have_no_demand():
    st = world([ship("E", berth=None, status="expected"), ship("G", berth=None, status="sailing")])
    assert [C.demand(st, s) for s in st.ships] == [0, 0]


# ---------------------------------------------------------------- allocate

def assert_safe_layout(st, want):
    pos = {c.crane_id: c.bay for c in st.cranes}
    pos.update({c: b for w in want.values() for c, b in w.items()})
    broken = {c.crane_id for c in st.cranes if c.status == "out_of_service"}
    target = {s.call_id: CallPlan(*BAYS[s.berth], s.cranes, want.get(s.call_id, {})) for s in st.ships if s.berth}
    assert unsafe(pos, {k: list(v) for k, v in want.items()}, broken, target) is None


def test_one_ship_gets_all_six_cranes_spread_over_its_berth():
    st = world([ship("A", "B1")])
    want, need = C.allocate(st)
    assert sorted(want["A"]) == CRANES and need == {"A": 6}
    assert all(1 <= b <= 30 for b in want["A"].values())
    assert_safe_layout(st, want)


def test_two_ships_both_get_cranes_in_rail_order_and_the_shorter_job_gets_more():
    st = world([ship("BIG", "B1", load=2000), ship("SMALL", "B3", load=300)])
    want, _ = C.allocate(st)
    assert want["BIG"] and want["SMALL"] and len(want["BIG"]) + len(want["SMALL"]) == 6
    assert len(want["SMALL"]) > len(want["BIG"])                  # shortest job first
    assert max(int(c[2:]) for c in want["BIG"]) < min(int(c[2:]) for c in want["SMALL"])
    assert_safe_layout(st, want)


def test_a_stalled_ship_loses_its_cranes_to_a_ship_that_can_work():
    first_seen(world([ship("S", "B1", discharge=100), ship("W", "B2", load=900)], manifests={"S": boxes("S", 100)}))
    st = world([ship("S", "B1", discharge=60, cranes=("QC1", "QC2")), ship("W", "B2", load=800)],
               pos={**HOME, "QC1": 5, "QC2": 20},
               manifests={"S": boxes("S", 40, placed="A") + boxes("S", 60, start=40)}, free=0)
    want, need = C.allocate(st)
    assert need["S"] == 0 and want["S"] == {}
    assert len(want["W"]) >= 4
    acts, _ = C.CraneAgent().propose(st, [])
    assert any(a.args["call_id"] == "S" and a.args["crane_ids"] == [] for a in acts)


def test_spare_cranes_stay_idle_rather_than_on_a_ship_that_cannot_use_them():
    st = world([ship("L", load=10)])                              # can use 2
    want, _ = C.allocate(st)
    assert len(want["L"]) == 2


def test_broken_crane_is_never_used_and_the_plan_stays_safe():
    st = world([ship("A", "B1"), ship("B", "B3")], broken=("QC3",))
    want, _ = C.allocate(st)
    assert all("QC3" not in w for w in want.values())
    assert want["A"] and want["B"]
    assert_safe_layout(st, want)


# ---------------------------------------------------------------- property test

def apply(pos, on, a):
    call = a.args["call_id"]
    pos = {**pos, **dict(zip(a.args["crane_ids"], a.args["bay_positions"]))}
    on = {c: k for c, k in on.items() if k != call}
    on.update({c: call for c in a.args["crane_ids"]})
    return pos, on


def state_of(base, pos, on):
    cranes = tuple(Crane(c.crane_id, c.status if c.status == "out_of_service" else ("working" if c.crane_id in on else "available"),
                         on.get(c.crane_id), pos[c.crane_id]) for c in base.cranes)
    ships = tuple(Ship(s.call_id, s.vessel, s.loa_m, s.draft_m, s.scheduled_eta, s.actual_arrival, s.status, s.berth,
                       tuple(sorted((c for c, k in on.items() if k == s.call_id), key=lambda c: int(c[2:]))),
                       s.trucks, s.discharge_remaining, s.load_remaining, s.departed_at) for s in base.ships)
    return WorldState(base.now, base.scenario_end, ships, cranes, base.trucks, base.blocks, base.holds,
                      base.manifests, base.berths)


def random_world(rng):
    pos = dict(zip(CRANES, sorted(rng.sample(range(1, 81), 6))))
    broken = set(rng.sample(CRANES, rng.choice([0, 0, 0, 1, 2])))
    ships, used = [], set()
    for i, b in enumerate(sorted(rng.sample(["B1", "B2", "B3"], rng.randint(1, 3)))):
        lo, hi = BAYS[b]
        cur = []
        for c in CRANES:
            if lo <= pos[c] <= hi and c not in broken and c not in used and rng.random() < 0.6 and \
                    all(abs(pos[c] - pos[x]) >= 2 for x in used | set(cur)):
                cur.append(c)
        used |= set(cur)
        ships.append(ship(f"C{i}", b, discharge=0, load=rng.choice([0, 5, 30, 300, 2000]), cranes=cur))
    return world(ships, pos=pos, broken=broken)


def test_3000_random_layouts_every_step_passes_the_guard():
    rng = random.Random(8)
    moved = 0
    for n in range(3000):
        st = random_world(rng)
        acts, reqs = C.CraneAgent().propose(st, [])
        pos = {c.crane_id: c.bay for c in st.cranes}
        on = {c.crane_id: c.call_id for c in st.cranes if c.call_id}
        for i, a in enumerate(acts):
            ok, why = check(a, state_of(st, pos, on))
            assert ok, f"case {n} step {i}: {why} {a.args}"
            assert a.reason and a.agent == "crane"
            pos, on = apply(pos, on, a)
        broken = {c.crane_id for c in st.cranes if c.status == "out_of_service"}
        assert not (set(on) & broken), n
        moved += bool(acts)
        assert all(r.resource == RES_CRANES and r.amount > 0 for r in reqs)
    assert moved > 1000


# ---------------------------------------------------------------- grants, coordinator, robustness

def test_grants_are_binding():
    st = world([ship("A", "B1"), ship("B", "B3")])
    agent = C.CraneAgent()
    _, reqs = agent.propose(st, [])
    assert {r.call_id: r.amount for r in reqs} and sum(r.amount for r in reqs) == 6
    agent.grants = {(RES_CRANES, "A"): 1, (RES_CRANES, "B"): 2}
    acts, _ = agent.propose(st, [])
    last = {}
    for a in acts:
        last[a.args["call_id"]] = a.args["crane_ids"]
    assert len(last.get("A", [])) <= 1 and len(last.get("B", [])) <= 2
    agent.grants = {}                                             # nothing granted: no cranes added
    acts, _ = agent.propose(st, [])
    assert all(a.args["crane_ids"] == [] for a in acts)


def test_coordinator_counts_every_in_service_crane_even_ones_on_ships():
    st = world([ship("A", "B1", cranes=("QC1", "QC2", "QC3", "QC4")), ship("B", "B3")], broken=("QC6",),
               pos={**HOME, "QC1": 3, "QC2": 8, "QC3": 13, "QC4": 18})
    reqs = [Request("crane", RES_CRANES, 3, "A", 1.0, "r"), Request("crane", RES_CRANES, 2, "B", 1.0, "r")]
    grants, _ = coordinator.decide(reqs, st)
    assert [(g.request.call_id, g.granted, g.status) for g in grants] == [("A", 3, GRANTED), ("B", 2, GRANTED)]


def test_never_raises_on_a_broken_state():
    agent = C.CraneAgent()
    assert agent.propose(None, []) == ([], [])
    assert agent.escalations


def test_no_ids_in_the_agent():
    src = (ROOT / "agents" / "crane.py").read_text(encoding="utf-8")
    strings = [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not [s for s in strings if s[:1] == "P" and s[1:3].isdigit()]


def test_run_py_uses_the_crane_agent():
    import run
    names = [type(a).__name__ for a in run.make_agents("multi")]
    assert "CraneAgent" in names and "SkeletonAgent" not in names


# ---------------------------------------------------------------- real scenarios

class Observer:
    """Sees the state at the start of every round; records stalled ships that still hold cranes."""
    name = "observer"

    def __init__(self):
        self.first = {}

    def propose(self, state, events):
        self.first.setdefault(state.now, {s.call_id for s in state.ships if s.cranes} & stall.stalled_calls(state))
        return [], []


def run(scenario, tmp_path, cranes):
    from core.loop import Loop
    from core.skeleton import SkeletonAgent
    from agents.berth import BerthAgent
    from agents.trucks import TruckAgent
    from agents.yard import YardAgent
    from vendor import TerminalAPI
    coordinator.reset()
    stall.reset()
    api = TerminalAPI(scenario, data_xlsx=str(WORKBOOK), log_path=str(tmp_path / "actions.jsonl"))
    obs = Observer()
    agents = [BerthAgent(), cranes, TruckAgent(), YardAgent(), obs]
    Loop(api, tmp_path, agents=agents).run()
    return api.run_summary(), obs


@needs_workbook
def test_practice_02_more_moves_per_crane_hour_than_the_skeleton_and_no_stalled_ship_keeps_cranes(tmp_path):
    from core.skeleton import SkeletonAgent
    ours, obs = run("practice_02", tmp_path / "crane", C.CraneAgent())
    base, _ = run("practice_02", tmp_path / "skeleton", SkeletonAgent(parts=("cranes",)))
    assert not ours["violations"]
    assert ours["crane_moves_per_hour"] > base["crane_moves_per_hour"] + 5
    assert ours["vessels_departed"] >= base["vessels_departed"]
    ticks = sorted(obs.first)
    held = [(a, obs.first[a] & obs.first[b]) for a, b in zip(ticks, ticks[1:]) if obs.first[a] & obs.first[b]]
    assert held == []                                             # never two ticks in a row


@needs_workbook
def test_practice_05_crane_moves_per_hour_not_below_the_skeleton(tmp_path):
    from core.skeleton import SkeletonAgent
    ours, _ = run("practice_05", tmp_path / "crane", C.CraneAgent())
    base, _ = run("practice_05", tmp_path / "skeleton", SkeletonAgent(parts=("cranes",)))
    assert not ours["violations"]
    assert ours["crane_moves_per_hour"] >= base["crane_moves_per_hour"]
    assert ours["avg_turnaround_h"] < base["avg_turnaround_h"]
