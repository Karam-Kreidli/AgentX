"""CORE-8 tests: the single-agent baseline is rule-safe, explained, and really has no negotiation."""
import ast
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import guard
from baselines.single_agent import SingleAgent
from core.skeleton import BERTHS
from core.types import (Block, Container, Crane, Ship, Truck, WorldState, SHIP_ARRIVED, SHIP_WORKING,
                        TOOL_ASSIGN_BERTH, TOOL_PLACE, TOOL_PLAN_CRANES, TOOL_DISPATCH_TRUCKS)

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
T0 = datetime(2026, 3, 1)
HOME = {"QC1": 1, "QC2": 14, "QC3": 27, "QC4": 40, "QC5": 53, "QC6": 66}


def ship(cid, loa=200.0, draft=10.0, status=SHIP_ARRIVED, berth=None, arrive_h=0, dis=10, load=0, cranes=()):
    return Ship(cid, "V", loa, draft, T0, T0 + timedelta(hours=arrive_h), status, berth, tuple(cranes), 0, dis, load, None)


def blocks():
    out = [Block(b, "standard", 0, 220) for b in "ABCDEF"]
    return out + [Block("G", "empty", 0, 200), Block("H", "IMDG", 0, 120),
                  Block("I", "reefer", 0, 200), Block("J", "reefer", 0, 200)]


def box(cid, call, cargo="dry", yard=None):
    return Container(cid, call, "import", 40, cargo, 3.0 if cargo == "IMDG" else None, 20.0, yard)


def world(ships, manifests=None, broken=(), trucks=12):
    cr = [Crane(c, "out_of_service" if c in broken else "available", None, bay) for c, bay in HOME.items()]
    return WorldState(T0 + timedelta(hours=5), T0 + timedelta(days=5), tuple(ships), tuple(cr),
                      tuple(Truck(f"T{i}", "available", None) for i in range(trucks)), tuple(blocks()), (),
                      manifests or {}, BERTHS)


def test_no_requests_no_grants():
    a = SingleAgent()
    assert not hasattr(a, "grants")
    acts, reqs = a.propose(world([ship("C1")]), [])
    assert reqs == []
    assert acts and all(x.reason and x.agent == "single" for x in acts)


def test_berth_best_fit_and_fcfs_order():
    # C1 (arrived first) fits B3 (220 m: LOA 200 + 20) -> smallest berth; C2 (270 m) needs B2 or B1.
    w = world([ship("C1", loa=200, arrive_h=1), ship("C2", loa=260, arrive_h=2), ship("C3", loa=200, arrive_h=3)])
    acts = [x for x in SingleAgent().propose(w, [])[0] if x.tool == TOOL_ASSIGN_BERTH]
    got = {x.args["call_id"]: x.args["berth_id"] for x in acts}
    assert got == {"C1": "B3", "C2": "B2", "C3": "B1"}


def test_unknown_size_never_berthed():
    w = world([ship("C1", loa=None)])
    assert [x for x in SingleAgent().propose(w, [])[0] if x.tool == TOOL_ASSIGN_BERTH] == []


def test_cranes_split_by_work_and_rail_safe():
    ships = [ship("C1", status=SHIP_WORKING, berth="B1", dis=0, load=300),
             ship("C2", status=SHIP_WORKING, berth="B3", dis=0, load=100)]
    w = world(ships)
    acts = [x for x in SingleAgent().propose(w, [])[0] if x.tool == TOOL_PLAN_CRANES]
    assert acts
    for x in acts:
        ok, why = guard.check(x, w)
        assert ok, why
    first = acts[0].args
    assert first["call_id"] in ("C1", "C2")


def test_broken_crane_never_assigned():
    ships = [ship("C1", status=SHIP_WORKING, berth="B1", dis=0, load=300)]
    w = world(ships, broken=("QC2",))
    for x in SingleAgent().propose(w, [])[0]:
        if x.tool == TOOL_PLAN_CRANES:
            assert "QC2" not in x.args["crane_ids"]


def test_unknown_cargo_never_placed():
    ships = [ship("C1", status=SHIP_WORKING, berth="B1", dis=3)]
    rows = (box("X1", "C1", "dry"), box("X2", "C1", "hazmat"), box("X3", "C1", ""), box("X4", "C1", "IMDG"),
            box("X5", "C1", "reefer"))
    acts = [x for x in SingleAgent().propose(world(ships, {"C1": rows}), [])[0] if x.tool == TOOL_PLACE]
    placed = {c: x.args["block_id"] for x in acts for c in x.args["container_ids"]}
    assert set(placed) == {"X1", "X4", "X5"}
    assert placed["X4"] == "H" and placed["X5"] in ("I", "J") and placed["X1"] in "ABCDEFG"


def test_yard_avoids_blocks_over_85_percent_when_possible():
    ships = [ship("C1", status=SHIP_WORKING, berth="B1", dis=2)]
    w = world(ships, {"C1": (box("X1", "C1"), box("X2", "C1"))})
    nearest = min("ABCDEF", key=lambda b: {"A": 2.3, "B": 2.6}.get(b, 9))      # A is nearest to B1
    blk = tuple(Block(b.block_id, b.block_type, 200 if b.block_id == nearest else b.used, b.slots) for b in w.blocks)
    w = WorldState(w.now, w.scenario_end, w.ships, w.cranes, w.trucks, blk, w.holds, w.manifests, w.berths)
    acts = [x for x in SingleAgent().propose(w, [])[0] if x.tool == TOOL_PLACE]
    assert acts and all(x.args["block_id"] != nearest for x in acts)


def test_trucks_three_per_crane():
    ships = [ship("C1", status=SHIP_WORKING, berth="B1", dis=0, load=100, cranes=("QC1", "QC2"))]
    acts = [x for x in SingleAgent().propose(world(ships), [])[0] if x.tool == TOOL_DISPATCH_TRUCKS]
    assert len(acts) == 1 and len(acts[0].args["truck_ids"]) == 6


ALLOWED = {"agents.crane_moves", "agents.yard", "core.skeleton", "core.types", "core.state", "math", "typing",
           "__future__"}


def test_imports_are_limited():
    """No coordinator, admission, stall, options, agent classes or LLM in the single agent."""
    tree = ast.parse((ROOT / "baselines" / "single_agent.py").read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mods.add(n.module)
        elif isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
    assert mods <= ALLOWED, sorted(mods - ALLOWED)
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "agents.yard" for a in n.names}
    assert names == {"TRAVEL_MIN"}                      # not YardAgent


@pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
def test_practice_01_runs_clean(tmp_path):
    from run import run_one
    s = run_one("practice_01", WORKBOOK, None, tmp_path, "single")
    assert s["violations"] == [] and s["skeleton_exceptions"] == 0
    assert s["loop"]["actions_refused"] == 0 and s["loop"]["breaches"] == 0
    assert s["vessels_departed"] >= 1
