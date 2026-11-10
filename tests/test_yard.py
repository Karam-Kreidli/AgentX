"""YARD-1 tests: legal blocks only, nearest first, 85% soft limit, H full, I/J full,
unknown cargo (not placed, escalated), and zero yard violations with the reefer
rule respected across the 5 practice scenarios."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType

from agents.yard import TRAVEL_MIN, YardAgent
from core.skeleton import BERTHS
from core.types import Block, Container, Ship, WorldState
from guard import check

T0 = datetime(2026, 9, 1, 6, 0)
TYPES = {"A": "standard", "B": "standard", "C": "standard", "D": "standard", "E": "standard",
         "F": "standard", "G": "empty", "H": "IMDG", "I": "reefer", "J": "reefer"}
SLOTS = {**{b: 220 for b in "ABCDEF"}, "G": 200, "H": 120, "I": 200, "J": 200}


def ship(call_id="C1", berth="B1", status="working", discharge=100):
    return Ship(call_id, "MV", 200.0, 11.0, T0, T0, status, berth, ("QC1",), 3, discharge, 10, None)


def boxes(n, cargo="dry", call="C1", start=0, placed=None):
    return [Container(f"{cargo[:1].upper()}{call}{i}", call, "import", 40, cargo,
                      3.0 if cargo == "IMDG" else None, 20.0, placed) for i in range(start, start + n)]


def world(ships, containers, used=None):
    used = used or {}
    blocks = tuple(Block(b, TYPES[b], used.get(b, 0), SLOTS[b]) for b in "ABCDEFGHIJ")
    by_call = {}
    for c in containers:
        by_call.setdefault(c.call_id, []).append(c)
    return WorldState(T0, T0 + timedelta(days=5), tuple(ships), (), (), blocks, (),
                      MappingProxyType({k: tuple(v) for k, v in by_call.items()}), BERTHS)


def plan(state):
    agent = YardAgent()
    actions, _ = agent.propose(state, [])
    placed = {}
    for a in actions:
        for c in a.args["container_ids"]:
            placed[c] = a.args["block_id"]
    return placed, actions, agent


# ---------------------------------------------------------------- legal blocks

def test_imdg_only_to_h_reefer_only_to_ij():
    s = world([ship()], boxes(5, "IMDG") + boxes(5, "reefer"))
    placed, actions, _ = plan(s)
    assert {placed[c] for c in placed if c.startswith("I")} == {"H"}
    assert {placed[c] for c in placed if c.startswith("R")} <= {"I", "J"}
    for a in actions:
        assert check(a, s) == (True, "ok") and a.reason


def test_dry_goes_to_nearest_standard_block_from_the_ships_berth():
    for berth in ("B1", "B2", "B3"):
        placed, _, _ = plan(world([ship(berth=berth)], boxes(10)))
        nearest = min("ABCDEF", key=lambda b: (TRAVEL_MIN[berth][b], b))
        assert set(placed.values()) == {nearest}, berth


def test_standard_blocks_kept_at_or_under_85_percent_while_others_have_room():
    s = world([ship()], boxes(100), used={"A": 180})          # 85% of 220 = 187
    placed, actions, _ = plan(s)
    assert sum(1 for b in placed.values() if b == "A") == 7     # 180 -> 187
    assert all(b in "ABCDEF" for b in placed.values())


def test_overflow_order_standard_then_g_then_ij():
    full = {b: 220 for b in "ABCDEF"}
    placed, _, _ = plan(world([ship()], boxes(50), used=full))
    assert set(placed.values()) == {"G"}
    placed, _, _ = plan(world([ship()], boxes(50), used={**full, "G": 200}))
    assert set(placed.values()) <= {"I", "J"}


def test_never_more_boxes_than_free_slots():
    s = world([ship()], boxes(30, "IMDG"), used={"H": 110})
    placed, actions, _ = plan(s)
    assert len(placed) == 10 and all(check(a, s)[0] for a in actions)


# ---------------------------------------------------------------- nowhere legal: queue, never force

def test_h_full_imdg_waits_and_is_escalated():
    s = world([ship()], boxes(4, "IMDG") + boxes(3), used={"H": 120})
    placed, _, agent = plan(s)
    assert not any(c.startswith("I") for c in placed)            # no IMDG box forced elsewhere
    assert len([c for c in placed if c.startswith("D")]) == 3    # the dry boxes still go
    assert len(agent.escalations) == 1 and "4 import boxes" in agent.escalations[0]


def test_ij_full_reefer_waits_and_is_never_put_in_a_standard_block():
    s = world([ship()], boxes(6, "reefer"), used={"I": 200, "J": 200})
    placed, actions, agent = plan(s)
    assert placed == {} and actions == []
    assert len(agent.escalations) == 1 and "no legal yard slot" in agent.escalations[0]


def test_unknown_cargo_is_not_placed_and_is_escalated_once():
    s = world([ship(), ship("C2", "B2")],
              boxes(30, "banana") + boxes(3, "kiwi") + boxes(2) + boxes(5, "banana", call="C2"))
    agent = YardAgent()
    actions, _ = agent.propose(s, [])
    placed = {c for a in actions for c in a.args["container_ids"]}
    assert not any(c.startswith(("B", "K")) for c in placed) and len(placed) == 2
    unknown = [e for e in agent.escalations if "unknown cargo" in e]
    assert len(unknown) == 3                       # one per ship and cargo type, not one per box
    assert any("30 import boxes of C1" in e and "'banana'" in e for e in unknown)
    assert any("3 import boxes of C1" in e and "'kiwi'" in e for e in unknown)
    assert any("5 import boxes of C2" in e for e in unknown)
    agent.propose(s, [])
    assert sum("unknown cargo" in e for e in agent.escalations) == 3   # not repeated next tick


def test_only_berthed_or_working_ships_with_work_left_and_unplaced_imports():
    exports = [Container("EX1", "C1", "export", 40, "dry", None, 20.0, None)]
    s = world([ship("C1"), ship("C2", status="arrived", berth=None), ship("C3", discharge=0)],
              boxes(2, call="C1", placed="A") + exports + boxes(2, call="C2") + boxes(2, call="C3"))
    placed, _, _ = plan(s)
    assert placed == {}


# ---------------------------------------------------------------- the 5 scenarios

def test_zero_yard_violations_and_reefer_rule_across_the_5_scenarios(multi_runs):
    root = Path(__file__).resolve().parents[1]
    import pandas as pd                      # test-only: check the outcome against the real cargo types
    cargo = pd.read_excel(root / "data" / "P2_Container_Terminal_Data.xlsx", "Practice_Manifest") \
              .set_index("container_id")["cargo_type"]
    for scenario, (summary, run_dir) in multi_runs.items():
        assert summary["violations"] == []
        decisions = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines()]
        places = [d for d in decisions if d["choice"] == "place_container"]
        assert places and all(d["agent"] == "yard" and d["guard_ok"] for d in places), scenario
        for d in places:
            block = d["saw"]["args"]["block_id"]
            for c in d["saw"]["args"]["container_ids"]:
                assert (cargo[c] == "IMDG") == (block == "H"), (scenario, c, block)
                if cargo[c] == "reefer":
                    assert block in ("I", "J"), (scenario, c, block)
