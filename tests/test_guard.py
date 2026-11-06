"""SAFE-1 tests: every hard rule has at least one passing and one failing test,
plus a fuzz test of crane moves against the real simulator."""
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

import pytest

from core.skeleton import BERTHS
from core.types import (Action, Block, Container, Crane, Hold, Ship, Truck, WorldState)
from guard import check

T0 = datetime(2026, 9, 1, 6, 0)
HOME = {"QC1": 1, "QC2": 14, "QC3": 27, "QC4": 40, "QC5": 53, "QC6": 66}


def ship(call_id, status="berthed", berth="B1", cranes=(), loa=250.0, draft=12.0):
    arrived = None if status == "expected" else T0
    return Ship(call_id, "MV " + call_id, loa, draft, T0, arrived, status, berth if status in
                ("berthed", "working", "sailing", "departed") else None, tuple(cranes), 0, 100, 50, None)


def box(cid, cargo="dry", call="C1", block=None):
    return Container(cid, call, "import", 40, cargo, 3.0 if cargo == "IMDG" else None, 20.0, block)


def world(ships=(), positions=None, broken=(), trucks=None, blocks=None, boxes=(), holds=()):
    positions = {**HOME, **(positions or {})}
    on = {c: s.call_id for s in ships for c in s.cranes}
    cranes = tuple(Crane(c, "out_of_service" if c in broken else ("working" if c in on else "available"),
                         on.get(c), positions[c]) for c in sorted(positions))
    trucks = trucks or (Truck("YT01", "available", None), Truck("YT02", "out_of_service", None),
                        Truck("YT03", "working", "C2"))
    blocks = blocks or (Block("A", "standard", 10, 220), Block("G", "empty", 0, 200),
                        Block("H", "IMDG", 0, 120), Block("I", "reefer", 0, 200), Block("J", "reefer", 199, 200))
    manifests = MappingProxyType({"C1": tuple(boxes)})
    return WorldState(T0, datetime(2026, 9, 6), tuple(ships), cranes, tuple(trucks), tuple(blocks),
                      tuple(holds), manifests, BERTHS)


def act(tool, **args):
    return Action(tool, args, "test", "because")


def cranes(call, ids, bays):
    return act("plan_crane_sequence", call_id=call, crane_ids=list(ids), bay_positions=list(bays))


def ok(action, state):
    return check(action, state)[0]


# ---------------------------------------------------------------- rule 1: rail order (incl. idle)

def test_rail_order_pass():
    s = world([ship("C1")])
    assert ok(cranes("C1", ["QC1", "QC2"], [5, 20]), s)


def test_rail_order_fail_passing_an_idle_crane():
    # QC2 is idle at bay 14; moving QC1 to bay 20 would pass it.
    s = world([ship("C1")])
    good, reason = check(cranes("C1", ["QC1"], [20]), s)
    assert not good and "rail order" in reason


def test_rail_order_fail_passing_a_broken_crane():
    s = world([ship("C2", berth="B2")], broken=["QC4"])
    good, reason = check(cranes("C2", ["QC3"], [45]), s)     # QC4 broken at 40
    assert not good and "rail order" in reason


# ---------------------------------------------------------------- rule 2: spacing

def test_spacing_pass_exactly_two_bays():
    s = world([ship("C1")], positions={"QC1": 5, "QC2": 20})
    assert ok(cranes("C1", ["QC1", "QC2"], [5, 7]), s)


def test_spacing_fail_one_bay():
    s = world([ship("C1")])
    good, reason = check(cranes("C1", ["QC1", "QC2"], [5, 6]), s)
    assert not good and "2 bays" in reason


def test_spacing_fail_against_another_ships_crane():
    # C2 on B2 already works QC3 at bay 31; QC2 moved to bay 30 for C1 would be 1 bay away.
    s = world([ship("C1"), ship("C2", berth="B2", cranes=["QC3"])], positions={"QC3": 31})
    good, reason = check(cranes("C1", ["QC2"], [30]), s)
    assert not good and "2 bays" in reason


def test_spacing_ignores_idle_cranes():
    # QC3 idle at 27 is 1 bay from QC2 at 26: allowed (only working cranes need 2 bays).
    s = world([ship("C1")])
    assert ok(cranes("C1", ["QC2"], [26]), s)


# ---------------------------------------------------------------- rule 3: IMDG only in H, H only IMDG

def test_imdg_to_h_pass():
    s = world(boxes=[box("X1", "IMDG")])
    assert ok(act("place_container", container_ids=["X1"], block_id="H"), s)


def test_imdg_outside_h_fail():
    s = world(boxes=[box("X1", "IMDG")])
    good, reason = check(act("place_container", container_ids=["X1"], block_id="A"), s)
    assert not good and "IMDG" in reason


def test_non_imdg_into_h_fail():
    s = world(boxes=[box("D1", "dry")])
    good, reason = check(act("place_container", container_ids=["D1"], block_id="H"), s)
    assert not good and "only IMDG" in reason


# ---------------------------------------------------------------- rule 4: reefers only in I/J

def test_reefer_to_i_pass():
    s = world(boxes=[box("R1", "reefer")])
    assert ok(act("place_container", container_ids=["R1"], block_id="I"), s)


def test_reefer_outside_ij_fail():
    s = world(boxes=[box("R1", "reefer")])
    good, reason = check(act("place_container", container_ids=["R1"], block_id="A"), s)
    assert not good and "reefer" in reason


def test_dry_box_may_use_any_block_but_h():
    s = world(boxes=[box("D1"), box("D2")])
    for b in ("A", "G", "I"):
        assert ok(act("place_container", container_ids=["D1"], block_id=b), s)


def test_mixed_batch_refused_if_one_box_breaks_a_rule():
    s = world(boxes=[box("D1"), box("X1", "IMDG")])
    assert not ok(act("place_container", container_ids=["D1", "X1"], block_id="A"), s)


def test_block_capacity_and_unknown_boxes():
    s = world(boxes=[box("R1", "reefer"), box("R2", "reefer"), box("U1", "banana")])
    assert not ok(act("place_container", container_ids=["R1", "R2"], block_id="J"), s)   # 1 free slot
    assert ok(act("place_container", container_ids=["R1"], block_id="J"), s)
    assert not ok(act("place_container", container_ids=["U1"], block_id="A"), s)         # unknown cargo
    assert not ok(act("place_container", container_ids=["NOPE"], block_id="A"), s)       # unknown box


# ---------------------------------------------------------------- rule 5: customs holds

def test_release_pass_when_not_held():
    s = world(holds=[Hold("Z1", T0, True)])       # hold lifted
    assert ok(act("release_to_consignee", container_id="Z1"), s)


def test_release_fail_when_held():
    s = world(holds=[Hold("Z1", T0, False)])
    good, reason = check(act("release_to_consignee", container_id="Z1"), s)
    assert not good and "customs hold" in reason


# ---------------------------------------------------------------- rule 6: berth fit / arrival / occupancy

def test_berth_pass():
    s = world([ship("C1", status="arrived", loa=290.0, draft=14.0)])
    assert ok(act("assign_berth", call_id="C1", berth_id="B1"), s)


def test_berth_fail_not_arrived():
    s = world([ship("C1", status="expected")])
    assert "arrived" in check(act("assign_berth", call_id="C1", berth_id="B1"), s)[1]


def test_berth_fail_too_long_and_too_deep():
    s = world([ship("C1", status="arrived", loa=261.0, draft=12.0)])
    assert "needs" in check(act("assign_berth", call_id="C1", berth_id="B2"), s)[1]      # 281 m > 280 m
    s = world([ship("C1", status="arrived", loa=200.0, draft=12.1)])
    assert "too deep" in check(act("assign_berth", call_id="C1", berth_id="B3"), s)[1]   # 12.1 > 12.0
    s = world([ship("C1", status="arrived", loa=200.0, draft=12.0)])
    assert ok(act("assign_berth", call_id="C1", berth_id="B3"), s)                       # exactly fits


def test_berth_fail_occupied_including_sailing():
    for status in ("berthed", "working", "sailing"):
        s = world([ship("C1", status="arrived"), ship("C2", status=status, berth="B1")])
        assert "occupied" in check(act("assign_berth", call_id="C1", berth_id="B1"), s)[1]
    s = world([ship("C1", status="arrived"), ship("C2", status="departed", berth="B1")])
    assert ok(act("assign_berth", call_id="C1", berth_id="B1"), s)


# ---------------------------------------------------------------- rule 7: broken / busy equipment

def test_crane_broken_or_busy_fail():
    s = world([ship("C1")], broken=["QC2"])
    assert "out of service" in check(cranes("C1", ["QC2"], [20]), s)[1]
    s = world([ship("C1"), ship("C2", berth="B2", cranes=["QC3"])])
    assert "working on C2" in check(cranes("C1", ["QC3"], [28]), s)[1]


def test_trucks_pass_and_fail():
    s = world([ship("C1"), ship("C2", berth="B2")])
    assert ok(act("dispatch_truck", call_id="C1", truck_ids=["YT01"]), s)
    assert ok(act("dispatch_truck", call_id="C1", truck_ids=[]), s)                       # release all
    assert ok(act("dispatch_truck", call_id="C2", truck_ids=["YT03"]), s)                 # already its own
    assert "out of service" in check(act("dispatch_truck", call_id="C1", truck_ids=["YT02"]), s)[1]
    assert "already serves" in check(act("dispatch_truck", call_id="C1", truck_ids=["YT03"]), s)[1]


# ---------------------------------------------------------------- rule 8: cranes inside berth bays

def test_crane_inside_berth_pass_and_fail():
    s = world([ship("C2", berth="B2")], positions={"QC4": 50})
    assert ok(cranes("C2", ["QC3"], [31]), s)
    good, reason = check(cranes("C2", ["QC3"], [30]), s)          # B2 is bays 31-58
    assert not good and "outside B2" in reason


# ---------------------------------------------------------------- robustness

@pytest.mark.parametrize("action", [
    Action("advance_time", {"minutes": 15}, "x", "r"),           # not an agent tool
    Action("drop_table", {}, "x", "r"),
    Action("plan_crane_sequence", {"call_id": "C1", "crane_ids": "QC1", "bay_positions": [5]}, "x", "r"),
    Action("plan_crane_sequence", {"call_id": "C1", "crane_ids": ["QC1"], "bay_positions": [5.5]}, "x", "r"),
    Action("plan_crane_sequence", {"call_id": "C1", "crane_ids": ["QC1", "QC1"], "bay_positions": [5, 8]}, "x", "r"),
    Action("plan_crane_sequence", {"call_id": "NOPE", "crane_ids": ["QC1"], "bay_positions": [5]}, "x", "r"),
    Action("place_container", {"container_ids": None, "block_id": "A"}, "x", "r"),
    Action("assign_berth", {"call_id": "C1"}, "x", "r"),
    Action("release_to_consignee", {}, "x", "r"),
    Action("dispatch_truck", {"call_id": "C1", "truck_ids": ["YT99"]}, "x", "r"),
    "not an action",
    None,
])
def test_malformed_or_unknown_is_refused_never_raises(action):
    good, reason = check(action, world([ship("C1")]))
    assert good is False and reason


def test_guard_does_not_modify_the_action_or_state():
    s = world([ship("C1")])
    a = cranes("C1", ["QC1"], [20])
    before = (a, s)
    check(a, s)
    assert (a, s) == before


# ---------------------------------------------------------------- fuzz against the real simulator

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"


def _put_ships_to_work(api, state):
    """Test-only driver: berth arrived ships and give each berth its fixed cranes,
    straight through the simulator, so the fuzz test sees realistic busy states."""
    from core.skeleton import CRANE_SPLIT
    for s in state.ships:
        if s.status == "arrived":
            for b in BERTHS:
                if b.fits(s.loa_m, s.draft_m) and not any(o.berth == b.berth_id and o.status in
                                                         ("berthed", "working", "sailing") for o in state.ships):
                    api.call("assign_berth", {"call_id": s.call_id, "berth_id": b.berth_id})
                    break
    for s in api.call("get_vessel_schedule"):
        if s["status"] in ("berthed", "working") and not s["cranes"]:
            broken = {e["equipment_id"] for e in api.call("get_equipment_status") if e["status"] == "out of service"}
            split = [(c, bay) for c, bay in CRANE_SPLIT[s["berth"]] if c not in broken]
            if split:
                api.call("plan_crane_sequence", {"call_id": s["call_id"], "crane_ids": [c for c, _ in split],
                                                 "bay_positions": [b for _, b in split]})


@pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
def test_crane_moves_agree_with_simulator(tmp_path):
    """Random crane moves on real states: the guard says ok exactly when the simulator
    accepts the move without recording a violation."""
    from core.state import build_state
    from vendor import TerminalAPI

    rng = random.Random(1)
    checked = refused = 0
    for scenario in ("practice_02", "practice_05"):          # several ships berthed at once
        api = TerminalAPI(scenario, data_xlsx=str(WORKBOOK), log_path=str(tmp_path / f"{scenario}.jsonl"))
        for tick in range(200):
            if tick % 10 == 0:
                _put_ships_to_work(api, build_state(api))
                state = build_state(api)
                active = [s for s in state.ships if s.status in ("berthed", "working")]
                for _ in range(60 if active else 0):
                    target = rng.choice(active)
                    n = rng.randint(1, 3)
                    ids = sorted(rng.sample(sorted(HOME), n))
                    bays = [rng.randint(1, 80) for _ in ids]
                    action = cranes(target.call_id, ids, bays)
                    verdict, why = check(action, state)

                    saved = (dict(api.crane_pos), {k: list(v["cranes"]) for k, v in api.calls.items()},
                             len(api.violations))
                    res = api.call("plan_crane_sequence", action.args)
                    sim_ok = "error" not in res and len(api.violations) == saved[2]
                    api.crane_pos.clear(); api.crane_pos.update(saved[0])
                    for k, v in saved[1].items():
                        api.calls[k]["cranes"] = v
                    del api.violations[saved[2]:]

                    assert verdict == sim_ok, (scenario, tick, action.args, why, res)
                    checked += 1
                    refused += not verdict
            api.call("advance_time", {"minutes": 15})
        assert api.violations == []
    assert checked >= 1000 and 0 < refused < checked
