"""Review follow-up: crane grants decided together with the yard slots and trucks they need
(coordinator._joint). Two ships share scarce yard room: the crane agent counts that room for
both, the coordinator gives it to one, and the joint check trims the other's cranes, with the
reason in the negotiation transcript, so the crane agent moves them to a ship that can work."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

import coordinator as C
from agents import crane as crane_agent
from agents import stall
from core.types import GRANTED, REFUSED, TRIMMED, Crane, Request, Truck, RES_CRANES
from tests.test_coordinator import T0, boxes, no_forecast, ship, world, yard_req  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"


@pytest.fixture(autouse=True)
def fresh_stall():
    stall.reset()
    yield
    stall.reset()


def crane_req(cid, n, priority=1.0):
    return Request("crane", RES_CRANES, n, cid, priority, f"{cid} can keep {n} cranes busy")


def trucks(n, broken=0):
    return [Truck(f"T{i:02d}", "out_of_service" if i < broken else "available", None) for i in range(n)]


CRANES = [Crane(f"QC{i}", "available", None, b) for i, b in zip(range(1, 7), (1, 14, 27, 40, 53, 66))]


def two_ships(free):
    """A on B1 and B on B3, both discharging 100 dry boxes, nothing placed yet."""
    ships = [replace(ship("A", "B1", discharge=100), cranes=()), replace(ship("B", "B3", discharge=100, order_eta=1), cranes=())]
    st = world(ships, boxes("A", 100) + boxes("B", 100), free={"A": free}, trucks=trucks(20), cranes=CRANES)
    stall.stalled_calls(st)                       # first sighting: nothing discharged yet
    return st


def test_yard_room_given_to_one_ship_trims_the_other_ships_cranes(monkeypatch):
    no_forecast(monkeypatch)
    st = two_ships(free=30)                       # room for 30 boxes: A (first) gets all 30
    g = {(x.request.resource, x.request.call_id): x for x in
         C.decide([yard_req("A", 100), yard_req("B", 100), crane_req("A", 3), crane_req("B", 3)], st)[0]}
    assert g[("yard_slots", "A")].granted == 30 and g[("yard_slots", "B")].granted == 0
    assert g[("cranes", "A")].status == GRANTED and g[("cranes", "A")].granted == 3
    b = g[("cranes", "B")]
    assert b.status == REFUSED and b.granted == 0
    assert b.reason.startswith("Joint check:") and "0 of the 100 boxes" in b.reason


def test_a_small_yard_grant_keeps_fewer_cranes_busy(monkeypatch):
    no_forecast(monkeypatch)
    monkeypatch.setattr(crane_agent, "TICK_H", 0.25)   # the 15-minute horizon; config/params.yaml may differ
    st = two_ships(free=12)                       # 12 boxes = ~2 cranes for a 15-minute tick
    g = {(x.request.resource, x.request.call_id): x for x in
         C.decide([yard_req("A", 100), crane_req("A", 6)], st)[0]}
    a = g[("cranes", "A")]
    assert a.status == TRIMMED and a.granted == 3 and "12 of the 100" in a.reason


def test_trucks_in_service_limit_cranes_in_priority_order(monkeypatch):
    no_forecast(monkeypatch)
    ships = [ship("A", "B1", discharge=0), ship("B", "B3", discharge=0, order_eta=1)]
    st = world(ships, [], trucks=trucks(20, broken=13), cranes=CRANES)      # 7 trucks in service
    g = {x.request.call_id: x for x in C.decide([crane_req("A", 3), crane_req("B", 3)], st)[0]}
    assert (g["A"].granted, g["A"].status) == (2, TRIMMED) and "only 7 trucks" in g["A"].reason
    assert (g["B"].granted, g["B"].status) == (1, TRIMMED)                  # the last truck runs one crane


def test_loading_ships_and_ships_without_a_yard_request_are_not_trimmed(monkeypatch):
    no_forecast(monkeypatch)
    ships = [ship("L", "B1", discharge=0), ship("P", "B3", discharge=40)]
    st = world(ships, boxes("P", 40, placed="A"), free={"A": 0}, trucks=trucks(20), cranes=CRANES)
    g = {x.request.call_id: x for x in C.decide([crane_req("L", 4), crane_req("P", 2)], st)[0]}
    assert g["L"].status == GRANTED and g["P"].status == GRANTED


def test_the_check_never_breaks_the_grants(monkeypatch):
    no_forecast(monkeypatch)
    st = two_ships(free=30)
    monkeypatch.setattr(crane_agent, "with_yard", lambda *a: 1 / 0)
    g = {(x.request.resource, x.request.call_id): x for x in
         C.decide([yard_req("B", 100), crane_req("B", 3)], st)[0]}
    assert g[("cranes", "B")].status == GRANTED                             # the pool's grant stands


def test_the_crane_agent_moves_trimmed_cranes_to_the_ship_that_can_work(monkeypatch):
    no_forecast(monkeypatch)
    st = two_ships(free=30)
    agent = crane_agent.CraneAgent()
    _, reqs = agent.propose(st, [])
    asked = {r.call_id: r.amount for r in reqs}
    assert asked.get("A") and asked.get("B")                                # both count the same 30 slots
    grants = C.decide([yard_req("A", 100), yard_req("B", 100)] + reqs, st)[0]
    agent.grants = {(g.request.resource, g.request.call_id): g.granted for g in grants
                    if g.request.agent == "crane"}
    acts, _ = agent.propose(st, [])
    final = {}
    for a in acts:
        final[a.args["call_id"]] = a.args["crane_ids"]
    assert not final.get("B")
    assert len(final.get("A", [])) >= asked["A"]


@pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
def test_practice_02_shows_joint_trims_in_the_transcript(tmp_path):
    from core.loop import Loop
    from run import make_agents
    from vendor import TerminalAPI
    C.reset()
    api = TerminalAPI("practice_02", data_xlsx=str(WORKBOOK), log_path=str(tmp_path / "actions.jsonl"))
    s = Loop(api, tmp_path, agents=make_agents("multi")).run()
    assert not s["violations"]
    lines = [json.loads(x) for x in (tmp_path / "negotiation.jsonl").read_text(encoding="utf-8").splitlines()]
    joint = [x for x in lines if "Joint check" in json.dumps(x)]
    assert joint
    md = (tmp_path / "negotiation.md").read_text(encoding="utf-8")
    assert "Joint check" in md and "boxes waiting to come off it" in md
