"""CORE-5 tests: coordinator (negotiation and admission control) and binding grants.

Hand-made conflicting requests (the task's named tests), plus: impossible ships refused by a
general check, the write-off judgement offered to the supervisor with FCFS as option 0,
a broken forecast never crashes, the transcript logs each conflict once, grants are binding
in the loop (the yard agent places at most what it was granted, in the same round), and the
5 practice scenarios depart at least as many ships as the skeleton.
"""
import json
import sys
import types
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import pytest

import coordinator as C
from agents.yard import YardAgent
from core import log as runlog
from core.skeleton import BERTHS
from core.types import (GRANTED, REFUSED, TRIMMED, Block, Container, Crane, Request, Ship, Truck,
                        WorldState)

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
T0 = datetime(2026, 9, 1, 6, 0)
TYPES = {"A": "standard", "G": "empty", "H": "IMDG", "I": "reefer"}


@pytest.fixture(autouse=True)
def fresh():
    from agents import admission
    C.reset()
    admission.reset()
    yield
    C.reset()
    admission.reset()


def ship(cid, berth="B1", status="working", discharge=None, imports=100, order_eta=0):
    eta = T0 - timedelta(hours=10 - order_eta)
    return Ship(cid, "MV", 200.0, 11.0, eta, eta, status, berth, ("QC1",), 3,
                imports if discharge is None else discharge, 10, None)


def boxes(cid, n, cargo="dry", start=0, placed=None):
    return [Container(f"{cid}-{cargo}-{i}", cid, "import", 40, cargo, 3.0 if cargo == "IMDG" else None, 20.0, placed)
            for i in range(start, start + n)]


def world(ships, containers, free=None, trucks=(), cranes=()):
    """Blocks A (standard), G (empty), H (IMDG), I (reefer) with the given free slots each."""
    free = free or {}
    blocks = tuple(Block(b, TYPES[b], 0, free.get(b, 0)) for b in "AGHI")
    by_call: dict = {}
    for c in containers:
        by_call.setdefault(c.call_id, []).append(c)
    return WorldState(T0, T0 + timedelta(days=5), tuple(ships), tuple(cranes), tuple(trucks), blocks, (),
                      MappingProxyType({k: tuple(v) for k, v in by_call.items()}), BERTHS)


def yard_req(cid, n, priority=1.0):
    return Request("yard", "yard_slots", n, cid, priority, f"{n} boxes of {cid}")


def no_forecast(monkeypatch):
    monkeypatch.setattr(C, "_forecast", lambda state: (None, None))


def by_call(grants):
    return {g.request.call_id: g for g in grants}


# ---------------------------------------------------------------- yard slots: space and order

def test_conflicting_yard_requests_first_come_first_served_then_trimmed_then_refused(monkeypatch):
    no_forecast(monkeypatch)
    # Arrival order Z1, M2, A3: neither the request order nor the ids' order.
    ships = [ship("Z1", order_eta=0), ship("M2", "B2", order_eta=1), ship("A3", "B3", order_eta=2)]
    st = world(ships, boxes("Z1", 60) + boxes("M2", 60) + boxes("A3", 60), free={"A": 100})
    g = by_call(C.decide([yard_req("A3", 60), yard_req("Z1", 60), yard_req("M2", 60)], st)[0])
    assert (g["Z1"].status, g["Z1"].granted) == (GRANTED, 60)
    assert (g["M2"].status, g["M2"].granted) == (TRIMMED, 40) and "40" in g["M2"].reason
    assert (g["A3"].status, g["A3"].granted) == (REFUSED, 0) and g["A3"].reason


def test_higher_priority_request_is_served_before_arrival_order(monkeypatch):
    no_forecast(monkeypatch)
    st = world([ship("S1"), ship("S2", "B2", order_eta=1)], boxes("S1", 50) + boxes("S2", 50), free={"A": 50})
    g = by_call(C.decide([yard_req("S1", 50, 0.5), yard_req("S2", 50, 1.0)], st)[0])
    assert g["S2"].granted == 50 and g["S1"].status == REFUSED


def test_yard_rules_imdg_to_h_reefer_to_i_dry_overflows_g_then_i(monkeypatch):
    no_forecast(monkeypatch)
    st = world([ship("S1"), ship("S2", "B2", order_eta=1)],
               boxes("S1", 5, "IMDG") + boxes("S1", 10, "dry") + boxes("S2", 10, "reefer"),
               free={"A": 4, "G": 3, "H": 2, "I": 12})
    g = by_call(C.decide([yard_req("S1", 15), yard_req("S2", 10)], st)[0])
    # S1: 2 IMDG (H), 10 dry = 4 A + 3 G + 3 I.  S2: reefer only in I, 12 - 3 = 9 left.
    assert g["S1"].granted == 12 and g["S1"].status == TRIMMED
    assert g["S2"].granted == 9 and g["S2"].status == TRIMMED


def test_reefer_never_goes_to_a_standard_block_even_when_it_has_room(monkeypatch):
    no_forecast(monkeypatch)
    st = world([ship("S1")], boxes("S1", 5, "reefer") + boxes("S1", 5, "IMDG"), free={"A": 50, "G": 50})
    g = C.decide([yard_req("S1", 10)], st)[0][0]
    assert (g.status, g.granted) == (REFUSED, 0)


def test_never_more_than_the_ship_actually_has_unplaced(monkeypatch):
    no_forecast(monkeypatch)
    st = world([ship("S1")], boxes("S1", 5) + boxes("S1", 5, start=5, placed="A"), free={"A": 100})
    g = C.decide([yard_req("S1", 50)], st)[0][0]
    assert g.granted == 5 and g.status == TRIMMED


# ---------------------------------------------------------------- admission control (YARD-3)

def test_impossible_ship_is_refused_by_the_forecast_without_any_id():
    # Must discharge 1,600 boxes but only 40 imports in its manifest: can never finish.
    st = world([ship("Z9", discharge=1600, imports=40)], boxes("Z9", 40), free={"A": 500})
    g = C.decide([yard_req("Z9", 40)], st)[0][0]
    assert g.status == REFUSED and "never finish" in g.reason and "Admission control" in g.reason


class FakeBound:
    def __init__(self, written_off, impossible=()):
        self.written_off, self.impossible = tuple(written_off), tuple(impossible)
        self.predicted_departed, self.predicted_departed_fcfs = 3, 2
        self.reasons = {c: f"Giving {c} no yard slots finishes 3 ships instead of 2." for c in written_off}


def judged(monkeypatch, cid="S2"):
    monkeypatch.setattr(C, "OFFER_FORECAST_WRITE_OFFS", True)
    monkeypatch.setattr(C, "_forecast", lambda st: ({"S1": 999, cid: 0}, FakeBound([cid])))
    return world([ship("S1"), ship(cid, "B2", order_eta=1)], boxes("S1", 10) + boxes(cid, 10), free={"A": 100})


def test_forecast_write_offs_are_off_by_default_and_the_ship_is_served(monkeypatch):
    monkeypatch.setattr(C, "_forecast", lambda st: ({"S1": 999, "S2": 0}, FakeBound(["S2"])))
    st = world([ship("S1"), ship("S2", "B2", order_eta=1)], boxes("S1", 10) + boxes("S2", 10), free={"A": 100})
    grants, options = C.decide([yard_req("S1", 10), yard_req("S2", 10)], st)
    assert options == [] and by_call(grants)["S2"].granted == 10 and C.OFFER_FORECAST_WRITE_OFFS is False


def test_impossible_ships_are_still_refused_with_write_offs_off(monkeypatch):
    monkeypatch.setattr(C, "_forecast", lambda st: ({"S2": 0}, FakeBound(["S2"], impossible=["S2"])))
    st = world([ship("S2")], boxes("S2", 10), free={"A": 100})
    assert C.decide([yard_req("S2", 10)], st)[0][0].status == REFUSED


def test_write_off_is_a_judgement_offered_once_with_serve_all_as_option_0(monkeypatch):
    st = judged(monkeypatch)
    grants, options = C.decide([yard_req("S1", 10), yard_req("S2", 10)], st)
    assert [o.id for o in options] == [C.OPT_FCFS, C.OPT_FORECAST]
    assert all(o.template for o in options) and options[1].predicted["written_off"] == ["S2"]
    assert by_call(grants)["S2"].granted == 10           # served until the supervisor picks the forecast
    C.chosen(options[0].id)
    grants, options = C.decide([yard_req("S1", 10), yard_req("S2", 10)], st)
    assert options == [] and by_call(grants)["S2"].granted == 10


def test_provisional_answer_with_options_pending_is_not_written_to_the_transcript(tmp_path, monkeypatch):
    st = judged(monkeypatch)
    runlog.start(tmp_path)
    try:
        _, options = C.decide([yard_req("S2", 10)], st)
        log = tmp_path / "negotiation.jsonl"
        assert options and (not log.exists() or not log.read_text(encoding="utf-8").strip())
        C.chosen(C.OPT_FORECAST)
        C.decide([yard_req("S2", 10)], st)
    finally:
        runlog.stop()
    lines = [json.loads(x) for x in (tmp_path / "negotiation.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(x["kind"], x["status"]) for x in lines] == [("request", None), ("grant", REFUSED)]


def test_supervisor_can_pick_the_forecast_and_the_ship_is_then_refused(monkeypatch):
    st = judged(monkeypatch)
    _, options = C.decide([yard_req("S2", 10)], st)
    C.chosen(C.OPT_FORECAST)
    g = C.decide([yard_req("S2", 10)], st)[0][0]
    assert g.status == REFUSED and "finishes 3 ships" in g.reason


def test_broken_forecast_never_crashes_and_grants_on_free_space(monkeypatch):
    from agents import admission
    monkeypatch.setattr(admission, "budgets", lambda st: 1 / 0)
    st = world([ship("S1")], boxes("S1", 10), free={"A": 6})
    g = C.decide([yard_req("S1", 10)], st)[0][0]
    assert (g.status, g.granted) == (TRIMMED, 6)


# ---------------------------------------------------------------- trucks, cranes, berths, unknown

def test_trucks_go_to_working_ships_first_then_arrival_order():
    trucks = [Truck(f"T{i}", "available", None) for i in range(3)] + [Truck("T9", "working", "S1")]
    st = world([ship("S1"), ship("S2", "B2", order_eta=1), ship("S3", "B3", order_eta=2)], [], trucks=trucks)
    reqs = [Request("trucks", "trucks", 2, "S1", 0.5, "berthed"), Request("trucks", "trucks", 2, "S2", 1.0, "w"),
            Request("trucks", "trucks", 2, "S3", 1.0, "w")]
    g = by_call(C.decide(reqs, st)[0])
    assert g["S2"].granted == 2 and g["S3"].granted == 1 and g["S3"].status == TRIMMED
    assert g["S1"].status == REFUSED


def test_cranes_and_berths_are_granted_from_what_is_free():
    cranes = [Crane("QC1", "available", None, 1), Crane("QC2", "working", "S1", 5)]
    st = world([ship("S1"), ship("S2", None, "arrived", order_eta=1)], [], cranes=cranes)
    g = C.decide([Request("berth", "cranes", 2, "S1", 1.0, "x"), Request("berth", "berth", 1, "S2", 1.0, "y")], st)[0]
    assert {(x.request.resource, x.status, x.granted) for x in g} == {("cranes", TRIMMED, 1), ("berth", GRANTED, 1)}


def test_unknown_resource_is_refused_not_crashed():
    g = C.decide([Request("x", "tugboats", 1, "S1", 1.0, "?")], world([ship("S1")], []))[0][0]
    assert g.status == REFUSED and "tugboats" in g.reason


# ---------------------------------------------------------------- transcript

def test_transcript_has_each_conflict_once_with_reasons(tmp_path, monkeypatch):
    no_forecast(monkeypatch)
    runlog.start(tmp_path)
    try:
        st = world([ship("S1"), ship("S2", "B2", order_eta=1)], boxes("S1", 60) + boxes("S2", 60), free={"A": 100})
        for _ in range(3):                                 # same outcome three ticks running: logged once
            C.decide([yard_req("S1", 60), yard_req("S2", 60)], st)
        st2 = replace(st, blocks=(Block("A", "standard", 0, 200),) + st.blocks[1:])
        C.decide([yard_req("S1", 60), yard_req("S2", 60)], st2)      # S2 now fits: logged again
    finally:
        runlog.stop()
    lines = [json.loads(x) for x in (tmp_path / "negotiation.jsonl").read_text(encoding="utf-8").splitlines()]
    grants = [(x["call_id"], x["status"]) for x in lines if x["kind"] == "grant"]
    assert grants == [("S1", GRANTED), ("S2", TRIMMED), ("S2", GRANTED)]
    assert sum(x["kind"] == "request" for x in lines) == 3
    md = (tmp_path / "negotiation.md").read_text(encoding="utf-8")
    assert "trimmed S2 from 60 to 40 yard_slots" in md and "wait for boxes to leave the yard" in md


# ---------------------------------------------------------------- binding grants

def test_yard_agent_places_at_most_its_grant_and_asks_for_the_rest():
    st = world([ship("S1")], boxes("S1", 30), free={"A": 100})
    agent = YardAgent()
    acts, reqs = agent.propose(st, [])
    assert sum(len(a.args["container_ids"]) for a in acts) == 30          # no coordinator: v0 behaviour
    assert [(r.resource, r.amount, r.call_id) for r in reqs] == [("yard_slots", 30, "S1")]
    agent.grants = {("yard_slots", "S1"): 12}
    acts, _ = agent.propose(st, [])
    assert sum(len(a.args["container_ids"]) for a in acts) == 12
    agent.grants = {}
    assert agent.propose(st, [])[0] == [] and not agent.escalations     # refused: nothing, no false alarm


def test_loop_hands_out_grants_and_agents_act_within_them_in_the_same_round(tmp_path, monkeypatch):
    from core.loop import Loop
    from tests.test_executor import FakeAPI
    fake = types.ModuleType("coordinator")
    seen = []

    def decide(requests, state):
        seen.extend(requests)
        from core.types import Grant
        return [Grant(r, 0, REFUSED, "test: nothing for anyone") for r in requests], []

    fake.decide = decide
    monkeypatch.setitem(sys.modules, "coordinator", fake)
    api = FakeAPI()
    api.schedule[0] = dict(api.schedule[0], status="working", berth="B1", cranes=["QC1"])
    agent = YardAgent()
    Loop(api, tmp_path, agents=[agent]).step()
    assert seen and seen[0].resource == "yard_slots"
    assert not [c for c in api.calls if c[0] == "place_container"]      # refused grant: nothing placed


def test_without_a_coordinator_module_nothing_changes(tmp_path, monkeypatch):
    from core.loop import Loop
    from tests.test_executor import FakeAPI
    monkeypatch.setattr("core.loop._optional", lambda m: None)         # no coordinator (or other hooks)
    api = FakeAPI()
    api.schedule[0] = dict(api.schedule[0], status="working", berth="B1", cranes=["QC1"])
    agent = YardAgent()
    Loop(api, tmp_path, agents=[agent]).step()
    assert agent.grants is None
    assert [c for c in api.calls if c[0] == "place_container"]         # placed as in v0


# ---------------------------------------------------------------- scenarios

@needs_workbook
@pytest.mark.parametrize("scenario", ["practice_01", "practice_02", "practice_03", "practice_04", "practice_05"])
def test_departed_at_least_skeleton_and_transcript_explains_every_conflict(tmp_path, scenario):
    import run
    skel = {"practice_01": 4, "practice_02": 3, "practice_03": 3, "practice_04": 3, "practice_05": 4}   # CORE-2
    s = run.run_one(scenario, WORKBOOK, None, tmp_path, "multi")
    n_ships = {"practice_01": 4, "practice_02": 6, "practice_03": 6, "practice_04": 5, "practice_05": 4}[scenario]
    assert n_ships - len(s["unfinished_calls"]) >= skel[scenario]
    lines = [json.loads(x) for x in (tmp_path / scenario / "negotiation.jsonl").read_text(encoding="utf-8").splitlines()]
    for g in (x for x in lines if x["kind"] == "grant" and x["status"] != GRANTED):
        assert g["grant_reason"] and g["amount"] is not None and g["granted"] is not None
