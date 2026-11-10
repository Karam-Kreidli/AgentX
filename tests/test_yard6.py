"""YARD-6: the yard parameters in config/params.yaml (keep a ship's boxes together, a share of the
reefer blocks kept for reefers, the soft fill limit), the coordinator granting exactly what the
yard agent can place under them, and the blocks-per-ship report."""
from dataclasses import replace
from datetime import timedelta

import coordinator
from agents import yard
from core.explain import Explainer
from core.types import CARGO_DRY, CARGO_REEFER
from tests.test_yard import T0, boxes, plan, ship, world


def test_default_is_nearest_block():
    s = world([ship()], boxes(10) + boxes(5, placed="C", start=100))
    placed, _, _ = plan(s)
    assert set(placed.values()) == {"A"}                       # A is nearest to B1; the ship's C boxes don't matter


def test_together_prefers_a_block_that_already_holds_the_ships_boxes(monkeypatch):
    monkeypatch.setattr(yard, "PREFER", "together")
    s = world([ship()], boxes(10) + boxes(5, placed="C", start=100))
    placed, actions, _ = plan(s)
    assert set(placed.values()) == {"C"}
    assert "already holds 5 of its boxes" in actions[0].reason


def test_together_still_respects_the_soft_fill_limit(monkeypatch):
    monkeypatch.setattr(yard, "PREFER", "together")
    s = world([ship()], boxes(10) + boxes(5, placed="C", start=100), used={"C": 187})   # C is at 85 %
    placed, _, _ = plan(s)
    assert "C" not in placed.values() and set(placed.values()) == {"A"}


def test_another_ships_boxes_do_not_attract(monkeypatch):
    monkeypatch.setattr(yard, "PREFER", "together")
    s = world([ship()], boxes(10) + boxes(5, call="C2", placed="C", start=100))
    placed, _, _ = plan(s)
    assert set(placed.values()) == {"A"}


def _full_dry_world(**used):
    base = {b: 220 for b in "ABCDEF"} | {"G": 200}
    return base | used


def test_reefer_keep_stops_dry_boxes_short_of_the_reefer_share(monkeypatch):
    monkeypatch.setattr(yard, "REEFER_KEEP", 0.5)
    s = world([ship()], boxes(300), used=_full_dry_world())   # standard and G full: dry can only go to I/J
    placed, _, _ = plan(s)
    per = {b: sum(1 for x in placed.values() if x == b) for b in "IJ"}
    assert per == {"I": 100, "J": 100}                          # half of each 200-slot reefer block


def test_reefer_keep_does_not_limit_reefers(monkeypatch):
    monkeypatch.setattr(yard, "REEFER_KEEP", 1.0)
    s = world([ship()], boxes(50, "reefer") + boxes(50), used=_full_dry_world())
    placed, _, _ = plan(s)
    assert sum(1 for c in placed if c.startswith("R")) == 50
    assert not any(c.startswith("D") for c in placed)


def test_coordinator_grants_what_the_yard_agent_can_place(monkeypatch):
    for keep in (0.0, 0.5, 1.0):
        monkeypatch.setattr(yard, "REEFER_KEEP", keep)
        s = world([ship()], boxes(300) + boxes(30, "reefer"), used=_full_dry_world(I=20))
        free = coordinator.free_space(s)
        n = coordinator.fit({CARGO_DRY: 300, CARGO_REEFER: 30}, free)
        placed, _, _ = plan(s)
        assert n == len(placed), keep


def test_fit_without_the_new_key_behaves_as_before():
    free = {"IMDG": 0, "reefer": 50, "standard": 0, "empty": 0}
    assert coordinator.fit({CARGO_DRY: 80}, free) == 50


def test_spread_counts_the_blocks_each_ships_boxes_went_to():
    e = Explainer()
    s1 = world([ship(), ship("C2", berth="B2")], boxes(3, placed="A") + boxes(2, placed="B", start=10)
               + boxes(4, call="C2", placed="D"))
    e.observe(s1)
    s2 = world([ship(status="departed"), ship("C2", berth="B2")], boxes(5, call="C2", placed="E", start=20))
    e.observe(replace(s2, now=T0 + timedelta(hours=1)))
    assert e.spread() == {"C1": 2, "C2": 2}                     # C1 kept A+B after it left; C2 added E
