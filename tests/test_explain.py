"""Review follow-up: one feed sentence per re-plan, and why each unfinished ship did not finish
(core/explain.py)."""
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from agents import stall
from core.explain import Explainer
from tests.test_coordinator import T0, boxes, ship, world  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"


@pytest.fixture(autouse=True)
def fresh_stall():
    stall.reset()
    yield
    stall.reset()


def at(st, h):
    """The same world h hours later."""
    return replace(st, now=st.now + timedelta(hours=h))


def run(ex, states):
    for s in states:
        ex.observe(s)


# ------------------------------------------------------------------ re-plan sentences
def test_berthing_and_crane_split_make_one_sentence():
    a = replace(ship("A", "B1"), cranes=("QC1", "QC2"))
    b0 = replace(ship("B", None, status="arrived"), cranes=())
    b1 = replace(ship("B", "B3", status="berthed"), cranes=())
    before = world([a, b0], boxes("A", 100, placed="A") + boxes("B", 100, placed="A"))
    after = world([replace(a, cranes=("QC1",)), replace(b1, cranes=("QC2",))],
                  boxes("A", 100, placed="A") + boxes("B", 100, placed="A"))
    s = Explainer().replan(before, after)
    assert s.startswith("Re-plan: ") and s.endswith(".") and s.count("Re-plan") == 1
    assert "B berthed on B3 after waiting 10.0 h" in s and "cranes now A 1, B 1" in s


def test_no_change_no_sentence():
    st = world([ship("A", "B1")], boxes("A", 100, placed="A"))
    assert Explainer().replan(st, st) is None


def test_a_ship_left_without_cranes_says_why():
    a = replace(ship("A", "B1"), cranes=("QC1",))
    b = replace(ship("B", "B3", order_eta=1), cranes=("QC2",))
    containers = boxes("A", 100, placed="A") + boxes("B", 100)        # B's boxes have nowhere to go
    before = world([a, b], containers, free={"A": 0})
    stall.stalled_calls(before)
    after = world([replace(a, cranes=("QC1", "QC2")), replace(b, cranes=())], containers, free={"A": 0})
    s = Explainer().replan(before, after)
    assert "cranes now A 2" in s and "B has none while it waits for yard space" in s


# ------------------------------------------------------------------ why unfinished
def test_hours_go_to_the_cause_seen_at_the_last_tick():
    st = world([replace(ship("A", "B1"), cranes=("QC1",), trucks=1)], boxes("A", 100, placed="A"))
    ex = Explainer()
    run(ex, [st, at(st, 2), at(st, 3)])
    assert ex.hours["A"]["trucks"] == pytest.approx(3) and ex.hours["A"]["working"] == 0


def test_short_of_trucks_is_named_as_the_main_cause():
    st = world([replace(ship("A", "B1"), cranes=("QC1", "QC2"), trucks=2)], boxes("A", 100, placed="A"))
    ex = Explainer()
    run(ex, [st, at(st, 10)])
    why = ex.unfinished(at(st, 10))["A"]
    assert "10.0 h short of trucks" in why and why.endswith("Main cause: short of trucks.")


def test_working_all_along_means_it_ran_out_of_time():
    st = world([replace(ship("A", "B1", discharge=600), cranes=("QC1", "QC2"), trucks=6)],
               boxes("A", 600, placed="A"))
    ex = Explainer()
    run(ex, [st, at(st, 5)])
    why = ex.unfinished(at(st, 5))["A"]
    assert "ran out of time with 610 moves left (about 4.4 h more" in why and why.endswith("Main cause: time.")


def test_a_ship_that_never_got_a_berth():
    st = world([replace(ship("A", None, status="arrived"), cranes=())], boxes("A", 100))
    ex = Explainer()
    run(ex, [st, at(st, 4)])
    why = ex.unfinished(at(st, 4))["A"]
    assert why.startswith("A waited 4.0 h for a berth and never got one")


def test_an_impossible_manifest_is_named_first():
    st = world([ship("A", "B1", discharge=100)], boxes("A", 40))       # 60 boxes the manifest never had
    stall.stalled_calls(st)
    why = Explainer().unfinished(st)["A"]
    assert "can never finish" in why and "only 40 import boxes" in why


def test_departed_ships_are_not_listed_and_nothing_raises():
    st = world([replace(ship("A", "B1"), status="departed")], boxes("A", 1, placed="A"))
    ex = Explainer()
    assert ex.unfinished(st) == {}
    ex.observe(None)                                     # never raises
    assert ex.replan(None, st) is None


# ------------------------------------------------------------------ whole run
@pytest.mark.skipif(not WORKBOOK.exists(), reason="needs the data workbook (git-ignored)")
def test_practice_02_writes_reasons_and_replan_sentences(tmp_path):
    import run as runner
    runner.main(["--scenario", "practice_02", "--no-llm", "--runs", str(tmp_path)])
    d = tmp_path / "practice_02"
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["unfinished_why"]) == set(summary["unfinished_calls"])
    assert all("Main cause" in w or "never" in w or "had not arrived" in w for w in summary["unfinished_why"].values())
    assert (d / "unfinished.md").exists()
    assert "Re-plan: " in (d / "supervisor_feed.txt").read_text(encoding="utf-8")


def test_stopped_for_yard_space_is_named_with_the_boxes_left_unplaced():
    a = replace(ship("A", "B1", discharge=100), cranes=("QC1",), trucks=3)
    st = world([a], boxes("A", 100), free={"A": 0})              # no room for any of A's boxes
    stall.stalled_calls(st)
    ex = Explainer()
    run(ex, [st, at(st, 6)])
    why = ex.unfinished(at(st, 6))["A"]
    assert "6.0 h stopped for yard space" in why and "100 of its boxes still had no yard slot" in why
    assert why.endswith("Main cause: stopped for yard space.")
