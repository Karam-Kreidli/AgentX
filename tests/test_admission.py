"""YARD-3 tests: forecast, feasibility bound and budgets (agents/admission.py)."""
import ast
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agents import admission as A
from core.skeleton import BERTHS
from core.types import Block, Container, Hold, Ship, WorldState

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
T0 = datetime(2026, 9, 1, 0, 0)


@pytest.fixture(autouse=True)
def fresh():
    A.reset()
    yield
    A.reset()


def ship(cid, status="arrived", berth=None, loa=200.0, imports=100, load=100, eta=T0, arrived=T0, discharge=None,
         cranes=()):
    return Ship(cid, "MV", loa, 11.0, eta, arrived if status != "expected" else None, status, berth, cranes, 0,
                imports if discharge is None else discharge, load, None)


def boxes(cid, n, cargo="dry", placed_in=None, start=0):
    return tuple(Container(f"{cid}-{i}", cid, "import", 40, cargo, None, 20.0, placed_in) for i in range(start, start + n))


def blocks(used=None):
    """The real terminal layout: A-F standard 220, G empty 200, H IMDG 120, I/J reefer 200."""
    used = used or {}
    out = [Block(b, "standard", used.get(b, 0), 220) for b in "ABCDEF"]
    out += [Block("G", "empty", used.get("G", 0), 200), Block("H", "IMDG", used.get("H", 0), 120),
            Block("I", "reefer", used.get("I", 0), 200), Block("J", "reefer", used.get("J", 0), 200)]
    return tuple(out)


def small_yard(dry_slots):
    """Only one dry block; no IMDG or reefer room."""
    return (Block("A", "standard", 0, dry_slots),)


def world(ships, manifests, blks=None, now=T0, holds=(), days=5):
    return WorldState(now, T0 + timedelta(days=days), tuple(ships), (), (), blks or blocks(), tuple(holds),
                      manifests, BERTHS)


# ---------------------------------------------------------------- feasibility bound

def test_impossible_ship_is_found_by_a_general_check():
    # Must discharge 1,600 boxes but has only 551 imports: can never finish (the P03-004 case, no ID).
    s = ship("X-1", imports=551, discharge=1600)
    b = A.bound(world([s], {"X-1": boxes("X-1", 551)}))
    assert b.impossible == ("X-1",) and "can never finish" in b.reasons["X-1"]
    assert A.budgets(world([s], {"X-1": boxes("X-1", 551)})) == {"X-1": 0}


def test_ship_with_all_imports_discharged_so_far_is_not_impossible():
    s = ship("X-1", status="working", berth="B1", imports=200, discharge=200, cranes=("QC1", "QC2"))
    st = world([s], {"X-1": boxes("X-1", 200)})
    A.bound(st)                                              # first sight: 200 to discharge, 200 imports
    s2 = ship("X-1", status="working", berth="B1", imports=200, discharge=50, cranes=("QC1", "QC2"))
    st2 = world([s2], {"X-1": boxes("X-1", 150, placed_in="A") + boxes("X-1", 50, start=150)},
                now=T0 + timedelta(hours=4))
    assert A.bound(st2).impossible == ()                     # 150 done, 50 left to discharge, 50 left to place


# ---------------------------------------------------------------- budgets

def test_no_shortage_every_ship_gets_all_its_slots():
    ships = [ship("C1"), ship("C2", eta=T0 + timedelta(hours=5), arrived=T0 + timedelta(hours=5))]
    st = world(ships, {"C1": boxes("C1", 300), "C2": boxes("C2", 300)})
    b = A.bound(st)
    assert not b.shortage and b.written_off == () and set(b.can_finish) == {"C1", "C2"}
    assert A.budgets(st) == {"C1": 300, "C2": 300}


def test_ship_written_off_only_when_that_finishes_strictly_more_ships():
    # 300 dry slots and a 2-day window (no box leaves before ~2.25 days). C1 arrived first and needs 280;
    # C2 and C3 need 140 each. FCFS: C1 takes 280, C2/C3 can't fit -> 1 ship. C1 written off -> 2 ships.
    ships = [ship("C1", imports=280), ship("C2", imports=140, arrived=T0 + timedelta(minutes=5)),
             ship("C3", imports=140, arrived=T0 + timedelta(minutes=10))]
    st = world(ships, {"C1": boxes("C1", 280), "C2": boxes("C2", 140), "C3": boxes("C3", 140)},
               small_yard(300), days=2)
    b = A.bound(st)
    assert b.predicted_departed_fcfs == 1 and b.predicted_departed == 2
    assert b.written_off == ("C1",) and "instead of 1" in b.reasons["C1"]
    assert A.budgets(st) == {"C1": 0, "C2": 140, "C3": 140}


def test_tie_keeps_first_come_first_served():
    # Only one of two ships can ever fit; writing either off doesn't finish more: nobody is written off.
    ships = [ship("C1", imports=250), ship("C2", imports=250, arrived=T0 + timedelta(minutes=5))]
    st = world(ships, {"C1": boxes("C1", 250), "C2": boxes("C2", 250)}, small_yard(300), days=2)
    b = A.bound(st)
    assert b.written_off == () and b.shortage
    assert A.budgets(st) == {"C1": 250, "C2": 250}


def test_unknown_cargo_and_placed_boxes_are_not_counted_in_budgets():
    rows = boxes("C1", 10) + boxes("C1", 5, cargo="mystery", start=10) + boxes("C1", 7, placed_in="A", start=20)
    st = world([ship("C1", imports=22)], {"C1": rows})
    assert A.budgets(st) == {"C1": 10}


def test_departed_and_sailing_ships_get_no_budget():
    ships = [ship("C1", status="departed"), ship("C2", status="sailing", berth="B1", discharge=0, load=0)]
    assert A.budgets(world(ships, {"C1": boxes("C1", 3), "C2": ()})) == {}


# ---------------------------------------------------------------- forecast

def test_release_prior_is_zero_before_min_dwell_and_one_after_max():
    h = lambda d: A.RELEASE_DELAY_H + d * 24
    assert A._left_by(h(A.DWELL_MIN_D)) == 0.0
    assert A._left_by(h(A.DWELL_MAX_D)) == 1.0
    assert A._left_by(h((A.DWELL_MIN_D + A.DWELL_MAX_D) / 2)) == pytest.approx(0.5)


def test_observed_releases_are_counted_and_calibrate_the_forecast():
    s = ship("C1", status="working", berth="B1", imports=100, cranes=("QC1", "QC2"))
    placed = boxes("C1", 100, placed_in="A")
    A.bound(world([s], {"C1": placed}, blocks(used={"A": 100})))
    later = T0 + timedelta(days=5)                            # prior expects half gone by ~5 d
    A.bound(world([s], {"C1": placed}, blocks(used={"A": 90}), now=later))   # only 10 left
    assert A._mem.released["dry"] == 10
    cal = A._calibration(world([s], {"C1": placed}, blocks(used={"A": 90}), now=later))
    assert cal["dry"] < 0.5                                   # releases slower than the prior: scale down


def test_held_boxes_are_never_forecast_to_leave():
    s = ship("C1", status="working", berth="B1", imports=10, cranes=("QC1", "QC2"))
    placed = boxes("C1", 10, placed_in="A")
    holds = tuple(Hold(c.container_id, T0, False) for c in placed)
    st = world([s], {"C1": placed}, blocks(used={"A": 10}), holds=holds)
    A.bound(st)
    assert A._cohorts(st) == []


def test_new_scenario_resets_memory():
    st = world([ship("C1")], {"C1": boxes("C1", 5, placed_in="A")}, blocks(used={"A": 5}))
    A.bound(st)
    assert A._mem.seen_at
    other = world([], {}, days=6)
    A.bound(other)
    assert not A._mem.seen_at


# ---------------------------------------------------------------- honest play

def test_never_reads_the_workbook_or_the_simulator():
    tree = ast.parse((ROOT / "agents" / "admission.py").read_text(encoding="utf-8"))
    names = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
            {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not names & {"vendor", "pandas", "openpyxl", "core.state", "core.executor"}
    src = (ROOT / "agents" / "admission.py").read_text(encoding="utf-8")
    assert "xlsx" not in src and "P03-004" not in src.split('"""', 2)[2]     # no IDs in the code


# ---------------------------------------------------------------- real scenarios

@needs_workbook
@pytest.mark.parametrize("scenario", [f"practice_0{i}" for i in range(1, 6)])
def test_budgets_are_sane_and_fast_on_every_tick(scenario, tmp_path):
    from core.loop import Loop
    from core.state import build_state
    from run import make_agents
    from vendor import TerminalAPI
    api = TerminalAPI(scenario, data_xlsx=str(WORKBOOK), log_path=str(tmp_path / "actions.jsonl"))
    loop = Loop(api, tmp_path, agents=make_agents("multi"))
    times, flagged = [], set()
    while True:
        loop.step()
        s = build_state(api)
        t0 = time.perf_counter()
        bud = A.budgets(s)
        b = A.bound(s)
        times.append(time.perf_counter() - t0)
        flagged |= set(b.impossible)
        for s_ in s.ships:
            if s_.call_id in bud:
                unplaced = sum(1 for c in s.manifests.get(s_.call_id, ()) if c.direction == "import"
                               and not c.yard_block and c.cargo_type in ("dry", "reefer", "IMDG"))
                assert 0 <= bud[s_.call_id] <= unplaced
        assert b.predicted_departed >= b.predicted_departed_fcfs
        if s.now >= s.scenario_end:
            break
        api.call("advance_time", {"minutes": loop._tick_minutes(s)})
        loop.tick += 1
    times.sort()
    assert times[len(times) // 2] < 0.05 and sum(times) < 30
    # P03-004 is the only practice ship that can never finish; found without its ID.
    assert flagged == ({"P03-004"} if scenario == "practice_03" else set())


def test_release_prior_in_hours():
    # A box seen a day ago hasn't left; after 3 days some have; after 9 days all have.
    assert A._left_by(24) == 0.0 and A._left_by(54) == 0.0          # 2 d + 6 h: none yet
    assert 0 < A._left_by(3 * 24) < 0.5 and A._left_by(9 * 24) == 1.0
