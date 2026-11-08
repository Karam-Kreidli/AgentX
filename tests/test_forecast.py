"""Review follow-up (Sun 8 Nov): the yard forecast and the berth options use the crane agent's
own way of sharing cranes, and the forecast never releases a box twice.

- agents/crane.py split(): one crane each for every ship with work, spare cranes to the
  shortest job, capped by what each ship's work keeps busy.
- agents/admission.py: boxes the yard has already let go are taken off the oldest end of each
  group, a cohort never releases more than it holds, the plan is redone when a yard group's
  free space or the boxes on hold change, and cranes are shared, not 2 per berth.
- agents/options.py: berth-option finish times use the same sharing.
"""
from datetime import timedelta

import pytest

from agents import admission as A
from agents import crane as C
from agents import options as O
from core.types import Crane, Hold
from tests.test_admission import T0, blocks, boxes, ship, world


@pytest.fixture(autouse=True)
def fresh():
    A.reset()
    yield
    A.reset()


# ---------------------------------------------------------------- split

def test_split_gives_everyone_with_work_a_crane_then_the_shortest_job_the_rest():
    out = C.split({"BIG": 900, "SMALL": 60, "IDLE": 0}, {"BIG": 1500, "SMALL": 60, "IDLE": 40}, 6, 1.0)
    assert out == {"BIG": 3, "SMALL": 3, "IDLE": 0}                       # SMALL: 60 moves keep 3 busy for 1 h
    assert C.split({"A": 500, "B": 500, "C": 500}, {"A": 9, "B": 8, "C": 7}, 2, 1.0) == {"A": 0, "B": 1, "C": 1}
    assert C.split({"A": 10_000}, {"A": 10_000}, 6, 1.0) == {"A": 6}
    assert C.split({"A": 10_000}, {"A": 10_000}, 0, 1.0) == {"A": 0}


def test_in_service_counts_working_cranes_and_six_when_none_are_listed():
    st = world([], {})
    assert C.in_service(st) == 6
    st = st.__class__(**{**st.__dict__, "cranes": (Crane("QC1", "available", None, 1),
                                                  Crane("QC2", "out_of_service", None, 14))})
    assert C.in_service(st) == 1


# ---------------------------------------------------------------- the forecast

def test_a_lone_ship_gets_all_the_cranes_in_the_model():
    # 1,200 load moves, nothing to discharge: 6 cranes x 23/h ~ 9.7 h; 2 cranes would take ~26 h.
    s = ship("L", status="working", berth="B1", imports=0, discharge=0, load=1200)
    b = A.bound(world([s], {}, days=0.5))                                  # a 12-hour window
    assert "L" in b.can_finish


def test_boxes_that_have_left_are_not_released_again():
    old = T0 - timedelta(days=3)
    for i in range(100):
        A._mem.seen_at[f"X-{i}"] = (old, A.G_DRY)
    st = world([], {})
    assert sum(n for _, _, n in A._cohorts(st)) == 100
    A._mem.released[A.G_DRY] = 60                                          # the yard count fell by 60
    left = A._cohorts(st)
    assert sum(n for _, _, n in left) == 40                                # only the 40 still there


def test_a_cohort_never_releases_more_than_it_holds():
    for cal in (0.5, 1.0, 1.5):
        shares = [A._released_share(-30.0, x, cal) for x in range(0, 400, 7)]
        assert shares[0] == 0.0 and max(shares) <= 1.0
        assert all(b >= a for a, b in zip(shares, shares[1:]))            # never negative releases
    overdue = -24 * (A.DWELL_MAX_D + 2)
    assert A._released_share(overdue, 0.0, 1.0) == 0.0 and A._released_share(overdue, 1.0, 1.0) == 1.0


def test_the_plan_is_redone_when_yard_space_or_holds_change():
    s = ship("S", status="working", berth="B1", imports=10, load=10)
    mani = {"S": boxes("S", 10)}
    b1 = A.bound(world([s], mani, blks=blocks()))
    assert A.bound(world([s], mani, blks=blocks())) is b1                 # nothing changed: cached
    b2 = A.bound(world([s], mani, blks=blocks({"A": 120})))               # 120 fewer free dry slots
    assert b2 is not b1
    b3 = A.bound(world([s], mani, blks=blocks({"A": 120}), holds=(Hold("S-1", T0, False),)))
    assert b3 is not b2


# ---------------------------------------------------------------- berth options

def test_berth_option_finish_times_share_the_cranes():
    from tests.test_options import ship as oship, world as oworld
    lone = oship("ONLY", arrived_h=-1, moves=690)
    st = oworld([lone])
    pred = O._predict(st, lone, "B1", [lone])
    # 690 moves / (6 x 23 per h) = 5 h, + 1 h before work starts + 1 h to sail = ~7 h (2 cranes: ~17 h)
    assert 6.5 <= pred["finish_h"]["ONLY"] <= 7.5
