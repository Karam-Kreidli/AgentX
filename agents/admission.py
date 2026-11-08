"""Yard flow forecast, feasibility bound and admission control (YARD-3).

    from agents.admission import budgets, bound
    budgets(state) -> dict[call_id, int]   # import slots each live ship may take now
    bound(state)   -> Bound                # forecast, which ships can finish, which are written off

Why: the yard is the bottleneck. First-come-first-served placement lets every berthed ship
take slots as soon as it can, so in a shortage a big early ship can soak up the yard and leave
later ships unable to finish. Admission control decides which ships get the scarce slots.

How (only what the tools show at the current time; no workbook, no hidden data):

1. Model. A small forward simulation of the terminal from the current snapshot to the end of
   the window, in STEP_H-hour steps: ships arrive (actual arrival, else scheduled ETA), take the
   first free berth they fit, place their imports in the yard (IMDG -> H, reefer -> I/J,
   dry -> A-G then I/J) as space allows, discharge only what is placed, then load and sail.
   Cranes (CRANE_RATE moves/h each) are shared among the ships at berth the way the crane
   agent shares them (agents/crane.py split(): one each for every ship with work it can reach,
   spare ones to the shortest job), not a fixed number per berth.
2. Yard forecast. Import boxes leave the yard some days after they are placed (customs-held
   ones stay). Per-box dwell is hidden, so the model assumes a uniform DWELL_MIN_D..DWELL_MAX_D
   days (+6 h) from when the box was first seen in the yard, and scales future releases by the
   releases actually observed so far versus expected.
3. Feasibility bound. A ship that must discharge more boxes than its manifest has imports left
   can never finish (the P03-004 case); found by a general check, never by ID.
4. Decision. The model is run with no ship written off, then with each single ship written off
   (no slots, no berth). A ship is written off only if that is predicted to finish strictly more
   ships in the window; ties keep first-come-first-served, which also gives the better
   turnaround. Impossible ships are always written off.
5. Budgets: a live ship that is not written off may take all its unplaced imports now (exactly
   what v0 does); a written-off or impossible ship gets 0. With no shortage nothing is written
   off, so behaviour equals first-come-first-served (throughput only).

Enforcement is CORE-5's job: the yard agent will ask for "yard_slots" and the coordinator will
grant at most these budgets; BERTH-2 will prefer ships with a budget. This module only computes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from agents import crane as crane_model

from core.types import (BLOCK_IMDG, BLOCK_REEFER, CARGO_DRY, CARGO_IMDG, CARGO_REEFER, SHIP_ARRIVED,
                        SHIP_BERTHED, SHIP_EXPECTED, SHIP_SAILING, SHIP_WORKING, WorldState)

# Assumptions (no tool exposes these). Tune in YARD-6 on practice + variants, never per scenario.
DWELL_MIN_D = 2.0          # earliest a placed import box leaves, days after placement
DWELL_MAX_D = 8.0          # latest
RELEASE_DELAY_H = 6.0      # the simulator adds 6 h to the dwell
CRANE_RATE = 23.0          # moves/h per crane at full trucks (rated ~27 x 0.85)
WORK_DELAY_H = 1.0         # work starts 1 h after berthing
SAIL_DELAY_H = 1.0         # a ship sails 1 h after its last move
STEP_H = 1.0               # model resolution
REPLAN_H = 1.0             # re-plan at least this often, and whenever a ship's status/berth changes
REPLAN_SLOTS = 50          # ... or a yard group's free space moves by this many slots

G_IMDG, G_REEFER, G_DRY = "imdg", "reefer", "dry"
GROUPS = (G_IMDG, G_REEFER, G_DRY)
LIVE = (SHIP_EXPECTED, SHIP_ARRIVED, SHIP_BERTHED, SHIP_WORKING)
AT_BERTH = (SHIP_BERTHED, SHIP_WORKING, SHIP_SAILING)


@dataclass(frozen=True)
class Bound:
    now: datetime
    shortage: bool                          # True if, under FCFS, some live ship is predicted not to finish
    can_finish: tuple[str, ...]             # predicted to finish in the window (with the chosen write-offs)
    cannot_finish: tuple[str, ...]          # predicted not to finish, but still served
    written_off: tuple[str, ...]            # given no slots: freeing theirs finishes more ships
    impossible: tuple[str, ...]             # more discharge than imports left: never finishes
    predicted_departed: int                 # live ships predicted to finish, with the chosen write-offs
    predicted_departed_fcfs: int            # with none (first-come-first-served)
    free_now: dict                          # free slots per group
    free_at_end: dict                       # forecast free slots per group at the window end
    reasons: dict = field(default_factory=dict)   # call_id -> one sentence

    def summary(self) -> str:
        return (f"predicted finishing {self.predicted_departed} (FCFS {self.predicted_departed_fcfs}); "
                f"written off {list(self.written_off) or 'none'}; impossible {list(self.impossible) or 'none'}")


# --------------------------------------------------------------------------
# Memory across calls in one scenario: when each box was first seen in the yard, observed
# releases, and each ship's discharge count when first seen. Resets on a new scenario.
# --------------------------------------------------------------------------

class _Memory:
    def __init__(self) -> None:
        self.reset(None)

    def reset(self, key) -> None:
        self.key = key
        self.seen_at: dict[str, tuple[datetime, str]] = {}
        self.first_discharge: dict[str, int] = {}
        self.last_used: Optional[dict] = None
        self.released = {g: 0 for g in GROUPS}
        self.last_now: Optional[datetime] = None
        self.cache: Optional[tuple] = None


_mem = _Memory()


def reset() -> None:
    """Forget the current scenario (a new scenario resets automatically)."""
    _mem.reset(None)


def _group_of_block(block_type: str) -> str:
    return G_IMDG if block_type == BLOCK_IMDG else G_REEFER if block_type == BLOCK_REEFER else G_DRY


def _group_of_cargo(cargo: str) -> Optional[str]:
    return {CARGO_IMDG: G_IMDG, CARGO_REEFER: G_REEFER, CARGO_DRY: G_DRY}.get(cargo)


def _observe(state: WorldState) -> None:
    key = (state.scenario_end,)
    if _mem.key != key or (_mem.last_now is not None and state.now < _mem.last_now):
        _mem.reset(key)
    btype = {b.block_id: b.block_type for b in state.blocks}
    new_here = {g: 0 for g in GROUPS}
    for rows in state.manifests.values():
        for c in rows:
            if c.direction == "import" and c.yard_block and c.container_id not in _mem.seen_at:
                g = _group_of_block(btype.get(c.yard_block, ""))
                _mem.seen_at[c.container_id] = (state.now, g)
                new_here[g] += 1
    used = {g: 0 for g in GROUPS}
    for b in state.blocks:
        used[_group_of_block(b.block_type)] += b.used
    if _mem.last_used is not None:
        for g in GROUPS:
            rel = _mem.last_used[g] + new_here[g] - used[g]       # used now = before + placed - released
            if rel > 0:
                _mem.released[g] += rel
    _mem.last_used = used
    for s in state.ships:
        _mem.first_discharge.setdefault(s.call_id, s.discharge_remaining)
    _mem.last_now = state.now


# --------------------------------------------------------------------------
# Release forecast
# --------------------------------------------------------------------------

def _left_by(age_h: float) -> float:
    """Expected share of boxes placed `age_h` hours ago that have left (uniform dwell prior)."""
    d = (age_h - RELEASE_DELAY_H) / 24
    if d <= DWELL_MIN_D:
        return 0.0
    if d >= DWELL_MAX_D:
        return 1.0
    return (d - DWELL_MIN_D) / (DWELL_MAX_D - DWELL_MIN_D)


def _calibration(state: WorldState) -> dict:
    """Observed / expected releases so far per group (1.0 until there is enough evidence)."""
    held = state.held_ids()
    exp = {g: 0.0 for g in GROUPS}
    for cid, (t0, g) in _mem.seen_at.items():
        if cid not in held:
            exp[g] += _left_by((state.now - t0).total_seconds() / 3600)
    return {g: (1.0 if exp[g] < 20 else max(0.25, min(1.5, _mem.released[g] / exp[g]))) for g in GROUPS}


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

@dataclass
class _Ship:
    call_id: str
    order: int
    arrive_h: float                 # hours from now (<= 0 if already arrived)
    loa: float
    draft: float
    berth: Optional[str]
    work_from_h: float              # hours from now when cranes may work
    cranes: int
    need: dict                      # unplaced imports per group
    placed_waiting: int             # placed but not yet discharged
    discharge: int                  # discharge moves left
    load: int                       # load moves left
    sail_h: Optional[float] = None  # hours from now when it sails (set when its work is done)


def _ships(state: WorldState) -> tuple[list[_Ship], dict, list[str]]:
    """Model ships for live and sailing calls, one-sentence reasons, and the impossible calls."""
    out, reasons, impossible = [], {}, []

    def hours(t: datetime) -> float:
        return (t - state.now).total_seconds() / 3600

    for i, s in enumerate(state.ships):
        if s.status not in LIVE and s.status != SHIP_SAILING:
            continue
        need = {g: 0 for g in GROUPS}
        total = placed = 0
        for c in state.manifests.get(s.call_id, ()):
            if c.direction != "import":
                continue
            total += 1
            g = _group_of_cargo(c.cargo_type)
            if c.yard_block:
                placed += 1
            elif g is not None:
                need[g] += 1
        done = max(0, _mem.first_discharge.get(s.call_id, s.discharge_remaining) - s.discharge_remaining)
        if s.status in LIVE and s.discharge_remaining > total - done:
            impossible.append(s.call_id)
            reasons[s.call_id] = (f"{s.call_id} must discharge {s.discharge_remaining} more boxes but its manifest "
                                  f"has only {max(0, total - done)} import boxes left, so it can never finish.")
            continue
        arrive = s.actual_arrival or s.scheduled_eta
        at_berth = s.status in AT_BERTH and bool(s.berth)
        out.append(_Ship(
            s.call_id, i, hours(arrive), s.loa_m, s.draft_m, s.berth if at_berth else None,
            0.0 if s.status in (SHIP_WORKING, SHIP_SAILING) else (WORK_DELAY_H if at_berth else 0.0),
            len(s.cranes), need, max(0, placed - done), s.discharge_remaining,
            s.load_remaining, sail_h=(SAIL_DELAY_H if s.status == SHIP_SAILING else None)))
    return out, reasons, impossible


def _cohorts(state: WorldState) -> list[tuple[float, str, float]]:
    """Boxes still in the yard, grouped by (hour first seen, group): (start hours from now, group, count).

    The manifest keeps a box's yard block after the box has left, so departures can't be seen
    one by one, only as a count per group (_mem.released). That many boxes are taken off the
    oldest end of each group (the dwell prior has older boxes leave first), so a box that has
    already gone is not released a second time in the forecast. Held boxes stay."""
    held = state.held_ids()
    per_group: dict = {g: [] for g in GROUPS}
    for cid, (t0, g) in _mem.seen_at.items():
        if cid not in held:
            per_group[g].append(t0)
    buckets: dict = {}
    for g, times in per_group.items():
        times.sort()
        for t0 in times[min(len(times), _mem.released[g]):]:
            k = (round((t0 - state.now).total_seconds() / 3600, 2), g)
            buckets[k] = buckets.get(k, 0) + 1
    return [(start, g, float(n)) for (start, g), n in buckets.items()]


def _released_share(start: float, x: float, cal: float) -> float:
    """Share of a cohort still in the yard now (or placed at `start` > 0 in the model) that has left
    by model time x: the dwell prior conditioned on not having left yet, scaled by the observed
    calibration, never more than all of it."""
    f0 = _left_by(max(0.0, -start))
    if f0 >= 1.0:
        return 1.0 if x > 0 else 0.0                # overdue by the prior: assume it leaves in the first step
    return min(1.0, cal * (_left_by(x - start) - f0) / (1.0 - f0))


def _simulate(state: WorldState, skip: frozenset, horizon_h: float, cal: dict) -> tuple[set, dict]:
    """Run the model; returns (call_ids that sail within the window, free slots per group at the end)."""
    ships, _, _ = _ships(state)
    ships = [s for s in ships if s.call_id not in skip]
    occupied = {s.berth: s.call_id for s in ships if s.berth}
    n_cranes = crane_model.in_service(state)
    free = {g: 0.0 for g in GROUPS}
    for b in state.blocks:
        free[_group_of_block(b.block_type)] += b.free
    cohorts = _cohorts(state)
    finished: set = set()
    t = 0.0
    while t < horizon_h:
        t1 = t + STEP_H
        for start, g, n in cohorts:                                  # yard releases in (t, t1]
            if t1 <= start:
                continue
            free[g] += n * (_released_share(start, t1, cal[g]) - _released_share(start, max(t, start), cal[g]))
        for s in sorted(ships, key=lambda x: (x.arrive_h, x.order)):  # berthing, arrival order
            if s.berth or s.sail_h is not None or s.arrive_h > t:
                continue
            for b in state.berths:
                if b.berth_id not in occupied and b.fits(s.loa, s.draft):
                    s.berth, occupied[b.berth_id] = b.berth_id, s.call_id
                    s.work_from_h = max(t, s.arrive_h) + WORK_DELAY_H
                    break
        for s in sorted(ships, key=lambda x: x.order):                # placement, schedule order (v0)
            if not s.berth or s.sail_h is not None or not any(s.need.values()):
                continue
            n_i = min(s.need[G_IMDG], int(free[G_IMDG]))
            n_r = min(s.need[G_REEFER], int(free[G_REEFER]))
            free[G_IMDG] -= n_i
            free[G_REEFER] -= n_r
            n_d1 = min(s.need[G_DRY], int(free[G_DRY]))
            free[G_DRY] -= n_d1
            n_d2 = min(s.need[G_DRY] - n_d1, int(free[G_REEFER]))
            free[G_REEFER] -= n_d2
            for g, n in ((G_IMDG, n_i), (G_REEFER, n_r + n_d2), (G_DRY, n_d1)):
                if n:
                    cohorts.append((t, g, float(n)))
            s.need[G_IMDG] -= n_i
            s.need[G_REEFER] -= n_r
            s.need[G_DRY] -= n_d1 + n_d2
            s.placed_waiting += n_i + n_r + n_d1 + n_d2
        work, left, avail = {}, {}, {}                                # cranes: shared the crane agent's way
        for s in ships:
            if not s.berth or s.sail_h is not None:
                continue
            avail[s.call_id] = max(0.0, t1 - max(t, s.work_from_h))
            reach = s.discharge + s.load if s.placed_waiting >= s.discharge else s.placed_waiting
            work[s.call_id] = reach if avail[s.call_id] > 0 else 0
            left[s.call_id] = s.discharge + s.load
        cranes = crane_model.split(work, left, n_cranes, STEP_H, CRANE_RATE)
        for s in ships:                                              # discharge placed, then load
            if not s.berth or s.sail_h is not None:
                continue
            cap = CRANE_RATE * cranes.get(s.call_id, 0) * avail.get(s.call_id, 0.0)
            d = int(min(cap, s.discharge, s.placed_waiting))
            s.discharge -= d
            s.placed_waiting -= d
            cap -= d
            if s.discharge <= 0:
                s.load -= int(min(cap, s.load))
                if s.load <= 0:
                    s.sail_h = t1 + SAIL_DELAY_H
        for s in ships:
            if s.sail_h is not None and s.sail_h <= t1 and s.call_id not in finished:
                if s.sail_h <= horizon_h:
                    finished.add(s.call_id)
                if s.berth and occupied.get(s.berth) == s.call_id:
                    del occupied[s.berth]
        t = t1
    return finished, {g: int(v) for g, v in free.items()}


def plan(state: WorldState) -> Bound:
    """The admission plan for this snapshot (cached per snapshot time)."""
    _observe(state)
    free = {g: 0 for g in GROUPS}
    for b in state.blocks:
        free[_group_of_block(b.block_type)] += b.free
    # Re-plan on a ship's status or berth, the cranes in service, the boxes on hold, or a yard group's
    # free space moving by REPLAN_SLOTS; otherwise at most every REPLAN_H. (Crane counts per ship
    # are not in the key: the model shares the cranes itself, as the crane agent does.)
    sig = (tuple((s.call_id, s.status, s.berth) for s in state.ships), crane_model.in_service(state),
           len(state.held_ids()), tuple(free[g] // REPLAN_SLOTS for g in GROUPS))
    if _mem.cache and _mem.cache[1] == sig and \
            timedelta(0) <= state.now - _mem.cache[0] < timedelta(hours=REPLAN_H):
        return _mem.cache[2]
    horizon = (state.scenario_end - state.now).total_seconds() / 3600
    cal = _calibration(state)
    ships, reasons, impossible = _ships(state)
    live_ids = [s.call_id for s in ships if s.sail_h is None]
    base_skip = frozenset(impossible)
    fcfs_done, fcfs_end = _simulate(state, base_skip, horizon, cal)
    best_skip, best_done, best_end = base_skip, fcfs_done, fcfs_end
    for cid in live_ids:
        done, end = _simulate(state, base_skip | {cid}, horizon, cal)
        if len(done & set(live_ids)) > len(best_done & set(live_ids)):   # strictly more: ties keep FCFS
            best_skip, best_done, best_end = base_skip | {cid}, done, end
    written_off = tuple(c for c in live_ids if c in best_skip)
    n_best, n_fcfs = len(best_done & set(live_ids)), len(fcfs_done & set(live_ids))
    for c in written_off:
        reasons[c] = (f"Giving {c} no yard slots is predicted to finish {n_best} ships in the window instead of "
                      f"{n_fcfs}: its slots go to ships that can finish.")
    can = tuple(c for c in live_ids if c in best_done)
    cannot = tuple(c for c in live_ids if c not in best_done and c not in best_skip)
    for c in can:
        reasons.setdefault(c, f"{c} is predicted to finish in the window: it gets all the slots it needs.")
    for c in cannot:
        reasons.setdefault(c, f"{c} is not predicted to finish in the window, but writing it off wouldn't "
                              f"finish more ships, so it keeps its first-come-first-served slots.")
    free_now = {g: 0 for g in GROUPS}
    for b in state.blocks:
        free_now[_group_of_block(b.block_type)] += b.free
    shortage = n_fcfs < len(live_ids)
    result = Bound(state.now, shortage, can, cannot, written_off, tuple(impossible), n_best, n_fcfs,
                   free_now, best_end, reasons)
    _mem.cache = (state.now, sig, result)
    return result


def bound(state: WorldState) -> Bound:
    """Forecast and feasibility: which ships can finish, which are written off, and why."""
    return plan(state)


def budgets(state: WorldState) -> dict[str, int]:
    """Import slots each live ship may take now: all its unplaced imports, or 0 if written off/impossible."""
    b = plan(state)
    out = {}
    for s in state.ships:
        if s.status not in LIVE:
            continue
        unplaced = sum(1 for c in state.manifests.get(s.call_id, ())
                       if c.direction == "import" and not c.yard_block and _group_of_cargo(c.cargo_type))
        out[s.call_id] = 0 if (s.call_id in b.written_off or s.call_id in b.impossible) else unplaced
    return out
