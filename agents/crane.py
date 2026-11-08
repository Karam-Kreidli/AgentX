"""Crane agent v1 (CRANE-2): cranes where the work is.

    from agents.crane import CraneAgent
    actions, requests = CraneAgent().propose(state, events)

The simulator has no limit on cranes per ship and moves cranes instantly, so every tick
we can re-split all six quay cranes. What limits a crane is the work it can reach:

  - a ship that is loading (discharge done) can use every crane it gets;
  - a ship that is discharging can only discharge boxes that already have a yard slot,
    plus the boxes the yard agent can still place this tick (legal free room);
  - a ship stalled for yard space (agents/stall.py) or that can never finish makes no
    moves at all, and a crane on it still counts crane-hours: it gets no cranes.

`demand(state, s)` turns that into the number of cranes a ship can keep busy for one
tick (0 to 6). `allocate(state)` then tries every split of the in-service cranes that
keeps the rail order (ships take crane groups in berth order along the rail; idle and
broken cranes stay where they are) and picks, in this order:
  1. no ship that has work is left without a crane;
  2. the most cranes doing useful work;
  3. the fewest cranes on ships that can't use them (those stay idle: no crane-hours);
  4. extra cranes go to the ship with the least work left (shortest job first: it frees
     its berth soonest, which lowers the average turnaround);
  5. the fewest cranes changing ship (so the split doesn't flip between equal plans).
Bays are spread evenly over each berth. Every split is checked with crane_moves.unsafe,
and the moves are planned with safe_crane_moves(): after every single call all cranes,
idle and broken included, stay in rail order and working cranes stay >= 2 bays apart.

It sends one crane Request per ship it wants to serve (amount = cranes planned). Grants
are binding: a ship granted fewer cranes than planned is capped and the split re-planned.
It only proposes; the guard re-checks every step before the executor sends it.
"""
from __future__ import annotations

import math
from functools import lru_cache
from itertools import product
from typing import Mapping, Optional

from agents import stall
from agents.crane_moves import _num, plans_from_state, safe_crane_moves, unsafe
from core.types import (BLOCK_EMPTY, BLOCK_IMDG, BLOCK_REEFER, BLOCK_STANDARD, CARGO_DRY, CARGO_IMDG,
                        CARGO_REEFER, EQ_OUT_OF_SERVICE, RES_CRANES, SHIP_BERTHED, SHIP_WORKING, Action,
                        Event, Request, WorldState)

AGENT = "crane"
RATE = 23.0             # moves per crane-hour: rated 26-28 x 0.85 (simulator model)
TICK_H = 0.25           # one normal tick: demand is the work a crane can reach in it
MAX_PER_SHIP = 6
AT_BERTH = (SHIP_BERTHED, SHIP_WORKING)
IDLE = -1


# --------------------------------------------------------------------------- demand

def _room(state: WorldState) -> dict[str, int]:
    free = {BLOCK_IMDG: 0, BLOCK_REEFER: 0, BLOCK_STANDARD: 0, BLOCK_EMPTY: 0}
    for b in state.blocks:
        if b.block_type in free:
            free[b.block_type] += max(0, b.free)
    return free


def demand(state: WorldState, s, room: Optional[dict] = None, impossible: Optional[set] = None,
           stalled: Optional[set] = None) -> int:
    """Cranes ship `s` can keep busy this tick (0..MAX_PER_SHIP)."""
    if s.status not in AT_BERTH or not s.berth:
        return 0
    impossible = stall.impossible_calls(state) if impossible is None else impossible
    stalled = stall.stalled_calls(state) if stalled is None else stalled
    if s.call_id in stalled:
        return 0
    if s.discharge_remaining <= 0:
        work = s.load_remaining
    else:
        total, placed, unplaced = stall._imports(state, s.call_id)
        at_quay = max(0, placed - stall._discharged(s))
        if s.call_id in impossible:
            reach = 0                                   # gets no yard slots (YARD-3 / coordinator)
        else:
            room = _room(state) if room is None else room
            reach = (min(unplaced[CARGO_IMDG], room[BLOCK_IMDG])
                     + min(unplaced[CARGO_REEFER], room[BLOCK_REEFER])
                     + min(unplaced[CARGO_DRY], room[BLOCK_STANDARD] + room[BLOCK_EMPTY] + room[BLOCK_REEFER]))
        return _cranes_for(s, at_quay + reach, total)
    return _cranes_for(s, work, None)


def with_yard(state: WorldState, s, new_slots: int) -> int:
    """Cranes ship `s` can keep busy this tick if the yard takes `new_slots` more of its boxes.
    Used by the coordinator's joint check (crane + yard grants decided together)."""
    if s.status not in AT_BERTH or not s.berth:
        return 0
    if s.discharge_remaining <= 0:
        return _cranes_for(s, s.load_remaining, None)
    total, placed, _ = stall._imports(state, s.call_id)
    at_quay = max(0, placed - stall._discharged(s))
    return _cranes_for(s, at_quay + max(0, new_slots), total)


def _cranes_for(s, work: int, total: Optional[int]) -> int:
    """Cranes that `work` moves keep busy for one tick (0..MAX_PER_SHIP)."""
    if s.discharge_remaining > 0:
        if total == 0:
            work = s.discharge_remaining                # no manifest rows: can't judge, assume it can work
        if work >= s.discharge_remaining:
            work = s.discharge_remaining + s.load_remaining
    if work <= 0:
        # The schedule rounds remaining moves: a ship showing 0 and 0 that has not sailed still has
        # a fraction of a move left. Without a crane it would never finish, so it keeps one.
        return 1 if s.discharge_remaining <= 0 and s.load_remaining <= 0 else 0
    return min(MAX_PER_SHIP, math.ceil(work / (RATE * TICK_H)))


# --------------------------------------------------------------------------- split

def _bays(n: int, lo: int, hi: int) -> Optional[list[int]]:
    """n bays spread over [lo, hi] (segment centres), strictly increasing and >= 2 apart."""
    if n == 0:
        return []
    if hi - lo < 2 * (n - 1):
        return None
    width = (hi - lo + 1) / n
    out = [int(lo + width * (i + 0.5)) for i in range(n)]
    for i in range(1, n):                               # rounding can squeeze neighbours
        out[i] = max(out[i], out[i - 1] + 2)
    return out if out[-1] <= hi else None


def _place(state: WorldState, ships: list, labels: tuple[int, ...], movable: list[str],
           fixed: Mapping[str, int]) -> Optional[dict[str, dict[str, int]]]:
    """Bays for one labelling (ship index per movable crane, IDLE = stays put), or None."""
    rail = sorted(list(movable) + list(fixed), key=_num)
    lab = dict(zip(movable, labels))
    want: dict[str, dict[str, int]] = {}
    prev_working = -10**6
    for i, s in enumerate(ships):
        group = [c for c in movable if lab[c] == i]
        b = state.berth(s.berth)
        if not group:
            want[s.call_id] = {}
            continue
        first, last = rail.index(group[0]), rail.index(group[-1])
        lo = max(b.start_bay, prev_working + 2)
        hi = b.end_bay
        for c in rail[:first]:                          # idle/broken cranes to the left stay left
            if c in fixed or lab.get(c) == IDLE:
                lo = max(lo, (fixed.get(c) if c in fixed else _pos(state, c)) + 1)
        for c in rail[last + 1:]:                       # ... and those to the right stay right
            if c in fixed or lab.get(c) == IDLE:
                hi = min(hi, (fixed.get(c) if c in fixed else _pos(state, c)) - 1)
        inner = [c for c in rail[first:last + 1] if c not in group]
        if inner:                                       # an idle or broken crane inside the group
            return None
        bays = _bays(len(group), lo, hi)
        if bays is None:
            return None
        want[s.call_id] = dict(zip(group, bays))
        prev_working = bays[-1]
    return want


@lru_cache(maxsize=None)
def _labelings(n_ships: int, n_cranes: int) -> tuple:
    """Every rail-order labelling: ship groups follow the berths along the rail, IDLE anywhere.
    Returns ((labels, cranes per ship), ...)."""
    out = []
    for labels in product([IDLE] + list(range(n_ships)), repeat=n_cranes):
        working = [x for x in labels if x != IDLE]
        if working == sorted(working):
            out.append((labels, tuple(working.count(i) for i in range(n_ships))))
    return tuple(out)


def _pos(state: WorldState, crane_id: str) -> int:
    c = next(c for c in state.cranes if c.crane_id == crane_id)
    return c.bay


def allocate(state: WorldState, caps: Optional[Mapping[str, int]] = None
             ) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """(want_by_call, demand_by_call): the best rail-safe split for this tick.

    want_by_call maps every berthed/working ship to {crane: bay} (empty = no cranes)."""
    ships = sorted((s for s in state.ships if s.status in AT_BERTH and s.berth and state.berth(s.berth)),
                   key=lambda s: state.berth(s.berth).start_bay)
    if not ships:
        return {}, {}
    room, imp, stl = _room(state), stall.impossible_calls(state), stall.stalled_calls(state)
    need = {s.call_id: demand(state, s, room, imp, stl) for s in ships}
    if caps:
        need = {k: min(v, caps.get(k, v)) for k, v in need.items()}
    broken = {c.crane_id for c in state.cranes if c.status == EQ_OUT_OF_SERVICE}
    movable = sorted((c.crane_id for c in state.cranes if c.crane_id not in broken), key=_num)
    fixed = {c.crane_id: c.bay for c in state.cranes if c.crane_id in broken}
    current = {c: s.call_id for s in ships for c in s.cranes}
    left = {s.call_id: s.discharge_remaining + s.load_remaining for s in ships}
    positions, broken_set, _ = plans_from_state(state, {})

    scored = []
    for labels, k in _labelings(len(ships), len(movable)):
        if any(k[i] > need[s.call_id] and k[i] > 0 and need[s.call_id] == 0 for i, s in enumerate(ships)):
            continue                                    # never a crane on a ship that can't use one
        starved = sum(1 for i, s in enumerate(ships) if need[s.call_id] > 0 and k[i] == 0)
        useful = sum(min(k[i], need[s.call_id]) for i, s in enumerate(ships))
        waste = sum(max(0, k[i] - need[s.call_id]) for i, s in enumerate(ships))
        spt = sum(min(k[i], need[s.call_id]) / max(1, left[s.call_id]) for i, s in enumerate(ships))
        churn = sum(1 for c, x in zip(movable, labels)
                    if (ships[x].call_id if x != IDLE else None) != current.get(c))
        scored.append(((starved, -useful, waste, -spt, churn), labels))
    scored.sort(key=lambda x: x[0])

    for _, labels in scored:
        want = _place(state, ships, labels, movable, fixed)
        if want is None:
            continue
        # keep cranes that stay on the same ship where they are, if that is still safe
        for s in ships:
            for c in list(want[s.call_id]):
                if current.get(c) == s.call_id:
                    trial = {k: dict(v) for k, v in want.items()}
                    trial[s.call_id][c] = positions[c]
                    if _final_safe(state, ships, trial, positions, broken_set):
                        want = trial
        if not _final_safe(state, ships, want, positions, broken_set):
            continue
        pos, brk, target = plans_from_state(state, want)
        plan = safe_crane_moves(pos, brk, target)
        if plan.complete:
            return want, need
    return {}, need


def _final_safe(state, ships, want, positions, broken) -> bool:
    _, _, target = plans_from_state(state, want)
    final_pos = {**positions, **{c: b for w in want.values() for c, b in w.items()}}
    final_assign = {k: list(w) for k, w in want.items()}
    for call in target:
        final_assign.setdefault(call, list(target[call].want))
    return unsafe(final_pos, final_assign, broken, target) is None


# --------------------------------------------------------------------------- agent

class CraneAgent:
    """propose(state, events) -> (actions, requests). Never raises."""

    name = AGENT

    def __init__(self) -> None:
        self.escalations: list[str] = []      # drained and logged by the loop
        self.grants: Optional[dict] = None    # set by the loop from the coordinator's grants (CORE-5)
        self._memo: dict = {}

    def propose(self, state: WorldState, events: list[Event]) -> tuple[list[Action], list[Request]]:
        try:
            return self._propose(state)
        except Exception as e:                # never crash the run; the loop logs the escalation
            self.escalations.append(f"Crane agent skipped this round ({type(e).__name__}); cranes stay as they are.")
            return [], []

    def _allocate(self, state: WorldState, caps=None):
        """allocate(), remembered for the same snapshot (the loop asks twice per round)."""
        key = (state.now, tuple((s.call_id, s.status, s.berth, s.cranes, s.discharge_remaining, s.load_remaining)
                                for s in state.ships),
               tuple((c.crane_id, c.status, c.bay, c.call_id) for c in state.cranes),
               tuple((b.block_id, b.used) for b in state.blocks),
               tuple(sorted(caps.items())) if caps else None)
        if key not in self._memo:
            self._memo = {key: allocate(state, caps)} if len(self._memo) > 8 else {**self._memo, key: allocate(state, caps)}
        want, need = self._memo[key]
        return {k: dict(v) for k, v in want.items()}, dict(need)

    def _propose(self, state: WorldState) -> tuple[list[Action], list[Request]]:
        asked, need = self._allocate(state)   # what we would do with no limits: that is what we ask for
        requests = []
        for s in state.ships:
            if asked.get(s.call_id):
                n = len(asked[s.call_id])
                requests.append(Request(AGENT, RES_CRANES, n, s.call_id,
                                        1.0 if s.status == SHIP_WORKING else 0.5,
                                        f"{s.call_id} can keep {need.get(s.call_id, n)} cranes busy this tick "
                                        f"({s.discharge_remaining} to discharge, {s.load_remaining} to load)."))
        want = asked
        if self.grants is not None:           # binding: a ship gets at most what it was granted
            caps = {s.call_id: self.grants.get((RES_CRANES, s.call_id), 0) for s in state.ships}
            if any(len(v) > caps.get(k, 0) for k, v in asked.items()):
                want, need = self._allocate(state, caps)
        if not want:
            return [], requests
        pos, brk, target = plans_from_state(state, want)
        plan = safe_crane_moves(pos, brk, target)
        actions = [Action(a.tool, a.args, AGENT, self._why(state, a, need)) for a in plan]
        return actions, requests

    @staticmethod
    def _why(state: WorldState, a: Action, need: Mapping[str, int]) -> str:
        cid = a.args.get("call_id")
        s = state.ship(cid)
        n = len(a.args.get("crane_ids") or ())
        if s is None:
            return a.reason
        if n == 0:
            what = ("it is waiting for yard space, so a crane on it would make no moves"
                    if need.get(cid, 0) == 0 else "its cranes go where there is more work")
            return f"{cid} releases its cranes: {what}. " + a.reason
        return (f"{cid} gets {n} crane{'s' if n != 1 else ''} (it can keep {need.get(cid, n)} busy; "
                f"{s.discharge_remaining} to discharge, {s.load_remaining} to load). " + a.reason)
