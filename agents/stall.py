"""Stalled and impossible ship detector (YARD-4).

    from agents import stall
    events = stall.detect(state)         # called every tick by core/events.py (CORE-3 hook)
    stall.stalled_calls(state)           # ships stalled right now (for CRANE-2)
    stall.impossible_calls(state)        # ships that can never finish

Impossible: a ship at the terminal (expected to working) must discharge more boxes than its
manifest still has: remaining discharge moves > import boxes in the manifest minus those it has
discharged so far (a ship with no manifest rows is not judged: that is missing data, YARD-5).
A general check on the tool data, no IDs: e.g. a ship that must discharge
3,100 boxes with 551 imports in its manifest. Discharged so far = first discharge count seen
minus the current one (if the ship was first seen mid-discharge, boxes discharged before that
still count as left, so the check can only be late, never wrong). Reported once per ship.

Stalled: a berthed or working ship with discharge left, not impossible, that has nothing to
discharge now (every import box placed in the yard has been discharged) and whose next boxes
have no legal free slot (IMDG -> H, reefer -> I/J, dry -> standard, G, then I/J). It waits for
yard space; its cranes would be better used elsewhere (CRANE-2). An event when a ship becomes
stalled; `stalled_calls()` gives the current set.

Events (core/types): ship_impossible (once per ship, detail.message says why; the loop
escalates it) and ship_stalled (each time a ship becomes stalled). Never raises: a failure
returns no events (core/events.py also guards the call). Escalations never block the run.
"""
from __future__ import annotations

from core.state import hours
from core.types import (BLOCK_EMPTY, BLOCK_IMDG, BLOCK_REEFER, BLOCK_STANDARD, CARGO_DRY, CARGO_IMDG,
                        CARGO_REEFER, EV_SHIP_IMPOSSIBLE, EV_SHIP_STALLED, KNOWN_CARGO, SHIP_ARRIVED,
                        SHIP_BERTHED, SHIP_EXPECTED, SHIP_WORKING, Event, WorldState)

AT_TERMINAL = (SHIP_EXPECTED, SHIP_ARRIVED, SHIP_BERTHED, SHIP_WORKING)
AT_BERTH = (SHIP_BERTHED, SHIP_WORKING)


class _Memory:
    def __init__(self) -> None:
        self.scenario = None
        self.first_discharge: dict[str, int] = {}     # discharge moves when first seen
        self.reported: set[str] = set()               # impossible ships already reported
        self.stalled: set[str] = set()                # stalled at the last check


_mem = _Memory()


def reset() -> None:
    global _mem
    _mem = _Memory()


def _scenario(state: WorldState) -> None:
    key = (state.scenario_end, tuple(sorted(s.call_id for s in state.ships)))
    if _mem.scenario != key:
        reset()
        _mem.scenario = key
    for s in state.ships:                     # discharged since first seen (all of it if seen before working)
        _mem.first_discharge.setdefault(s.call_id, s.discharge_remaining)


def _imports(state: WorldState, call_id: str) -> tuple[int, int, dict]:
    """(import boxes in the manifest, placed in the yard, unplaced per known cargo type)."""
    total = placed = 0
    unplaced = {CARGO_IMDG: 0, CARGO_REEFER: 0, CARGO_DRY: 0}
    for c in state.manifests.get(call_id, ()):
        if c.direction != "import":
            continue
        total += 1
        if c.yard_block:
            placed += 1
        elif c.cargo_type in KNOWN_CARGO:
            unplaced[c.cargo_type] += 1
    return total, placed, unplaced


def _discharged(s) -> int:
    return max(0, _mem.first_discharge.get(s.call_id, s.discharge_remaining) - s.discharge_remaining)


def _impossible_reason(state: WorldState, s):
    if s.status not in AT_TERMINAL or s.discharge_remaining <= 0:
        return None
    total, _, _ = _imports(state, s.call_id)
    if total == 0:
        return None                                    # no manifest rows: unknown, not impossible
    left = max(0, total - _discharged(s))
    if s.discharge_remaining > left:
        return (f"{s.call_id} must discharge {s.discharge_remaining} more boxes but its manifest has only "
                f"{left} import boxes left, so it can never finish.")
    return None


def _room(state: WorldState) -> dict:
    free = {BLOCK_IMDG: 0, BLOCK_REEFER: 0, BLOCK_STANDARD: 0, BLOCK_EMPTY: 0}
    for b in state.blocks:
        if b.block_type in free:
            free[b.block_type] += max(0, b.free)
    return free


def _no_room_for(unplaced: dict, free: dict) -> bool:
    """True if none of the waiting boxes has a legal free slot."""
    if unplaced[CARGO_IMDG] and free[BLOCK_IMDG]:
        return False
    if unplaced[CARGO_REEFER] and free[BLOCK_REEFER]:
        return False
    if unplaced[CARGO_DRY] and (free[BLOCK_STANDARD] or free[BLOCK_EMPTY] or free[BLOCK_REEFER]):
        return False
    return True


def impossible_calls(state: WorldState) -> set[str]:
    """Ships that can never finish (general check). Never raises."""
    try:
        _scenario(state)
        return {s.call_id for s in state.ships if _impossible_reason(state, s)}
    except Exception:
        return set()


def stalled_calls(state: WorldState) -> set[str]:
    """Berthed/working ships waiting for yard space right now (not impossible ones). Never raises."""
    try:
        _scenario(state)
        free = _room(state)
        out = set()
        for s in state.ships:
            if s.status not in AT_BERTH or s.discharge_remaining <= 0 or _impossible_reason(state, s):
                continue
            total, placed, unplaced = _imports(state, s.call_id)
            waiting_at_quay = placed - _discharged(s)
            if waiting_at_quay <= 0 and sum(unplaced.values()) > 0 and _no_room_for(unplaced, free):
                out.add(s.call_id)
        return out
    except Exception:
        return set()


def detect(state: WorldState) -> list[Event]:
    """New ship_impossible (once per ship) and ship_stalled (on becoming stalled) events."""
    try:
        _scenario(state)
        t = hours(state)
        events = []
        for s in state.ships:
            if s.call_id in _mem.reported:
                continue
            why = _impossible_reason(state, s)
            if why:
                _mem.reported.add(s.call_id)
                events.append(Event(EV_SHIP_IMPOSSIBLE, s.call_id, t, {
                    "message": why, "discharge_remaining": s.discharge_remaining}))
        now = stalled_calls(state)
        for cid in sorted(now - _mem.stalled):
            _, _, unplaced = _imports(state, cid)
            events.append(Event(EV_SHIP_STALLED, cid, t, {
                "message": f"{cid} has nothing left to discharge and no legal yard slot for its next "
                           f"{sum(unplaced.values())} import boxes; it waits for the yard to release some.",
                "imports_without_slot": sum(unplaced.values())}))
        _mem.stalled = now
        return events
    except Exception:
        return []
