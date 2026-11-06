"""Berth agent v0 (BERTH-1): every arrived ship gets a legal berth quickly.

    from agents.berth import BerthAgent
    actions, requests = BerthAgent().propose(state, events)

Proposes `assign_berth` for ships that have arrived, oldest arrival first, on the
first free berth they fit (berth order B1, B2, B3). Fit rule: berth length >=
LOA + 20 m and berth max draft >= ship draft. Berth sizes come from the static
reference data in the WorldState (`state.berths`), never hard-coded per scenario.

A berth is busy while a ship on it is berthed, working or sailing (the simulator
frees it only when the ship has departed). Ships that fit no berth, or have no
valid length or draft, are escalated once and left waiting.

It only proposes; the loop sends each action through the guard and the executor.
BERTH-2 adds priorities (keeping B1 for big ships, shorter waits).
"""
from __future__ import annotations

from core.types import (SHIP_ARRIVED, SHIP_BERTHED, SHIP_SAILING, SHIP_WORKING, TOOL_ASSIGN_BERTH,
                        Action, Event, Request, WorldState)

AGENT = "berth"
BUSY = (SHIP_BERTHED, SHIP_WORKING, SHIP_SAILING)


class BerthAgent:
    name = AGENT

    def __init__(self) -> None:
        self.escalations: list[str] = []      # drained and logged by the loop
        self._flagged: set[str] = set()

    def _escalate(self, key: str, message: str) -> None:
        if key not in self._flagged:
            self._flagged.add(key)
            self.escalations.append(message)

    def propose(self, state: WorldState, events: list[Event]) -> tuple[list[Action], list[Request]]:
        actions: list[Action] = []
        occupied = {s.berth for s in state.ships if s.berth and s.status in BUSY}
        waiting = sorted((s for s in state.ships if s.status == SHIP_ARRIVED),
                         key=lambda s: (s.actual_arrival is None, s.actual_arrival, s.call_id))
        for s in waiting:
            if s.loa_m is None or s.draft_m is None or s.loa_m <= 0 or s.draft_m <= 0:
                self._escalate(f"size:{s.call_id}",
                               f"{s.call_id} has no valid length or draft; it is not berthed.")
                continue
            fits = [b for b in state.berths if b.fits(s.loa_m, s.draft_m)]
            if not fits:
                self._escalate(f"nofit:{s.call_id}",
                               f"{s.call_id} (LOA {s.loa_m} m, draft {s.draft_m} m) fits no berth; it will wait.")
                continue
            free = [b for b in fits if b.berth_id not in occupied]
            if not free:
                continue                      # waits for one of the berths it fits
            b = free[0]
            occupied.add(b.berth_id)
            others = [x.berth_id for x in fits if x.berth_id != b.berth_id]
            actions.append(Action(
                TOOL_ASSIGN_BERTH, {"call_id": s.call_id, "berth_id": b.berth_id}, AGENT,
                f"{s.call_id} has arrived; {b.berth_id} is free and fits it "
                f"({b.length_m:g} m for LOA {s.loa_m:g} m + 20 m, depth {b.max_draft_m:g} m for draft {s.draft_m:g} m)"
                + (f"; it also fits {', '.join(others)}." if others else "; it fits no other berth.")))
        return actions, []
