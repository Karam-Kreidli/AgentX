"""Berth agent v1 (BERTH-2): shorter waits and more departures than first-free-berth.

    from agents.berth import BerthAgent
    actions, requests = BerthAgent().propose(state, events)

Every tick, for the ships that have arrived and fit some berth:

1. Order (who goes first when berths are scarce). Ships the yard forecast says can never finish
   (YARD-3 `impossible`, e.g. more discharge than imports) go last; then shortest remaining crane
   work first (moves left), then earliest arrival. Shortest-work-first lowers the average wait.
2. Best fit (which berth). Each ship takes the SMALLEST free berth it fits, so the long, deep
   berths stay free for the ships that only fit there. On practice_02 first-free-berth put a
   220 m ship on B1, and the 290 m ship that arrived 1.5 h later (it fits only B1) waited 24 h.
3. Operator directives (LLM-6, checked by llm/operator.py before they get here): prioritised
   ships go first (after impossible ones), held ships are not berthed, a reserved berth goes
   only to its ship (or to nobody while it is kept free).
4. Keep a berth for a ship that only fits there, when that ship is due soon: a free berth is not
   given to a ship that fits another berth if a ship that fits ONLY this berth has its scheduled
   ETA within HOLD_H hours (scheduled ETA from the schedule; the actual arrival is only used once
   a ship has arrived, never the hidden one).

Negotiation (CORE-5): for each ship it wants to berth now it sends `Request("berth", 1, call_id,
priority)` and berths only what the coordinator grants (`grants`; None = no coordinator, all
granted). A waiting ship with no free berth that fits it sends a request too, so the transcript
shows why it waits.

Fit rule: berth length >= LOA + 20 m and berth max draft >= ship draft, from the static berth data
in the WorldState (`state.berths`). A berth is busy while a ship on it is berthed, working or
sailing. Ships with no valid size, or that fit no berth, are escalated once and left waiting.
It only proposes; the loop sends each action through the guard and the executor.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

from core.types import (RES_BERTH, SHIP_ARRIVED, SHIP_BERTHED, SHIP_EXPECTED, SHIP_SAILING, SHIP_WORKING,
                        TOOL_ASSIGN_BERTH, Action, Event, Request, WorldState)

AGENT = "berth"
BUSY = (SHIP_BERTHED, SHIP_WORKING, SHIP_SAILING)
HOLD_H = 2.0          # keep a berth this long ahead of a ship that fits only that berth


def _impossible(state: WorldState) -> set:
    """Ships YARD-3 says can never finish (general check), or empty if the forecast fails."""
    try:
        from agents import admission
        return set(admission.bound(state).impossible)
    except Exception:
        return set()


def _operator(state: WorldState):
    """LLM-6 directives (prioritised / held ships, reserved berths), or None. Never raises."""
    try:
        from llm import operator
        return operator.directives(state)
    except Exception:
        return None


def rank(state: WorldState, ships: list) -> list:
    """Who goes first: possible ships before impossible ones, then ships the operator
    prioritised (LLM-6), then least work, then arrival."""
    hopeless = _impossible(state)
    d = _operator(state)
    prio = {c: i for i, c in enumerate(d.priority)} if d is not None else {}
    return sorted(ships, key=lambda s: (s.call_id in hopeless, prio.get(s.call_id, len(prio)), s.remaining_moves,
                                        s.actual_arrival or state.now, s.call_id))


def plan(state: WorldState) -> list[tuple]:
    """[(ship, berth, why)] the agent would berth now, in order. Shared with BC-3's option 0."""
    occupied = {s.berth for s in state.ships if s.berth and s.status in BUSY}
    free = sorted((b for b in state.berths if b.berth_id not in occupied), key=lambda b: (b.length_m, b.max_draft_m))
    d = _operator(state)
    on_hold = set(d.held) if d is not None else set()
    reserved = dict(d.reserved) if d is not None else {}          # berth -> call id, or None = keep free
    waiting = [s for s in state.ships if s.status == SHIP_ARRIVED and _valid(s) and s.call_id not in on_hold]
    soon = state.now + timedelta(hours=HOLD_H)
    held = set()
    for e in state.ships:
        if e.status == SHIP_EXPECTED and _valid(e) and e.scheduled_eta and e.scheduled_eta <= soon:
            only = [b.berth_id for b in state.berths if b.fits(e.loa_m, e.draft_m)]
            if len(only) == 1:
                held.add(only[0])
    out = []
    for s in rank(state, waiting):
        fits = [b for b in free if b.fits(s.loa_m, s.draft_m)]
        n_fit = sum(1 for b in state.berths if b.fits(s.loa_m, s.draft_m))
        own = [b for b in fits if reserved.get(b.berth_id, "-") == s.call_id]     # reserved for this ship
        open_ = [b for b in fits if b.berth_id not in reserved and (b.berth_id not in held or n_fit == 1)]
        choice = (own or open_ or [None])[0]
        if choice is None:
            continue
        free.remove(choice)
        why = "the smallest free berth it fits" + ("" if n_fit > 1 else "; it fits no other berth")
        if own:
            why = "the operator reserved it for this ship"
        elif d is not None and s.call_id in d.priority:
            why += "; the operator asked to prioritise it"
        out.append((s, choice, why))
    return out


def _valid(s) -> bool:
    return s.loa_m is not None and s.draft_m is not None and s.loa_m > 0 and s.draft_m > 0


class BerthAgent:
    name = AGENT

    def __init__(self) -> None:
        self.escalations: list[str] = []      # drained and logged by the loop
        self._flagged: set[str] = set()
        self.grants: Optional[dict] = None    # set by the loop from the coordinator's grants (CORE-5)

    def _escalate(self, key: str, message: str) -> None:
        if key not in self._flagged:
            self._flagged.add(key)
            self.escalations.append(message)

    def propose(self, state: WorldState, events: list[Event]) -> tuple[list[Action], list[Request]]:
        actions: list[Action] = []
        requests: list[Request] = []
        for s in state.ships:
            if s.status != SHIP_ARRIVED:
                continue
            if not _valid(s):
                self._escalate(f"size:{s.call_id}", f"{s.call_id} has no valid length or draft; it is not berthed.")
            elif not any(b.fits(s.loa_m, s.draft_m) for b in state.berths):
                self._escalate(f"nofit:{s.call_id}",
                               f"{s.call_id} (LOA {s.loa_m} m, draft {s.draft_m} m) fits no berth; it will wait.")
        planned = plan(state)
        chosen = {s.call_id for s, _, _ in planned}
        for i, (s, b, why) in enumerate(planned):
            requests.append(Request(AGENT, RES_BERTH, 1, s.call_id, round(1.0 - i * 0.01, 2),
                                    f"{s.call_id} has arrived and {b.berth_id} is free and fits it."))
            if self.grants is not None and self.grants.get((RES_BERTH, s.call_id), 0) < 1:
                continue
            waited = (state.now - s.actual_arrival).total_seconds() / 3600 if s.actual_arrival else 0.0
            actions.append(Action(
                TOOL_ASSIGN_BERTH, {"call_id": s.call_id, "berth_id": b.berth_id}, AGENT,
                f"{s.call_id} has waited {waited:.1f} h with {s.remaining_moves} moves to do; {b.berth_id} is "
                f"{why} ({b.length_m:g} m for LOA {s.loa_m:g} m + 20 m, depth {b.max_draft_m:g} m for "
                f"draft {s.draft_m:g} m)."))
        for s in rank(state, [x for x in state.ships if x.status == SHIP_ARRIVED and _valid(x)]):
            if s.call_id not in chosen and any(b.fits(s.loa_m, s.draft_m) for b in state.berths):
                requests.append(Request(AGENT, RES_BERTH, 1, s.call_id, 0.1,
                                        f"{s.call_id} has arrived but no free berth fits it yet."))
        return actions, requests
