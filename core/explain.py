"""Explaining the run (review follow-up, Sun 8 Nov).

    from core.explain import Explainer
    ex = Explainer()
    ex.observe(state)                    # start of every tick: hours spent per ship and cause
    ex.replan(before, after)             # end of every tick: one feed sentence if the plan changed
    why = ex.unfinished(final_state)     # end of run: call_id -> one sentence

1. One sentence per re-plan (stretch goal 1). Whenever a tick changes the plan - a ship is
   berthed, or the cranes are re-split between ships - one plain sentence goes to
   supervisor_feed.txt, built from what changed (no LLM): "Re-plan: P02-003 berthed on B2;
   cranes now P02-001 4, P02-003 2; P02-002 has none while it waits for yard space." The
   supervisor's own option picks (LLM-4) still write their sentence too.

2. Why a ship did not finish. For every ship still at the terminal at the end, the hours it
   spent are split by what held it up, and the biggest one is named:
     - its manifest can never cover its discharge (impossible, YARD-4);
     - it had not arrived yet;
     - waiting for a berth (every berth it fits was busy; or it fits only one);
     - at berth but stopped for yard space (stalled, YARD-4);
     - at berth with no crane while it could have worked;
     - working with fewer than 3 trucks per crane;
     - or it was working all along and simply ran out of time.
   The sentences go to unfinished.md and summary.json ("unfinished_why").

Read-only: it never changes the plan. Never raises.
"""
from __future__ import annotations

from typing import Optional

from agents import stall
from core.state import hours
from core.types import (SHIP_ARRIVED, SHIP_BERTHED, SHIP_DEPARTED, SHIP_EXPECTED, SHIP_SAILING, SHIP_WORKING,
                        WorldState)

AT_BERTH = (SHIP_BERTHED, SHIP_WORKING)
RATE = 23.0                  # moves per crane-hour (same as the crane agent)
CAUSES = ("berth", "yard", "no_crane", "trucks", "working")


class Explainer:
    def __init__(self) -> None:
        self.hours: dict[str, dict[str, float]] = {}
        self.last_t: Optional[float] = None
        self.last: Optional[WorldState] = None
        self.blocks: dict[str, set] = {}        # YARD-6: every block that has held a ship's import boxes

    # ------------------------------------------------------------------ during the run
    def observe(self, state: WorldState) -> None:
        """Add the time since the last tick to each ship's cause, as seen at the last tick."""
        try:
            t = hours(state)
            if self.last is not None and self.last_t is not None and t > self.last_t:
                dt = t - self.last_t
                stalled = stall.stalled_calls(self.last)
                for s in self.last.ships:
                    cause = _cause(s, stalled)
                    if cause:
                        h = self.hours.setdefault(s.call_id, {c: 0.0 for c in CAUSES})
                        h[cause] += dt
            self.last, self.last_t = state, t
            for s in state.ships:                # boxes are only placed while their ship is at a berth
                if s.status in AT_BERTH:
                    seen = self.blocks.setdefault(s.call_id, set())
                    seen.update(c.yard_block for c in state.manifests.get(s.call_id, ())
                                if c.direction == "import" and c.yard_block)
        except Exception:
            pass

    def spread(self) -> dict[str, int]:
        """YARD-6: in how many yard blocks each ship's import boxes were placed (ships with any)."""
        return {cid: len(b) for cid, b in sorted(self.blocks.items()) if b}

    def replan(self, before: WorldState, after: WorldState) -> Optional[str]:
        """One sentence if this tick berthed a ship or re-split the cranes, else None."""
        try:
            prev = {s.call_id: s for s in before.ships}
            parts = []
            for s in after.ships:
                p = prev.get(s.call_id)
                if p is not None and s.berth and not p.berth and s.status in AT_BERTH:
                    waited = (after.now - s.actual_arrival).total_seconds() / 3600 if s.actual_arrival else None
                    parts.append(f"{s.call_id} berthed on {s.berth}"
                                 + (f" after waiting {waited:.1f} h" if waited and waited >= 0.25 else ""))
            old = {s.call_id: len(s.cranes) for s in before.ships if s.status in AT_BERTH}
            new = {s.call_id: len(s.cranes) for s in after.ships if s.status in AT_BERTH}
            if any(old.get(k, 0) != n for k, n in new.items()) or any(new.get(k, 0) != n for k, n in old.items()):
                with_cranes = [f"{k} {n}" for k, n in new.items() if n]
                if with_cranes:
                    parts.append("cranes now " + ", ".join(with_cranes))
                stalled = stall.stalled_calls(after)
                idle = [k for k, n in new.items() if not n]
                for k in idle:
                    s = after.ship(k)
                    if k in stalled:
                        parts.append(f"{k} has none while it waits for yard space")
                    elif s is not None and s.status == SHIP_BERTHED:
                        parts.append(f"{k} has none yet (still mooring)")
                    else:
                        parts.append(f"{k} has none this tick")
            if not parts:
                return None
            return "Re-plan: " + "; ".join(parts) + "."
        except Exception:
            return None

    # ------------------------------------------------------------------ at the end
    def unfinished(self, state: WorldState) -> dict[str, str]:
        """call_id -> one sentence for every ship that did not depart."""
        out: dict[str, str] = {}
        try:
            impossible = stall.impossible_calls(state)
            for s in state.ships:
                if s.status == SHIP_DEPARTED:
                    continue
                out[s.call_id] = _why(state, s, impossible, self.hours.get(s.call_id))
        except Exception as e:
            out["?"] = f"Could not explain the unfinished ships ({type(e).__name__})."
        return out


def _on_hold(state: WorldState) -> set:
    """Ships the operator put on hold (LLM-6), so the reason names the hold, not the berths. Never raises."""
    try:
        from agents.berth import _operator
        d = _operator(state)
        return set(d.held) if d is not None else set()
    except Exception:
        return set()


def _cause(s, stalled: set) -> Optional[str]:
    if s.status == SHIP_ARRIVED:
        return "berth"
    if s.status not in AT_BERTH:
        return None
    if s.call_id in stalled:
        return "yard"
    if not s.cranes:
        return "no_crane" if s.status == SHIP_WORKING else None
    if s.trucks < 3 * len(s.cranes):
        return "trucks"
    return "working"


def _why(state: WorldState, s, impossible: set, h: Optional[dict]) -> str:
    cid, left = s.call_id, s.discharge_remaining + s.load_remaining
    if cid in impossible:
        total, placed, _ = stall._imports(state, cid)
        return (f"{cid} can never finish: it must discharge {s.discharge_remaining} more boxes but its manifest "
                f"has only {max(0, total - stall._discharged(s))} import boxes left.")
    if s.status == SHIP_EXPECTED:
        return f"{cid} had not arrived by the end of the window."
    if s.status == SHIP_SAILING:
        return f"{cid} finished its moves and was sailing at the end."
    h = h or {c: 0.0 for c in CAUSES}
    if s.status == SHIP_ARRIVED:
        if cid in _on_hold(state):
            return f"{cid} waited {h['berth']:.1f} h at anchor and was not berthed: the operator put it on hold."
        fits = [b.berth_id for b in state.berths if s.loa_m and s.draft_m and b.fits(s.loa_m, s.draft_m)]
        where = (f"it fits only {fits[0]}" if len(fits) == 1 else
                 "it fits no berth" if not fits else f"every berth it fits ({', '.join(fits)}) was busy")
        return f"{cid} waited {h['berth']:.1f} h for a berth and never got one: {where}."
    at = sum(h[c] for c in ("yard", "no_crane", "trucks", "working"))
    held = {"yard": "stopped for yard space", "no_crane": "without a crane", "trucks": "short of trucks"}
    main = max(("yard", "no_crane", "trucks"), key=lambda c: h[c])
    detail = ", ".join(f"{h[c]:.1f} h {held[c]}" for c in ("yard", "no_crane", "trucks") if h[c] >= 0.25)
    head = f"{cid} spent {h['berth']:.1f} h waiting for a berth and {at:.1f} h at berth" if h["berth"] >= 0.25 else \
        f"{cid} spent {at:.1f} h at berth"
    if h[main] >= 0.25 and h[main] >= 0.2 * max(at, 0.01):
        _, _, unplaced = stall._imports(state, cid)
        tail = (f"; {sum(unplaced.values())} of its boxes still had no yard slot at the end"
                if main == "yard" and sum(unplaced.values()) else "")
        return f"{head} ({detail}){tail}; {left} moves were left. Main cause: {held[main]}."
    need = left / (RATE * 6)
    return (f"{head}, working nearly all of it ({detail or 'no delays'}); it ran out of time with {left} moves "
            f"left (about {need:.1f} h more even with all six cranes). Main cause: time.")
