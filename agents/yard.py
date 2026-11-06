"""Yard agent v0 (YARD-1): every import box goes to a legal block.

    from agents.yard import YardAgent
    actions, requests = YardAgent().propose(state, events)

For each berthed or working ship that still has boxes to discharge, its import
boxes without a yard slot are placed:
  - IMDG   -> block H only (the IMDG block)
  - reefer -> blocks I or J only (the reefer blocks), nearest first
  - dry    -> the nearest standard block while it stays at or under 85% full,
              then any legal block: standard blocks over 85%, then G (meant for
              empties), then I/J last, so reefers keep their only blocks as long as possible.
"Nearest" is the truck travel time from the ship's berth to the block (static
reference data from the terminal, the same in every scenario).

A box with nowhere legal to go is never forced: it stays queued and is tried
again next tick (the ship cannot discharge it until then), and the ship is
escalated once. A box with an unknown cargo type is never placed and is
escalated once.

It only proposes; the loop sends each action through the guard and the executor.
"""
from __future__ import annotations

import math

from core.types import (BLOCK_EMPTY, BLOCK_IMDG, BLOCK_REEFER, BLOCK_STANDARD, CARGO_DRY, CARGO_IMDG,
                        CARGO_REEFER, KNOWN_CARGO, SHIP_BERTHED, SHIP_WORKING, TOOL_PLACE, Action,
                        Event, Request, WorldState)

AGENT = "yard"
SOFT_FILL = 0.85            # keep standard blocks at or under 85% while other room exists

# Truck travel time in minutes, berth -> block. Static reference data (terminal layout).
TRAVEL_MIN: dict[str, dict[str, float]] = {
    "B1": {"A": 2.3, "B": 2.6, "C": 2.6, "D": 3.1, "E": 3.1, "F": 2.8, "G": 2.9, "H": 3.1, "I": 3.4, "J": 3.6},
    "B2": {"A": 2.4, "B": 2.7, "C": 3.2, "D": 3.3, "E": 3.3, "F": 3.1, "G": 3.3, "H": 3.4, "I": 3.5, "J": 3.9},
    "B3": {"A": 2.7, "B": 3.1, "C": 3.2, "D": 3.5, "E": 3.9, "F": 3.5, "G": 3.4, "H": 3.6, "I": 4.1, "J": 4.2},
}


def _travel(berth: str | None, block: str) -> float:
    return TRAVEL_MIN.get(berth or "", {}).get(block, 99.0)


class YardAgent:
    name = AGENT

    def __init__(self) -> None:
        self.escalations: list[str] = []      # drained and logged by the loop
        self._flagged: set[str] = set()

    def _escalate(self, key: str, message: str) -> None:
        if key not in self._flagged:
            self._flagged.add(key)
            self.escalations.append(message)

    # ------------------------------------------------------------------ plan
    def _targets(self, cargo: str, berth: str | None, blocks: dict, free: dict) -> list[tuple[str, int]]:
        """(block, how many may go there now) in order of preference for one cargo type."""
        near = lambda b: (_travel(berth, b), b)                                  # noqa: E731
        of = lambda kind: sorted((b for b, blk in blocks.items() if blk.block_type == kind), key=near)  # noqa: E731
        if cargo == CARGO_IMDG:
            return [(b, free[b]) for b in of(BLOCK_IMDG)]
        if cargo == CARGO_REEFER:
            return [(b, free[b]) for b in of(BLOCK_REEFER)]
        # dry: standard under 85%, then standard over 85%, then G, then I/J
        out = []
        for b in of(BLOCK_STANDARD):
            soft = math.floor(SOFT_FILL * blocks[b].slots) - (blocks[b].slots - free[b])
            out.append((b, max(0, min(free[b], soft))))
        out += [(b, free[b]) for b in of(BLOCK_STANDARD)]
        out += [(b, free[b]) for b in of(BLOCK_EMPTY)]
        out += [(b, free[b]) for b in of(BLOCK_REEFER)]
        return out

    def propose(self, state: WorldState, events: list[Event]) -> tuple[list[Action], list[Request]]:
        actions: list[Action] = []
        blocks = {b.block_id: b for b in state.blocks}
        free = {b.block_id: b.free for b in state.blocks}
        for s in state.ships:
            if s.status not in (SHIP_BERTHED, SHIP_WORKING) or s.discharge_remaining <= 0:
                continue
            todo: dict[str, list[str]] = {}
            for c in state.manifests.get(s.call_id, ()):
                if c.direction != "import" or c.yard_block:
                    continue
                if c.cargo_type not in KNOWN_CARGO:
                    self._escalate(f"cargo:{c.container_id}",
                                   f"An import box of {s.call_id} has unknown cargo type "
                                   f"{str(c.cargo_type)[:20]!r}; it is not placed.")
                    continue
                todo.setdefault(c.cargo_type, []).append(c.container_id)

            queued = 0
            for cargo in (CARGO_IMDG, CARGO_REEFER, CARGO_DRY):
                boxes = todo.get(cargo, [])
                for block_id, room in self._targets(cargo, s.berth, blocks, free):
                    if not boxes:
                        break
                    n = min(room, free[block_id], len(boxes))
                    if n <= 0:
                        continue
                    batch, boxes = boxes[:n], boxes[n:]
                    blk = blocks[block_id]
                    after = blk.slots - free[block_id] + n
                    actions.append(Action(
                        TOOL_PLACE, {"container_ids": batch, "block_id": block_id}, AGENT,
                        f"{n} {cargo} import boxes of {s.call_id} need a slot before discharge; "
                        f"{block_id} is the nearest legal {blk.block_type} block with room "
                        f"({_travel(s.berth, block_id):g} min from {s.berth}, {after}/{blk.slots} after)."))
                    free[block_id] -= n
                queued += len(boxes)
            if queued:
                self._escalate(f"queue:{s.call_id}",
                               f"{queued} import boxes of {s.call_id} have no legal yard slot; they wait "
                               f"for space and the ship cannot discharge them until then.")
        return actions, []
