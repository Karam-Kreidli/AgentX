"""Walking skeleton (CORE-2), now the rules-only agent used by the main loop (CORE-4).

A deliberately simple, rule-safe planner. From one WorldState it proposes:
  1. berth arrived ships (oldest arrival first) on the first free berth they fit (B1, B2, B3)
  2. give each berthed ship its berth's fixed cranes, skipping broken ones
  3. give it 3 trucks per crane
  4. place its import boxes in legal blocks (IMDG -> H, reefer -> I/J, dry -> A-F, then G, then I/J)

It only *proposes* Actions; core/loop.py sends them through the guard and the
executor (the only code that changes the simulator). It is replaced piece by
piece by the real agents and stays as the rules-only fallback. It reads only
the WorldState (toolkit read tools + static reference data): no workbook, no
future information.

Why the fixed crane split can never break rail order or spacing
----------------------------------------------------------------
Each crane only ever sits at its home bay (where the simulator starts it) or at
its fixed working bay below. For every pair of rail neighbours, every possible
position of the left crane is below every possible position of the right one:

    crane   home  work   berth (bays)
    QC1      1     5     B1 (1-30)
    QC2     14    20     B1
    QC3     27    35     B2 (31-58)
    QC4     40    50     B2
    QC5     53    62     B3 (59-80)
    QC6     66    75     B3

So whatever order the moves happen in, and whichever cranes are broken (a broken
crane is skipped and simply stays where it is), all six stay in strict rail
order. Working positions are at least 15 bays apart, so spacing always holds.
"""
from __future__ import annotations

from core.types import (BLOCK_EMPTY, BLOCK_IMDG, BLOCK_REEFER, BLOCK_STANDARD, CARGO_DRY,
                        CARGO_IMDG, CARGO_REEFER, EQ_AVAILABLE, EQ_OUT_OF_SERVICE, SHIP_ARRIVED,
                        SHIP_BERTHED, SHIP_SAILING, SHIP_WORKING, TOOL_ASSIGN_BERTH,
                        TOOL_DISPATCH_TRUCKS, TOOL_PLACE, TOOL_PLAN_CRANES, Action, Berth, Event,
                        Request, WorldState)

AGENT = "skeleton"
TICK_MINUTES = 15
TRUCKS_PER_CRANE = 3          # the simulator reaches full crane speed at 3 trucks per crane

# Static reference data (terminal facts, the same in every scenario; never scenario sheets).
BERTHS: tuple[Berth, ...] = (
    Berth("B1", 320.0, 14.5, 1, 30),
    Berth("B2", 280.0, 13.5, 31, 58),
    Berth("B3", 220.0, 12.0, 59, 80),
)

# Legal block types per import cargo, in order of preference ("any legal block").
# Hard rules: IMDG only in H and H only IMDG; reefers only in I/J. Dry boxes are
# legal anywhere except H: standard blocks first, then G (meant for empties), then
# the reefer blocks I/J as a last resort. Trade-off for YARD-1/YARD-3: dry boxes in
# I/J can leave a later reefer with nowhere to go. On the practice data this never
# lowered ships departed and got practice_01 from 3/4 to 4/4.
LEGAL_BLOCK_TYPES: dict[str, tuple[str, ...]] = {
    CARGO_IMDG: (BLOCK_IMDG,),
    CARGO_REEFER: (BLOCK_REEFER,),
    CARGO_DRY: (BLOCK_STANDARD, BLOCK_EMPTY, BLOCK_REEFER),
}

# berth -> ((crane, working bay), ...) in rail order. See the module docstring.
CRANE_SPLIT: dict[str, tuple[tuple[str, int], ...]] = {
    "B1": (("QC1", 5), ("QC2", 20)),
    "B2": (("QC3", 35), ("QC4", 50)),
    "B3": (("QC5", 62), ("QC6", 75)),
}


class SkeletonAgent:
    """Rules-only planner. propose(state, events) -> (actions, requests)."""

    name = AGENT

    def __init__(self) -> None:
        self.escalations: list[str] = []     # drained and logged by the loop
        self._flagged: set[str] = set()

    def _escalate(self, key: str, message: str) -> None:
        if key not in self._flagged:
            self._flagged.add(key)
            self.escalations.append(message)

    def propose(self, state: WorldState, events: list[Event]) -> tuple[list[Action], list[Request]]:
        actions: list[Action] = []
        actions += self._berths(state)
        actions += self._cranes(state)
        actions += self._trucks(state)
        actions += self._placements(state)
        return actions, []

    # ------------------------------------------------------------ berths
    def _berths(self, state: WorldState) -> list[Action]:
        out = []
        occupied = {s.berth for s in state.ships
                    if s.berth and s.status in (SHIP_BERTHED, SHIP_WORKING, SHIP_SAILING)}
        waiting = sorted((s for s in state.ships if s.status == SHIP_ARRIVED),
                         key=lambda s: (s.actual_arrival is None, s.actual_arrival, s.call_id))
        for s in waiting:
            if s.loa_m is None or s.draft_m is None:
                self._escalate(f"size:{s.call_id}", f"{s.call_id} has no valid length or draft; it will wait.")
                continue
            fits = [b for b in BERTHS if b.fits(s.loa_m, s.draft_m)]
            if not fits:
                self._escalate(f"nofit:{s.call_id}", f"{s.call_id} fits no berth (LOA {s.loa_m} m, "
                                                    f"draft {s.draft_m} m); it will wait.")
                continue
            free = [b for b in fits if b.berth_id not in occupied]
            if not free:
                continue
            b = free[0]
            occupied.add(b.berth_id)
            out.append(Action(TOOL_ASSIGN_BERTH, {"call_id": s.call_id, "berth_id": b.berth_id}, AGENT,
                              f"{s.call_id} has arrived and {b.berth_id} is the first free berth it fits."))
        return out

    # ------------------------------------------------------------ cranes
    def _cranes(self, state: WorldState) -> list[Action]:
        out = []
        broken = {c.crane_id for c in state.cranes if c.status == EQ_OUT_OF_SERVICE}
        for s in state.ships:
            if s.status not in (SHIP_BERTHED, SHIP_WORKING):
                continue
            split = CRANE_SPLIT.get(s.berth, ())
            want = [(c, bay) for c, bay in split if c not in broken]
            if not want or sorted(s.cranes) == sorted(c for c, _ in want):
                continue
            skipped = [c for c, _ in split if c in broken]
            reason = (f"{s.call_id} on {s.berth} gets that berth's fixed cranes "
                      f"{', '.join(c for c, _ in want)}"
                      + (f" ({', '.join(skipped)} out of service, skipped)" if skipped else "") + ".")
            out.append(Action(TOOL_PLAN_CRANES, {"call_id": s.call_id, "crane_ids": [c for c, _ in want],
                                                 "bay_positions": [bay for _, bay in want]}, AGENT, reason))
        return out

    # ------------------------------------------------------------ trucks
    def _trucks(self, state: WorldState) -> list[Action]:
        out = []
        free = [t.truck_id for t in state.trucks if t.status == EQ_AVAILABLE]
        for s in state.ships:
            if s.status not in (SHIP_BERTHED, SHIP_WORKING):
                continue
            want = TRUCKS_PER_CRANE * len(s.cranes)
            have = [t.truck_id for t in state.trucks if t.call_id == s.call_id and t.status != EQ_OUT_OF_SERVICE]
            if len(have) >= want or not free:
                continue
            add = free[: want - len(have)]
            del free[: len(add)]
            out.append(Action(TOOL_DISPATCH_TRUCKS, {"call_id": s.call_id, "truck_ids": have + add}, AGENT,
                              f"{s.call_id} has {len(s.cranes)} cranes and {len(have)} trucks; adding "
                              f"{len(add)} to reach {TRUCKS_PER_CRANE} per crane where possible."))
        return out

    # ------------------------------------------------------------ yard
    def _placements(self, state: WorldState) -> list[Action]:
        out = []
        free = {b.block_id: b.free for b in state.blocks}
        btype = {b.block_id: b.block_type for b in state.blocks}
        for s in state.ships:
            if s.status not in (SHIP_BERTHED, SHIP_WORKING) or s.discharge_remaining <= 0:
                continue
            todo: dict[str, list[str]] = {}
            for c in state.manifests.get(s.call_id, ()):
                if c.direction != "import" or c.yard_block:
                    continue
                if c.cargo_type not in LEGAL_BLOCK_TYPES:
                    self._escalate(f"cargo:{c.container_id}",
                                   f"An import box of {s.call_id} has an unknown cargo type; it is not placed.")
                    continue
                todo.setdefault(c.cargo_type, []).append(c.container_id)
            for cargo, boxes in todo.items():
                prefs = LEGAL_BLOCK_TYPES[cargo]
                legal = sorted((b for b in free if btype[b] in prefs), key=lambda b: (prefs.index(btype[b]), b))
                for b in legal:
                    if not boxes:
                        break
                    n = min(free[b], len(boxes))
                    if n <= 0:
                        continue
                    batch, boxes = boxes[:n], boxes[n:]
                    out.append(Action(TOOL_PLACE, {"container_ids": batch, "block_id": b}, AGENT,
                                      f"{n} {cargo} import boxes of {s.call_id} need a yard slot before "
                                      f"discharge; block {b} is a legal {btype[b]} block with {free[b]} free."))
                    free[b] -= n
        return out
