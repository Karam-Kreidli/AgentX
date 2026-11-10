"""Single-agent baseline (CORE-8): ONE planner that does berth, crane, truck and yard in one pass.

    python run.py --agent single --scenario practice_01 --no-llm

It exists to answer "why multi-agent?". It is meant to be an honest, reasonable planner, not a
strawman: best-fit berths, cranes split in proportion to the work, trucks 3 per crane, yard blocks
chosen by travel time. What it does NOT have, on purpose: no negotiation (no Requests, no
`grants`), no admission control, no budgets per ship, no impossible-ship detection, no stall
detector, no options, no LLM. The loop runs it with the coordinator off, through the same guard,
executor and rules-only fallback as every other system.

It reads only the WorldState. Allowed imports: core.types, core.state.hours, core.skeleton
(static berths and legal block order), agents.crane_moves (rail-safe crane moves), agents.yard
TRAVEL_MIN (static truck travel times). A test checks that nothing else is imported.

One joint pass per tick:
  1. berth   arrived ships in arrival order, each on the smallest free berth it fits
  2. cranes  in-service cranes split over the ships that can work now, in proportion to their
             remaining moves, in rail order, bays spread evenly
  3. trucks  3 per assigned crane; trucks of ships with no cranes are released
  4. yard    import boxes of berthed ships go to the legal block nearest by travel time that is
             under 85 % full, else any legal block with room
"""
from __future__ import annotations

import math
from typing import Optional

from agents.crane_moves import plans_from_state, safe_crane_moves
from agents.yard import TRAVEL_MIN
from core.skeleton import BERTHS, LEGAL_BLOCK_TYPES
from core.types import (EQ_AVAILABLE, EQ_OUT_OF_SERVICE, SHIP_ARRIVED, SHIP_BERTHED, SHIP_SAILING,
                        SHIP_WORKING, TOOL_ASSIGN_BERTH, TOOL_DISPATCH_TRUCKS, TOOL_PLACE,
                        Action, Event, Request, WorldState)

AGENT = "single"
TRUCKS_PER_CRANE = 3
FILL_LIMIT = 0.85            # GROUND_RULES §1: placements into blocks over 85 % are counted
AT_BERTH = (SHIP_BERTHED, SHIP_WORKING)


def _rail(crane_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in crane_id if ch.isdigit())
    return (int(digits) if digits else 10**6, crane_id)


def _bays(n: int, lo: int, hi: int) -> Optional[list[int]]:
    """n bays spread evenly over [lo, hi], increasing and at least 2 apart; None if they do not fit."""
    if n == 0:
        return []
    if hi - lo < 2 * (n - 1):
        return None
    width = (hi - lo + 1) / n
    out = [int(lo + width * (i + 0.5)) for i in range(n)]
    for i in range(1, n):
        out[i] = max(out[i], out[i - 1] + 2)
    return out if out[-1] <= hi else None


def _largest_remainder(total: int, weights: list[float]) -> list[int]:
    """Split `total` over the weights, at least 1 each (needs total >= len(weights))."""
    n = len(weights)
    out = [1] * n
    spare = total - n
    if spare <= 0:
        return out
    s = sum(weights) or float(n)
    share = [spare * (w / s if sum(weights) else 1 / n) for w in weights]
    base = [int(x) for x in share]
    left = spare - sum(base)
    order = sorted(range(n), key=lambda i: (-(share[i] - base[i]), i))
    for i in order[:left]:
        base[i] += 1
    return [o + b for o, b in zip(out, base)]


class SingleAgent:
    name = AGENT

    def __init__(self) -> None:
        self.escalations: list[str] = []          # read by the loop; this agent never escalates
        self._first_discharge: dict[str, int] = {}

    def propose(self, state: WorldState, events: list[Event]) -> tuple[list[Action], list[Request]]:
        for s in state.ships:
            self._first_discharge.setdefault(s.call_id, s.discharge_remaining)
        actions = self._berths(state)
        cranes = self._cranes(state)
        actions += cranes
        actions += self._trucks(state)
        actions += self._yard(state)
        return actions, []

    # ------------------------------------------------------------ berth
    def _berths(self, state: WorldState) -> list[Action]:
        berths = state.berths or BERTHS
        occupied = {s.berth for s in state.ships
                    if s.berth and s.status in (SHIP_BERTHED, SHIP_WORKING, SHIP_SAILING)}
        waiting = sorted((s for s in state.ships if s.status == SHIP_ARRIVED),
                         key=lambda s: (s.actual_arrival is None, s.actual_arrival, s.call_id))
        out = []
        for s in waiting:
            if s.loa_m is None or s.draft_m is None:
                continue                                           # size unknown: never berthed
            free = [b for b in berths if b.berth_id not in occupied and b.fits(s.loa_m, s.draft_m)]
            if not free:
                continue
            b = min(free, key=lambda b: (b.length_m, b.berth_id))   # best fit: smallest berth that fits
            occupied.add(b.berth_id)
            out.append(Action(TOOL_ASSIGN_BERTH, {"call_id": s.call_id, "berth_id": b.berth_id}, AGENT,
                              f"{s.call_id} has arrived and {b.berth_id} is the smallest free berth that fits "
                              f"its {s.loa_m:g} m length, in arrival order."))
        return out

    # ------------------------------------------------------------ facts about a berthed ship
    def _imports(self, state: WorldState, s):
        total = placed = 0
        unplaced = []
        for c in state.manifests.get(s.call_id, ()):
            if c.direction != "import":
                continue
            total += 1
            if c.yard_block:
                placed += 1
            elif c.cargo_type in LEGAL_BLOCK_TYPES:
                unplaced.append(c)
        return total, placed, unplaced

    def _room(self, state: WorldState, cargo: str) -> int:
        types = LEGAL_BLOCK_TYPES[cargo]
        return sum(b.free for b in state.blocks if b.block_type in types)

    def _can_work(self, state: WorldState, s) -> bool:
        """Loading, or has placed boxes still to discharge, or can still get yard space for its imports."""
        if s.discharge_remaining <= 0:
            return True                      # loading, or a last fraction of a move that needs a crane
        total, placed, unplaced = self._imports(state, s)
        if total == 0:
            return True                      # no manifest rows: cannot judge, let it work
        done = max(0, self._first_discharge.get(s.call_id, s.discharge_remaining) - s.discharge_remaining)
        if placed - done > 0:
            return True
        need: dict[str, int] = {}
        for c in unplaced:
            need[c.cargo_type] = need.get(c.cargo_type, 0) + 1
        return any(self._room(state, cargo) > 0 for cargo in need)

    # ------------------------------------------------------------ cranes
    def _cranes(self, state: WorldState) -> list[Action]:
        berths = state.berths or BERTHS
        at = sorted((s for s in state.ships if s.status in AT_BERTH and s.berth),
                    key=lambda s: (state.berth(s.berth).start_bay if state.berth(s.berth) else 10**6, s.call_id))
        if not at:
            return []
        broken = {c.crane_id for c in state.cranes if c.status == EQ_OUT_OF_SERVICE}
        usable = sorted((c.crane_id for c in state.cranes if c.crane_id not in broken), key=_rail)
        workers = [s for s in at if self._can_work(state, s)]
        if len(workers) > len(usable):           # more ships than cranes: the earliest arrivals first
            keep = sorted(workers, key=lambda s: (s.actual_arrival is None, s.actual_arrival, s.call_id))
            workers = [s for s in at if s in keep[:len(usable)]]
        if self._current_ok(at, workers, usable, broken):
            return []                            # nothing broke and nobody started or finished: no churn
        want: dict[str, dict[str, int]] = {s.call_id: {} for s in at}
        if workers and usable:
            counts = _largest_remainder(len(usable), [max(1, s.remaining_moves) for s in workers])
            i = 0
            for s, n in zip(workers, counts):
                group = usable[i:i + n]
                i += n
                b = state.berth(s.berth)
                bays = _bays(n, b.start_bay, b.end_bay) if b else None
                if bays:
                    want[s.call_id] = dict(zip(group, bays))
        pos, brk, target = plans_from_state(state, want)
        out = []
        for a in safe_crane_moves(pos, brk, target):
            out.append(Action(a.tool, a.args, AGENT, a.reason))
        return out

    @staticmethod
    def _current_ok(at, workers, usable, broken) -> bool:
        """Is today's crane layout already a valid split? Every worker has a crane, nobody else has
        one, no broken crane is assigned, and every in-service crane is on some worker."""
        worker_ids = {s.call_id for s in workers}
        used: list[str] = []
        for s in at:
            mine = [c for c in s.cranes]
            if any(c in broken for c in mine):
                return False
            if s.call_id in worker_ids and not mine:
                return False
            if s.call_id not in worker_ids and mine:
                return False
            used += mine
        return (not workers) or sorted(used) == sorted(usable)

    # ------------------------------------------------------------ trucks
    def _trucks(self, state: WorldState) -> list[Action]:
        broken = {t.truck_id for t in state.trucks if t.status == EQ_OUT_OF_SERVICE}
        free = sorted(t.truck_id for t in state.trucks if t.status == EQ_AVAILABLE)
        serving: dict[str, list[str]] = {}
        for t in state.trucks:
            if t.call_id and t.truck_id not in broken:
                serving.setdefault(t.call_id, []).append(t.truck_id)
        ships = sorted((s for s in state.ships if s.status in AT_BERTH and s.berth),
                       key=lambda s: (state.berth(s.berth).start_bay if state.berth(s.berth) else 10**6, s.call_id))
        plan: dict[str, list[str]] = {}
        reasons: dict[str, str] = {}
        for s in ships:                                               # release first
            want = TRUCKS_PER_CRANE * len(s.cranes)
            have = sorted(serving.get(s.call_id, []))
            plan[s.call_id] = have[:want]
            if len(have) > want:
                free += have[want:]
                reasons[s.call_id] = (f"{s.call_id} has {len(s.cranes)} crane(s), so it keeps {want} trucks and "
                                      f"releases {len(have) - want}.")
        for s in ships:                                               # then top up in berth order
            want = TRUCKS_PER_CRANE * len(s.cranes)
            short = want - len(plan[s.call_id])
            if short > 0 and free:
                add, free = free[:short], free[short:]
                plan[s.call_id] = plan[s.call_id] + add
                reasons[s.call_id] = (f"{s.call_id} has {len(s.cranes)} crane(s) and "
                                      f"{len(plan[s.call_id]) - len(add)} trucks; adding {len(add)} "
                                      f"toward {TRUCKS_PER_CRANE} per crane.")
        return [Action(TOOL_DISPATCH_TRUCKS, {"call_id": cid, "truck_ids": plan[cid]}, AGENT, why)
                for cid, why in reasons.items()]

    # ------------------------------------------------------------ yard
    def _yard(self, state: WorldState) -> list[Action]:
        free = {b.block_id: b.free for b in state.blocks}
        used = {b.block_id: b.used for b in state.blocks}
        slots = {b.block_id: b.slots for b in state.blocks}
        btype = {b.block_id: b.block_type for b in state.blocks}
        ships = sorted((s for s in state.ships if s.status in AT_BERTH and s.discharge_remaining > 0),
                       key=lambda s: (s.actual_arrival is None, s.actual_arrival, s.call_id))
        out = []
        for s in ships:
            _, _, unplaced = self._imports(state, s)
            by_cargo: dict[str, list[str]] = {}
            for c in unplaced:
                ids = by_cargo.setdefault(c.cargo_type, [])
                if c.container_id not in ids:
                    ids.append(c.container_id)
            for cargo, boxes in by_cargo.items():
                prefs = LEGAL_BLOCK_TYPES[cargo]
                legal = [b for b in free if btype[b] in prefs]

                def dist(b: str, s=s) -> float:
                    return TRAVEL_MIN.get(s.berth or "", {}).get(b, 99.0)

                # Pass 1: nearest legal block under 85 %, most preferred block type first.
                # Pass 2: any legal block with room.
                for limited in (True, False):
                    for t in prefs:
                        for b in sorted((x for x in legal if btype[x] == t), key=lambda x: (dist(x), x)):
                            if not boxes:
                                break
                            room = free[b]
                            if limited:
                                room = min(room, math.floor(FILL_LIMIT * slots[b]) - used[b])
                            n = min(room, len(boxes))
                            if n <= 0:
                                continue
                            batch, boxes = boxes[:n], boxes[n:]
                            out.append(Action(TOOL_PLACE, {"container_ids": batch, "block_id": b}, AGENT,
                                              f"{n} {cargo} import boxes of {s.call_id} need a yard slot before "
                                              f"discharge; block {b} is a legal {btype[b]} block "
                                              + (f"{dist(b):g} min from {s.berth} and under 85 % full."
                                                 if limited else f"with room (every nearer legal block is over 85 %).")))
                            free[b] -= n
                            used[b] += n
                    if not boxes:
                        break
        return out
