"""Rail-order-safe crane moves (CRANE-1).

    from agents.crane_moves import CallPlan, safe_crane_moves, plans_from_state
    plan = safe_crane_moves(positions, broken, target)
    plan            # list[Action]: plan_crane_sequence calls, in the order to send them
    plan.complete   # True if the target is reached
    plan.reason     # one sentence: why the plan stops short (or "" when complete)

Inputs
  positions  {crane_id: bay} for ALL cranes on the rail, idle and broken included
  broken     set of crane ids that are out of service
  target     {call_id: CallPlan} for EVERY ship that is berthed or working:
             CallPlan(start_bay, end_bay, current=(cranes now on the ship), want={crane: bay})
             A ship whose crane list should not change gets want == its current cranes at their bays.
             `plans_from_state(state, want_by_call)` builds this from a WorldState.

How the simulator works (and why this module exists)
  plan_crane_sequence(call, cranes, bays) REPLACES the ship's crane list with `cranes`
  and moves them to `bays`. A crane can only be moved by assigning it to a berthed ship,
  inside that ship's berth bays. After every call the simulator checks:
    - all cranes, idle and broken included, are in strict rail order (QC1 < ... < QC6), and
    - cranes working on ships are at least 2 bays apart.
  Breaking either costs a -10 violation. So the order of moves matters.

The plan, after every single step, keeps:
  1. all cranes in strict rail order,
  2. working cranes at least 2 bays apart,
  3. every listed crane inside its ship's berth, never broken, never on another ship.
It works in two phases:
  a. release: ships drop the cranes they will not keep (nothing moves, so this is always safe);
  b. place: one crane per step, each step picked so the state after it is safe; a crane is
     added or moved only when its new bay keeps the rail order with every other crane's
     current position. Repeated until every ship has its target or no safe step is left.
If no safe sequence reaches the target, the plan is the safe prefix plus the reason.
It only proposes; the guard re-checks every step before the executor sends it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from core.types import SHIP_BERTHED, SHIP_WORKING, TOOL_PLAN_CRANES, Action, WorldState

AGENT = "crane"
GAP = 2


@dataclass(frozen=True)
class CallPlan:
    start_bay: int
    end_bay: int
    current: tuple[str, ...] = ()                 # cranes on the ship now
    want: Mapping[str, int] = field(default_factory=dict)   # target cranes -> bays


class MovePlan(list):
    """A list of Actions plus whether it reaches the target, and why not."""

    def __init__(self, actions: Iterable[Action] = (), complete: bool = True, reason: str = ""):
        super().__init__(actions)
        self.complete = complete
        self.reason = reason


def _num(crane_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in crane_id if ch.isdigit())
    return (int(digits) if digits else 10**6, crane_id)


# --------------------------------------------------------------------------- safety check

def unsafe(pos: Mapping[str, int], assign: Mapping[str, list[str]], broken: set[str],
           target: Mapping[str, CallPlan]) -> Optional[str]:
    """Why this crane state would be a violation or a refused call, or None if it is safe."""
    rail = sorted(pos, key=_num)
    for a, b in zip(rail, rail[1:]):
        if not pos[a] < pos[b]:
            return f"{a} (bay {pos[a]}) would not be left of {b} (bay {pos[b]})"
    seen: dict[str, str] = {}
    for call, cranes in assign.items():
        plan = target[call]
        for c in cranes:
            if c in broken:
                return f"{c} is out of service"
            if c in seen:
                return f"{c} would be on {seen[c]} and {call}"
            seen[c] = call
            if not plan.start_bay <= pos[c] <= plan.end_bay:
                return f"{c} at bay {pos[c]} would be outside {call}'s berth ({plan.start_bay}-{plan.end_bay})"
    working = sorted((pos[c], c) for c in seen)
    for (p1, c1), (p2, c2) in zip(working, working[1:]):
        if p2 - p1 < GAP:
            return f"working cranes {c1} (bay {p1}) and {c2} (bay {p2}) would be under {GAP} bays apart"
    return None


# --------------------------------------------------------------------------- planner

def _action(call: str, cranes: list[str], pos: Mapping[str, int], reason: str) -> Action:
    ordered = sorted(cranes, key=_num)
    return Action(TOOL_PLAN_CRANES, {"call_id": call, "crane_ids": ordered,
                                     "bay_positions": [pos[c] for c in ordered]}, AGENT, reason)


def safe_crane_moves(positions: Mapping[str, int], broken: Iterable[str],
                     target: Mapping[str, CallPlan]) -> MovePlan:
    """Ordered plan_crane_sequence calls that reach `target` without ever breaking a crane rule."""
    broken = set(broken)
    pos = dict(positions)
    assign = {call: [c for c in p.current if c not in broken] for call, p in target.items()}
    actions: list[Action] = []

    # Sanity: the target itself must be reachable as a final state.
    wanted_by: dict[str, str] = {}
    for call, p in target.items():
        for c in p.want:
            if c not in pos:
                return MovePlan([], False, f"{call} wants unknown crane {c}")
            if c in broken:
                return MovePlan([], False, f"{call} wants {c}, which is out of service")
            if c in wanted_by:
                return MovePlan([], False, f"{c} is wanted by both {wanted_by[c]} and {call}")
            wanted_by[c] = call
    final_pos = {**pos, **{c: b for p in target.values() for c, b in p.want.items()}}
    final_assign = {call: list(p.want) for call, p in target.items()}
    why = unsafe(final_pos, final_assign, broken, target)
    if why:
        return MovePlan([], False, f"The target itself is not safe: {why}.")

    # The state right now must be safe too (otherwise nothing we send can be checked against it).
    why = unsafe(pos, assign, broken, target)
    if why:
        return MovePlan([], False, f"The current crane state is not safe: {why}.")

    # Phase a: release cranes each ship will not keep (positions unchanged -> always safe).
    for call in sorted(target):
        keep = [c for c in assign[call] if c in target[call].want]
        if keep != assign[call]:
            dropped = [c for c in assign[call] if c not in keep]
            assign[call] = keep
            actions.append(_action(call, keep, pos,
                                   f"{call} releases {', '.join(dropped)} before cranes are re-planned "
                                   f"(nothing moves, so rail order and spacing stay safe)."))

    # Phase b: place one crane per step, any step whose resulting state is safe.
    pending = {c: (call, b) for call, p in target.items() for c, b in p.want.items()
               if not (c in assign[call] and pos[c] == b)}
    while pending:
        progressed = False
        # Try cranes in an order that tends to work: moves away from neighbours first.
        for c in sorted(pending, key=lambda c: (-abs(pending[c][1] - pos[c]), _num(c))):
            call, bay = pending[c]
            trial_pos = {**pos, c: bay}
            trial_assign = {k: list(v) for k, v in assign.items()}
            if c not in trial_assign[call]:
                trial_assign[call].append(c)
            if unsafe(trial_pos, trial_assign, broken, target) is None:
                moved = pos[c] != bay
                pos, assign = trial_pos, trial_assign
                del pending[c]
                verb = f"moves {c} from bay {positions[c]} to bay {bay}" if moved else f"takes {c} at bay {bay}"
                actions.append(_action(call, assign[call], pos,
                                       f"{call} {verb}; every crane stays in rail order and working "
                                       f"cranes stay at least {GAP} bays apart."))
                progressed = True
                break
        if not progressed:
            stuck = ", ".join(f"{c}→{b} for {call}" for c, (call, b) in sorted(pending.items(), key=lambda x: _num(x[0])))
            return MovePlan(actions, False, f"No safe next step for {stuck}: every order would put two cranes out "
                                            f"of rail order or too close; the plan stops at the last safe state.")
    return MovePlan(actions, True, "")


# --------------------------------------------------------------------------- from a WorldState

def plans_from_state(state: WorldState, want_by_call: Mapping[str, Mapping[str, int]]
                     ) -> tuple[dict[str, int], set[str], dict[str, CallPlan]]:
    """(positions, broken, target) for safe_crane_moves, from a snapshot.

    Every berthed or working ship is included. Ships not in `want_by_call` keep their
    current (in-service) cranes where they are.
    """
    positions = {c.crane_id: c.bay for c in state.cranes}
    broken = {c.crane_id for c in state.cranes if c.status == "out_of_service"}
    target: dict[str, CallPlan] = {}
    for s in state.ships:
        if s.status not in (SHIP_BERTHED, SHIP_WORKING) or not s.berth:
            continue
        b = state.berth(s.berth)
        if b is None:
            continue
        current = tuple(c for c in s.cranes if c in positions)
        want = want_by_call.get(s.call_id)
        if want is None:
            want = {c: positions[c] for c in current if c not in broken}
        target[s.call_id] = CallPlan(b.start_bay, b.end_bay, current, dict(want))
    return positions, broken, target
