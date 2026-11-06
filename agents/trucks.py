"""Truck dispatch v0 (YARD-2): no working crane without trucks.

    from agents.trucks import TruckAgent
    actions, requests = TruckAgent().propose(state, events)

The simulator runs a crane at full speed with 3 trucks per crane on its ship
(speed x min(1, trucks / (3 x cranes))). Each tick, for every berthed or working ship:
  - target = 3 trucks per crane on the ship (0 if it has no cranes)
  - broken trucks are dropped from the ship's list and replaced the same tick
    (the simulator only drops them by itself once the ship is working)
  - short of the target: add available trucks; still short when none are free:
    send a truck Request to the coordinator with the shortfall
  - more than the target (cranes removed, or the ship has none): release the extras,
    so other ships can use them
Ships that finished are released by the simulator itself when they sail.

When trucks are scarce, ships that are already working are served first, then those
waiting for mooring to finish; within each, ships with more cranes first.
It never hands out spare trucks beyond 3 per crane just to lower the idle-hours
number (GROUND_RULES: no metric gaming).

It only proposes; the loop sends each action through the guard and the executor.
"""
from __future__ import annotations

from core.types import (EQ_AVAILABLE, EQ_OUT_OF_SERVICE, RES_TRUCKS, SHIP_BERTHED, SHIP_WORKING,
                        TOOL_DISPATCH_TRUCKS, Action, Event, Request, WorldState)

AGENT = "trucks"
PER_CRANE = 3


class TruckAgent:
    name = AGENT

    def __init__(self) -> None:
        self.escalations: list[str] = []      # drained and logged by the loop

    def propose(self, state: WorldState, events: list[Event]) -> tuple[list[Action], list[Request]]:
        actions: list[Action] = []
        requests: list[Request] = []
        broken = {t.truck_id for t in state.trucks if t.status == EQ_OUT_OF_SERVICE}
        free = sorted(t.truck_id for t in state.trucks if t.status == EQ_AVAILABLE)
        serving: dict[str, list[str]] = {}
        for t in state.trucks:
            if t.call_id:
                serving.setdefault(t.call_id, []).append(t.truck_id)

        ships = [s for s in state.ships if s.status in (SHIP_BERTHED, SHIP_WORKING)]
        # Release first, so freed trucks can go to ships that are short this same tick.
        plan: dict[str, list[str]] = {}
        for s in ships:
            want = PER_CRANE * len(s.cranes)
            have = sorted(serving.get(s.call_id, []))
            healthy = [t for t in have if t not in broken]
            gone = [t for t in have if t in broken]
            keep = healthy[:want]
            extra = healthy[want:]
            free += extra
            plan[s.call_id] = keep
            if gone or extra:
                why = []
                if gone:
                    why.append(f"{', '.join(gone)} out of service")
                if extra:
                    why.append(f"{len(extra)} more than {PER_CRANE} per crane for {len(s.cranes)} crane(s)")
                actions.append(Action(
                    TOOL_DISPATCH_TRUCKS, {"call_id": s.call_id, "truck_ids": keep}, AGENT,
                    f"{s.call_id} keeps {len(keep)} trucks and lets go of "
                    f"{len(gone) + len(extra)} ({'; '.join(why)})."))
        free = sorted(set(free))

        # Then top up: working ships first, then berthed; more cranes first.
        order = sorted(ships, key=lambda s: (s.status != SHIP_WORKING, -len(s.cranes), s.call_id))
        for s in order:
            want = PER_CRANE * len(s.cranes)
            keep = plan[s.call_id]
            short = want - len(keep)
            if short <= 0:
                continue
            add, free = free[:short], free[short:]
            if add:
                new = keep + add
                plan[s.call_id] = new
                # replace an earlier release action for this ship, if any, with the final list
                actions = [a for a in actions if a.args["call_id"] != s.call_id]
                actions.append(Action(
                    TOOL_DISPATCH_TRUCKS, {"call_id": s.call_id, "truck_ids": new}, AGENT,
                    f"{s.call_id} has {len(s.cranes)} crane(s) and {len(keep)} working trucks; adding "
                    f"{len(add)} to reach {len(new)} of the {want} needed for full crane speed."))
            still = want - len(plan[s.call_id])
            if still > 0:
                requests.append(Request(
                    AGENT, RES_TRUCKS, still, s.call_id, 1.0 if s.status == SHIP_WORKING else 0.5,
                    f"{s.call_id} needs {still} more truck(s) for {len(s.cranes)} crane(s) at full speed; "
                    f"none are free."))
        return actions, requests
