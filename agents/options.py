"""Option generator for the supervisor (BC-3).

    from agents import options
    opts = options.build(state)        # [] when no berth/crane plans genuinely compete

When two or three reasonable plans compete, this builds them as `Option`s for the LLM
supervisor (LLM-4) to choose from, each with predicted effects and a code-written template
explanation (used word for word when the LLM is off). The coordinator returns them from
`decide()`; the loop runs the chosen option's actions first and drops the agents' own actions
of the same kind on the same ship, berth or cranes.

Rules every option follows:
  - Rule-safe: each option's actions are checked with the guard, step by step, on the state
    as it would be after the earlier steps. An option that fails is dropped; if option 0
    fails, no options are offered (the agents' plan stands).
  - Option 0 is the safest reasonable default: what the agents would do anyway.
  - No metric-only options: an alternative is offered only if it changes ships finishing in
    the window or the waiting ships' average finish time by at least MIN_GAIN_H hours
    (LLM-1: models took metric-gaming bait, so the code never offers it).

Berth choice (on): a berth is free and more arrived ships fit it than there are free berths
for them. Option 0 = the berth agent's own choice (BERTH-2: least work first, best fit);
alternatives = the oldest arrival, and a ship that fits only this berth. Predicted effects
come from a small crane-time model: the cranes in service shared among the ships at berth the way
the crane agent shares them (agents/crane.py split()), CRANE_RATE moves/h each, berths freeing as
ships finish; yard space is not modelled, and the template says so.

Crane choice (CRANE_OPTIONS = False, stays off): a ship at berth has cranes but can't
discharge (waiting for yard space) while another berthed ship can work. Option 0 = keep the
cranes; option 1 = move them to the working ship through safe_crane_moves (CRANE-1). Since
CRANE-2 (agents/crane.py) the crane agent makes this call itself every tick: a stalled ship
gets no cranes, so "keep" is never the agents' plan and the choice no longer competes.
"""
from __future__ import annotations

from dataclasses import replace
import guard
from agents.crane_moves import _num, plans_from_state, safe_crane_moves
from core.types import (EQ_AVAILABLE, EQ_WORKING, SHIP_ARRIVED, SHIP_BERTHED, SHIP_SAILING, SHIP_WORKING,
                        TOOL_ASSIGN_BERTH, TOOL_PLAN_CRANES, Action, Option, WorldState)

AGENT = "options"
CRANE_RATE = 23.0            # moves/h per crane at full trucks (same as YARD-3)
PREDICT_STEP_H = 0.25        # berth-option model step (one normal tick)
WORK_DELAY_H = 1.0           # work starts ~1 h after berthing
SAIL_DELAY_H = 1.0           # a ship sails ~1 h after its last move
MIN_GAIN_H = 1.0             # an alternative must change the outcome by at least this much
CRANE_OPTIONS = False        # stays off: CRANE-2 decides this itself (see module docstring)
BUSY = (SHIP_BERTHED, SHIP_WORKING, SHIP_SAILING)


# --------------------------------------------------------------------------
# rule safety: every step through the guard, on the state as it would be
# --------------------------------------------------------------------------

def apply(action: Action, state: WorldState) -> WorldState:
    """The state after `action`, as far as the guard's rules are concerned."""
    a = action.args
    if action.tool == TOOL_ASSIGN_BERTH:
        ships = tuple(replace(s, status=SHIP_BERTHED, berth=a["berth_id"]) if s.call_id == a["call_id"] else s
                      for s in state.ships)
        return replace(state, ships=ships)
    if action.tool == TOOL_PLAN_CRANES:
        call, ids, bays = a["call_id"], list(a["crane_ids"]), list(a["bay_positions"])
        at = dict(zip(ids, bays))
        old = set(next((s.cranes for s in state.ships if s.call_id == call), ()))
        cranes = []
        for c in state.cranes:
            if c.crane_id in at:
                cranes.append(replace(c, status=EQ_WORKING, call_id=call, bay=at[c.crane_id]))
            elif c.crane_id in old:
                cranes.append(replace(c, status=EQ_AVAILABLE, call_id=None))
            else:
                cranes.append(c)
        ships = tuple(replace(s, cranes=tuple(ids)) if s.call_id == call else s for s in state.ships)
        return replace(state, cranes=tuple(cranes), ships=ships)
    return state


def rule_safe(option: Option, state: WorldState) -> tuple[bool, str]:
    """(True, "ok") if every action passes the guard in order; else (False, why)."""
    st = state
    for i, act in enumerate(option.actions):
        ok, why = guard.check(act, st)
        if not ok:
            return False, f"step {i + 1} ({act.tool}): {why}"
        st = apply(act, st)
    return True, "ok"


# --------------------------------------------------------------------------
# berth choice
# --------------------------------------------------------------------------

def _hours(state, when) -> float:
    return (when - state.now).total_seconds() / 3600


def _predict(state: WorldState, first, berth_id: str, waiting: list) -> dict:
    """Finish times (hours from now) of the waiting ships if `first` takes `berth_id` now and
    the rest follow in the berth agent's order onto whichever fitting berth frees first.

    A small time-step model: the cranes in service are shared among the ships at berth the way
    the crane agent shares them (agents/crane.py split(): one each, spare cranes to the shortest
    job), CRANE_RATE moves/h each; work starts WORK_DELAY_H after berthing, a ship sails
    SAIL_DELAY_H after its last move. Yard space is not modelled (the template says so)."""
    from agents import crane as crane_model
    n = crane_model.in_service(state)
    left = {}                                        # call_id -> moves left
    start = {}                                       # call_id -> hours from now when cranes may work
    berth_of = {}
    for s in state.ships:                            # ships already at a berth
        if s.berth and s.status in BUSY:
            left[s.call_id] = max(0, s.remaining_moves)
            start[s.call_id] = 0.0
            berth_of[s.call_id] = s.berth
    left[first.call_id], start[first.call_id], berth_of[first.call_id] = \
        max(0, first.remaining_moves), WORK_DELAY_H, berth_id
    queue = [s for s in waiting if s.call_id != first.call_id and
             any(b.fits(s.loa_m, s.draft_m) for b in state.berths)]
    finish: dict = {}
    sailed: dict = {}
    t, step, horizon = 0.0, PREDICT_STEP_H, max(1.0, _hours(state, state.scenario_end)) + 48.0
    while t < horizon and (queue or any(k not in sailed for k in left)):
        busy = {berth_of[k] for k in left if k not in sailed or sailed[k] > t}
        for s in list(queue):                        # next ships onto berths that are free now
            b = next((x.berth_id for x in state.berths if x.berth_id not in busy and x.fits(s.loa_m, s.draft_m)), None)
            if b is None:
                continue
            queue.remove(s)
            busy.add(b)
            left[s.call_id], start[s.call_id], berth_of[s.call_id] = max(0, s.remaining_moves), t + WORK_DELAY_H, b
        active = {k: max(0.0, min(step, t + step - start[k])) for k in left if k not in sailed}
        work = {k: (left[k] if active[k] > 0 else 0) for k in active}
        cranes = crane_model.split(work, left, n, step, CRANE_RATE)
        for k in active:
            if k in sailed:
                continue
            left[k] -= min(left[k], CRANE_RATE * cranes.get(k, 0) * active[k])
            if left[k] <= 0 and start[k] <= t + step:
                sailed[k] = t + step + SAIL_DELAY_H
        t += step
    for k, at in sailed.items():
        finish[k] = at
    finish = {k: v for k, v in finish.items() if k == first.call_id or k in {s.call_id for s in waiting}}
    for s in [first] + list(waiting):                # not finished within the model horizon
        finish.setdefault(s.call_id, horizon)
    window = _hours(state, state.scenario_end)
    done = sum(1 for h in finish.values() if h <= window)
    return {"finish_h": {k: round(v, 1) for k, v in finish.items()},
            "ships_finishing_in_window": done,
            "avg_finish_h": round(sum(finish.values()) / len(finish), 1) if finish else 0.0}


def _berth_option(state, oid, ship, b, waiting, why) -> Option:
    pred = _predict(state, ship, b.berth_id, waiting)
    pred["ship"] = ship.call_id
    pred["berth"] = b.berth_id
    pred["waited_h"] = round((state.now - ship.actual_arrival).total_seconds() / 3600, 1)
    act = Action(TOOL_ASSIGN_BERTH, {"call_id": ship.call_id, "berth_id": b.berth_id}, AGENT,
                 f"Supervisor's choice: berth {ship.call_id} on {b.berth_id} ({why}).")
    tmpl = (f"Berth {ship.call_id} on {b.berth_id} now ({why}): by crane time alone it finishes in about "
            f"{pred['finish_h'][ship.call_id]:g} h, {pred['ships_finishing_in_window']} of the waiting ships finish "
            f"in the window and they finish {pred['avg_finish_h']:g} h from now on average (yard space not modelled).")
    return Option(oid, (act,), pred, tmpl)


def berth_options(state: WorldState) -> list[Option]:
    from agents import berth as berth_agent
    occupied = {s.berth for s in state.ships if s.berth and s.status in BUSY}
    waiting = berth_agent.rank(state, [s for s in state.ships if s.status == SHIP_ARRIVED and s.actual_arrival
                                       and s.loa_m and s.draft_m and s.loa_m > 0 and s.draft_m > 0])
    free = [b for b in state.berths if b.berth_id not in occupied]
    planned = {b.berth_id: (s, why) for s, b, why in berth_agent.plan(state)}
    for b in free:
        fit = [s for s in waiting if b.fits(s.loa_m, s.draft_m)]
        if len(fit) < 2 or b.berth_id not in planned:
            continue
        # Competition: more ships want this berth than there are free berths they could take.
        room = {x.berth_id for s in fit for x in free if x.fits(s.loa_m, s.draft_m)}
        if len(fit) <= len(room):
            continue
        ship, why = planned[b.berth_id]
        base = _berth_option(state, "berth_agent_plan", ship, b, waiting,
                             f"the berth agent's choice: least work first, {why}")
        alts = []
        oldest = min(fit, key=lambda s: (s.actual_arrival, s.call_id))
        if oldest.call_id != ship.call_id:
            alts.append(_berth_option(state, "oldest_first", oldest, b, waiting, "it arrived first"))
        only = [s for s in fit if [x.berth_id for x in state.berths if x.fits(s.loa_m, s.draft_m)] == [b.berth_id]]
        if only and only[0].call_id not in {ship.call_id, oldest.call_id}:
            alts.append(_berth_option(state, "only_fits_here", only[0], b, waiting,
                                      f"it fits no berth but {b.berth_id}"))
        alts = [o for o in alts if _real_gain(o, base)]
        if alts:
            return [base] + alts[:2]
        return []
    return []


def _real_gain(alt: Option, base: Option) -> bool:
    """Not metric-only: changes ships finishing in the window, or the average finish by >= MIN_GAIN_H."""
    a, b = alt.predicted, base.predicted
    return (a["ships_finishing_in_window"] != b["ships_finishing_in_window"]
            or abs(a["avg_finish_h"] - b["avg_finish_h"]) >= MIN_GAIN_H)


# --------------------------------------------------------------------------
# crane choice (stalled ship), off until CRANE-2
# --------------------------------------------------------------------------

def _can_work(state: WorldState, s) -> bool:
    """Has something to do now: boxes placed in the yard but not yet discharged, or loading."""
    if s.discharge_remaining == 0:
        return s.load_remaining > 0
    imports = [c for c in state.manifests.get(s.call_id, ()) if c.direction == "import"]
    placed = sum(1 for c in imports if c.yard_block)
    discharged = max(0, len(imports) - s.discharge_remaining)
    return placed - discharged > 0


def _spread(cranes: list[str], start: int, end: int) -> dict[str, int]:
    """Bays for `cranes` (rail order) spread evenly over a berth, at least 2 apart."""
    n = len(cranes)
    if n == 0:
        return {}
    step = max(2, (end - start) // n)
    return {c: min(end, start + 1 + i * step) for i, c in enumerate(cranes)}


def crane_options(state: WorldState) -> list[Option]:
    at = [s for s in state.ships if s.status in (SHIP_BERTHED, SHIP_WORKING) and s.berth]
    stalled = [s for s in at if s.cranes and s.discharge_remaining > 0 and not _can_work(state, s)]
    busy = [s for s in at if s.call_id not in {x.call_id for x in stalled} and _can_work(state, s)]
    if not stalled or not busy:
        return []
    s = stalled[0]
    o = max(busy, key=lambda x: (x.remaining_moves, x.call_id))
    b = state.berth(o.berth)
    if b is None:
        return []
    new = sorted(set(o.cranes) | set(s.cranes), key=_num)
    want = {s.call_id: {}, o.call_id: _spread(new, b.start_bay, b.end_bay)}
    pos, broken, target = plans_from_state(state, want)
    plan = safe_crane_moves(pos, broken, target)
    if not plan.complete:
        return []
    n = len(s.cranes)
    keep = Option("keep_cranes", (), {"ship": s.call_id, "moves_per_h_now": 0},
                  f"Keep the cranes on {s.call_id}: it is waiting for yard space and makes no moves until "
                  f"slots free up; no crane moves, so nothing can go wrong on the rail.")
    gain = round(CRANE_RATE * n)
    move = Option("move_cranes", tuple(replace(a, agent=AGENT) for a in plan),
                  {"from": s.call_id, "to": o.call_id, "cranes": n, "extra_moves_per_h": gain,
                   "moves_left": o.remaining_moves},
                  f"Move {n} cranes from {s.call_id} to {o.call_id} on {o.berth}: about {gain} more moves per "
                  f"hour on a ship with {o.remaining_moves} moves left, in {len(plan)} rail-safe steps.")
    return [keep, move]


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build(state: WorldState) -> list[Option]:
    """2-3 rule-safe options for one competing decision, or []. Never raises."""
    try:
        opts = berth_options(state)
        if not opts and CRANE_OPTIONS:
            opts = crane_options(state)
    except Exception:
        return []
    safe = [o for o in opts if rule_safe(o, state)[0]]
    if not safe or not opts or safe[0] is not opts[0] or len(safe) < 2:
        return []                       # option 0 must be safe, and a choice needs two
    return safe[:3]
