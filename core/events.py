"""Event detector (CORE-3).

    events = detect_events(prev_state, cur_state)

Compares two snapshots and lists what changed. `prev` may be None for the
first snapshot: everything already true at the start (ships already arrived,
cranes already down, holds already placed) is then reported once.

Event kinds (core.types.EVENT_KINDS):
    arrival, berthed, departed            ship status changes
    crane_down, crane_back                crane goes out of / back into service
    truck_down, truck_back                same for yard trucks
    customs_hold, hold_released           a new hold appears / customs lifts it
    anomaly                               a new finding from YARD-5 validation
    ship_stalled, ship_impossible         from YARD-4: agents.stall.detect(state)
    operator_message                      from LLM-6's inbox: core.inbox.poll()

The last two groups come from modules other tasks build. detect_events calls
them when they exist and skips them until then. A failing hook is reported as
an anomaly event and never crashes the run.

Event.subject is a call_id, equipment_id or container_id (untrusted text: pass
it to an LLM only as quoted, truncated data). Event.t is simulated hours since
scenario start.
"""
from __future__ import annotations

from typing import Optional

from core.state import hours
from core.types import (EQ_OUT_OF_SERVICE, EV_ANOMALY, EV_ARRIVAL, EV_BERTHED, EV_CRANE_BACK,
                        EV_CRANE_DOWN, EV_CUSTOMS_HOLD, EV_DEPARTED, EV_HOLD_RELEASED,
                        EV_OPERATOR_MESSAGE, EV_TRUCK_BACK, EV_TRUCK_DOWN, SHIP_DEPARTED,
                        SHIP_EXPECTED, Event, WorldState)

MAX_TEXT = 500      # operator messages are kept as data, truncated; never instructions


def _ship_events(prev: Optional[WorldState], cur: WorldState, t: float) -> list[Event]:
    out: list[Event] = []
    before = {s.call_id: s for s in prev.ships} if prev else {}
    for s in cur.ships:
        p = before.get(s.call_id)
        arrived_now = s.status != SHIP_EXPECTED and (p is None or p.status == SHIP_EXPECTED)
        if arrived_now:
            out.append(Event(EV_ARRIVAL, s.call_id, t, {
                "actual_arrival": s.actual_arrival, "scheduled_eta": s.scheduled_eta,
                "loa_m": s.loa_m, "draft_m": s.draft_m}))
        if s.berth and (p is None or p.berth != s.berth):
            out.append(Event(EV_BERTHED, s.call_id, t, {"berth": s.berth}))
        if s.status == SHIP_DEPARTED and (p is None or p.status != SHIP_DEPARTED):
            out.append(Event(EV_DEPARTED, s.call_id, t, {
                "berth": s.berth, "departed_at": s.departed_at, "actual_arrival": s.actual_arrival}))
    return out


def _equipment_events(prev: Optional[WorldState], cur: WorldState, t: float) -> list[Event]:
    out: list[Event] = []
    pairs = ((prev.cranes if prev else (), cur.cranes, EV_CRANE_DOWN, EV_CRANE_BACK, "crane_id"),
             (prev.trucks if prev else (), cur.trucks, EV_TRUCK_DOWN, EV_TRUCK_BACK, "truck_id"))
    for before_list, now_list, down, back, key in pairs:
        before = {getattr(e, key): e for e in before_list}
        for e in now_list:
            eid = getattr(e, key)
            was_down = eid in before and before[eid].status == EQ_OUT_OF_SERVICE
            is_down = e.status == EQ_OUT_OF_SERVICE
            if is_down and not was_down:
                out.append(Event(down, eid, t, {"was_serving": before[eid].call_id if eid in before else None}))
            elif was_down and not is_down:
                out.append(Event(back, eid, t, {}))
    return out


def _hold_events(prev: Optional[WorldState], cur: WorldState, t: float) -> list[Event]:
    out: list[Event] = []
    before = {h.container_id: h for h in prev.holds} if prev else {}
    call_of = {c.container_id: call for call, rows in cur.manifests.items() for c in rows}
    for h in cur.holds:
        p = before.get(h.container_id)
        detail = {"hold_since": h.hold_since, "call_id": call_of.get(h.container_id)}
        if p is None and not h.released:
            out.append(Event(EV_CUSTOMS_HOLD, h.container_id, t, detail))
        if h.released and (p is None or not p.released):
            out.append(Event(EV_HOLD_RELEASED, h.container_id, t, detail))
    return out


def _anomaly_events(prev: Optional[WorldState], cur: WorldState, t: float) -> list[Event]:
    seen = set(prev.anomalies) if prev else set()
    return [Event(EV_ANOMALY, "validation", t, {"message": a}) for a in cur.anomalies if a not in seen]


def _optional(module: str):
    """Import a module another task builds, or None if it doesn't exist yet.
    A module that exists but fails to import raises, so the caller reports it."""
    import importlib
    import importlib.util
    import sys
    if module in sys.modules:
        return sys.modules[module]
    if importlib.util.find_spec(module) is None:
        return None
    return importlib.import_module(module)


def _hook_events(cur: WorldState, t: float) -> list[Event]:
    """Events from other tasks' modules, once they exist (YARD-4 stall detector, LLM-6 inbox)."""
    out: list[Event] = []
    try:
        stall = _optional("agents.stall")             # YARD-4: detect(state) -> list[Event]
    except Exception as e:
        stall = None
        out.append(Event(EV_ANOMALY, "stall_detector", t, {"message": f"import failed: {type(e).__name__}"}))
    if stall is not None:
        try:
            out.extend(stall.detect(cur))
        except Exception as e:
            out.append(Event(EV_ANOMALY, "stall_detector", t, {"message": f"{type(e).__name__}: {e}"[:120]}))
    try:
        inbox = _optional("core.inbox")               # LLM-6: poll() -> list[str], new messages only
    except Exception as e:
        inbox = None
        out.append(Event(EV_ANOMALY, "inbox", t, {"message": f"import failed: {type(e).__name__}"}))
    if inbox is not None:
        try:
            for msg in inbox.poll():
                out.append(Event(EV_OPERATOR_MESSAGE, "operator", t, {"text": str(msg)[:MAX_TEXT]}))
        except Exception as e:
            out.append(Event(EV_ANOMALY, "inbox", t, {"message": f"{type(e).__name__}: {e}"[:120]}))
    return out


def detect_events(prev: Optional[WorldState], cur: WorldState) -> list[Event]:
    """What changed between two snapshots (CORE-3 interface). Never raises."""
    t = hours(cur)
    events: list[Event] = []
    for part in (_ship_events, _equipment_events, _hold_events, _anomaly_events):
        try:
            events.extend(part(prev, cur, t))
        except Exception as e:
            events.append(Event(EV_ANOMALY, "event_detector", t,
                                {"message": f"{part.__name__}: {type(e).__name__}: {e}"[:120]}))
    events.extend(_hook_events(cur, t))
    return events
