"""World-state builder (CORE-3).

One clean, read-only snapshot of the terminal per tick, so agents never call
read tools themselves:

    from core.state import build_state
    state = build_state(api)            # WorldState (frozen)

Reads only the toolkit's read tools (get_current_time, get_vessel_schedule,
get_equipment_status, get_yard_status, get_customs_status, get_manifest) plus
static reference data. Never the workbook or answer key (GROUND_RULES §3).

Never crashes the run: if a read tool returns an error, that part of the
snapshot is taken from the previous snapshot and an anomaly is recorded.

Manifests are read once per call and cached, because a manifest never changes
and get_manifest is slow. A container's `yard_block` in the cache is therefore
only as fresh as the last read: call `invalidate(api, call_id)` after placing
that call's boxes (CORE-4's read-after-write does this), or pass
`refresh_manifests=True`.

Every snapshot goes through YARD-5's `core.validate.validate(state)` once that
module exists; until then validation is a no-op.
"""
from __future__ import annotations

import weakref
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any, Optional

from core.skeleton import BERTHS          # static reference data (berth sizes, bays)
from core.types import (EQ_AVAILABLE, EQ_OUT_OF_SERVICE, EQ_WORKING, Block, Container,
                        Crane, Hold, Ship, Truck, WorldState)

# The toolkit's scenario window: 5 days from midnight before the first ETA.
SCENARIO_DAYS = 5


# --------------------------------------------------------------------------
# Parsing helpers (tolerant: bad values become None, never raise)
# --------------------------------------------------------------------------

def _dt(v: Any) -> Optional[datetime]:
    if v in (None, ""):
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except ValueError:
        return None


def _f(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _rail_key(crane_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in crane_id if ch.isdigit())
    return (int(digits) if digits else 10**6, crane_id)


def start_of(state: WorldState) -> datetime:
    """Scenario start (the toolkit's window is SCENARIO_DAYS long)."""
    return state.scenario_end - timedelta(days=SCENARIO_DAYS)


def hours(state: WorldState, when: Optional[datetime] = None) -> float:
    """Simulated hours since scenario start (Event.t, Decision.t)."""
    return round(((when or state.now) - start_of(state)).total_seconds() / 3600, 3)


# --------------------------------------------------------------------------
# Converters: raw tool output -> shared types
# --------------------------------------------------------------------------

def ship_from(r: dict) -> Ship:
    return Ship(call_id=str(r.get("call_id")), vessel=str(r.get("vessel") or ""),
                loa_m=_f(r.get("loa_m")), draft_m=_f(r.get("draft_m")),
                scheduled_eta=_dt(r.get("scheduled_eta")), actual_arrival=_dt(r.get("actual_arrival")),
                status=str(r.get("status")), berth=r.get("berth") or None,
                cranes=tuple(r.get("cranes") or ()), trucks=_i(r.get("trucks")),
                discharge_remaining=_i(r.get("discharge_remaining"), -1),     # -1 = unreadable (YARD-5 fixes)
                load_remaining=_i(r.get("load_remaining"), -1), departed_at=_dt(r.get("departed_at")))


def _equipment_status(text: str) -> tuple[str, Optional[str]]:
    """'available' | 'out of service' | 'working P01-001' | 'serving P01-001' -> (status, call_id)."""
    text = str(text or "")
    if text == "out of service":
        return EQ_OUT_OF_SERVICE, None
    for prefix in ("working ", "serving "):
        if text.startswith(prefix):
            return EQ_WORKING, text[len(prefix):]
    if text == "available":
        return EQ_AVAILABLE, None
    return text, None          # unknown wording: kept as-is, YARD-5 flags it


def equipment_from(rows: list[dict]) -> tuple[tuple[Crane, ...], tuple[Truck, ...]]:
    cranes, trucks = [], []
    for r in rows:
        status, call = _equipment_status(r.get("status"))
        if r.get("type") == "quay crane":
            cranes.append(Crane(str(r.get("equipment_id")), status, call, _i(r.get("bay"), -1)))
        else:
            trucks.append(Truck(str(r.get("equipment_id")), status, call))
    cranes.sort(key=lambda c: _rail_key(c.crane_id))
    trucks.sort(key=lambda t: t.truck_id)
    return tuple(cranes), tuple(trucks)


def block_from(r: dict) -> Block:
    return Block(str(r.get("block_id")), str(r.get("block_type")), _i(r.get("used"), -1), _i(r.get("slots")))


def hold_from(r: dict) -> Hold:
    return Hold(str(r.get("container_id")), _dt(r.get("hold_since")), bool(r.get("released")))


def container_from(r: dict, call_id: str) -> Container:
    size = r.get("size_ft")
    return Container(container_id=str(r.get("container_id")), call_id=call_id,
                     direction=str(r.get("direction")), size_ft=None if size is None else _i(size),
                     cargo_type=str(r.get("cargo_type")), imdg_class=_f(r.get("imdg_class")),
                     gross_weight_t=_f(r.get("gross_weight_t")), yard_block=r.get("yard_block") or None)


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------

class StateBuilder:
    """Builds snapshots for one simulator instance and keeps the manifest cache."""

    def __init__(self, api: Any):
        self.api = api
        self.manifests: dict[str, tuple[Container, ...]] = {}
        self.prev: Optional[WorldState] = None

    def _read(self, tool: str, args: Optional[dict], anomalies: list[str]) -> Any:
        try:
            out = self.api.call(tool, args or {})
        except Exception as e:          # api.call should never raise; be safe anyway
            out = {"error": f"{type(e).__name__}: {e}"}
        if isinstance(out, dict) and "error" in out:
            anomalies.append(f"read tool {tool} failed: {str(out['error'])[:80]}")
            return None
        return out

    def invalidate(self, call_id: Optional[str] = None) -> None:
        """Forget cached manifests (one call, or all) so the next build re-reads them."""
        if call_id is None:
            self.manifests.clear()
        else:
            self.manifests.pop(call_id, None)

    def build(self, refresh_manifests: bool = False) -> WorldState:
        anomalies: list[str] = []
        prev = self.prev

        clock = self._read("get_current_time", None, anomalies)
        schedule = self._read("get_vessel_schedule", None, anomalies)
        equipment = self._read("get_equipment_status", None, anomalies)
        yard = self._read("get_yard_status", None, anomalies)
        customs = self._read("get_customs_status", None, anomalies)

        now = _dt(clock.get("now")) if isinstance(clock, dict) else None
        end = _dt(clock.get("scenario_end")) if isinstance(clock, dict) else None
        if now is None or end is None:
            if prev is None:
                raise RuntimeError("cannot read the simulator clock on the first snapshot")
            now, end = prev.now, prev.scenario_end

        def parse(rows, conv, fallback, what):
            if not isinstance(rows, list):
                return fallback
            out = []
            for r in rows:
                try:
                    out.append(conv(r))
                except Exception as e:      # one bad row never sinks the snapshot
                    anomalies.append(f"bad {what} row skipped: {type(e).__name__}")
            return tuple(out)

        ships = parse(schedule, ship_from, prev.ships if prev else (), "schedule")
        if isinstance(equipment, list):
            cranes, trucks = equipment_from(equipment)
        else:
            cranes, trucks = (prev.cranes, prev.trucks) if prev else ((), ())
        blocks = parse(yard, block_from, prev.blocks if prev else (), "yard")
        holds = parse(customs, hold_from, prev.holds if prev else (), "customs")

        if refresh_manifests:
            self.manifests.clear()
        for s in ships:
            if s.call_id in self.manifests:
                continue
            rows = self._read("get_manifest", {"call_id": s.call_id}, anomalies)
            if isinstance(rows, list):
                self.manifests[s.call_id] = parse(rows, lambda r, c=s.call_id: container_from(r, c),
                                                  (), "manifest")

        state = WorldState(now=now, scenario_end=end, ships=ships, cranes=cranes, trucks=trucks,
                           blocks=blocks, holds=holds,
                           manifests=MappingProxyType({s.call_id: self.manifests.get(s.call_id, ())
                                                       for s in ships}),
                           berths=BERTHS, anomalies=tuple(anomalies))
        state = _validate(state)
        self.prev = state
        return state


def _validate(state: WorldState) -> WorldState:
    """YARD-5 hook: core.validate.validate(state) -> (state, anomalies), once it exists."""
    import importlib
    import importlib.util
    try:
        import sys
        v = sys.modules.get("core.validate")
        if v is None:
            if importlib.util.find_spec("core.validate") is None:   # not built yet (YARD-5)
                return state
            v = importlib.import_module("core.validate")
        new_state, found = v.validate(state)
    except Exception as e:                       # validation must never crash the run
        return _with_anomalies(state, [f"validation failed: {type(e).__name__}: {e}"[:120]])
    return _with_anomalies(new_state, list(found))


def _with_anomalies(state: WorldState, extra: list[str]) -> WorldState:
    if not extra:
        return state
    from dataclasses import replace
    return replace(state, anomalies=tuple(dict.fromkeys((*state.anomalies, *extra))))


# One builder per simulator instance, so build_state(api) keeps its manifest cache.
_BUILDERS: "weakref.WeakKeyDictionary[Any, StateBuilder]" = weakref.WeakKeyDictionary()


def builder_for(api: Any) -> StateBuilder:
    b = _BUILDERS.get(api)
    if b is None:
        b = _BUILDERS[api] = StateBuilder(api)
    return b


def build_state(api: Any, refresh_manifests: bool = False) -> WorldState:
    """Read the terminal into one frozen WorldState (CORE-3 interface)."""
    return builder_for(api).build(refresh_manifests=refresh_manifests)


def invalidate(api: Any, call_id: Optional[str] = None) -> None:
    """Drop cached manifests for one call (or all) after placing its boxes."""
    builder_for(api).invalidate(call_id)
