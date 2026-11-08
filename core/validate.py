"""Input validation layer (YARD-5): bad or fake data never crashes a run or tricks an agent.

    from core.validate import validate
    state, anomalies = validate(state)      # CORE-3 calls this on every snapshot

Every snapshot is checked before any agent sees it. Each problem gets the most restrictive
safe handling, so agents never act on a number we can't trust, and is reported once per
scenario as one plain sentence ("Bad data: ... Safest default taken: ..."). The loop
escalates those sentences. Trust order: simulator > static data > operator messages > LLM:
a value the simulator gives is only overridden when it is impossible, and then by static
reference data (berths, the crane rail, the manifest) or by the last good snapshot.

What is checked, and what is done about it:
  ships        duplicate call ids (first row kept); length/draft missing, NaN, <= 0 or absurd
               (set to unknown: never berthed); fits no berth (never berthed); remaining moves
               missing or negative (last good value, else the manifest count); unknown status or
               berth; ETA outside the 5-day window or arrival in the future (flagged); cranes on
               a ship that are not on the rail (dropped from the ship)
  cranes       duplicate ids; unknown status wording or a bay off the rail (treated as out of
               service: never moved, never assigned); working for a ship that doesn't exist
  trucks       duplicate ids; unknown status (out of service)
  yard blocks  duplicate ids; unknown block type (no free slots); used < 0, used > slots or
               slots <= 0 (clamped so free space is never overstated); capacity bigger than first
               seen (first seen kept)
  manifests    the same box id twice, on one ship or two (never placed: the simulator can't tell them apart); unknown cargo type (kept: agents never place
               it); an IMDG class on a box not marked IMDG (never placed: H refuses it, elsewhere may be unsafe);
               unknown direction, bad size or weight (flagged)
  holds        a hold with no start time (treated as held since now)
  clock        the clock past the scenario end (flagged)

It never raises: if a check itself fails, the snapshot passes on unchanged with an anomaly.
It never reads the workbook: only the snapshot and static reference data.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import timedelta
from types import MappingProxyType
from typing import Any, Optional

from core.skeleton import CRANE_SPLIT
from core.types import (BLOCK_TYPES, CARGO_IMDG, EQ_OUT_OF_SERVICE, EQ_STATUSES, KNOWN_CARGO, SHIP_STATUSES,
                        WorldState)

SCENARIO_DAYS = 5
MAX_LOA_M = 500.0          # nothing afloat is longer
MAX_DRAFT_M = 30.0
RAIL = (1, 80)             # quay crane rail bays (static: B1 1-30, B2 31-58, B3 59-80)
CRANES = tuple(c for split in CRANE_SPLIT.values() for c, _ in split)       # QC1..QC6 (static)
DIRECTIONS = ("import", "export")
PREFIX = "Bad data: "
CONFLICT = "conflicting (IMDG class, not marked IMDG)"   # not in KNOWN_CARGO: never placed


class _Memory:
    def __init__(self) -> None:
        self.scenario: Optional[tuple] = None
        self.reported: set[str] = set()          # anomaly sentences already reported
        self.remaining: dict[str, tuple[int, int]] = {}   # call_id -> last good (discharge, load)
        self.slots: dict[str, int] = {}          # block -> capacity when first seen


_mem = _Memory()


def reset() -> None:
    global _mem
    _mem = _Memory()
    _rows_cache.clear()
    _shared_cache.clear()


def _q(v: Any, n: int = 30) -> str:
    """Untrusted text, quoted and cut for a log line."""
    s = str(v)
    return repr(s if len(s) <= n else s[:n] + "…")


def _bad_number(x: Any) -> bool:
    return x is None or not isinstance(x, (int, float)) or isinstance(x, bool) or \
        (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


class _Report:
    def __init__(self) -> None:
        self.found: list[str] = []

    def add(self, problem: str, handling: str) -> None:
        msg = f"{PREFIX}{problem} Safest default taken: {handling}"
        if msg not in _mem.reported:
            _mem.reported.add(msg)
            self.found.append(msg)


# --------------------------------------------------------------------------- parts

def _ships(state: WorldState, r: _Report) -> tuple:
    berths = {b.berth_id: b for b in state.berths}
    rail = {c.crane_id for c in state.cranes} or set(CRANES)
    start = state.scenario_end - timedelta(days=SCENARIO_DAYS)
    out, seen = [], set()
    for s in state.ships:
        cid = s.call_id
        if cid in seen:
            r.add(f"ship {_q(cid)} appears twice in the schedule.", "the second row is ignored.")
            continue
        seen.add(cid)
        loa, draft = s.loa_m, s.draft_m
        if _bad_number(loa) or loa <= 0 or loa > MAX_LOA_M or _bad_number(draft) or draft <= 0 or draft > MAX_DRAFT_M:
            r.add(f"{_q(cid)} has no valid length or draft (LOA {_q(loa, 12)}, draft {_q(draft, 12)}).",
                  "its size is treated as unknown, so it is never berthed.")
            loa = draft = None
        elif berths and not any(b.fits(loa, draft) for b in berths.values()):
            r.add(f"{_q(cid)} (LOA {loa:g} m, draft {draft:g} m) fits no berth.", "it is never berthed.")
        dis, load = s.discharge_remaining, s.load_remaining
        bad_dis = _bad_number(dis) or dis < 0
        bad_load = _bad_number(load) or load < 0
        if bad_dis or bad_load:
            last = _mem.remaining.get(cid)
            rows = state.manifests.get(cid, ())
            if last is not None:
                fix, src = last, "the last good snapshot's numbers are used"
            else:
                fix = (sum(1 for c in rows if c.direction == "import"), sum(1 for c in rows if c.direction == "export"))
                src = "its manifest counts are used"
            r.add(f"{_q(cid)} has unreadable remaining moves (discharge {_q(dis, 12)}, load {_q(load, 12)}).",
                  f"{src}; the simulator still decides when it is done.")
            dis = fix[0] if bad_dis else dis
            load = fix[1] if bad_load else load
        else:
            _mem.remaining[cid] = (dis, load)
        if s.status not in SHIP_STATUSES:
            r.add(f"{_q(cid)} has an unknown status {_q(s.status, 20)}.", "no agent acts on it until the status is known.")
        berth = s.berth
        if berth is not None and berth not in berths:
            r.add(f"{_q(cid)} is at an unknown berth {_q(berth, 10)}.", "no agent acts on it at that berth.")
        if s.scheduled_eta is not None and not (start - timedelta(days=1) <= s.scheduled_eta <= state.scenario_end):
            r.add(f"{_q(cid)} has an ETA outside the scenario window.", "it is planned only once it actually arrives.")
        if s.actual_arrival is not None and s.actual_arrival > state.now:
            r.add(f"{_q(cid)} shows an arrival time in the future.", "its status from the simulator is trusted.")
        cranes = s.cranes
        ghost = [c for c in cranes if c not in rail]
        if ghost:
            r.add(f"{_q(cid)} lists cranes that are not on the rail: {', '.join(_q(c, 8) for c in ghost[:3])}.",
                  "they are dropped from the ship.")
            cranes = tuple(c for c in cranes if c in rail)
        trucks = s.trucks if not _bad_number(s.trucks) and s.trucks >= 0 else 0
        if trucks != s.trucks:
            r.add(f"{_q(cid)} has an unreadable truck count {_q(s.trucks, 8)}.", "it is counted as 0.")
        new = replace(s, loa_m=loa, draft_m=draft, discharge_remaining=dis, load_remaining=load,
                      cranes=cranes, trucks=trucks)
        out.append(new)
    return tuple(out)


def _cranes(state: WorldState, ships: tuple, r: _Report) -> tuple:
    calls = {s.call_id for s in ships}
    out, seen = [], set()
    for c in state.cranes:
        if c.crane_id in seen:
            r.add(f"crane {_q(c.crane_id, 8)} appears twice.", "the second row is ignored.")
            continue
        seen.add(c.crane_id)
        status = c.status
        if status not in EQ_STATUSES:
            r.add(f"crane {_q(c.crane_id, 8)} has an unknown status {_q(status, 20)}.",
                  "it is treated as out of service: never moved or assigned.")
            status = EQ_OUT_OF_SERVICE
        if _bad_number(c.bay) or not RAIL[0] <= c.bay <= RAIL[1]:
            r.add(f"crane {_q(c.crane_id, 8)} reports a bay off the rail ({_q(c.bay, 8)}).",
                  "it is treated as out of service: never moved or assigned.")
            status = EQ_OUT_OF_SERVICE
        if c.call_id and c.call_id not in calls:
            r.add(f"crane {_q(c.crane_id, 8)} works for an unknown ship {_q(c.call_id)}.",
                  "it is left where it is and not reassigned by us.")
        out.append(replace(c, status=status))
    return tuple(out)


def _trucks(state: WorldState, r: _Report) -> tuple:
    out, seen = [], set()
    for t in state.trucks:
        if t.truck_id in seen:
            r.add(f"truck {_q(t.truck_id, 10)} appears twice.", "the second row is ignored.")
            continue
        seen.add(t.truck_id)
        if t.status not in EQ_STATUSES:
            r.add(f"truck {_q(t.truck_id, 10)} has an unknown status {_q(t.status, 20)}.",
                  "it is treated as out of service.")
            t = replace(t, status=EQ_OUT_OF_SERVICE)
        out.append(t)
    return tuple(out)


def _blocks(state: WorldState, r: _Report) -> tuple:
    out, seen = [], set()
    for b in state.blocks:
        if b.block_id in seen:
            r.add(f"yard block {_q(b.block_id, 8)} appears twice.", "the second row is ignored.")
            continue
        seen.add(b.block_id)
        slots, used = b.slots, b.used
        if _bad_number(slots) or slots <= 0:
            r.add(f"yard block {_q(b.block_id, 8)} reports no valid capacity ({_q(slots, 8)}).",
                  "it is treated as full.")
            slots = max(0, int(used)) if not _bad_number(used) else 0
            used = slots
        first = _mem.slots.setdefault(b.block_id, slots)
        if slots > first:
            r.add(f"yard block {_q(b.block_id, 8)} capacity grew from {first} to {slots} slots.",
                  f"the smaller capacity ({first}) is used.")
            slots = first
        elif slots < first:
            r.add(f"yard block {_q(b.block_id, 8)} capacity shrank from {first} to {slots} slots.",
                  f"the smaller capacity ({slots}) is used.")
            _mem.slots[b.block_id] = slots
        if _bad_number(used) or used < 0:
            r.add(f"yard block {_q(b.block_id, 8)} reports an unreadable fill ({_q(used, 8)}).",
                  "it is treated as full.")
            used = slots
        elif used > slots:
            r.add(f"yard block {_q(b.block_id, 8)} reports more boxes ({used}) than slots ({slots}).",
                  "it is treated as full.")
            used = slots
        btype = b.block_type
        if btype not in BLOCK_TYPES:
            r.add(f"yard block {_q(b.block_id, 8)} has an unknown type {_q(btype, 20)}.",
                  "it is treated as full, so nothing is placed there.")
            used = slots
        out.append(replace(b, used=used, slots=slots))
    return tuple(out)


def _clean_rows(rows: tuple) -> tuple[tuple, frozenset, dict]:
    """One ship's manifest on its own: (rows to use, box ids, counts of problems)."""
    keep, ids, unknown = [], set(), {}
    counts = {"twice": 0, "imdg": 0, "dir": 0, "size": 0}
    seen_once: dict = {}
    for c in rows:
        seen_once[c.container_id] = seen_once.get(c.container_id, 0) + 1
    twice = {k for k, n in seen_once.items() if n > 1}
    changed = bool(twice)
    counts["twice"] = len(twice)
    for c in rows:
        ids.add(c.container_id)
        if c.container_id in twice:
            continue
        if c.imdg_class is not None and not _bad_number(c.imdg_class) and c.cargo_type != CARGO_IMDG:
            c = replace(c, cargo_type=CONFLICT)       # not placed: H would refuse it, elsewhere may be unsafe
            counts["imdg"] += 1
            changed = True
        elif c.cargo_type not in KNOWN_CARGO:
            unknown[str(c.cargo_type)] = unknown.get(str(c.cargo_type), 0) + 1
        if c.direction not in DIRECTIONS:
            counts["dir"] += 1
        if (c.size_ft is not None and (_bad_number(c.size_ft) or c.size_ft <= 0)) or \
                (c.gross_weight_t is not None and (_bad_number(c.gross_weight_t) or c.gross_weight_t < 0)):
            counts["size"] += 1
        keep.append(c)
    counts["unknown"] = unknown
    return (tuple(keep) if changed else rows), frozenset(ids), counts


_rows_cache: dict = {}          # call_id -> (rows object, _clean_rows result); same rows = same answer
_shared_cache: dict = {}        # box ids listed on more than one ship, for the last set of manifests


def _manifests(state: WorldState, ships: tuple, r: _Report) -> MappingProxyType:
    out: dict = {}
    cleaned = []
    count: dict = {}
    for s in ships:
        rows = state.manifests.get(s.call_id, ())
        hit = _rows_cache.get(s.call_id)
        if hit is None or hit[0] is not rows:
            hit = (rows, _clean_rows(rows))
            _rows_cache[s.call_id] = hit
        cleaned.append((s, hit[1]))
    key = tuple((hash(ids), len(ids)) for _, (_, ids, _) in cleaned)   # same box ids as last time: same answer
    if _shared_cache.get("key") != key:
        for _, (_, ids, _) in cleaned:
            for i in ids:
                count[i] = count.get(i, 0) + 1
        _shared_cache.update(key=key, shared={i for i, n in count.items() if n > 1})
    shared = _shared_cache["shared"]
    for s, (keep, ids, n) in cleaned:
        cid = _q(s.call_id)
        clash = ids & shared
        if clash:              # the same box on two ships: the simulator can't tell them apart, so nobody moves it
            r.add(f"{len(clash)} boxes of {cid} are also listed on another ship.",
                  "those boxes are not placed by anyone, so the ship waits for them and is escalated.")
            keep = tuple(c for c in keep if c.container_id not in clash)
        if n["twice"]:
            r.add(f"{n['twice']} boxes of {cid} are listed twice.",
                  "those boxes are not placed, so the ship waits for them and is escalated.")
        for kind, k in sorted(n["unknown"].items()):
            r.add(f"{k} boxes of {cid} have an unknown cargo type {_q(kind, 20)}.",
                  "they are never placed, so the ship waits for them and is escalated.")
        if n["imdg"]:
            r.add(f"{n['imdg']} boxes of {cid} carry an IMDG class but are not marked IMDG.",
                  "they are not placed (block H only takes boxes marked IMDG, and anywhere else could be "
                  "unsafe), so the ship waits for them and is escalated.")
        if n["dir"]:
            r.add(f"{n['dir']} boxes of {cid} have an unknown direction.", "no agent moves them.")
        if n["size"]:
            r.add(f"{n['size']} boxes of {cid} have an unreadable size or weight.", "they are handled as listed.")
        out[s.call_id] = keep
    return MappingProxyType(out)


def _holds(state: WorldState, r: _Report) -> tuple:
    out = []
    for h in state.holds:
        if h.hold_since is None:
            r.add(f"customs hold on {_q(h.container_id, 15)} has no start time.", "the box is treated as held.")
            h = replace(h, hold_since=state.now)
        out.append(h)
    return tuple(out)


# --------------------------------------------------------------------------- entry point

def validate(state: WorldState) -> tuple[WorldState, list[str]]:
    """(cleaned state, new anomaly sentences). Each sentence is reported once per scenario."""
    r = _Report()
    try:
        key = (state.scenario_end, tuple(sorted({str(s.call_id) for s in state.ships})))
        if _mem.scenario != key:
            reset()
            _mem.scenario = key
        if state.now > state.scenario_end:
            r.add("the clock is past the scenario end.", "the run stops at the end as usual.")
        ships = _ships(state, r)
        cranes = _cranes(state, ships, r)
        trucks = _trucks(state, r)
        blocks = _blocks(state, r)
        manifests = _manifests(state, ships, r)
        holds = _holds(state, r)
        return replace(state, ships=ships, cranes=cranes, trucks=trucks, blocks=blocks,
                       manifests=manifests, holds=holds), r.found
    except Exception as e:                      # never crash: pass the snapshot on as it came
        return state, [f"{PREFIX}validation itself failed ({type(e).__name__}). Safest default taken: the "
                       f"snapshot is used as read."]
