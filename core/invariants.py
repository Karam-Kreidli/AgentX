"""Invariant monitor (SAFE-2): catch anything the guard missed, on the live state, every tick.

    from core.invariants import check_state
    breaches = check_state(state)      # core/loop.py calls this after every tick

Written from the hard-rules list in GROUND_RULES.md §2, NOT from guard.py, so a blind spot
in one is not copied into the other. This module never imports the guard.

What it checks
--------------
Hard rules on the live snapshot. A breach makes the loop log CRITICAL and run the next
tick on the rules-only agent:
    rail_order       all six cranes, idle and broken included, in strict order QC1 < ... < QC6
    crane_spacing    cranes on berthed/working ships at least 2 bays apart
    imdg_block       every placed IMDG box is in the IMDG block (H), and H holds only IMDG boxes
    reefer_block     every placed reefer box is in a reefer block (I or J)
    berth_*          every ship at a berth has arrived, fits it (length >= LOA + 20 m,
                     depth >= draft) and has it to itself
    crane_two_ships  no crane is listed on two ships
    crane_in_berth   every assigned crane stands inside its ship's berth bays
    unverifiable     something a rule refers to that the snapshot doesn't know (refuse to assume)

The simulator's own record: actions.jsonl in the run folder, the file the organizers score:
    simulator_violation   every new VIOLATION line (customs-held box released, and anything
                          else the simulator flags). This is how "a customs-held box is never
                          released" is watched: a released box can't be seen in the snapshot.
    simulator_refusal     every new tool_error line. The toolkit README says "Attempting a
                          forbidden action counts as a violation".

Soft warnings, logged once and with no rules-only switch: broken equipment still listed
on a ship, trucks parked on a ship that isn't being worked, export or unknown-cargo boxes
in the yard, block counts outside 0..slots, equipment status that disagrees with the ships.

Behaviour
---------
- A breach is reported once when it appears, and again if it clears and comes back. So one
  misplaced box doesn't keep the whole run in rules-only mode.
- Every new breach is also logged as an escalation. It never blocks the run.
- Never raises. An internal error comes back as a breach, which is the safe direction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core import log
from core.types import (BLOCK_IMDG, BLOCK_REEFER, CARGO_IMDG, CARGO_REEFER, EQ_OUT_OF_SERVICE,
                        KNOWN_CARGO, SHIP_BERTHED, SHIP_SAILING, SHIP_WORKING, WorldState)

MIN_GAP_BAYS = 2              # GROUND_RULES §2: two working cranes at least 2 bays apart
LENGTH_MARGIN_M = 20          # GROUND_RULES §2: berth length >= ship + 20 m
AT_BERTH = (SHIP_BERTHED, SHIP_WORKING, SHIP_SAILING)
CRANES_WORKING = (SHIP_BERTHED, SHIP_WORKING)    # cranes on these ships are "working cranes"
SIM_LOG = "actions.jsonl"
AGENT = "invariants"


@dataclass(frozen=True)
class Breach:
    """One broken invariant. `str(b)` is what the loop logs."""
    rule: str
    subject: str
    detail: str
    key: str = ""                 # identity for de-duplication; default rule|subject

    @property
    def ident(self) -> str:
        return self.key or f"{self.rule}|{self.subject}"

    def __str__(self) -> str:
        return f"[{self.rule}] {self.subject}: {self.detail}"


# --------------------------------------------------------------------------
# Snapshot checks (pure: state in, findings out)
# --------------------------------------------------------------------------

def _rail_number(crane_id: str) -> Optional[int]:
    digits = "".join(ch for ch in str(crane_id) if ch.isdigit())
    return int(digits) if digits else None


def _is_bay(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _cranes(state: WorldState, hard: list, soft: list) -> None:
    bays = {c.crane_id: c.bay for c in state.cranes}

    # Rail order over ALL cranes, idle and broken included.
    order = []
    for c in state.cranes:
        n = _rail_number(c.crane_id)
        if n is None or not _is_bay(c.bay):
            hard.append(Breach("unverifiable", c.crane_id, "crane has no rail number or bay position"))
        else:
            order.append((n, c.crane_id, c.bay))
    order.sort()
    for (_, left, lb), (_, right, rb) in zip(order, order[1:]):
        if not lb < rb:
            hard.append(Breach("rail_order", f"{left}/{right}",
                               f"{left} at bay {lb} is not left of {right} at bay {rb}"))

    # Which ship each crane is listed on (ships at a berth only).
    on: dict[str, str] = {}
    berths = {b.berth_id: b for b in state.berths}
    for s in state.ships:
        if s.status not in CRANES_WORKING:
            continue
        berth = berths.get(s.berth) if s.berth else None
        for cid in s.cranes:
            if cid in on and on[cid] != s.call_id:
                hard.append(Breach("crane_two_ships", cid, f"listed on both {on[cid]} and {s.call_id}"))
            on.setdefault(cid, s.call_id)
            if cid not in bays or not _is_bay(bays[cid]):
                hard.append(Breach("unverifiable", cid, f"crane on {s.call_id} is not in the equipment status"))
                continue
            if berth is None:
                continue                      # reported by the berth check
            if not berth.start_bay <= bays[cid] <= berth.end_bay:
                hard.append(Breach("crane_in_berth", cid,
                                   f"at bay {bays[cid]}, outside {berth.berth_id} (bays {berth.start_bay}-"
                                   f"{berth.end_bay}) where {s.call_id} is"))

    # Spacing among working cranes.
    working = sorted((bays[c], c) for c in on if c in bays and _is_bay(bays[c]))
    for (b1, c1), (b2, c2) in zip(working, working[1:]):
        if b2 - b1 < MIN_GAP_BAYS:
            hard.append(Breach("crane_spacing", f"{c1}/{c2}",
                               f"working cranes at bays {b1} and {b2}, under {MIN_GAP_BAYS} bays apart"))

    # Soft: equipment status vs. the ships' lists.
    for c in state.cranes:
        if c.crane_id in on and c.status == EQ_OUT_OF_SERVICE:
            soft.append((f"broken_listed|{c.crane_id}",
                         f"{c.crane_id} is out of service but still listed on {on[c.crane_id]}."))
        elif c.call_id and on.get(c.crane_id) not in (None, c.call_id):
            soft.append((f"crane_mismatch|{c.crane_id}",
                         f"{c.crane_id} reports work on {c.call_id} but is listed on {on[c.crane_id]}."))


def _berths(state: WorldState, hard: list) -> None:
    berths = {b.berth_id: b for b in state.berths}
    occupant: dict[str, str] = {}
    for s in state.ships:
        if s.status not in AT_BERTH:
            continue
        if not s.berth:
            hard.append(Breach("berth_missing", s.call_id, f"is '{s.status}' but has no berth"))
            continue
        b = berths.get(s.berth)
        if b is None:
            hard.append(Breach("unverifiable", s.call_id, f"berth {s.berth!r} is not in the reference data"))
            continue
        if s.actual_arrival is None:
            hard.append(Breach("berth_before_arrival", s.call_id, f"is at {s.berth} but has no arrival time"))
        if s.loa_m is None or s.draft_m is None:
            hard.append(Breach("unverifiable", s.call_id, "has no valid length or draft"))
        else:
            if b.length_m < s.loa_m + LENGTH_MARGIN_M:
                hard.append(Breach("berth_too_short", s.call_id,
                                   f"LOA {s.loa_m} m needs {s.loa_m + LENGTH_MARGIN_M} m; {s.berth} is {b.length_m} m"))
            if s.draft_m > b.max_draft_m:
                hard.append(Breach("berth_too_shallow", s.call_id,
                                   f"draft {s.draft_m} m is deeper than {s.berth} ({b.max_draft_m} m)"))
        if s.berth in occupant and occupant[s.berth] != s.call_id:
            hard.append(Breach("berth_shared", s.berth, f"{occupant[s.berth]} and {s.call_id} are both at {s.berth}"))
        occupant.setdefault(s.berth, s.call_id)


def _trucks(state: WorldState, soft: list) -> None:
    for s in state.ships:
        if s.trucks > 0 and s.status not in CRANES_WORKING:
            soft.append((f"trucks_parked|{s.call_id}",
                         f"{s.trucks} trucks are assigned to {s.call_id}, which is '{s.status}', not being worked."))
    counted = sum(max(0, s.trucks) for s in state.ships)
    serving = sum(1 for t in state.trucks if t.call_id)
    broken = sum(1 for t in state.trucks if t.status == EQ_OUT_OF_SERVICE)
    if counted > serving + broken:
        soft.append(("trucks_count", f"ships count {counted} trucks but only {serving} are serving a ship."))


def _blocks(state: WorldState, soft: list) -> None:
    for b in state.blocks:
        if b.used < 0 or b.used > b.slots:
            soft.append((f"block_count|{b.block_id}", f"block {b.block_id} reports {b.used}/{b.slots} slots used."))


def _manifest_findings(rows: tuple, types: dict) -> tuple[list, list]:
    """Yard rules for one ship's boxes: (hard breaches, soft warnings)."""
    hard, soft = [], []
    for c in rows:
        block = c.yard_block
        if not block:
            continue
        btype = types.get(block)
        if btype is None:
            hard.append(Breach("unverifiable", c.container_id, f"is in block {block!r}, which the yard status doesn't list"))
            continue
        if c.cargo_type == CARGO_IMDG and btype != BLOCK_IMDG:
            hard.append(Breach("imdg_block", c.container_id, f"IMDG box is in block {block}, not the IMDG block"))
        elif c.cargo_type != CARGO_IMDG and btype == BLOCK_IMDG:
            hard.append(Breach("imdg_block", c.container_id, f"{c.cargo_type} box is in IMDG block {block}"))
        if c.cargo_type == CARGO_REEFER and btype != BLOCK_REEFER:
            hard.append(Breach("reefer_block", c.container_id, f"reefer box is in block {block}, not a reefer block"))
        if c.cargo_type not in KNOWN_CARGO:
            soft.append((f"unknown_cargo|{c.container_id}", f"{c.container_id} with unknown cargo type is in block {block}."))
        if c.direction != "import":
            soft.append((f"export_in_yard|{c.container_id}",
                         f"{c.direction} box {c.container_id} has yard block {block} (only imports are placed)."))
    return hard, soft


def _yard(state: WorldState, hard: list, soft: list, memo: Optional[dict] = None) -> None:
    types = {b.block_id: b.block_type for b in state.blocks}
    sig = tuple(sorted(types.items()))
    for call_id, rows in state.manifests.items():
        hit = memo.get(call_id) if memo is not None else None
        if hit is not None and hit[0] is rows and hit[1] == sig:   # same cached tuple: nothing changed
            h, s = hit[2], hit[3]
        else:
            h, s = _manifest_findings(rows, types)
            if memo is not None:
                memo[call_id] = (rows, sig, h, s)
        hard.extend(h)
        soft.extend(s)


def find_breaches(state: WorldState, memo: Optional[dict] = None) -> tuple[list[Breach], list[tuple[str, str]]]:
    """Every hard breach and soft warning in one snapshot. Pure apart from the optional memo."""
    hard: list[Breach] = []
    soft: list[tuple[str, str]] = []
    _cranes(state, hard, soft)
    _berths(state, hard)
    _yard(state, hard, soft, memo)
    _trucks(state, soft)
    _blocks(state, soft)
    return hard, soft


# --------------------------------------------------------------------------
# The simulator's own record: new VIOLATION / tool_error lines in actions.jsonl
# --------------------------------------------------------------------------

def _sim_findings(path: Path, offset: int) -> tuple[list[Breach], int]:
    """Read complete new lines from `offset`. Returns (breaches, new offset)."""
    size = path.stat().st_size
    if size < offset:                     # the file was restarted
        offset = 0
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    out: list[Breach] = []
    pos = offset
    for raw in data[: end + 1].splitlines(keepends=True):
        start, pos = pos, pos + len(raw)
        if b'"VIOLATION"' not in raw and b'"tool_error"' not in raw:
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        kind, at = rec.get("action"), rec.get("sim_time")
        if kind == "VIOLATION":
            extra = {k: v for k, v in rec.items() if k not in ("action", "type", "sim_time")}
            out.append(Breach("simulator_violation", str(rec.get("type")),
                              f"recorded by the simulator at {at}: {json.dumps(extra)[:160]}", key=f"sim|{start}"))
        elif kind == "tool_error":
            out.append(Breach("simulator_refusal", str(rec.get("tool")),
                              f"refused at {at}: {str(rec.get('error'))[:160]}", key=f"sim|{start}"))
    return out, offset + end + 1


# --------------------------------------------------------------------------
# The monitor: de-duplication, logging, never raising
# --------------------------------------------------------------------------

class Monitor:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.token: Any = None
        self.last_now: Any = None
        self.active: set[str] = set()     # hard breaches present in the last snapshot
        self.soft_seen: set[str] = set()
        self.memo: dict = {}
        self.sim_offset = 0

    def _new_run_check(self, state: WorldState) -> None:
        token = (log.stats().get("dir"), state.scenario_end)
        if token != self.token or (self.last_now is not None and state.now < self.last_now):
            self.reset()
            self.token = token
        self.last_now = state.now

    def _sim(self) -> list[Breach]:
        d = log.stats().get("dir")
        if not d:
            return []
        path = Path(d) / SIM_LOG
        if not path.exists():
            return []
        found, self.sim_offset = _sim_findings(path, self.sim_offset)
        return found

    def check(self, state: WorldState) -> list[Breach]:
        try:
            self._new_run_check(state)
            hard, soft = find_breaches(state, self.memo)
            current = {b.ident: b for b in hard}
            new = [b for k, b in current.items() if k not in self.active]
            self.active = set(current)
            new += self._sim()
            tick = log.current_tick()
            for b in new:
                log.escalation(AGENT, f"Live-state check found a rule breach: {b}.",
                               "the next tick runs on the rules-only agent", state=state)
            for key, msg in soft:
                if key not in self.soft_seen:
                    self.soft_seen.add(key)
                    log.note(tick if tick is not None else -1, state, AGENT, "warning", msg)
            return new
        except Exception as e:            # never raise: a failing monitor counts as a breach
            return [Breach("monitor_error", AGENT, f"{type(e).__name__}: {e}"[:200], key="monitor_error")]


_monitor = Monitor()


def check_state(state: WorldState) -> list[Breach]:
    """New hard breaches in this snapshot (and new simulator violations/refusals). Never raises."""
    return _monitor.check(state)


def reset() -> None:
    """Forget what was already reported (a new run resets automatically)."""
    _monitor.reset()
