"""Guard (SAFE-1): nothing that breaks a rule ever reaches the simulator.

    from guard import check
    ok, reason = check(action, state)

Written from GROUND_RULES.md §2 (the hard rules), not from agent code. The
guard only says yes or no: it never "fixes" an action, and it never raises.
Anything it cannot verify (unknown ship, crane, container, malformed args) is
refused. Changes to this file need approval from Members D and E.

Checks run in this order and stop at the first failure:
  1. crane rail order, including idle and broken cranes (QC1 < QC2 < ... < QC6)
  2. working cranes at least 2 bays apart
  3. IMDG boxes only in block H, and block H only takes IMDG boxes
  4. reefer boxes only in blocks I or J
  5. a customs-held box is never released
  6. a ship is berthed only after it has arrived, on a free berth it fits
     (berth length >= LOA + 20 m, berth max draft >= ship draft)
  7. broken or busy equipment is never assigned
  8. cranes stay inside their berth's bays
Plus basic sanity: known tool, well-formed arguments, known ids, block capacity.
"""
from __future__ import annotations

from typing import Any

from core.types import (AGENT_TOOLS, BLOCK_IMDG, BLOCK_REEFER, CARGO_IMDG, CARGO_REEFER,
                        EQ_OUT_OF_SERVICE, KNOWN_CARGO, SHIP_ARRIVED, SHIP_BERTHED, SHIP_SAILING,
                        SHIP_WORKING, TOOL_ASSIGN_BERTH, TOOL_DISPATCH_TRUCKS, TOOL_PLACE,
                        TOOL_PLAN_CRANES, TOOL_RELEASE, Action, WorldState)

MIN_CRANE_GAP = 2           # bays between two working cranes
BERTH_LENGTH_MARGIN = 20    # metres: berth length >= LOA + 20 m

Result = tuple[bool, str]
OK: Result = (True, "ok")


def _no(reason: str) -> Result:
    return False, reason


def _strs(v: Any, what: str) -> list[str]:
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, str) for x in v):
        raise ValueError(f"{what} must be a list of ids")
    return list(v)


def _rail_key(crane_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in crane_id if ch.isdigit())
    return (int(digits) if digits else 10**6, crane_id)


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------

def check(action: Action, state: WorldState) -> Result:
    """Return (True, "ok") if the action breaks no hard rule, else (False, reason)."""
    try:
        tool = getattr(action, "tool", None)
        args = getattr(action, "args", None)
        if tool not in AGENT_TOOLS:
            return _no(f"unknown or forbidden tool {tool!r}")
        if not isinstance(args, dict) and not hasattr(args, "get"):
            return _no("arguments must be a mapping")
        if tool == TOOL_PLAN_CRANES:
            return _check_cranes(args, state)
        if tool == TOOL_PLACE:
            return _check_place(args, state)
        if tool == TOOL_RELEASE:
            return _check_release(args, state)
        if tool == TOOL_ASSIGN_BERTH:
            return _check_berth(args, state)
        if tool == TOOL_DISPATCH_TRUCKS:
            return _check_trucks(args, state)
        return _no(f"no rule set for tool {tool!r}")
    except Exception as e:      # malformed input: refuse, never raise
        return _no(f"cannot verify action ({type(e).__name__}: {e})"[:200])


# --------------------------------------------------------------------------
# plan_crane_sequence: rules 1, 2, 7, 8
# --------------------------------------------------------------------------

def _check_cranes(args, state: WorldState) -> Result:
    call_id = args.get("call_id")
    crane_ids = _strs(args.get("crane_ids"), "crane_ids")
    bays = args.get("bay_positions")
    if not isinstance(bays, (list, tuple)) or not all(isinstance(b, int) and not isinstance(b, bool) for b in bays):
        return _no("bay_positions must be a list of whole bay numbers")
    bays = list(bays)
    if len(crane_ids) != len(bays):
        return _no("one bay position per crane is required")
    if len(set(crane_ids)) != len(crane_ids):
        return _no("a crane is listed twice")
    ship = state.ship(call_id) if isinstance(call_id, str) else None
    if ship is None:
        return _no(f"unknown call {call_id!r}")
    if ship.status not in (SHIP_BERTHED, SHIP_WORKING) or not ship.berth:
        return _no(f"{call_id} is not berthed")
    berth = state.berth(ship.berth)
    if berth is None:
        return _no(f"no reference data for berth {ship.berth!r}")
    known = {c.crane_id: c for c in state.cranes}
    for c in crane_ids:
        if c not in known:
            return _no(f"unknown crane {c!r}")

    # Positions after the call: listed cranes move, every other crane stays put.
    after = {cid: c.bay for cid, c in known.items()}
    after.update(dict(zip(crane_ids, bays)))

    # 1. Rail order over ALL cranes, idle and broken included.
    order = sorted(after, key=_rail_key)
    for a, b in zip(order, order[1:]):
        if not after[a] < after[b]:
            return _no(f"rail order broken: {a} at bay {after[a]} would not be left of {b} at bay {after[b]}")

    # 2. Spacing among working cranes after the call. This call's crane list is
    #    replaced by crane_ids; other berthed/working ships keep theirs.
    working = set(crane_ids)
    for s in state.ships:
        if s.call_id != call_id and s.status in (SHIP_BERTHED, SHIP_WORKING):
            working |= set(s.cranes)
    pos = sorted((after[c], c) for c in working if c in after)
    for (p1, c1), (p2, c2) in zip(pos, pos[1:]):
        if p2 - p1 < MIN_CRANE_GAP:
            return _no(f"working cranes {c1} (bay {p1}) and {c2} (bay {p2}) would be under {MIN_CRANE_GAP} bays apart")

    # 7. Broken or busy cranes are never assigned.
    busy = {c: s.call_id for s in state.ships if s.call_id != call_id for c in s.cranes}
    for c in crane_ids:
        if known[c].status == EQ_OUT_OF_SERVICE:
            return _no(f"{c} is out of service")
        if c in busy:
            return _no(f"{c} is working on {busy[c]}")

    # 8. Cranes stay inside their berth's bays.
    for c, b in zip(crane_ids, bays):
        if not berth.start_bay <= b <= berth.end_bay:
            return _no(f"{c} at bay {b} is outside {berth.berth_id} (bays {berth.start_bay}-{berth.end_bay})")
    return OK


# --------------------------------------------------------------------------
# place_container: rules 3, 4 (+ capacity, known boxes)
# --------------------------------------------------------------------------

def _container_index(state: WorldState) -> dict:
    return {c.container_id: c for rows in state.manifests.values() for c in rows}


def _check_place(args, state: WorldState) -> Result:
    boxes = _strs(args.get("container_ids"), "container_ids")
    block_id = args.get("block_id")
    if not boxes:
        return _no("no containers to place")
    if len(set(boxes)) != len(boxes):
        return _no("a container is listed twice")
    block = state.block(block_id) if isinstance(block_id, str) else None
    if block is None:
        return _no(f"unknown block {block_id!r}")
    index = _container_index(state)
    for cid in boxes:
        c = index.get(cid)
        if c is None:
            return _no(f"container {cid!r} is not in any known manifest")
        if c.cargo_type not in KNOWN_CARGO:
            return _no(f"container {cid} has unknown cargo type {c.cargo_type!r}")
        # 3. IMDG only in H; H only IMDG.
        if c.cargo_type == CARGO_IMDG and block.block_type != BLOCK_IMDG:
            return _no(f"IMDG container {cid} may only go to the IMDG block (H), not {block_id}")
        if c.cargo_type != CARGO_IMDG and block.block_type == BLOCK_IMDG:
            return _no(f"block {block_id} takes only IMDG containers; {cid} is {c.cargo_type}")
        # 4. Reefers only in I/J.
        if c.cargo_type == CARGO_REEFER and block.block_type != BLOCK_REEFER:
            return _no(f"reefer container {cid} may only go to a reefer block (I or J), not {block_id}")
    new = [cid for cid in boxes if index[cid].yard_block is None]
    if len(new) > block.free:
        return _no(f"block {block_id} has {block.free} free slots for {len(new)} containers")
    return OK


# --------------------------------------------------------------------------
# release_to_consignee: rule 5
# --------------------------------------------------------------------------

def _check_release(args, state: WorldState) -> Result:
    cid = args.get("container_id")
    if not isinstance(cid, str):
        return _no("container_id must be an id")
    if cid in state.held_ids():
        return _no(f"{cid} is under a customs hold and must never leave the terminal")
    return OK


# --------------------------------------------------------------------------
# assign_berth: rule 6
# --------------------------------------------------------------------------

def _check_berth(args, state: WorldState) -> Result:
    call_id, berth_id = args.get("call_id"), args.get("berth_id")
    ship = state.ship(call_id) if isinstance(call_id, str) else None
    if ship is None:
        return _no(f"unknown call {call_id!r}")
    berth = state.berth(berth_id) if isinstance(berth_id, str) else None
    if berth is None:
        return _no(f"unknown berth {berth_id!r}")
    if ship.status != SHIP_ARRIVED:
        return _no(f"{call_id} is '{ship.status}'; only an arrived ship can be berthed")
    for s in state.ships:
        if s.berth == berth_id and s.status in (SHIP_BERTHED, SHIP_WORKING, SHIP_SAILING):
            return _no(f"{berth_id} is occupied by {s.call_id}")
    if ship.loa_m is None or ship.draft_m is None:
        return _no(f"{call_id} has no valid length or draft")
    if berth.length_m < ship.loa_m + BERTH_LENGTH_MARGIN:
        return _no(f"{call_id} (LOA {ship.loa_m} m) needs {ship.loa_m + BERTH_LENGTH_MARGIN} m; "
                   f"{berth_id} is {berth.length_m} m")
    if ship.draft_m > berth.max_draft_m:
        return _no(f"{call_id} (draft {ship.draft_m} m) is too deep for {berth_id} ({berth.max_draft_m} m)")
    return OK


# --------------------------------------------------------------------------
# dispatch_truck: rule 7
# --------------------------------------------------------------------------

def _check_trucks(args, state: WorldState) -> Result:
    call_id = args.get("call_id")
    trucks = _strs(args.get("truck_ids"), "truck_ids")
    if not isinstance(call_id, str) or state.ship(call_id) is None:
        return _no(f"unknown call {call_id!r}")
    if len(set(trucks)) != len(trucks):
        return _no("a truck is listed twice")
    known = {t.truck_id: t for t in state.trucks}
    for t in trucks:
        if t not in known:
            return _no(f"unknown truck {t!r}")
        if known[t].status == EQ_OUT_OF_SERVICE:
            return _no(f"{t} is out of service")
        if known[t].call_id and known[t].call_id != call_id:
            return _no(f"{t} already serves {known[t].call_id}")
    return OK

