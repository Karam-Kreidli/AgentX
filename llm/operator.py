"""Operator channel: plain English to safe commands (LLM-6).

    from llm import operator
    cmd = operator.translate(message, state)          # Command, or None if the LLM can't answer
    ok, why = operator.validate(cmd, state)           # our rules, never the LLM's
    result = operator.handle(message, state)          # translate -> validate -> apply -> Result

A judge types a message (core/inbox.py). The LLM may only map it to one of five fixed commands
(llm/schemas.py OperatorCommand) or "not_a_command"; there is no override command and nothing
turns the guard off. Our code then checks the command against the rules and the live state:

  prioritise_ship  the ship exists, has not left, and can finish (an impossible ship is refused)
  reserve_berth    the berth exists; a named ship must exist, not have left, and FIT the berth
                   (length >= LOA + 20 m, depth >= draft: the guard's berth rule). Without a ship
                   the berth is kept free for RESERVE_FREE_H hours at most, so it can't stall.
  hold_ship        the ship exists and has not been berthed yet (a ship at berth is not moved)
  approve/decline  only if something is waiting for a human; today nothing is, so it is refused
                   with that reason
  not_a_command    refused, and escalated so a human sees it

Accepted commands become directives the agents read: the berth agent puts prioritised ships
first, berths reserved berths only to their ship, and skips held ships; the coordinator serves
prioritised ships first for yard slots. A directive never bypasses the guard: every action
still goes through it. Refusals are logged with a clear reason and escalated.

LLM: one call per message, through the shared one-call-per-tick slot (llm/slot.py); if the
slot is taken the message waits for the next tick. LLM off or no valid answer -> nothing is
done and the message is escalated ("could not be understood"). Never raises.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional

from core.types import SHIP_ARRIVED, SHIP_DEPARTED, SHIP_EXPECTED, SHIP_SAILING, WorldState
from llm import client as llm_client
from llm import slot
from llm.client import untrusted
from llm.schemas import (CMD_APPROVE, CMD_DECLINE, CMD_HOLD_SHIP, CMD_NONE, CMD_PRIORITISE_SHIP, CMD_RESERVE_BERTH,
                         OperatorCommand)

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "operator.md"
DRILLS_PATH = Path(__file__).resolve().parent / "operator_drills.json"
RESERVE_FREE_H = 6.0        # a berth kept free with no ship named is released after this
GONE = (SHIP_SAILING, SHIP_DEPARTED)

ACCEPTED, REFUSED, DEFERRED = "accepted", "refused", "deferred"


@dataclass(frozen=True)
class Command:
    kind: str
    ship: Optional[str] = None
    berth: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True)
class Result:
    status: str                     # accepted | refused | deferred
    command: Optional[Command]
    message: str                    # the operator's text, capped
    reply: str                      # one sentence for the log / the operator


@dataclass
class Directives:
    priority: list = field(default_factory=list)          # call ids, first = most urgent
    reserved: dict = field(default_factory=dict)          # berth -> call id, or None (keep free)
    reserved_until: dict = field(default_factory=dict)    # berth -> datetime (keep-free expiry)
    held: set = field(default_factory=set)


_d = Directives()
_mem = {"scenario": None}


def directives(state: Optional[WorldState] = None) -> Directives:
    """Current directives (expired keep-free reservations and departed ships dropped)."""
    try:
        if state is not None:
            _scenario(state)
            gone = {s.call_id for s in state.ships if s.status in GONE}
            _d.priority[:] = [c for c in _d.priority if c not in gone]
            _d.held -= gone
            for b in list(_d.reserved):
                ship = _d.reserved[b]
                expired = ship is None and _d.reserved_until.get(b) and state.now >= _d.reserved_until[b]
                if expired or (ship is not None and (ship in gone or _berthed_on(state, ship, b))):
                    _d.reserved.pop(b, None)
                    _d.reserved_until.pop(b, None)
    except Exception:
        pass
    return _d


def reset() -> None:
    global _d
    _d = Directives()
    _mem["scenario"] = None


def _scenario(state: WorldState) -> None:
    key = (state.scenario_end, tuple(sorted(s.call_id for s in state.ships)))
    if _mem["scenario"] != key:
        reset()
        _mem["scenario"] = key


def _berthed_on(state, call_id, berth_id) -> bool:
    s = state.ship(call_id)
    return s is not None and s.berth == berth_id and s.status not in (SHIP_EXPECTED, SHIP_ARRIVED)


# --------------------------------------------------------------------------
# translate (LLM)
# --------------------------------------------------------------------------

def _ships_view(state: Optional[WorldState]) -> list:
    if state is None:
        return []
    return [{"call_id": s.call_id, "vessel": untrusted(s.vessel, 60), "status": s.status} for s in state.ships]


def translate(message: str, state: Optional[WorldState] = None) -> Optional[Command]:
    """The command the message asks for, or None if the LLM is unavailable/invalid. Never raises."""
    try:
        payload = {"message": untrusted(message, 500), "ships": _ships_view(state)}
        ans = llm_client.ask("operator", payload, OperatorCommand, system=PROMPT_PATH.read_text(encoding="utf-8"))
        if ans is None:
            return None
        return Command(ans["command"], ans.get("ship"), ans.get("berth"), untrusted(ans.get("reason", ""), 200))
    except Exception:
        return None


# --------------------------------------------------------------------------
# validate (our rules)
# --------------------------------------------------------------------------

def _impossible(state: WorldState) -> set:
    try:
        from agents import stall
        return set(stall.impossible_calls(state))
    except Exception:
        return set()


def validate(cmd: Command, state: WorldState) -> tuple[bool, str]:
    """(ok, reason). The rules are ours: the LLM's reading is only a proposal."""
    if cmd is None:
        return False, "the message could not be understood, so nothing was changed"
    k = cmd.kind
    if k == CMD_NONE and "keep_at_berth" in (cmd.reason or ""):
        return False, ("a ship can't be kept at its berth or have its departure delayed: the terminal "
                       "sails it automatically one hour after its last move, and no command can change that")
    if k == CMD_NONE:
        return False, ("that is not one of the commands I can carry out (prioritise a ship, reserve a "
                       "berth, hold a ship, approve or decline an escalation); the safety rules can't be "
                       "changed or switched off")
    ship = state.ship(cmd.ship) if cmd.ship else None
    if k in (CMD_PRIORITISE_SHIP, CMD_HOLD_SHIP) and ship is None:
        return False, f"there is no ship {untrusted(cmd.ship or '(none named)', 40)} in this scenario"
    if k == CMD_PRIORITISE_SHIP:
        if ship.status in GONE:
            return False, f"{ship.call_id} has already left the berth"
        if ship.call_id in _impossible(state):
            return False, (f"{ship.call_id} can never finish (it must discharge more boxes than its manifest "
                           f"has), so it would take space that ships which can finish need")
        return True, f"{ship.call_id} goes first for a berth and for yard space"
    if k == CMD_HOLD_SHIP:
        if ship.status not in (SHIP_EXPECTED, SHIP_ARRIVED):
            return False, f"{ship.call_id} is already at a berth ({ship.status}); a working ship is not sent away"
        return True, f"{ship.call_id} will not be berthed until the hold is lifted"
    if k == CMD_RESERVE_BERTH:
        b = state.berth(cmd.berth) if cmd.berth else None
        if b is None:
            return False, f"there is no berth {untrusted(cmd.berth or '(none named)', 10)}"
        if cmd.ship is None:
            return True, f"{b.berth_id} is kept free for up to {RESERVE_FREE_H:g} h"
        if ship is None:
            return False, f"there is no ship {untrusted(cmd.ship, 40)} in this scenario"
        if ship.status in GONE:
            return False, f"{ship.call_id} has already left"
        if not b.fits(ship.loa_m or 0, ship.draft_m or 0):
            return False, (f"{ship.call_id} (LOA {ship.loa_m:g} m, draft {ship.draft_m:g} m) does not fit "
                           f"{b.berth_id} ({b.length_m:g} m, depth {b.max_draft_m:g} m; it needs LOA + 20 m)")
        return True, f"{b.berth_id} is kept for {ship.call_id}"
    if k in (CMD_APPROVE, CMD_DECLINE):
        return False, "nothing is waiting for a human decision right now"
    return False, f"unknown command {untrusted(k, 30)}"


def apply(cmd: Command, state: WorldState) -> None:
    d = directives(state)
    if cmd.kind == CMD_PRIORITISE_SHIP:
        if cmd.ship in d.priority:
            d.priority.remove(cmd.ship)
        d.priority.insert(0, cmd.ship)
        d.held.discard(cmd.ship)
    elif cmd.kind == CMD_HOLD_SHIP:
        d.held.add(cmd.ship)
        if cmd.ship in d.priority:
            d.priority.remove(cmd.ship)
    elif cmd.kind == CMD_RESERVE_BERTH:
        d.reserved[cmd.berth] = cmd.ship
        if cmd.ship is None:
            d.reserved_until[cmd.berth] = state.now + timedelta(hours=RESERVE_FREE_H)
        else:
            d.reserved_until.pop(cmd.berth, None)
            for b, s in list(d.reserved.items()):          # one berth per ship
                if s == cmd.ship and b != cmd.berth:
                    d.reserved.pop(b)


# --------------------------------------------------------------------------
# the whole channel
# --------------------------------------------------------------------------

def handle(message: str, state: WorldState, use_llm: bool = True) -> Result:
    """translate -> validate -> apply. DEFERRED when this tick's LLM slot is taken. Never raises."""
    text = untrusted(message, 500)
    try:
        directives(state)
        if use_llm and not slot.claim("operator"):
            return Result(DEFERRED, None, text, "waiting for the next tick to read this message")
        cmd = translate(text, state) if use_llm else None
        ok, why = validate(cmd, state)
        if ok:
            apply(cmd, state)
            return Result(ACCEPTED, cmd, text, f"Done: {why}.")
        return Result(REFUSED, cmd, text, f"Not done: {why}.")
    except Exception as e:
        return Result(REFUSED, None, text, f"Not done: the message could not be handled ({type(e).__name__}).")


# --------------------------------------------------------------------------
# drills (the CLI that runs them on the real model: python tools/operator_drills.py)
# --------------------------------------------------------------------------

def run_drills(state: WorldState) -> list[dict]:
    """Each drill message through translate + validate. Pass = expected command (where given)
    and expected outcome."""
    rows = []
    for d in json.loads(DRILLS_PATH.read_text(encoding="utf-8")):
        reset()
        _mem["scenario"] = (state.scenario_end, tuple(sorted(s.call_id for s in state.ships)))
        cmd = translate(d["message"], state)
        ok, why = validate(cmd, state)
        got = ACCEPTED if ok else REFUSED
        passed = got == d["outcome"] and (d.get("command") is None or (cmd is not None and cmd.kind == d["command"]))
        rows.append({"message": d["message"], "expected": d["outcome"], "got": got,
                     "command": cmd.kind if cmd else None, "why": why, "pass": passed})
    return rows
