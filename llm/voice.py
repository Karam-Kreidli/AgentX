"""Negotiation voice for the agents (LLM-5).

    from llm import voice
    sentence = voice.voice(request_or_grant)                 # one line
    sentences = voice.voice_batch(items, use_llm=True)       # one LLM call for the whole batch

Every line of the negotiation transcript (core/log.py negotiation.md) gets one sentence: a
request in the requesting agent's voice, a decision in the coordinator's voice. The sentence
is built from the structured fields; the LLM may only reword it.

Numbers are never invented: an LLM sentence is used only if every number in it appears in
that item's own fields, it names the item's ship, has no markup and is one short line.
Otherwise that item gets its template (code-written from the same fields, always correct).

Cost (the LLM is slow at ~20 tokens/s, and a tick has room for one ~3 s call):
  - one call per batch, at most MAX_LINES items worded by the LLM (the rest use templates);
  - only when this tick's LLM slot is free (llm/slot.py: the supervisor and the operator
    channel come first), the call budget keeps BUDGET_RESERVE calls for them, and at most
    MAX_CALLS voice calls per scenario.
Never raises.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core import log as runlog
from core.types import GRANTED, REFUSED, TRIMMED
from llm import client as llm_client
from llm import slot
from llm.client import untrusted

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "voice.md"
MAX_LINES = 3                # items worded by the LLM per call (~35 tokens each at ~20 tokens/s)
MAX_CALLS = 12               # voice calls per scenario
BUDGET_RESERVE = 20          # calls of the scenario budget kept for the supervisor and operator
MAX_WORDS = 40
_NUM = re.compile(r"\d+(?:\.\d+)?")
_RES_WORDS = {"yard_slots": "yard slots", "trucks": "trucks", "cranes": "cranes", "berth": "berth"}

_mem = {"tick": None, "calls": 0}


class _Line(BaseModel):
    model_config = ConfigDict(extra="forbid")
    i: int
    sentence: str = Field(max_length=300)


class VoiceBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lines: list[_Line] = Field(max_length=MAX_LINES)


def reset() -> None:
    _mem.update(tick=None, calls=0)


# --------------------------------------------------------------------------
# fields and templates
# --------------------------------------------------------------------------

def _parts(item: Any) -> tuple[Any, Any]:
    """(request, grant or None) for a Request or a Grant."""
    if hasattr(item, "request") and hasattr(item, "status"):
        return item.request, item
    return item, None


def fields(item: Any) -> dict:
    """What the sentence may use: the item's structured fields (untrusted text capped)."""
    req, g = _parts(item)
    out = {"speaker": "coordinator" if g is not None else f"{untrusted(getattr(req, 'agent', '?'), 20)} agent",
           "to": f"{untrusted(getattr(req, 'agent', '?'), 20)} agent" if g is not None else "coordinator",
           "kind": "decision" if g is not None else "request",
           "ship": untrusted(getattr(req, "call_id", ""), 40),
           "resource": _RES_WORDS.get(getattr(req, "resource", ""), untrusted(getattr(req, "resource", ""), 20)),
           "asked": getattr(req, "amount", None),
           "why": untrusted(getattr(req, "reason", ""), 200)}
    if g is not None:
        out.update(status=untrusted(getattr(g, "status", ""), 20), granted=getattr(g, "granted", None),
                   why=untrusted(getattr(g, "reason", ""), 200))
    return out


def template(item: Any) -> str:
    """The sentence from the fields alone (used whenever the LLM is not)."""
    f = fields(item)
    ship, res, asked, why = f["ship"], f["resource"], f["asked"], f["why"].rstrip(".")
    if f["kind"] == "request":
        return f"I'm asking for {asked} {res} for {ship}: {why}." if why else f"I'm asking for {asked} {res} for {ship}."
    status, got = f.get("status"), f.get("granted")
    if status == GRANTED:
        head = f"You can have {got} {res} for {ship}"
    elif status == TRIMMED:
        head = f"I can give you {got} of the {asked} {res} you asked for {ship}"
    elif status == REFUSED:
        head = f"No {res} for {ship} right now"
    else:
        head = f"{got} of {asked} {res} for {ship}"
    return f"{head}: {why}." if why else f"{head}."


# --------------------------------------------------------------------------
# checks on an LLM sentence
# --------------------------------------------------------------------------

def _canon(n: str) -> str:
    return n.rstrip("0").rstrip(".") if "." in n else (n.lstrip("0") or "0")


def _numbers(obj: Any) -> set[str]:
    return {_canon(n) for n in _NUM.findall(json.dumps(obj, ensure_ascii=False, default=str))}


def check(sentence: str, item_fields: dict) -> bool:
    """Every number from the fields, names the ship, one short plain line."""
    s = str(sentence)
    if not s.strip() or "\n" in s or any(ch in s for ch in "<>`|#*[]{}"):
        return False
    if len(s.split()) > MAX_WORDS:
        return False
    ship = item_fields.get("ship") or ""
    if ship and ship not in s:
        return False
    return _numbers(s) <= _numbers(item_fields)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def _may_call() -> bool:
    tick = runlog.current_tick()
    if tick is not None and _mem["tick"] is not None and tick < _mem["tick"]:
        reset()                                        # a new scenario started
    if _mem["calls"] >= MAX_CALLS:
        return False
    c = llm_client.get_client()
    if not getattr(c, "enabled", False) or c.budget - c.used <= BUDGET_RESERVE:
        return False
    return slot.claim("voice")


def voice_batch(items: list, use_llm: bool = True) -> list[str]:
    """One sentence per item, in order. Never raises."""
    try:
        out = [template(x) for x in items]
    except Exception:
        return ["" for _ in items]
    try:
        if not use_llm or not items or not _may_call():
            return out
        _mem["tick"] = runlog.current_tick()
        _mem["calls"] += 1
        chosen = list(range(min(MAX_LINES, len(items))))
        payload = {"items": [dict(i=i, **fields(items[i])) for i in chosen]}
        system = PROMPT_PATH.read_text(encoding="utf-8")
        ans = llm_client.ask("voice", payload, VoiceBatch, system=system, max_tokens=60 * len(chosen) + 40)
        if ans is None:
            return out
        for line in ans.get("lines", []):
            i = line.get("i")
            if i in chosen:
                s = " ".join(str(line.get("sentence", "")).split())
                if check(s, fields(items[i])):
                    out[i] = s
    except Exception:
        pass
    return out


def voice(item: Any) -> str:
    """One item (interface from the task spec)."""
    return voice_batch([item])[0]
