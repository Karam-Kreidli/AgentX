"""One LLM call per tick, shared by every LLM user (LLM-4 supervisor, LLM-5 voice, LLM-6 operator).

    from llm import slot
    if slot.claim("voice"):
        ...one llm.ask(...)

The tick watchdog is 10 s and one call takes ~3 s at p95, so a tick has room for one call.
The first user to claim a tick gets it; the same user may claim it again (it guards its own
repeats itself); everyone else gets False and uses their template/fallback. Outside a run
(no current tick, e.g. in a unit test) every claim succeeds. Never raises.
"""
from __future__ import annotations

from typing import Optional

from core import log as runlog

_owner: dict = {"tick": None, "who": None}


def claim(who: str) -> bool:
    try:
        tick: Optional[int] = runlog.current_tick()
    except Exception:
        return True
    if tick is None:
        return True
    if _owner["tick"] == tick and _owner["who"] != who:
        return False
    _owner.update(tick=tick, who=who)
    return True


def holder(tick: Optional[int]) -> Optional[str]:
    """Who has claimed `tick` (None if nobody)."""
    return _owner["who"] if _owner["tick"] == tick else None


def reset() -> None:
    _owner.update(tick=None, who=None)
