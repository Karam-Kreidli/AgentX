"""One LLM call per tick, shared by every LLM user (LLM-4 supervisor, LLM-5 voice, LLM-6 operator).

    from llm import slot
    if slot.claim("voice"):
        ...one llm.ask(...)

The tick watchdog is 10 s and one call takes ~3-4 s (up to the 8 s deadline), so a tick has
room for exactly one call. The first claim in a tick gets it; every later claim in that tick
gets False, the same user's included, and uses its template/fallback. (Letting the same user
claim again let the voice make one call per coordinator round, up to three ~4 s calls in one
tick: ticks ran past 10 s and the watchdog finished them on rules-only decisions.) Outside a
run (no current tick, e.g. in a unit test) every claim succeeds. Never raises.
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
    if _owner["tick"] == tick:
        return False                    # this tick's one call is already spent
    _owner.update(tick=tick, who=who)
    return True


def holder(tick: Optional[int]) -> Optional[str]:
    """Who has claimed `tick` (None if nobody)."""
    return _owner["who"] if _owner["tick"] == tick else None


def reset() -> None:
    _owner.update(tick=None, who=None)
