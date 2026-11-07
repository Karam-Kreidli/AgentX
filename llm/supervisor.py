"""Supervisor: choose among options and explain (LLM-4).

    from llm import supervisor
    option_id, sentence = supervisor.decide(event, options, state=state)

Called by the loop when the coordinator returns 2-3 `Option`s (plans that genuinely compete).
The model sees the goal ladder (llm/prompts/supervisor.md), a compact terminal summary and the
options with their predicted effects and code-written explanations, and answers with one option
id and one sentence. It never creates actions: it only picks among options our code built,
which the guard checks like every other action.

Fallback, always: (options[0].id, options[0].template). Used when the LLM is off, down, slow,
over budget, answers outside the schema, or has already been asked this tick.

Safety:
  - The option id is constrained by the schema (structured output on vLLM; pydantic elsewhere).
  - Untrusted text (vessel names, event details) goes in only through llm.client.untrusted:
    quoted data, capped, control characters removed.
  - The sentence may only use numbers that appear in what the model was shown; otherwise the
    chosen option's template is used instead (the choice stands, the wording is ours).
  - At most one LLM call per tick (the tick watchdog is 10 s; p95 is ~3 s).

Every decision also writes one sentence to supervisor_feed.txt (SAFE-3).

The bake-off (LLM-3) builds its prompts with `payload()` and `summary()` from this module, so
re-running it measures exactly what the supervisor sends.
"""
from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

from core import log as runlog
from core.types import SHIP_ARRIVED, SHIP_BERTHED, SHIP_EXPECTED, SHIP_SAILING, SHIP_WORKING
from llm import client as llm_client
from llm.client import untrusted
from llm.schemas import REASON_MAX, choice_model

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "supervisor.md"
WINDOW_DAYS = 5
MAX_WORDS = 60
_NUM = re.compile(r"\d+(?:\.\d+)?")

_state = {"tick": None, "backend": "rules"}


def _prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def last_backend() -> str:
    """Who made the last decision: 'rules' (fallback), or the LLM backend ('local' / 'groq')."""
    return _state["backend"]


def reset() -> None:
    _state.update(tick=None, backend="rules")


# --------------------------------------------------------------------------
# what the model sees
# --------------------------------------------------------------------------

def _clock(state) -> str:
    """'Day 2 06:15': day 1 is the first day of the 5-day window."""
    start = state.scenario_end - timedelta(days=WINDOW_DAYS)
    return f"Day {(state.now - start).days + 1} {state.now:%H:%M}"


def _unplaced(state, call_id) -> int:
    return sum(1 for c in state.manifests.get(call_id, ()) if c.direction == "import" and not c.yard_block)


def _group(block_type: str) -> str:
    return {"IMDG": "imdg", "reefer": "reefer"}.get(block_type, "dry")


def summary(state) -> dict:
    """Compact, prompt-ready view of the terminal. Vessel names are untrusted and capped."""
    free = {"imdg": 0, "reefer": 0, "dry": 0}
    for b in state.blocks:
        free[_group(b.block_type)] += b.free
    ships = []
    for s in state.ships:
        if s.status not in (SHIP_EXPECTED, SHIP_ARRIVED, SHIP_BERTHED, SHIP_WORKING):
            continue
        waited = round((state.now - s.actual_arrival).total_seconds() / 3600, 1) if s.actual_arrival else None
        ships.append({"call_id": s.call_id, "vessel": untrusted(s.vessel), "status": s.status, "berth": s.berth,
                      "waiting_h": waited if s.status == SHIP_ARRIVED else None,
                      "moves_left": s.remaining_moves, "imports_without_slot": _unplaced(state, s.call_id)})
    busy = {s.berth for s in state.ships if s.berth and s.status in (SHIP_BERTHED, SHIP_WORKING, SHIP_SAILING)}
    return {"time": _clock(state),
            "hours_left_in_window": round((state.scenario_end - state.now).total_seconds() / 3600, 1),
            "ships": ships, "yard_free_slots": free,
            "free_berths": [b.berth_id for b in state.berths if b.berth_id not in busy]}


def payload(question: str, terminal: Optional[dict], options: list[dict]) -> dict:
    """The user message: the question, the terminal summary and the options (dicts with id,
    predicted, template, and optionally label)."""
    keys = ("id", "label", "predicted", "template")
    return {"question": question, "terminal": terminal,
            "options": [{k: o[k] for k in keys if k in o} for o in options]}


def _question(event) -> str:
    base = "Our code built these plans for one decision. Which should the terminal follow?"
    if event is None:
        return base
    kind = untrusted(getattr(event, "kind", ""), 30)
    subject = untrusted(getattr(event, "subject", ""), 40)
    return f"{base} Trigger: {kind} ({subject})."


def _numbers(obj: Any) -> set[str]:
    return {_canon(n) for n in _NUM.findall(json.dumps(obj, ensure_ascii=False, default=str))}


def _canon(n: str) -> str:
    return n.rstrip("0").rstrip(".") if "." in n else n.lstrip("0") or "0"


def _clean(sentence: str) -> str:
    s = " ".join(str(sentence).split())
    words = s.split(" ")
    if len(words) > MAX_WORDS:
        s = " ".join(words[:MAX_WORDS]).rstrip(",;") + "."
    return s[:REASON_MAX]


def check_sentence(sentence: str, shown: dict) -> bool:
    """True if every number in the sentence appears in what the model was shown, and it is
    one plain sentence of text (no markup to forge log lines)."""
    s = str(sentence)
    if not s.strip() or any(ch in s for ch in "<>`|#*[]{}"):
        return False
    return _numbers(s) <= _numbers(shown)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def decide(event, options, state=None) -> tuple[str, str]:
    """(option_id, sentence). Falls back to (options[0].id, options[0].template). Never raises."""
    if not options:
        return "", ""
    first = options[0]
    choice, sentence, backend = first.id, first.template, "rules"
    try:
        tick = runlog.current_tick()
        if len(options) > 1 and not (tick is not None and tick == _state["tick"]):
            _state["tick"] = tick
            opts = [{"id": o.id, "predicted": dict(o.predicted), "template": o.template} for o in options]
            shown = payload(_question(event), summary(state) if state is not None else None, opts)
            ans = llm_client.ask("supervisor", shown, choice_model([o.id for o in options]), system=_prompt())
            if ans is not None:
                picked = next((o for o in options if o.id == ans["option_id"]), None)
                if picked is not None:
                    backend = llm_client.get_client().last_backend()
                    choice = picked.id
                    reason = _clean(ans.get("reason", ""))
                    sentence = reason if check_sentence(reason, shown) else picked.template
    except Exception as e:                     # never crash the loop: the default plan stands
        choice, sentence, backend = first.id, first.template, "rules"
        runlog.note(runlog.current_tick() or 0, state, "supervisor", "exception",
                    f"{type(e).__name__}: {str(e)[:150]}; using option 0.")
    _state["backend"] = backend
    runlog.feed(sentence, untrusted(getattr(event, "subject", ""), 40) or None)
    return choice, sentence
