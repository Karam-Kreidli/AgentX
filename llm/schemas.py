"""LLM output schemas (LLM-2).

Every LLM answer is checked against one of these pydantic models before anyone
sees it. A reply that doesn't match is treated as "no answer" (the client returns
None), so invalid JSON is never passed on (Ground rules §5).

- SupervisorChoice / choice_model(): LLM-4 picks one of the option ids our code made.
- VoiceLine: LLM-5 writes one sentence for the negotiation transcript.
- OperatorCommand: LLM-6 turns an operator message into one fixed command.

The LLM can only fill these fields. None of them is an Action: turning an answer
into something the simulator sees is always our code's job, behind the guard.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model


class _Strict(BaseModel):
    """No extra keys allowed; unknown fields make the answer invalid."""
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- LLM-4 supervisor

REASON_MAX = 300


class SupervisorChoice(_Strict):
    """Generic form. Use choice_model(ids) so the model can only name a real option."""
    option_id: str = Field(max_length=40)
    reason: str = Field(max_length=REASON_MAX)


def choice_model(option_ids) -> type[BaseModel]:
    """A SupervisorChoice whose option_id must be one of `option_ids`.

    With structured output on our vLLM server the model can't even produce another
    id; on other backends pydantic rejects it.
    """
    ids = tuple(dict.fromkeys(str(i) for i in option_ids))
    if not ids:
        raise ValueError("choice_model needs at least one option id")
    return create_model(
        "SupervisorChoice",
        __base__=_Strict,
        option_id=(Literal[ids], ...),
        reason=(str, Field(max_length=REASON_MAX)),
    )


# ---------------------------------------------------------------- LLM-5 voice

class VoiceLine(_Strict):
    sentence: str = Field(max_length=300)


# ---------------------------------------------------------------- LLM-6 operator

CMD_PRIORITISE_SHIP = "prioritise_ship"
CMD_RESERVE_BERTH = "reserve_berth"
CMD_HOLD_SHIP = "hold_ship"
CMD_APPROVE = "approve_escalation"
CMD_DECLINE = "decline_escalation"
CMD_NONE = "not_a_command"
COMMANDS = (CMD_PRIORITISE_SHIP, CMD_RESERVE_BERTH, CMD_HOLD_SHIP, CMD_APPROVE, CMD_DECLINE, CMD_NONE)
# There is deliberately no override / admin / disable-guard command (Ground rules §4-5).


class OperatorCommand(_Strict):
    command: Literal[COMMANDS]
    ship: Optional[str] = Field(default=None, max_length=40)
    berth: Optional[Literal["B1", "B2", "B3"]] = None
    reason: str = Field(max_length=200)
