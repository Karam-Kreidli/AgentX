"""Executor (CORE-4): the only code that changes the simulator.

    ex = Executor(api, run_dir)
    result = ex.execute(action, state)     # state = the snapshot the action was planned on
    ex.state                               # fresh snapshot after the action (read-after-write)

For every action:
  1. refuse it if it has no reason (every state-changing action must explain itself)
  2. guard.check(action, state); a refusal is logged and the action is NOT sent
  3. api.call(tool, args); simulator errors come back as data and are logged, never raised
  4. log one line in decisions.jsonl (who, what, why, guard result, tool result)
  5. re-read the state, and check the simulator now shows what we did; if not,
     the simulator is believed and the mismatch is logged as an anomaly

Agents never import this module or the toolkit: they return Actions, the loop
hands them here.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import guard
from core.state import build_state, hours, invalidate
from core.types import (TOOL_ASSIGN_BERTH, TOOL_DISPATCH_TRUCKS, TOOL_PLACE, TOOL_PLAN_CRANES,
                        TOOL_RELEASE, Action, Decision, WorldState)


def _json_default(o: Any) -> Any:
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if hasattr(o, "items"):
        return dict(o)
    return str(o)


class DecisionLog:
    """Append-only decisions.jsonl (minimal; SAFE-3 extends it with transcript and feed)."""

    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "decisions.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, d: Decision) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(d), ensure_ascii=False, default=_json_default) + "\n")

    def note(self, tick: int, state: WorldState, agent: str, choice: str, reason: str,
             saw: Optional[dict] = None) -> None:
        """Non-action entries: escalations, exceptions, invariant breaches, mismatches."""
        self.write(Decision(tick=tick, t=hours(state), agent=agent, saw=saw or {}, options=(),
                            choice=choice, reason=reason, guard_ok=True, guard_reason="n/a (not an action)",
                            tool_result=None, backend="rules"))


class Executor:
    def __init__(self, api: Any, run_dir: Path, log: Optional[DecisionLog] = None):
        self.api = api
        self.log = log or DecisionLog(run_dir)
        self.tick = 0
        self.state: Optional[WorldState] = None
        self.sent = 0
        self.refused = 0
        self.mismatches = 0

    # ------------------------------------------------------------------
    def execute(self, action: Action, state: WorldState, backend: str = "rules",
                options: tuple[str, ...] = ()) -> dict:
        reason = (getattr(action, "reason", "") or "").strip()
        if not reason:
            ok, why = False, "refused: every state-changing action needs a reason"
        else:
            ok, why = guard.check(action, state)
        agent = getattr(action, "agent", "?")
        tool = getattr(action, "tool", "?")
        args = dict(getattr(action, "args", {}) or {})
        if not ok:
            self.refused += 1
            self.log.write(Decision(tick=self.tick, t=hours(state), agent=agent, saw={"args": args},
                                    options=options, choice=tool, reason=reason or "(none)",
                                    guard_ok=False, guard_reason=why, tool_result=None, backend=backend))
            self.state = state
            return {"refused": why}

        result = self.api.call(tool, args)          # never raises; errors come back as {"error": ...}
        self.sent += 1
        self.log.write(Decision(tick=self.tick, t=hours(state), agent=agent, saw={"args": args},
                                options=options, choice=tool, reason=reason, guard_ok=True,
                                guard_reason=why, tool_result=result if isinstance(result, dict) else {"result": result},
                                backend=backend))
        if isinstance(result, dict) and "error" in result:
            self.state = state
            return result

        # Read-after-write.
        if tool == TOOL_PLACE:
            for call_id in {c.call_id for rows in state.manifests.values() for c in rows
                            if c.container_id in set(args.get("container_ids", ()))}:
                invalidate(self.api, call_id)
        new = build_state(self.api)
        problem = self._mismatch(tool, args, new)
        if problem:
            self.mismatches += 1
            self.log.note(self.tick, new, "executor", "mismatch",
                          f"After {tool} the simulator shows something else ({problem}); trusting the simulator.",
                          {"args": args})
        self.state = new
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _mismatch(tool: str, args: dict, s: WorldState) -> Optional[str]:
        if tool == TOOL_ASSIGN_BERTH:
            ship = s.ship(args["call_id"])
            if ship is None or ship.berth != args["berth_id"]:
                return f"{args['call_id']} is not on {args['berth_id']}"
        elif tool == TOOL_PLAN_CRANES:
            ship = s.ship(args["call_id"])
            if ship is None or sorted(ship.cranes) != sorted(args["crane_ids"]):
                return f"{args['call_id']} has cranes {list(ship.cranes) if ship else None}"
            bays = {c.crane_id: c.bay for c in s.cranes}
            wrong = [c for c, b in zip(args["crane_ids"], args["bay_positions"]) if bays.get(c) != b]
            if wrong:
                return f"cranes not at their bays: {wrong}"
        elif tool == TOOL_DISPATCH_TRUCKS:
            serving = {t.truck_id for t in s.trucks if t.call_id == args["call_id"]}
            if serving != set(args["truck_ids"]):
                return f"{args['call_id']} is served by {sorted(serving)}"
        elif tool == TOOL_PLACE:
            where = {c.container_id: c.yard_block for rows in s.manifests.values() for c in rows}
            wrong = [c for c in args["container_ids"] if where.get(c) != args["block_id"]]
            if wrong:
                return f"{len(wrong)} boxes not in block {args['block_id']}"
        elif tool == TOOL_RELEASE:
            pass
        return None
