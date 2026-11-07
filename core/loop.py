"""Main loop (CORE-4).

    from core.loop import Loop
    summary = Loop(api, run_dir).run()

Every tick:
  1. build state            core.state.build_state            (CORE-3)
  2. detect events          core.events.detect_events         (CORE-3)
  3. agents propose         each agent's propose(state, events) -> (actions, requests)
  4. coordinator            coordinator.decide(requests, state) -> (grants, options)   (CORE-5, once it exists)
  5. supervisor             llm.supervisor.decide(event, options) -> (option_id, sentence)  (LLM-4, once it exists)
  6. execute                core.executor: guard -> simulator -> log -> re-read
  7. invariant monitor      core.invariants.check_state(state) -> breaches   (SAFE-2, once it exists)
  8. advance the clock      5 min while a ship is arriving or a crane is idle, else 15 min

Safety:
  - An agent that raises is logged; that tick falls back to the rules-only agent.
  - Watchdog: if a tick takes more than TICK_BUDGET_S of wall time, the rest of the
    tick runs on rules-only decisions.
  - An invariant breach is logged as CRITICAL and the next tick runs rules-only.
  - Each tick may run a few propose/execute rounds, so a ship berthed this tick
    also gets its cranes, trucks and yard slots this tick.
Modules from later tasks are called once their files exist (see CONTRACTS.md).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

from core import log as runlog
from core.events import detect_events
from core.executor import DecisionLog, Executor
from core.skeleton import SkeletonAgent
from core.state import build_state, hours
from core.types import (EQ_AVAILABLE, SHIP_ARRIVED, SHIP_BERTHED, SHIP_DEPARTED, SHIP_EXPECTED,
                        SHIP_WORKING, Action, WorldState)

TICK_NORMAL_MIN = 15
TICK_BUSY_MIN = 5
TICK_BUDGET_S = 10.0          # watchdog: wall-clock seconds per tick before falling back to rules-only
MAX_ROUNDS = 3                # propose/execute rounds per tick
NO_STATE_MAX_TICKS = 5 * 24 * 60 // TICK_NORMAL_MIN   # a 5-day window at 15 min: stop if no snapshot ever succeeds


def _optional(module: str):
    """A module another task builds, or None until it exists."""
    if module in sys.modules:
        return sys.modules[module]
    if importlib.util.find_spec(module) is None:
        return None
    return importlib.import_module(module)


class Loop:
    def __init__(self, api: Any, run_dir: Path, agents: Optional[list] = None,
                 tick_budget_s: float = TICK_BUDGET_S, max_ticks: int = 100_000):
        self.api = api
        self.run_dir = Path(run_dir)
        self.log = DecisionLog(self.run_dir)
        self.ex = Executor(api, self.run_dir, self.log)
        self.rules = SkeletonAgent()
        self.agents = agents if agents is not None else [self.rules]
        self.tick_budget_s = tick_budget_s
        self.max_ticks = max_ticks
        self.tick = 0
        self.prev: Optional[WorldState] = None
        self.rules_only_next = False
        self.counts = {"exceptions": 0, "watchdog": 0, "breaches": 0, "rules_only_ticks": 0}

    # ------------------------------------------------------------------ helpers
    def _note(self, state, agent, choice, reason, saw=None):
        self.log.note(self.tick, state, agent, choice, reason, saw)

    def _drain_escalations(self, state, agents):
        for a in agents:
            for msg in getattr(a, "escalations", ())[:]:
                self._note(state, getattr(a, "name", "agent"), "escalate", msg)
            if hasattr(a, "escalations"):
                a.escalations.clear()

    def _propose(self, state, events, agents) -> tuple[list[Action], list, bool]:
        """Collect actions and requests. Returns (actions, requests, fell_back)."""
        actions, requests = [], []
        for a in agents:
            try:
                acts, reqs = a.propose(state, events)
                actions += list(acts)
                requests += list(reqs)
            except Exception as e:
                self.counts["exceptions"] += 1
                self._note(state, getattr(a, "name", "agent"), "exception",
                           f"{type(e).__name__}: {e}"[:200] + "; this tick runs rules-only.",
                           {"trace": traceback.format_exc(limit=3)})
                acts, _ = self.rules.propose(state, events)
                return list(acts), [], True
        return actions, requests, False

    def _coordinate(self, state, events, requests) -> list[Action]:
        """CORE-5 coordinator and LLM-4 supervisor, once they exist. Returns extra actions to run."""
        if not requests:
            return []
        coord = _optional("coordinator")
        if coord is None:
            return []
        grants, options = coord.decide(requests, state)
        for g in grants:
            self._note(state, "coordinator", f"grant:{g.status}", g.reason,
                       {"request": g.request, "granted": g.granted})
        if not options:
            return []
        chosen, sentence = options[0].id, options[0].template
        sup = _optional("llm.supervisor")
        if sup is not None:
            try:
                chosen, sentence = sup.decide(events[0] if events else None, options)
            except Exception as e:
                self._note(state, "supervisor", "exception", f"{type(e).__name__}: {e}"[:200] + "; using option 0.")
        pick = next((o for o in options if o.id == chosen), options[0])
        self._note(state, "supervisor", f"option:{pick.id}", sentence, {"options": [o.id for o in options]})
        return list(pick.actions)

    def _invariants(self, state) -> None:
        inv = _optional("core.invariants")
        if inv is None:
            return
        try:
            breaches = inv.check_state(state)
        except Exception as e:
            breaches = [f"invariant monitor failed: {type(e).__name__}: {e}"]
        if breaches:
            self.counts["breaches"] += len(breaches)
            self.rules_only_next = True
            for b in breaches:
                self._note(state, "invariants", "CRITICAL", str(b)[:300] + " Next tick runs rules-only.")

    def _watchdog(self, state) -> bool:
        self.counts["watchdog"] += 1
        self._note(state, "loop", "watchdog",
                   f"Tick took over {self.tick_budget_s:g} s; finishing it on rules-only decisions.")
        return True

    @staticmethod
    def _tick_minutes(state: WorldState) -> int:
        """5 minutes while a ship is arriving or a crane is idle, else 15."""
        soon = state.now + timedelta(minutes=TICK_NORMAL_MIN)
        arriving = any(s.status == SHIP_ARRIVED or
                       (s.status == SHIP_EXPECTED and s.scheduled_eta and s.scheduled_eta <= soon)
                       for s in state.ships)
        waiting_for_work = any(s.status in (SHIP_ARRIVED, SHIP_BERTHED, SHIP_WORKING) for s in state.ships)
        crane_idle = waiting_for_work and any(c.status == EQ_AVAILABLE for c in state.cranes)
        return TICK_BUSY_MIN if (arriving or crane_idle) else TICK_NORMAL_MIN

    # ------------------------------------------------------------------ one tick
    def step(self) -> WorldState:
        t0 = time.perf_counter()
        state = build_state(self.api)
        runlog.tick_snapshot(self.tick, state)          # SAFE-3: ticks.jsonl for the timeline
        events = detect_events(self.prev, state)
        self.prev = state
        self.ex.tick = self.tick
        rules_only = self.rules_only_next
        self.rules_only_next = False
        if rules_only:
            self.counts["rules_only_ticks"] += 1

        for rnd in range(MAX_ROUNDS):
            agents = [self.rules] if rules_only else self.agents
            actions, requests, fell_back = self._propose(state, events if rnd == 0 else [], agents)
            rules_only = rules_only or fell_back
            if not rules_only and time.perf_counter() - t0 > self.tick_budget_s:
                rules_only = self._watchdog(state)
                actions, requests = list(self.rules.propose(state, events if rnd == 0 else [])[0]), []
            if not rules_only:
                try:
                    actions += self._coordinate(state, events if rnd == 0 else [], requests)
                except Exception as e:
                    self.counts["exceptions"] += 1
                    self._note(state, "coordinator", "exception", f"{type(e).__name__}: {e}"[:200])
            self._drain_escalations(state, agents)
            if not actions:
                break
            sent_any = False
            for a in actions:
                if not rules_only and time.perf_counter() - t0 > self.tick_budget_s:
                    rules_only = self._watchdog(state)
                    break
                res = self.ex.execute(a, state)
                if "refused" not in res and "error" not in res:
                    sent_any = True
                    state = self.ex.state
            if not sent_any:
                break
            state = build_state(self.api)

        self._invariants(state)
        return state

    # ------------------------------------------------------------------ whole run
    def run(self) -> dict:
        est_now = None                        # our own clock: last snapshot time + minutes advanced since
        while self.tick < self.max_ticks:
            try:
                state = self.step()
                est_now = state.now
            except Exception as e:            # never crash: log, then keep the clock moving
                self.counts["exceptions"] += 1
                state = self.prev
                if state is not None:
                    self._note(state, "loop", "exception", f"{type(e).__name__}: {e}"[:200],
                               {"trace": traceback.format_exc(limit=3)})
                self.rules_only_next = True
            if state is None:
                # No snapshot yet (e.g. the very first clock read failed). Keep the clock moving
                # and try again next tick, but give up after a full scenario's worth of ticks.
                if self.tick >= NO_STATE_MAX_TICKS:
                    break
                minutes = TICK_NORMAL_MIN
            else:
                # A failed tick leaves `state` at the previous snapshot, so compare our own clock
                # estimate with the end of the window, not the stale snapshot time.
                if (est_now or state.now) >= state.scenario_end:
                    break
                minutes = self._tick_minutes(state)
            self.api.call("advance_time", {"minutes": minutes})
            if est_now is not None:
                est_now += timedelta(minutes=minutes)
            self.tick += 1
        return self.summary()

    def summary(self) -> dict:
        try:
            s = self.api.run_summary()
        except Exception as e:                # never crash, even at the very end
            s = {"summary_error": f"{type(e).__name__}: {e}"[:200]}
        try:
            st = build_state(self.api)
        except Exception as e:
            st = self.prev                    # last good snapshot, if any
            s["state_error"] = f"{type(e).__name__}: {e}"[:200]
        if st is not None:
            honest = [((v.departed_at or st.scenario_end) - v.actual_arrival).total_seconds() / 3600
                      for v in st.ships if v.actual_arrival]
            s["honest_turnaround_h"] = round(sum(honest) / len(honest), 2) if honest else None
            s["unfinished_calls"] = [v.call_id for v in st.ships if v.status != SHIP_DEPARTED]
        else:
            s["honest_turnaround_h"] = None
            s["unfinished_calls"] = []
        s["skeleton_exceptions"] = self.counts["exceptions"]
        s["loop"] = {**self.counts, "ticks": self.tick, "actions_sent": self.ex.sent,
                     "actions_refused": self.ex.refused, "mismatches": self.ex.mismatches}
        (self.run_dir / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
        return s
