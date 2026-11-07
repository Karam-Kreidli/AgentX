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
  - SAFE-4 (core/safety.py): agents, coordinator, supervisor and event detection run with
    a hard timeout on a worker thread; a call that hangs, raises or returns garbage is
    logged and that tick runs rules-only. Simulator calls stay on this thread.
  - Clock guard: if advance_time doesn't move the clock, retry with smaller steps; after
    safety.CLOCK_ATTEMPTS (6) failures stop cleanly. Run budget: at 70 % of RUN_BUDGET_S the
    LLM is switched off, at 100 % the rest of the run is rules-only. summary.json is always written.
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
from core import safety
from core.events import detect_events
from core.executor import DecisionLog, Executor
from core.skeleton import SkeletonAgent
from core.state import build_state, hours
from core.types import (EQ_AVAILABLE, EV_ANOMALY, SHIP_ARRIVED, SHIP_BERTHED, SHIP_DEPARTED,
                        SHIP_EXPECTED, SHIP_WORKING, Action, WorldState)

TICK_NORMAL_MIN = 15
TICK_BUSY_MIN = 5
TICK_BUDGET_S = 10.0          # watchdog: wall-clock seconds per tick before falling back to rules-only
MAX_ROUNDS = 3                # propose/execute rounds per tick
FIRST_SNAPSHOT_TRIES = 5      # SAFE-4: a failed first read is retried, not taken as the end of the run
_FAULT_COUNTS = {safety.EXCEPTION: "exceptions", safety.TIMEOUT: "timeouts", safety.GARBAGE: "garbage"}


def _optional(module: str):
    """A module another task builds, or None until it exists."""
    if module in sys.modules:
        return sys.modules[module]
    if importlib.util.find_spec(module) is None:
        return None
    return importlib.import_module(module)


class Loop:
    def __init__(self, api: Any, run_dir: Path, agents: Optional[list] = None,
                 tick_budget_s: float = TICK_BUDGET_S, max_ticks: int = 100_000,
                 call_timeout_s: Optional[float] = None, run_budget_s: Optional[float] = None):
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
        self.counts = {"exceptions": 0, "watchdog": 0, "breaches": 0, "rules_only_ticks": 0,
                       "timeouts": 0, "garbage": 0, "busy_ticks": 0, "clock_retries": 0}
        # SAFE-4
        self.calls = safety.Guarded()
        self.call_timeout_s = call_timeout_s or safety.call_timeout_s()
        self.budget = safety.RunBudget(run_budget_s)
        self.phase = safety.FULL
        self.llm_prev: Optional[bool] = None
        self.llm_switched = False
        self.stopped: Optional[str] = None       # why the run ended early, if it did

    # ------------------------------------------------------------------ helpers
    def _note(self, state, agent, choice, reason, saw=None):
        self.log.note(self.tick, state, agent, choice, reason, saw)

    def _call(self, state, key, label, fn, *args, timeout_s=None, default=None, check=None):
        """SAFE-4: run decision code with a hard timeout (see core/safety.py)."""
        out = self.calls.call(key, label, fn, *args, timeout_s=timeout_s or self.call_timeout_s,
                              default=default, check=check)
        if out.back:
            self._note(state, "loop", "recovered", f"{label} finished its earlier call; it is used again.")
        return out

    def _fault(self, state, agent, out: safety.Outcome, consequence: str) -> None:
        """Count and log a failed guarded call. Busy calls were logged when they timed out."""
        if out.kind == safety.BUSY:
            self.counts["busy_ticks"] += 1
            return
        self.counts[_FAULT_COUNTS.get(out.kind, "exceptions")] += 1
        if out.kind == safety.TIMEOUT:
            what = f"{out.label}: {out.error}; " + ("it is benched for the rest of the run" if out.benched
                                                   else "it is skipped until that call returns")
        elif out.kind == safety.GARBAGE:
            what = f"{out.label} returned garbage ({out.error})"
        else:
            what = out.error
        self._note(state, agent, out.kind, f"{what}; {consequence}", {"trace": out.trace} if out.trace else None)

    def _drain_escalations(self, state, agents):
        for a in agents:
            name = getattr(a, "name", "agent")
            try:
                esc = getattr(a, "escalations", None)
                if not esc:
                    continue
                msgs = list(esc)
                del esc[:len(msgs)]       # keep anything a still-running call appended meanwhile
            except Exception as e:
                self._note(state, name, "exception", f"could not read its escalations: {type(e).__name__}")
                continue
            for msg in msgs:
                self._note(state, name, "escalate", msg)

    def _rules_fallback(self, state, events, failed=None) -> list[Action]:
        """The rules-only agent's actions, also guarded. Nothing at all if it fails too."""
        if failed is self.rules:
            return []
        out = self._call(state, id(self.rules), "rules.propose", self.rules.propose, state, events,
                         check=safety.proposal)
        if out.ok:
            return out.value[0]
        self._fault(state, "rules", out, "no actions this tick.")
        return []

    def _propose(self, state, events, agents) -> tuple[list[Action], list, bool]:
        """Collect actions and requests. Returns (actions, requests, fell_back)."""
        actions, requests = [], []
        for a in agents:
            name = getattr(a, "name", "agent")
            out = self._call(state, id(a), f"{name}.propose", a.propose, state, events, check=safety.proposal)
            if not out.ok:
                self._fault(state, name, out, "this tick runs rules-only.")
                return self._rules_fallback(state, events, a), [], True
            acts, reqs = out.value
            actions += acts
            requests += reqs
        return actions, requests, False

    def _reread(self, before: WorldState) -> WorldState:
        """A fresh snapshot mid-tick. Failed reads it adds (vs `before`) are logged, not silent."""
        st = build_state(self.api)
        for a in st.anomalies:
            if a not in before.anomalies:
                self._note(st, "state", "anomaly", f"Re-read: {a}. Logged; the run carries on.")
        return st

    def _events(self, state):
        """detect_events with a hard timeout; data anomalies (failed reads, bad rows) are logged."""
        out = self._call(state, "events", "events.detect", detect_events, self.prev, state,
                         timeout_s=safety.EVENTS_TIMEOUT_S, default=[])
        if not out.ok:
            self._fault(state, "events", out, "no events this tick.")
            return []
        events = out.value if isinstance(out.value, list) else []
        for e in events:
            if getattr(e, "kind", None) == EV_ANOMALY:
                msg = (getattr(e, "detail", None) or {}).get("message", "")
                self._note(state, "state", "anomaly", f"Data anomaly ({e.subject}): {msg}. Logged; the run carries on.")
        return events

    def _coordinate(self, state, events, requests) -> list[Action]:
        """CORE-5 coordinator and LLM-4 supervisor, once they exist. Returns extra actions to run."""
        if not requests:
            return []
        coord = _optional("coordinator")
        if coord is None:
            return []
        out = self._call(state, "coordinator", "coordinator.decide", coord.decide, requests, state,
                         check=safety.coordination)
        if not out.ok:
            self._fault(state, "coordinator", out, "no coordinator actions this round.")
            return []
        grants, options = out.value
        for g in grants:
            self._note(state, "coordinator", f"grant:{g.status}", g.reason,
                       {"request": g.request, "granted": g.granted})
        if not options:
            return []
        chosen, sentence = options[0].id, options[0].template
        sup = _optional("llm.supervisor")
        if sup is not None:
            out = self._call(state, "supervisor", "supervisor.decide", sup.decide,
                             events[0] if events else None, options, check=safety.choice)
            if out.ok:
                chosen, sentence = out.value
            else:
                self._fault(state, "supervisor", out, "using option 0.")
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
        events = self._events(state)
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
                actions, requests = self._rules_fallback(state, events if rnd == 0 else []), []
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
                state = self.ex.state or state      # fresh after every sent call, failed ones too
                if "refused" not in res and "error" not in res:
                    sent_any = True
            if not sent_any:
                break
            state = self._reread(state)

        self._invariants(state)
        return state

    # ------------------------------------------------------------------ whole run
    def run(self) -> dict:
        self.budget.start()
        try:
            self._run()
        except Exception as e:                # SAFE-4: nothing outside step() may crash the run
            self.counts["exceptions"] += 1
            self.stopped = self.stopped or "error"
            self._note(self.prev, "loop", "exception", f"{type(e).__name__}: {e}"[:200] + "; stopping the run cleanly.",
                       {"trace": traceback.format_exc(limit=3)})
        finally:
            if self.llm_switched:
                safety.llm_restore(self.llm_prev)   # the next scenario starts with the LLM as configured
        return self.summary()

    def _run(self) -> None:
        no_state = 0
        while self.tick < self.max_ticks:
            self._budget()
            try:
                state = self.step()
                est_now = state.now
            except Exception as e:            # never crash: log, then keep the clock moving
                self.counts["exceptions"] += 1
                state = self.prev
                self._note(state, "loop", "exception", f"{type(e).__name__}: {e}"[:200],
                           {"trace": traceback.format_exc(limit=3)})
                self.rules_only_next = True
            if state is None:                 # not one snapshot yet (the first reads failed): try again
                no_state += 1
                if no_state >= FIRST_SNAPSHOT_TRIES:
                    self.stopped = "no_state"
                    self._note(None, "loop", "CRITICAL", f"Could not read the simulator in {no_state} tries; "
                               "stopping the run cleanly.")
                    return
                continue
            if state.now >= state.scenario_end:
                return
            if not self._advance(state):
                return
            self.tick += 1

    def _budget(self) -> None:
        """Run budget (SAFE-4): LLM off at 70 %, rules-only from 100 %."""
        phase = self.budget.phase()
        if phase != self.phase:
            used, total = self.budget.used_s(), self.budget.budget_s
            if not self.llm_switched:
                self.llm_prev, self.llm_switched = safety.llm_off(), True
                self._note(self.prev, "loop", "budget", f"{used:.0f} s of the {total:g} s run budget used; "
                           "the LLM is switched off for the rest of this run.")
            if phase == safety.RULES_ONLY:
                self._note(self.prev, "loop", "budget", f"Run budget of {total:g} s used up; "
                           "the rest of this run is rules-only.")
            self.phase = phase
        if phase == safety.RULES_ONLY:
            self.rules_only_next = True

    def _advance(self, state: WorldState) -> bool:
        """Advance the clock and check it moved (SAFE-4). False = stuck: stop the run cleanly."""
        res = safety.advance_clock(lambda m: self.api.call("advance_time", {"minutes": m}),
                                   lambda: self._reread(state).now, state.now, self._tick_minutes(state))
        for i, p in enumerate(res.problems):
            self.counts["clock_retries"] += 1
            last = not res.ok and i == len(res.problems) - 1
            self._note(state, "loop", "clock", f"{p}; re-read the clock" +
                       ("." if last else " and retried with a smaller step."))
        if res.lost_ack is not None:
            self._note(state, "loop", "clock", f"advance_time answered with an error ({res.lost_ack}), "
                       "but the clock did move; carrying on.")
        if not res.ok:
            self.stopped = "clock_stuck"
            self._note(state, "loop", "CRITICAL", f"The simulator clock is stuck at {state.now:%Y-%m-%d %H:%M} "
                       f"after {len(res.problems)} tries; stopping the run cleanly instead of spinning. "
                       "Unfinished ships stay unfinished.")
            return False
        return True

    def summary(self) -> dict:
        """KPIs plus our own counters. Never raises: summary.json is always written (SAFE-4)."""
        try:
            s = dict(self.api.run_summary())
        except Exception as e:
            s = {"scenario": getattr(self.api, "scenario", None),
                 "violations": list(getattr(self.api, "violations", None) or []),
                 "summary_error": f"run_summary failed: {type(e).__name__}: {e}"[:200]}
        try:
            st = build_state(self.api)
            honest = [((v.departed_at or st.scenario_end) - v.actual_arrival).total_seconds() / 3600
                      for v in st.ships if v.actual_arrival]
            s["honest_turnaround_h"] = round(sum(honest) / len(honest), 2) if honest else None
            s["unfinished_calls"] = [v.call_id for v in st.ships if v.status != SHIP_DEPARTED]
            if st.anomalies:
                s["final_state_anomalies"] = list(st.anomalies)[:20]
        except Exception as e:
            s.setdefault("summary_error", f"final state failed: {type(e).__name__}: {e}"[:200])
        s["skeleton_exceptions"] = self.counts["exceptions"]
        s["loop"] = {**self.counts, "ticks": self.tick, "actions_sent": self.ex.sent,
                     "actions_refused": self.ex.refused, "mismatches": self.ex.mismatches,
                     "lost_acks": self.ex.lost_acks, "stopped_early": self.stopped,
                     "benched": self.calls.benched_labels(), "budget": self.budget.report()}
        try:
            (self.run_dir / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            print(f"[loop] could not write summary.json: {type(e).__name__}: {e}", file=sys.stderr)
        return s
