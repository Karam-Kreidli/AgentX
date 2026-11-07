"""Crash-proofing (SAFE-4): hard timeouts, a clock-progress guard and a run budget.

    from core import safety
    out = safety.safe_call(agent.propose, state, events, timeout_s=10, label="berth.propose",
                           check=safety.proposal)
    out.ok, out.value, out.kind, out.error         # kind: ok | exception | timeout | garbage

    calls = safety.Guarded()                       # one per run
    out = calls.call(key, label, fn, *args, timeout_s=10, default=None, check=None)
    #   like safe_call, and a caller whose last call timed out is "busy" until that call returns;
    #   if it is still running a grace period later, its thread is stopped and it is benched
    calls.shutdown()                               # end of run: stop every thread still running

    budget = safety.RunBudget()                    # RUN_BUDGET_S, default 300 s
    budget.phase()                                 # full | no_llm (>= 70 %) | rules_only (>= 100 %)
    prev = safety.llm_off(); ...; safety.llm_restore(prev)

    res = safety.advance_clock(advance, read_now, before, minutes)   # ClockResult(ok, now, problems)

Why these, in GROUND_RULES order ("never crash" is rule 2, and a hang is a crash on the day):
- safe_call runs untrusted decision code (agents, coordinator, supervisor, event hooks and
  the LLM calls inside them) on a daemon thread and waits at most timeout_s. Not a
  ThreadPoolExecutor (its threads block interpreter exit) and not signal.alarm (Unix only,
  main thread only). Any exception, even SystemExit, comes back as data.
- A call that times out keeps running on its thread: Python can't kill a thread cleanly.
  Guarded never calls the same caller again while it runs, so one agent object never runs on
  two threads at once. Then, whichever comes first:
    * the call finishes on its own -> "back": its state is consistent, it is used again;
    * it is still running KILL_AFTER_S later (a CPU spin or a real hang) -> its thread is
      stopped (an async exception, delivered at its next bytecode, so a spin stops at once; a
      thread blocked in sleep or I/O stops when that returns) and the caller is BENCHED for
      the rest of the run. A thread stopped from outside may leave the agent's state
      half-updated (its own cleanup is skipped), so a killed agent is never reused.
  After MAX_TIMEOUTS timeouts a caller is benched too, which caps the wall time a flaky agent
  can cost. shutdown() stops whatever is still running at the end of a run, so a spinning
  thread can't slow the next scenario.
- Simulator calls never go to a worker thread: a thread we gave up on must not be able to
  change the simulator. Agents can't call it anyway (CONTRACTS.md), and the loop and executor
  call it on the main thread. This module never calls the simulator itself.
- proposal / coordination / choice check the *shape* of what came back (garbage = wrong
  types, unknown tool, more than MAX_ITEMS items). They don't judge whether an action is
  safe: that is guard.check's job, and the executor still refuses actions with no reason.
- advance_clock: if the clock doesn't move, re-read it (the reply may be lost while the
  clock did move), then retry with smaller steps. After CLOCK_ATTEMPTS failures the caller
  stops the run cleanly instead of spinning through max_ticks.
- RunBudget: each scenario gets RUN_BUDGET_S (default 300 s) of wall time, GROUND_RULES §1.9
  ("under 5 minutes per scenario"); 3 hidden scenarios then use 15 of the 20 minutes. At 70 %
  the LLM is switched off; at 100 % the rest of the run is rules-only.

Never raises: everything here returns data. The caller logs every fault (see core/loop.py).
"""
from __future__ import annotations

import ctypes
import math
import os
import threading
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Hashable, Optional

from core.types import AGENT_TOOLS, Action, Grant, Option, Request

LLM_TIMEOUT_S = 8.0         # llm/client.py default for one ask(), all attempts included
CALL_MARGIN_S = 2.0         # a guarded call may take one full LLM ask plus this
EVENTS_TIMEOUT_S = 5.0      # event detection and its hooks (stall detector, operator inbox)
MAX_TIMEOUTS = 3            # timeouts before a caller is benched for the rest of the run
KILL_AFTER_S = None         # grace after a timeout before the thread is stopped (None = one more
                            # timeout_s of that call, so 10 s by default)
MAX_ITEMS = 500             # actions / requests / options in one answer (busiest tick so far: 16)
RUN_BUDGET_S = 300.0        # GROUND_RULES §1.9: under 5 minutes per scenario
LLM_OFF_AT = 0.7            # fraction of the run budget at which the LLM is switched off
RULES_ONLY_AT = 1.0         # fraction at which the rest of the run is rules-only
CLOCK_ATTEMPTS = 6          # advance_time tries before the run stops cleanly. Not 3: with 5 %
                            # transient errors, 3 in a row happen in ~1 run of 6 (~1,400 ticks);
                            # 6 in a row ~1 in 40,000. A clock that is really stuck costs ms.

OK, EXCEPTION, TIMEOUT, GARBAGE, BUSY = "ok", "exception", "timeout", "garbage", "busy"
FULL, NO_LLM, RULES_ONLY = "full", "no_llm", "rules_only"


class GarbageError(ValueError):
    """A guarded call returned something of the wrong shape."""


class Cancelled(BaseException):
    """Raised inside a worker thread we gave up on. BaseException, so an agent's own
    `except Exception` can't swallow it and carry on."""


# --------------------------------------------------------------------------
# Hard timeouts
# --------------------------------------------------------------------------

@dataclass
class Outcome:
    ok: bool
    value: Any                       # the checked result, or the default
    kind: str = OK                   # ok | exception | timeout | garbage | busy
    error: str = ""                  # one line, for the log
    trace: Optional[str] = None      # short traceback for exceptions
    elapsed_s: float = 0.0
    label: str = ""
    thread: Optional[threading.Thread] = field(default=None, repr=False)   # still running after a timeout
    benched: bool = False            # this call benched the caller for the rest of the run
    back: bool = False               # first call after an earlier timed-out call finished ON ITS OWN
    killed: bool = False             # this call stopped the caller's still-running thread (and benched it)


def call_timeout_s() -> float:
    """Hard timeout for one agent / coordinator / supervisor call: one LLM ask plus a margin."""
    try:
        t = float(os.environ.get("LLM_TIMEOUT_S", LLM_TIMEOUT_S))
    except ValueError:
        t = LLM_TIMEOUT_S
    return (t if math.isfinite(t) and t > 0 else LLM_TIMEOUT_S) + CALL_MARGIN_S


def _one_line(e: BaseException) -> str:
    return " ".join(f"{type(e).__name__}: {e}".split())[:200]


def cancel(thread: Optional[threading.Thread]) -> bool:
    """Ask a worker thread to stop by raising Cancelled inside it. Best effort; never raises."""
    try:
        if thread is None or not thread.is_alive() or thread.ident is None:
            return False
        set_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
        n = set_exc(ctypes.c_ulong(thread.ident), ctypes.py_object(Cancelled))
        if n > 1:                                   # should never happen: undo it
            set_exc(ctypes.c_ulong(thread.ident), None)
            return False
        return n == 1
    except Exception:
        return False


def safe_call(fn: Callable[..., Any], *args: Any, timeout_s: float, default: Any = None,
              label: str = "call", check: Optional[Callable[[Any], Any]] = None, **kwargs: Any) -> Outcome:
    """Run fn(*args, **kwargs) on a daemon thread; give up after timeout_s.

    check(value) runs on the same thread, inside the timeout, so a check that consumes the
    answer can't hang the caller either. It returns the cleaned value or raises GarbageError.
    A call that times out is left running (out.thread); stopping it is the caller's choice,
    with cancel() (Guarded does it after a grace period and benches the caller).
    """
    if not (isinstance(timeout_s, (int, float)) and math.isfinite(timeout_s) and timeout_s > 0):
        timeout_s = call_timeout_s()
    box: dict[str, Any] = {}
    done = threading.Event()

    def work() -> None:
        try:
            try:
                value = fn(*args, **kwargs)
                box["value"] = check(value) if check is not None else value
            except Cancelled:
                box["cancelled"] = True
            except GarbageError as e:
                box["garbage"] = e
            except BaseException as e:      # noqa: BLE001 - every failure becomes data
                box["error"] = e
                box["trace"] = traceback.format_exc(limit=4)
            finally:
                done.set()
        except BaseException:               # a Cancelled landing inside a handler above:
            pass                            # nobody is waiting any more, nothing to report

    t0 = time.perf_counter()
    th = threading.Thread(target=work, name=f"safe:{label}"[:60], daemon=True)
    try:
        th.start()
    except Exception as e:                   # e.g. can't start a new thread
        return Outcome(False, default, EXCEPTION, f"could not start a worker thread: {_one_line(e)}",
                       label=label)
    finished = done.wait(timeout_s)
    elapsed = time.perf_counter() - t0
    if not finished:
        return Outcome(False, default, TIMEOUT, f"no answer within {timeout_s:g} s", elapsed_s=elapsed,
                       label=label, thread=th)
    if "error" in box:
        return Outcome(False, default, EXCEPTION, _one_line(box["error"]), box.get("trace"), elapsed, label)
    if "garbage" in box:
        return Outcome(False, default, GARBAGE, str(box["garbage"])[:200], elapsed_s=elapsed, label=label)
    if box.get("cancelled"):
        return Outcome(False, default, EXCEPTION, "cancelled", elapsed_s=elapsed, label=label)
    return Outcome(True, box.get("value"), OK, elapsed_s=elapsed, label=label)


class Guarded:
    """safe_call with memory, one per run. A caller (key) whose last call timed out is BUSY
    until that call finishes on its own ("back"). If it is still running kill_after_s after the
    timeout, its thread is stopped and the caller is benched for the rest of the run. Benched
    for good after MAX_TIMEOUTS timeouts too."""

    def __init__(self, max_timeouts: int = MAX_TIMEOUTS, kill_after_s: Optional[float] = KILL_AFTER_S,
                 clock: Callable[[], float] = time.monotonic):
        self.max_timeouts = max_timeouts
        self.kill_after_s = kill_after_s
        self._clock = clock
        self._running: dict[Hashable, tuple[threading.Thread, float]] = {}   # key -> (thread, kill at)
        self._labels: dict[Hashable, str] = {}
        self.timeouts: dict[Hashable, int] = {}
        self.benched: set[Hashable] = set()
        self.killed: set[Hashable] = set()

    def benched_labels(self) -> list[str]:
        return sorted(self._labels.get(k, str(k)) for k in self.benched)

    def is_busy(self, key: Hashable) -> bool:
        if key in self.benched:
            return True
        run = self._running.get(key)
        return run is not None and run[0].is_alive()

    def _kill(self, key: Hashable) -> None:
        th, _ = self._running.pop(key)
        cancel(th)
        self.killed.add(key)
        self.benched.add(key)

    def call(self, key: Hashable, label: str, fn: Callable[..., Any], *args: Any, timeout_s: float,
             default: Any = None, check: Optional[Callable[[Any], Any]] = None, **kwargs: Any) -> Outcome:
        if key in self.benched:
            return Outcome(False, default, BUSY, f"{label} is benched for the rest of the run", label=label)
        run = self._running.get(key)
        back = False
        if run is not None:
            th, kill_at = run
            if th.is_alive():
                if self._clock() < kill_at:
                    return Outcome(False, default, BUSY, f"{label} has not finished its last call", label=label)
                late = max(0.0, self._clock() - kill_at)
                self._kill(key)
                return Outcome(False, default, BUSY, f"{late + self._grace(timeout_s):.1f} s after its timeout",
                               label=label, benched=True, killed=True)
            del self._running[key]
            back = True                       # finished on its own: its state is consistent
        out = safe_call(fn, *args, timeout_s=timeout_s, default=default, label=label, check=check, **kwargs)
        out.back = back
        if out.kind == TIMEOUT:
            self._labels[key] = label
            self.timeouts[key] = self.timeouts.get(key, 0) + 1
            if out.thread is not None:
                self._running[key] = (out.thread, self._clock() + self._grace(timeout_s))
            if self.timeouts[key] >= self.max_timeouts:
                self.benched.add(key)
                out.benched = True
        return out

    def _grace(self, timeout_s: float) -> float:
        g = self.kill_after_s
        return float(g) if isinstance(g, (int, float)) and math.isfinite(g) and g >= 0 else float(timeout_s)

    def shutdown(self) -> list[str]:
        """End of run: stop every thread still running. Returns their labels. Never raises."""
        stopped = []
        for key in list(self._running):
            try:
                th, _ = self._running[key]
                if th.is_alive():
                    self._kill(key)
                    stopped.append(self._labels.get(key, str(key)))
                else:
                    del self._running[key]
            except Exception:
                pass
        return sorted(stopped)


# --------------------------------------------------------------------------
# Shape checks for what guarded calls return (garbage in -> GarbageError)
# --------------------------------------------------------------------------

def _seq(x: Any, what: str) -> list:
    if not isinstance(x, (list, tuple)):
        raise GarbageError(f"{what} is a {type(x).__name__}, not a list")
    if len(x) > MAX_ITEMS:
        raise GarbageError(f"{what} has {len(x)} items (more than {MAX_ITEMS})")
    return list(x)


def _action(a: Any, what: str) -> Action:
    if not isinstance(a, Action):
        raise GarbageError(f"{what} contains a {type(a).__name__}, not an Action")
    if not isinstance(a.tool, str) or a.tool not in AGENT_TOOLS:
        raise GarbageError(f"{what} contains an action for unknown tool {str(a.tool)[:40]!r}")
    if not isinstance(a.args, Mapping):
        raise GarbageError(f"{what} contains an action whose args are not a mapping")
    if not isinstance(a.reason, str) or not isinstance(a.agent, str):
        raise GarbageError(f"{what} contains an action whose agent or reason is not text")
    return a


def proposal(value: Any) -> tuple[list[Action], list[Request]]:
    """Agent.propose(...) -> (list[Action], list[Request])."""
    pair = _seq(value, "the answer")
    if len(pair) != 2:
        raise GarbageError(f"the answer has {len(pair)} parts, not (actions, requests)")
    actions = [_action(a, "actions") for a in _seq(pair[0], "actions")]
    requests = _seq(pair[1], "requests")
    for r in requests:
        if not isinstance(r, Request):
            raise GarbageError(f"requests contains a {type(r).__name__}, not a Request")
    return actions, requests


def coordination(value: Any) -> tuple[list[Grant], list[Option]]:
    """coordinator.decide(...) -> (list[Grant], list[Option])."""
    pair = _seq(value, "the answer")
    if len(pair) != 2:
        raise GarbageError(f"the answer has {len(pair)} parts, not (grants, options)")
    grants = _seq(pair[0], "grants")
    for g in grants:
        if not isinstance(g, Grant):
            raise GarbageError(f"grants contains a {type(g).__name__}, not a Grant")
    options = _seq(pair[1], "options")
    for o in options:
        if not isinstance(o, Option) or not isinstance(o.id, str) or not isinstance(o.template, str):
            raise GarbageError("options contains something that is not a complete Option")
        for a in _seq(o.actions, f"option {o.id[:20]!r} actions"):
            _action(a, f"option {o.id[:20]!r}")
    return grants, options


def choice(value: Any) -> tuple[str, str]:
    """supervisor.decide(...) -> (option_id, sentence)."""
    pair = _seq(value, "the answer")
    if len(pair) != 2 or not all(isinstance(x, str) for x in pair):
        raise GarbageError("the answer is not (option_id, sentence)")
    return pair[0], pair[1]


# --------------------------------------------------------------------------
# Run budget and the LLM switch
# --------------------------------------------------------------------------

def run_budget_s() -> float:
    try:
        b = float(os.environ.get("RUN_BUDGET_S", RUN_BUDGET_S))
    except ValueError:
        return RUN_BUDGET_S
    return b if math.isfinite(b) and b > 0 else RUN_BUDGET_S


class RunBudget:
    """Wall-clock budget for one scenario run."""

    def __init__(self, budget_s: Optional[float] = None, clock: Callable[[], float] = time.monotonic):
        ok = isinstance(budget_s, (int, float)) and math.isfinite(budget_s) and budget_s > 0
        self.budget_s = float(budget_s) if ok else run_budget_s()
        self._clock = clock
        self._t0: Optional[float] = None
        self.reached: dict[str, float] = {}      # phase -> seconds into the run when first reached

    def start(self) -> None:
        self._t0 = self._clock()

    def used_s(self) -> float:
        return 0.0 if self._t0 is None else max(0.0, self._clock() - self._t0)

    def phase(self) -> str:
        used = self.used_s()
        frac = used / self.budget_s
        p = RULES_ONLY if frac >= RULES_ONLY_AT else NO_LLM if frac >= LLM_OFF_AT else FULL
        if p != FULL and p not in self.reached:
            self.reached[p] = round(used, 1)
            if p == RULES_ONLY:
                self.reached.setdefault(NO_LLM, round(used, 1))
        return p

    def report(self) -> dict:
        return {"budget_s": self.budget_s, "used_s": round(self.used_s(), 1),
                "llm_off_at_s": self.reached.get(NO_LLM), "rules_only_at_s": self.reached.get(RULES_ONLY)}


def llm_off() -> Optional[bool]:
    """Switch the shared LLM client off; returns its previous setting (None if unavailable)."""
    try:
        from llm import client
        c = client.get_client()
        prev = bool(c.enabled)
        c.enabled = False
        return prev
    except Exception:
        return None


def llm_restore(prev: Optional[bool]) -> None:
    """Undo llm_off() so the next scenario starts with the LLM as configured."""
    if prev is None:
        return
    try:
        from llm import client
        client.get_client().enabled = prev
    except Exception:
        pass


# --------------------------------------------------------------------------
# Clock-progress guard
# --------------------------------------------------------------------------

@dataclass
class ClockResult:
    ok: bool
    now: Optional[datetime]
    problems: list[str] = field(default_factory=list)   # one per failed attempt
    lost_ack: Optional[str] = None                      # the reply was this error, yet the clock moved


def _when(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except (TypeError, ValueError):
        return None


def clock_steps(minutes: int, attempts: int = CLOCK_ATTEMPTS) -> list[int]:
    """The step tried on each attempt: the planned one, then smaller ones (15, 5, 1, 1, ...)."""
    m = max(1, int(minutes))
    steps = [m]
    while len(steps) < attempts:
        steps.append(max(1, steps[-1] // 3))
    return steps


def advance_clock(advance: Callable[[int], Any], read_now: Callable[[], Any], before: datetime,
                  minutes: int, attempts: int = CLOCK_ATTEMPTS) -> ClockResult:
    """Move the simulated clock forward and check that it moved.

    advance(m) calls the simulator's advance_time; read_now() re-reads the clock. Both are
    passed in so this module never calls the simulator itself.
    """
    problems: list[str] = []
    for m in clock_steps(minutes, attempts):
        try:
            res = advance(m)
        except Exception as e:                 # api.call should never raise; be safe anyway
            res = {"error": _one_line(e)}
        err = res.get("error") if isinstance(res, Mapping) else None
        now = _when(res.get("now")) if isinstance(res, Mapping) and err is None else None
        try:
            if now is not None and now > before:
                return ClockResult(True, now, problems)
        except TypeError:                       # naive vs aware datetimes: trust the re-read
            pass
        try:
            seen = _when(read_now())            # the reply may be lost while the clock did move
            if seen is not None and seen > before:
                return ClockResult(True, seen, problems,
                                   lost_ack=None if err is None else str(err)[:120])
        except Exception:
            pass
        problems.append(f"advance_time({m} min): " +
                        (f"error {str(err)[:80]!r}" if err is not None else "the clock did not move"))
    return ClockResult(False, None, problems)
