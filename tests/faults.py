"""Fault injection for the crash-proofing tests (SAFE-4). Test-only code.

    from tests.faults import FaultyAPI, FaultyAgent
    api = FaultyAPI(TerminalAPI(...), seed=1, tool_error_rate=0.05)
    agents = [FaultyAgent(BerthAgent(), "raise", every=7), ...]

FaultyAPI wraps a simulator instance. It never changes the simulator's rules or files; it
only gets between our code and `api.call`, the way a flaky network or a buggy toolkit would:
    tool errors         a call is answered with {"error": ...} and never reaches the simulator
    lost acknowledgment a state-changing call reaches the simulator, but the answer is lost
    stuck clock         from some point on, advance_time stops moving the clock, either
                        silently ("silent": claims success) or loudly ("error")
Every fault it injects is recorded in `injected`, and its error text carries a unique tag
("#17"), so a test can check that each one was logged.

TinyAPI is a terminal with a clock and nothing else, for the tests that must run without
the workbook (CI doesn't have it, F8).

FaultyAgent wraps a real agent and, on every `every`-th propose(), raises, hangs, spins the
CPU or returns garbage. A hang waits on an event that `release()` sets, so no thread
outlives the test.
"""
from __future__ import annotations

import random
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from core.types import Action

MUTATING = {"assign_berth", "plan_crane_sequence", "dispatch_truck", "place_container",
            "release_to_consignee", "advance_time"}


class FaultyAPI:
    def __init__(self, api: Any, seed: int = 0, tool_error_rate: float = 0.0,
                 lost_ack_rate: float = 0.0, stuck_clock_after_h: Optional[float] = None,
                 stuck_mode: str = "silent", tools: Optional[set[str]] = None, grace_calls: int = 0):
        self._api = api
        self._rng = random.Random(seed)
        self.tool_error_rate = tool_error_rate
        self.lost_ack_rate = lost_ack_rate
        self.stuck_clock_after_h = stuck_clock_after_h
        self.stuck_mode = stuck_mode
        self.tools = tools                       # None = every tool
        self.grace_calls = grace_calls           # no random faults in the first n calls (first snapshot)
        self.injected: list[dict] = []
        self.calls = 0
        self._start = api.now

    def __getattr__(self, name: str) -> Any:     # violations, run_summary, now, ... from the real one
        return getattr(self._api, name)

    def _record(self, kind: str, tool: str, args: Any) -> str:
        tag = f"#{len(self.injected) + 1}"
        self.injected.append({"kind": kind, "tool": tool, "args": args, "tag": tag,
                              "sim_time": str(self._api.now)})
        return tag

    def _stuck(self) -> bool:
        if self.stuck_clock_after_h is None:
            return False
        return (self._api.now - self._start).total_seconds() / 3600 >= self.stuck_clock_after_h

    def call(self, name: str, arguments: Optional[dict] = None) -> Any:
        self.calls += 1
        if name == "advance_time" and self._stuck():
            tag = self._record("stuck_clock", name, arguments)
            if self.stuck_mode == "error":
                return {"error": f"injected {tag}: clock service unavailable"}
            return {"now": self._api.now.isoformat(), "finished": False}     # claims success, no move
        if self.calls > self.grace_calls and (self.tools is None or name in self.tools):
            if self.tool_error_rate and self._rng.random() < self.tool_error_rate:
                tag = self._record("tool_error", name, arguments)
                return {"error": f"injected {tag}: {name} unavailable"}
            if name in MUTATING and self.lost_ack_rate and self._rng.random() < self.lost_ack_rate:
                self._api.call(name, arguments)          # it happens ...
                tag = self._record("lost_ack", name, arguments)
                return {"error": f"injected {tag}: connection reset before the reply"}   # ... we never hear
        return self._api.call(name, arguments)


class TinyAPI:
    """An empty terminal whose clock works: no ships, one crane, one block. `hours` before
    the end of the 5-day window it starts. For fault tests that don't need the workbook."""

    def __init__(self, hours: float = 6.0):
        self.end = datetime(2026, 9, 6)
        self.now = self.end - timedelta(hours=hours)
        self.scenario = "tiny"
        self.violations: list = []
        self.calls: list[str] = []

    def call(self, name: str, arguments: Optional[dict] = None) -> Any:
        self.calls.append(name)
        if name == "get_current_time":
            return {"now": self.now.isoformat(), "scenario_end": self.end.isoformat(),
                    "finished": self.now >= self.end}
        if name == "advance_time":
            self.now = min(self.now + timedelta(minutes=int((arguments or {})["minutes"])), self.end)
            return {"now": self.now.isoformat(), "finished": self.now >= self.end}
        if name == "get_equipment_status":
            return [{"equipment_id": "QC1", "type": "quay crane", "status": "available", "bay": 1}]
        if name == "get_yard_status":
            return [{"block_id": "A", "block_type": "standard", "used": 0, "slots": 10}]
        if name in ("get_vessel_schedule", "get_customs_status", "get_manifest"):
            return []
        return {"error": f"unknown tool {name}"}

    def run_summary(self) -> dict:
        return {"scenario": self.scenario, "violations": list(self.violations),
                "vessels_departed": 0, "vessels_total": 0}


GARBAGE = (
    None,
    "garbage",
    42,
    ([None], []),
    (["assign_berth"], []),
    ({"tool": "assign_berth", "args": {}}, []),
    ([Action("format_disk", {}, "evil", "Wipe the terminal.")], []),
    ([Action("assign_berth", "B1", "evil", "args are not a mapping.")], []),
    ([], [object()]),
    ([], [], []),
    (iter(()), iter(())),
)


class FaultyAgent:
    """A real agent that misbehaves on every `every`-th call.
    mode: raise | hang | spin | garbage | slow (sleeps slow_s, then answers normally)."""

    def __init__(self, inner: Any, mode: str, every: int = 5, slow_s: float = 0.0, limit: Optional[int] = None):
        self.inner = inner
        self.mode = mode
        self.every = every
        self.slow_s = slow_s
        self.limit = limit                    # stop misbehaving after this many faults
        self.name = getattr(inner, "name", "agent")
        self.escalations = getattr(inner, "escalations", [])
        self.calls = 0
        self.faults = 0
        self.in_flight = 0                    # concurrent calls (must never exceed 1)
        self.max_in_flight = 0
        self._release = threading.Event()
        self._lock = threading.Lock()

    def release(self) -> None:
        self._release.set()

    def propose(self, state, events):
        with self._lock:
            self.calls += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            n = self.calls
        try:
            if n % self.every == 0 and (self.limit is None or self.faults < self.limit):
                self.faults += 1
                if self.mode == "raise":
                    raise RuntimeError(f"injected fault in {self.name}")
                if self.mode == "hang":
                    self._release.wait()
                    return [], []
                if self.mode == "spin":
                    while not self._release.is_set():
                        pass
                    return [], []
                if self.mode == "garbage":
                    return GARBAGE[(self.faults - 1) % len(GARBAGE)]
                if self.mode == "slow":
                    time.sleep(self.slow_s)
            return self.inner.propose(state, events)
        finally:
            with self._lock:
                self.in_flight -= 1
