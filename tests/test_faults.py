"""SAFE-4 tests: crash-proofing (core/safety.py, its wiring in core/loop.py, tests/faults.py).

TASKS.md "Done when": fault injection (exceptions in each agent, tool errors in 5 % of
calls) never stops a run. plan.md adds lost replies, agents that hang, spin or return
garbage, and a stuck clock: every run finishes with 0 violations, and every fault is logged.

Parts 1 and 2 need nothing (they run in CI without the workbook). Part 3 runs the real
practice scenarios and needs the workbook in data/.
"""
import json
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

from core import log, safety
from core.loop import Loop
from core.skeleton import SkeletonAgent
from core.types import Action, Grant, Option, Request
from tests.faults import GARBAGE, FaultyAgent, FaultyAPI, TinyAPI

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")

ACTION = Action("assign_berth", {"call_id": "C1", "berth_id": "B1"}, "berth", "arrived and B1 is free")
REQUEST = Request("trucks", "trucks", 3, "C1", 1.0, "one crane needs three trucks")


@pytest.fixture
def faulty():
    """Make FaultyAgents; every hung or spinning call is released when the test ends."""
    made = []

    def make(inner, mode, **kw):
        a = FaultyAgent(inner, mode, **kw)
        made.append(a)
        return a
    yield make
    for a in made:
        a.release()


def _decisions(run_dir):
    return [json.loads(line) for line in (Path(run_dir) / "decisions.jsonl").read_text().splitlines()]


def _summary(run_dir):
    return json.loads((Path(run_dir) / "summary.json").read_text())


def _no_live_workers(timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not [t for t in threading.enumerate() if t.name.startswith("safe:") and t.is_alive()]:
            return True
        time.sleep(0.02)
    return False


# =========================================================================
# Part 1: core/safety.py
# =========================================================================

def test_safe_call_returns_the_value():
    out = safety.safe_call(lambda a, b=0: a + b, 2, b=3, timeout_s=1, label="add")
    assert out.ok and out.value == 5 and out.kind == safety.OK and out.label == "add"


@pytest.mark.parametrize("exc", [RuntimeError("agent bug"), SystemExit(3), RecursionError("deep"),
                                 KeyboardInterrupt()])
def test_safe_call_turns_any_exception_into_data(exc):
    def boom():
        raise exc
    out = safety.safe_call(boom, timeout_s=1, default="fallback", label="boom")
    assert not out.ok and out.kind == safety.EXCEPTION and out.value == "fallback"
    assert type(exc).__name__ in out.error and out.trace


def test_safe_call_gives_up_on_a_hang_and_returns_the_default():
    gate = threading.Event()
    try:
        t0 = time.perf_counter()
        out = safety.safe_call(gate.wait, timeout_s=0.1, default=([], []), label="hang")
        assert time.perf_counter() - t0 < 1.0
        assert out.kind == safety.TIMEOUT and out.value == ([], []) and "0.1 s" in out.error
        assert out.thread is not None and out.thread.daemon      # a daemon: it can't block exit
    finally:
        gate.set()


def test_safe_call_leaves_a_timed_out_call_running_and_cancel_stops_a_spin():
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            pass
    try:
        out = safety.safe_call(spin, timeout_s=0.1, label="spin")
        assert out.kind == safety.TIMEOUT
        assert out.thread.is_alive()               # not killed by safe_call itself
        assert safety.cancel(out.thread)
        out.thread.join(2.0)
        assert not out.thread.is_alive()           # stopped by the cancel, not by `stop`
    finally:
        stop.set()


def test_check_runs_inside_the_timeout():
    gate = threading.Event()
    try:
        out = safety.safe_call(lambda: 1, timeout_s=0.1, check=lambda v: gate.wait(), label="slow check")
        assert out.kind == safety.TIMEOUT
    finally:
        gate.set()

    def bad(v):
        raise safety.GarbageError("not a proposal")
    out = safety.safe_call(lambda: 1, timeout_s=1, check=bad, default="d")
    assert out.kind == safety.GARBAGE and out.value == "d" and "not a proposal" in out.error


def test_proposal_accepts_a_valid_answer():
    assert safety.proposal(([ACTION], [REQUEST])) == ([ACTION], [REQUEST])
    assert safety.proposal(((), ())) == ([], [])


@pytest.mark.parametrize("value", GARBAGE, ids=[f"garbage{i}" for i in range(len(GARBAGE))])
def test_proposal_rejects_garbage(value):
    with pytest.raises(safety.GarbageError):
        safety.proposal(value)


def test_proposal_rejects_an_answer_too_big_to_execute():
    with pytest.raises(safety.GarbageError, match="more than"):
        safety.proposal(([ACTION] * (safety.MAX_ITEMS + 1), []))


def test_coordination_and_choice_shapes():
    opt = Option("opt0", (ACTION,), {}, "Berth C1 on B1 now.")
    grant = Grant(REQUEST, 3, "granted", "trucks are free")
    assert safety.coordination(([grant], [opt])) == ([grant], [opt])
    with pytest.raises(safety.GarbageError):
        safety.coordination(([grant], [Option("x", ("rm -rf",), {}, "t")]))
    with pytest.raises(safety.GarbageError):
        safety.coordination(([{"granted": 3}], []))
    assert safety.choice(("opt0", "Because it is safest.")) == ("opt0", "Because it is safest.")
    for bad in (None, "opt0", ("opt0",), ("opt0", None), (0, "x")):
        with pytest.raises(safety.GarbageError):
            safety.choice(bad)


def test_guarded_never_runs_one_caller_on_two_threads(faulty):
    agent = faulty(SkeletonAgent(), "hang", every=1, limit=1)
    calls = safety.Guarded(kill_after_s=60)              # long grace: it must come back on its own
    first = calls.call("a", "agent.propose", agent.propose, None, [], timeout_s=0.05)
    assert first.kind == safety.TIMEOUT and not first.benched
    for _ in range(3):                                   # still hanging: not called again
        assert calls.call("a", "agent.propose", agent.propose, None, [], timeout_s=0.05).kind == safety.BUSY
    assert agent.calls == 1
    agent.release()
    first.thread.join(1.0)
    back = calls.call("a", "agent.propose", lambda s, e: ([], []), None, [], timeout_s=1)
    assert back.ok and back.back and not back.killed     # finished on its own: used again
    assert agent.max_in_flight == 1


def test_guarded_kills_a_call_still_running_after_the_grace_and_benches_it_for_good():
    stop = threading.Event()
    now = [0.0]

    def spin():
        while not stop.is_set():
            pass
    calls = safety.Guarded(kill_after_s=5, clock=lambda: now[0])
    try:
        first = calls.call("k", "spin.propose", spin, timeout_s=0.05)
        assert first.kind == safety.TIMEOUT and not first.benched
        now[0] = 4.9                                       # inside the grace: just busy
        out = calls.call("k", "spin.propose", spin, timeout_s=0.05)
        assert out.kind == safety.BUSY and not out.killed and not out.benched
        now[0] = 5.0                                       # grace over, still spinning: killed
        out = calls.call("k", "spin.propose", spin, timeout_s=0.05)
        assert out.kind == safety.BUSY and out.killed and out.benched
        first.thread.join(2.0)
        assert not first.thread.is_alive()                 # the spin was stopped
        later = calls.call("k", "spin.propose", lambda: 1, timeout_s=1)
        assert later.kind == safety.BUSY and not later.back   # a killed caller never comes back
        assert calls.benched_labels() == ["spin.propose"] and calls.killed == {"k"}
    finally:
        stop.set()


def test_guarded_shutdown_stops_every_thread_still_running():
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            pass
    calls = safety.Guarded(kill_after_s=60)
    try:
        out = calls.call("k", "spin.propose", spin, timeout_s=0.05)
        assert calls.shutdown() == ["spin.propose"]
        out.thread.join(2.0)
        assert not out.thread.is_alive() and calls.benched_labels() == ["spin.propose"]
        assert calls.shutdown() == []                      # nothing left
    finally:
        stop.set()


def test_guarded_benches_a_caller_after_3_timeouts():
    calls = safety.Guarded(kill_after_s=60)
    for i in range(safety.MAX_TIMEOUTS):
        out = calls.call("k", "slow.propose", time.sleep, 0.2, timeout_s=0.02)
        assert out.kind == safety.TIMEOUT and out.benched == (i == safety.MAX_TIMEOUTS - 1)
        out.thread.join(1.0)                             # its call returns ...
    assert calls.call("k", "slow.propose", lambda: 1, timeout_s=1).kind == safety.BUSY   # ... benched anyway
    assert calls.benched_labels() == ["slow.propose"]
    assert calls.call("other", "fine", lambda: 1, timeout_s=1).ok     # others are unaffected


def test_run_budget_phases():
    now = [0.0]
    b = safety.RunBudget(100, clock=lambda: now[0])
    b.start()
    assert b.phase() == safety.FULL
    now[0] = 69.9
    assert b.phase() == safety.FULL
    now[0] = 70.0
    assert b.phase() == safety.NO_LLM
    now[0] = 100.0
    assert b.phase() == safety.RULES_ONLY
    assert b.report() == {"budget_s": 100.0, "used_s": 100.0, "llm_off_at_s": 70.0, "rules_only_at_s": 100.0}


@pytest.mark.parametrize("env,expected", [("120", 120.0), ("abc", 300.0), ("-5", 300.0), ("nan", 300.0),
                                          (None, 300.0)])
def test_run_budget_reads_run_budget_s(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv("RUN_BUDGET_S", raising=False)
    else:
        monkeypatch.setenv("RUN_BUDGET_S", env)
    assert safety.RunBudget().budget_s == expected


@pytest.mark.parametrize("env,expected", [("5", 7.0), ("oops", 10.0), ("0", 10.0), (None, 10.0)])
def test_call_timeout_is_one_llm_ask_plus_a_margin(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv("LLM_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("LLM_TIMEOUT_S", env)
    assert safety.call_timeout_s() == expected


def test_llm_off_and_restore(monkeypatch):
    from llm import client
    test_client = client.LLMClient([], enabled=True)
    monkeypatch.setattr(client, "_default", test_client)
    assert safety.llm_off() is True and test_client.enabled is False
    safety.llm_restore(True)
    assert test_client.enabled is True
    safety.llm_restore(None)                             # nothing to restore: no-op
    assert test_client.enabled is True


def test_clock_steps():
    assert safety.CLOCK_ATTEMPTS == 6
    assert safety.clock_steps(15) == [15, 5, 1, 1, 1, 1]
    assert safety.clock_steps(5, attempts=3) == [5, 1, 1]
    assert safety.clock_steps(0, attempts=2) == [1, 1]


class Clock:
    """advance()/read_now() for advance_clock, with scripted failures."""

    def __init__(self, replies):
        from datetime import datetime, timedelta
        self.t = datetime(2026, 9, 1, 6)
        self.start = self.t
        self.replies = list(replies)          # per attempt: "ok" | "stuck" | "error" | "lost" | "raise"
        self.asked = []
        self.td = timedelta

    def advance(self, m):
        self.asked.append(m)
        r = self.replies.pop(0) if self.replies else "ok"
        if r == "raise":
            raise RuntimeError("toolkit bug")
        if r == "error":
            return {"error": "clock unavailable"}
        if r in ("ok", "lost"):
            self.t += self.td(minutes=m)
        if r == "lost":
            return {"error": "reply lost"}
        return {"now": self.t.isoformat(), "finished": False}

    def read_now(self):
        return self.t


def test_advance_clock_moves_on_the_first_try():
    c = Clock(["ok"])
    res = safety.advance_clock(c.advance, c.read_now, c.start, 15)
    assert res.ok and res.problems == [] and c.asked == [15] and res.lost_ack is None


def test_advance_clock_retries_with_smaller_steps():
    c = Clock(["stuck", "error", "ok"])
    res = safety.advance_clock(c.advance, c.read_now, c.start, 15)
    assert res.ok and c.asked == [15, 5, 1] and len(res.problems) == 2
    assert "did not move" in res.problems[0] and "clock unavailable" in res.problems[1]


def test_advance_clock_rides_out_a_burst_of_transient_errors():
    c = Clock(["error"] * 5 + ["ok"])                 # 3 in a row happen by chance at 5 % errors
    res = safety.advance_clock(c.advance, c.read_now, c.start, 15)
    assert res.ok and c.asked == [15, 5, 1, 1, 1, 1] and len(res.problems) == 5


def test_advance_clock_believes_the_clock_when_the_reply_is_lost():
    c = Clock(["lost"])
    res = safety.advance_clock(c.advance, c.read_now, c.start, 15)
    assert res.ok and c.asked == [15] and res.lost_ack == "reply lost"     # not re-sent


def test_advance_clock_gives_up_after_6_tries():
    c = Clock(["stuck", "raise", "error", "stuck", "stuck", "stuck", "ok"])
    res = safety.advance_clock(c.advance, c.read_now, c.start, 15)
    assert not res.ok and len(res.problems) == 6 and c.asked == [15, 5, 1, 1, 1, 1]
    assert "toolkit bug" in res.problems[1]


# =========================================================================
# Part 2: the loop under faults, on a tiny terminal (no workbook needed)
# =========================================================================

def _tiny_loop(tmp_path, agents=None, api=None, **kw):
    return Loop(api or TinyAPI(hours=6), tmp_path, agents=agents, **kw)


def test_hanging_agent_times_out_and_the_run_finishes(tmp_path, faulty):
    agent = faulty(SkeletonAgent(), "hang", every=3)
    t0 = time.perf_counter()
    s = _tiny_loop(tmp_path, [agent], call_timeout_s=0.1, kill_after_s=60).run()
    assert time.perf_counter() - t0 < 10
    assert s["loop"]["stopped_early"] is None and s["loop"]["ticks"] == 24     # 6 h of 15-min ticks
    assert s["loop"]["timeouts"] == 1 and s["loop"]["busy_ticks"] == 25 - 3     # 25 steps; busy after #3
    assert agent.calls == 3 and agent.max_in_flight == 1
    (line,) = [d for d in _decisions(tmp_path) if d["choice"] == "timeout"]
    assert "rules-only" in line["reason"] and "skipped until that call returns" in line["reason"]
    (end,) = [d for d in _decisions(tmp_path) if d["choice"] == "killed"]       # still hanging at the end
    assert "end of the run" in end["reason"] and s["loop"]["killed"] == 1


def test_spinning_agent_is_killed_once_and_benched_never_reused(tmp_path, faulty):
    # Leader's review of PR #13: stopping a thread from outside skips the agent's own cleanup,
    # so its state may be half-updated. A killed agent must never be "recovered".
    agent = faulty(SkeletonAgent(), "spin", every=2)
    s = _tiny_loop(tmp_path, [agent], call_timeout_s=0.1, kill_after_s=0).run()
    assert s["loop"]["timeouts"] == 1 and s["loop"]["killed"] == 1
    assert s["loop"]["benched"] == ["skeleton.propose"]
    assert agent.calls == 2 and agent.max_in_flight == 1     # never called again after the timeout
    assert _no_live_workers()                                 # no spinning thread left behind
    lines = _decisions(tmp_path)
    assert not any(d["choice"] == "recovered" for d in lines)
    (kill,) = [d for d in lines if d["choice"] == "killed"]
    assert "benched for the rest of the run" in kill["reason"] and kill["tick"] == 2
    assert s["loop"]["stopped_early"] is None and s["loop"]["ticks"] == 24     # the run still finishes


def test_garbage_answers_are_logged_and_never_executed(tmp_path, faulty):
    agent = faulty(SkeletonAgent(), "garbage", every=1)
    s = _tiny_loop(tmp_path, [agent]).run()
    assert s["loop"]["garbage"] == agent.faults == agent.calls > 0
    assert s["loop"]["exceptions"] == 0 and s["loop"]["actions_sent"] == 0 and s["loop"]["actions_refused"] == 0
    assert sum(d["choice"] == "garbage" for d in _decisions(tmp_path)) == agent.faults


def test_rules_agent_failing_too_means_no_actions_not_a_crash(tmp_path, faulty):
    loop = _tiny_loop(tmp_path)
    loop.rules = faulty(SkeletonAgent(), "raise", every=1)
    loop.agents = [loop.rules]
    s = loop.run()
    assert s["loop"]["stopped_early"] is None and s["loop"]["ticks"] == 24
    assert s["loop"]["exceptions"] == loop.rules.calls == 25          # one per tick, each logged once
    assert sum(d["choice"] == "exception" for d in _decisions(tmp_path)) == 25


@pytest.mark.parametrize("mode", ["silent", "error"])
def test_stuck_clock_stops_the_run_cleanly(tmp_path, mode):
    api = FaultyAPI(TinyAPI(hours=6), stuck_clock_after_h=1, stuck_mode=mode)
    s = _tiny_loop(tmp_path, api=api).run()
    assert s["loop"]["stopped_early"] == "clock_stuck"
    assert s["loop"]["ticks"] == 4 and s["loop"]["clock_retries"] == 6     # 1 h of 15-min ticks
    lines = _decisions(tmp_path)
    assert sum(d["choice"] == "clock" for d in lines) == 6
    assert any(d["choice"] == "CRITICAL" and "clock is stuck" in d["reason"] for d in lines)
    assert _summary(tmp_path)["loop"]["stopped_early"] == "clock_stuck"


class FirstReadsFail(TinyAPI):
    def __init__(self, failures):
        super().__init__(hours=1)
        self.failures = failures

    def call(self, name, arguments=None):
        if name == "get_current_time" and self.failures > 0:
            self.failures -= 1
            return {"error": "simulator still loading"}
        return super().call(name, arguments)


def test_a_failed_first_read_is_retried_not_the_end_of_the_run(tmp_path):
    s = _tiny_loop(tmp_path, api=FirstReadsFail(2)).run()
    assert s["loop"]["stopped_early"] is None and s["loop"]["ticks"] == 4 and s["loop"]["exceptions"] == 2
    assert sum("first snapshot" in d["reason"] for d in _decisions(tmp_path)) == 2


class ClockReadsDie(TinyAPI):
    """The clock can't be read after `after_h` hours, for the rest of the run; advance_time works.
    core/state.py then keeps the OLD time in every snapshot (it falls back to the last one)."""

    def __init__(self, hours, after_h):
        super().__init__(hours=hours)
        self.dies_at = self.now + timedelta(hours=after_h)

    def call(self, name, arguments=None):
        if name == "get_current_time" and self.now >= self.dies_at:
            return {"error": "clock service down"}
        return super().call(name, arguments)


def test_clock_reads_dying_for_good_still_stop_at_the_window_end(tmp_path):
    # Leader's review of PR #13: without tracking advance_time's own answer, the loop compared
    # the stale snapshot time with the window end and ran to max_ticks (5,000 vs ~1,440 there).
    api = ClockReadsDie(hours=6, after_h=1)
    loop = _tiny_loop(tmp_path, api=api, max_ticks=5000)
    s = loop.run()
    assert api.now >= api.end
    assert loop.tick == 24 and s["loop"]["stopped_early"] is None             # 6 h of 15-min ticks
    assert loop.clock_now == api.end


def test_a_simulator_that_never_answers_stops_after_5_tries(tmp_path):
    s = _tiny_loop(tmp_path, api=FirstReadsFail(10**6)).run()
    assert s["loop"]["stopped_early"] == "no_state" and s["loop"]["exceptions"] == 5
    assert (tmp_path / "summary.json").exists()


def test_summary_is_written_even_when_run_summary_fails(tmp_path):
    class Broken(TinyAPI):
        def run_summary(self):
            raise KeyError("baseline")
    s = _tiny_loop(tmp_path, api=Broken(hours=1)).run()
    on_disk = _summary(tmp_path)
    assert "run_summary failed" in on_disk["summary_error"] and on_disk["violations"] == []
    assert on_disk["loop"]["ticks"] == s["loop"]["ticks"] == 4


def test_run_budget_switches_the_llm_off_then_goes_rules_only(tmp_path, monkeypatch):
    from llm import client
    test_client = client.LLMClient([], enabled=True)
    monkeypatch.setattr(client, "_default", test_client)
    seen = []

    class Probe:
        name = "probe"
        escalations = []

        def propose(self, state, events):
            seen.append(test_client.enabled)
            return [], []
    loop = _tiny_loop(tmp_path, [Probe()])
    tick = [0.0]
    loop.budget = safety.RunBudget(10, clock=lambda: tick[0])

    def budget():                     # the run uses 1 s of its 10 s budget per tick
        tick[0] = float(loop.tick)
        Loop._budget(loop)
    loop._budget = budget
    s = loop.run()
    assert seen[:7] == [True] * 7 and seen[7:] == [False] * 3          # LLM off from 70 %
    assert len(seen) == 10                                              # rules-only from 100 %
    assert s["loop"]["rules_only_ticks"] == 25 - 10
    assert s["loop"]["budget"]["llm_off_at_s"] == 7.0 and s["loop"]["budget"]["rules_only_at_s"] == 10.0
    assert sum(d["choice"] == "budget" for d in _decisions(tmp_path)) == 2
    assert test_client.enabled is True                                  # restored for the next scenario


def test_run_one_writes_a_summary_when_the_scenario_cannot_start(tmp_path):
    import run
    with pytest.raises(Exception):
        run.run_one("practice_01", tmp_path / "missing.xlsx", None, tmp_path)
    stub = _summary(tmp_path / "practice_01")
    assert stub["crashed"] is True and stub["violations"] == [] and stub["error"]


# =========================================================================
# Part 3: the real practice scenarios under faults (needs the workbook)
# =========================================================================

def _real(scenario, tmp_path, **faults):
    from vendor import TerminalAPI
    run_dir = tmp_path / scenario
    real = TerminalAPI(scenario, data_xlsx=str(WORKBOOK), log_path=str(run_dir / "actions.jsonl"))
    return real, (FaultyAPI(real, **faults) if faults else real), run_dir


def _check_clean(s, run_dir, real):
    """The run reached the end of the scenario with nothing the organizers could score against us."""
    assert s["loop"]["stopped_early"] is None and real.now >= real.end
    assert s["violations"] == []
    counts = log.simulator_log_counts(run_dir)
    assert counts["VIOLATION"] == 0 and counts["tool_error"] == 0, counts
    assert log.count_missing_reasons(run_dir) == 0


def _unlogged(api, run_dir):
    text = (run_dir / "decisions.jsonl").read_text() + (run_dir / "summary.json").read_text()
    return [f for f in api.injected if f["kind"] != "stuck_clock" and f"injected {f['tag']}:" not in text]


@needs_workbook
@pytest.mark.parametrize("i", range(5))
def test_agent_exceptions_and_5pct_tool_errors_never_stop_a_run(tmp_path, faulty, i, multi_runs):
    """Done when (TASKS.md): exceptions in each agent + tool errors in 5 % of calls.
    Scenario i also makes agent i % 4 raise on every 5th call, so all four agents are covered."""
    from run import make_agents
    scenario = f"practice_0{i + 1}"
    real, api, run_dir = _real(scenario, tmp_path, seed=i, tool_error_rate=0.05, grace_calls=10)
    agents = make_agents("multi")
    agents[i % 4] = bad = faulty(agents[i % 4], "raise", every=5)
    s = Loop(api, run_dir, agents=agents).run()
    _check_clean(s, run_dir, real)
    assert len(api.injected) > 100 and bad.faults > 10
    assert _unlogged(api, run_dir) == []                        # every injected fault is in the log
    raised = [d for d in _decisions(run_dir) if d["choice"] == "exception" and d["agent"] == bad.name]
    assert len(raised) == bad.faults == s["loop"]["exceptions"]
    clean, _ = multi_runs[scenario]
    assert s["vessels_departed"] >= clean["vessels_departed"] - 1


@needs_workbook
def test_lost_replies_never_make_us_redo_work(tmp_path):
    from run import make_agents
    real, api, run_dir = _real("practice_02", tmp_path, seed=7, lost_ack_rate=0.1)
    s = Loop(api, run_dir, agents=make_agents("multi")).run()
    _check_clean(s, run_dir, real)
    lost = [f for f in api.injected if f["kind"] == "lost_ack"]
    assert len(lost) > 10 and s["loop"]["lost_acks"] > 0
    assert _unlogged(api, run_dir) == []
    placed = [c for line in (run_dir / "actions.jsonl").read_text().splitlines()
              for r in [json.loads(line)] if r["action"] == "tool_call" and r.get("tool") == "place_container"
              for c in r["arguments"]["container_ids"]]
    assert len(placed) == len(set(placed))                       # no box placed twice (F2 drift)
    in_block = {}
    for c, b in real.slot.items():
        if c in real.placed_at:
            in_block[b] = in_block.get(b, 0) + 1
    assert all(real.yard_fill[b] == in_block.get(b, 0) for b in real.yard_fill)   # yard counts agree


@needs_workbook
def test_a_hanging_agent_costs_one_timeout_not_the_run(tmp_path, faulty):
    from run import make_agents
    real, api, run_dir = _real("practice_05", tmp_path)
    agents = make_agents("multi")
    agents[0] = faulty(agents[0], "hang", every=40)
    t0 = time.perf_counter()
    s = Loop(api, run_dir, agents=agents, call_timeout_s=0.5).run()
    assert time.perf_counter() - t0 < 60
    _check_clean(s, run_dir, real)
    assert s["loop"]["timeouts"] == 1 and s["vessels_departed"] > 0


@needs_workbook
def test_garbage_from_every_agent_is_caught_before_the_guard(tmp_path, faulty):
    from run import make_agents
    real, api, run_dir = _real("practice_02", tmp_path)
    agents = [faulty(a, "garbage", every=3) for a in make_agents("multi")]
    s = Loop(api, run_dir, agents=agents).run()
    _check_clean(s, run_dir, real)
    assert s["loop"]["garbage"] == sum(a.faults for a in agents) > 100
    assert s["loop"]["actions_refused"] == 0                     # nothing malformed reached the guard


@needs_workbook
def test_stuck_clock_on_a_real_scenario(tmp_path):
    real, api, run_dir = _real("practice_02", tmp_path, stuck_clock_after_h=24)
    s = Loop(api, run_dir).run()
    assert s["loop"]["stopped_early"] == "clock_stuck" and s["violations"] == []
    assert s["loop"]["ticks"] < 400                              # stopped, not spinning to max_ticks
    assert log.simulator_log_counts(run_dir)["tool_error"] == 0


@needs_workbook
def test_everything_at_once(tmp_path, faulty):
    """The chaos run: 5 % tool errors, 5 % lost replies, and every agent misbehaving its own way."""
    from run import make_agents
    real, api, run_dir = _real("practice_03", tmp_path, seed=3, tool_error_rate=0.05, lost_ack_rate=0.05,
                               grace_calls=10)
    agents = make_agents("multi")
    # The hanging agent goes last: a busy agent makes the rest of each tick rules-only, so
    # the agents after it would never be called again.
    for k, (mode, every) in enumerate([("raise", 23), ("garbage", 7), ("spin", 17), ("hang", 13)]):
        agents[k] = faulty(agents[k], mode, every=every)
    s = Loop(api, run_dir, agents=agents, call_timeout_s=0.5).run()
    _check_clean(s, run_dir, real)
    assert _unlogged(api, run_dir) == []
    assert s["loop"]["exceptions"] and s["loop"]["timeouts"] and s["loop"]["garbage"]
    assert s["vessels_departed"] > 0
