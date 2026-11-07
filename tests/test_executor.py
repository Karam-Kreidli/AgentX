"""CORE-4 tests: executor and main loop.

- A refused action is logged and not sent (the task's named test).
- Every state-changing call has a matching decisions.jsonl entry, on all 5 scenarios, with zero violations.
- Errors from the simulator are data; agent exceptions, the watchdog and invariant breaches
  fall back to rules-only without stopping the run.
"""
import json
import sys
import time
import types
from pathlib import Path

import pytest

from core.executor import Executor
from core.loop import Loop
from core.state import StateBuilder
from core.types import Action

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
MUTATING = {"assign_berth", "plan_crane_sequence", "dispatch_truck", "place_container", "release_to_consignee"}


class FakeAPI:
    """Minimal recorded terminal: one arrived ship, two cranes, a yard. Records every call."""

    def __init__(self):
        self.calls = []
        self.schedule = [{"call_id": "C1", "vessel": "MV Test", "loa_m": 200, "draft_m": 11.0,
                          "scheduled_eta": "2026-09-01T01:00:00", "actual_arrival": "2026-09-01T01:00:00",
                          "status": "arrived", "berth": None, "cranes": [], "trucks": 0,
                          "discharge_remaining": 2, "load_remaining": 0, "departed_at": None}]
        self.manifest = [{"container_id": "X1", "direction": "import", "size_ft": 40, "cargo_type": "IMDG",
                          "imdg_class": 3.0, "gross_weight_t": 20.0, "yard_block": None}]
        self.lie = False

    def call(self, tool, args=None):
        self.calls.append((tool, dict(args or {})))
        if tool == "get_current_time":
            return {"now": "2026-09-01T06:00:00", "scenario_end": "2026-09-06T00:00:00", "finished": False}
        if tool == "get_vessel_schedule":
            return self.schedule
        if tool == "get_equipment_status":
            return [{"equipment_id": "QC1", "type": "quay crane", "status": "available", "bay": 1},
                    {"equipment_id": "QC2", "type": "quay crane", "status": "available", "bay": 14}]
        if tool == "get_yard_status":
            return [{"block_id": "A", "block_type": "standard", "used": 0, "slots": 220},
                    {"block_id": "H", "block_type": "IMDG", "used": 0, "slots": 120}]
        if tool == "get_customs_status":
            return []
        if tool == "get_manifest":
            return self.manifest
        if tool == "assign_berth":
            if not self.lie:
                self.schedule[0] = dict(self.schedule[0], status="berthed", berth=args["berth_id"])
            return {"ok": True}
        if tool == "place_container":
            return {"error": "block H is under maintenance"}     # simulator error, returned as data
        return {"error": f"unexpected tool {tool}"}

    def sent(self):
        return [c for c in self.calls if c[0] in MUTATING]


def _setup(tmp_path):
    api = FakeAPI()
    ex = Executor(api, tmp_path)
    state = StateBuilder(api).build()
    return api, ex, state


def _log(tmp_path):
    return [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------- executor

def test_refused_action_is_logged_and_not_sent(tmp_path):
    api, ex, state = _setup(tmp_path)
    bad = Action("place_container", {"container_ids": ["X1"], "block_id": "A"}, "yard", "A is closest")
    res = ex.execute(bad, state)
    assert "refused" in res and "IMDG" in res["refused"]
    assert api.sent() == []                                       # never reached the simulator
    (entry,) = _log(tmp_path)
    assert entry["guard_ok"] is False and entry["tool_result"] is None
    assert entry["choice"] == "place_container" and "IMDG" in entry["guard_reason"]
    assert ex.refused == 1 and ex.sent == 0


def test_action_without_reason_is_refused(tmp_path):
    api, ex, state = _setup(tmp_path)
    res = ex.execute(Action("assign_berth", {"call_id": "C1", "berth_id": "B3"}, "berth", "  "), state)
    assert "refused" in res and "reason" in res["refused"]
    assert api.sent() == []


def test_allowed_action_is_sent_logged_and_state_reread(tmp_path):
    api, ex, state = _setup(tmp_path)
    res = ex.execute(Action("assign_berth", {"call_id": "C1", "berth_id": "B3"}, "berth", "arrived, B3 free"), state)
    assert res == {"ok": True}
    assert api.sent() == [("assign_berth", {"call_id": "C1", "berth_id": "B3"})]
    (entry,) = _log(tmp_path)
    assert entry["guard_ok"] and entry["tool_result"] == {"ok": True} and entry["reason"] == "arrived, B3 free"
    assert ex.state.ship("C1").berth == "B3"                      # read-after-write
    assert ex.mismatches == 0


def test_simulator_error_is_data_not_exception(tmp_path):
    api, ex, state = _setup(tmp_path)
    res = ex.execute(Action("place_container", {"container_ids": ["X1"], "block_id": "H"}, "yard", "IMDG to H"), state)
    assert res == {"error": "block H is under maintenance"}
    assert _log(tmp_path)[0]["tool_result"] == {"error": "block H is under maintenance"}


def test_mismatch_after_write_is_logged_and_simulator_believed(tmp_path):
    api, ex, state = _setup(tmp_path)
    api.lie = True                                                # simulator says ok but nothing changes
    ex.execute(Action("assign_berth", {"call_id": "C1", "berth_id": "B3"}, "berth", "arrived, B3 free"), state)
    log = _log(tmp_path)
    assert log[-1]["choice"] == "mismatch" and ex.mismatches == 1
    assert ex.state.ship("C1").berth is None                      # we believe the simulator


# ---------------------------------------------------------------- loop on the real simulator

def _api(scenario, tmp_path):
    from vendor import TerminalAPI
    return TerminalAPI(scenario, data_xlsx=str(WORKBOOK), log_path=str(tmp_path / scenario / "actions.jsonl"))


def _pairs(run_dir):
    actions = [json.loads(l) for l in (run_dir / "actions.jsonl").read_text().splitlines()]
    sent = [a for a in actions if a["action"] in ("tool_call", "tool_error") and a.get("tool") in MUTATING]
    decisions = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines()]
    logged = [d for d in decisions if d["choice"] in MUTATING and d["guard_ok"]]
    return sent, logged, decisions


@needs_workbook
@pytest.mark.parametrize("scenario", [f"practice_0{i}" for i in range(1, 6)])
def test_all_scenarios_zero_violations_and_every_call_logged(tmp_path, scenario):
    api = _api(scenario, tmp_path)
    s = Loop(api, tmp_path / scenario).run()
    assert s["violations"] == []
    assert s["loop"]["exceptions"] == 0
    sent, logged, _ = _pairs(tmp_path / scenario)
    assert len(sent) == len(logged) > 0
    for a, d in zip(sent, logged):
        assert a["tool"] == d["choice"] and a["arguments"] == d["saw"]["args"] and d["reason"]


class Boom:
    name = "boom"
    escalations = []

    def propose(self, state, events):
        raise RuntimeError("agent bug")


class Slow:
    name = "slow"

    def __init__(self):
        from core.skeleton import SkeletonAgent
        self.inner = SkeletonAgent()

    def propose(self, state, events):
        time.sleep(0.02)
        return self.inner.propose(state, events)


@needs_workbook
def test_agent_exception_falls_back_to_rules_only(tmp_path):
    api = _api("practice_05", tmp_path)
    s = Loop(api, tmp_path / "practice_05", agents=[Boom()], max_ticks=200).run()
    assert s["violations"] == [] and s["loop"]["exceptions"] > 0
    assert s["loop"]["actions_sent"] > 0                          # rules-only still berthed and worked ships
    _, _, decisions = _pairs(tmp_path / "practice_05")
    assert any(d["choice"] == "exception" and "rules-only" in d["reason"] for d in decisions)


@needs_workbook
def test_watchdog_finishes_slow_ticks_on_rules_only(tmp_path):
    api = _api("practice_05", tmp_path)
    s = Loop(api, tmp_path / "practice_05", agents=[Slow()], tick_budget_s=0.0, max_ticks=100).run()
    assert s["violations"] == [] and s["loop"]["watchdog"] > 0
    _, _, decisions = _pairs(tmp_path / "practice_05")
    assert any(d["choice"] == "watchdog" for d in decisions)


@needs_workbook
def test_invariant_breach_logs_critical_and_next_tick_is_rules_only(tmp_path, monkeypatch):
    inv = types.ModuleType("core.invariants")
    calls = {"n": 0}

    def check_state(state):
        calls["n"] += 1
        return ["planted breach"] if calls["n"] == 3 else []

    inv.check_state = check_state
    monkeypatch.setitem(sys.modules, "core.invariants", inv)
    api = _api("practice_05", tmp_path)
    s = Loop(api, tmp_path / "practice_05", max_ticks=10).run()
    assert s["loop"]["breaches"] == 1 and s["loop"]["rules_only_ticks"] == 1
    _, _, decisions = _pairs(tmp_path / "practice_05")
    assert any(d["choice"] == "CRITICAL" and "planted breach" in d["reason"] for d in decisions)


def test_tick_is_5_minutes_when_busy_else_15():
    from datetime import datetime
    from core.skeleton import BERTHS
    from core.types import Crane, Ship, WorldState
    t = datetime(2026, 9, 1, 6)

    def st(status, crane_status="available", eta=datetime(2026, 9, 2)):
        ship = Ship("C1", "V", 200.0, 11.0, eta, None if status == "expected" else t, status, None,
                    (), 0, 1, 1, None)
        return WorldState(t, datetime(2026, 9, 6), (ship,), (Crane("QC1", crane_status, None, 1),), (), (), (),
                          {}, BERTHS)
    assert Loop._tick_minutes(st("expected")) == 15                                  # nothing due
    assert Loop._tick_minutes(st("expected", eta=datetime(2026, 9, 1, 6, 10))) == 5  # ship due soon
    assert Loop._tick_minutes(st("arrived")) == 5                                    # waiting for a berth
    assert Loop._tick_minutes(st("working", crane_status="working")) == 15           # all cranes busy


# ---------------------------------------------------------------- snapshot failures (never crash, never run away)

def _failing_build_state(monkeypatch, fail):
    """Make core.loop.build_state raise whenever fail(call_number) is true."""
    import core.loop as loop_mod
    real = loop_mod.build_state
    n = {"calls": 0}

    def flaky(api):
        n["calls"] += 1
        if fail(n["calls"]):
            raise RuntimeError("clock read failed")
        return real(api)

    monkeypatch.setattr(loop_mod, "build_state", flaky)
    return n


@needs_workbook
def test_first_snapshot_failing_does_not_end_the_run(tmp_path, monkeypatch):
    # The first two snapshots fail. Before the fix the loop stopped at once and summary() crashed.
    _failing_build_state(monkeypatch, lambda k: k <= 2)
    api = _api("practice_05", tmp_path)
    s = Loop(api, tmp_path / "practice_05").run()
    assert s["loop"]["exceptions"] == 2
    assert s["violations"] == [] and s["vessels_departed"] == 4        # same as a clean run


@needs_workbook
def test_snapshots_failing_for_good_still_stop_at_the_window_end(tmp_path, monkeypatch):
    # Every snapshot after the 10th fails. The loop must stop near the end of the 5-day window
    # (its own clock estimate), not keep advancing on a stale snapshot until max_ticks.
    _failing_build_state(monkeypatch, lambda k: k > 10)
    api = _api("practice_05", tmp_path)
    loop = Loop(api, tmp_path / "practice_05", max_ticks=5000)
    s = loop.run()
    assert loop.tick < 5 * 24 * 12 + 5                                  # at most a 5-min tick per step
    assert api.now >= api.end and s["loop"]["stopped_early"] is None    # it reached the window end
    assert "summary_error" in s and s["violations"] == []


@needs_workbook
def test_no_snapshot_ever_stops_cleanly_after_5_tries(tmp_path, monkeypatch):
    # SAFE-4: a simulator that can't be read at all is retried FIRST_SNAPSHOT_TRIES times,
    # then the run stops cleanly (it used to keep advancing for one full window).
    from core.loop import FIRST_SNAPSHOT_TRIES
    _failing_build_state(monkeypatch, lambda k: True)
    api = _api("practice_05", tmp_path)
    loop = Loop(api, tmp_path / "practice_05", max_ticks=100_000)
    s = loop.run()                                                       # must not raise
    assert loop.tick == 0 and s["loop"]["exceptions"] == FIRST_SNAPSHOT_TRIES == 5
    assert s["loop"]["stopped_early"] == "no_state"
    assert s["honest_turnaround_h"] is None and "summary_error" in s
