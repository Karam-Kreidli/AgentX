"""CORE-7/8: Loop(coordinate=False) never calls the coordinator (FCFS reference, single agent, ablation)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
pytestmark = pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")


def _run(tmp_path, monkeypatch, agent, coordinate):
    import coordinator
    from run import make_agents
    from core.loop import Loop
    from vendor import TerminalAPI
    calls = []
    real = coordinator.decide
    monkeypatch.setattr(coordinator, "decide", lambda *a, **k: calls.append(1) or real(*a, **k))
    api = TerminalAPI("practice_01", data_xlsx=str(WORKBOOK), log_path=str(tmp_path / "actions.jsonl"))
    summary = Loop(api, tmp_path, agents=make_agents(agent), coordinate=coordinate).run()
    return summary, len(calls)


def test_coordinate_false_never_calls_the_coordinator(tmp_path, monkeypatch):
    summary, n = _run(tmp_path, monkeypatch, "multi", coordinate=False)
    assert n == 0
    assert not summary["violations"]


def test_coordinate_true_calls_it(tmp_path, monkeypatch):
    _, n = _run(tmp_path, monkeypatch, "multi", coordinate=True)
    assert n > 0
