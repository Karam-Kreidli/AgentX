"""BC-4: config/params.yaml is valid, loads without a new dependency, never crashes a run on bad
content, and holds no scenario-specific parameter."""
import re
from pathlib import Path

from core import params

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "params.yaml"


def test_committed_file_is_valid_and_complete():
    vals = params.load(CONFIG)
    assert params.problems() == []
    assert set(vals) == set(params.SPEC)


def test_no_value_names_a_scenario_ship_berth_crane_or_box():
    text = CONFIG.read_text(encoding="utf-8")
    body = "\n".join(line.split("#", 1)[0] for line in text.split("\n"))
    assert not re.search(r"practice|hidden|P\d\d-\d|\bB[123]\b|\bQC\d|\bYT\d|[A-Z]{4}\d{6,}", body, re.I)


def test_agents_read_their_values_from_the_file():
    from agents import berth, crane, options
    vals = params.load(CONFIG)
    assert berth.HOLD_H == vals["berth.hold_h"] and berth.AGE_WEIGHT == vals["berth.age_weight"]
    assert crane.TICK_H == vals["crane.tick_h"] and crane.SPARE_TO == vals["crane.spare_to"]
    assert options.MIN_GAIN_H == vals["options.min_gain_h"]


def test_missing_file_gives_the_defaults(tmp_path):
    vals = params.load(tmp_path / "nope.yaml")
    assert vals == {k: d for k, (d, _) in params.SPEC.items()}
    assert any("cannot read" in p for p in params.problems())
    params.load(CONFIG)


def test_bad_entries_keep_their_default_and_are_reported(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("berth:\n  hold_h: -5\n  age_weight: banana\n  colour: red\ncrane:\n  spare_to: everywhere\n"
                 "  tick_h: 0.5   # fine\nnot a section line\n", encoding="utf-8")
    vals = params.load(f)
    assert vals["berth.hold_h"] == params.SPEC["berth.hold_h"][0]
    assert vals["berth.age_weight"] == params.SPEC["berth.age_weight"][0]
    assert vals["crane.spare_to"] == params.SPEC["crane.spare_to"][0]
    assert vals["crane.tick_h"] == 0.5
    probs = " | ".join(params.problems())
    for word in ("hold_h", "age_weight", "colour", "spare_to", "line 8"):
        assert word in probs, word
    params.load(CONFIG)


def test_env_var_points_at_another_file(tmp_path, monkeypatch):
    f = tmp_path / "p.yaml"
    f.write_text("berth:\n  hold_h: 3\n", encoding="utf-8")
    monkeypatch.setenv("AGENTX_PARAMS", str(f))
    assert params.load()["berth.hold_h"] == 3.0
    monkeypatch.delenv("AGENTX_PARAMS")
    params.load(CONFIG)


def test_age_weight_lets_a_long_wait_overtake_less_work(monkeypatch):
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from agents import berth
    now = datetime(2026, 9, 2, 12, 0)
    old = SimpleNamespace(call_id="X1", remaining_moves=1000, actual_arrival=now - timedelta(hours=20))
    new = SimpleNamespace(call_id="X2", remaining_moves=600, actual_arrival=now - timedelta(hours=1))
    state = SimpleNamespace(now=now, ships=[], berths=[])
    monkeypatch.setattr(berth, "_impossible", lambda st: set())
    monkeypatch.setattr(berth, "_operator", lambda st: None)
    monkeypatch.setattr(berth, "AGE_WEIGHT", 0.0)
    assert [s.call_id for s in berth.rank(state, [old, new])] == ["X2", "X1"]     # least work first
    monkeypatch.setattr(berth, "AGE_WEIGHT", 30.0)     # 19 h more wait x 30 = 570 moves > 400 moves more work
    assert [s.call_id for s in berth.rank(state, [old, new])] == ["X1", "X2"]


def test_a_bad_file_is_noted_once_in_the_run_log(tmp_path, monkeypatch):
    workbook = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
    if not workbook.exists():
        import pytest
        pytest.skip("workbook not in data/")
    from core.loop import Loop
    from vendor import TerminalAPI
    monkeypatch.setattr(params, "problems", lambda: ["berth.hold_h: '-5' outside 0.0..24.0; default 2.0 used"])
    api = TerminalAPI("practice_01", data_xlsx=str(workbook), log_path=str(tmp_path / "actions.jsonl"))
    Loop(api, tmp_path, max_ticks=3).run()
    text = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8")
    assert text.count("config/params.yaml: berth.hold_h") == 1
