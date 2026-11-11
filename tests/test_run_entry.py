"""CORE-9: how the organizers start us. find_data (D16), --all-hidden (D11), the toolkit-default log
copy (D10), run_agent (D12)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
needs_wb = pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")


# ------------------------------------------------------------------ D16 find_data
def test_find_data_explicit_wins(tmp_path):
    import run
    assert run.find_data(tmp_path / "x.xlsx") == tmp_path / "x.xlsx"


def test_find_data_next_to_key(tmp_path, monkeypatch):
    import run
    monkeypatch.setattr(run, "ROOT", tmp_path / "repo")          # no data/ or toolskit/ here
    (tmp_path / "keys").mkdir()
    (tmp_path / "keys" / run.DATA_NAME).write_bytes(b"")
    found = run.find_data(None, tmp_path / "keys" / "KEY.xlsx", cwd=tmp_path / "empty")
    assert found == (tmp_path / "keys" / run.DATA_NAME).resolve()


def test_find_data_order_and_nothing(tmp_path, monkeypatch):
    import run
    repo = tmp_path / "repo"
    monkeypatch.setattr(run, "ROOT", repo)
    assert run.find_data(None, None, cwd=tmp_path / "empty") is None
    (repo / "toolskit" / "data").mkdir(parents=True)
    (repo / "toolskit" / "data" / run.DATA_NAME).write_bytes(b"")
    (tmp_path / "cwd").mkdir()
    (tmp_path / "cwd" / run.DATA_NAME).write_bytes(b"")
    assert run.find_data(None, None, cwd=tmp_path / "cwd") == tmp_path / "cwd" / run.DATA_NAME   # cwd before toolskit
    (repo / "data").mkdir()
    (repo / "data" / run.DATA_NAME).write_bytes(b"")
    assert run.find_data(None, None, cwd=tmp_path / "cwd") == repo / "data" / run.DATA_NAME      # data/ first


def test_main_says_which_workbook_it_used(tmp_path, monkeypatch, capsys):
    import run
    wb = tmp_path / "d.xlsx"
    wb.write_bytes(b"")
    monkeypatch.setattr(run, "run_one", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    run.main(["--scenario", "practice_01", "--no-llm", "--data", str(wb), "--runs", str(tmp_path / "r")])
    assert f"data workbook: {wb}" in capsys.readouterr().out


def test_main_errors_when_no_workbook(tmp_path, monkeypatch):
    import run
    monkeypatch.setattr(run, "find_data", lambda *a, **k: None)
    with pytest.raises(SystemExit) as e:
        run.main(["--scenario", "practice_01", "--no-llm"])
    assert e.value.code == 2


# ------------------------------------------------------------------ D11 --all-hidden
def _fake_key(tmp_path, names=("hidden_02", "hidden_01", "hidden_02")):
    import pandas as pd
    key = tmp_path / "KEY.xlsx"
    with pd.ExcelWriter(key) as w:
        pd.DataFrame({"scenario": list(names), "call_id": list("abc"), "vessel_name": ["SECRET"] * 3}
                     ).to_excel(w, sheet_name="Hidden_Vessels", index=False)
        pd.DataFrame({"scenario": ["hidden_01"], "x": [1]}).to_excel(w, sheet_name="Hidden_Events", index=False)
    return key


def test_hidden_scenarios_sorted_unique(tmp_path):
    import run
    assert run.hidden_scenarios(_fake_key(tmp_path)) == ["hidden_01", "hidden_02"]


def test_all_hidden_reads_only_the_scenario_column(tmp_path, monkeypatch):
    import pandas as pd
    import run
    key = _fake_key(tmp_path)
    reads = []
    real = pd.read_excel
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: reads.append((k.get("sheet_name"), k.get("usecols"))) or real(*a, **k))
    assert run.hidden_scenarios(key) == ["hidden_01", "hidden_02"]
    assert reads == [("Hidden_Vessels", ["scenario"])]


def test_all_hidden_runs_every_name_and_passes_only_paths(tmp_path, monkeypatch):
    import run
    key = _fake_key(tmp_path)
    wb = tmp_path / "d.xlsx"
    wb.write_bytes(b"")
    seen = []

    def fake(s, data, k, runs, agent="multi"):
        seen.append((s, data, k))
        return {"violations": [], "vessels_departed": 0, "vessels_total": 0}
    monkeypatch.setattr(run, "run_one", fake)
    monkeypatch.setattr(run, "print_table", lambda rows: None)
    monkeypatch.chdir(tmp_path)
    assert run.main(["--all-hidden", "--key", str(key), "--data", str(wb), "--no-llm", "--runs", str(tmp_path / "r")]) == 0
    assert seen == [("hidden_01", wb, key), ("hidden_02", wb, key)]      # names + paths: nothing else from the key


def test_all_hidden_needs_key(tmp_path):
    import run
    with pytest.raises(SystemExit) as e:
        run.main(["--all-hidden", "--data", str(tmp_path / "x.xlsx")])
    assert e.value.code == 2


def test_agents_get_no_key_path():
    """run_agent takes the api only; the key path never leaves run_one's TerminalAPI call."""
    import inspect
    import run
    assert list(inspect.signature(run.run_agent).parameters) == ["api", "run_dir", "agent"]


# ------------------------------------------------------------------ D10 log copy
def test_copy_toolkit_log(tmp_path, monkeypatch):
    import run
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "runs" / "practice_01"
    d.mkdir(parents=True)
    (d / "actions.jsonl").write_text('{"a": 1}\n')
    dst = run.copy_toolkit_log(d, "practice_01")
    assert dst == Path("runs/P2/practice_01/actions.jsonl")
    assert dst.read_text() == '{"a": 1}\n'
    assert run.copy_toolkit_log(tmp_path / "nope", "x") is None      # no source: nothing, no crash


# ------------------------------------------------------------------ D12 run_agent, end to end
@needs_wb
def test_run_agent_on_own_api_and_main_leaves_both_logs(tmp_path, monkeypatch):
    import run
    from vendor import TerminalAPI
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    # (a) a caller that builds the api itself, with the toolkit's default log path
    api = TerminalAPI("practice_01", data_xlsx=str(WORKBOOK))
    summary = run.run_agent(api, tmp_path / "mine", "skeleton")
    assert not summary["violations"]
    assert (tmp_path / "mine" / "summary.json").is_file()
    assert (tmp_path / "runs" / "P2" / "practice_01" / "actions.jsonl").stat().st_size > 0
    # (b) the command line: our folder and the toolkit-default copy hold the same actions
    assert run.main(["--scenario", "practice_02", "--no-llm", "--agent", "skeleton", "--runs", str(runs)]) == 0
    ours = (runs / "practice_02" / "actions.jsonl").read_text()
    assert ours and (runs / "P2" / "practice_02" / "actions.jsonl").read_text() == ours
    assert json.loads((runs / "practice_02" / "summary.json").read_text())["scenario"] == "practice_02"


@needs_wb
def test_hidden_rehearsal_with_mock_key(tmp_path, monkeypatch):
    """The judges' path: --all-hidden with a key and no --data (found in data/)."""
    import run
    sys.path.insert(0, str(ROOT))
    from tools import perturb
    perturb.main(["--n", "1", "--malformed", "0", "--hidden-key", "--out", str(tmp_path / "v")])
    key = tmp_path / "v" / "MOCK_ANSWER_KEY.xlsx"
    assert key.is_file()
    names = run.hidden_scenarios(key)
    assert names == [f"hidden_0{i}" for i in range(1, 6)]
    monkeypatch.chdir(tmp_path)
    assert run.main(["--scenario", "hidden_01", "--key", str(key), "--no-llm", "--agent", "skeleton",
                     "--runs", str(tmp_path / "runs")]) == 0


# ------------------------------------------------------------------ clean-clone script and README
def test_clean_clone_script_is_valid_bash_and_needs_a_workbook():
    import subprocess
    script = ROOT / "tools" / "clean_clone_check.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert subprocess.run(["bash", str(script)], capture_output=True).returncode == 2
    assert subprocess.run(["bash", str(script), "/no/such/file.xlsx"], capture_output=True).returncode == 2


def test_readme_documents_every_entry():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for needle in ("--all-hidden", "--no-llm", "run_agent", "clean_clone_check.sh", "runs/P2/", "Freeze checklist"):
        assert needle in text, needle
