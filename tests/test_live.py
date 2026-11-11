"""tools/live.py: the judges' scenario builder must give every strategy the same scenario, and the
workbook snapshot must hand out fresh, untouched copies."""
from pathlib import Path

import pytest

from tools import live

WB = Path(__file__).resolve().parents[1] / "data" / "P2_Container_Terminal_Data.xlsx"
pytestmark = pytest.mark.skipif(not WB.exists(), reason="workbook not in data/")


@pytest.fixture(autouse=True)
def _own_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "CACHE_DIR", tmp_path / "cache")


def _arrivals(api):
    return {c: v["actual_arrival"] for c, v in api.calls.items()}


def test_same_knobs_same_scenario():
    k = {"arrivals": "late", "seed": 3, "breakdowns": [{"crane": "QC2", "at_h": 6, "hours": 4}]}
    a, b = live.build_api("practice_02", WB, None, k), live.build_api("practice_02", WB, None, k)
    assert _arrivals(a) == _arrivals(b)
    assert a.down == b.down and "QC2" in a.down


def test_snapshot_copies_are_not_shared():
    plain = _arrivals(live.build_api("practice_02", WB, None, {}))
    live.build_api("practice_02", WB, None, {"arrivals": "bunched"})
    assert _arrivals(live.build_api("practice_02", WB, None, {})) == plain
    assert list(live.CACHE_DIR.glob("practice_02-*.pkl"))


def test_late_arrivals_never_run_past_the_window():
    api = live.build_api("practice_02", WB, None, {"arrivals": "late", "seed": 9})
    assert all(t <= api.end for t in _arrivals(api).values())


def test_unknown_crane_in_a_breakdown_is_ignored():
    api = live.build_api("practice_02", WB, None, {"breakdowns": [{"crane": "QC99", "at_h": 1, "hours": 2}]})
    assert "QC99" not in api.down


def test_meta_describes_the_terminal():
    m = live.scenario_meta(live.build_api("practice_02", WB, None, {}))
    assert m["ships"] and m["cranes"] and m["berths"]


# ---- scenarios dropped onto the page as a workbook

def _book(tmp_path, name, sheets_prefix, scen_map, skip=()):
    import pandas as pd
    path = tmp_path / name
    with pd.ExcelWriter(path) as w:
        for part in ("Vessels", "Events", "Manifest", "Baseline_FCFS"):
            if part in skip:
                continue
            df = pd.read_excel(WB, sheet_name="Practice_" + part)
            df = df[df["scenario"].isin(scen_map)].copy()
            df["scenario"] = df["scenario"].map(scen_map)
            df.to_excel(w, sheet_name=sheets_prefix + part, index=False)
    return path.read_bytes()


@pytest.fixture
def _uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setitem(live.STATE, "uploads", {})
    monkeypatch.setitem(live.STATE, "data", WB)
    monkeypatch.setitem(live.STATE, "key", None)


def test_a_practice_style_workbook_adds_a_runnable_scenario(tmp_path, _uploads):
    raw = _book(tmp_path, "day.xlsx", "Practice_", {"practice_03": "judges_day"}, skip=("Baseline_FCFS",))
    res = live.import_workbook(raw, "day.xlsx")
    assert [a["id"] for a in res["added"]] == ["judges_day"] and not res["errors"]
    assert "judges_day" in live.scenarios()
    api = live.build_api("judges_day", live.data_for("judges_day"), None, {"arrivals": "late", "seed": 2})
    assert len(api.calls) == res["added"][0]["ships"]


def test_answer_key_style_names_are_renamed_so_they_do_not_clash(tmp_path, _uploads):
    raw = _book(tmp_path, "k.xlsx", "Hidden_", {"practice_04": "hidden_01", "practice_01": "practice_01"})
    ids = sorted(a["id"] for a in live.import_workbook(raw, "k.xlsx")["added"])
    assert ids == ["new_hidden_01", "new_practice_01"]


def test_a_workbook_without_the_sheets_is_refused(tmp_path, _uploads):
    import pandas as pd
    p = tmp_path / "x.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(p, index=False)
    res = live.import_workbook(p.read_bytes(), "x.xlsx")
    assert not res["added"] and res["errors"]
    assert not live.import_workbook(b"not a workbook", "y.xlsx")["added"]


def test_missing_columns_are_named(tmp_path, _uploads):
    import pandas as pd
    p = tmp_path / "c.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame({"scenario": ["s"], "call_id": ["X"]}).to_excel(w, sheet_name="Practice_Vessels", index=False)
        pd.DataFrame({"scenario": ["s"]}).to_excel(w, sheet_name="Practice_Manifest", index=False)
    res = live.import_workbook(p.read_bytes(), "c.xlsx")
    assert not res["added"] and any("loa_m" in e for e in res["errors"])


def test_the_template_goes_straight_back_in(tmp_path, _uploads):
    res = live.import_workbook(live.template_workbook(), "template.xlsx")
    assert [a["id"] for a in res["added"]] == ["my_scenario"]
