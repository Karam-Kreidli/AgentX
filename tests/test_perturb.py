"""CORE-7: tools/perturb.py makes variant workbooks. These tests need the workbook in data/."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import perturb  # noqa: E402

WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")


@pytest.fixture(scope="module")
def source():
    return perturb.load(WORKBOOK)


@pytest.fixture(scope="module")
def identity_set(tmp_path_factory):
    out = tmp_path_factory.mktemp("variants")
    index = perturb.build(WORKBOOK, out, n=2, seed=1, families=["identity"])
    return out, index


def _same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=True)


@needs_workbook
def test_identity_round_trip_is_lossless(identity_set, source):
    out, _ = identity_set
    back = perturb.load(out / "v01.xlsx")
    for k in ("V", "E", "M"):                      # dtypes, NaT, millisecond times, NaN imdg_class
        _same(source.sheets[k], back.sheets[k])
    assert back.sheets["B"].empty
    assert list(back.sheets["B"].columns) == list(source.sheets["B"].columns)


@needs_workbook
def test_static_sheets_are_unchanged(identity_set, source):
    out, _ = identity_set
    back = perturb.load(out / "v02.xlsx")
    for name in perturb.STATIC_SHEETS:
        _same(source.static[name], back.static[name])


@needs_workbook
def test_source_workbook_is_untouched(source, tmp_path):
    before = perturb.sha256(WORKBOOK)
    perturb.build(WORKBOOK, tmp_path, n=1, seed=1, families=["identity"])
    assert perturb.sha256(WORKBOOK) == before == source.sha256


@needs_workbook
def test_same_seed_same_result_and_index_records_it(identity_set, tmp_path):
    out, index = identity_set
    again = perturb.build(WORKBOOK, tmp_path, n=2, seed=1, families=["identity"])
    assert again["books"] == index["books"]
    assert json.loads((out / "index.json").read_text())["source_sha256"] == index["source_sha256"]
    assert set(index["books"]["v01"]) == set(perturb.SCENARIOS)


@needs_workbook
def test_toolkit_loads_every_scenario_of_a_variant(identity_set):
    from vendor import TerminalAPI
    out, _ = identity_set
    for sc in perturb.SCENARIOS:
        api = TerminalAPI(sc, data_xlsx=str(out / "v01.xlsx"), log_path=str(out / f"log_{sc}.jsonl"))
        assert len(api.calls) >= 4
        assert api.baseline.empty                     # FCFS sheet is empty on purpose (corePlan D2)


@needs_workbook
def test_nothing_is_left_half_written(identity_set):
    out, _ = identity_set
    assert not list(out.glob("*.tmp.xlsx"))


@needs_workbook
def test_unknown_family_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        perturb.build(WORKBOOK, tmp_path, n=1, seed=1, families=["nope"])
