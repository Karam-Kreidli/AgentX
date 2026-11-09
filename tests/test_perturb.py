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


# ---------------------------------------------------------------------------
# Families, malformed kinds, the sets (CORE-7 steps 3-8)
# ---------------------------------------------------------------------------

import ast  # noqa: E402

import numpy as np  # noqa: E402


@pytest.fixture(scope="module")
def base(source):
    return perturb.split(source)


def _apply(base, fam, sc="practice_02", i=0, registry=None, seed=1):
    rng = np.random.default_rng([seed, 0, i])
    return (registry or perturb.FAMILIES)[fam](base[sc], rng)


@needs_workbook
def test_static_assumptions_match_the_workbook(source):
    perturb.check_static(source.static)             # exits if berths, yard capacity, crane or truck ids differ


@needs_workbook
@pytest.mark.parametrize("fam", [f for f in perturb.FAMILIES if f != "identity"])
def test_every_family_makes_valid_data_on_every_scenario(base, fam):
    for sc in perturb.SCENARIOS:
        f, meta = _apply(base, fam, sc)
        assert perturb.check_variant(f, fam, base[sc]) == [], (fam, sc)
        assert set(meta) == {"params", "expect", "shortage"}


@needs_workbook
@pytest.mark.parametrize("fam", [f for f in perturb.FAMILIES if f != "identity"])
def test_families_are_deterministic_and_never_change_their_input(base, fam):
    before = base["practice_03"].M.copy()
    a, ma = _apply(base, fam, "practice_03")
    b, mb = _apply(base, fam, "practice_03")
    _same(a.V, b.V), _same(a.E, b.E), _same(a.M, b.M)
    assert ma == mb
    _same(base["practice_03"].M, before)


@needs_workbook
def test_moves_still_match_boxes_the_way_the_source_does(base):
    # P03-004 is the deliberate impossible manifest: 1,600 moves for 551 boxes. It must stay exactly that.
    for fam in ("tight", "noshort_fewer", "noshort_dwell", "mixed", "renamed"):
        f, _ = _apply(base, fam, "practice_03")
        assert perturb._deltas(f) == perturb._deltas(base["practice_03"])
    assert max(d for d, _ in perturb._deltas(base["practice_03"])) > 1000


@needs_workbook
@pytest.mark.parametrize("fam,share", [("noshort_fewer", 0.60), ("noshort_dwell", 0.85)])
def test_no_shortage_imports_fit_the_legal_slots_of_every_cargo_group(base, fam, share):
    for sc in perturb.SCENARIOS:
        f, meta = _apply(base, fam, sc)
        imp = perturb.imports(f)
        for group, capacity in perturb.GROUP_CAPACITY.items():
            n = int((imp["cargo_type"] == group).sum())
            assert n <= share * capacity, (sc, group, n)
            assert meta["shortage"][group]["imports_after"] == n
        assert meta["expect"] == {"noshort": True}


@needs_workbook
def test_no_shortage_dwell_is_shortened_and_holds_follow_their_boxes(base):
    f, meta = _apply(base, "noshort_dwell", "practice_01")
    imp = perturb.imports(f)
    assert imp["planned_dwell_days"].max() <= meta["params"]["dwell_cap_days"] <= 3
    holds = f.E[f.E.event_type == "customs_hold"]
    assert set(holds.target) <= set(f.M.container_id)


@needs_workbook
def test_renamed_keeps_sort_order_and_changes_only_ids_and_names(base):
    sc = "practice_04"
    f, meta = _apply(base, "renamed", sc)
    o = base[sc]
    assert meta["expect"] == {"same_as": True}
    # order-preserving: sorting by the new id gives the old ships in the old order
    assert list(f.V.sort_values("call_id").vessel_name) == [f"MV Test {i:02d}" for i in range(1, len(f.V) + 1)]
    assert list(o.V.sort_values("call_id").index) == list(f.V.sort_values("call_id").index)
    assert list(o.M.sort_values("container_id").index) == list(f.M.sort_values("container_id").index)
    other = [c for c in o.V.columns if c not in ("call_id", "vessel_name")]
    _same(o.V[other], f.V[other])
    other = [c for c in o.M.columns if c not in ("call_id", "container_id")]
    _same(o.M[other], f.M[other])
    other = [c for c in o.E.columns if c not in ("target", "detail")]
    _same(o.E[other], f.E[other])
    assert set(f.E[f.E.event_type == "customs_hold"].target) <= set(f.M.container_id)
    assert not set(f.V.call_id) & set(o.V.call_id) and not set(f.M.container_id) & set(o.M.container_id)


@needs_workbook
def test_shifted_never_moves_the_window_and_stays_inside_it(base):
    for sc in perturb.SCENARIOS:
        f, _ = _apply(base, "shifted", sc)
        start, end = perturb.window(f)
        assert (start, end) == perturb.window(base[sc])
        assert pd.to_datetime(f.V.actual_arrival).max() <= end - perturb.EDGE


@needs_workbook
def test_breakdowns_put_two_rail_neighbours_down_together_and_use_real_equipment(base):
    f, meta = _apply(base, "breakdowns", "practice_01")
    new = f.E.iloc[[i for i in f.E.index if (f.E.at[i, "event_type"], f.E.at[i, "target"], str(f.E.at[i, "start"]))
                    in {(("crane_breakdown" if t.startswith("QC") else "truck_breakdown"), t, s) for t, s, _ in meta["params"]["events"]}]]
    cranes = new[new.event_type == "crane_breakdown"].sort_values("start")
    assert set(new.target) <= set(perturb.CRANE_IDS) | set(perturb.TRUCK_IDS)
    ids = [int(t[2:]) for t in cranes.target]
    assert any(abs(a - b) == 1 and ps < pe and qs < pe
               for (a, ps, pe) in zip(ids, cranes.start, cranes.end) for (b, qs, qe) in zip(ids, cranes.start, cranes.end)
               if a != b and qs >= ps)


@needs_workbook
def test_holds_target_import_boxes_and_some_are_never_released(base):
    f, meta = _apply(base, "holds", "practice_02")
    new = f.E[f.E.event_type == "customs_hold"]
    assert set(new.target) <= set(perturb.imports(f).container_id)
    assert not new.target.duplicated().any()
    assert 10 <= meta["params"]["extra_holds"] <= 30


@needs_workbook
def test_resized_swaps_sizes_and_every_ship_still_fits_a_berth(base):
    f, meta = _apply(base, "resized", "practice_02")
    o = base["practice_02"]
    assert sorted(zip(f.V.loa_m, f.V.draft_m)) == sorted(zip(o.V.loa_m, o.V.draft_m))
    assert meta["params"]["swapped"]


@needs_workbook
def test_tight_adds_imports_with_new_ids_and_matching_moves(base):
    f, meta = _apply(base, "tight", "practice_05")
    o = base["practice_05"]
    assert len(perturb.imports(f)) - len(perturb.imports(o)) == meta["params"]["added_imports"] > 0
    assert (perturb.imports(f).planned_dwell_days.max() == perturb.imports(o).planned_dwell_days.max() + 1)
    assert f.V.discharge_moves.sum() - o.V.discharge_moves.sum() == meta["params"]["added_imports"]


@needs_workbook
def test_assign_meets_every_family_on_every_scenario_over_a_full_set():
    fams = perturb.DEFAULT_FAMILIES
    for i in range(5):
        seen = {perturb.assign(k, i, fams) for k in range(len(fams))}
        assert seen == set(fams)


# ---- malformed ----

def _mal(base, kind, sc="practice_02"):
    return _apply(base, kind, sc, registry=perturb.MALFORMED)


@needs_workbook
def test_nan_size_and_too_big_name_their_targets(base):
    f, m = _mal(base, "nan_size")
    a, b = m["expect"]["never_berthed"]
    assert pd.isna(f.V.loc[f.V.call_id == a, "loa_m"]).all() and pd.isna(f.V.loc[f.V.call_id == b, "draft_m"]).all()
    f, m = _mal(base, "too_big")
    a, b = m["expect"]["never_berthed"]
    assert (f.V.loc[f.V.call_id == a, "loa_m"] == 340).all() and (f.V.loc[f.V.call_id == b, "draft_m"] == 15.5).all()


@needs_workbook
def test_impossible_has_three_times_the_moves_of_its_boxes(base):
    f, m = _mal(base, "impossible")
    c = m["params"]["ship"]
    n = len(perturb.imports(f)[perturb.imports(f).call_id == c])
    assert int(f.V.loc[f.V.call_id == c, "discharge_moves"].iloc[0]) == 3 * n
    assert m["expect"]["not_departed"] == [c] and m["expect"]["event"] == "ship_impossible"


@needs_workbook
def test_bad_cargo_kinds_change_exactly_the_boxes_they_name(base):
    o = base["practice_02"]
    for kind, bad in (("unknown_cargo", lambda s: ~s.isin(perturb.KNOWN_CARGO) & s.notna()),
                      ("nan_cargo", lambda s: s.isna())):
        f, m = _mal(base, kind)
        ids = m["expect"]["never_placed"]
        assert len(ids) == 5
        changed = f.M[f.M.cargo_type.fillna("<nan>") != o.M.cargo_type.fillna("<nan>")]
        assert sorted(changed.container_id) == sorted(ids)
        assert bad(f.M.loc[f.M.container_id.isin(ids), "cargo_type"]).all()
    f, m = _mal(base, "imdg_conflict")
    ids = m["expect"]["never_placed"]
    sub = f.M[f.M.container_id.isin(ids)]
    assert len(sub) == 3 and (sub.cargo_type == "dry").all() and (sub.imdg_class == 3.0).all()


@needs_workbook
def test_duplicate_box_repeats_ids_once_on_a_ship_and_once_across_ships(base):
    f, m = _mal(base, "duplicate_box")
    a, b = m["expect"]["never_placed"]
    assert (f.M.container_id == a).sum() == 2 and (f.M.container_id == b).sum() == 2
    assert f.M[f.M.container_id == a].call_id.nunique() == 1 and f.M[f.M.container_id == b].call_id.nunique() == 2
    ships = set(m["params"]["two_ships"]) | {m["params"]["same_ship"]}
    assert set(m["expect"]["anomaly"]) == ships            # YARD-5 names the ships, not the boxes


@needs_workbook
def test_negative_moves_bad_times_bad_box_data_and_orphan_hold(base):
    f, m = _mal(base, "negative_moves")
    assert (f.V.discharge_moves < 0).sum() == 1 and (f.V.load_moves < 0).sum() == 1
    f, m = _mal(base, "bad_times")
    nat, = m["expect"]["never_berthed"]
    assert f.V.loc[f.V.call_id == nat, "actual_arrival"].isna().all()
    late = m["params"]["eta_after_window"]
    assert pd.to_datetime(f.V.loc[f.V.call_id == late, "scheduled_eta"]).iloc[0] >= perturb.window(base["practice_02"])[1]
    assert perturb.window(f)[0] == perturb.window(base["practice_02"])[0]
    f, m = _mal(base, "bad_box_data")
    sub = perturb.imports(f)
    assert sub.gross_weight_t.isna().sum() == 1 and (sub.gross_weight_t == -3.0).sum() == 1
    assert (sub.size_ft == 45).sum() == 1 and sub.size_ft.isna().sum() == 1
    f, m = _mal(base, "orphan_hold")
    assert m["params"]["target"] in set(f.E.target) and m["params"]["target"] not in set(f.M.container_id)


@needs_workbook
def test_malformed_books_leave_static_sheets_alone_except_the_yard_book(tmp_path, source):
    idx = perturb.build(WORKBOOK, tmp_path, n=0, seed=1, families=["identity"], malformed=1)
    assert set(idx["books"]) == {"m01", "m_yard"}
    m01 = perturb.load(tmp_path / "m01.xlsx")
    for name in perturb.STATIC_SHEETS:
        _same(source.static[name], m01.static[name])
    yard = perturb.load(tmp_path / "m_yard.xlsx")
    for name in perturb.STATIC_SHEETS:
        if name != "Yard_Blocks":
            _same(source.static[name], yard.static[name])
    y = yard.static["Yard_Blocks"].set_index("block_id").slots
    assert y["A"] == 0 and y["H"] == 10 and y["B"] == 220
    assert {m["family"] for m in idx["books"]["m_yard"].values()} == {"yard_capacity"}
    assert not list(tmp_path.glob("*.tmp.xlsx"))


@needs_workbook
def test_four_malformed_books_cover_every_kind(tmp_path):
    idx = perturb.build(WORKBOOK, tmp_path, n=0, seed=1, families=["identity"], malformed=4)
    kinds = {m["family"] for b in idx["books"].values() for m in b.values()}
    assert kinds == set(perturb.MALFORMED) | {perturb.YARD_KIND}
    scenarios = sum(len(b) for k, b in idx["books"].items() if k.startswith("m"))
    assert scenarios >= 10


@needs_workbook
def test_the_toolkit_loads_a_perturbed_a_malformed_and_a_yard_book(tmp_path):
    from vendor import TerminalAPI
    perturb.build(WORKBOOK, tmp_path, n=1, seed=1, families=["mixed"], malformed=1)
    for book in ("v01", "m01", "m_yard"):
        api = TerminalAPI("practice_03", data_xlsx=str(tmp_path / f"{book}.xlsx"), log_path=str(tmp_path / f"{book}.jsonl"))
        assert len(api.calls) >= 4


# ---- the other sets ----

@needs_workbook
def test_ci_set_is_fixed_and_small(tmp_path):
    a = perturb.build_ci(WORKBOOK, tmp_path / "a")
    b = perturb.build_ci(WORKBOOK, tmp_path / "b")
    assert a["books"] == b["books"] and a["seed"] == 7
    assert sorted(a["books"]) == ["m01", "m02", "v01", "v02"]
    assert {m["family"] for m in a["books"]["v01"].values()} == {"mixed"}
    assert {m["family"] for m in a["books"]["v02"].values()} == {"noshort_fewer", "noshort_dwell"}
    kinds = [m["family"] for k in ("m01", "m02") for m in a["books"][k].values()]
    assert sorted(kinds) == sorted(list(perturb.MALFORMED)[:10])      # the first ten kinds, once each


@needs_workbook
def test_limits_set_has_one_toolkit_limit_per_book(tmp_path):
    idx = perturb.build_limits(WORKBOOK, tmp_path)
    assert {next(iter(b.values()))["family"] for b in idx["books"].values()} == set(perturb.LIMITS)


@needs_workbook
def test_hidden_key_loads_the_way_the_organizers_do(tmp_path):
    from vendor import TerminalAPI
    base = perturb.split(perturb.load(WORKBOOK))
    used = perturb.write_hidden_key(tmp_path / "MOCK_ANSWER_KEY.xlsx", base, seed=1)
    assert sorted(used) == [f"hidden_0{i}" for i in range(1, 6)]
    api = TerminalAPI("hidden_01", data_xlsx=str(WORKBOOK), key_xlsx=str(tmp_path / "MOCK_ANSWER_KEY.xlsx"),
                      log_path=str(tmp_path / "h.jsonl"))
    assert len(api.calls) >= 4 and api.baseline.empty


@needs_workbook
def test_clean_removes_old_books_but_keeps_other_files(tmp_path):
    (tmp_path / "v99.xlsx").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("keep")
    perturb.build(WORKBOOK, tmp_path, n=1, seed=1, families=["identity"], clean=True)
    assert not (tmp_path / "v99.xlsx").exists() and (tmp_path / "notes.txt").exists()


def test_agent_code_never_names_the_variant_tools_or_folder():
    from tools import check_honest
    for path in check_honest.agent_files(ROOT):
        text = path.read_text(encoding="utf-8")
        for needle in ("data/variants", "tools.perturb", "tools/perturb", "perturb.py", "index.json"):
            assert needle not in text, f"{path.relative_to(ROOT)} mentions {needle}"
        ast.parse(text)


@needs_workbook
def test_eval_all_runs_the_ci_books_on_one_scenario(tmp_path):
    from tools import eval_all
    perturb.build_ci(WORKBOOK, tmp_path / "v")
    code = eval_all.main(["--scenario", "practice_01", "--variants", "4", "--variants-dir", str(tmp_path / "v"),
                          "--fcfs", "--no-llm", "--out", str(tmp_path / "out")])
    card = json.loads((tmp_path / "out" / "scorecard.json").read_text(encoding="utf-8"))
    assert code == 0, card["failures"]
    assert [r["label"] for r in card["rows"]] == ["practice_01", "v01/practice_01", "v02/practice_01",
                                                  "m01/practice_01", "m02/practice_01"]
    assert {r["family"] for r in card["rows"]} == {"practice", "mixed", "noshort_fewer", "nan_size", "imdg_conflict"}
