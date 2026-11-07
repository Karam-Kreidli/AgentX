"""CORE-6 tests: the scorecard's checks, comparison with main, output, and a real run in < 1 minute."""
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tools import eval_all as ev

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")


def row(label="practice_01", **kw):
    r = {"label": label, "crashed": None, "violations": 0, "refused_calls": 0, "breaches": 0, "exceptions": 0,
         "departed": 4, "total": 4, "honest_h": 40.0, "sim_h": 40.0, "fcfs_h": 30.0, "moves_h": 15.0,
         "truck_idle_h": 100.0, "over85": 10, "delivered": 500, "missing_reasons": 0, "guard_refusals": 0,
         "mismatches": 0, "watchdog": 0, "rules_only_ticks": 0, "escalations": 0, "llm_calls": 0,
         "llm_budget": 150, "wall_s": 12.0, "unfinished": []}
    r.update(kw)
    return r


def card(*rows):
    return {"commit": "abc", "rows": list(rows)}


# ---------------------------------------------------------------- absolute checks

def test_clean_row_passes_with_and_without_main():
    assert ev.checks(row(), None) == []
    assert ev.checks(row(), row()) == []


@pytest.mark.parametrize("field,value,needle", [
    ("violations", 1, "violations 1"),
    ("refused_calls", 2, "refused calls 2"),        # a refused call counts as a violation (toolkit README)
    ("breaches", 1, "breaches 1"),
    ("exceptions", 3, "3 unhandled exceptions"),
    ("missing_reasons", 1, "1 actions without a reason"),
    ("wall_s", 300.0, "runtime 300.0 s"),
    ("llm_calls", 151, "LLM calls 151 > budget 150"),
])
def test_each_must_pass_column_fails_on_its_own(field, value, needle):
    fails = ev.checks(row(**{field: value}), None)
    assert len(fails) == 1 and needle in fails[0]


def test_crashed_scenario_fails():
    assert ev.checks(row(crashed="KeyError: x"), row()) == ["crashed: KeyError: x"]


def test_violations_total_counts_refusals_and_breaches():
    assert ev.violations_total(row(violations=1, refused_calls=2, breaches=3)) == 6


# ---------------------------------------------------------------- comparison with main

def test_fewer_ships_departed_than_main_fails():
    fails = ev.checks(row(departed=3), row(departed=4))
    assert fails == ["ships departed 3 < main 4"]


def test_more_ships_departed_passes():
    assert ev.checks(row(departed=4), row(departed=3)) == []


def test_turnaround_within_3_percent_passes_and_beyond_fails():
    assert ev.checks(row(honest_h=41.2), row(honest_h=40.0)) == []            # exactly +3%
    fails = ev.checks(row(honest_h=41.3), row(honest_h=40.0))
    assert len(fails) == 1 and "honest turnaround 41.3 h > main 40.0 h + 3%" in fails[0]


def test_crashed_main_row_is_not_compared():
    assert ev.checks(row(departed=1), row(crashed="boom", departed=None)) == []


def test_compare_matches_rows_by_label_and_skips_labels_main_lacks():
    fails = ev.compare([row("practice_01", departed=3), row("practice_02", departed=1)],
                       card(row("practice_01", departed=4)))
    assert fails == {"practice_01": ["ships departed 3 < main 4"], "practice_02": []}


def test_no_baseline_only_absolute_checks():
    assert ev.compare([row(departed=0)], None) == {"practice_01": []}


# ---------------------------------------------------------------- rows, totals, output

def test_make_row_from_summary_and_logs():
    summary = {"violations": [], "vessels_departed": 3, "vessels_total": 6, "honest_turnaround_h": 49.93,
               "avg_turnaround_h": 29.36, "fcfs_baseline_avg_turnaround_h_same_calls": 25.93,
               "crane_moves_per_hour": 8.5, "truck_idle_hours": 648.0, "placements_into_blocks_over_85pct": 1060,
               "import_boxes_delivered": 773, "unfinished_calls": ["P02-003"], "wall_s": 12.6,
               "loop": {"exceptions": 0, "breaches": 0, "actions_refused": 2, "mismatches": 0,
                        "watchdog": 1, "rules_only_ticks": 1},
               "llm": {"requests_used": 7, "budget": 150}}
    sim = {"VIOLATION": 1, "tool_error": 2}
    r = ev.make_row("practice_02", summary, sim, missing=0, escalations=3)
    assert (r["departed"], r["total"], r["honest_h"], r["fcfs_h"]) == (3, 6, 49.93, 25.93)
    assert (r["violations"], r["refused_calls"], r["guard_refusals"], r["escalations"]) == (1, 2, 2, 3)
    assert (r["llm_calls"], r["llm_budget"], r["watchdog"]) == (7, 150, 1)


def test_totals():
    t = ev.totals([row(departed=4, total=4, honest_h=40.0), row("p2", departed=3, total=6, honest_h=50.0,
                                                                refused_calls=1), row("p3", crashed="x")])
    assert (t["departed"], t["total"], t["honest_h_mean"], t["violations"], t["crashes"]) == (7, 10, 45.0, 1, 1)


def test_markdown_is_the_pr_template_table():
    md = ev.markdown([row(departed=3, total=6, honest_h=50.0)], card(row(departed=4, total=6, honest_h=49.0)))
    lines = md.splitlines()
    assert lines[0] == "| Scenario | Ships departed (before → after) | Honest turnaround h | Violations | Crashes |"
    assert lines[2] == "| practice_01 | 4/6 → 3/6 | 49.0 → 50.0 | 0 → 0 | 0 → 0 |"


def test_plan_adds_variant_workbooks(tmp_path, monkeypatch):
    for name in ("v02", "v01", "v03"):
        (tmp_path / f"{name}.xlsx").write_bytes(b"")
    monkeypatch.setattr(ev, "VARIANTS_DIR", tmp_path)
    spec = ev.plan(["practice_01", "practice_02"], 2, Path("data.xlsx"))
    assert [label for label, _, _ in spec] == ["practice_01", "practice_02", "v01/practice_01",
                                               "v01/practice_02", "v02/practice_01", "v02/practice_02"]
    assert ev.plan(["practice_01"], 0, Path("d.xlsx")) == [("practice_01", "practice_01", Path("d.xlsx"))]


def test_save_writes_latest_by_commit_and_main_only_when_asked(tmp_path):
    c = {"commit": "0123456789abcdef", "rows": []}
    ev.save(c, tmp_path, None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["0123456789ab.json", "scorecard.json"]
    ev.save(c, tmp_path, ev.main_file(tmp_path, "multi", True))
    assert (tmp_path / "main-multi-no-llm.json").exists()
    assert json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))["commit"] == c["commit"]


# ---------------------------------------------------------------- real runs

@needs_workbook
def test_all_practice_scenarios_in_under_a_minute(tmp_path, capsys):
    t0 = time.perf_counter()
    code = ev.main(["--all", "--no-llm", "--out", str(tmp_path), "--baseline", str(_baseline_of_itself(tmp_path))])
    took = time.perf_counter() - t0
    assert took < 60, f"scorecard took {took:.0f} s (spec: under 1 minute with --no-llm)"
    assert code == 0
    saved = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    assert [r["label"] for r in saved["rows"]] == ev.PRACTICE
    for r in saved["rows"]:
        assert r["crashed"] is None and ev.violations_total(r) == 0 and r["missing_reasons"] == 0
        # The row agrees with the loop's own summary.json for that scenario.
        s = json.loads((tmp_path / "runs" / r["label"] / r["label"] / "summary.json").read_text(encoding="utf-8"))
        assert (r["departed"], r["honest_h"]) == (s["vessels_departed"], s["honest_turnaround_h"])
    out = capsys.readouterr().out
    assert "FAIL" not in out and "total:" in out


def _baseline_of_itself(tmp_path):
    """A baseline equal to the floor every change must hold (main on 7 Nov), as a file."""
    floor = {"practice_01": (4, 4, 38.69), "practice_02": (3, 6, 49.93), "practice_03": (4, 6, 34.18),
             "practice_04": (3, 5, 37.02), "practice_05": (4, 4, 34.52)}
    rows = [row(k, departed=d, total=t, honest_h=h) for k, (d, t, h) in floor.items()]
    path = tmp_path / "floor.json"
    path.write_text(json.dumps(card(*rows)), encoding="utf-8")
    return path


@needs_workbook
def test_a_scenario_that_cannot_run_is_a_crash_and_fails(tmp_path):
    code = ev.main(["--scenario", "practice_99", "--scenario", "practice_05", "--no-llm", "--out", str(tmp_path)])
    saved = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    bad, good = saved["rows"]
    assert bad["crashed"] and good["crashed"] is None and good["departed"] == 4   # one bad scenario doesn't stop others
    assert code == 1 and saved["failures"]["practice_99"][0].startswith("crashed")


@needs_workbook
@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_against_a_git_ref_runs_that_commit_in_a_temporary_worktree(tmp_path):
    before = subprocess.run(["git", "worktree", "list"], cwd=ROOT, capture_output=True, text=True).stdout
    code = ev.main(["--scenario", "practice_05", "--no-llm", "--out", str(tmp_path), "--against", "HEAD"])
    saved = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    assert code == 0 and saved["baseline"]["source"] == "fresh run of HEAD"
    assert saved["failures"] == {"practice_05": []}
    after = subprocess.run(["git", "worktree", "list"], cwd=ROOT, capture_output=True, text=True).stdout
    assert after == before                                         # the temporary checkout is removed


def test_compare_does_not_depend_on_row_order():
    rows = [row("practice_01", departed=4), row("practice_02", departed=2)]
    main = card(row("practice_02", departed=3), row("practice_01", departed=4))      # different order
    assert ev.compare(rows, main) == {"practice_01": [], "practice_02": ["ships departed 2 < main 3"]}


def test_output_survives_a_cp1252_console(monkeypatch):
    # Windows writes redirected output in cp1252, which has no "→".
    import io
    import sys
    buf = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buf, encoding="cp1252"))
    ev._utf8_stdout()
    print(ev.markdown([row()], card(row())))
    sys.stdout.flush()
    assert "4/4 → 4/4".encode("utf-8") in buf.getvalue()
