"""CORE-8 tests for tools/compare.py: aggregation, win/tie/loss, and a well-formed chart."""
import sys
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare  # noqa: E402


def row(label, dep, h, total=4, **kw):
    r = {"label": label, "crashed": None, "violations": 0, "refused_calls": 0, "breaches": 0, "departed": dep,
         "total": total, "honest_h": h, "sim_h": h, "moves_h": 22.0, "truck_idle_h": 10.0, "over85": 5, "wall_s": 10.0}
    r.update(kw)
    return r


def test_set_of():
    assert compare.set_of("practice_01") == "practice"
    assert compare.set_of("v03/practice_02") == "variants"
    assert compare.set_of("m01/practice_02") == "malformed"


def test_aggregate_sums_and_counts_crashes():
    rows = [row("a", 3, 20.0), row("b", 2, 30.0, violations=1, refused_calls=2), {"label": "c", "crashed": "boom"}]
    a = compare.aggregate(rows)
    assert (a["departed"], a["total"], a["honest_h"], a["violations"], a["crashes"], a["scenarios"]) == (5, 8, 25.0, 3, 1, 3)


def test_paired_win_tie_loss():
    multi = [row("s1", 4, 20.0), row("s2", 3, 20.0), row("s3", 3, 20.0), row("s4", 2, 20.0)]
    single = [row("s1", 3, 25.0), row("s2", 3, 20.0), row("s3", 3, 18.0), row("s4", 4, 30.0)]
    p = compare.paired(multi, single)
    assert p["n"] == 4
    assert p["departed"] == {"win": 1, "tie": 2, "loss": 1}
    assert p["honest_h"] == {"win": 2, "tie": 1, "loss": 1}
    assert p["not_worse"] == 2          # s1 and s2; s3 is 11 % slower; s4 lost on departed


def test_paired_ignores_crashed_and_missing():
    p = compare.paired([row("a", 1, 1.0), {"label": "b", "crashed": "x"}], [row("a", 1, 1.0)])
    assert p["n"] == 1


def test_chart_is_wellformed_xml():
    card = {"practice_rows": {k: [row(f"practice_0{i}", i, 10.0 * i) for i in range(1, 6)] for k, _ in compare.SYSTEMS}}
    for metric in ("departed", "honest_h"):
        svg = compare.svg_chart(card, metric, "A & B <title>", " h")
        root = ElementTree.fromstring(svg)
        assert root.tag.endswith("svg")
        assert svg.count("<rect") > 20
