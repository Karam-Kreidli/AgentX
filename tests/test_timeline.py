"""SAFE-7: tools/timeline.py builds from any run directory and never crashes on bad logs."""
import json

from tools import timeline


def _w(p, lines):
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tick(tick, t, status, berth=None, crane="available", used=0):
    return json.dumps({"tick": tick, "t": t, "now": f"T{tick}",
                       "ships": [{"id": "C-1", "status": status, "berth": berth}],
                       "cranes": [{"id": "QC1", "status": crane, "call": None}],
                       "blocks": {"A": [used, 100]}})


def test_empty_run_dir_gives_an_empty_but_valid_timeline(tmp_path):
    jp, hp = timeline.export(tmp_path)
    d = json.loads(jp.read_text())
    assert d["schema"] == timeline.SCHEMA and d["ships"] == [] and d["actions"] == []
    assert "<script" in hp.read_text()


def test_intervals_events_and_classification(tmp_path):
    _w(tmp_path / "ticks.jsonl", [_tick(0, 0.0, "expected"), "not json", _tick(1, 1.0, "arrived"),
                                  _tick(2, 2.0, "working", "B1", "out_of_service", 50), _tick(3, 3.0, "working", "B1")])
    dec = [{"t": 1.0, "agent": "berth", "choice": "assign_berth", "reason": "fits", "guard_ok": True,
            "tool_result": {"ok": True}},
           {"t": 1.5, "agent": "yard", "choice": "place_container", "reason": "r", "guard_ok": False,
            "guard_reason": "reefer into a non-reefer block"},
           {"t": 2.0, "agent": "yard", "choice": "escalate", "reason": "no slot", "guard_ok": True},
           {"t": 2.5, "agent": "loop", "choice": "CRITICAL", "reason": "stuck clock", "guard_ok": True}]
    _w(tmp_path / "decisions.jsonl", [json.dumps(d) for d in dec] + ["[1,2]"])
    _w(tmp_path / "negotiation.jsonl", [json.dumps({"t": 2.0, "kind": "request", "agent": "yard",
                                                    "resource": "slots", "amount": 5, "call_id": "C-1"})])
    d = timeline.build(tmp_path)
    assert [iv["status"] for iv in d["ships"][0]["intervals"]] == ["expected", "arrived", "working"]
    assert d["ships"][0]["intervals"][-1]["t1"] == 3.0
    assert any("out_of_service" in e["text"] for e in d["events"])
    assert d["blocks"][0]["fill"] == [[0.0, 0], [2.0, 50], [3.0, 0]]
    assert len(d["actions"]) == 1 and d["refusals"][0]["by"] == "guard"
    assert len(d["escalations"]) == 1 and d["alerts"][0]["choice"] == "CRITICAL"
    assert "asks for 5 slots" in d["negotiation"][0]["text"]
    assert d["meta"]["skipped_lines"] == {"ticks.jsonl": 1, "decisions.jsonl": 1}


def test_log_text_cannot_break_out_of_the_page(tmp_path):
    _w(tmp_path / "decisions.jsonl", [json.dumps({"agent": "yard", "choice": "escalate", "guard_ok": True,
                                                  "reason": "<!-- </script><script>alert(1)</script>"})])
    page = timeline.render_html(timeline.build(tmp_path))
    assert page.count("</script>") == 2          # only the page's own two script tags
    assert "<!--" not in page
    a = page.index('application/json">') + 18
    json.loads(page[a:page.index("</script>", a)])   # embedded data must still parse as JSON


def test_cli_reports_a_missing_dir_without_a_traceback(tmp_path, capsys):
    assert timeline.main([str(tmp_path / "nope")]) == 1
    assert "not a directory" in capsys.readouterr().err
