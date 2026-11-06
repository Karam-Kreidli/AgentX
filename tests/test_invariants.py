"""SAFE-2 tests: the invariant monitor.

Done when: zero breaches across all scenarios; a planted breach in a test is caught.
Plus: every rule has a passing and a failing case, breaches are reported once,
the monitor never raises, and it never imports the guard (independent implementation).
"""
import json
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

import pytest

import core.invariants as inv
from core import log
from core.invariants import Breach, Monitor, check_state, find_breaches
from core.skeleton import BERTHS, SkeletonAgent
from core.types import Action, Block, Container, Crane, Ship, Truck, WorldState

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
T0 = datetime(2026, 9, 1, 6, 0)
END = datetime(2026, 9, 6)
HOME = {"QC1": 1, "QC2": 14, "QC3": 27, "QC4": 40, "QC5": 53, "QC6": 66}


@pytest.fixture(autouse=True)
def _fresh():
    inv.reset()
    log.stop()
    yield
    inv.reset()
    log.stop()


# ---------------------------------------------------------------- helpers

def ship(cid, status="working", berth="B1", cranes=(), loa=250.0, draft=12.0, trucks=0, arrived=True):
    at_berth = status in ("berthed", "working", "sailing", "departed")
    return Ship(cid, "MV " + cid, loa, draft, T0, T0 if arrived else None, status,
                berth if at_berth else None, tuple(cranes), trucks, 100, 50, None)


def box(cid, cargo="dry", block=None, direction="import"):
    return Container(cid, "C1", direction, 40, cargo, 3.0 if cargo == "IMDG" else None, 20.0, block)


def world(ships=(), bays=None, broken=(), boxes=(), trucks=(), blocks=None, end=END):
    bays = {**HOME, **(bays or {})}
    on = {c: s.call_id for s in ships for c in s.cranes}
    cranes = tuple(Crane(c, "out_of_service" if c in broken else ("working" if c in on else "available"),
                         on.get(c), bays[c]) for c in sorted(bays))
    blocks = blocks or (Block("A", "standard", 10, 220), Block("G", "empty", 0, 200), Block("H", "IMDG", 1, 120),
                        Block("I", "reefer", 1, 200), Block("J", "reefer", 0, 200))
    return WorldState(T0, end, tuple(ships), cranes, tuple(trucks), blocks, (),
                      MappingProxyType({"C1": tuple(boxes)}), BERTHS)


def rules(state):
    return sorted({b.rule for b in find_breaches(state)[0]})


def soft_keys(state):
    return sorted(k.split("|")[0] for k, _ in find_breaches(state)[1])


CLEAN = dict(ships=[ship("C1", berth="B1", cranes=("QC1", "QC2")), ship("C2", berth="B2", cranes=("QC3", "QC4"))],
             bays={"QC1": 5, "QC2": 20, "QC3": 35, "QC4": 50},
             boxes=[box("D1", "dry", "A"), box("H1", "IMDG", "H"), box("R1", "reefer", "I"), box("U1", "reefer")])


def test_clean_state_has_no_breaches_and_no_warnings():
    s = world(**CLEAN)
    assert find_breaches(s) == ([], [])
    assert check_state(s) == []


# ---------------------------------------------------------------- rail order (idle and broken included)

def test_rail_order_idle_crane_out_of_order():
    s = world([ship("C1", cranes=("QC1",))], bays={"QC1": 5, "QC4": 20})     # idle QC4 left of QC2/QC3
    assert "rail_order" in rules(s)
    assert any(b.subject == "QC3/QC4" for b in find_breaches(s)[0])


def test_rail_order_broken_crane_counts_and_equal_bays_are_a_breach():
    assert "rail_order" in rules(world(bays={"QC3": 45}, broken=("QC3",)))
    assert "rail_order" in rules(world(bays={"QC2": 27}))                 # QC2 and QC3 on the same bay
    assert rules(world()) == []                                            # home positions are fine


# ---------------------------------------------------------------- spacing among working cranes

def test_working_cranes_one_bay_apart_is_a_breach():
    assert "crane_spacing" in rules(world([ship("C1", cranes=("QC1", "QC2"))], bays={"QC1": 10, "QC2": 11}))


def test_working_cranes_two_bays_apart_pass_and_idle_cranes_dont_count():
    assert rules(world([ship("C1", cranes=("QC1", "QC2"))], bays={"QC1": 10, "QC2": 12})) == []
    assert rules(world([ship("C1", cranes=("QC1",))], bays={"QC1": 10, "QC2": 11})) == []   # QC2 idle


# ---------------------------------------------------------------- IMDG and reefer blocks

def test_imdg_only_in_h_and_h_only_imdg():
    assert rules(world(boxes=[box("H1", "IMDG", "H")])) == []
    assert rules(world(boxes=[box("H1", "IMDG", "A")])) == ["imdg_block"]
    assert rules(world(boxes=[box("D1", "dry", "H")])) == ["imdg_block"]


def test_reefer_only_in_i_or_j_but_dry_may_use_i_j():
    assert rules(world(boxes=[box("R1", "reefer", "J")])) == []
    assert rules(world(boxes=[box("R1", "reefer", "A")])) == ["reefer_block"]
    assert rules(world(boxes=[box("R1", "reefer", "G")])) == ["reefer_block"]
    assert rules(world(boxes=[box("D1", "dry", "I")])) == []                # allowed: the rule is reefer -> I/J


def test_box_in_unknown_block_is_unverifiable():
    assert rules(world(boxes=[box("D1", "dry", "Z")])) == ["unverifiable"]


# ---------------------------------------------------------------- berths

def test_berth_fit_rules():
    assert rules(world([ship("C1", berth="B1", loa=290)])) == []
    assert rules(world([ship("C1", berth="B2", loa=290)])) == ["berth_too_short"]     # 280 m < 290 + 20
    assert rules(world([ship("C1", berth="B3", loa=180, draft=13.0)])) == ["berth_too_shallow"]
    assert rules(world([ship("C1", berth="B9")])) == ["unverifiable"]


def test_berth_shared_and_before_arrival():
    assert "berth_shared" in rules(world([ship("C1", berth="B1"), ship("C2", berth="B1")]))
    assert rules(world([ship("C1", status="berthed", arrived=False)])) == ["berth_before_arrival"]


def test_departed_ship_keeps_berth_label_without_breach():
    assert rules(world([ship("C1", berth="B1"), ship("C0", status="departed", berth="B1")])) == []


# ---------------------------------------------------------------- equipment

def test_crane_on_two_ships_and_crane_outside_its_berth():
    assert "crane_two_ships" in rules(world([ship("C1", berth="B1", cranes=("QC1",)),
                                             ship("C2", berth="B2", cranes=("QC1",))], bays={"QC1": 5}))
    s = world([ship("C1", berth="B1", cranes=("QC3",))], bays={"QC3": 35, "QC4": 45})   # QC3 is in B2's bays
    assert rules(s) == ["crane_in_berth"]


# ---------------------------------------------------------------- soft warnings never flip to rules-only

def test_soft_warnings_are_not_breaches():
    s = world([ship("C1", cranes=("QC1",), trucks=3), ship("C0", status="departed", berth="B2", trucks=2)],
              bays={"QC1": 5}, broken=("QC1",),
              boxes=[box("E1", "dry", "A", direction="export"), box("X1", "ammo", "A")],
              blocks=(Block("A", "standard", 230, 220), Block("H", "IMDG", 0, 120), Block("I", "reefer", 0, 200)))
    assert rules(s) == []
    assert soft_keys(s) == ["block_count", "broken_listed", "export_in_yard", "trucks_count",
                            "trucks_parked", "unknown_cargo"]
    assert check_state(s) == []


# ---------------------------------------------------------------- reporting behaviour

def test_breach_reported_once_then_again_after_it_clears():
    m = Monitor()
    bad, good = world(boxes=[box("R1", "reefer", "A")]), world()
    assert [b.rule for b in m.check(bad)] == ["reefer_block"]
    assert m.check(bad) == []                         # still there: not reported again
    assert m.check(good) == []
    assert [b.rule for b in m.check(bad)] == ["reefer_block"]     # came back: reported again


def test_new_run_resets_what_was_reported():
    m = Monitor()
    assert len(m.check(world(boxes=[box("R1", "reefer", "A")]))) == 1
    assert len(m.check(world(boxes=[box("R1", "reefer", "A")], end=datetime(2026, 9, 11)))) == 1


def test_changed_manifest_is_rechecked():
    m = Monitor()
    rows_ok = (box("R1", "reefer"),)
    s1 = world(boxes=rows_ok)
    assert m.check(s1) == [] and m.check(s1) == []    # same cached tuple: reused
    assert [b.rule for b in m.check(world(boxes=(box("R1", "reefer", "A"),)))] == ["reefer_block"]


def test_monitor_never_raises():
    out = check_state(object())                       # garbage in
    assert len(out) == 1 and out[0].rule == "monitor_error"   # the safe direction: counts as a breach


def test_breach_string_is_what_the_loop_logs():
    b = Breach("rail_order", "QC3/QC4", "QC3 at bay 45 is not left of QC4 at bay 40")
    assert str(b) == "[rail_order] QC3/QC4: QC3 at bay 45 is not left of QC4 at bay 40"
    assert b.ident == "rail_order|QC3/QC4"


def test_breaches_are_escalated_and_warnings_logged_once(tmp_path):
    log.start(tmp_path)
    s = world([ship("C0", status="departed", berth="B2", trucks=2)], boxes=[box("R1", "reefer", "A")])

    def logged():
        return [json.loads(l) for l in (tmp_path / log.DECISIONS).read_text().splitlines()]

    check_state(s)
    first = logged()
    check_state(s)
    assert logged() == first                                           # nothing new the second time
    choices = [r["choice"] for r in first if r["agent"] == "invariants"]
    assert choices[0] == "escalate" and set(choices[1:]) == {"warning"}
    assert "reefer_block" in first[0]["reason"] and "rules-only" in first[0]["reason"]
    assert any("C0" in r["reason"] for r in first if r["choice"] == "warning")   # trucks parked on a departed ship


# ---------------------------------------------------------------- the simulator's own record

def test_new_simulator_violations_and_refusals_are_breaches(tmp_path):
    log.start(tmp_path)
    sim = tmp_path / "actions.jsonl"
    lines = [{"action": "session_start"},
             {"action": "tool_call", "tool": "get_yard_status", "result": "(read)"},
             {"action": "VIOLATION", "type": "customs_held_container_released", "container_id": "CVLU3445284",
              "sim_time": "2026-09-01T07:00:00"},
             {"action": "tool_error", "tool": "plan_crane_sequence", "error": "QC3 is out of service",
              "sim_time": "2026-09-01T07:05:00"}]
    sim.write_text("".join(json.dumps(r) + "\n" for r in lines))
    found = check_state(world())
    assert [(b.rule, b.subject) for b in found] == [("simulator_violation", "customs_held_container_released"),
                                                    ("simulator_refusal", "plan_crane_sequence")]
    assert check_state(world()) == []                 # already read
    with sim.open("a") as f:
        f.write(json.dumps({"action": "VIOLATION", "type": "imdg_outside_block_H"}) + "\n")
        f.write('{"action": "VIOLATION", "type": "half a li')                    # incomplete line: wait
    assert [b.subject for b in check_state(world())] == ["imdg_outside_block_H"]


# ---------------------------------------------------------------- independence from the guard

def test_monitor_is_written_independently_of_the_guard():
    import ast
    tree = ast.parse((ROOT / "core" / "invariants.py").read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m == "guard" or m.startswith("guard.") for m in imported)


# ---------------------------------------------------------------- real simulator

def _api(scenario, run_dir):
    from vendor import TerminalAPI
    run_dir.mkdir(parents=True, exist_ok=True)
    return TerminalAPI(scenario, data_xlsx=str(WORKBOOK), log_path=str(run_dir / "actions.jsonl"))


@needs_workbook
@pytest.mark.parametrize("scenario", [f"practice_0{i}" for i in range(1, 6)])
def test_zero_breaches_on_the_practice_scenarios(scenario, tmp_path):
    try:
        import run
        agents = run.make_agents("multi")
    except ImportError as e:                           # run.py itself is broken: other tests report it
        pytest.skip(f"run.py can't be imported: {e}")
    from core.loop import Loop
    d = tmp_path / scenario
    summary = Loop(_api(scenario, d), d, agents=agents).run()
    assert summary["loop"]["breaches"] == 0 and summary["loop"]["rules_only_ticks"] == 0
    assert summary["violations"] == []
    recs = [json.loads(l) for l in (d / "decisions.jsonl").read_text().splitlines()]
    assert not [r for r in recs if r["agent"] == "invariants" and r["choice"] in ("CRITICAL", "escalate")]


class EvilAgent(SkeletonAgent):
    """The rules-only planner, plus two planted rule breaks the first time it can:
    a reefer box into block A, and QC3 moved past QC1/QC2 onto bay 2 of B1."""
    name = "evil"

    def __init__(self):
        super().__init__()
        self.fired_tick_t = None

    def propose(self, state, events):
        acts, reqs = super().propose(state, events)
        if self.fired_tick_t is not None:
            return acts, reqs
        qc3 = state.crane("QC3")
        target = next((s for s in state.ships if s.berth == "B1" and s.status in ("berthed", "working")), None)
        if target is None or qc3 is None or qc3.status != "available" or any("QC3" in s.cranes for s in state.ships):
            return acts, reqs
        reefer = next((c.container_id for c in state.manifests.get(target.call_id, ())
                       if c.cargo_type == "reefer" and c.direction == "import" and not c.yard_block), None)
        if reefer is None:
            return acts, reqs
        self.fired_tick_t = state.now
        keep = []
        for a in acts:                               # don't let the skeleton place that reefer legally
            if a.tool == "place_container" and reefer in a.args["container_ids"]:
                ids = [c for c in a.args["container_ids"] if c != reefer]
                if ids:
                    keep.append(Action(a.tool, {**a.args, "container_ids": ids}, a.agent, a.reason))
            elif a.tool == "plan_crane_sequence" and a.args["call_id"] == target.call_id:
                continue
            else:
                keep.append(a)
        planted = [Action("place_container", {"container_ids": [reefer], "block_id": "A"}, "evil",
                          "planted breach: reefer box in a standard block"),
                   Action("plan_crane_sequence", {"call_id": target.call_id, "crane_ids": ["QC3"],
                                                  "bay_positions": [2]}, "evil",
                          "planted breach: QC3 moved past QC1 and QC2")]
        return planted + keep, reqs                  # first, before the skeleton fills block A


@needs_workbook
def test_planted_breaches_are_caught_in_the_same_tick(tmp_path, monkeypatch):
    import guard
    from core.loop import Loop
    monkeypatch.setattr(guard, "check", lambda action, state: (True, "guard switched off for this test"))
    d = tmp_path / "practice_01"
    evil = EvilAgent()
    summary = Loop(_api("practice_01", d), d, agents=[evil]).run()

    assert evil.fired_tick_t is not None, "the planted breach never happened"
    recs = [json.loads(l) for l in (d / "decisions.jsonl").read_text().splitlines()]
    critical = [r for r in recs if r["agent"] == "invariants" and r["choice"] == "CRITICAL"]
    reasons = " ".join(r["reason"] for r in critical)
    assert "[reefer_block]" in reasons                 # the simulator never flags this one: only we do
    assert "[rail_order]" in reasons
    assert "[simulator_violation] cranes_out_of_rail_order" in reasons
    planted = [r for r in recs if r["agent"] == "evil"]
    assert critical[0]["tick"] == planted[0]["tick"]   # caught at the end of the same tick
    assert summary["loop"]["rules_only_ticks"] >= 1    # and the next tick ran rules-only
    assert sum("[reefer_block]" in r["reason"] for r in critical) == 1   # reported once, not every tick
    assert any(r["choice"] == "escalate" and r["agent"] == "invariants" for r in recs)
