"""Collect real decision points for the LLM-3 bake-off.

    python -m llm.bakeoff.collect                 # -> llm/bakeoff/data/decision_points.jsonl

Runs the 5 practice scenarios with the current agents, rules only (no LLM), and records the
moments where the terminal faces a genuine choice, with the options our code would offer the
supervisor (LLM-4) and a template explanation for each:

  berth      a berth is free and two or more arrived ships fit it: which one goes first?
  admission  the yard can't take every ship (YARD-3 model): keep first-come-first-served,
             or write one ship off so more ships finish?
  stall      a ship at berth can't discharge (no yard slot for its next boxes) while another
             ship at berth has work: keep the cranes, or move them to the ship that can work?

Option A is always the safest reasonable default (first-come-first-served). "rules_choice" is
what our deterministic rules would pick; agreement with it is reported, not required.
Deterministic: same code + same workbook -> same file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents import admission  # noqa: E402
from core.loop import Loop  # noqa: E402
from core.state import build_state  # noqa: E402
from core.types import SHIP_ARRIVED, SHIP_BERTHED, SHIP_WORKING  # noqa: E402
from llm import client as llm_client  # noqa: E402
from llm.supervisor import summary  # noqa: E402,F401  (the supervisor's own terminal summary)

PRACTICE = [f"practice_0{i}" for i in range(1, 6)]
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "decision_points.jsonl"   # git-ignored: workbook data
TARGET = 50
RATE = admission.CRANE_RATE * admission.DEFAULT_CRANES       # moves/h for one berthed ship


def _unplaced(state, call_id) -> int:
    return sum(1 for c in state.manifests.get(call_id, ()) if c.direction == "import" and not c.yard_block)


def berth_point(state) -> dict | None:
    busy = {s.berth for s in state.ships if s.berth and s.status in (SHIP_BERTHED, SHIP_WORKING, "sailing")}
    waiting = sorted((s for s in state.ships if s.status == SHIP_ARRIVED and s.actual_arrival),
                     key=lambda s: (s.actual_arrival, s.call_id))
    for b in state.berths:
        if b.berth_id in busy:
            continue
        fit = [s for s in waiting if b.fits(s.loa_m, s.draft_m)]
        if len(fit) < 2:
            continue
        opts = []
        for oid, s in zip("ABC", fit[:3]):
            waited = (state.now - s.actual_arrival).total_seconds() / 3600
            work = s.remaining_moves / RATE + admission.WORK_DELAY_H + admission.SAIL_DELAY_H
            slots = _unplaced(state, s.call_id)
            opts.append({"id": oid, "label": f"berth {s.call_id} on {b.berth_id}",
                         "predicted": {"ship": s.call_id, "waited_h": round(waited, 1), "work_h": round(work, 1),
                                       "yard_slots_needed": slots},
                         "template": (f"Berth {s.call_id} on {b.berth_id}: it has waited {waited:.1f} h and needs "
                                      f"about {work:.0f} h of crane work and {slots} yard slots.")})
        return {"kind": "berth", "options": opts, "rules_choice": "A",
                "question": f"Berth {b.berth_id} is free. Which waiting ship should take it?"}
    return None


def admission_point(state) -> dict | None:
    b = admission.bound(state)
    if not b.shortage:
        return None
    horizon = (state.scenario_end - state.now).total_seconds() / 3600
    cal = admission._calibration(state)
    base = frozenset(b.impossible)
    fcfs, _ = admission._simulate(state, base, horizon, cal)
    live = [c for c in b.can_finish + b.cannot_finish + b.written_off]
    cands = []
    for cid in live:
        done, _ = admission._simulate(state, base | {cid}, horizon, cal)
        cands.append((-len(done), cid, len(done)))
    cands.sort()
    opts = [{"id": "A", "label": "keep first-come-first-served yard slots",
             "predicted": {"ships_finishing_in_window": len(fcfs)},
             "template": f"Keep first-come-first-served: {len(fcfs)} ships are predicted to finish in the window."}]
    for oid, (_, cid, n) in zip("BC", cands[:2]):
        opts.append({"id": oid, "label": f"give {cid} no more yard slots",
                     "predicted": {"ships_finishing_in_window": n, "ship_written_off": cid},
                     "template": f"Give {cid} no more yard slots: {n} ships are predicted to finish in the window."})
    best = max(opts, key=lambda o: (o["predicted"]["ships_finishing_in_window"], o["id"] == "A"))
    return {"kind": "admission", "options": opts, "rules_choice": best["id"],
            "question": "The yard can't take every ship's boxes. How should the scarce slots be used?"}


def stall_point(state) -> dict | None:
    """A ship at berth with cranes but nothing it can discharge, while another berthed ship can work."""
    first = admission._mem.first_discharge
    def workable(s):
        placed = sum(1 for c in state.manifests.get(s.call_id, ()) if c.direction == "import" and c.yard_block)
        done = max(0, first.get(s.call_id, s.discharge_remaining) - s.discharge_remaining)
        return (placed - done) > 0 or (s.discharge_remaining == 0 and s.load_remaining > 0)
    at = [s for s in state.ships if s.status in (SHIP_BERTHED, SHIP_WORKING) and s.berth]
    stalled = [s for s in at if s.cranes and s.discharge_remaining > 0 and not workable(s)]
    busy = [s for s in at if workable(s)]
    if not stalled or not busy:
        return None
    s, o = stalled[0], max(busy, key=lambda x: x.remaining_moves)
    n = len(s.cranes)
    opts = [{"id": "A", "label": f"keep {n} cranes on {s.call_id}",
             "predicted": {"ship": s.call_id, "moves_per_h_now": 0, "imports_waiting_for_yard": _unplaced(state, s.call_id)},
             "template": (f"Keep the cranes on {s.call_id}: it is waiting for yard space for "
                          f"{_unplaced(state, s.call_id)} boxes and makes no moves until slots free up.")},
            {"id": "B", "label": f"move {n} cranes from {s.call_id} to {o.call_id}",
             "predicted": {"ship": o.call_id, "extra_moves_per_h": round(admission.CRANE_RATE * n), "moves_left": o.remaining_moves},
             "template": (f"Move {n} cranes to {o.call_id} (berth {o.berth}): about {admission.CRANE_RATE * n:.0f} more "
                          f"moves per hour on a ship with {o.remaining_moves} moves left.")}]
    return {"kind": "stall", "options": opts, "rules_choice": "B",
            "question": f"{s.call_id} can't discharge until yard slots free up. What should its cranes do?"}


def collect(data: Path, scenarios=PRACTICE) -> list[dict]:
    from run import make_agents
    from vendor import TerminalAPI
    import tempfile
    llm_client.configure(enabled=False)
    found = []
    for sc in scenarios:
        admission.reset()
        with tempfile.TemporaryDirectory() as tmp:
            api = TerminalAPI(sc, data_xlsx=str(data), log_path=str(Path(tmp) / "actions.jsonl"))
            loop = Loop(api, Path(tmp), agents=make_agents("multi"))
            last = {}
            while True:
                st = build_state(api)                     # the terminal as the agents see it, before they act
                for make in (berth_point, admission_point, stall_point):
                    p = make(st)
                    if not p:
                        continue
                    key = (p["kind"], tuple(o["label"] for o in p["options"]),
                           tuple(o["predicted"].get("ships_finishing_in_window") for o in p["options"]))
                    if last.get(p["kind"]) == key:
                        continue                                  # same choice as last tick: skip
                    last[p["kind"]] = key
                    p.update({"scenario": sc, "time": st.now.isoformat(), "state": summary(st)})
                    found.append(p)
                loop.step()
                st = build_state(api)
                if st.now >= st.scenario_end:
                    break
                api.call("advance_time", {"minutes": loop._tick_minutes(st)})
                loop.tick += 1
    return found


def _informative(p: dict) -> bool:
    """A choice where the options' predicted outcomes differ (not a tie)."""
    if p["kind"] != "admission":
        return True
    return len({o["predicted"].get("ships_finishing_in_window") for o in p["options"]}) > 1


def select(points: list[dict], n: int = TARGET) -> list[dict]:
    """Informative choices first, spread evenly over time; ties only to fill up to n. Stable ids."""
    def spread(xs, k):
        if k <= 0:
            return []
        if len(xs) <= k:
            return list(xs)
        step = len(xs) / k
        return [xs[int(i * step)] for i in range(k)]
    good = [p for p in points if _informative(p)]
    ties = [p for p in points if not _informative(p)]
    chosen = spread(good, n)
    chosen += spread(ties, n - len(chosen))
    chosen.sort(key=lambda p: (p["scenario"], p["time"], p["kind"]))
    for i, p in enumerate(chosen, 1):
        p["id"] = f"dp{i:03d}"
    return chosen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "P2_Container_Terminal_Data.xlsx")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("-n", type=int, default=TARGET)
    a = p.parse_args(argv)
    pts = select(collect(a.data), a.n)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pts), encoding="utf-8")
    kinds = {}
    for x in pts:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    print(f"{len(pts)} decision points -> {a.out} ({', '.join(f'{k}: {v}' for k, v in sorted(kinds.items()))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
