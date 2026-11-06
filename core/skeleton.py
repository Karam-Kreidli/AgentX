"""Walking skeleton (CORE-2) and the fallback rules-only mode.

A deliberately simple, rule-safe loop:
  each tick (15 simulated minutes):
    1. read schedule, equipment and yard
    2. berth arrived ships (oldest arrival first) on the first free berth they fit (B1, B2, B3)
    3. give each berthed ship its berth's fixed cranes, skipping broken ones
    4. give it 3 trucks per crane
    5. place its import boxes in legal blocks (IMDG -> H, reefer -> I/J, dry -> A-F)
    6. advance the clock

It is replaced piece by piece by CORE-3/4 and the agents, and stays as the
rules-only fallback. It talks to the simulator only through `api.call`, so tool
errors come back as data and never raise. It reads only the toolkit's read
tools and static reference data: no workbook, no future information.

Why the fixed crane split can never break rail order or spacing
----------------------------------------------------------------
Each crane only ever sits at its home bay (where the simulator starts it) or at
its fixed working bay below. For every pair of rail neighbours, every possible
position of the left crane is below every possible position of the right one:

    crane   home  work   berth (bays)
    QC1      1     5     B1 (1-30)
    QC2     14    20     B1
    QC3     27    35     B2 (31-58)
    QC4     40    50     B2
    QC5     53    62     B3 (59-80)
    QC6     66    75     B3

So whatever order the moves happen in, and whichever cranes are broken (a broken
crane is skipped and simply stays where it is), all six stay in strict rail
order. Working positions are at least 15 bays apart, so spacing always holds.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from core.types import (BLOCK_EMPTY, BLOCK_IMDG, BLOCK_REEFER, BLOCK_STANDARD, CARGO_DRY,
                        CARGO_IMDG, CARGO_REEFER, SHIP_ARRIVED, SHIP_BERTHED, SHIP_WORKING,
                        TOOL_ASSIGN_BERTH, TOOL_DISPATCH_TRUCKS, TOOL_PLACE, TOOL_PLAN_CRANES,
                        Berth, Decision)

AGENT = "skeleton"
TICK_MINUTES = 15
TRUCKS_PER_CRANE = 3          # the simulator reaches full crane speed at 3 trucks per crane

# Static reference data (terminal facts, the same in every scenario; never scenario sheets).
BERTHS: tuple[Berth, ...] = (
    Berth("B1", 320.0, 14.5, 1, 30),
    Berth("B2", 280.0, 13.5, 31, 58),
    Berth("B3", 220.0, 12.0, 59, 80),
)

# Legal block types per import cargo, in order of preference ("any legal block").
# Hard rules: IMDG only in H and H only IMDG; reefers only in I/J. Dry boxes are
# legal anywhere except H: standard blocks first, then G (meant for empties), then
# the reefer blocks I/J as a last resort. Trade-off for YARD-1/YARD-3: dry boxes in
# I/J can leave a later reefer with nowhere to go. On the practice data this never
# lowered ships departed and got practice_01 from 3/4 to 4/4.
LEGAL_BLOCK_TYPES: dict[str, tuple[str, ...]] = {
    CARGO_IMDG: (BLOCK_IMDG,),
    CARGO_REEFER: (BLOCK_REEFER,),
    CARGO_DRY: (BLOCK_STANDARD, BLOCK_EMPTY, BLOCK_REEFER),
}

# berth -> ((crane, working bay), ...) in rail order. See the module docstring.
CRANE_SPLIT: dict[str, tuple[tuple[str, int], ...]] = {
    "B1": (("QC1", 5), ("QC2", 20)),
    "B2": (("QC3", 35), ("QC4", 50)),
    "B3": (("QC5", 62), ("QC6", 75)),
}

OUT_OF_SERVICE = "out of service"


def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class Skeleton:
    def __init__(self, api: Any, run_dir: Path):
        self.api = api
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.run_dir / "decisions.jsonl"
        self.decisions_path.write_text("", encoding="utf-8")
        self.tick = 0
        self.start: datetime | None = None
        self.pending: dict[str, dict[str, list[str]]] = {}   # call -> cargo -> boxes without a slot
        self.flagged: set[str] = set()          # things already escalated once
        self.errors = 0

    # ------------------------------------------------------------ logging
    def _hours(self, now: datetime) -> float:
        return round((now - self.start).total_seconds() / 3600, 3) if self.start else 0.0

    def _log(self, choice: str, reason: str, saw: dict, result: dict | None, now: datetime) -> None:
        ok = not (isinstance(result, dict) and "error" in result)
        d = Decision(tick=self.tick, t=self._hours(now), agent=AGENT, saw=saw, options=(),
                     choice=choice, reason=reason, guard_ok=True,
                     guard_reason="no guard yet (skeleton); rule-safe by construction",
                     tool_result=result if ok else {"error": result.get("error")}, backend="rules")
        with self.decisions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(d), ensure_ascii=False, default=str) + "\n")

    def _act(self, tool: str, args: dict, reason: str, saw: dict, now: datetime) -> dict:
        result = self.api.call(tool, args)
        self._log(tool, reason, {**saw, "args": args}, result, now)
        return result

    def _escalate(self, key: str, message: str, now: datetime) -> None:
        if key in self.flagged:
            return
        self.flagged.add(key)
        self._log("escalate", message, {"key": key}, None, now)

    # ------------------------------------------------------------ main loop
    def run(self) -> dict:
        while True:
            clock = self.api.call("get_current_time")
            now = _ts(clock["now"])
            if self.start is None:
                self.start = now
            if clock.get("finished"):
                break
            try:
                self._decide(now)
            except Exception as e:  # never crash: log and carry on to the next tick
                self.errors += 1
                self._log("exception", f"{type(e).__name__}: {e}",
                          {"trace": traceback.format_exc(limit=3)}, None, now)
            self.api.call("advance_time", {"minutes": TICK_MINUTES})
            self.tick += 1
        return self.summary()

    def _decide(self, now: datetime) -> None:
        ships = self.api.call("get_vessel_schedule")
        equipment = self.api.call("get_equipment_status")
        yard = self.api.call("get_yard_status")
        broken = {e["equipment_id"] for e in equipment if e["status"] == OUT_OF_SERVICE}

        self._berth_ships(ships, now)
        ships = self.api.call("get_vessel_schedule")     # re-read after berthing
        active = [s for s in ships if s["status"] in (SHIP_BERTHED, SHIP_WORKING)]
        for s in active:
            self._cranes(s, broken, now)
        equipment = self.api.call("get_equipment_status")
        self._trucks(self.api.call("get_vessel_schedule"), equipment, now)
        self._place_imports(active, yard, now)

    # ------------------------------------------------------------ berths
    def _berth_ships(self, ships: list[dict], now: datetime) -> None:
        occupied = {s["berth"] for s in ships if s["berth"] and s["status"] in (SHIP_BERTHED, SHIP_WORKING, "sailing")}
        waiting = sorted((s for s in ships if s["status"] == SHIP_ARRIVED), key=lambda s: s["actual_arrival"])
        for s in waiting:
            fits = [b for b in BERTHS if b.fits(s["loa_m"], s["draft_m"])]
            if not fits:
                self._escalate(f"nofit:{s['call_id']}", f"{s['call_id']} fits no berth "
                               f"(LOA {s['loa_m']} m, draft {s['draft_m']} m); it will wait.", now)
                continue
            free = [b for b in fits if b.berth_id not in occupied]
            if not free:
                continue
            b = free[0]
            res = self._act(TOOL_ASSIGN_BERTH, {"call_id": s["call_id"], "berth_id": b.berth_id},
                            f"{s['call_id']} has arrived and {b.berth_id} is the first free berth it fits.",
                            {"arrived": s["actual_arrival"], "loa_m": s["loa_m"], "draft_m": s["draft_m"]}, now)
            if "error" not in res:
                occupied.add(b.berth_id)

    # ------------------------------------------------------------ cranes
    def _cranes(self, s: dict, broken: set[str], now: datetime) -> None:
        split = CRANE_SPLIT.get(s["berth"], ())
        want = [(c, bay) for c, bay in split if c not in broken]
        if not want or sorted(s["cranes"]) == sorted(c for c, _ in want):
            return
        skipped = [c for c, _ in split if c in broken]
        reason = (f"{s['call_id']} on {s['berth']} gets that berth's fixed cranes "
                  f"{', '.join(c for c, _ in want)}" + (f" ({', '.join(skipped)} out of service, skipped)" if skipped else "") + ".")
        self._act(TOOL_PLAN_CRANES, {"call_id": s["call_id"], "crane_ids": [c for c, _ in want],
                                     "bay_positions": [bay for _, bay in want]},
                  reason, {"berth": s["berth"], "current": s["cranes"]}, now)

    # ------------------------------------------------------------ trucks
    def _trucks(self, ships: list[dict], equipment: list[dict], now: datetime) -> None:
        serving: dict[str, list[str]] = {}
        free: list[str] = []
        for e in equipment:
            if e["type"] != "yard truck":
                continue
            st = e["status"]
            if st == "available":
                free.append(e["equipment_id"])
            elif st.startswith("serving "):
                serving.setdefault(st[len("serving "):], []).append(e["equipment_id"])
        for s in ships:
            if s["status"] not in (SHIP_BERTHED, SHIP_WORKING):
                continue
            want = TRUCKS_PER_CRANE * len(s["cranes"])
            have = serving.get(s["call_id"], [])
            if len(have) >= want or not free:
                continue
            add = free[: want - len(have)]
            del free[: len(add)]
            self._act(TOOL_DISPATCH_TRUCKS, {"call_id": s["call_id"], "truck_ids": have + add},
                      f"{s['call_id']} has {len(s['cranes'])} cranes and {len(have)} trucks; "
                      f"adding {len(add)} to reach {TRUCKS_PER_CRANE} per crane where possible.",
                      {"cranes": s["cranes"], "trucks_before": len(have)}, now)

    # ------------------------------------------------------------ yard
    def _pending_imports(self, cid: str, now: datetime) -> dict[str, list[str]] | None:
        """Import boxes of a call still without a slot, by cargo type.

        The manifest of a call never changes, and only we place boxes, so it is
        read once and then kept up to date from our own successful placements.
        """
        if cid not in self.pending:
            manifest = self.api.call("get_manifest", {"call_id": cid})
            if isinstance(manifest, dict):          # tool error: try again next tick
                return None
            todo: dict[str, list[str]] = {}
            for m in manifest:
                if m["direction"] != "import" or m["yard_block"]:
                    continue
                if m["cargo_type"] not in LEGAL_BLOCK_TYPES:
                    self._escalate(f"cargo:{m['container_id']}",
                                   f"An import box of {cid} has an unknown cargo type; it is not placed.", now)
                    continue
                todo.setdefault(m["cargo_type"], []).append(m["container_id"])
            self.pending[cid] = todo
        return self.pending[cid]

    def _place_imports(self, active: list[dict], yard: list[dict], now: datetime) -> None:
        free = {b["block_id"]: b["slots"] - b["used"] for b in yard}
        btype = {b["block_id"]: b["block_type"] for b in yard}
        for s in active:
            cid = s["call_id"]
            if s["discharge_remaining"] <= 0:
                continue
            todo = self._pending_imports(cid, now)
            if not todo:
                continue
            for cargo, boxes in todo.items():
                prefs = LEGAL_BLOCK_TYPES[cargo]
                legal = sorted((b for b in free if btype[b] in prefs), key=lambda b: (prefs.index(btype[b]), b))
                for b in legal:
                    if not boxes:
                        break
                    n = min(free[b], len(boxes))
                    if n <= 0:
                        continue
                    batch = boxes[:n]
                    res = self._act(TOOL_PLACE, {"container_ids": batch, "block_id": b},
                                    f"{n} {cargo} import boxes of {cid} need a yard slot before discharge; "
                                    f"block {b} is a legal {btype[b]} block with {free[b]} free.",
                                    {"cargo": cargo, "free_before": free[b]}, now)
                    if "error" not in res:
                        free[b] -= n
                        del boxes[:n]

    # ------------------------------------------------------------ summary
    def summary(self) -> dict:
        s = self.api.run_summary()
        ships = self.api.call("get_vessel_schedule")
        end = _ts(self.api.call("get_current_time")["scenario_end"])
        honest = [((_ts(v["departed_at"]) or end) - _ts(v["actual_arrival"])).total_seconds() / 3600
                  for v in ships if v["actual_arrival"]]
        s["honest_turnaround_h"] = round(sum(honest) / len(honest), 2) if honest else None
        s["unfinished_calls"] = [v["call_id"] for v in ships if v["status"] != "departed"]
        s["skeleton_exceptions"] = self.errors
        (self.run_dir / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
        return s
