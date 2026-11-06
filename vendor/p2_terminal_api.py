"""
Problem 2 - Container Terminal Orchestra: mock terminal simulator.

    from p2_terminal_api import TerminalAPI
    api = TerminalAPI("practice_01", data_xlsx="data/P2_Container_Terminal_Data.xlsx")
    api.get_vessel_schedule(); api.advance_time(60) ...

Hidden scenarios: TerminalAPI("hidden_01", data_xlsx=..., key_xlsx="P2_Container_Terminal_ANSWER_KEY.xlsx")

Model (simplified but consistent):
- A vessel can be berthed once it has arrived. Work starts 1 h after berthing (mooring).
- Each assigned crane moves rated_moves_per_h x 0.85 x truck_factor boxes per hour,
  truck_factor = min(1, trucks assigned to the call / (3 x cranes)).
- Import boxes can only be discharged once they have a yard slot (place_container).
- When all moves are done the vessel sails 1 h later; berth, cranes and trucks are freed.
- Breakdowns and customs holds from the scenario appear at their start time.
Hard rules recorded as violations: cranes closer than 2 bays, IMDG outside block H,
customs-held box released, vessel placed on a berth it does not fit.
"""
import math
import numpy as np
import pandas as pd
from common import MockAPI, ClockMixin, tool, S, APIError, sheets

STEP = pd.Timedelta(minutes=5)


class TerminalAPI(ClockMixin, MockAPI):
    PROBLEM = "P2"

    def __init__(self, scenario, data_xlsx, key_xlsx=None, log_path=None):
        d = sheets(data_xlsx, "Berths", "Quay_Cranes", "Yard_Trucks", "Yard_Blocks", "Truck_Travel_Times")
        if scenario.startswith("hidden"):
            k = sheets(key_xlsx, "Hidden_Vessels", "Hidden_Events", "Hidden_Manifest", "Hidden_Baseline_FCFS")
            V, E, M, B = k["Hidden_Vessels"], k["Hidden_Events"], k["Hidden_Manifest"], k["Hidden_Baseline_FCFS"]
        else:
            k = sheets(data_xlsx, "Practice_Vessels", "Practice_Events", "Practice_Manifest", "Practice_Baseline_FCFS")
            V, E, M, B = k["Practice_Vessels"], k["Practice_Events"], k["Practice_Manifest"], k["Practice_Baseline_FCFS"]
        V, E, M, B = [x[x.scenario == scenario].drop(columns="scenario").reset_index(drop=True) for x in (V, E, M, B)]
        if V.empty:
            raise APIError(f"unknown scenario {scenario}")
        for c in ("scheduled_eta", "actual_arrival"):
            V[c] = pd.to_datetime(V[c])
        E["start"] = pd.to_datetime(E["start"]); E["end"] = pd.to_datetime(E["end"])
        start = V.scheduled_eta.min().floor("D")
        super().__init__(scenario, log_path, start, start + pd.Timedelta(days=5))
        self.berths = d["Berths"].set_index("berth_id"); self.cranes = d["Quay_Cranes"].set_index("crane_id")
        self.trucks = d["Yard_Trucks"].set_index("truck_id"); self.blocks = d["Yard_Blocks"].set_index("block_id")
        self.travel = d["Truck_Travel_Times"]; self.events = E; self.baseline = B
        self.calls = {r.call_id: dict(r._asdict(), state="expected", berth=None, berthed_at=None, work_from=None, departed_at=None,
                                      done_discharge=0.0, done_load=0.0, cranes=[], trucks=[]) for r in V.itertuples(index=False)}
        self.manifest = M.set_index("container_id")
        self.down, self.holds = {}, {}
        for r in E.itertuples():
            if r.event_type in ("crane_breakdown", "truck_breakdown"):
                self.down.setdefault(r.target, []).append((r.start, r.end))
            elif r.event_type == "customs_hold":
                self.holds.setdefault(r.target, []).append((r.start, r.end))
        self.placed_at, self.leave_at, self.n_placed = {}, {}, {}; self.delivered = 0
        self.imports = {c: list(g.index) for c, g in self.manifest[self.manifest.direction == "import"].groupby("call_id")}
        self.slot = {}          # container -> block
        self.yard_fill = {b: 0 for b in self.blocks.index}
        self.released = set()
        self.crane_pos = {c: int(self.cranes.loc[c, "home_bay"]) for c in self.cranes.index}
        self.crane_hours = 0.0; self.moves_done = 0.0; self.truck_idle_h = 0.0; self.over85 = 0

    # ------------------------------------------------------------ helpers
    def _broken(self, eq, t=None):
        t = t or self.now
        return any(s <= t < e for s, e in self.down.get(eq, ()))

    def _held(self, cid, t=None):
        t = t or self.now
        return any(s <= t and (pd.isna(e) or e > t) for s, e in self.holds.get(cid, ()))

    def _check_spacing(self):
        working = sorted([(self.crane_pos[c], c) for v in self.calls.values() if v["state"] in ("berthed", "working") for c in v["cranes"]])
        for (p1, c1), (p2, c2) in zip(working, working[1:]):
            if p2 - p1 < 2:
                self._violation("crane_spacing_under_2_bays", cranes=[c1, c2], bays=[p1, p2])

    def _step(self, until):
        t = self.now
        while t < until:
            t2 = min(t + STEP, until); h = (t2 - t).total_seconds() / 3600
            busy_trucks = set()
            for cid, v in self.calls.items():
                if v["state"] == "expected" and v["actual_arrival"] <= t2:
                    v["state"] = "arrived"
                if v["state"] in ("berthed", "working") and v["work_from"] <= t2:
                    v["state"] = "working"
                    v["cranes"] = [c for c in v["cranes"] if not self._broken(c, t2)]
                    v["trucks"] = [k for k in v["trucks"] if not self._broken(k, t2)]
                    busy_trucks |= set(v["trucks"])
                    n = len(v["cranes"])
                    if n:
                        tf = min(1.0, len(v["trucks"]) / (3 * n))
                        cap = sum(self.cranes.loc[c, "rated_moves_per_h"] for c in v["cranes"]) * 0.85 * tf * h
                        imp_placed = self.n_placed.get(cid, 0)
                        can_dis = max(0.0, min(v["discharge_moves"], imp_placed) - v["done_discharge"])
                        d = min(cap, can_dis); v["done_discharge"] += d; cap -= d; l = 0.0
                        if v["done_discharge"] >= v["discharge_moves"] - 1e-6:
                            l = min(cap, v["load_moves"] - v["done_load"]); v["done_load"] += l
                        self.moves_done += d + l
                        self.crane_hours += n * h
                    if v["done_discharge"] >= v["discharge_moves"] - 1e-6 and v["done_load"] >= v["load_moves"] - 1e-6:
                        v["state"] = "sailing"; v["departed_at"] = t2 + pd.Timedelta(hours=1)
                        v["cranes"], v["trucks"] = [], []
                if v["state"] == "sailing" and v["departed_at"] <= t2:
                    v["state"] = "departed"
            for c, lt in list(self.leave_at.items()):   # import boxes leave after their planned dwell unless held
                if lt <= t2 and not self._held(c, t2):
                    self.yard_fill[self.slot[c]] -= 1; del self.leave_at[c]; del self.placed_at[c]; self.delivered += 1
            avail = [k for k in self.trucks.index if not self._broken(k, t2)]
            if any(v["state"] == "working" for v in self.calls.values()):
                self.truck_idle_h += h * len([k for k in avail if k not in busy_trucks])
            t = t2

    # ------------------------------------------------------------ tools
    @tool("Vessel schedule: ETA, status and remaining moves for every call. Actual arrival is shown only once the ship has arrived.")
    def get_vessel_schedule(self):
        out = []
        for cid, v in self.calls.items():
            out.append(dict(call_id=cid, vessel=v["vessel_name"], loa_m=v["loa_m"], draft_m=v["draft_m"], scheduled_eta=v["scheduled_eta"],
                            actual_arrival=v["actual_arrival"] if v["state"] != "expected" else None, status=v["state"], berth=v["berth"],
                            cranes=v["cranes"], trucks=len(v["trucks"]), discharge_remaining=round(v["discharge_moves"] - v["done_discharge"]),
                            load_remaining=round(v["load_moves"] - v["done_load"]), departed_at=v["departed_at"] if v["state"] == "departed" else None))
        return out

    @tool("Berth an arrived vessel. Work starts 1 h later. The berth must be free, long enough (LOA + 20 m) and deep enough.",
          dict(call_id=S("string"), berth_id=S("string", "B1, B2 or B3")), ["call_id", "berth_id"], mutating=True)
    def assign_berth(self, call_id, berth_id):
        v = self.calls.get(call_id)
        if v is None or berth_id not in self.berths.index:
            raise APIError("unknown call or berth")
        if v["state"] != "arrived":
            raise APIError(f"vessel is '{v['state']}' - only arrived vessels can be berthed")
        if any(o["berth"] == berth_id and o["state"] in ("berthed", "working", "sailing") for o in self.calls.values()):
            raise APIError(f"{berth_id} is occupied")
        b = self.berths.loc[berth_id]
        if b.length_m < v["loa_m"] + 20 or b.max_draft_m < v["draft_m"]:
            raise APIError(f"{v['vessel_name']} (LOA {v['loa_m']} m, draft {v['draft_m']} m) does not fit {berth_id} ({b.length_m} m, {b.max_draft_m} m)")
        v.update(state="berthed", berth=berth_id, berthed_at=self.now, work_from=self.now + pd.Timedelta(hours=1))
        return dict(ok=True, work_starts=v["work_from"])

    @tool("Assign quay cranes to a berthed vessel and set their bay positions along the rail (cranes keep their rail order and must be at least 2 bays apart).",
          dict(call_id=S("string"), crane_ids=S("array", items=S("string")), bay_positions=S("array", "One bay number per crane, within the berth's bay range", items=S("integer"))),
          ["call_id", "crane_ids", "bay_positions"], mutating=True)
    def plan_crane_sequence(self, call_id, crane_ids, bay_positions):
        v = self.calls.get(call_id)
        if v is None or v["state"] not in ("berthed", "working"):
            raise APIError("vessel must be berthed")
        if len(crane_ids) != len(bay_positions):
            raise APIError("one bay position per crane")
        b = self.berths.loc[v["berth"]]
        for c, p in zip(crane_ids, bay_positions):
            if c not in self.cranes.index:
                raise APIError(f"unknown crane {c}")
            if self._broken(c):
                raise APIError(f"{c} is out of service")
            if any(c in o["cranes"] for k, o in self.calls.items() if k != call_id):
                raise APIError(f"{c} is working on another vessel")
            if not b.start_bay <= p <= b.end_bay:
                raise APIError(f"bay {p} is outside berth {v['berth']} ({b.start_bay}-{b.end_bay})")
        v["cranes"] = list(crane_ids)
        for c, p in zip(crane_ids, bay_positions):
            self.crane_pos[c] = int(p)
        order = sorted(self.crane_pos.items(), key=lambda x: x[0])
        if any(p2 <= p1 for (_, p1), (_, p2) in zip(order, order[1:])):
            self._violation("cranes_out_of_rail_order", positions=dict(order))
        self._check_spacing()
        return dict(ok=True, cranes=v["cranes"])

    @tool("Assign yard trucks to serve a vessel (about 3 trucks per crane gives full speed). Pass an empty list to release them.",
          dict(call_id=S("string"), truck_ids=S("array", items=S("string"))), ["call_id", "truck_ids"], mutating=True)
    def dispatch_truck(self, call_id, truck_ids):
        v = self.calls.get(call_id)
        if v is None:
            raise APIError("unknown call")
        for k in truck_ids:
            if k not in self.trucks.index:
                raise APIError(f"unknown truck {k}")
            if self._broken(k):
                raise APIError(f"{k} is out of service")
            if any(k in o["trucks"] for c, o in self.calls.items() if c != call_id):
                raise APIError(f"{k} already serves another vessel")
        v["trucks"] = list(truck_ids)
        return dict(ok=True, trucks=len(truck_ids))

    @tool("Give import containers a yard block (needed before they can be discharged). IMDG boxes may only go to block H; reefers need block I or J.",
          dict(container_ids=S("array", items=S("string")), block_id=S("string")), ["container_ids", "block_id"], mutating=True)
    # note: import boxes leave the yard automatically after planned_dwell_days unless customs holds them
    def place_container(self, container_ids, block_id):
        if block_id not in self.blocks.index:
            raise APIError("unknown block")
        blk = self.blocks.loc[block_id]; placed = []
        for c in container_ids:
            if c not in self.manifest.index:
                raise APIError(f"unknown container {c}")
            if self.yard_fill[block_id] >= blk.slots:
                raise APIError(f"block {block_id} is full")
            m = self.manifest.loc[c]
            if m.cargo_type == "IMDG" and block_id != "H":
                self._violation("imdg_outside_block_H", container_id=c, block_id=block_id)
            if m.cargo_type != "IMDG" and block_id == "H":
                raise APIError("block H is reserved for IMDG boxes")
            if c not in self.slot:
                self.yard_fill[block_id] += 1; self.placed_at[c] = self.now
                self.leave_at[c] = self.now + pd.Timedelta(days=int(m.planned_dwell_days)) + pd.Timedelta(hours=6)
                self.n_placed[m.call_id] = self.n_placed.get(m.call_id, 0) + 1
                if self.yard_fill[block_id] > 0.85 * blk.slots:
                    self.over85 += 1
            self.slot[c] = block_id; placed.append(c)
        return dict(ok=True, placed=len(placed), block_fill=f"{self.yard_fill[block_id]}/{blk.slots}")

    @tool("Customs holds currently known (optionally for one container).", dict(container_id=S("string")))
    def get_customs_status(self, container_id=None):
        e = self.events[(self.events.event_type == "customs_hold") & (self.events.start <= self.now)]
        if container_id:
            return dict(container_id=container_id, held=self._held(container_id))
        return [dict(container_id=r.target, hold_since=r.start, released=(pd.notna(r.end) and r.end <= self.now)) for r in e.itertuples()]

    @tool("Release an import container to the consignee's truck (it leaves the terminal). Never release a customs-held box.",
          dict(container_id=S("string")), ["container_id"], mutating=True)
    def release_to_consignee(self, container_id):
        if container_id not in self.placed_at:
            raise APIError("container is not in the yard")
        if self._held(container_id):
            self._violation("customs_held_container_released", container_id=container_id)
        self.released.add(container_id); self.yard_fill[self.slot[container_id]] -= 1; del self.placed_at[container_id]; self.leave_at.pop(container_id, None)
        return dict(ok=True)

    @tool("Status of cranes and trucks: available, working (on which call) or out of service.")
    def get_equipment_status(self):
        out = []
        for c in self.cranes.index:
            w = [k for k, v in self.calls.items() if c in v["cranes"]]
            out.append(dict(equipment_id=c, type="quay crane", status="out of service" if self._broken(c) else ("working " + w[0] if w else "available"), bay=self.crane_pos[c]))
        for k in self.trucks.index:
            w = [c for c, v in self.calls.items() if k in v["trucks"]]
            out.append(dict(equipment_id=k, type="yard truck", status="out of service" if self._broken(k) else ("serving " + w[0] if w else "available")))
        return out

    @tool("Yard occupancy by block.")
    def get_yard_status(self):
        return [dict(block_id=b, block_type=self.blocks.loc[b, "block_type"], used=self.yard_fill[b], slots=int(self.blocks.loc[b, "slots"])) for b in self.blocks.index]

    @tool("Containers of one vessel call (import and export), with type and whether they have a yard slot.", dict(call_id=S("string")), ["call_id"])
    def get_manifest(self, call_id):
        m = self.manifest[self.manifest.call_id == call_id].reset_index()
        m["yard_block"] = m.container_id.map(self.slot)
        return m[["container_id", "direction", "size_ft", "cargo_type", "imdg_class", "gross_weight_t", "yard_block"]]

    def run_summary(self):
        dep = [v for v in self.calls.values() if v["state"] == "departed"]
        ta = [(v["departed_at"] - v["actual_arrival"]).total_seconds() / 3600 for v in dep]
        base = self.baseline[self.baseline.call_id.isin([c for c, v in self.calls.items() if v["state"] == "departed"])]
        return dict(super().run_summary(), vessels_departed=len(dep), vessels_total=len(self.calls),
                    avg_turnaround_h=round(float(np.mean(ta)), 2) if ta else None,
                    fcfs_baseline_avg_turnaround_h_same_calls=round(float(base.turnaround_h.mean()), 2) if len(base) else None,
                    crane_moves_per_hour=round(self.moves_done / self.crane_hours, 1) if self.crane_hours else None,
                    truck_idle_hours=round(self.truck_idle_h, 1), placements_into_blocks_over_85pct=self.over85, import_boxes_delivered=self.delivered)
