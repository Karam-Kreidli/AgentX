"""Scenario variants (CORE-7): workbooks made from the practice data, to catch overfitting.

    python tools/perturb.py --n 20 --seed 1            # 20 variant books + 4 malformed books + the yard book
    python tools/perturb.py --n 20 --malformed 4 --seed 1
    python tools/perturb.py --n 3 --malformed 0 --families shifted,holds
    python tools/perturb.py --ci                       # the fixed CI set (seed 7) in runs/check/variants/
    python tools/perturb.py --limits                   # toolkit-limit set in data/variants/limits/
    python tools/perturb.py --hidden-key               # also MOCK_ANSWER_KEY.xlsx (Hidden_* sheets, hidden_01..05)

Each variant workbook holds all five scenario names (practice_01..05), so tools/eval_all.py
--variants N and `run.py --data data/variants/v01.xlsx --scenario practice_02` load it through the
toolkit's normal path. The five static sheets (Berths, Quay_Cranes, Yard_Trucks, Yard_Blocks,
Truck_Travel_Times) are copied unchanged (agent code has berths, the crane split and truck travel
times built in), except in the one yard_capacity book. The four Practice_* sheets are rewritten per
scenario by one family each. Practice_Baseline_FCFS is written empty: the workbook's FCFS numbers
only match the original data (corePlan D2).

Files: v01.xlsx ... (perturbed and no-shortage), m01.xlsx ... (malformed), m_yard.xlsx (yard capacity),
index.json (what was done to every scenario and what must hold afterwards: "expect").

Moves versus boxes: every family keeps (discharge_moves - import rows) and (load_moves - export rows)
of each ship as in the source, so an "impossible manifest" in the practice data (3,100 moves for
551 boxes) stays exactly that, and a normal ship stays normal.

This is a test tool. It may read IDs and the Practice_* sheets; agent code never reads data/variants/.
The source workbook is never modified (its SHA-256 is checked before and after).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
DEFAULT_OUT = ROOT / "data" / "variants"
CI_OUT = ROOT / "runs" / "check" / "variants"
SCENARIOS = [f"practice_0{i}" for i in range(1, 6)]
STATIC_SHEETS = ["Berths", "Quay_Cranes", "Yard_Trucks", "Yard_Blocks", "Truck_Travel_Times"]
PRACTICE_SHEETS = {"V": "Practice_Vessels", "E": "Practice_Events", "M": "Practice_Manifest",
                   "B": "Practice_Baseline_FCFS"}
HIDDEN_SHEETS = {"V": "Hidden_Vessels", "E": "Hidden_Events", "M": "Hidden_Manifest", "B": "Hidden_Baseline_FCFS"}
WINDOW_DAYS = 5          # the toolkit: start = floor(min scheduled_eta, day), end = start + 5 days
EDGE = pd.Timedelta(hours=12)

# Static terminal facts the generator relies on. build() checks them against the source workbook.
CRANE_IDS = [f"QC{i}" for i in range(1, 7)]
TRUCK_IDS = [f"YT{i:02d}" for i in range(1, 21)]
BERTH_IDS = ["B1", "B2", "B3"]
BERTHS = {"B1": (320.0, 14.5), "B2": (280.0, 13.5), "B3": (220.0, 12.0)}      # length m, max draft m
GROUP_CAPACITY = {"dry": 6 * 220 + 200, "reefer": 400, "IMDG": 120}          # legal slots per cargo group
KNOWN_CARGO = ("dry", "reefer", "IMDG")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@dataclass
class Frames:
    """One scenario's four practice tables (scenario column dropped, as the toolkit does)."""
    V: pd.DataFrame
    E: pd.DataFrame
    M: pd.DataFrame
    B: pd.DataFrame

    def copy(self) -> "Frames":
        return Frames(self.V.copy(), self.E.copy(), self.M.copy(), self.B.copy())


@dataclass
class Source:
    static: dict                      # sheet name -> DataFrame (copied unchanged)
    sheets: dict                      # "V"/"E"/"M"/"B" -> full practice sheet (with scenario column)
    sha256: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(data: Path) -> Source:
    """Read the five static sheets and the four Practice_* sheets once."""
    static = {s: pd.read_excel(data, sheet_name=s) for s in STATIC_SHEETS}
    sheets = {k: pd.read_excel(data, sheet_name=name) for k, name in PRACTICE_SHEETS.items()}
    return Source(static, sheets, sha256(data))


def split(src: Source) -> dict[str, Frames]:
    """{scenario: Frames}: each sheet filtered by scenario, the scenario column dropped."""
    out = {}
    for sc in SCENARIOS:
        parts = {k: df[df["scenario"] == sc].drop(columns="scenario").reset_index(drop=True)
                 for k, df in src.sheets.items()}
        out[sc] = Frames(**parts)
    return out


def check_static(static: dict) -> None:
    """The generator's built-in terminal facts must match the workbook (else the variants are meaningless)."""
    b = static["Berths"].set_index("berth_id")
    got_b = {i: (float(r.length_m), float(r.max_draft_m)) for i, r in b.iterrows()}
    y = static["Yard_Blocks"]
    cap = {"dry": int(y[y.block_type == "standard"].slots.sum() + y[y.block_type == "empty"].slots.sum()),
           "reefer": int(y[y.block_type == "reefer"].slots.sum()), "IMDG": int(y[y.block_type == "IMDG"].slots.sum())}
    problems = []
    if got_b != BERTHS:
        problems.append(f"berths {got_b} != {BERTHS}")
    if cap != GROUP_CAPACITY:
        problems.append(f"yard capacity {cap} != {GROUP_CAPACITY}")
    if list(static["Quay_Cranes"].crane_id) != CRANE_IDS:
        problems.append("crane ids differ")
    if list(static["Yard_Trucks"].truck_id) != TRUCK_IDS:
        problems.append("truck ids differ")
    if problems:
        raise SystemExit("the workbook's static data differs from what perturb.py assumes: " + "; ".join(problems))


def window(f: Frames) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The toolkit's scenario window."""
    start = pd.to_datetime(f.V["scheduled_eta"]).min().floor("D")
    return start, start + pd.Timedelta(days=WINDOW_DAYS)


def imports(f: Frames) -> pd.DataFrame:
    return f.M[f.M["direction"] == "import"]


def _r15(t: pd.Timestamp) -> pd.Timestamp:
    return t.round("15min")


def _clamp(t: pd.Timestamp, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Timestamp:
    return min(max(t, lo), hi)


def _fresh_ids(rng: np.random.Generator, used: set, n: int) -> list[str]:
    """n container ids in the workbook's format (3 letters + U + 7 digits) that nobody uses yet."""
    out = []
    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    while len(out) < n:
        cid = "".join(rng.choice(letters, 3)) + "U" + "".join(str(d) for d in rng.integers(0, 10, 7))
        if cid not in used:
            used.add(cid)
            out.append(cid)
    return out


def _event_rows(rows: list[dict], like: pd.DataFrame) -> pd.DataFrame:
    """New event rows with the dtypes of the source events table."""
    df = pd.DataFrame(rows, columns=list(like.columns))
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c])
    return df.astype(like.dtypes.to_dict(), errors="ignore") if len(df) else like.iloc[0:0].copy()


def _add_events(f: Frames, rows: list[dict]) -> None:
    if not rows:
        return
    new = _event_rows(rows, f.E)
    f.E = pd.concat([f.E, new], ignore_index=True).sort_values("start", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# Perturbed families. Each is a pure function (frames, rng) -> (frames, meta); never in place.
# meta = {"params": {...}, "expect": {...}, "shortage": {...}}
# --------------------------------------------------------------------------

def _meta(params=None, expect=None, shortage=None) -> dict:
    return {"params": params or {}, "expect": expect or {}, "shortage": shortage or {}}


def fam_identity(f: Frames, rng: np.random.Generator) -> tuple[Frames, dict]:
    """Nothing changes. Tests the writer and the toolkit path."""
    return f.copy(), _meta()


def _shift(f: Frames, rng: np.random.Generator, k: float) -> tuple[Frames, dict]:
    """Ships arrive earlier or later than the data says. The earliest scheduled ETA never moves,
    so the scenario window stays where it is. k = strength (1.0 full)."""
    out = f.copy()
    start, end = window(out)
    hi = end - EDGE
    first = out.V["scheduled_eta"].idxmin()
    shifts = {}
    for idx in out.V.index:
        d_act = float(rng.uniform(-3, 8)) * k
        d_eta = float(rng.uniform(-2, 2)) * k
        act = _clamp(_r15(out.V.at[idx, "actual_arrival"] + pd.Timedelta(hours=d_act)), start, hi)
        out.V.at[idx, "actual_arrival"] = act
        if idx != first:
            out.V.at[idx, "scheduled_eta"] = _clamp(_r15(out.V.at[idx, "scheduled_eta"] + pd.Timedelta(hours=d_eta)), start, hi)
        shifts[out.V.at[idx, "call_id"]] = round(d_act, 2)
    return out, _meta({"arrival_shift_h": shifts})


def fam_shifted(f, rng):
    return _shift(f, rng, 1.0)


def fam_bunched(f: Frames, rng: np.random.Generator) -> tuple[Frames, dict]:
    """Arrivals pulled toward the first one: more ships wait for a berth at the same time."""
    out = f.copy()
    start, end = window(out)
    hi = end - EDGE
    t0 = out.V["scheduled_eta"].min()
    factor = float(rng.uniform(0.4, 0.7))
    for idx in out.V.index:
        out.V.at[idx, "scheduled_eta"] = _r15(t0 + (out.V.at[idx, "scheduled_eta"] - t0) * factor)
        out.V.at[idx, "actual_arrival"] = _clamp(_r15(t0 + (out.V.at[idx, "actual_arrival"] - t0) * factor), start, hi)
    return out, _meta({"factor": round(factor, 3)})


def _breakdowns(f: Frames, rng: np.random.Generator, k: float) -> tuple[Frames, dict]:
    """Extra crane and truck breakdowns; one pair of rail neighbours is down at the same time."""
    out = f.copy()
    start, end = window(out)
    n_crane = max(1, round(int(rng.integers(2, 5)) * k))
    n_truck = max(1, round(int(rng.integers(3, 7)) * k))
    rows, made = [], []

    def crane(cid, t, hours):
        e = min(t + pd.Timedelta(hours=hours), end)
        rows.append({"event_type": "crane_breakdown", "target": cid, "berth_id": str(rng.choice(BERTH_IDS)),
                     "start": t, "end": e, "detail": "Crane out of service"})
        made.append([cid, str(t), str(e)])

    span = (end - pd.Timedelta(hours=30) - start).total_seconds() / 3600
    if n_crane >= 2:
        j = int(rng.integers(0, len(CRANE_IDS) - 1))
        t = (start + pd.Timedelta(hours=float(rng.uniform(2, span)))).round("ms")
        d1 = float(rng.uniform(1, 8))
        crane(CRANE_IDS[j], t, d1)
        crane(CRANE_IDS[j + 1], (t + pd.Timedelta(hours=float(rng.uniform(0, d1 * 0.5)))).round("ms"), float(rng.uniform(1, 8)))
    for _ in range(n_crane - (2 if n_crane >= 2 else 0)):
        crane(str(rng.choice(CRANE_IDS)), (start + pd.Timedelta(hours=float(rng.uniform(2, span)))).round("ms"),
              float(rng.uniform(1, 8)))
    for _ in range(n_truck):
        t = (start + pd.Timedelta(hours=float(rng.uniform(2, span)))).round("ms")
        e = min(t + pd.Timedelta(hours=float(rng.uniform(2, 16))), end)
        rows.append({"event_type": "truck_breakdown", "target": str(rng.choice(TRUCK_IDS)), "berth_id": np.nan,
                     "start": t, "end": e, "detail": "Yard truck unavailable"})
        made.append([rows[-1]["target"], str(t), str(e)])
    _add_events(out, rows)
    return out, _meta({"extra_cranes": n_crane, "extra_trucks": n_truck, "events": made})


def fam_breakdowns(f, rng):
    return _breakdowns(f, rng, 1.0)


def _holds(f: Frames, rng: np.random.Generator, k: float) -> tuple[Frames, dict]:
    """Extra customs holds on import boxes; about half are never released inside the window."""
    out = f.copy()
    start, end = window(out)
    imp = imports(out)
    held = set(out.E.loc[out.E["event_type"] == "customs_hold", "target"])
    pool = imp[~imp["container_id"].isin(held)]
    n = min(max(1, round(int(rng.integers(10, 31)) * k)), len(pool))
    pick = pool.iloc[rng.choice(len(pool), n, replace=False)]
    span = (end - pd.Timedelta(hours=12) - start).total_seconds() / 3600
    rows = []
    for r in pick.itertuples():
        t = (start + pd.Timedelta(hours=float(rng.uniform(0, span)))).round("ms")
        e = min(t + pd.Timedelta(hours=float(rng.uniform(6, 72))), end) if rng.random() < 0.5 else pd.NaT
        rows.append({"event_type": "customs_hold", "target": r.container_id, "berth_id": np.nan, "start": t, "end": e,
                     "detail": f"Hold on import box from {r.call_id}"})
    _add_events(out, rows)
    return out, _meta({"extra_holds": n, "never_released": sum(1 for r in rows if pd.isna(r["end"]))})


def fam_holds(f, rng):
    return _holds(f, rng, 1.0)


def fam_mixed(f: Frames, rng: np.random.Generator) -> tuple[Frames, dict]:
    """Shifted arrivals, extra breakdowns and extra holds, each at half strength."""
    a, m1 = _shift(f, rng, 0.5)
    b, m2 = _breakdowns(a, rng, 0.5)
    c, m3 = _holds(b, rng, 0.5)
    return c, _meta({**m1["params"], **m2["params"], **m3["params"]})


def fam_renamed(f: Frames, rng: np.random.Generator) -> tuple[Frames, dict]:
    """New call ids, container ids and vessel names, in the same sort order: nothing else changes.
    Any KPI difference from the original scenario means some code reads the content of an ID."""
    out = f.copy()
    cmap = {c: f"K{i + 1:03d}" for i, c in enumerate(sorted(out.V["call_id"]))}
    cids = sorted(set(out.M["container_id"]) | set(out.E.loc[out.E["event_type"] == "customs_hold", "target"]))
    bmap = {c: f"ZQXU{i:07d}" for i, c in enumerate(cids)}
    order = {c: i for i, c in enumerate(sorted(cmap))}
    out.V["vessel_name"] = [f"MV Test {order[c] + 1:02d}" for c in out.V["call_id"]]
    out.V["call_id"] = out.V["call_id"].map(cmap)
    out.M["call_id"] = out.M["call_id"].map(cmap)
    out.M["container_id"] = out.M["container_id"].map(bmap)
    hold = out.E["event_type"] == "customs_hold"
    out.E.loc[hold, "target"] = out.E.loc[hold, "target"].map(lambda t: bmap.get(t, t))

    def fix(text):
        for old, new in cmap.items():
            text = str(text).replace(old, new)
        return text
    out.E["detail"] = out.E["detail"].map(fix)
    return out, _meta({"calls": len(cmap), "boxes": len(bmap)}, {"same_as": True})


def fam_tight(f: Frames, rng: np.random.Generator) -> tuple[Frames, dict]:
    """20-40 % more import boxes and a day longer in the yard: the yard is the bottleneck."""
    out = f.copy()
    frac = float(rng.uniform(0.2, 0.4))
    imp_mask = out.M["direction"] == "import"
    out.M.loc[imp_mask, "planned_dwell_days"] = out.M.loc[imp_mask, "planned_dwell_days"] + 1
    imp = out.M[imp_mask]
    n_add = round(len(imp) * frac)
    copies = imp.iloc[rng.integers(0, len(imp), n_add)].copy()
    used = set(out.M["container_id"]) | set(out.E["target"])
    copies["container_id"] = _fresh_ids(rng, used, n_add)
    added = copies.groupby("call_id").size()
    out.M = pd.concat([out.M, copies], ignore_index=True)
    out.V["discharge_moves"] = out.V["discharge_moves"] + out.V["call_id"].map(added).fillna(0).astype(int)
    return out, _meta({"added_imports": n_add, "fraction": round(frac, 3), "dwell_plus_days": 1})


def fam_resized(f: Frames, rng: np.random.Generator) -> tuple[Frames, dict]:
    """Ship sizes swapped between ships, only where both still fit some berth."""
    out = f.copy()
    order = list(rng.permutation(out.V.index))
    swapped = []

    def fits(loa, draft):
        return any(length >= loa + 20 and depth >= draft for length, depth in BERTHS.values())

    for a, b in zip(order[0::2], order[1::2]):
        la, da = out.V.at[a, "loa_m"], out.V.at[a, "draft_m"]
        lb, db = out.V.at[b, "loa_m"], out.V.at[b, "draft_m"]
        if fits(lb, db) and fits(la, da):          # always true for ships that fit today; kept as a guard
            out.V.at[a, "loa_m"], out.V.at[a, "draft_m"] = lb, db
            out.V.at[b, "loa_m"], out.V.at[b, "draft_m"] = la, da
            swapped.append([out.V.at[a, "call_id"], out.V.at[b, "call_id"]])
    return out, _meta({"swapped": swapped})


def _thin(f: Frames, rng: np.random.Generator, share: float, dwell_cap: Optional[int]) -> tuple[Frames, dict]:
    """Remove import boxes until each cargo group holds at most `share` of its legal yard slots.
    Then the yard can never be full for that group, even if no box ever leaves."""
    out = f.copy()
    imp = imports(out)
    drop = []
    shortage = {}
    for g, capacity in GROUP_CAPACITY.items():
        rows = imp[imp["cargo_type"] == g]
        limit = int(share * capacity)
        if len(rows) > limit:
            keep = set(rng.choice(rows.index.to_numpy(), limit, replace=False))
            drop += [i for i in rows.index if i not in keep]
        shortage[g] = {"capacity": capacity, "limit": limit, "imports_before": int(len(rows)),
                       "imports_after": int(min(len(rows), limit))}
    removed = out.M.loc[drop]
    gone = removed.groupby("call_id").size()
    out.V["discharge_moves"] = out.V["discharge_moves"] - out.V["call_id"].map(gone).fillna(0).astype(int)
    out.M = out.M.drop(index=drop).reset_index(drop=True)
    out.E = out.E[~((out.E["event_type"] == "customs_hold") & out.E["target"].isin(set(removed["container_id"])))]\
        .reset_index(drop=True)
    params = {"share_of_capacity": share, "removed": len(drop)}
    if dwell_cap is not None:
        m = out.M["direction"] == "import"
        out.M.loc[m, "planned_dwell_days"] = out.M.loc[m, "planned_dwell_days"].clip(upper=dwell_cap)
        params["dwell_cap_days"] = dwell_cap
    return out, _meta(params, {"noshort": True}, shortage)


def fam_noshort_fewer(f, rng):
    return _thin(f, rng, 0.60, None)


def fam_noshort_dwell(f, rng):
    return _thin(f, rng, 0.85, int(rng.choice([1, 2, 3])))


FAMILIES: dict[str, Callable] = {
    "identity": fam_identity,
    "shifted": fam_shifted,
    "bunched": fam_bunched,
    "breakdowns": fam_breakdowns,
    "holds": fam_holds,
    "mixed": fam_mixed,
    "renamed": fam_renamed,
    "tight": fam_tight,
    "resized": fam_resized,
    "noshort_fewer": fam_noshort_fewer,
    "noshort_dwell": fam_noshort_dwell,
}
DEFAULT_FAMILIES = [f for f in FAMILIES if f != "identity"]


# --------------------------------------------------------------------------
# Malformed kinds (fair tests only: the agent can see the problem through the toolkit's read tools).
# Targets are picked by rule, never by ID; their IDs go into "expect" for the checker.
# --------------------------------------------------------------------------

def _ships_by_discharge(f: Frames) -> list[str]:
    """Call ids, most discharge moves first (ties: call id)."""
    v = f.V.sort_values(["discharge_moves", "call_id"], ascending=[False, True])
    return list(v["call_id"])


def _normal_ships(f: Frames) -> list[str]:
    """Ships whose discharge moves equal their import boxes, most moves first."""
    n = imports(f).groupby("call_id").size()
    return [c for c in _ships_by_discharge(f) if int(f.V.loc[f.V.call_id == c, "discharge_moves"].iloc[0]) == int(n.get(c, 0))]


def _first_imports(f: Frames, call_id: str, k: int, cargo: Optional[str] = None) -> list[int]:
    m = imports(f)
    m = m[m["call_id"] == call_id]
    if cargo:
        m = m[m["cargo_type"] == cargo]
    return list(m.index[:k])


def _set(df: pd.DataFrame, index, col: str, value) -> None:
    """Set a value, widening the column dtype when needed (int -> float for NaN, int -> object for text)."""
    if isinstance(value, float) and np.isnan(value) and not pd.api.types.is_float_dtype(df[col]):
        df[col] = df[col].astype("float64")
    df.loc[index, col] = value


def mal_nan_size(f, rng):
    out = f.copy()
    a, b = _ships_by_discharge(out)[:2]
    _set(out.V, out.V.call_id == a, "loa_m", np.nan)
    _set(out.V, out.V.call_id == b, "draft_m", np.nan)
    return out, _meta({"nan_loa": a, "nan_draft": b}, {"never_berthed": [a, b], "anomaly": [a, b]})


def mal_too_big(f, rng):
    out = f.copy()
    a, b = _ships_by_discharge(out)[:2]
    out.V.loc[out.V.call_id == a, "loa_m"] = 340
    _set(out.V, out.V.call_id == b, "draft_m", 15.5)
    return out, _meta({"too_long": a, "too_deep": b}, {"never_berthed": [a, b], "anomaly": [a, b]})


def mal_impossible(f, rng):
    out = f.copy()
    c = _normal_ships(out)[0]
    n = int(out.V.loc[out.V.call_id == c, "discharge_moves"].iloc[0])
    out.V.loc[out.V.call_id == c, "discharge_moves"] = 3 * n
    return out, _meta({"ship": c, "discharge_moves": 3 * n, "imports": n},
                      {"event": "ship_impossible", "not_departed": [c], "anomaly": [c]})


def _bad_cargo(f, rng, values, name):
    out = f.copy()
    c = _ships_by_discharge(out)[1]
    idx = _first_imports(out, c, len(values), "dry")
    ids = list(out.M.loc[idx, "container_id"])
    out.M["cargo_type"] = out.M["cargo_type"].astype(object)
    for i, v in zip(idx, values):
        out.M.at[i, "cargo_type"] = v
    return out, _meta({"ship": c, "values": [str(v) for v in values]}, {"never_placed": ids, "anomaly": [c]})


def mal_unknown_cargo(f, rng):
    return _bad_cargo(f, rng, ["hazmat", "Reefer", "frozen", "HAZMAT", "dry "], "unknown_cargo")


def mal_nan_cargo(f, rng):
    return _bad_cargo(f, rng, [np.nan] * 5, "nan_cargo")


def mal_imdg_conflict(f, rng):
    out = f.copy()
    c = _ships_by_discharge(out)[1]
    idx = _first_imports(out, c, 3, "dry")
    ids = list(out.M.loc[idx, "container_id"])
    out.M.loc[idx, "imdg_class"] = 3.0
    return out, _meta({"ship": c}, {"never_placed": ids, "anomaly": [c]})


def mal_duplicate_box(f, rng):
    out = f.copy()
    ships = _ships_by_discharge(out)
    a, b, c = ships[0], ships[1], ships[2]
    i1 = _first_imports(out, a, 1, "dry")[0]
    i2 = _first_imports(out, b, 1, "dry")[0]
    same_ship = out.M.loc[[i1]].copy()
    other_ship = out.M.loc[[i2]].copy()
    other_ship["call_id"] = c
    ids = [out.M.at[i1, "container_id"], out.M.at[i2, "container_id"]]
    out.M = pd.concat([out.M, same_ship, other_ship], ignore_index=True)
    # validate names the ship, not the box: "N boxes of 'P02-004' are listed twice / are also listed on another ship"
    return out, _meta({"same_ship": a, "two_ships": [b, c]}, {"never_placed": ids, "anomaly": [a, b, c]})


def mal_negative_moves(f, rng):
    out = f.copy()
    a, b = _ships_by_discharge(out)[:2]
    out.V.loc[out.V.call_id == a, "discharge_moves"] = -10
    out.V.loc[out.V.call_id == b, "load_moves"] = -5
    return out, _meta({"negative_discharge": a, "negative_load": b}, {"anomaly": [a, b]})


def mal_bad_times(f, rng):
    out = f.copy()
    ships = list(out.V.sort_values(["scheduled_eta", "call_id"])["call_id"])
    late, nat = ships[-1], _ships_by_discharge(out)[0]
    if nat == late:
        nat = _ships_by_discharge(out)[1]
    start, end = window(out)
    out.V.loc[out.V.call_id == late, "scheduled_eta"] = end + pd.Timedelta(hours=24)
    out.V.loc[out.V.call_id == nat, "actual_arrival"] = pd.NaT
    return out, _meta({"eta_after_window": late, "no_arrival": nat}, {"never_berthed": [nat], "anomaly": [late]})


def mal_bad_box_data(f, rng):
    out = f.copy()
    c = _ships_by_discharge(out)[1]
    idx = _first_imports(out, c, 4, "dry")
    out.M["gross_weight_t"] = out.M["gross_weight_t"].astype("float64")
    out.M["size_ft"] = out.M["size_ft"].astype("float64")
    out.M.at[idx[0], "gross_weight_t"] = np.nan
    out.M.at[idx[1], "gross_weight_t"] = -3.0
    out.M.at[idx[2], "size_ft"] = 45
    out.M.at[idx[3], "size_ft"] = np.nan
    return out, _meta({"ship": c}, {"anomaly": [c]})


def mal_orphan_hold(f, rng):
    out = f.copy()
    start, end = window(out)
    _add_events(out, [{"event_type": "customs_hold", "target": "ZZZU0000000", "berth_id": np.nan,
                       "start": start + pd.Timedelta(hours=30), "end": pd.NaT, "detail": "Hold on import box from nowhere"}])
    return out, _meta({"target": "ZZZU0000000"})


# Kinds in book order. The yard_capacity kind changes a static sheet, so it has its own book (m_yard).
MALFORMED: dict[str, Callable] = {
    "nan_size": mal_nan_size, "too_big": mal_too_big, "impossible": mal_impossible,
    "unknown_cargo": mal_unknown_cargo, "nan_cargo": mal_nan_cargo, "imdg_conflict": mal_imdg_conflict,
    "duplicate_box": mal_duplicate_box, "negative_moves": mal_negative_moves, "bad_times": mal_bad_times,
    "bad_box_data": mal_bad_box_data, "orphan_hold": mal_orphan_hold,
}
YARD_KIND = "yard_capacity"


def yard_capacity(static: dict) -> dict:
    """Yard_Blocks with block A at 0 slots and block H at 10: free space is overstated if anyone trusts the old numbers."""
    s = dict(static)
    y = s["Yard_Blocks"].copy()
    y.loc[y.block_id == "A", "slots"] = 0
    y.loc[y.block_id == "H", "slots"] = 10
    s["Yard_Blocks"] = y
    return s


# --------------------------------------------------------------------------
# Toolkit-limit kinds: the organizers' toolkit itself fails on this data and the agent cannot see why.
# They live in their own set with their own pass rule (corePlan §3.6).
# --------------------------------------------------------------------------

def lim_nan_moves(f, rng):
    out = f.copy()
    a = _ships_by_discharge(out)[0]
    _set(out.V, out.V.call_id == a, "discharge_moves", np.nan)
    return out, _meta({"ship": a})


def lim_nan_dwell(f, rng):
    out = f.copy()
    c = _ships_by_discharge(out)[1]
    idx = _first_imports(out, c, 3, "dry")
    out.M["planned_dwell_days"] = out.M["planned_dwell_days"].astype("float64")
    out.M.loc[idx, "planned_dwell_days"] = np.nan
    return out, _meta({"ship": c})


def lim_duplicate_call(f, rng):
    out = f.copy()
    extra = out.V.iloc[[0]].copy()
    out.V = pd.concat([out.V, extra], ignore_index=True)
    return out, _meta({"ship": extra.call_id.iloc[0]})


LIMITS: dict[str, Callable] = {"nan_moves": lim_nan_moves, "nan_dwell": lim_nan_dwell, "duplicate_call": lim_duplicate_call}


# --------------------------------------------------------------------------
# Own sanity check: a generator bug fails here, not halfway through an eval
# --------------------------------------------------------------------------

def _deltas(f: Frames) -> list[tuple]:
    """Per ship in call-id order: (discharge - imports, load - exports)."""
    m = f.M.groupby(["call_id", "direction"]).size().unstack(fill_value=0)
    out = []
    for r in f.V.sort_values("call_id").itertuples():
        imp = int(m["import"].get(r.call_id, 0)) if "import" in m else 0
        exp = int(m["export"].get(r.call_id, 0)) if "export" in m else 0
        out.append((int(r.discharge_moves) - imp, int(r.load_moves) - exp))
    return out


def check_variant(f: Frames, family: str, base: Optional[Frames] = None, malformed: bool = False) -> list[str]:
    """Problems with a generated scenario ([] = fine). `malformed` skips the checks that bad data breaks on purpose."""
    bad = []
    if f.V.empty:
        return ["no vessels"]
    start, end = window(f)
    if base is not None and start != window(base)[0]:
        bad.append("window start moved")
    if not malformed:
        if f.V["call_id"].duplicated().any():
            bad.append("duplicate call ids")
        if f.M["container_id"].duplicated().any():
            bad.append("duplicate container ids")
        for col in ("scheduled_eta", "actual_arrival"):
            t = pd.to_datetime(f.V[col])
            if ((t < start) | (t >= end)).any():
                bad.append(f"{col} outside the window")
        if not f.E.empty:
            s = pd.to_datetime(f.E["start"])
            e = pd.to_datetime(f.E["end"])
            if ((s < start) | (s >= end)).any():
                bad.append("event start outside the window")
            if (e.notna() & (e <= s)).any():
                bad.append("event ends before it starts")
            if (e.notna() & (e > end)).any():
                bad.append("event ends after the window")
            holds = f.E[f.E["event_type"] == "customs_hold"]
            if not set(holds["target"]) <= set(f.M["container_id"]):
                bad.append("customs hold on a box that is in no manifest")
        if (f.V["discharge_moves"] < 0).any() or (f.V["load_moves"] < 0).any():
            bad.append("negative moves")
        if not set(f.M["call_id"]) <= set(f.V["call_id"]):
            bad.append("manifest rows for an unknown call")
        for r in f.V.itertuples():
            if not any(length >= r.loa_m + 20 and depth >= r.draft_m for length, depth in BERTHS.values()):
                bad.append(f"{r.call_id} fits no berth")
        if base is not None and _deltas(f) != _deltas(base):
            bad.append("moves no longer match boxes the way the source does")
        if base is not None and len(f.V) != len(base.V):
            bad.append("number of ships changed")
    return bad


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def write_book(path: Path, static: dict, by_scenario: dict[str, Frames], sheets: Optional[dict] = None,
               static_too: bool = True) -> None:
    """Static sheets unchanged, Practice_* sheets for all scenarios, FCFS sheet empty.
    Written to a temporary file and renamed, so a half-written book never exists.
    sheets: sheet names to use instead of Practice_* (the mock answer key's Hidden_*)."""
    names = sheets or PRACTICE_SHEETS
    parts = {k: [] for k in names}
    for sc, f in by_scenario.items():
        for k in ("V", "E", "M"):
            df = getattr(f, k).copy()
            df.insert(0, "scenario", sc)
            parts[k].append(df)
    tmp = path.with_name(path.stem + ".tmp.xlsx")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        if static_too:
            for name, df in static.items():
                df.to_excel(xw, sheet_name=name, index=False)
        for k in ("V", "E", "M"):
            pd.concat(parts[k], ignore_index=True).to_excel(xw, sheet_name=names[k], index=False)
        first = next(iter(by_scenario.values())).B
        pd.DataFrame(columns=["scenario"] + list(first.columns)).to_excel(xw, sheet_name=names["B"], index=False)
    os.replace(tmp, path)


def git_sha() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------
# Building sets
# --------------------------------------------------------------------------

def assign(book: int, scen: int, families: list[str]) -> str:
    """Family of scenario `scen` (0-4) in book `book` (0-based). The `book // 2` term makes the stride
    coprime with 5 scenarios, so over a set every scenario meets every family."""
    return families[(book * len(SCENARIOS) + scen + book // 2) % len(families)]


def _make_book(out: Path, name: str, kinds: list[str], registry: dict, src: Source, base: dict, rng_key: int,
               seed: int, static: Optional[dict] = None, malformed: bool = False) -> dict:
    by_scenario, meta = {}, {}
    for i, sc in enumerate(SCENARIOS):
        kind = kinds[i]
        rng = np.random.default_rng([seed, rng_key, i])
        frames, m = registry[kind](base[sc], rng)
        problems = check_variant(frames, kind, base[sc], malformed=malformed)
        if problems:
            raise SystemExit(f"{name}/{sc} ({kind}): generated data is wrong: {'; '.join(problems)}")
        by_scenario[sc] = frames
        meta[sc] = {"family": kind, **m}
    write_book(out / f"{name}.xlsx", static or src.static, by_scenario)
    return meta


def _index(src: Source, data: Path, seed: int) -> dict:
    return {"seed": seed, "source": data.name, "source_sha256": src.sha256, "generator": git_sha(), "books": {}}


def _finish(out: Path, data: Path, src: Source, index: dict) -> None:
    if sha256(data) != src.sha256:
        raise SystemExit("the source workbook changed while generating (it must never be modified)")
    (out / "index.json").write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")


def _clean(out: Path) -> None:
    for pattern in ("v*.xlsx", "m*.xlsx", "MOCK_ANSWER_KEY.xlsx", "index.json"):
        for p in out.glob(pattern):
            p.unlink(missing_ok=True)


def build(data: Path, out: Path, n: int, seed: int, families: list[str], clean: bool = False,
          malformed: int = 0, kinds: Optional[list[str]] = None, yard_book: bool = True,
          hidden_key: bool = False) -> dict:
    """Write n variant books (v01.xlsx ...), `malformed` malformed books (m01.xlsx ...), the yard book
    (m_yard.xlsx, when malformed > 0) and index.json into `out`. Returns the index."""
    unknown = [x for x in families if x not in FAMILIES]
    if unknown:
        raise SystemExit(f"unknown family {unknown}; known: {sorted(FAMILIES)}")
    src = load(data)
    check_static(src.static)
    base = split(src)
    out.mkdir(parents=True, exist_ok=True)
    if clean:
        _clean(out)
    index = _index(src, data, seed)
    for k in range(n):
        name = f"v{k + 1:02d}"
        kk = [assign(k, i, families) for i in range(len(SCENARIOS))]
        index["books"][name] = _make_book(out, name, kk, FAMILIES, src, base, k, seed)
    mk = kinds or list(MALFORMED)
    for j in range(malformed):
        name = f"m{j + 1:02d}"
        kk = [mk[(j * len(SCENARIOS) + i) % len(mk)] for i in range(len(SCENARIOS))]
        index["books"][name] = _make_book(out, name, kk, MALFORMED, src, base, 1000 + j, seed, malformed=True)
    if malformed and yard_book:
        meta = _make_book(out, "m_yard", ["identity"] * len(SCENARIOS), FAMILIES, src, base, 2000, seed,
                          static=yard_capacity(src.static))
        for sc, m in meta.items():
            m["family"] = YARD_KIND
            m["expect"] = {"yard_capacity": True}
        index["books"]["m_yard"] = meta
    if hidden_key:
        index["hidden_key"] = write_hidden_key(out / "MOCK_ANSWER_KEY.xlsx", base, seed)
    _finish(out, data, src, index)
    return index


def build_ci(data: Path, out: Path, seed: int = 7) -> dict:
    """The fixed CI set: 1 mixed book, 1 no-shortage book (the two kinds alternating), 2 malformed books
    (the first ten kinds once each). 4 books = 20 scenario runs."""
    src = load(data)
    check_static(src.static)
    base = split(src)
    out.mkdir(parents=True, exist_ok=True)
    _clean(out)
    index = _index(src, data, seed)
    index["books"]["v01"] = _make_book(out, "v01", ["mixed"] * 5, FAMILIES, src, base, 0, seed)
    ns = ["noshort_fewer", "noshort_dwell", "noshort_fewer", "noshort_dwell", "noshort_fewer"]
    index["books"]["v02"] = _make_book(out, "v02", ns, FAMILIES, src, base, 1, seed)
    mk = list(MALFORMED)[:10]
    for j in range(2):
        name = f"m{j + 1:02d}"
        kk = [mk[(j * len(SCENARIOS) + i) % len(mk)] for i in range(len(SCENARIOS))]
        index["books"][name] = _make_book(out, name, kk, MALFORMED, src, base, 1000 + j, seed, malformed=True)
    _finish(out, data, src, index)
    return index


def build_limits(data: Path, out: Path, seed: int = 1) -> dict:
    """data/variants/limits/L1.xlsx ...: one toolkit-limit kind on each scenario of its book."""
    src = load(data)
    check_static(src.static)
    base = split(src)
    out.mkdir(parents=True, exist_ok=True)
    for p in list(out.glob("L*.xlsx")) + [out / "index.json"]:
        p.unlink(missing_ok=True)
    index = _index(src, data, seed)
    for j, kind in enumerate(LIMITS):
        name = f"L{j + 1}"
        index["books"][name] = _make_book(out, name, [kind] * len(SCENARIOS), LIMITS, src, base, 3000 + j, seed,
                                          malformed=True)
    _finish(out, data, src, index)
    return index


def write_hidden_key(path: Path, base: dict, seed: int) -> dict:
    """MOCK_ANSWER_KEY.xlsx: Hidden_Vessels / Hidden_Events / Hidden_Manifest / Hidden_Baseline_FCFS with
    scenarios hidden_01..hidden_05 (one perturbed family each), to rehearse the organizers' load path:
    TerminalAPI("hidden_01", data_xlsx=<the data workbook>, key_xlsx=<this file>)."""
    fams = ["mixed", "shifted", "breakdowns", "holds", "noshort_fewer"]
    frames, used = {}, {}
    for i, (sc, fam) in enumerate(zip(SCENARIOS, fams)):
        rng = np.random.default_rng([seed, 4000, i])
        f, _ = FAMILIES[fam](base[sc], rng)
        frames[f"hidden_{i + 1:02d}"] = f
        used[f"hidden_{i + 1:02d}"] = {"base": sc, "family": fam}
    write_book(path, {}, frames, sheets=HIDDEN_SHEETS, static_too=False)
    return used


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20, help="number of variant workbooks (default 20)")
    p.add_argument("--malformed", type=int, default=4, help="number of malformed workbooks, plus the yard book (default 4)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=None, help=f"default {DEFAULT_OUT} (--ci: {CI_OUT})")
    p.add_argument("--families", default=",".join(DEFAULT_FAMILIES), help="comma-separated perturbed families")
    p.add_argument("--clean", action="store_true", help="delete old books and index.json in --out first")
    p.add_argument("--ci", action="store_true", help="the fixed CI set (seed 7) in runs/check/variants/")
    p.add_argument("--limits", action="store_true", help="the toolkit-limit set in <out>/limits/")
    p.add_argument("--hidden-key", action="store_true", help="also write MOCK_ANSWER_KEY.xlsx (hidden_01..05)")
    a = p.parse_args(argv)
    if not a.data.exists():
        p.error(f"data workbook not found: {a.data}")
    if a.ci:
        out = a.out or CI_OUT
        idx = build_ci(a.data, out)
        print(f"wrote the CI set ({len(idx['books'])} books) to {out}")
        return 0
    out = a.out or DEFAULT_OUT
    if a.limits:
        idx = build_limits(a.data, out / "limits", a.seed)
        print(f"wrote {len(idx['books'])} toolkit-limit books to {out / 'limits'}")
        return 0
    fams = [x for x in a.families.split(",") if x]
    if not fams:
        p.error("no families given")
    idx = build(a.data, out, a.n, a.seed, fams, a.clean, a.malformed, hidden_key=a.hidden_key)
    kinds = sorted({m["family"] for b in idx["books"].values() for m in b.values()})
    print(f"wrote {len(idx['books'])} workbooks to {out} (seed {a.seed}); families: {', '.join(kinds)}"
          + (f"; mock answer key {out / 'MOCK_ANSWER_KEY.xlsx'}" if a.hidden_key else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
