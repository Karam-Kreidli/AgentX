"""Scenario variants (CORE-7): workbooks made from the practice data, to catch overfitting.

    python tools/perturb.py --n 20 --seed 1            # 20 variant workbooks in data/variants/
    python tools/perturb.py --n 20 --families identity # only some families

Each variant workbook holds all five scenario names (practice_01..05), so tools/eval_all.py
--variants N and `run.py --data data/variants/v01.xlsx --scenario practice_02` load it through the
toolkit's normal path. The five static sheets (Berths, Quay_Cranes, Yard_Trucks, Yard_Blocks,
Truck_Travel_Times) are copied unchanged: agent code has berths, crane split and truck travel times
built in, so a variant must not change them. The four Practice_* sheets are rewritten per scenario by
one family each. Practice_Baseline_FCFS is written empty: the workbook's FCFS numbers only match the
original data (corePlan D2).

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
SCENARIOS = [f"practice_0{i}" for i in range(1, 6)]
STATIC_SHEETS = ["Berths", "Quay_Cranes", "Yard_Trucks", "Yard_Blocks", "Truck_Travel_Times"]
PRACTICE_SHEETS = {"V": "Practice_Vessels", "E": "Practice_Events", "M": "Practice_Manifest",
                   "B": "Practice_Baseline_FCFS"}
WINDOW_DAYS = 5          # the toolkit: start = floor(min scheduled_eta, day), end = start + 5 days


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


def empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def window(f: Frames) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The toolkit's scenario window."""
    start = pd.to_datetime(f.V["scheduled_eta"]).min().floor("D")
    return start, start + pd.Timedelta(days=WINDOW_DAYS)


# --------------------------------------------------------------------------
# Families. Each is a pure function (frames, rng) -> (frames, params); never in place.
# --------------------------------------------------------------------------

def fam_identity(f: Frames, rng: np.random.Generator) -> tuple[Frames, dict]:
    """Nothing changes. Tests the writer and the toolkit path."""
    return f.copy(), {}


FAMILIES: dict[str, Callable] = {
    "identity": fam_identity,
}


# --------------------------------------------------------------------------
# Own sanity check: a generator bug fails here, not halfway through an eval
# --------------------------------------------------------------------------

def check_variant(f: Frames, family: str) -> list[str]:
    """Problems with a generated scenario ([] = fine)."""
    bad = []
    if f.V.empty:
        return ["no vessels"]
    if f.V["call_id"].duplicated().any() and family != "malformed":
        bad.append("duplicate call ids")
    start, end = window(f)
    for col in ("scheduled_eta", "actual_arrival"):
        t = pd.to_datetime(f.V[col])
        if t.notna().all() and ((t < start) | (t >= end)).any():
            bad.append(f"{col} outside the window")
    if not f.E.empty:
        s = pd.to_datetime(f.E["start"])
        if s.notna().all() and ((s < start) | (s >= end)).any():
            bad.append("event start outside the window")
    return bad


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def write_book(path: Path, static: dict, by_scenario: dict[str, Frames]) -> None:
    """Static sheets unchanged, Practice_* sheets for all scenarios, FCFS sheet empty.
    Written to a temporary file and renamed, so a half-written book never exists."""
    parts = {k: [] for k in PRACTICE_SHEETS}
    for sc, f in by_scenario.items():
        for k in ("V", "E", "M"):
            df = getattr(f, k).copy()
            df.insert(0, "scenario", sc)
            parts[k].append(df)
    tmp = path.with_name(path.stem + ".tmp.xlsx")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        for name, df in static.items():
            df.to_excel(xw, sheet_name=name, index=False)
        for k in ("V", "E", "M"):
            pd.concat(parts[k], ignore_index=True).to_excel(xw, sheet_name=PRACTICE_SHEETS[k], index=False)
        first = next(iter(by_scenario.values())).B
        cols = ["scenario"] + list(first.columns)
        pd.DataFrame(columns=cols).to_excel(xw, sheet_name=PRACTICE_SHEETS["B"], index=False)
    os.replace(tmp, path)


def git_sha() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------
# Building a set
# --------------------------------------------------------------------------

def assign(book: int, scen: int, families: list[str]) -> str:
    """Family of scenario `scen` (0-4) in book `book` (0-based): spread so every family meets different bases."""
    return families[(book * len(SCENARIOS) + scen) % len(families)]


def build(data: Path, out: Path, n: int, seed: int, families: list[str], clean: bool = False) -> dict:
    """Write n variant workbooks (v01.xlsx ...) and index.json into `out`. Returns the index."""
    unknown = [x for x in families if x not in FAMILIES]
    if unknown:
        raise SystemExit(f"unknown family {unknown}; known: {sorted(FAMILIES)}")
    src = load(data)
    base = split(src)
    out.mkdir(parents=True, exist_ok=True)
    if clean:
        for p in list(out.glob("v*.xlsx")) + [out / "index.json"]:
            p.unlink(missing_ok=True)
    index = {"seed": seed, "source": data.name, "source_sha256": src.sha256, "generator": git_sha(), "books": {}}
    for k in range(n):
        name = f"v{k + 1:02d}"
        by_scenario, meta = {}, {}
        for i, sc in enumerate(SCENARIOS):
            fam = assign(k, i, families)
            rng = np.random.default_rng([seed, k, i])
            frames, params = FAMILIES[fam](base[sc], rng)
            problems = check_variant(frames, fam)
            if problems:
                raise SystemExit(f"{name}/{sc} ({fam}): generated data is wrong: {'; '.join(problems)}")
            by_scenario[sc] = frames
            meta[sc] = {"family": fam, "params": params}
        write_book(out / f"{name}.xlsx", src.static, by_scenario)
        index["books"][name] = meta
    if sha256(data) != src.sha256:
        raise SystemExit("the source workbook changed while generating (it must never be modified)")
    (out / "index.json").write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
    return index


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20, help="number of variant workbooks (default 20)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--families", default=",".join(f for f in FAMILIES if f != "identity"),
                   help="comma-separated families (default: all except identity)")
    p.add_argument("--clean", action="store_true", help="delete old v*.xlsx and index.json in --out first")
    a = p.parse_args(argv)
    if not a.data.exists():
        p.error(f"data workbook not found: {a.data}")
    fams = [x for x in a.families.split(",") if x]
    if not fams:
        p.error("no families to build: none implemented besides 'identity' yet; pass --families identity")
    idx = build(a.data, a.out, a.n, a.seed, fams, a.clean)
    print(f"wrote {len(idx['books'])} workbooks to {a.out} (seed {a.seed}, families {','.join(fams)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
