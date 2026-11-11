"""Tuning parameters (BC-4): config/params.yaml, read once per process.

    from core import params
    HOLD_H = params.get("berth.hold_h")

The file is a small YAML subset (sections, `key: value`, `#` comments), so no new dependency is
needed. Every key has a default here, equal to the value the agents used before BC-4, so a missing
or broken file changes nothing: the run never crashes on it, and `problems()` says what was wrong
(the loop logs it as a note). AGENTX_PARAMS=<path> points at another file (used by the tuning
sweep in tools/tune.py; never set in a real run).

Rules for adding a parameter (GROUND_RULES §3, BC-4 "watch out"): it must make sense on any
terminal and any scenario. No parameter may name a scenario, ship, berth, crane or box.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "config" / "params.yaml"

# key -> (default, (min, max) or tuple of allowed values)
SPEC: dict[str, tuple[Any, tuple]] = {
    "berth.hold_h": (2.0, (0.0, 24.0)),          # keep a berth this long ahead of a ship that fits only it
    "berth.age_weight": (0.0, (0.0, 1000.0)),    # moves of priority a waiting ship gains per hour waited
    "crane.tick_h": (0.25, (0.05, 4.0)),         # horizon for "cranes a ship can keep busy"
    "crane.spare_to": ("shortest_job", ("shortest_job", "most_work")),   # where spare cranes go
    "options.min_gain_h": (1.0, (0.0, 24.0)),    # an alternative plan must change the outcome by this much
    "yard.soft_fill": (0.85, (0.5, 1.0)),        # YARD-6: keep standard blocks at or under this while other room exists
    "yard.prefer": ("nearest", ("nearest", "together")),   # YARD-6: nearest block, or the ship's own blocks first
    "yard.reefer_keep": (0.0, (0.0, 1.0)),       # YARD-6: share of each reefer block dry boxes may not use
    "yard.space_order": ("arrival", ("arrival", "least_left")),   # YARD-7: who gets scarce yard space first
}

_values: Optional[dict] = None
_problems: list[str] = []


def _parse(text: str) -> tuple[dict, list[str]]:
    """'section:' lines and indented 'key: value' lines; returns ({"section.key": raw}, problems)."""
    out, bad, section = {}, [], None
    for n, raw in enumerate(text.split("\n"), 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            if not line.endswith(":") or " " in line[:-1]:
                bad.append(f"line {n}: expected 'section:'")
                section = None
                continue
            section = line[:-1]
            continue
        if section is None or ":" not in line:
            bad.append(f"line {n}: expected 'key: value' under a section")
            continue
        key, value = (x.strip() for x in line.split(":", 1))
        out[f"{section}.{key}"] = value
    return out, bad


def _convert(key: str, raw: str) -> Any:
    default, allowed = SPEC[key]
    raw = raw.strip().strip("'\"")
    if isinstance(default, str):
        if raw not in allowed:
            raise ValueError(f"{raw!r} not one of {', '.join(allowed)}")
        return raw
    v = float(raw)
    if v != v or not allowed[0] <= v <= allowed[1]:
        raise ValueError(f"{raw!r} outside {allowed[0]}..{allowed[1]}")
    return v


def load(path: Optional[Path] = None) -> dict:
    """Read (or re-read) the parameters. Never raises: bad entries keep their default."""
    global _values, _problems
    vals = {k: d for k, (d, _) in SPEC.items()}
    probs: list[str] = []
    p = Path(path or os.environ.get("AGENTX_PARAMS") or DEFAULT_PATH)
    try:
        raw, probs = _parse(p.read_text(encoding="utf-8"))
    except OSError as e:
        raw = {}
        probs.append(f"cannot read {p.name}: {type(e).__name__}; defaults used")
    for key, value in raw.items():
        if key not in SPEC:
            probs.append(f"unknown parameter {key!r} ignored")
            continue
        try:
            vals[key] = _convert(key, value)
        except ValueError as e:
            probs.append(f"{key}: {e}; default {SPEC[key][0]!r} used")
    _values, _problems = vals, probs
    return dict(vals)


def get(key: str) -> Any:
    if _values is None:
        load()
    return _values[key]


def problems() -> list[str]:
    if _values is None:
        load()
    return list(_problems)
