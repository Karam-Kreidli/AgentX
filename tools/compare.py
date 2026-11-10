"""CORE-8: compare the four systems side by side. Why multi-agent?

    python tools/compare.py                       # 4 systems x the 5 practice scenarios, no LLM
    python tools/compare.py --variants 20         # + the first 20 variant workbooks (CORE-7)
    python tools/compare.py --llm                 # + multi with the LLM (one scenario at a time)

Systems (all use the same guard, executor, validate, invariant monitor and rules-only fallback;
only the decision logic differs):
    skeleton       FCFS rules (oldest ship first, first free berth, fixed cranes). Coordinator off.
    single         ONE planner for berth, crane, truck and yard (baselines/single_agent.py). Coordinator off.
    multi-nocoord  our four agents without the coordinator (ablation: no negotiation, no admission control).
    multi          our four agents with the coordinator.

Sets: practice (5 scenarios); variants (perturbed + no-shortage books, "v*"); malformed ("m*",
robustness columns only: violations and crashes; their KPIs are not comparable because a ship
with bad data cannot finish).

Writes runs/compare/compare.json, compare.md and compare_*.svg. It reuses tools/eval_all.py
(run_all, make_row, violations_total), so every metric is defined once.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import eval_all  # noqa: E402

OUT_DIR = ROOT / "runs" / "compare"
SYSTEMS = [("skeleton", "FCFS"), ("single", "Single agent"), ("multi-nocoord", "Multi, no coordinator"),
           ("multi", "Multi (ours)")]
LLM_KEY, LLM_LABEL = "multi+llm", "Multi + LLM"
SETS = ("practice", "variants", "malformed")


def set_of(label: str) -> str:
    """practice_01 -> practice; v03/practice_02 -> variants; m02/practice_01 -> malformed."""
    if "/" not in label:
        return "practice"
    return "malformed" if label.split("/", 1)[0].startswith("m") else "variants"


def _mean(xs: list) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def aggregate(rows: list[dict]) -> dict:
    """One system on one set: the Done-when table columns plus truck idle, >85 %, wall time."""
    ok = [r for r in rows if not r.get("crashed")]
    return {
        "scenarios": len(rows),
        "departed": sum(r["departed"] or 0 for r in ok),
        "total": sum(r["total"] or 0 for r in ok),
        "honest_h": _mean([r["honest_h"] for r in ok]),
        "sim_h": _mean([r["sim_h"] for r in ok]),
        "moves_h": _mean([r["moves_h"] for r in ok]),
        "violations": sum(eval_all.violations_total(r) for r in ok),
        "crashes": sum(1 for r in rows if r.get("crashed")),
        "truck_idle_h": round(sum(r["truck_idle_h"] or 0 for r in ok), 1),
        "over85": sum(r["over85"] or 0 for r in ok),
        "wall_s": _mean([r["wall_s"] for r in ok]),
    }


def _cmp(a, b, tol: float = 0.0, lower_is_better: bool = False) -> str:
    """'win' / 'tie' / 'loss' of a against b. None on either side is a tie (nothing to compare)."""
    if a is None or b is None or abs(a - b) <= tol:
        return "tie"
    better = a < b if lower_is_better else a > b
    return "win" if better else "loss"


def paired(a_rows: list[dict], b_rows: list[dict]) -> dict:
    """A against B on every label both finished: win/tie/loss on ships departed, then on honest turnaround."""
    bm = {r["label"]: r for r in b_rows if not r.get("crashed")}
    out = {"n": 0, "departed": {"win": 0, "tie": 0, "loss": 0}, "honest_h": {"win": 0, "tie": 0, "loss": 0},
           "not_worse": 0}
    for r in a_rows:
        o = bm.get(r["label"])
        if r.get("crashed") or o is None:
            continue
        out["n"] += 1
        d = _cmp(r["departed"], o["departed"])
        t = _cmp(r["honest_h"], o["honest_h"], tol=0.005, lower_is_better=True)
        out["departed"][d] += 1
        out["honest_h"][t] += 1
        # Same rule as GROUND_RULES §6 / the FCFS check: departed >= and turnaround within 3 %.
        if d != "loss" and (r["honest_h"] is None or o["honest_h"] is None
                            or r["honest_h"] <= o["honest_h"] * (1 + eval_all.TURNAROUND_SLACK) + 1e-9):
            out["not_worse"] += 1
    return out


# --------------------------------------------------------------------------- output

def _f(v, nd: int = 2) -> str:
    return "-" if v is None else (f"{v:.{nd}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v))


COLS = [("Ships departed", lambda a: f"{a['departed']}/{a['total']}"),
        ("Honest TA h", lambda a: _f(a["honest_h"])), ("Sim TA h", lambda a: _f(a["sim_h"])),
        ("Crane moves/h", lambda a: _f(a["moves_h"])), ("Violations", lambda a: str(a["violations"])),
        ("Crashes", lambda a: str(a["crashes"])), ("Truck idle h", lambda a: _f(a["truck_idle_h"], 1)),
        (">85 % placements", lambda a: str(a["over85"])), ("Wall s", lambda a: _f(a["wall_s"], 1))]


def markdown(card: dict) -> str:
    names = dict(SYSTEMS, **{LLM_KEY: LLM_LABEL})
    lines = []
    for st in SETS:
        keys = [k for k in names if k in card["tables"] and st in card["tables"][k]]
        if not keys:
            continue
        n = card["tables"][keys[0]][st]["scenarios"]
        lines += [f"### {st} ({n} scenarios)", ""]
        cols = COLS if st != "malformed" else [c for c in COLS if c[0] in ("Violations", "Crashes", "Wall s")]
        lines += ["| System | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---|" * len(cols)]
        for k in keys:
            a = card["tables"][k][st]
            lines.append(f"| {names[k]} | " + " | ".join(c[1](a) for c in cols) + " |")
        lines.append("")
    pr = card.get("paired") or {}
    if pr:
        lines += ["### Paired view on variant scenarios (first system against second)", "",
                  "| Pair | Scenarios | Departed win/tie/loss | Honest TA win/tie/loss | Not worse (departed >=, TA within 3 %) |",
                  "|---|---|---|---|---|"]
        for name, p in pr.items():
            d, t = p["departed"], p["honest_h"]
            lines.append(f"| {name} | {p['n']} | {d['win']}/{d['tie']}/{d['loss']} | {t['win']}/{t['tie']}/{t['loss']} "
                         f"| {p['not_worse']} of {p['n']} |")
        lines.append("")
    return "\n".join(lines)


PALETTE = {"skeleton": "#8a8f98", "single": "#d98e04", "multi-nocoord": "#2f8f83", "multi": "#2b6cb0",
           LLM_KEY: "#7b4bb7"}


def svg_chart(card: dict, metric: str, title: str, unit: str) -> str:
    """Grouped bars: one group per practice scenario, one bar per system. Hand-written SVG (no plotting
    library is installed); colours come from CSS so it reads in light and dark."""
    names = dict(SYSTEMS, **{LLM_KEY: LLM_LABEL})
    keys = [k for k in names if k in card["practice_rows"]]
    scen = sorted({r["label"] for k in keys for r in card["practice_rows"][k]})
    val = {(k, r["label"]): r.get(metric) for k in keys for r in card["practice_rows"][k] if not r.get("crashed")}
    top = max([v for v in val.values() if v is not None] or [1]) * 1.12
    W, H, L, R, T, B = 900, 460, 60, 20, 60, 90
    pw, ph = W - L - R, H - T - B
    gw = pw / len(scen)
    bw = gw * 0.8 / len(keys)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="{escape(title)}" font-family="system-ui, sans-serif">',
         "<style>.bg{fill:#fff}.tx{fill:#1f2933}.mu{fill:#52606d}.gr{stroke:#d9dee3}"
         "@media (prefers-color-scheme: dark){.bg{fill:#14181d}.tx{fill:#e8ecf0}.mu{fill:#9aa5b1}.gr{stroke:#2c333b}}</style>",
         f'<rect class="bg" width="{W}" height="{H}"/>',
         f'<text class="tx" x="{L}" y="28" font-size="18" font-weight="600">{escape(title)}</text>']
    for i in range(5):
        y = T + ph - ph * i / 4
        o.append(f'<line class="gr" x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="mu" x="{L - 8}" y="{y + 4:.1f}" font-size="12" text-anchor="end">{_f(top * i / 4 / 1.12, 1)}</text>')
    for gi, sc in enumerate(scen):
        x0 = L + gi * gw + gw * 0.1
        for ki, k in enumerate(keys):
            v = val.get((k, sc))
            if v is None:
                continue
            h = ph * v / top
            x, y = x0 + ki * bw, T + ph - h
            o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" fill="{PALETTE[k]}">'
                     f"<title>{escape(names[k])} {escape(sc)}: {_f(v)}{unit}</title></rect>")
            o.append(f'<text class="tx" x="{x + (bw - 2) / 2:.1f}" y="{y - 4:.1f}" font-size="11" text-anchor="middle">{_f(v, 1)}</text>')
        o.append(f'<text class="mu" x="{L + gi * gw + gw / 2:.1f}" y="{T + ph + 20}" font-size="12" text-anchor="middle">{escape(sc)}</text>')
    lx = L
    for k in keys:
        o.append(f'<rect x="{lx}" y="{H - 40}" width="14" height="14" fill="{PALETTE[k]}"/>')
        o.append(f'<text class="tx" x="{lx + 20}" y="{H - 28}" font-size="13">{escape(names[k])}</text>')
        lx += 40 + 8 * len(names[k])
    o.append("</svg>")
    return "\n".join(o)


# --------------------------------------------------------------------------- main

def main(argv: Optional[list[str]] = None) -> int:
    eval_all._utf8_stdout()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", type=int, default=0, metavar="N", help="also run the first N variant workbooks")
    p.add_argument("--variants-dir", type=Path, default=None, metavar="DIR", help="default data/variants")
    p.add_argument("--llm", action="store_true", help="also run multi with the LLM (one scenario at a time)")
    p.add_argument("--data", type=Path, default=eval_all.DEFAULT_DATA)
    p.add_argument("--jobs", type=int, default=None, help="scenarios in parallel (default: all CPUs)")
    p.add_argument("--out", type=Path, default=OUT_DIR, help="default runs/compare/")
    a = p.parse_args(argv)
    if not a.data.exists():
        p.error(f"data workbook not found: {a.data}")

    vdir = a.variants_dir or eval_all.VARIANTS_DIR
    spec = eval_all.plan(eval_all.PRACTICE, a.variants, a.data, vdir)
    if a.variants and len(spec) == len(eval_all.PRACTICE):
        print(f"note: no variant workbooks in {vdir} (python tools/perturb.py makes them); practice only")
    meta = eval_all.meta_for(spec, vdir)
    jobs = a.jobs or (os.cpu_count() or 1)

    t0 = time.perf_counter()
    runs: dict[str, list[dict]] = {}
    for key, label in SYSTEMS:
        print(f"running {label} ({key}) on {len(spec)} scenarios ...", flush=True)
        runs[key] = eval_all.run_all(ROOT, spec, a.out / "runs" / key, key, True, jobs, meta)
    if a.llm:
        print(f"running {LLM_LABEL} on {len(spec)} scenarios, one at a time ...", flush=True)
        runs[LLM_KEY] = eval_all.run_all(ROOT, spec, a.out / "runs" / LLM_KEY, "multi", False, 1, meta)

    tables = {k: {st: aggregate([r for r in rows if set_of(r["label"]) == st]) for st in SETS
                  if any(set_of(r["label"]) == st for r in rows)} for k, rows in runs.items()}
    var = {k: [r for r in rows if set_of(r["label"]) == "variants"] for k, rows in runs.items()}
    pairs = {}
    if var.get("multi"):
        pairs = {"Multi vs Single": paired(var["multi"], var["single"]),
                 "Multi vs Multi without coordinator": paired(var["multi"], var["multi-nocoord"]),
                 "Multi vs FCFS": paired(var["multi"], var["skeleton"]),
                 "Single vs FCFS": paired(var["single"], var["skeleton"])}
    card = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": eval_all.head_info()["commit"],
            "settings": {"variants": a.variants, "llm": a.llm, "data": a.data.name},
            "tables": tables, "paired": pairs, "rows": runs,
            "practice_rows": {k: [r for r in rows if set_of(r["label"]) == "practice"] for k, rows in runs.items()},
            "wall_s": round(time.perf_counter() - t0, 1)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "compare.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    md = markdown(card)
    (a.out / "compare.md").write_text(md, encoding="utf-8")
    (a.out / "compare_departed.svg").write_text(
        svg_chart(card, "departed", "Ships departed per practice scenario (more is better)", ""), encoding="utf-8")
    (a.out / "compare_turnaround.svg").write_text(
        svg_chart(card, "honest_h", "Honest turnaround per practice scenario (lower is better)", " h"), encoding="utf-8")
    print("\n" + md)
    print(f"saved {a.out}/compare.json, compare.md, compare_departed.svg, compare_turnaround.svg · {card['wall_s']} s")
    bad = sum(t[st]["violations"] + t[st]["crashes"] for t in tables.values() for st in t)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
