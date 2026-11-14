"""Live demo: build a scenario in front of the judges and race our agents against the baselines.

    python tools/live.py                       # opens http://127.0.0.1:8765 in the browser
    python tools/live.py --key path/to/ANSWER_KEY.xlsx --port 8765 --no-browser

Each strategy you tick runs in its own process with the real code (run.make_agents + core.loop.Loop),
on the same scenario, through the same guard and executor as `python run.py`. The page reads what
the runs write (ticks.jsonl, negotiation.jsonl, decisions.jsonl, llm_calls.jsonl, summary.json) as it is written, so
every decision on screen was made seconds earlier. Nothing is pre-recorded.

The scenario builder changes the simulator instance in memory before the run (arrival pattern,
crane and truck breakdowns, customs holds, yard size), the same way for every strategy, from one
seed. A new scenario can also be dropped onto the page as a workbook laid out like the practice data
(Practice_* or Hidden_* sheets; terminal sheets optional). During a run the judges can break cranes or trucks (every strategy gets the same breakdown at
the same simulated time) and, with the model on, send an operator message to our agents.

A test and demo tool: agent code never reads the scenario it is given here except through the
simulator's tools. Standard library only; no new dependency.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import shutil
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGE = ROOT / "tools" / "live_dashboard.html"
LIVE_DIR = ROOT / "runs" / "live"
CACHE_DIR = ROOT / "runs" / "live_cache"
_CACHE_LOCK = threading.Lock()
DEFAULT_DATA = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"
SYSTEMS = {                       # key: (label, run.make_agents kind)
    "multi": ("VerteX", "multi"),
    "nocoord": ("Agents, no coordinator", "multi-nocoord"),
    "single": ("Single agent", "single"),
    "fcfs": ("First come, first served", "skeleton"),
}
CRANES = ["QC1", "QC2", "QC3", "QC4", "QC5", "QC6"]


# --------------------------------------------------------------------------- the scenario

def base_api(base: str, data: Path, key: Optional[Path]):
    """A fresh, untouched TerminalAPI for `base`. Reading the workbook takes seconds, so the first build
    is saved as a snapshot (keyed on the workbook's size and time) and later runs load that instead."""
    from vendor import TerminalAPI
    stamp = lambda f: f"{f.stat().st_size}-{int(f.stat().st_mtime)}" if f else "none"
    snap = CACHE_DIR / f"{base}-{stamp(Path(data))}-{stamp(Path(key) if key else None)}.pkl"
    try:
        return pickle.loads(snap.read_bytes())
    except Exception:
        pass
    api = TerminalAPI(base, data_xlsx=str(data), key_xlsx=str(key) if key else None, log_path=os.devnull)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = snap.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(pickle.dumps(api))
        os.replace(tmp, snap)
    except Exception:
        pass                                             # no snapshot: the next run just reads the workbook again
    return api


def warm(bases: list, data: Path, key: Optional[Path]) -> None:
    """Prepare snapshots in the background so the first run in front of the judges starts quickly."""
    for b in bases:
        with _CACHE_LOCK:
            try:
                base_api(b, STATE["uploads"].get(b, data), key)
            except Exception:
                pass


def build_api(base: str, data: Path, key: Optional[Path], knobs: dict, log_path: Optional[Path] = None):
    """A TerminalAPI for `base`, changed in memory by the scenario builder's knobs. Same knobs + seed
    give the same scenario in every process."""
    import pandas as pd
    api = base_api(base, data, key)
    api.log_path = Path(log_path) if log_path else Path(os.devnull)
    if log_path:
        api.log_path.parent.mkdir(parents=True, exist_ok=True)
        api.log_path.write_text("")
    rng = random.Random(int(knobs.get("seed", 1)))
    calls = api.calls
    ids = sorted(calls, key=lambda c: calls[c]["actual_arrival"])
    t0 = min(c["scheduled_eta"] for c in calls.values())
    end = api.end - pd.Timedelta(hours=12)

    pattern = knobs.get("arrivals", "schedule")
    for cid in ids:
        c = calls[cid]
        if pattern == "bunched":            # everyone shows up within about half the time
            for f in ("scheduled_eta", "actual_arrival"):
                c[f] = (t0 + (c[f] - t0) * 0.5).floor("15min")
        elif pattern == "late":             # ships run late by 2 to 10 hours
            c["actual_arrival"] = min(end, (c["actual_arrival"] + pd.Timedelta(hours=rng.uniform(2, 10))).floor("15min"))
        elif pattern == "early":            # ships run early by up to 6 hours (never before the window starts)
            c["actual_arrival"] = max(t0, (c["actual_arrival"] - pd.Timedelta(hours=rng.uniform(0, 6))).floor("15min"))

    first = min(c["actual_arrival"] for c in calls.values())
    for b in knobs.get("breakdowns") or []:            # [{"crane": "QC2", "at_h": 6, "hours": 4}]
        cid = str(b.get("crane"))
        if cid in api.cranes.index:
            s = first + pd.Timedelta(hours=float(b.get("at_h", 6)))
            api.down.setdefault(cid, []).append((s, s + pd.Timedelta(hours=float(b.get("hours", 4)))))
    n_tr = int(knobs.get("truck_breakdowns", 0))
    if n_tr:
        s = first + pd.Timedelta(hours=float(knobs.get("truck_at_h", 8)))
        for tid in sorted(api.trucks.index)[:n_tr]:
            api.down.setdefault(tid, []).append((s, s + pd.Timedelta(hours=float(knobs.get("truck_hours", 6)))))

    n_hold = int(knobs.get("holds", 0))
    if n_hold:
        imp = api.manifest[api.manifest.direction == "import"]
        pick = rng.sample(sorted(imp.index), min(n_hold, len(imp)))
        rows = []
        for cid in pick:
            s = first + pd.Timedelta(hours=rng.uniform(0, 48))
            e = s + pd.Timedelta(hours=rng.uniform(12, 48))
            api.holds.setdefault(cid, []).append((s, e))
            rows.append({"event_type": "customs_hold", "target": cid, "start": s, "end": e, "detail": "Customs hold (live demo)"})
        api.events = pd.concat([api.events, pd.DataFrame(rows)], ignore_index=True)

    yard = float(knobs.get("yard_pct", 100)) / 100.0
    if yard != 1.0:
        api.blocks["slots"] = [max(20, int(round(s * yard / 4) * 4)) for s in api.blocks["slots"]]
    return api


def scenario_meta(api) -> dict:
    """What the page needs to draw the terminal (static layout + ship particulars)."""
    from core.state import build_state
    st = build_state(api)
    return {"start": api.now.isoformat(), "end_h": (api.end - api.now).total_seconds() / 3600,
            "berths": [{"id": b.berth_id, "len": b.length_m, "a": b.start_bay, "b": b.end_bay} for b in st.berths],
            "blocks": [{"id": b.block_id, "type": b.block_type, "slots": b.slots} for b in st.blocks],
            "cranes": [c.crane_id for c in st.cranes], "trucks": len(st.trucks),
            "ships": [{"id": s.call_id, "name": s.vessel, "loa": s.loa_m, "dis": s.discharge_remaining,
                       "load": s.load_remaining, "eta": s.scheduled_eta.isoformat() if s.scheduled_eta else None}
                      for s in st.ships]}


# --------------------------------------------------------------------------- one strategy, one process

KPI_KEYS = ("vessels_departed", "vessels_total", "avg_turnaround_h", "fcfs_baseline_avg_turnaround_h_same_calls",
            "crane_moves_per_hour", "truck_idle_hours", "placements_into_blocks_over_85pct", "import_boxes_delivered")


def write_kpis(api, path: Path, sim_h: float) -> None:
    """One line of the simulator's run_summary() (read-only) for the page. Never raises."""
    try:
        rs = api.run_summary()
        rec = {k: rs.get(k) for k in KPI_KEYS}
        rec.update(t=round(sim_h, 3), violations=len(rs.get("violations") or []))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def worker(key: str, agent: str, base: str, data: str, keyfile: Optional[str], knobs: dict, run_dir: str,
           cmd_path: str, llm_on: bool, inbox: str, rate: float, progress=None, slot: int = 0) -> None:
    os.chdir(ROOT)
    os.environ["OPERATOR_INBOX"] = inbox
    run_dir_p = Path(run_dir)
    run_dir_p.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        import run as runner
        from core.loop import Loop
        api = build_api(base, Path(data), Path(keyfile) if keyfile else None, knobs, run_dir_p / "actions.jsonl")
        if agent == "multi" and llm_on:
            runner.llm_client.configure()
            runner.llm_client.warmup()
        else:
            runner.llm_client.configure(enabled=False)
        runner.llm_client.new_scenario(run_dir_p / "llm_calls.jsonl")

        # Pace by simulated time so every strategy stays side by side, and pick up the judges' commands.
        start = api.now
        pace = {"rate": rate, "wall0": time.monotonic(), "sim0": 0.0, "off": 0, "kpi_next": 0.0}
        kpi_path = run_dir_p / "kpis.jsonl"
        orig_step = api._step

        def step(until):
            orig_step(until)
            try:
                with open(cmd_path, "rb") as f:
                    f.seek(pace["off"])
                    data_ = f.read()
                pace["off"] += len(data_)
                for line in data_.decode("utf-8").splitlines():
                    cmd = json.loads(line)
                    if cmd["op"] == "break":
                        s = pd.Timestamp(cmd["at"])
                        for eq in cmd["ids"]:
                            api.down.setdefault(eq, []).append((s, s + pd.Timedelta(hours=float(cmd["hours"]))))
                    elif cmd["op"] == "rate":
                        sim_h = (until - start).total_seconds() / 3600
                        pace.update(rate=float(cmd["value"]), wall0=time.monotonic(), sim0=sim_h)
            except OSError:
                pass
            sim_h = (until - start).total_seconds() / 3600
            if sim_h >= pace["kpi_next"]:               # the simulator's own KPI summary, every half hour
                pace["kpi_next"] = sim_h + 0.5
                write_kpis(api, kpi_path, sim_h)
            if progress is not None:                    # lockstep: never more than half an hour ahead of the slowest
                progress[slot] = sim_h
                t_wait = time.monotonic()
                while sim_h > min(progress[:]) + 0.5 and time.monotonic() - t_wait < 20:
                    time.sleep(0.02)
            if pace["rate"] > 0:
                wait = pace["wall0"] + (sim_h - pace["sim0"]) / pace["rate"] - time.monotonic()
                if wait > 0:
                    time.sleep(min(wait, 2.0))

        api._step = step
        summary = Loop(api, run_dir_p, agents=runner.make_agents(agent), coordinate=agent == "multi").run()
        (run_dir_p / "summary.json").write_text(json.dumps(summary, default=str, indent=1), encoding="utf-8")
    except Exception as e:                                  # the page shows it; the other strategies carry on
        (run_dir_p / "error.txt").write_text(f"{type(e).__name__}: {e}"[:500], encoding="utf-8")
    finally:
        if progress is not None:
            progress[slot] = 1e9                         # finished: never hold the others back


# --------------------------------------------------------------------------- the session

class Session:
    def __init__(self, cfg: dict, data: Path, keyfile: Optional[Path]):
        self.id = uuid.uuid4().hex[:8]
        self.dir = LIVE_DIR / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.systems = [k for k in SYSTEMS if k in (cfg.get("systems") or ["multi", "fcfs"])] or ["multi"]
        self.meta: Optional[dict] = None
        self.meta_error: Optional[str] = None
        self.start = None
        self.trucks: list = []
        self.inbox = self.dir / "operator_inbox.txt"
        self.inbox.write_text("", encoding="utf-8")
        self.procs: dict = {}
        self.stopped = False
        self.progress = mp.get_context("spawn").Array("d", [0.0] * len(self.systems), lock=False)
        threading.Thread(target=self._prepare, args=(cfg, data, keyfile), daemon=True).start()

    def _spawn(self, cfg: dict, data: Path, keyfile: Optional[Path]) -> None:
        ctx = mp.get_context("spawn")
        for slot, k in enumerate(self.systems):
            d = self.dir / k
            d.mkdir(parents=True, exist_ok=True)
            (d / "commands.jsonl").write_text("", encoding="utf-8")
            inbox = str(self.inbox) if k == "multi" else str(d / "no_inbox.txt")
            p = ctx.Process(target=worker, daemon=True,
                            args=(k, SYSTEMS[k][1], cfg["base"], str(data), str(keyfile) if keyfile else None,
                                  cfg.get("knobs") or {}, str(d), str(d / "commands.jsonl"),
                                  bool(cfg.get("llm")) and k == "multi", inbox, float(cfg.get("rate", 2.0)),
                                  self.progress, slot))
            if self.stopped:
                return
            p.start()
            self.procs[k] = p

    def _prepare(self, cfg: dict, data: Path, keyfile: Optional[Path]) -> None:
        """Build the scenario here to describe it to the page (this also leaves the snapshot every run then
        loads), then start the runs."""
        try:
            with _CACHE_LOCK:
                api = build_api(cfg["base"], data, keyfile, cfg.get("knobs") or {})
            meta = scenario_meta(api)
            meta.update(systems=[{"key": k, "label": SYSTEMS[k][0]} for k in self.systems], llm=bool(cfg.get("llm")),
                        base=cfg["base"], knobs=cfg.get("knobs") or {})
            self.start = api.now
            self.trucks = sorted(str(x) for x in api.trucks.index)
            self._spawn(cfg, data, keyfile)
            self.meta = meta
        except Exception as e:
            self.meta_error = f"{type(e).__name__}: {e}"[:300]

    def command(self, cmd: dict) -> None:
        line = json.dumps(cmd) + "\n"
        for k in self.systems:
            with open(self.dir / k / "commands.jsonl", "a", encoding="utf-8") as f:
                f.write(line)

    def latest_now(self):
        """The furthest simulated time any strategy has reached (from the last line of ticks.jsonl)."""
        import pandas as pd
        best = None
        for k in self.systems:
            p = self.dir / k / "ticks.jsonl"
            try:
                with open(p, "rb") as f:
                    f.seek(max(0, p.stat().st_size - 20000))
                    tail = f.read().decode("utf-8", "replace").strip().splitlines()
                rec = json.loads(tail[-1])
                now = pd.Timestamp(rec["now"])
                best = now if best is None else max(best, now)
            except (OSError, ValueError, IndexError, KeyError):
                pass
        return best

    def stop(self) -> None:
        self.stopped = True
        for p in list(self.procs.values()):
            if p.is_alive():
                p.terminate()


STATE: dict = {"session": None, "data": DEFAULT_DATA, "key": None, "uploads": {}}
UPLOAD_DIR = ROOT / "runs" / "live_uploads"
MAX_UPLOAD = 40 * 1024 * 1024
TERMINAL_SHEETS = ("Berths", "Quay_Cranes", "Yard_Trucks", "Yard_Blocks", "Truck_Travel_Times")
SCEN_SHEETS = ("Vessels", "Events", "Manifest", "Baseline_FCFS")
NEED = {"Vessels": ("scenario", "call_id", "vessel_name", "loa_m", "draft_m", "scheduled_eta", "actual_arrival",
                    "discharge_moves", "load_moves"),
        "Manifest": ("scenario", "container_id", "call_id", "direction")}


def data_for(base: str) -> Path:
    """The workbook a scenario lives in: an uploaded one, or the normal data workbook."""
    return STATE["uploads"].get(base, STATE["data"])


def import_workbook(raw: bytes, filename: str) -> dict:
    """Read a dropped workbook laid out like the practice data and make it runnable here.

    It may hold Practice_* sheets (like the data workbook) or Hidden_* sheets (like the answer key);
    the terminal sheets (berths, cranes, trucks, yard) are taken from it when present, else from the
    normal data workbook. It is rewritten as a practice-style workbook under runs/live_uploads/, and
    every scenario in it is built once to check it runs. Returns {"added": [...], "errors": [...]}."""
    import io
    import pandas as pd
    try:
        book = pd.read_excel(io.BytesIO(raw), sheet_name=None)
    except Exception as e:
        return {"added": [], "errors": [f"Could not read {filename} as an Excel workbook ({type(e).__name__})."]}
    prefix = next((p for p in ("Practice_", "Hidden_") if p + "Vessels" in book), None)
    if prefix is None:
        return {"added": [], "errors": ["No Practice_Vessels or Hidden_Vessels sheet: the workbook needs the same sheets as the practice data."]}
    errors: list[str] = []
    for part, cols in NEED.items():
        df = book.get(prefix + part)
        if df is None:
            errors.append(f"Sheet {prefix + part} is missing.")
        else:
            miss = [c for c in cols if c not in df.columns]
            if miss:
                errors.append(f"Sheet {prefix + part} lacks column(s): {', '.join(miss)}.")
    if errors:
        return {"added": [], "errors": errors}
    base = pd.read_excel(STATE["data"], sheet_name=list(TERMINAL_SHEETS) + ["Practice_Events", "Practice_Baseline_FCFS"])
    out = {n: book[n] if n in book else base[n] for n in TERMINAL_SHEETS}
    taken = set(scenarios())
    rename = {}
    for name in sorted({str(x) for x in book[prefix + "Vessels"]["scenario"].dropna()}):
        new = name
        if new.startswith("hidden") or new in taken:          # the simulator reads hidden_* from the answer key
            new = "new_" + new
        while new in taken:
            new += "b"
        rename[name] = new
    for part in SCEN_SHEETS:
        df = book.get(prefix + part)
        if df is None:                                        # no events / no baseline: empty, same columns as practice
            df = base["Practice_" + part].iloc[0:0].copy()
        df = df.copy()
        df["scenario"] = df["scenario"].astype(str).map(lambda s: rename.get(s, s))
        out["Practice_" + part] = df[df["scenario"].isin(rename.values())]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(filename).stem)[:40] or "scenario"
    path = UPLOAD_DIR / f"{stem}-{uuid.uuid4().hex[:6]}.xlsx"
    with pd.ExcelWriter(path) as w:
        for n, df in out.items():
            df.to_excel(w, sheet_name=n, index=False)
    added = []
    for old, new in rename.items():
        try:
            api = base_api(new, path, None)
            v = api.calls.values()
            ev = api.events["event_type"].value_counts().to_dict() if len(api.events) else {}
            STATE["uploads"][new] = path
            added.append({"id": new, "from": old, "file": filename, "ships": len(api.calls),
                          "moves": int(sum(c["discharge_moves"] + c["load_moves"] for c in v)),
                          "events": {str(k): int(n) for k, n in ev.items()}})
        except Exception as e:
            errors.append(f"Scenario {old} does not run: {type(e).__name__}: {e}"[:300])
    return {"added": added, "errors": errors}


def reload_uploads() -> None:
    """Scenarios dropped in earlier stay available after a restart."""
    import pandas as pd
    for f in sorted(UPLOAD_DIR.glob("*.xlsx")) if UPLOAD_DIR.exists() else []:
        try:
            for s in sorted({str(x) for x in pd.read_excel(f, sheet_name="Practice_Vessels", usecols=["scenario"])["scenario"].dropna()}):
                STATE["uploads"].setdefault(s, f)
        except Exception:
            pass


def template_workbook() -> bytes:
    """practice_01 as a stand-alone workbook named my_scenario: a starting point to edit."""
    import io
    import pandas as pd
    src = pd.read_excel(STATE["data"], sheet_name=["Practice_" + p for p in SCEN_SHEETS])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        for n, df in src.items():
            df = df[df["scenario"] == "practice_01"].copy()
            df["scenario"] = "my_scenario"
            df.to_excel(w, sheet_name=n, index=False)
    return buf.getvalue()
LOCK = threading.Lock()


def scenarios() -> list[str]:
    out = [f"practice_0{i}" for i in range(1, 6)]
    if STATE["key"]:
        try:
            import pandas as pd
            col = pd.read_excel(STATE["key"], sheet_name="Hidden_Vessels", usecols=["scenario"])["scenario"]
            out += sorted({str(x) for x in col.dropna()})
        except Exception:
            pass
    return out + [k for k in STATE["uploads"] if k not in out]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code: int, obj) -> None:
        raw = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            raw = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/api/template":
            try:
                raw = template_workbook()
            except Exception as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"[:300]})
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            self.send_header("Content-Disposition", 'attachment; filename="VerteX_scenario_template.xlsx"')
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/api/setup":
            self._json(200, {"scenarios": scenarios(), "uploaded": sorted(STATE["uploads"]), "systems": [{"key": k, "label": v[0]} for k, v in SYSTEMS.items()],
                             "cranes": CRANES, "llm_configured": bool(os.environ.get("LLM_BASE_URL") or (ROOT / ".env").exists())})
        elif self.path.startswith("/api/stream"):
            self.stream()
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/upload":
            n = int(self.headers.get("Content-Length") or 0)
            if not n or n > MAX_UPLOAD:
                return self._json(400, {"error": "send one .xlsx file under 40 MB"})
            raw = self.rfile.read(n)
            from urllib.parse import unquote
            name = unquote(self.headers.get("X-Filename") or "scenario.xlsx")
            try:
                with LOCK:
                    res = import_workbook(raw, name)
            except Exception as e:
                res = {"added": [], "errors": [f"{type(e).__name__}: {e}"[:300]]}
            return self._json(200 if res["added"] else 400, {**res, "scenarios": scenarios()})
        try:
            body = self._body()
        except ValueError:
            return self._json(400, {"error": "bad JSON"})
        s: Optional[Session] = STATE["session"]
        if self.path == "/api/run":
            if body.get("base") not in scenarios():
                return self._json(400, {"error": "unknown scenario"})
            with LOCK:
                if s:
                    s.stop()
                try:
                    STATE["session"] = Session(body, data_for(body["base"]), STATE["key"])
                except Exception as e:
                    return self._json(500, {"error": f"{type(e).__name__}: {e}"[:300]})
                return self._json(200, {"id": STATE["session"].id})
        if s is None:
            return self._json(409, {"error": "no run is going"})
        if self.path == "/api/break":
            import pandas as pd
            ids = [c for c in body.get("cranes", []) if c in CRANES]
            n_tr = max(0, min(20, int(body.get("trucks", 0))))
            ids += s.trucks[:n_tr]
            now = s.latest_now()
            if now is None or s.start is None:
                return self._json(409, {"error": "the runs have not started yet; try again in a few seconds"})
            at = now + pd.Timedelta(minutes=30)
            hours = max(0.5, min(24.0, float(body.get("hours", 4))))
            s.command({"op": "break", "ids": ids, "at": at.isoformat(), "hours": hours})
            return self._json(200, {"at_h": (at - s.start).total_seconds() / 3600, "ids": ids, "hours": hours})
        if self.path == "/api/rate":
            s.command({"op": "rate", "value": max(0.0, min(50.0, float(body.get("rate", 2))))})
            return self._json(200, {"ok": True})
        if self.path == "/api/operator":
            text = " ".join(str(body.get("text", "")).split())[:300]
            if text:
                with open(s.inbox, "a", encoding="utf-8") as f:
                    f.write(text + "\n")
            return self._json(200, {"ok": bool(text)})
        if self.path == "/api/stop":
            s.stop()
            return self._json(200, {"ok": True})
        self._json(404, {"error": "not found"})

    def stream(self):
        """Server-sent events: every new line the runs write, as it is written."""
        s: Optional[Session] = STATE["session"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if s is None:
            return
        files = {(k, n): 0 for k in s.systems for n in ("ticks.jsonl", "negotiation.jsonl", "decisions.jsonl", "llm_calls.jsonl", "kpis.jsonl")}
        done: set = set()
        try:
            while s.meta is None and s.meta_error is None and STATE["session"] is s:
                self.wfile.write(b": preparing\n\n")
                self.wfile.flush()
                time.sleep(0.3)
            if s.meta is None:
                self.wfile.write(f"event: fail\ndata: {json.dumps(s.meta_error or 'stopped')}\n\n".encode())
                return
            self.wfile.write(f"event: meta\ndata: {json.dumps(s.meta, default=str)}\n\n".encode())
            while STATE["session"] is s:
                out = []
                for (k, n), off in files.items():
                    p = s.dir / k / n
                    try:
                        if p.stat().st_size <= off:
                            continue
                        with open(p, "rb") as f:
                            f.seek(off)
                            chunk = f.read()
                    except OSError:
                        continue
                    end = chunk.rfind(b"\n")
                    if end < 0:
                        continue
                    files[(k, n)] = off + end + 1
                    kind = n.split(".")[0]
                    for line in chunk[:end].decode("utf-8", "replace").splitlines():
                        if line.strip():
                            out.append(f'{{"sys":"{k}","kind":"{kind}","rec":{line}}}')
                for k in s.systems:
                    if k in done:
                        continue
                    sm, er = s.dir / k / "summary.json", s.dir / k / "error.txt"
                    if er.exists() or (sm.exists() and not s.procs[k].is_alive()):
                        done.add(k)
                        rec = json.loads(sm.read_text(encoding="utf-8")) if sm.exists() else {"error": er.read_text(encoding="utf-8")}
                        out.append(json.dumps({"sys": k, "kind": "done", "rec": rec}, default=str))
                for i in range(0, len(out), 200):
                    self.wfile.write(("data: [" + ",".join(out[i:i + 200]) + "]\n\n").encode())
                if not out:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--key", type=Path, help="answer key, to offer the hidden scenarios too")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args(argv)
    if not a.data.is_file():
        p.error(f"data workbook not found: {a.data}")
    try:
        from llm import client as llm_client
        llm_client.load_dotenv()
    except Exception:
        pass
    STATE.update(data=a.data, key=a.key)
    if LIVE_DIR.exists():
        shutil.rmtree(LIVE_DIR, ignore_errors=True)
    reload_uploads()
    threading.Thread(target=warm, args=(sorted(scenarios(), key=lambda s: s != "practice_02"), a.data, a.key), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{a.port}/"
    print(f"VerteX live demo on {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if STATE["session"]:
            STATE["session"].stop()
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
