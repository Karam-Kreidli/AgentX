"""Audit log, negotiation transcript and supervisor feed (SAFE-3).

    from core import log
    log.start(run_dir, meta=None)              # once per run; DecisionLog(run_dir) calls it for you
    log.decision(d)                            # one Decision -> decisions.jsonl
    log.note(tick, state, agent, choice, reason, saw=None)   # non-action entries
    log.escalation(agent, issue, default_taken, state=None)
    log.negotiation(request, grant=None, sentence=None)      # CORE-5: one line per request / grant
    log.feed(sentence, subject=None)           # LLM-4: one sentence per re-plan (stretch goal 1)
    log.tick_snapshot(tick, state)             # loop: compact per-tick state for SAFE-7 / the website
    log.update_meta(**fields)                  # add facts to run_meta.json (e.g. backend actually used)
    log.count_missing_reasons(run_dir) -> int  # CORE-6 / CI: must be 0
    log.simulator_log_counts(run_dir) -> dict  # CORE-6 / CI: tool_error and VIOLATION must be 0

Files written per run, in runs/<scenario>/:
    decisions.jsonl      what we saw, the options, the choice, the reason, guard and tool results
    negotiation.jsonl    every Request / Grant as data
    negotiation.md       the same exchange in plain English, grouped by tick
    supervisor_feed.txt  one sentence per re-plan, for a human supervisor
    ticks.jsonl          one compact snapshot per tick (ships, cranes, blocks)
    run_meta.json        when, which commit, which LLM settings (never keys or URLs)

Rules (GROUND_RULES §5):
- Never raises. A failed write is counted and reported once on stderr; the run goes on.
- Before start() every call is a no-op, so agents and the coordinator can be unit
  tested without a run directory.
- Thread-safe: SAFE-4 may run agents on worker threads.
- Secrets are redacted from every string: the values of *_API_KEY / *_TOKEN /
  *_SECRET variables (environment and .env) and the patterns gsk_..., sk-..., Bearer ...
- Untrusted text (vessel names, container ids, operator messages) is data. Invisible
  characters (control, zero-width, bidi overrides) are shown as [U+XXXX] instead of being
  kept or silently dropped. In the one-line files (negotiation.md, supervisor_feed.txt)
  newlines are collapsed and '<' is escaped, so a vessel name can't forge a heading or HTML.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from core.types import AGENT_TOOLS, GRANTED, REFUSED, TRIMMED, Decision

ROOT = Path(__file__).resolve().parents[1]

DECISIONS = "decisions.jsonl"
NEGOTIATION_JSONL = "negotiation.jsonl"
NEGOTIATION_MD = "negotiation.md"
FEED = "supervisor_feed.txt"
TICKS = "ticks.jsonl"
META = "run_meta.json"
RUN_FILES = (DECISIONS, NEGOTIATION_JSONL, NEGOTIATION_MD, FEED, TICKS, META)

MAX_SAW_STR = 200        # strings inside Decision.saw
MAX_TEXT = 500           # any other string in a JSON line
MAX_LINE = 400           # one transcript line
MAX_SENTENCE = 300       # one feed sentence
MAX_ITEMS = 1000         # items kept from any one list
MAX_DEPTH = 6            # nesting kept in a JSON value
REDACTED = "[REDACTED]"

_SECRET_NAME = re.compile(r"(_API_KEY|_KEY|_TOKEN|_SECRET)$")
_SECRET_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
)
# Control characters (except \t \n \r, handled separately), zero-width and bidi controls.
_INVISIBLE = re.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f"
                        "​-‏\u2028\u2029‪-‮⁠-⁤⁦-⁩﻿]")
_WHITESPACE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# Run state
# --------------------------------------------------------------------------

class _Run:
    def __init__(self, run_dir: Path, secrets: tuple[str, ...]):
        self.dir = run_dir
        self.secrets = secrets
        self.lock = threading.Lock()
        self.tick: Optional[int] = None
        self.t: Optional[float] = None
        self.md_tick: object = object()          # last tick header written to negotiation.md
        self.lines = {name: 0 for name in RUN_FILES}
        self.errors = 0
        self._warned: set[str] = set()

    def warn_once(self, name: str, e: BaseException) -> None:
        self.errors += 1
        if name not in self._warned:
            self._warned.add(name)
            _stderr(f"[log] could not write {name}: {type(e).__name__}: {e}")


_run: Optional[_Run] = None


def _stderr(msg: str) -> None:
    try:
        print(msg, file=sys.stderr)
    except Exception:
        pass


def _secret_values() -> tuple[str, ...]:
    """Values of secret-looking variables from the environment and .env (never printed)."""
    found: set[str] = set()
    for name, value in os.environ.items():
        if _SECRET_NAME.search(name) and value and len(value) >= 8:
            found.add(value)
    try:
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            name, sep, value = line.strip().partition("=")
            value = value.strip().strip('"').strip("'")
            if sep and not name.startswith("#") and _SECRET_NAME.search(name.strip()) and len(value) >= 8:
                found.add(value)
    except OSError:
        pass
    return tuple(sorted(found, key=len, reverse=True))   # longest first, so overlaps redact fully


def start(run_dir: Path | str, meta: Optional[Mapping[str, Any]] = None) -> None:
    """Begin a run: (re)create the run files and write run_meta.json. Never raises."""
    global _run
    _run = None
    try:
        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name in (DECISIONS, NEGOTIATION_JSONL, FEED, TICKS):
            (d / name).write_text("", encoding="utf-8")
        (d / NEGOTIATION_MD).write_text(f"# Negotiation transcript · {d.name}\n", encoding="utf-8")
        _run = _Run(d, _secret_values())
        _write_meta(dict(meta or {}), replace=True)
    except Exception as e:                      # e.g. run_dir is a file, or no permission
        _run = None
        _stderr(f"[log] could not start the run log in {run_dir}: {type(e).__name__}: {e}")


def stop() -> None:
    """End the run (later calls are no-ops). Mainly for tests."""
    global _run
    _run = None


def active() -> bool:
    return _run is not None


def current_tick() -> Optional[int]:
    """The tick of the latest decision or snapshot, or None before any."""
    return _run.tick if _run is not None else None


def stats() -> dict:
    """Lines written per file and write errors for the current run."""
    if _run is None:
        return {"active": False}
    return {"active": True, "dir": str(_run.dir), "lines": dict(_run.lines), "errors": _run.errors}


# --------------------------------------------------------------------------
# Cleaning: redaction, invisible characters, truncation, JSON-safe values
# --------------------------------------------------------------------------

def redact(text: str) -> str:
    """Remove secret values and secret-looking tokens from one string."""
    try:
        s = str(text)
        for secret in (_run.secrets if _run else _secret_values()):
            if secret in s:
                s = s.replace(secret, REDACTED)
        if "sk" in s or "earer" in s.lower():      # cheap pre-check before the regexes
            for p in _SECRET_PATTERNS:
                s = p.sub(REDACTED, s)
        return s
    except Exception:
        return REDACTED


def _mark_invisible(s: str) -> str:
    return _INVISIBLE.sub(lambda m: f"[U+{ord(m.group()):04X}]", s)


def _cut(s: str, cap: int) -> str:
    return s if len(s) <= cap else s[: cap - 1] + "…"


def _text(s: Any, cap: int) -> str:
    """A safe string for a JSON line: redacted, invisible characters shown, truncated."""
    return _cut(_mark_invisible(redact(s)), cap)


def one_line(s: Any, cap: int = MAX_LINE) -> str:
    """A safe single line for Markdown / plain-text files."""
    s = _mark_invisible(redact("" if s is None else s))
    s = _WHITESPACE.sub(" ", s).strip().replace("<", "&lt;")
    return _cut(s, cap)


def _plain(obj: Any, cap: int = MAX_TEXT, depth: int = 0) -> Any:
    """JSON-safe copy of anything; never raises."""
    try:
        if obj is None or isinstance(obj, bool):
            return obj
        if isinstance(obj, int):
            return obj
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, str):
            return _text(obj, cap)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if depth >= MAX_DEPTH:
            return "<…>"
        if is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: _plain(getattr(obj, f.name, None), cap, depth + 1) for f in fields(obj)}
        if isinstance(obj, Mapping) or hasattr(obj, "items"):
            return {_text(k, 100): _plain(v, cap, depth + 1) for k, v in list(obj.items())[:MAX_ITEMS]}
        if isinstance(obj, (list, tuple, set, frozenset)):
            items = list(obj)
            if isinstance(obj, (set, frozenset)):
                try:
                    items = sorted(items)
                except TypeError:
                    pass
            out = [_plain(v, cap, depth + 1) for v in items[:MAX_ITEMS]]
            if len(items) > MAX_ITEMS:
                out.append(f"… (+{len(items) - MAX_ITEMS} more)")
            return out
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if hasattr(obj, "item"):                 # numpy scalars
            return _plain(obj.item(), cap, depth + 1)
        return _text(repr(obj), cap)
    except Exception as e:
        return f"<unloggable {type(obj).__name__}: {type(e).__name__}>"


def _append(name: str, text: str) -> None:
    run = _run
    if run is None:
        return
    try:
        with run.lock:
            with (run.dir / name).open("a", encoding="utf-8") as f:
                f.write(text)
            run.lines[name] += text.count("\n")
    except Exception as e:
        run.warn_once(name, e)


def _json_line(name: str, rec: Any) -> None:
    try:
        line = json.dumps(rec, ensure_ascii=False, allow_nan=False)
    except Exception as e:
        line = json.dumps({"unloggable": f"{type(e).__name__}: {e}"[:200]})
    _append(name, line + "\n")


def clock(t: Optional[float]) -> str:
    """Simulated hours since scenario start -> 'Day 2 06:15'."""
    if t is None:
        return "Day ? --:--"
    try:
        minutes = int(round(float(t) * 60))
        return f"Day {minutes // 1440 + 1} {(minutes % 1440) // 60:02d}:{minutes % 60:02d}"
    except Exception:
        return "Day ? --:--"


def _now(tick: Optional[int], t: Optional[float]) -> tuple[Optional[int], Optional[float]]:
    run = _run
    return (run.tick if tick is None and run else tick), (run.t if t is None and run else t)


def _hours(state: Any) -> Optional[float]:
    try:
        from core.state import hours
        return hours(state)
    except Exception:
        return _run.t if _run else None


# --------------------------------------------------------------------------
# decisions.jsonl
# --------------------------------------------------------------------------

def decision(d: Decision) -> None:
    """Write one Decision line. Also remembers its tick and time for later lines."""
    run = _run
    if run is None:
        return
    try:
        if is_dataclass(d):
            rec = {f.name: _plain(getattr(d, f.name, None), MAX_SAW_STR if f.name == "saw" else MAX_TEXT)
                   for f in fields(d)}
        else:
            rec = _plain(d)
        if isinstance(rec.get("tick"), int):
            run.tick = rec["tick"]
        if isinstance(rec.get("t"), (int, float)):
            run.t = rec["t"]
    except Exception as e:
        rec = {"unloggable_decision": f"{type(e).__name__}: {e}"[:200]}
    _json_line(DECISIONS, rec)


def note(tick: int, state: Any, agent: str, choice: str, reason: str,
         saw: Optional[Mapping[str, Any]] = None) -> None:
    """Non-action entries: escalations, exceptions, invariant breaches, mismatches."""
    if _run is None:
        return
    decision(Decision(tick=tick, t=_hours(state), agent=agent, saw=saw or {}, options=(),
                      choice=choice, reason=reason, guard_ok=True, guard_reason="n/a (not an action)",
                      tool_result=None, backend="rules"))


def escalation(agent: str, issue: str, default_taken: str, state: Any = None,
               tick: Optional[int] = None, saw: Optional[Mapping[str, Any]] = None) -> None:
    """Flag an issue for a human. Never blocks: it records the safe default already taken."""
    if _run is None:
        return
    tick, t = _now(tick, _hours(state) if state is not None else None)
    decision(Decision(tick=tick if tick is not None else -1, t=t, agent=agent, saw=saw or {}, options=(),
                      choice="escalate", reason=f"{issue} Safest default taken: {default_taken}",
                      guard_ok=True, guard_reason="n/a (not an action)", tool_result=None, backend="rules"))


# --------------------------------------------------------------------------
# negotiation.jsonl + negotiation.md (CORE-5 writes through this)
# --------------------------------------------------------------------------

def _template(req: Any, grant: Any) -> tuple[str, str]:
    """(speaker, plain-English line) built from the structured fields only."""
    agent = one_line(getattr(req, "agent", "?"), 40)
    res = one_line(getattr(req, "resource", "?"), 40)
    call = one_line(getattr(req, "call_id", "?"), 40)
    amount = getattr(req, "amount", "?")
    if grant is None:
        return f"{agent} → coordinator", f"{call} asks for {amount} {res}. {one_line(getattr(req, 'reason', ''))}"
    status = getattr(grant, "status", "?")
    granted = getattr(grant, "granted", "?")
    why = one_line(getattr(grant, "reason", ""))
    if status == GRANTED:
        text = f"granted {granted} {res} for {call}. {why}"
    elif status == TRIMMED:
        text = f"trimmed {call} from {amount} to {granted} {res}. {why}"
    elif status == REFUSED:
        text = f"refused {amount} {res} for {call}. {why}"
    else:
        text = f"{one_line(status, 20)} {granted}/{amount} {res} for {call}. {why}"
    return f"coordinator → {agent}", text


def negotiation(request: Any = None, grant: Any = None, sentence: Optional[str] = None, *,
                tick: Optional[int] = None, t: Optional[float] = None) -> None:
    """Log one request (grant=None) or one decision on it. `sentence` is LLM-5's wording;
    without it a template built from the fields is used. Numbers always come from the fields."""
    run = _run
    if run is None:
        return
    try:
        req = getattr(grant, "request", None) if grant is not None and request is None else request
        tick, t = _now(tick, t)
        rec = {
            "tick": tick, "t": t, "kind": "request" if grant is None else "grant",
            "agent": _plain(getattr(req, "agent", None)), "resource": _plain(getattr(req, "resource", None)),
            "amount": _plain(getattr(req, "amount", None)), "call_id": _plain(getattr(req, "call_id", None)),
            "priority": _plain(getattr(req, "priority", None)), "reason": _plain(getattr(req, "reason", None)),
            "status": _plain(getattr(grant, "status", None)), "granted": _plain(getattr(grant, "granted", None)),
            "grant_reason": _plain(getattr(grant, "reason", None)),
            "sentence": _plain(sentence) if sentence else None,
        }
        _json_line(NEGOTIATION_JSONL, rec)
        speaker, text = _template(req, grant)
        line = f"- **{speaker}:** {one_line(sentence) if sentence else one_line(text)}\n"
        with run.lock:
            header = run.md_tick != tick
            run.md_tick = tick
        if header:
            line = f"\n### {clock(t)} (tick {tick if tick is not None else '?'})\n" + line
        _append(NEGOTIATION_MD, line)
    except Exception as e:
        run.warn_once(NEGOTIATION_MD, e)


# --------------------------------------------------------------------------
# supervisor_feed.txt (LLM-4 writes through this)
# --------------------------------------------------------------------------

def feed(sentence: str, subject: Optional[str] = None, *,
         tick: Optional[int] = None, t: Optional[float] = None) -> None:
    """One sentence per re-plan: '[Day 2 06:15 | tick 117 | P02-004] Moved QC4 to B3 because ...'."""
    if _run is None:
        return
    tick, t = _now(tick, t)
    tag = f" | {one_line(subject, 40)}" if subject else ""
    _append(FEED, f"[{clock(t)} | tick {tick if tick is not None else '?'}{tag}] {one_line(sentence, MAX_SENTENCE)}\n")


# --------------------------------------------------------------------------
# ticks.jsonl (the loop writes one per tick; SAFE-7 and the demo website read it)
# --------------------------------------------------------------------------

def tick_snapshot(tick: int, state: Any) -> None:
    """Compact per-tick state. No vessel names: the timeline only needs call ids."""
    run = _run
    if run is None:
        return
    try:
        t = _hours(state)
        run.tick, run.t = tick, t
        rec = {
            "tick": tick, "t": t, "now": _plain(getattr(state, "now", None)),
            "ships": [{"id": _plain(s.call_id, 40), "status": s.status, "berth": s.berth,
                       "cranes": list(s.cranes), "trucks": s.trucks,
                       "dis": s.discharge_remaining, "load": s.load_remaining} for s in state.ships],
            "cranes": [{"id": c.crane_id, "status": c.status, "bay": c.bay, "call": c.call_id}
                       for c in state.cranes],
            "blocks": {b.block_id: [b.used, b.slots] for b in state.blocks},
            "holds": len(state.held_ids()),
            "anomalies": len(getattr(state, "anomalies", ()) or ()),
        }
        _json_line(TICKS, _plain(rec, 40))
    except Exception as e:
        run.warn_once(TICKS, e)


# --------------------------------------------------------------------------
# run_meta.json
# --------------------------------------------------------------------------

def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=2)
        return out.stdout.strip() or None
    except Exception:
        return None


def _write_meta(extra: dict, replace: bool = False) -> None:
    run = _run
    if run is None:
        return
    path = run.dir / META
    try:
        with run.lock:
            if replace:
                env = os.environ
                meta: dict = {
                    "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "run": run.dir.name, "git_sha": _git_sha(), "python": sys.version.split()[0],
                    "llm_model": env.get("LLM_MODEL") or None,
                    "llm_configured": bool(env.get("LLM_BASE_URL") and env.get("LLM_API_KEY")),
                    "groq_configured": bool(env.get("GROQ_API_KEY")),
                    "llm_timeout_s": env.get("LLM_TIMEOUT_S") or None,
                    "llm_call_budget": env.get("LLM_CALL_BUDGET") or None,
                }
            else:
                try:
                    meta = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            meta.update(extra)
            path.write_text(json.dumps(_plain(meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            run.lines[META] = 1
    except Exception as e:
        run.warn_once(META, e)


def update_meta(**extra: Any) -> None:
    """Merge facts into run_meta.json, e.g. update_meta(llm_backend_used="local", llm_calls=23)."""
    _write_meta(extra)


# --------------------------------------------------------------------------
# Checks used by the scorecard (CORE-6) and CI (SAFE-6)
# --------------------------------------------------------------------------

def count_missing_reasons(run_dir: Path | str) -> int:
    """State-changing calls that were sent without a reason (must be 0).
    Raises if decisions.jsonl is missing or unreadable: that is a failure in itself."""
    n = 0
    for line in (Path(run_dir) / DECISIONS).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        reason = str(rec.get("reason") or "").strip()
        if rec.get("choice") in AGENT_TOOLS and rec.get("guard_ok") and reason in ("", "(none)"):
            n += 1
    return n


def simulator_log_counts(run_dir: Path | str) -> dict:
    """Counts from the organizers' own actions.jsonl, the file the run is scored from.
    The toolkit README says "Attempting a forbidden action counts as a violation", so a
    tool_error (a call the simulator refused) must be treated as seriously as a VIOLATION:
    both must be 0. Raises if actions.jsonl is missing or unreadable."""
    counts = {"tool_call": 0, "tool_error": 0, "VIOLATION": 0, "tool_error_tools": {}, "violation_types": {}}
    for line in (Path(run_dir) / "actions.jsonl").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        kind = rec.get("action")
        if kind in ("tool_call", "tool_error", "VIOLATION"):
            counts[kind] += 1
        if kind == "tool_error":
            tool = str(rec.get("tool"))
            counts["tool_error_tools"][tool] = counts["tool_error_tools"].get(tool, 0) + 1
        elif kind == "VIOLATION":
            vtype = str(rec.get("type"))
            counts["violation_types"][vtype] = counts["violation_types"].get(vtype, 0) + 1
    return counts


# --------------------------------------------------------------------------
# Compatibility: the executor / loop API from CORE-4
# --------------------------------------------------------------------------

class DecisionLog:
    """decisions.jsonl writer used by core/executor.py and core/loop.py.
    Creating one starts the run log for that directory."""

    def __init__(self, run_dir: Path | str, meta: Optional[Mapping[str, Any]] = None):
        self.path = Path(run_dir) / DECISIONS
        start(run_dir, meta)

    def write(self, d: Decision) -> None:
        decision(d)

    def note(self, tick: int, state: Any, agent: str, choice: str, reason: str,
             saw: Optional[Mapping[str, Any]] = None) -> None:
        note(tick, state, agent, choice, reason, saw)
