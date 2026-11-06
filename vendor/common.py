"""
Shared base for the competition mock APIs (Problems 2-12).

Every problem module defines a class derived from MockAPI. Methods decorated with
@tool become agent tools: they appear in `tool_schemas()` (ready for LLM function
calling) and can be invoked by name with `api.call(name, arguments)`.

Every tool call is appended to a JSON-lines action log with the simulated time,
so organizers can score runs afterwards. Hard-constraint breaches are recorded as
VIOLATION entries and returned by `run_summary()`.
"""
import json, math, inspect
from pathlib import Path
import numpy as np
import pandas as pd


class APIError(Exception):
    pass


def tool(description, params=None, required=None, mutating=False):
    """Mark a method as an agent tool. params = {name: json-schema}."""
    def deco(fn):
        fn._tool = dict(name=fn.__name__, description=description,
                        input_schema=dict(type="object", properties=params or {}, required=required or []))
        fn._mutating = mutating
        return fn
    return deco


def S(t, desc=None, **kw):
    """JSON-schema property helper: S('string', 'what it is')."""
    d = dict(type=t, **kw)
    if desc:
        d["description"] = desc
    return d


def clean(x):
    """Make values JSON-safe (timestamps -> ISO strings, NaN -> None, numpy -> python)."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return [clean(r) for r in x.to_dict(orient="records")]
    if isinstance(x, pd.Series):
        return clean(x.to_dict())
    if x is None or x is pd.NaT:
        return None
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(x) else pd.Timestamp(x).isoformat()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else round(float(x), 4)
    return x


def sheets(path, *names):
    return {n: pd.read_excel(path, sheet_name=n) for n in names}


class MockAPI:
    PROBLEM = "?"

    def __init__(self, scenario, log_path=None, start_time=None, end_time=None):
        self.scenario = scenario
        self.now = pd.Timestamp(start_time) if start_time is not None else None
        self.end = pd.Timestamp(end_time) if end_time is not None else None
        self.log_path = Path(log_path or f"runs/{self.PROBLEM}/{scenario}/actions.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("")
        self.violations = []
        self._log("session_start", problem=self.PROBLEM, scenario=scenario)

    # ------------------------------------------------------------- logging
    def _log(self, action, **kw):
        kw.pop("sim_time", None)
        rec = dict(action=action, sim_time=self.now.isoformat() if self.now is not None else None, **clean(kw))
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _violation(self, kind, **kw):
        v = dict(type=kind, sim_time=self.now.isoformat() if self.now is not None else None, **clean(kw))
        self.violations.append(v)
        self._log("VIOLATION", **v)

    # ------------------------------------------------------------- dispatch
    @classmethod
    def tool_schemas(cls):
        out = []
        for name in dir(cls):
            m = getattr(cls, name, None)
            if callable(m) and hasattr(m, "_tool"):
                out.append(m._tool)
        return out

    def call(self, name, arguments=None):
        """Dispatch a tool call by name (for LLM loops). Returns a JSON-safe result or {'error': ...}."""
        if name not in {t["name"] for t in self.tool_schemas()}:
            return dict(error=f"unknown tool {name}")
        try:
            return clean(getattr(self, name)(**(arguments or {})))
        except (APIError, TypeError, KeyError, ValueError) as e:
            self._log("tool_error", tool=name, arguments=arguments, error=str(e))
            return dict(error=str(e))

    def __getattribute__(self, item):
        attr = object.__getattribute__(self, item)
        meta = getattr(attr, "_tool", None)
        if meta is not None and callable(attr):
            log = object.__getattribute__(self, "_log")
            def wrapped(*a, **kw):
                res = clean(attr(*a, **kw))
                log("tool_call", tool=item, arguments=kw if kw else list(a), result=res if attr._mutating else "(read)")
                return res
            return wrapped
        return attr

    def run_summary(self):
        """Organizer helper: KPIs and violations for the run (modules extend this)."""
        return clean(dict(problem=self.PROBLEM, scenario=self.scenario, violations=self.violations))


class ClockMixin:
    """Simulated clock tools for time-based problems. Subclasses implement _step(until)."""
    MAX_ADVANCE_MIN = 240

    @tool("Current simulated time and the end of the scenario.")
    def get_current_time(self):
        return dict(now=self.now, scenario_end=self.end, finished=self.now >= self.end)

    @tool("Advance the simulated clock (1-240 minutes). New data and events become visible; the simulation runs forward.",
          dict(minutes=S("integer", "Minutes to advance")), ["minutes"], mutating=True)
    def advance_time(self, minutes):
        minutes = int(minutes)
        if not 1 <= minutes <= self.MAX_ADVANCE_MIN:
            raise APIError(f"minutes must be between 1 and {self.MAX_ADVANCE_MIN}")
        target = min(self.now + pd.Timedelta(minutes=minutes), self.end)
        self._step(target)
        self.now = target
        return dict(now=self.now, finished=self.now >= self.end)

    def _step(self, until):
        pass

    def _visible(self, df, col):
        return df[pd.to_datetime(df[col]) <= self.now]
