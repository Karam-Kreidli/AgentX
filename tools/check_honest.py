"""Honest-play check (SAFE-6): GROUND_RULES §3 and the CLAUDE.md hard limits, on our own code.

    python tools/check_honest.py                 # agent code, names from the workbooks found
    python tools/check_honest.py --data X.xlsx   # also take vessel names from this workbook

Reads our Python source with `ast`, so comments and docstrings are ignored, and fails on:

  ids        a call id (P01-001), scenario name (practice_02), container id (UFCU4182103)
             or vessel name in a string: no logic may be keyed on them
  workbook   the workbook, its Practice_/Hidden_ sheets, the answer key or a data/ path
  toolkit    importing the toolkit (vendor, p2_terminal_api, common) or an Excel reader
  simulator  a simulator tool called outside core/executor.py (any tool), core/state.py
             (read tools) and core/loop.py (advance_time)
  internals  reading simulator internals (api.events, api.down, api.holds, ...): they show
             the future
  override   an override, admin mode or guard switch (GROUND_RULES §4: there is none)

Scope: every .py file of ours, except tests/, tools/, vendor/, toolskit/, ops/, runs/, data/
and virtualenvs. run.py is the harness: it imports the toolkit and opens the workbook for
it, and its help text names scenarios, so the scenario, workbook and toolkit rules skip it.

A line may be let through with `# honest-play: ok <reason>`. Every such line is printed,
so the reviewer sees it. Vessel names are also read from the workbooks (this check may
read them; agent code may not). Exit code 1 if anything is found.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOKS = (ROOT / "data" / "P2_Container_Terminal_Data.xlsx",
                     ROOT / "toolskit" / "data" / "P2_Container_Terminal_Data.xlsx")

SKIP_TOP = {"tests", "tools", "vendor", "toolskit", "ops", "runs", "data", "mydocs", "context"}
SKIP_ANY = {"__pycache__", "node_modules", "venv", "env", "site-packages"}
HARNESS = {"run.py"}                        # skips the scenario, workbook and toolkit rules
# Offline harnesses that build their own simulator, like run.py; agent code never imports them.
SKIP_PREFIX = ("llm/bakeoff/",)             # LLM-3 bake-off: collects decision points from runs

READ_TOOLS = {"get_current_time", "get_vessel_schedule", "get_equipment_status", "get_yard_status",
              "get_customs_status", "get_manifest"}
WRITE_TOOLS = {"assign_berth", "plan_crane_sequence", "dispatch_truck", "place_container",
               "release_to_consignee", "advance_time"}
TOOLS = READ_TOOLS | WRITE_TOOLS
# Which file may call which tool through api.call(...); "*" = also with a computed tool name.
CALL_ALLOWED = {"core/executor.py": TOOLS | {"*"}, "core/state.py": READ_TOOLS | {"*"},
                "core/loop.py": {"advance_time"}}
API_NAMES = {"api", "_api", "sim", "terminal"}            # how our code names the simulator
API_PUBLIC = {"call", "scenario", "violations", "run_summary", "tool_schemas", "PROBLEM"}

CALL_ID = re.compile(r"\b[A-Z]\d{2}-\d{3}\b")                          # P01-001, H02-004
SCENARIO = re.compile(r"(?i)\b(?:practice|hidden)_\d{2}\b")            # practice_02, hidden_01
CONTAINER = re.compile(r"\b[A-Z]{3}[UJZ]\d{7}\b")                      # ISO 6346: UFCU4182103
VESSEL = re.compile(r"\bM/?[VT]\.?\s+[A-Z][a-z]+")                      # MV Desert Spirit
WORKBOOK = re.compile(r"(?i)\.xls[xmb]?\b|answer[ _-]?key|\b(?:practice|hidden)_(?:vessels|events|manifest)\b"
                      r"|baseline_fcfs|vessel_calls_history|container_manifest|customs_holds_history"
                      r"|equipment_failures|(?:^|[\\/])data[\\/]")
EXCEL_READERS = {"read_excel", "ExcelFile", "load_workbook"}
TOOLKIT_MODULES = {"vendor", "p2_terminal_api", "common", "toolskit"}
EXCEL_MODULES = {"openpyxl", "xlrd", "xlsxwriter", "pyxlsb"}
NAME_PREFIX = re.compile(r"(?i)^m/?[vt]\.?\s+")            # "MV Desert Spirit" is also matched as "Desert Spirit"
OVERRIDE = re.compile(r"(?i)^(?:.*_)?(?:admin_?mode|god_?mode|unsafe_?mode|(?:override|bypass|skip|disable|no)_?guard"
                      r"|guard_?(?:off|override|bypass|disabled?))(?:_.*)?$")
PRAGMA = re.compile(r"#\s*honest-play:\s*ok\b(.*)$")


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


# --------------------------------------------------------------------------
# Which files
# --------------------------------------------------------------------------

def agent_files(root: Path) -> list[Path]:
    """Our Python files, minus tests, tools, the toolkit and anything that isn't agent code."""
    out = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).parts
        if rel[0] in SKIP_TOP or any(part in SKIP_ANY or part.startswith(".") for part in rel[:-1]):
            continue
        if "/".join(rel).startswith(SKIP_PREFIX):
            continue
        out.append(p)
    return out


# --------------------------------------------------------------------------
# Scanning one file
# --------------------------------------------------------------------------

def _docstrings(tree: ast.AST) -> set[int]:
    """ids of string statements that do nothing (docstrings, and strings used as comments)."""
    return {id(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}


def _is_api(node: ast.AST) -> bool:
    """`api`, `self.api`, `self._api`, `sim`, ... : an expression naming the simulator."""
    if isinstance(node, ast.Name):
        return node.id in API_NAMES
    return isinstance(node, ast.Attribute) and node.attr in API_NAMES


def _snip(s: str, n: int = 60) -> str:
    s = repr(s)
    return s if len(s) <= n else s[:n - 3] + "..."


class _Scanner(ast.NodeVisitor):
    def __init__(self, rel: str, names: set[str]):
        self.rel = rel
        self.harness = rel in HARNESS
        self.names = names
        self.found: list[Finding] = []
        self.skip: set[int] = set()

    def add(self, node: ast.AST, rule: str, message: str) -> None:
        self.found.append(Finding(self.rel, getattr(node, "lineno", 0), rule, message))

    # strings ---------------------------------------------------------------
    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and id(node) not in self.skip:
            self.check_string(node, node.value)

    def check_string(self, node: ast.AST, s: str) -> None:
        for rx, what in ((CALL_ID, "call id"), (CONTAINER, "container id"), (VESSEL, "vessel name")):
            m = rx.search(s)
            if m:
                self.add(node, "ids", f"{what} {m.group()!r} in a string: no logic may be keyed on it")
        low = s.lower()
        for name in self.names:
            if name in low:
                self.add(node, "ids", f"vessel name {name!r} (from the workbook) in a string")
                break
        if not self.harness:
            m = SCENARIO.search(s)
            if m:
                self.add(node, "ids", f"scenario name {m.group()!r} in a string: no logic may be keyed on it")
            m = WORKBOOK.search(s)
            if m:
                self.add(node, "workbook", f"workbook / data reference {m.group()!r} in {_snip(s)}: "
                                           "agents see the world only through the read tools")
        if (re.fullmatch(r"[A-Z0-9_]+", s) or re.fullmatch(r"-{1,2}[a-z0-9-]+", s)) \
                and OVERRIDE.match(s.lstrip("-").replace("-", "_")):
            self.add(node, "override", f"{s!r} looks like a guard switch: there is no override")

    # imports -----------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self.check_module(node, a.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            self.check_module(node, node.module)
        for a in node.names:
            if a.name in EXCEL_READERS:
                self.add(node, "workbook", f"imports the Excel reader {a.name}")

    def check_module(self, node: ast.AST, module: str) -> None:
        top = module.split(".")[0]
        if top in EXCEL_MODULES:
            self.add(node, "workbook", f"imports the Excel library {module}: agents never open the workbook")
        elif top in TOOLKIT_MODULES and not self.harness:
            self.add(node, "toolkit", f"imports {module}: only run.py builds the simulator; agents get a WorldState")

    # names, attributes, calls -----------------------------------------------------
    def check_identifier(self, node: ast.AST, name: Optional[str]) -> None:
        if name and OVERRIDE.match(name):
            self.add(node, "override", f"identifier {name!r} looks like a guard switch: there is no override")

    def visit_Name(self, node: ast.Name) -> None:
        self.check_identifier(node, node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self.check_identifier(node, node.arg)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        self.check_identifier(node, node.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.check_identifier(node, node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.check_identifier(node, node.name)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.check_identifier(node, node.attr)
        if node.attr in EXCEL_READERS:
            self.add(node, "workbook", f"uses the Excel reader {node.attr}")
        if _is_api(node.value) and node.attr not in API_PUBLIC and node.attr not in TOOLS:
            self.add(node, "internals", f"reads simulator internals .{node.attr}: use the read tools "
                                        "(core/state.py); internals show future events")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "call" and _is_api(f.value):
            self.check_tool_call(node)
        elif isinstance(f, ast.Attribute) and f.attr in TOOLS and (
                _is_api(f.value) or f.attr in WRITE_TOOLS - {"advance_time"}):
            self.add(node, "simulator", f"calls the simulator tool {f.attr}() directly: "
                                        "state changes go through core/executor.py only")
        elif isinstance(f, ast.Name) and f.id == "getattr" and len(node.args) >= 2 and _is_api(node.args[0]) \
                and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str) \
                and node.args[1].value not in API_PUBLIC:
            self.add(node, "internals", f"reads simulator internals getattr(api, {node.args[1].value!r})")
        self.generic_visit(node)

    def check_tool_call(self, node: ast.Call) -> None:
        allowed = CALL_ALLOWED.get(self.rel, set())
        arg = node.args[0] if node.args else None
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if arg.value not in allowed:
                where = "core/executor.py" if arg.value in WRITE_TOOLS else "core/state.py"
                self.add(node, "simulator", f"api.call({arg.value!r}) here: only {where} may call it")
        elif "*" not in allowed:
            self.add(node, "simulator", "api.call() with a computed tool name: only core/executor.py "
                                        "and core/state.py call the simulator")


def scan_file(path: Path, root: Path, names: set[str]) -> tuple[list[Finding], list[Finding]]:
    """(findings, allowed): allowed are findings on lines marked `# honest-play: ok <reason>`."""
    rel = path.relative_to(root).as_posix()
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=rel)
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as e:
        return [Finding(rel, getattr(e, "lineno", 0) or 0, "parse", f"could not read: {type(e).__name__}")], []
    sc = _Scanner(rel, names)
    sc.skip = _docstrings(tree)
    sc.visit(tree)
    lines = text.splitlines()
    found, allowed = [], []
    for f in sc.found:
        m = PRAGMA.search(lines[f.line - 1]) if 0 < f.line <= len(lines) else None
        if m and m.group(1).strip(" -—:"):
            allowed.append(f)
        elif m:
            found.append(Finding(f.path, f.line, f.rule, f.message + " (the honest-play pragma needs a reason)"))
        else:
            found.append(f)
    return found, allowed


def scan(root: Path = ROOT, files: Optional[Iterable[Path]] = None,
         names: Iterable[str] = ()) -> tuple[list[Finding], list[Finding], int]:
    """(findings, allowed, files checked) for the agent code under root."""
    files = list(files) if files is not None else agent_files(root)
    bare = {NAME_PREFIX.sub("", n.strip()) for n in names if n}
    keep = {n.lower() for n in bare if len(n) >= 6}
    found, allowed = [], []
    for p in files:
        f, a = scan_file(p, root, keep)
        found += f
        allowed += a
    return found, allowed, len(files)


def workbook_names(paths: Iterable[Path]) -> tuple[set[str], list[str]]:
    """Vessel names from the workbooks' vessel sheets (practice and history). (names, notes)."""
    names, notes = set(), []
    for p in paths:
        try:
            import pandas as pd
            with pd.ExcelFile(p) as book:
                for sheet in book.sheet_names:
                    if "vessel" in sheet.lower():
                        df = book.parse(sheet, usecols=lambda c: c == "vessel_name")
                        if "vessel_name" in df:
                            names |= {str(v) for v in df["vessel_name"].dropna().unique()}
        except Exception as e:                    # a missing or odd workbook only weakens the check
            notes.append(f"could not read vessel names from {p}: {type(e).__name__}")
    return names, notes


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, action="append", help="workbook to take vessel names from (repeatable)")
    a = p.parse_args(argv)
    books = a.data or [b for b in DEFAULT_WORKBOOKS if b.exists()]
    names, notes = workbook_names(books)
    for n in notes:
        print(f"note: {n}")
    if not books:
        print("note: no workbook found: vessel names are matched by pattern only")
    found, allowed, n_files = scan(ROOT, names=names)
    for f in allowed:
        print(f"allowed  {f}")
    for f in found:
        print(f"FAIL     {f}")
    print(f"honest play: {n_files} files, {len(names)} vessel names from {len(books)} workbook(s); "
          f"{len(found)} finding(s), {len(allowed)} allowed by pragma")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
