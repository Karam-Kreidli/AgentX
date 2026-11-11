#!/usr/bin/env bash
# CORE-9: does a fresh clone run? Clones this repo into a temp folder, makes a fresh venv, installs
# requirements.txt, copies the workbook in and runs the scenarios the way the README says.
#
#   bash tools/clean_clone_check.sh WORKBOOK [ANSWER_KEY]
#
#   WORKBOOK     P2_Container_Terminal_Data.xlsx (never committed, so it is copied in)
#   ANSWER_KEY   optional. Also runs every hidden scenario in it: --all-hidden --key KEY.
#                Make a mock key with:  python tools/perturb.py --hidden-key
#
# Checks (any failure -> exit 1):
#   - practice_01 --no-llm ends with exit 0, a summary.json, 0 violations and 0 crashes, in under 5 min
#   - runs/P2/practice_01/actions.jsonl (the toolkit-default path) exists and is not empty
#   - with a key: every hidden scenario does the same
#   - with a key: the answer key and workbook are not written anywhere inside the clone's tracked files
#
# Settings (environment variables):
#   CLONE_SOURCE=<path or url>   what to clone (default: this repo's folder; only COMMITTED work is tested,
#                                so commit or stash first. The script says which commit it tested)
#   CLONE_REF=<branch or tag>    check this ref out after cloning (default: the source's current branch)
#   PYTHON_BIN=python3.11        interpreter for the venv (default: python3.11, then python3)
#   KEEP=1                       keep the temp folder (default: delete it at the end)
#   RUN_LLM_OFF_TOO=0            skip the "endpoint unreachable" run (see below). Default 1
#
# The last run sets LLM_BASE_URL to a closed port and runs WITHOUT --no-llm on practice_01. It proves
# the automatic fall-back to rules-only (the run must still end with 0 violations and 0 crashes).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)

WB=${1:-}; KEY=${2:-}
if [ -z "$WB" ] || [ ! -f "$WB" ]; then echo "usage: bash tools/clean_clone_check.sh WORKBOOK [ANSWER_KEY]" >&2; exit 2; fi
if [ -n "$KEY" ] && [ ! -f "$KEY" ]; then echo "answer key not found: $KEY" >&2; exit 2; fi
WB=$(cd "$(dirname "$WB")" && pwd)/$(basename "$WB")
[ -n "$KEY" ] && KEY=$(cd "$(dirname "$KEY")" && pwd)/$(basename "$KEY")

SRC=${CLONE_SOURCE:-$HERE}
PYBIN=${PYTHON_BIN:-}
if [ -z "$PYBIN" ]; then
  if command -v python3.11 >/dev/null 2>&1; then PYBIN=python3.11; else PYBIN=python3; fi
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/agentx-clean-clone.XXXXXX")
FAILS=0
cleanup() { if [ "${KEEP:-0}" = 1 ]; then echo "kept: $TMP"; else rm -rf "$TMP"; fi; }
trap cleanup EXIT
fail() { echo "FAIL: $1"; FAILS=$((FAILS + 1)); }
ok()   { echo "ok:   $1"; }

echo "=== clone $SRC"
git clone --quiet "$SRC" "$TMP/repo" || { echo "clone failed"; exit 1; }
cd "$TMP/repo" || exit 1
if [ -n "${CLONE_REF:-}" ]; then git checkout --quiet "$CLONE_REF" || { echo "no such ref: $CLONE_REF"; exit 1; }; fi
echo "tested commit: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"

echo "=== fresh venv + requirements"
"$PYBIN" -m venv "$TMP/venv" || { echo "could not make a venv with $PYBIN"; exit 1; }
PY="$TMP/venv/bin/python"; [ -x "$PY" ] || PY="$TMP/venv/Scripts/python.exe"
"$PY" -m pip install --quiet -r requirements.txt || { echo "pip install failed"; exit 1; }
echo "python: $("$PY" --version 2>&1)"

echo "=== workbook into data/"
mkdir -p data && cp "$WB" data/P2_Container_Terminal_Data.xlsx

# check_run NAME DIR SECONDS: summary.json present, no crash, 0 violations, toolkit-default log, time
check_run() {
  local name=$1 dir=$2 secs=$3
  "$PY" - "$dir/summary.json" "$secs" "$name" <<'PY' || return 1
import json, sys
try:
    s = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError) as e:
    print(f"  {sys.argv[3]}: no readable summary.json ({e})"); sys.exit(1)
bad = []
if s.get("crashed"): bad.append(f"crashed: {s.get('error')}")
if s.get("violations"): bad.append(f"{len(s['violations'])} violations")
loop = s.get("loop") or {}
if loop.get("exceptions"): bad.append(f"{loop['exceptions']} loop exceptions")
if loop.get("breaches"): bad.append(f"{loop['breaches']} breaches")
if loop.get("actions_refused"): bad.append(f"{loop['actions_refused']} refused actions")
secs = float(s.get("wall_s") or sys.argv[2])       # the scenario's own time, else the whole command's
if secs > 300: bad.append(f"{secs} s is over 5 min")
print(f"  {sys.argv[3]}: departed {s.get('vessels_departed')}/{s.get('vessels_total')}, "
      f"honest {s.get('honest_turnaround_h')} h, {secs} s" + ("" if not bad else "  <- " + "; ".join(bad)))
sys.exit(1 if bad else 0)
PY
  [ -s "runs/P2/$name/actions.jsonl" ] || { echo "  no runs/P2/$name/actions.jsonl"; return 1; }
}

timed() { local t0=$SECONDS; "$@"; local rc=$?; ELAPSED=$((SECONDS - t0)); return $rc; }

echo "=== practice_01 --no-llm"
timed "$PY" run.py --scenario practice_01 --no-llm
RC=$?
if [ $RC -eq 0 ] && check_run practice_01 runs/practice_01 "$ELAPSED"; then ok "practice_01 --no-llm"; else fail "practice_01 --no-llm (exit $RC)"; fi

if [ -n "$KEY" ]; then
  echo "=== hidden scenarios with the key"
  timed "$PY" run.py --all-hidden --key "$KEY" --no-llm
  RC=$?
  COUNT=$(ls -d runs/hidden_* 2>/dev/null | wc -l | tr -d " "); PER=$((ELAPSED / (COUNT > 0 ? COUNT : 1)))   # average per scenario
  [ $RC -eq 0 ] || fail "--all-hidden exit $RC"
  N=0
  for d in runs/hidden_*; do
    [ -d "$d" ] || continue
    N=$((N + 1))
    if check_run "$(basename "$d")" "$d" "$PER"; then ok "$(basename "$d")"; else fail "$(basename "$d")"; fi
  done
  [ $N -gt 0 ] || fail "no hidden scenario ran"
  # secrets: nothing from the key or the workbook may sit in a tracked or untracked-but-not-ignored file
  if [ -n "$(git status --porcelain)" ]; then fail "the run left untracked or changed files: $(git status --porcelain | head -3 | tr '\n' ' ')"; else ok "the run left the clone clean"; fi
fi

if [ "${RUN_LLM_OFF_TOO:-1}" = 1 ]; then
  echo "=== LLM endpoint unreachable (fall-back to rules-only)"
  mkdir -p "$TMP/fallback"
  timed env LLM_BASE_URL=http://127.0.0.1:9/v1 LLM_API_KEY=x LLM_MODEL=none "$PY" run.py --scenario practice_01 --runs "$TMP/fallback/runs"
  RC=$?
  if [ $RC -eq 0 ] && "$PY" - "$TMP/fallback/runs/practice_01/summary.json" "$ELAPSED" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
bad = s.get("crashed") or s.get("violations") or (s.get("loop") or {}).get("exceptions")
print(f"  endpoint down: departed {s.get('vessels_departed')}/{s.get('vessels_total')}, {sys.argv[2]} s")
sys.exit(1 if bad or float(sys.argv[2]) > 300 else 0)
PY
  then ok "run ends cleanly with the endpoint down"; else fail "run with the endpoint down (exit $RC)"; fi
fi

echo
if [ $FAILS -eq 0 ]; then echo "CLEAN CLONE: PASS"; exit 0; fi
echo "CLEAN CLONE: FAIL ($FAILS)"; exit 1
