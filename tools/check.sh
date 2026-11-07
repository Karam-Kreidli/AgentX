#!/usr/bin/env bash
# CI check (SAFE-6). Run it before you open a PR; GitHub runs it on every PR.
#
#   bash tools/check.sh                    # every step
#   bash tools/check.sh honest secrets     # only these steps
#
# Steps, in this order (any FAIL -> exit code 1):
#   pytest     pytest -m "not llm"
#   scorecard  tools/eval_all.py --all --no-llm --against main, on every workbook. Fails on any
#              violation / refused call / breach, any crash, an action without a reason, a scenario
#              over 5 min, fewer ships departed than main, honest turnaround > main + 3 %
#   redteam    tools/check_redteam.py: hostile vessel names (tests/redteam/) change no action
#   honest     tools/check_honest.py: no IDs, vessel names, workbook reads, simulator calls or
#              guard switches in agent code
#   secrets    tools/check_secrets.py: no keys, .env or workbooks in committable files or run logs
#
# Settings (environment variables):
#   CHECK_BASE=origin/main          the git ref the scorecard compares with (default origin/main,
#                                   else main). A ref that doesn't exist: absolute checks only.
#   CHECK_WORKBOOKS="a.xlsx b.xlsx" workbooks to run (default: the copies in data/ and toolskit/data/
#                                   that exist; F14: the team and website copies differ)
#   CHECK_NO_WORKBOOK_OK=1          no workbook: skip scorecard and redteam instead of failing
#   CHECK_ALLOW_KPI_REGRESSION=1    fewer ships or a slower turnaround than main is a warning, not a
#                                   failure. The PR must explain why (GROUND_RULES §6); CI sets this
#                                   from the PR label "kpi-regression-explained". Violations, crashes
#                                   and missing reasons always fail.
#   PYTHON=/path/to/python          interpreter (default: .venv, then python3, then python)
#
# Output goes to runs/check/ (git-ignored). Works with macOS's bash 3.2.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

ALL_STEPS="pytest scorecard redteam honest secrets"
OUT=runs/check

# ------------------------------------------------------------------ setup
if [ -n "${PYTHON:-}" ]; then PY=$PYTHON
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python
fi

if [ $# -eq 0 ]; then STEPS=$ALL_STEPS; else STEPS="$*"; fi
for s in $STEPS; do
  case " $ALL_STEPS " in *" $s "*) ;; *) echo "unknown step '$s' (steps: $ALL_STEPS)" >&2; exit 2 ;; esac
done

if [ -n "${CHECK_WORKBOOKS:-}" ]; then
  WORKBOOKS=$CHECK_WORKBOOKS
else
  WORKBOOKS=""
  for wb in data/P2_Container_Terminal_Data.xlsx toolskit/data/P2_Container_Terminal_Data.xlsx; do
    [ -f "$wb" ] && WORKBOOKS="$WORKBOOKS $wb"
  done
fi

BASE=${CHECK_BASE:-}
if [ -z "$BASE" ]; then
  if git rev-parse --verify --quiet "origin/main^{commit}" >/dev/null 2>&1; then BASE=origin/main; else BASE=main; fi
fi

group()    { if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::group::$1"; else printf '\n=== %s\n' "$1"; fi; }
endgroup() { if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::endgroup::"; fi; }
warn()     { if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::warning::$1"; else echo "WARNING: $1"; fi; }
label_of() { local d; d=$(dirname "$1"); echo "${d//\//-}"; }      # data/x.xlsx -> data

# 0 = workbooks present, 2 = skip (allowed), 1 = fail
need_workbooks() {
  local wb
  for wb in $WORKBOOKS; do
    if [ ! -f "$wb" ]; then echo "workbook not found: $wb"; return 1; fi
  done
  [ -n "${WORKBOOKS// /}" ] && return 0
  echo "No workbook: put P2_Container_Terminal_Data.xlsx in data/ (CI: ci/workbooks/team.xlsx.gpg + the"
  echo "WORKBOOK_PASSPHRASE secret). Without it nothing proves 0 violations, so this step fails."
  if [ "${CHECK_NO_WORKBOOK_OK:-0}" = 1 ]; then echo "CHECK_NO_WORKBOOK_OK=1: skipped."; return 2; fi
  return 1
}

# true if every failure in an eval_all scorecard is a KPI regression versus main
kpi_only() {
  "$PY" - "$1" <<'PY'
import json, sys
try:
    card = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    sys.exit(1)
fails = [f for fl in (card.get("failures") or {}).values() for f in fl]
sys.exit(0 if fails and all(f.startswith(("ships departed", "honest turnaround")) for f in fails) else 1)
PY
}

summary_md() { if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "$1" >> "$GITHUB_STEP_SUMMARY"; fi; }

# ------------------------------------------------------------------ steps (0 pass, 1 fail, 2 skip, 3 warn)
step_pytest() {
  "$PY" -m pytest -q -m "not llm"
}

step_scorecard() {
  need_workbooks; local rc=$?; [ $rc -ne 0 ] && return $rc
  if [ ! -f tools/eval_all.py ]; then
    echo "tools/eval_all.py (CORE-6) is missing: rebase on main."; return 1
  fi
  local against="" wb dir r result=0
  if git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null 2>&1; then
    against="--against $BASE"
  else
    warn "git ref '$BASE' not found: nothing to compare with, only the absolute checks run"
  fi
  for wb in $WORKBOOKS; do
    dir="$OUT/$(label_of "$wb")"
    mkdir -p "$dir"
    echo "--- scorecard: $wb (compared with ${against:-nothing})"
    # shellcheck disable=SC2086
    "$PY" tools/eval_all.py --all --no-llm --data "$wb" --out "$dir" --md $against | tee "$dir/eval_all.txt"
    r=${PIPESTATUS[0]}
    summary_md "#### Scorecard: \`$wb\`"
    summary_md "$(grep '^|' "$dir/eval_all.txt" || true)"
    if [ "$r" -ne 0 ] && [ "${CHECK_ALLOW_KPI_REGRESSION:-0}" = 1 ] && kpi_only "$dir/scorecard.json"; then
      warn "KPI regression versus main on $wb, allowed by CHECK_ALLOW_KPI_REGRESSION: the PR must explain it"
      [ $result -eq 0 ] && result=3
    elif [ "$r" -ne 0 ]; then
      result=1
    fi
  done
  return $result
}

step_redteam() {
  need_workbooks; local rc=$?; [ $rc -ne 0 ] && return $rc
  local wb result=0
  for wb in $WORKBOOKS; do
    echo "--- red team: $wb"
    "$PY" tools/check_redteam.py --data "$wb" --out "$OUT/$(label_of "$wb")/redteam" || result=1
  done
  return $result
}

step_honest() {
  local args="" wb
  for wb in $WORKBOOKS; do [ -f "$wb" ] && args="$args --data $wb"; done
  # shellcheck disable=SC2086
  "$PY" tools/check_honest.py $args
}

step_secrets() {
  if [ -d "$OUT" ]; then "$PY" tools/check_secrets.py --logs "$OUT"; else "$PY" tools/check_secrets.py; fi
}

# ------------------------------------------------------------------ run
echo "check: steps [$STEPS] · python $("$PY" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo '?')" \
     "· workbooks [${WORKBOOKS# }] · base $BASE"
RESULTS=""
FAILED=0
for s in $STEPS; do
  t0=$SECONDS
  group "$s"
  "step_$s"
  rc=$?
  endgroup
  case $rc in
    0) res=PASS ;;
    2) res=SKIP ;;
    3) res=WARN ;;
    *) res=FAIL; FAILED=1 ;;
  esac
  RESULTS="$RESULTS$(printf '  %-10s %-5s %4ss' "$s" "$res" $((SECONDS - t0)))
"
done

printf '\n=== check summary\n%s' "$RESULTS"
summary_md "#### check.sh: \`$STEPS\`"
summary_md "$(printf '```\n%s```' "$RESULTS")"
if [ $FAILED -ne 0 ]; then
  echo "check: FAIL"
  exit 1
fi
echo "check: PASS"
