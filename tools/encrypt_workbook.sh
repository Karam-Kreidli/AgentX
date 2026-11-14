#!/usr/bin/env bash
# Give CI the competition workbook without putting it in git in the clear (F8).
#
#   bash tools/encrypt_workbook.sh [WORKBOOK]      # default: data/P2_Container_Terminal_Data.xlsx
#
# gpg asks you for a passphrase (twice) and writes ci/workbooks/team.xlsx.gpg (AES-256). Then:
#   1. git add ci/workbooks/team.xlsx.gpg, commit and push it in a PR.
#   2. On GitHub: Settings -> Secrets and variables -> Actions -> New repository secret,
#      name WORKBOOK_PASSPHRASE, value = the same passphrase.
# The CI "Workbooks" step decrypts it into data/ for the run. The passphrase never goes in a file,
# a commit or a log; keep it in your password manager.
set -euo pipefail
WB=${1:-data/P2_Container_Terminal_Data.xlsx}
OUT=ci/workbooks/team.xlsx.gpg
command -v gpg >/dev/null || { echo "gpg not found (Windows: run this in Git Bash, which includes gpg)" >&2; exit 2; }
[ -f "$WB" ] || { echo "workbook not found: $WB" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
gpg --symmetric --cipher-algo AES256 --yes --output "$OUT" "$WB"
echo "wrote $OUT ($(wc -c < "$OUT") bytes). Now commit it and add the WORKBOOK_PASSPHRASE secret (see the top of this script)."
