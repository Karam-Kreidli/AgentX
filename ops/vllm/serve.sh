#!/usr/bin/env bash
# LLM-7: start vLLM on the L40S, reachable only from this machine (the HTTPS front in
# ops/vllm/Caddyfile + Tailscale Funnel is the only way in). See ops/vllm/runbook.md.
#
#   bash ops/vllm/serve.sh            # foreground (inside tmux: tmux new -s vllm)
#
# Settings come from ~/.vllm.env (or VLLM_ENV=<file>), never from this file:
#   MODEL=Qwen/...            the model chosen in LLM-3
#   REVISION=<commit sha>     pinned: the exact weights we tested (Hugging Face commit, not "main")
#   VLLM_API_KEY=<key>        the key in everyone's .env; vLLM reads it from the environment,
#                             so it never shows in `ps`
#   SERVED_NAME=...           optional: the name clients send as LLM_MODEL (default: MODEL)
#   MAX_MODEL_LEN=8192  GPU_MEM=0.90  PORT=8000  HOST=127.0.0.1
set -euo pipefail

ENV_FILE="${VLLM_ENV:-$HOME/.vllm.env}"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE (see the header of this script)"; exit 1; }
set -a; . "$ENV_FILE"; set +a

: "${MODEL:?set MODEL in $ENV_FILE}"
: "${VLLM_API_KEY:?set VLLM_API_KEY in $ENV_FILE: never serve without a key}"
if [ -z "${REVISION:-}" ] || [ "${REVISION}" = "main" ]; then
  echo "set REVISION in $ENV_FILE to the commit sha we tested (not 'main')"; exit 1
fi
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
if [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "localhost" ]; then
  echo "warning: HOST=$HOST serves vLLM beyond this machine; the HTTPS front is meant to be the only way in"
fi

export VLLM_API_KEY
exec vllm serve "$MODEL" \
  --revision "$REVISION" \
  --served-model-name "${SERVED_NAME:-$MODEL}" \
  --host "$HOST" --port "$PORT" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_MEM:-0.90}" \
  --generation-config vllm
