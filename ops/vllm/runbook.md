# Model endpoint runbook (LLM-7)

The run never needs the model to finish. If the model is down, every LLM call falls back
(local vLLM → Groq if configured → rules), the run carries on, and the run log says so
(`decisions.jsonl`: an `llm_down` note when it goes, an `llm_up` note if it comes back).
This runbook is about keeping the model *there*, so the judges see the LLM parts working.

## How it fits together

```
Admin's machine / our laptops                      L40S VM
  .env: LLM_BASE_URL=https://<vm>.<tailnet>.ts.net/v1
        LLM_API_KEY=<key>
            │ HTTPS (public)
            ▼
  Tailscale Funnel :443  ──►  Caddy 127.0.0.1:8080  ──►  vLLM 127.0.0.1:8000
  (TLS certificate)          (64 KB request limit,       (checks the API key,
                              only /v1/models and          pinned model revision)
                              /v1/chat/completions)
```

Nothing listens on a public interface: vLLM and Caddy bind to 127.0.0.1, and Funnel is the only
way in. Teammates use the same HTTPS URL as the Admin, so every test goes the real way.

Files: `serve.sh` (vLLM), `Caddyfile` (front), `Caddyfile.allowlist` (optional),
`healthcheck.py` (check, warm-up, watch), `smoke_test.py` (LLM-1 structured-output checks).
Secrets live only in `~/.vllm.env` on the VM and in each person's `.env`; never in the repo,
the chat history of a public channel, or a log.

## Questions for the organizers (ask before Sat 14 Nov)

1. Can the machine that runs our code reach an external HTTPS endpoint (outbound 443)? If not,
   the run is rules-only on the day: say so in the pitch, and show the LLM parts from our replay.
2. Which address does it call out from? (Only needed for the IP allowlist.)
3. How do we hand the Admin the `.env` (URL + key)? Use their channel, not a public one.

## One-time setup on the VM (Member A)

1. `~/.vllm.env` (chmod 600):
   ```
   MODEL=<the model chosen in LLM-3>
   REVISION=<the Hugging Face commit sha we tested>
   VLLM_API_KEY=<a new long random key: openssl rand -hex 32>
   SERVED_NAME=<the LLM_MODEL value in our .env>
   ```
2. Install Caddy (v2.7 or later) from the official package; nothing else to configure.
3. Funnel: Tailscale v1.38.3+, MagicDNS and HTTPS on, and the `funnel` node attribute for the VM
   in the tailnet policy (the first `tailscale funnel` command walks you through it).
4. Three tmux sessions, in this order:
   ```
   tmux new -s vllm   'bash ops/vllm/serve.sh'
   tmux new -s front  'caddy run --config ops/vllm/Caddyfile'
   tailscale funnel --bg --https=443 http://127.0.0.1:8080
   tailscale funnel status          # shows the https://<vm>.<tailnet>.ts.net URL
   ```
5. Send the URL and the new key to the team in the private chat; everyone updates `.env`:
   `LLM_BASE_URL=https://<vm>.<tailnet>.ts.net/v1`, `LLM_API_KEY=<key>`.
6. From a laptop: `python ops/vllm/healthcheck.py --warmup 20` → ALL PASS, size limit OK (413).
   Write down the model load time (how long `serve.sh` takes until `/v1/models` answers) here:
   **load time: ___ min**. It is the time a restart costs on the day.

## Before the day (Sun 15 Nov, after the freeze)

- [ ] `python ops/vllm/healthcheck.py --warmup 20`: ALL PASS, p95 well under `LLM_TIMEOUT_S` (8 s)
- [ ] `python ops/vllm/smoke_test.py`: ALL PASS
- [ ] `python tools/stress.py --llm real --drill conflict` and `--drill llm_dies` pass
      (llm_dies: the served model is cut off mid-run; the run finishes and `decisions.jsonl` has the
      `llm_down` note)
- [ ] One full run with the LLM on: `python run.py --all`; check `llm_calls.jsonl` is mostly `ok`
- [ ] Rotate the key for the day (new `VLLM_API_KEY`, restart `serve.sh`, new `.env` for the Admin
      and the team), so no key used during development works on the day

## On the day (Mon 16 Nov)

| When | Who | What |
|---|---|---|
| T−60 min | A | VM up; `tmux ls` shows vllm and front; `tailscale funnel status` shows the URL |
| T−45 min | A | `python ops/vllm/healthcheck.py --warmup 20` from a laptop: ALL PASS |
| T−30 min | E | The Admin's machine: `python ops/vllm/healthcheck.py` there if they allow it (proves *their* network reaches us); otherwise our `.env` is handed over as agreed |
| T−10 min | A | `python ops/vllm/healthcheck.py --watch 30` on a laptop, left running for the whole slot |
| During | A | Watch the status lines. Do not touch the VM unless the watch says DOWN |
| After | A | `tailscale funnel --https=443 off`; rotate the key |

## If the model is down

The run keeps going rules-only; nobody needs to stop or restart the run. The client tries the
model again every 60 s, so a model that comes back mid-run is used again automatically
(`llm_up` note). Fix it in this order, one step at a time, re-running
`python ops/vllm/healthcheck.py --backend local` after each:

| healthcheck says | Likely cause | Fix |
|---|---|---|
| `reachable: ConnectError` / `ConnectTimeout` | VM off, Funnel off, or Caddy down | `tailscale status` (VM online?), `tailscale funnel status`, `tmux attach -t front` |
| `reachable: HTTP 502` | Caddy is up, vLLM is not | `tmux attach -t vllm`: read the error; restart `bash ops/vllm/serve.sh` (costs the load time above) |
| `reachable: HTTP 401` | key mismatch | `.env` key ≠ `VLLM_API_KEY`: fix `.env`, never print the key in a shared screen |
| `model is not served` | `LLM_MODEL` ≠ `SERVED_NAME` | fix `.env` |
| `structured call` slow or `timeout` | GPU busy or out of memory | `nvidia-smi`; another process on the GPU? Restart `serve.sh` |
| `key enforced: FAIL` | vLLM started without the key | stop it at once; `serve.sh` refuses to start without `VLLM_API_KEY` |

If it cannot be fixed within the slot: leave it. The run finishes rules-only, the log shows when
and why the model went away, and that *is* the fallback the brief asks for: say so in the Q&A.

## Optional: IP allowlist

If the organizers give us their address, run `Caddyfile.allowlist` instead of `Caddyfile`
(both commands are at the top of that file; Funnel then forwards raw TCP with the PROXY protocol
so Caddy sees the caller's address). Add teammates' addresses too. Test it from an allowed
machine (healthcheck ALL PASS) and from one that is not (the request gets 403) before relying on it.

## What is logged where

| File | What |
|---|---|
| `runs/<scenario>/decisions.jsonl` | `llm_down` / `llm_up` notes (agent `llm`): which backend, why, what the run does instead |
| `runs/<scenario>/llm_calls.jsonl` | every call: backend, status (`ok`, `connect_error`, `breaker_open`, …), latency, tokens; no keys, no prompts |
| `runs/<scenario>/summary.json` | `llm`: calls used, by status, p50 latency |
| `runs/health/watch.jsonl` | the watch: one line per check |
