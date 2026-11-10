"""LLM-7: health check, warm-up and watch for the model endpoint. Same .env and the same client
(llm/client.py) as the run, so a PASS here means the run will reach the model the same way.

    python ops/vllm/healthcheck.py                  # one full check of every backend in .env
    python ops/vllm/healthcheck.py --warmup 20      # ...then 20 real structured calls (latency p50/p95)
    python ops/vllm/healthcheck.py --watch 30       # a status line every 30 s until Ctrl+C (during our slot)
    python ops/vllm/healthcheck.py --backend local  # only the served model (skip Groq)

Checks, per backend:
  1. reachable     GET /models answers within 20 s and serves LLM_MODEL
  2. key enforced  the same request without the key is refused (401/403)
  3. size limit    a 2 MB request is refused with 413 before it reaches vLLM (the HTTPS front, see
                   ops/vllm/runbook.md); WARN, not FAIL, when nothing in front limits it
  4. structured    one supervisor-style call through llm.client answers valid JSON within LLM_TIMEOUT_S
Exit code 0 = every check passed (WARNs allowed), 1 = something failed. Never prints the key or the
full endpoint address (the URL is shared in the private chat only).
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from llm import client as llm_client                      # noqa: E402
from llm.client import Backend, LLMClient                 # noqa: E402
from llm.schemas import choice_model                      # noqa: E402

BIG_BODY_BYTES = 2_000_000
WATCH_LOG = ROOT / "runs" / "health" / "watch.jsonl"
OPTIONS = ["opt0", "opt1", "opt2"]
PAYLOAD = {"event": "QC3 broke down at 14:20; ship P-2 has 400 moves left",
           "options": [{"id": "opt0", "summary": "Keep the plan; P-2 finishes 2 h later."},
                       {"id": "opt1", "summary": "Move QC4 from P-1 to P-2; P-1 finishes 1 h later."},
                       {"id": "opt2", "summary": "Hold P-3 at anchor until QC3 is back."}]}
SYSTEM = "Pick the option that keeps the most ships on time. Answer with its id and one short reason."


def masked(url: str) -> str:
    """https://abc.tail1234.ts.net/v1 -> https://…ts.net ; http://100.64.1.2:8000/v1 -> http://…:8000"""
    try:
        u = urlsplit(url)
        host = u.hostname or ""
        try:
            ipaddress.ip_address(host)
            shown = "…"
        except ValueError:
            shown = "…" + ".".join(host.split(".")[-2:]) if "." in host else "…"
        return f"{u.scheme}://{shown}" + (f":{u.port}" if u.port else "")
    except Exception:
        return "…"


class Report:
    def __init__(self) -> None:
        self.fails = 0
        self.warns = 0

    def line(self, status: str, what: str, detail: str = "") -> None:
        if status == "FAIL":
            self.fails += 1
        elif status == "WARN":
            self.warns += 1
        print(f"  [{status:4}] {what}" + (f": {detail}" if detail else ""), flush=True)


def check_backend(b: Backend, rep: Report, timeout_s: float, http: httpx.Client) -> None:
    print(f"\n{b.name}  model {b.model}  at {masked(b.base_url)}")
    auth = {"Authorization": f"Bearer {b.api_key}"}
    # 1. reachable + model served
    t0 = time.perf_counter()
    try:
        r = http.get(f"{b.base_url}/models", headers=auth, timeout=llm_client.WARMUP_TIMEOUT_S)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            rep.line("FAIL", "reachable", f"HTTP {r.status_code} in {dt:.1f} s")
            return
        served = sorted(m.get("id") for m in (r.json().get("data") or []) if isinstance(m, dict))
        if served and b.model not in served:
            rep.line("FAIL", "reachable", f"answers, but LLM_MODEL is not served (served: {', '.join(served)[:120]})")
            return
        rep.line("OK", "reachable", f"{dt:.2f} s, serves {b.model}")
    except httpx.HTTPError as e:
        rep.line("FAIL", "reachable", f"{type(e).__name__} after {time.perf_counter() - t0:.1f} s "
                 "(VM off? HTTPS front or tunnel down? see runbook 'If the model is down')")
        return
    # 2. key enforced
    try:
        r = http.get(f"{b.base_url}/models", timeout=10)
        rep.line("OK" if r.status_code in (401, 403) else "FAIL", "key enforced",
                 f"no-key request got HTTP {r.status_code}" + ("" if r.status_code in (401, 403) else
                                                               " (the endpoint is open to anyone: restart vLLM with --api-key)"))
    except httpx.HTTPError as e:
        rep.line("FAIL", "key enforced", type(e).__name__)
    # 3. size limit (only meaningful for our own endpoint)
    if b.name == "local":
        big = {"model": b.model, "max_tokens": 1, "messages": [{"role": "user", "content": "x" * BIG_BODY_BYTES}]}
        try:
            r = http.post(f"{b.base_url}/chat/completions", json=big, headers=auth, timeout=30)
            if r.status_code == 413:
                rep.line("OK", "size limit", "2 MB request refused with 413")
            else:
                rep.line("WARN", "size limit", f"2 MB request got HTTP {r.status_code}: nothing in front limits "
                         "request size (fine on the tailnet; put the HTTPS front up before the day)")
        except httpx.HTTPError as e:
            rep.line("WARN", "size limit", f"{type(e).__name__} (connection dropped on a 2 MB body)")
    # 4. one structured call through the run's own client
    one = LLMClient([b], timeout_s=timeout_s, budget=5)
    try:
        t0 = time.perf_counter()
        out = one.ask("supervisor", PAYLOAD, choice_model(OPTIONS), system=SYSTEM)
        dt = time.perf_counter() - t0
        last = one.records[-1]["status"] if one.records else "no call"
        if out and out.get("option_id") in OPTIONS:
            rep.line("OK" if dt < timeout_s * 0.6 else "WARN", "structured call",
                     f"{dt:.2f} s (limit {timeout_s:g} s) -> {out['option_id']}")
        else:
            rep.line("FAIL", "structured call", f"no valid answer ({last}) in {dt:.2f} s")
    finally:
        one.close()


def warmup(b: Backend, n: int, timeout_s: float, rep: Report) -> None:
    """n real calls: loads the model's caches before our slot and measures latency."""
    c = LLMClient([b], timeout_s=timeout_s, budget=n * 2 + 2)
    lat, bad = [], 0
    try:
        for i in range(n):
            t0 = time.perf_counter()
            out = c.ask("supervisor", {**PAYLOAD, "round": i}, choice_model(OPTIONS), system=SYSTEM)
            if out:
                lat.append(time.perf_counter() - t0)
            else:
                bad += 1
    finally:
        c.close()
    if not lat:
        rep.line("FAIL", f"warm-up {b.name}", f"0 of {n} calls answered")
        return
    lat.sort()
    p50, p95 = lat[len(lat) // 2], lat[min(len(lat) - 1, int(len(lat) * 0.95))]
    status = "OK" if not bad and p95 < timeout_s * 0.6 else ("FAIL" if bad > n // 4 else "WARN")
    rep.line(status, f"warm-up {b.name}", f"{len(lat)}/{n} answered, p50 {p50:.2f} s, p95 {p95:.2f} s, "
             f"max {lat[-1]:.2f} s (limit {timeout_s:g} s)")


def watch(backends: list[Backend], every_s: float, timeout_s: float, log: Path) -> int:
    """One status line per interval; a loud line when a backend changes state. Ctrl+C stops."""
    log.parent.mkdir(parents=True, exist_ok=True)
    last: dict[str, Optional[bool]] = {b.name: None for b in backends}
    print(f"watching {', '.join(b.name for b in backends)} every {every_s:g} s (Ctrl+C to stop); log: {log}")
    try:
        while True:
            for b in backends:
                c = LLMClient([b], timeout_s=timeout_s, budget=3)
                t0 = time.perf_counter()
                out = c.ask("supervisor", PAYLOAD, choice_model(OPTIONS), system=SYSTEM)
                dt = time.perf_counter() - t0
                status = "ok" if out else (c.records[-1]["status"] if c.records else "no_call")
                c.close()
                up = out is not None
                stamp = time.strftime("%H:%M:%S")
                flag = "" if last[b.name] in (None, up) else ("   <<< BACK UP" if up else "   <<< DOWN: see runbook")
                print(f"{stamp}  {b.name:6} {'UP  ' if up else 'DOWN'} {status:14} {dt:5.2f} s{flag}", flush=True)
                last[b.name] = up
                with log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": round(time.time(), 1), "backend": b.name, "up": up,
                                        "status": status, "latency_s": round(dt, 3)}) + "\n")
            time.sleep(every_s)
    except KeyboardInterrupt:
        print("stopped")
        return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["local", "groq"], help="only this backend (default: every one in .env)")
    p.add_argument("--warmup", type=int, default=0, metavar="N", help="after the checks, N real calls per backend")
    p.add_argument("--watch", type=float, default=0, metavar="S", help="status line every S seconds until Ctrl+C")
    p.add_argument("--log", type=Path, default=WATCH_LOG, help=f"watch log (default {WATCH_LOG.relative_to(ROOT)})")
    a = p.parse_args(argv)

    llm_client.load_dotenv()
    timeout_s = llm_client._env_float("LLM_TIMEOUT_S", llm_client.DEFAULT_TIMEOUT_S)
    backends = [b for b in llm_client.backends_from_env() if not a.backend or b.name == a.backend]
    if not backends:
        print("No LLM backend configured: fill LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in .env "
              "(the run would go rules-only, which is allowed but not what we want on the day).")
        return 1
    if a.watch:
        return watch(backends, a.watch, timeout_s, a.log)
    rep = Report()
    with httpx.Client(follow_redirects=False) as http:
        for b in backends:
            check_backend(b, rep, timeout_s, http)
    for b in backends if a.warmup else ():
        warmup(b, a.warmup, timeout_s, rep)
    print(f"\n{'ALL PASS' if not rep.fails else f'{rep.fails} FAIL'}" + (f", {rep.warns} WARN" if rep.warns else ""))
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
