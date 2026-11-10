"""LLM-7: the run log says when the model goes down and when it comes back, and a run whose model
server is killed mid-run still finishes (the Done-when fallback test, on a local stand-in server;
the same drill on the real endpoint is `python tools/stress.py --drill llm_dies`)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from llm import client as llm_client
from llm.client import Backend, LLMClient
from llm.schemas import OperatorCommand

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "P2_Container_Terminal_Data.xlsx"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _client(handler, clock, n=1):
    names = ["local", "groq"][:n]
    backends = [Backend(nm, f"http://{nm}.test/v1", "k", "qwen", "vllm") for nm in names]
    return LLMClient(backends, transport=httpx.MockTransport(handler), clock=clock, sleep=lambda s: None)


def _ok(request):
    if request.method == "GET":
        return httpx.Response(200, json={"data": [{"id": "qwen"}]})
    body = json.loads(request.content)
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(_instance(body))}}]})


def _down(request):
    raise httpx.ConnectError("refused", request=request)


def ask(c):
    return c.ask("operator", {"operator_message": "hold B1"}, OperatorCommand)


def test_first_success_is_not_news_but_going_down_and_coming_back_are():
    state = {"up": True}
    clock = Clock()
    c = _client(lambda r: _ok(r) if state["up"] else _down(r), clock)
    assert ask(c) is not None
    assert c.pop_transitions() == []
    state["up"] = False
    assert ask(c) is None
    [down] = c.pop_transitions()
    assert down["backend"] == "local" and not down["up"] and "connect_error" in down["why"]
    assert "stopped answering" in down["sentence"] and "rules" in down["sentence"]
    assert ask(c) is None and c.pop_transitions() == []          # still down: no second note
    state["up"] = True
    clock.t += 61                                                 # breaker cool-down over
    assert ask(c) is not None
    [up] = c.pop_transitions()
    assert up["up"] and "answering again" in up["sentence"]


def test_down_note_names_the_next_backend_when_there_is_one():
    clock = Clock()
    c = _client(lambda r: _down(r) if r.url.host == "local.test" else _ok(r), clock, n=2)
    assert ask(c) is not None                                     # groq answered
    ch = c.pop_transitions()
    assert [x["backend"] for x in ch] == ["local"] and "go to groq" in ch[0]["sentence"]


def test_a_failed_warmup_is_a_down_note():
    c = _client(_down, Clock())
    assert c.warmup() == {"local": "connect_error"}
    [d] = c.pop_transitions()
    assert not d["up"] and d["why"].startswith("warm-up")


def test_module_level_pop_is_empty_without_a_client(monkeypatch):
    monkeypatch.setattr(llm_client, "_default", None)
    assert llm_client.pop_transitions() == []


# ------------------------------------------------------------------ the server dies mid-run

def _instance(body):
    """A minimal answer that satisfies the JSON schema the client sent (structured_outputs)."""
    schema = (body.get("structured_outputs") or {}).get("json") or {}
    defs = schema.get("$defs", {})

    def make(s):
        if "$ref" in s:
            return make(defs[s["$ref"].split("/")[-1]])
        for key in ("anyOf", "oneOf", "allOf"):
            if key in s:
                return make(s[key][0])
        if "const" in s:
            return s["const"]
        if "enum" in s:
            return s["enum"][0]
        t = s.get("type")
        if t == "object":
            return {k: make(v) for k, v in (s.get("properties") or {}).items() if k in s.get("required", [])}
        if t == "array":
            return [make(s.get("items") or {}) for _ in range(s.get("minItems", 0))]
        if t == "integer":
            return int(s.get("minimum", 0))
        if t == "number":
            return float(s.get("minimum", 0))
        if t == "boolean":
            return False
        if t == "null":
            return None
        return "x" * max(1, s.get("minLength", 1))
    return make(schema)


class _Dying(BaseHTTPRequestHandler):
    """Answers valid JSON for every call, then the machine goes off after `die_after` chat calls."""
    die_after = 3
    posts = 0
    server_ref = None

    def log_message(self, *a):
        pass

    def _send(self, obj):
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send({"data": [{"id": "qwen"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        cls = type(self)
        cls.posts += 1
        if cls.posts > cls.die_after:
            self.close_connection = True
            threading.Thread(target=lambda: (cls.server_ref.shutdown(), cls.server_ref.server_close()),
                             daemon=True).start()
            return
        self._send({"choices": [{"message": {"content": json.dumps(_instance(body))}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


@pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook not in data/")
def test_killing_the_model_server_mid_run_the_run_finishes_and_the_log_says_so(tmp_path):
    import run
    handler = type("D", (_Dying,), {"posts": 0})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    handler.server_ref = srv
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = LLMClient([Backend("local", f"http://127.0.0.1:{srv.server_address[1]}/v1", "k", "qwen", "vllm")])
    before = llm_client._default
    llm_client.configure(c)
    try:
        assert c.warmup() == {"local": "ok"}
        s = run.run_one("practice_01", WORKBOOK, None, tmp_path, "multi")
    finally:
        llm_client.configure(before) if before is not None else setattr(llm_client, "_default", None)
        c.close()
    assert handler.posts > handler.die_after, "the run never used the LLM enough for the server to die"
    assert s["violations"] == [] and s["loop"]["breaches"] == 0 and s["loop"]["exceptions"] == 0
    assert s["vessels_departed"] >= 1
    calls = [json.loads(x) for x in (tmp_path / "practice_01" / "llm_calls.jsonl").read_text().splitlines()]
    assert any(r["status"] == "ok" for r in calls) and any(r["status"] == "connect_error" for r in calls)
    notes = [json.loads(x) for x in (tmp_path / "practice_01" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    down = [n for n in notes if "stopped answering" in json.dumps(n)]
    assert len(down) == 1, "one note when the model goes down, not one per failed call"
