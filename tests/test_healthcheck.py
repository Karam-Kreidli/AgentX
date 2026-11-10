"""LLM-7: ops/vllm/healthcheck.py against a local stand-in endpoint (no VM needed)."""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.test_llm_health import _instance

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("healthcheck", ROOT / "ops" / "vllm" / "healthcheck.py")
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)


class _Endpoint(BaseHTTPRequestHandler):
    key_required = True
    size_limit = True

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authed(self):
        return not self.key_required or self.headers.get("Authorization") == "Bearer sekret-key"

    def do_GET(self):
        if not self._authed():
            return self._send(401, {"error": "unauthorized"})
        self._send(200, {"data": [{"id": "qwen"}]})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if self.size_limit and n > 65536:
            self.close_connection = True
            return self._send(413, {"error": "too large"})
        body = json.loads(self.rfile.read(n) or b"{}")
        if not self._authed():
            return self._send(401, {"error": "unauthorized"})
        if n > 65536:
            return self._send(400, {"error": "context length exceeded"})
        self._send(200, {"choices": [{"message": {"content": json.dumps(_instance(body))}}]})


@pytest.fixture
def endpoint(monkeypatch):
    servers = []

    def start(**kw):
        h = type("E", (_Endpoint,), kw)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), h)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
        monkeypatch.setenv("LLM_API_KEY", "sekret-key")
        monkeypatch.setenv("LLM_MODEL", "qwen")
        return srv
    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def test_a_healthy_endpoint_passes_and_the_key_is_never_printed(endpoint, capsys, tmp_path):
    endpoint()
    assert hc.main(["--backend", "local", "--warmup", "3"]) == 0
    out = capsys.readouterr().out
    assert "ALL PASS" in out and "WARN" not in out
    assert "sekret-key" not in out and "127.0.0.1" not in out


def test_an_endpoint_without_the_key_fails(endpoint, capsys):
    endpoint(key_required=False)
    assert hc.main(["--backend", "local"]) == 1
    assert "[FAIL] key enforced" in capsys.readouterr().out


def test_no_size_limit_in_front_is_a_warning_not_a_failure(endpoint, capsys):
    endpoint(size_limit=False)
    assert hc.main(["--backend", "local"]) == 0
    out = capsys.readouterr().out
    assert "[WARN] size limit" in out and "1 WARN" in out


def test_a_dead_endpoint_fails_fast(monkeypatch, capsys):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "qwen")
    assert hc.main(["--backend", "local"]) == 1
    assert "[FAIL] reachable" in capsys.readouterr().out


def test_masked_never_shows_the_host():
    assert hc.masked("https://agentx-vm.tail1234.ts.net/v1") == "https://…ts.net"
    assert hc.masked("http://100.101.102.103:8000/v1") == "http://…:8000"
