"""Secret scan (SAFE-6): GROUND_RULES §5, keys live only in .env and are never committed or logged.

    python tools/check_secrets.py                    # every file git would commit
    python tools/check_secrets.py --logs runs/check  # ... and these run logs

Fails on:
  file      a file that must never be committed: .env, keys and certificates, and any Excel
            workbook (the repo is public; the workbook and answer key are private)
  token     a key-shaped token: Groq gsk_, OpenAI / Anthropic sk-, Hugging Face hf_, GitHub
            ghp_ / github_pat_, AWS AKIA, Tailscale tskey-, Bearer tokens, private-key blocks
  literal   a literal value given to a *_KEY / *_TOKEN / *_SECRET / *_PASSWORD name
  value     the value of a secret from .env or the environment, anywhere (exact match)

"Files git would commit" = tracked plus untracked-but-not-ignored (what `git add -A` adds).
Placeholders (values containing SECRET, FAKE, EXAMPLE, 1234567890, <...>, ${...}) are not
keys, so tests can use fake ones. Findings name the file, line and rule, never the value.
This file has its own patterns on purpose: a PR that weakens core/log.py's redaction must
not weaken this scan too. Exit code 1 if anything is found.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5_000_000
WALK_SKIP = {".git", ".venv", "venv", "env", "runs", "data", "toolskit", "__pycache__", "node_modules"}

NEVER_COMMIT = re.compile(r"(?i)(^|/)(\.env(\.(?!example$|sample$|template$)[^/]+)?|\.vllm\.env|id_(rsa|dsa|ecdsa|ed25519))$"
                          r"|\.(key|pem|p12|pfx|keystore|xlsx|xlsm|xlsb|xls)$")
TOKENS = (
    ("Groq key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}")),
    ("OpenAI / Anthropic key", re.compile(r"\bsk-(?:proj-|ant-|or-)?[A-Za-z0-9_\-]{20,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Tailscale key", re.compile(r"\btskey-[A-Za-z0-9\-]{20,}")),
    ("Bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{20,}=*")),
    ("private key", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")),
)
_NAME = r"[A-Za-z0-9_]*(?:API_KEY|_KEY|TOKEN|SECRET|PASSWORD|PASSPHRASE)"
LITERALS = (
    re.compile(rf"(?i)\b({_NAME})\b['\"]?\s*[:=]\s*['\"]([^'\"\s]{{12,}})['\"]"),     # x_KEY = "..." / "token": "..."
    re.compile(rf"(?im)^\s*(?:export\s+)?({_NAME})\s*=\s*([^\s'\"#]{{12,}})\s*$"),   # .env style: X_KEY=...
)
PLACEHOLDER = re.compile(r"(?i)secret|fake|dummy|example|placeholder|changeme|redacted|your[_-]|xxxx|abcdefgh"
                         r"|1234567890|0987654321|<|>|\$\{|\{\{|os\.environ|getenv|\*\*\*")
SECRET_NAME = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSPHRASE)$")


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    what: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.what}"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def committable(root: Path) -> list[Path]:
    """Files `git add -A` would commit: tracked plus untracked-not-ignored. Walks the tree without git."""
    try:
        r = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                           cwd=root, capture_output=True, timeout=60)
        if r.returncode == 0:
            names = sorted({n for n in r.stdout.decode("utf-8", "replace").split("\0") if n})
            return [root / n for n in names if (root / n).is_file()]
    except (OSError, subprocess.SubprocessError):
        pass
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in WALK_SKIP]
        out += [Path(dirpath) / f for f in filenames]
    return sorted(out)


def secret_values(root: Path, environ: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """{value: variable name} for secret-looking variables in .env and the environment."""
    found: dict[str, str] = {}
    env = os.environ if environ is None else environ
    for name, value in env.items():
        if SECRET_NAME.search(name) and value and len(value.strip()) >= 8:
            found[value.strip()] = name
    try:
        for line in (root / ".env").read_text(encoding="utf-8").splitlines():
            name, sep, value = line.strip().removeprefix("export ").partition("=")
            value = value.strip().strip('"').strip("'")
            if sep and not name.startswith("#") and SECRET_NAME.search(name.strip()) and len(value) >= 8:
                found[value] = name.strip()
    except OSError:
        pass
    return found


def _log_files(dirs: Iterable[Path]) -> list[Path]:
    out = []
    for d in dirs:
        if d.is_file():
            out.append(d)
        elif d.is_dir():
            out += sorted(p for p in d.rglob("*") if p.is_file())
    return out


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------

def _read_text(path: Path) -> Optional[str]:
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw[:8192]:                          # binary
        return None
    return raw.decode("utf-8", "replace")


def scan_text(rel: str, text: str, values: Mapping[str, str], patterns: bool = True) -> list[Finding]:
    found = []
    for i, line in enumerate(text.splitlines(), 1):
        for value, name in values.items():
            if value in line:
                found.append(Finding(rel, i, "value", f"the value of {name} (from .env or the environment)"))
        if not patterns:
            continue
        for what, rx in TOKENS:
            for m in rx.finditer(line):
                if not PLACEHOLDER.search(m.group()):
                    found.append(Finding(rel, i, "token", f"{what} {m.group()[:4]}…"))
        for rx in LITERALS:
            for m in rx.finditer(line):
                if not PLACEHOLDER.search(m.group(2)):
                    found.append(Finding(rel, i, "literal", f"a literal value for {m.group(1)}"))
    return found


def scan(root: Path = ROOT, files: Optional[Iterable[Path]] = None, logs: Iterable[Path] = (),
         environ: Optional[Mapping[str, str]] = None) -> tuple[list[Finding], int]:
    """(findings, files scanned) for the committable files under root, plus the log files."""
    values = secret_values(root, environ)
    files = list(files) if files is not None else committable(root)
    found: list[Finding] = []
    n = 0
    for p in files:
        rel = p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p)
        if NEVER_COMMIT.search(rel):
            found.append(Finding(rel, 0, "file", "this file must never be committed (keys, .env or a workbook)"))
            continue
        text = _read_text(p)
        if text is not None:
            n += 1
            found += scan_text(rel, text, values)
    for p in _log_files(logs):
        text = _read_text(p)
        if text is not None:
            n += 1
            rel = p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p)
            found += scan_text(rel, text, values)
    return found, n


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", type=Path, action="append", default=[], help="run-log folder to scan too (repeatable)")
    a = p.parse_args(argv)
    found, n = scan(ROOT, logs=[d if d.is_absolute() else ROOT / d for d in a.logs])
    for f in found:
        print(f"FAIL     {f}")
    print(f"secrets: {n} files scanned; {len(found)} finding(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
