"""Operator inbox (LLM-6): plain-English messages from a human during a run.

    python -m core.inbox "Please prioritise P02-004, its cargo is urgent"   # operator side
    from core import inbox; inbox.poll()      # loop side (CORE-3 hook): new messages only

Messages are lines appended to one text file, OPERATOR_INBOX in the environment or
runs/operator_inbox.txt by default, so a judge can type them from another terminal while a run
is going. poll() returns the lines added since the last poll: at most MAX_PER_POLL per call
(the rest wait for the next tick), each one line of at most MAX_LEN characters. Lines already in
the file when the process starts are not replayed. A truncated or deleted file starts over.

Messages are untrusted text: core/events.py turns each into an operator_message event and
llm/operator.py translates it into one of a few fixed commands, or none. Never raises.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "runs" / "operator_inbox.txt"
MAX_PER_POLL = 3
MAX_LEN = 500

_state: dict = {"path": None, "offset": None, "queue": []}


def path() -> Path:
    p = os.environ.get("OPERATOR_INBOX")
    return Path(p) if p else DEFAULT_PATH


def reset(skip_existing: bool = True) -> None:
    """Forget the read position. skip_existing: start from the current end of the file."""
    p = path()
    size = p.stat().st_size if (skip_existing and p.exists()) else 0
    _state.update(path=str(p), offset=size, queue=[])


def poll() -> list[str]:
    """New messages since the last poll, oldest first. Never raises."""
    try:
        p = path()
        if _state["path"] != str(p) or _state["offset"] is None:
            reset()
        if p.exists():
            size = p.stat().st_size
            if size < _state["offset"]:                  # truncated or replaced: start over
                _state["offset"] = 0
            if size > _state["offset"]:
                with p.open("rb") as f:
                    f.seek(_state["offset"])
                    data = f.read(size - _state["offset"])
                end = data.rfind(b"\n")
                if end >= 0:                             # only complete lines; a half-written one waits
                    _state["offset"] += end + 1
                    for raw in data[:end].split(b"\n"):
                        text = " ".join(raw.decode("utf-8", errors="replace").split())[:MAX_LEN]
                        if text:
                            _state["queue"].append(text)
        else:
            _state["offset"] = 0
        out, _state["queue"] = _state["queue"][:MAX_PER_POLL], _state["queue"][MAX_PER_POLL:]
        return out
    except Exception:
        return []


def send(message: str, target: Optional[Path] = None) -> Path:
    """Append one message (operator side)."""
    p = Path(target) if target else path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(" ".join(str(message).split())[:MAX_LEN] + "\n")
    return p


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python -m core.inbox "your message to the terminal"')
    print(f"sent to {send(' '.join(sys.argv[1:]))}")
