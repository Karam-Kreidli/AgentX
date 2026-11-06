"""Organizer toolkit, unmodified (p2_terminal_api.py, common.py).

The toolkit uses `from common import ...`, so this package puts its own folder
on sys.path before importing it. Do not edit the two toolkit files; wrap them.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from p2_terminal_api import TerminalAPI  # noqa: E402
from common import APIError  # noqa: E402

__all__ = ["TerminalAPI", "APIError"]
