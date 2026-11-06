"""Test helpers.

Loading the workbook takes ~6 s each time a simulator is created. Tests create
many, so the workbook's sheets are read once per test session and copies are
handed out. Test-only: the toolkit in vendor/ stays unmodified, and run.py
always reads the workbook the normal way.
"""
import pytest


@pytest.fixture(autouse=True, scope="session")
def _cache_workbook_reads():
    try:
        import vendor  # noqa: F401  (puts vendor/ on sys.path)
        import p2_terminal_api
    except Exception:
        yield
        return
    original = p2_terminal_api.sheets
    cache = {}

    def cached(path, *names):
        key = (str(path), names)
        if key not in cache:
            cache[key] = original(path, *names)
        return {n: df.copy() for n, df in cache[key].items()}

    p2_terminal_api.sheets = cached
    yield
    p2_terminal_api.sheets = original
