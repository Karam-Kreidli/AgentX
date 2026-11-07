"""Coordinator: negotiation and admission control (CORE-5).

    import coordinator
    grants, options = coordinator.decide(requests, state)

Agents ask for scarce resources with `Request`s; the coordinator answers each with a `Grant`
(granted / trimmed / refused, with a reason). Grants are binding: the loop hands every agent
its grants (`agent.grants`, keyed `(resource, call_id)`) and asks it to act again, so an agent
never places more than it was granted in the same round.

Yard slots (the bottleneck):
  1. YARD-3's budgets first: a written-off or impossible ship gets 0 (reason from the forecast).
  2. Then the space actually free, per cargo group, in the goal-ladder order: ships that are
     not written off, by arrival (first come, first served). YARD-3 found that this order
     finishes the most ships unless writing one off is predicted to finish strictly more.
     Space follows the yard rules: IMDG -> H, reefer -> I/J, dry -> standard, then G, then I/J.
     The yard agent fills it in the same order, so a grant is exactly what it can place.
Trucks, cranes, berths: what is free right now, highest priority first, then arrival order.

Forecast write-offs are switched off for now (OFFER_FORECAST_WRITE_OFFS, see below): every
ship that is not impossible is served. When switched on: when YARD-3 writes off a ship that is
not impossible (a judgement call: give up one ship so others finish), the coordinator offers the supervisor two Options: keep serving every ship
first-come-first-served (option 0, the safe default: no ship is given up on a forecast) or
follow the forecast. Until the supervisor picks the forecast, every such ship is served. The
pick comes back through `chosen(option_id)` and holds until the set of written-off ships
changes. Why FCFS is the default: on practice_03 the forecast wrote off P03-003 on arrival and
changed its mind 7.5 h later; following it delayed P03-003 and P03-005 by ~7 h each.
Impossible ships (more discharge than imports left) are never a judgement call: they get 0.

The negotiation transcript (core/log.py: negotiation.jsonl / negotiation.md) gets every
request and decision whose status (granted / trimmed / refused) changed since the last one for
that agent, resource and ship: every conflict shows up once, not every tick.

Never raises: a failure in the forecast grants requests on free space alone.
"""
from __future__ import annotations

from typing import Optional

from core import log as runlog
from core.types import (BLOCK_EMPTY, BLOCK_IMDG, BLOCK_REEFER, BLOCK_STANDARD, CARGO_DRY, CARGO_IMDG,
                        CARGO_REEFER, EQ_AVAILABLE, GRANTED, KNOWN_CARGO, REFUSED, RES_BERTH, RES_CRANES,
                        RES_TRUCKS, RES_YARD_SLOTS, TRIMMED, Grant, Option, Request, WorldState)

OPT_FORECAST = "follow_forecast"
OPT_FCFS = "serve_all"
# Forecast write-offs of ships that are not impossible are switched off: on practice_03 the
# forecast wrote off P03-003 on arrival and reversed itself 7.5 h later; following it (as the
# LLM supervisor does, since it predicts more departures) raised honest turnaround 34.18 -> 36.4 h.
# Turn back on when the YARD-3 forecast is more reliable. Impossible ships are always refused.
OFFER_FORECAST_WRITE_OFFS = False


class _Memory:
    def __init__(self) -> None:
        self.scenario: Optional[tuple] = None
        self.logged: dict = {}               # (agent, resource, call_id) -> (status, granted, amount)
        self.choice: dict = {}               # frozenset(written off) -> option id picked
        self.pending: Optional[frozenset] = None


_mem = _Memory()


def reset() -> None:
    """Forget everything (new scenario). Called automatically when the scenario changes."""
    global _mem
    _mem = _Memory()


def _scenario_key(state: WorldState) -> tuple:
    return (state.scenario_end, tuple(sorted(s.call_id for s in state.ships)))


# --------------------------------------------------------------------------
# yard slots
# --------------------------------------------------------------------------

def unplaced_imports(state: WorldState, call_id: str) -> dict[str, int]:
    """Import boxes of a call with no yard slot yet, per known cargo type."""
    out = {CARGO_IMDG: 0, CARGO_REEFER: 0, CARGO_DRY: 0}
    for c in state.manifests.get(call_id, ()):
        if c.direction == "import" and not c.yard_block and c.cargo_type in KNOWN_CARGO:
            out[c.cargo_type] += 1
    return out


def free_space(state: WorldState) -> dict[str, int]:
    """Free slots per block family: H (IMDG), I/J (reefer), standard, G (empties)."""
    free = {BLOCK_IMDG: 0, BLOCK_REEFER: 0, BLOCK_STANDARD: 0, BLOCK_EMPTY: 0}
    for b in state.blocks:
        if b.block_type in free:
            free[b.block_type] += b.free
    return free


def fit(need: dict[str, int], free: dict[str, int]) -> int:
    """How many of `need` fit in `free` under the yard rules; takes them out of `free`."""
    n_imdg = min(need.get(CARGO_IMDG, 0), free[BLOCK_IMDG])
    free[BLOCK_IMDG] -= n_imdg
    n_reefer = min(need.get(CARGO_REEFER, 0), free[BLOCK_REEFER])
    free[BLOCK_REEFER] -= n_reefer
    dry = need.get(CARGO_DRY, 0)
    n_dry = 0
    for kind in (BLOCK_STANDARD, BLOCK_EMPTY, BLOCK_REEFER):      # dry: standard, then G, then I/J
        take = max(0, min(dry - n_dry, free[kind]))
        free[kind] -= take
        n_dry += take
    return n_imdg + n_reefer + n_dry


def _forecast(state: WorldState):
    """(budgets, bound) from YARD-3, or (None, None) if it fails."""
    try:
        from agents import admission
        return admission.budgets(state), admission.bound(state)
    except Exception as e:                      # never crash: fall back to free space only
        runlog.note(runlog.current_tick() or 0, state, "coordinator", "forecast_failed",
                    f"Yard forecast failed ({type(e).__name__}: {str(e)[:120]}); granting on free space only.")
        return None, None


def _yard(reqs: list[Request], state: WorldState, order: dict) -> tuple[list[Grant], list[Option]]:
    budgets, bnd = _forecast(state)
    options: list[Option] = []
    written_off = frozenset(bnd.written_off) if bnd else frozenset()
    judged = written_off - frozenset(bnd.impossible if bnd else ())
    serve_all = False
    if judged and not OFFER_FORECAST_WRITE_OFFS:
        serve_all = True                          # forecast write-offs off: serve them (see above)
    elif judged:
        pick = _mem.choice.get(written_off)
        serve_all = pick != OPT_FORECAST
        if pick is None:
            options = _write_off_options(bnd, judged)
            _mem.pending = written_off

    free = free_space(state)
    grants = []
    ranked = sorted(reqs, key=lambda r: (budgets is not None and budgets.get(r.call_id, 1) == 0
                                         and not (serve_all and r.call_id in judged),
                                         -r.priority, order.get(r.call_id, 1e9), r.call_id))
    for r in ranked:
        need = unplaced_imports(state, r.call_id)
        asked = max(0, r.amount)
        budget = None if budgets is None else budgets.get(r.call_id, asked)
        if serve_all and r.call_id in judged:
            budget = asked
        if budget is not None and budget <= 0:
            why = (bnd.reasons.get(r.call_id) if bnd else None) or "The yard forecast gives it no slots."
            grants.append(Grant(r, 0, REFUSED, f"Admission control: {why}"))
            continue
        cap = asked if budget is None else min(asked, budget)
        scaled = {k: v for k, v in need.items()}
        if cap < sum(need.values()):              # asked for fewer than it has: keep the cargo mix order
            left = cap
            for k in (CARGO_IMDG, CARGO_REEFER, CARGO_DRY):
                scaled[k] = min(scaled[k], left)
                left -= scaled[k]
        got = fit(scaled, free)
        if got >= r.amount:
            grants.append(Grant(r, r.amount, GRANTED, "Room for all of them in legal blocks."))
        elif got > 0:
            grants.append(Grant(r, got, TRIMMED,
                                f"Only {got} legal slots are free for its cargo once ships ahead of it in "
                                f"arrival order are served; the other {r.amount - got} wait for boxes to "
                                f"leave the yard."))
        else:
            grants.append(Grant(r, 0, REFUSED,
                                "No legal slot is free for its cargo; its boxes wait for the yard to release some."))
    return grants, options


def _write_off_options(bnd, judged: frozenset) -> list[Option]:
    ships = ", ".join(sorted(judged))
    why = " ".join(bnd.reasons.get(c, "") for c in sorted(judged)).strip()
    return [
        Option(OPT_FCFS, (), {"ships_departed": bnd.predicted_departed_fcfs, "written_off": []},
               f"Keep serving every ship in arrival order, {ships} included: no ship is given up on a "
               f"forecast; the forecast expects {bnd.predicted_departed_fcfs} ships to finish this way."),
        Option(OPT_FORECAST, (), {"ships_departed": bnd.predicted_departed, "written_off": sorted(judged)},
               f"Stop giving yard slots to {ships}: the forecast finishes {bnd.predicted_departed} ships "
               f"this way against {bnd.predicted_departed_fcfs} if every ship is served. {why}".strip()),
    ]


def chosen(option_id: str) -> None:
    """The supervisor's (or the default) pick for the options returned last; holds until the
    set of written-off ships changes."""
    if _mem.pending is not None:
        _mem.choice[_mem.pending] = option_id
        _mem.pending = None


# --------------------------------------------------------------------------
# trucks, cranes, berths: what is free right now
# --------------------------------------------------------------------------

def _pool(reqs: list[Request], free: int, what: str, order: dict) -> list[Grant]:
    grants = []
    for r in sorted(reqs, key=lambda r: (-r.priority, order.get(r.call_id, 1e9), r.call_id)):
        got = max(0, min(r.amount, free))
        free -= got
        if got >= r.amount:
            grants.append(Grant(r, got, GRANTED, f"{got} {what} free."))
        elif got > 0:
            grants.append(Grant(r, got, TRIMMED, f"Only {got} {what} free; higher-priority or earlier ships "
                                                 f"were served first."))
        else:
            grants.append(Grant(r, 0, REFUSED, f"No {what} free right now; asked again when one frees up."))
    return grants


def _free_berths(state: WorldState) -> int:
    used = {s.berth for s in state.ships if s.berth}
    return sum(1 for b in state.berths if b.berth_id not in used)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def decide(requests: list[Request], state: WorldState) -> tuple[list[Grant], list[Option]]:
    """Grant, trim or refuse each request; options when a write-off is a judgement call."""
    key = _scenario_key(state)
    if _mem.scenario != key:
        reset()
        _mem.scenario = key
    order = {s.call_id: i for i, s in enumerate(state.ships)}
    by_res: dict[str, list[Request]] = {}
    for r in requests:
        by_res.setdefault(r.resource, []).append(r)

    grants: list[Grant] = []
    options: list[Option] = []
    for res, reqs in by_res.items():
        if res == RES_YARD_SLOTS:
            g, options = _yard(reqs, state, order)
        elif res == RES_TRUCKS:
            g = _pool(reqs, sum(1 for t in state.trucks if t.status == EQ_AVAILABLE and not t.call_id),
                      "trucks", order)
        elif res == RES_CRANES:
            g = _pool(reqs, sum(1 for c in state.cranes if c.status == EQ_AVAILABLE and not c.call_id),
                      "cranes", order)
        elif res == RES_BERTH:
            g = _pool(reqs, _free_berths(state), "berths", order)
        else:
            g = [Grant(r, 0, REFUSED, f"Unknown resource {str(res)[:20]!r}.") for r in reqs]
        grants += g
    if not options:                       # with options pending the loop decides again after the pick
        _transcript(grants)
    return grants, options


def _transcript(grants: list[Grant]) -> None:
    """Request + decision lines, only when the status changed for that agent/resource/ship."""
    for g in grants:
        r = g.request
        k = (r.agent, r.resource, r.call_id)
        now = g.status
        if _mem.logged.get(k) == now:
            continue
        _mem.logged[k] = now
        runlog.negotiation(r)
        runlog.negotiation(grant=g)
