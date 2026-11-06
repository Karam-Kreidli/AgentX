"""Shared types for the P2 Container Terminal agent (CORE-1).

FROZEN after Fri 6 Nov 2026. Changing or removing a field needs Member E's
approval and a dated note in CONTRACTS.md. Adding a new optional field at the
end of a class (with a default) also needs a CONTRACTS.md note.

Rules for everyone:
- Every type is an immutable (frozen) dataclass. Agents receive a WorldState and
  must never try to change it; build new objects instead.
- Collections inside types are tuples, never lists, so they cannot be mutated.
- Times are timezone-naive `datetime` objects in simulated time (the toolkit's
  clock). `core/state.py` (CORE-3) converts the toolkit's ISO strings.
- Only the stdlib is imported here, so `import core.types` works in a bare venv.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

# --------------------------------------------------------------------------
# Vocabularies (use these constants, not string literals, in agent code)
# --------------------------------------------------------------------------

# Ship.status: exactly the simulator's call states.
SHIP_EXPECTED = "expected"
SHIP_ARRIVED = "arrived"
SHIP_BERTHED = "berthed"     # berthed, mooring (work starts 1 h later)
SHIP_WORKING = "working"
SHIP_SAILING = "sailing"     # all moves done, leaves 1 h later
SHIP_DEPARTED = "departed"
SHIP_STATUSES = (SHIP_EXPECTED, SHIP_ARRIVED, SHIP_BERTHED, SHIP_WORKING, SHIP_SAILING, SHIP_DEPARTED)

# Crane.status / Truck.status (normalised from get_equipment_status text).
EQ_AVAILABLE = "available"
EQ_WORKING = "working"           # crane on a call, or truck serving a call
EQ_OUT_OF_SERVICE = "out_of_service"
EQ_STATUSES = (EQ_AVAILABLE, EQ_WORKING, EQ_OUT_OF_SERVICE)

# Container.cargo_type values we know. Anything else is "unknown cargo" and is
# never placed (YARD-1 / YARD-5 escalate it).
# Spelling matches the official workbook exactly (case-sensitive).
CARGO_DRY = "dry"
CARGO_REEFER = "reefer"
CARGO_IMDG = "IMDG"
KNOWN_CARGO = (CARGO_DRY, CARGO_REEFER, CARGO_IMDG)

# Block.block_type values (official workbook): A-F standard, G empty,
# H IMDG (IMDG only), I/J reefer (reefers only, our rule).
BLOCK_STANDARD = "standard"
BLOCK_EMPTY = "empty"
BLOCK_IMDG = "IMDG"
BLOCK_REEFER = "reefer"
BLOCK_TYPES = (BLOCK_STANDARD, BLOCK_EMPTY, BLOCK_IMDG, BLOCK_REEFER)

# Request.resource
RES_TRUCKS = "trucks"
RES_CRANES = "cranes"
RES_YARD_SLOTS = "yard_slots"
RES_BERTH = "berth"
RESOURCES = (RES_TRUCKS, RES_CRANES, RES_YARD_SLOTS, RES_BERTH)

# Grant.status
GRANTED = "granted"
TRIMMED = "trimmed"
REFUSED = "refused"
GRANT_STATUSES = (GRANTED, TRIMMED, REFUSED)

# Event.kind (CORE-3 detect_events, YARD-4 stall detector, inbox).
EV_ARRIVAL = "arrival"
EV_BERTHED = "berthed"
EV_DEPARTED = "departed"
EV_CRANE_DOWN = "crane_down"
EV_CRANE_BACK = "crane_back"
EV_TRUCK_DOWN = "truck_down"
EV_TRUCK_BACK = "truck_back"
EV_CUSTOMS_HOLD = "customs_hold"
EV_HOLD_RELEASED = "hold_released"
EV_SHIP_STALLED = "ship_stalled"
EV_SHIP_IMPOSSIBLE = "ship_impossible"
EV_OPERATOR_MESSAGE = "operator_message"
EV_ANOMALY = "anomaly"            # from YARD-5 validation
EVENT_KINDS = (
    EV_ARRIVAL, EV_BERTHED, EV_DEPARTED, EV_CRANE_DOWN, EV_CRANE_BACK,
    EV_TRUCK_DOWN, EV_TRUCK_BACK, EV_CUSTOMS_HOLD, EV_HOLD_RELEASED,
    EV_SHIP_STALLED, EV_SHIP_IMPOSSIBLE, EV_OPERATOR_MESSAGE, EV_ANOMALY,
)

# Action.tool: the only state-changing toolkit tools. advance_time is called by
# the loop, never proposed by agents.
TOOL_ASSIGN_BERTH = "assign_berth"
TOOL_PLAN_CRANES = "plan_crane_sequence"
TOOL_DISPATCH_TRUCKS = "dispatch_truck"
TOOL_PLACE = "place_container"
TOOL_RELEASE = "release_to_consignee"   # not used unless organizers allow it
AGENT_TOOLS = (TOOL_ASSIGN_BERTH, TOOL_PLAN_CRANES, TOOL_DISPATCH_TRUCKS, TOOL_PLACE, TOOL_RELEASE)


# --------------------------------------------------------------------------
# Static reference data (trust level 2: below the live simulator)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Berth:
    berth_id: str            # "B1"
    length_m: float
    max_draft_m: float
    start_bay: int
    end_bay: int

    def fits(self, loa_m: float, draft_m: float) -> bool:
        """Simulator fit rule: length >= LOA + 20 m and draft <= max draft."""
        return self.length_m >= loa_m + 20 and draft_m <= self.max_draft_m


# --------------------------------------------------------------------------
# World state (built by core/state.py from the read tools, CORE-3)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Ship:
    """One vessel call, from get_vessel_schedule."""
    call_id: str
    vessel: str                         # untrusted text: never put in a prompt unquoted
    loa_m: float
    draft_m: float
    scheduled_eta: datetime
    actual_arrival: Optional[datetime]  # None until the ship has arrived
    status: str                         # one of SHIP_STATUSES
    berth: Optional[str]
    cranes: tuple[str, ...]             # cranes currently assigned
    trucks: int                         # number of trucks assigned
    discharge_remaining: int
    load_remaining: int
    departed_at: Optional[datetime]

    @property
    def remaining_moves(self) -> int:
        return self.discharge_remaining + self.load_remaining


@dataclass(frozen=True)
class Crane:
    """One quay crane, from get_equipment_status."""
    crane_id: str                       # "QC1".."QC6"
    status: str                         # one of EQ_STATUSES
    call_id: Optional[str]              # call it works on, if any
    bay: int                            # current rail position (idle cranes keep theirs)


@dataclass(frozen=True)
class Truck:
    """One yard truck, from get_equipment_status."""
    truck_id: str
    status: str                         # one of EQ_STATUSES
    call_id: Optional[str]              # call it serves, if any


@dataclass(frozen=True)
class Block:
    """One yard block, from get_yard_status."""
    block_id: str                       # "A".."J"
    block_type: str
    used: int
    slots: int

    @property
    def free(self) -> int:
        return max(0, self.slots - self.used)

    @property
    def fill(self) -> float:
        return self.used / self.slots if self.slots else 1.0


@dataclass(frozen=True)
class Container:
    """One manifest row, from get_manifest(call_id)."""
    container_id: str                   # untrusted text
    call_id: str
    direction: str                      # "import" | "export"
    size_ft: Optional[int]
    cargo_type: str                     # see KNOWN_CARGO; anything else = unknown
    imdg_class: Optional[float]         # e.g. 3.0, 2.1; None if not IMDG
    gross_weight_t: Optional[float]
    yard_block: Optional[str]           # None until placed


@dataclass(frozen=True)
class Hold:
    """One customs hold, from get_customs_status."""
    container_id: str
    hold_since: datetime
    released: bool                      # True once customs has lifted it


@dataclass(frozen=True)
class WorldState:
    """One validated snapshot of the terminal at `now`. Read-only for agents."""
    now: datetime
    scenario_end: datetime
    ships: tuple[Ship, ...]
    cranes: tuple[Crane, ...]           # in rail order (QC1..QC6)
    trucks: tuple[Truck, ...]
    blocks: tuple[Block, ...]
    holds: tuple[Hold, ...]
    manifests: Mapping[str, tuple[Container, ...]] = field(default_factory=dict)  # call_id -> rows
    berths: tuple[Berth, ...] = ()      # static reference data
    anomalies: tuple[str, ...] = ()     # YARD-5 findings for this snapshot

    # Convenience lookups (return None when missing; never raise).
    def ship(self, call_id: str) -> Optional[Ship]:
        return next((s for s in self.ships if s.call_id == call_id), None)

    def crane(self, crane_id: str) -> Optional[Crane]:
        return next((c for c in self.cranes if c.crane_id == crane_id), None)

    def block(self, block_id: str) -> Optional[Block]:
        return next((b for b in self.blocks if b.block_id == block_id), None)

    def berth(self, berth_id: str) -> Optional[Berth]:
        return next((b for b in self.berths if b.berth_id == berth_id), None)

    def held_ids(self) -> frozenset[str]:
        """Containers under an active (not yet released) customs hold."""
        return frozenset(h.container_id for h in self.holds if not h.released)


# --------------------------------------------------------------------------
# Agent <-> coordinator <-> executor messages
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    kind: str                           # one of EVENT_KINDS
    subject: str                        # call_id, equipment_id, container_id or "operator"
    t: float                            # simulated hours since scenario start
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """A proposed state-changing tool call. Only core/executor.py sends it."""
    tool: str                           # one of AGENT_TOOLS
    args: Mapping[str, Any]             # exactly the toolkit tool's arguments
    agent: str                          # "berth" | "crane" | "yard" | "trucks" | "coordinator" | ...
    reason: str                         # required, one plain-English sentence


@dataclass(frozen=True)
class Request:
    agent: str
    resource: str                       # one of RESOURCES
    amount: int
    call_id: str
    priority: float                     # higher = more important
    reason: str


@dataclass(frozen=True)
class Grant:
    request: Request
    granted: int
    status: str                         # one of GRANT_STATUSES
    reason: str


@dataclass(frozen=True)
class Option:
    """A complete alternative plan for the supervisor to choose from."""
    id: str
    actions: tuple[Action, ...]
    predicted: Mapping[str, Any]        # e.g. {"ships_departed": 9, "finish_h": {"P02-003": 41.5}}
    template: str                       # code-written explanation; used when no LLM


@dataclass(frozen=True)
class Decision:
    """One line of runs/<scenario>/decisions.jsonl (SAFE-3)."""
    tick: int
    t: float                            # simulated hours since scenario start
    agent: str
    saw: Mapping[str, Any]              # compact facts the decision was based on
    options: tuple[str, ...]            # option ids considered (empty if none)
    choice: str                         # chosen option id, or the action tool name
    reason: str
    guard_ok: bool
    guard_reason: str
    tool_result: Optional[Mapping[str, Any]] = None   # None if the guard refused
    backend: str = "rules"              # "rules" | "local" | "groq": who made the choice


# --------------------------------------------------------------------------
# Interfaces (see CONTRACTS.md)
# --------------------------------------------------------------------------

@runtime_checkable
class Agent(Protocol):
    name: str

    def propose(self, state: WorldState, events: list[Event]) -> tuple[list[Action], list[Request]]:
        """Return actions and resource requests. Must not call the simulator."""
        ...
