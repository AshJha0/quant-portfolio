"""Kill-switch hierarchy: the reference implementation of docs/KILL_CONTRACT.md.

Rules K1-K7 are named in the docstrings where they are enforced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

WILD = "*"


class Level(StrEnum):
    FIRM = "FIRM"
    DESK = "DESK"
    STRATEGY = "STRATEGY"
    SYMBOL = "SYMBOL"


_DEPTH = {Level.FIRM: 1, Level.DESK: 2, Level.STRATEGY: 3, Level.SYMBOL: 4}


class KillState(StrEnum):
    ENGAGED = "ENGAGED"
    KILLED = "KILLED"
    RESET_PENDING = "RESET_PENDING"
    CLEARED = "CLEARED"


Path = tuple[str, str, str, str]


def make_scope(level: Level, path: Path) -> Path:
    """Truncate an order path to a kill scope at ``level`` (suffix wildcards)."""
    n = _DEPTH[level]
    return (path[0], *(path[i] if i < n else WILD for i in range(1, 4)))  # type: ignore[return-value]


def validate_scope(scope: Path) -> Level:
    """Return the level of a scope; raise if wildcards are not a pure suffix."""
    if len(scope) != 4:
        raise ValueError("scope must have four fields")
    n = 0
    while n < 4 and scope[n] != WILD:
        n += 1
    if n == 0 or any(s != WILD for s in scope[n:]):
        raise ValueError(f"invalid scope {scope!r}: wildcards must be a suffix after firm")
    return {1: Level.FIRM, 2: Level.DESK, 3: Level.STRATEGY, 4: Level.SYMBOL}[n]


def covers(scope: Path, path: Path) -> bool:
    return all(s == WILD or s == p for s, p in zip(scope, path, strict=True))


@dataclass
class Kill:
    kill_id: int
    level: Level
    scope: Path
    actor: str
    reason: str
    state: KillState = KillState.ENGAGED
    outstanding: set[int] = field(default_factory=set)
    reset_requester: str | None = None


@dataclass(frozen=True)
class KillEvent:
    seq: int
    event: str
    kill_id: int
    level: str
    scope: Path
    actor: str
    state: str
    cancel: tuple[int, ...] = ()
    reason: str = ""

    def to_wire(self) -> str:
        """Canonical JSON: sorted keys, no spaces (the parity form)."""
        d = {
            "seq": self.seq,
            "event": self.event,
            "kill_id": self.kill_id,
            "level": self.level,
            "scope": list(self.scope),
            "actor": self.actor,
            "state": self.state,
            "cancel": list(self.cancel),
            "reason": self.reason,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":"))


class KillSwitch:
    """One firm-wide kill hierarchy.

    The switch does not send messages. It tells the caller which orders to
    cancel and whether a new order may go out; the OMS/gateway acts on that.
    """

    def __init__(self) -> None:
        self._kills: dict[int, Kill] = {}
        self._next_id = 1
        self._seq = 0
        self.events: list[KillEvent] = []
        # order id -> (path, terminal?)
        self._working: dict[int, Path] = {}

    # ---- order tracking -------------------------------------------------
    def on_order_working(self, order_id: int, path: Path) -> None:
        self._working[order_id] = path

    def on_order_terminal(self, order_id: int) -> None:
        """K3: a terminal report (cancel ack, fill, reject, expiry) for an order."""
        self._working.pop(order_id, None)
        for k in self._kills.values():
            if order_id in k.outstanding:
                k.outstanding.discard(order_id)
                if k.state is KillState.ENGAGED and not k.outstanding:
                    k.state = KillState.KILLED
                    self._emit("killed", k, "system", "")

    # ---- engage ---------------------------------------------------------
    def engage(self, scope: Path, actor: str, reason: str = "") -> tuple[int, list[int]]:
        """Engage a kill. Returns (kill_id, sorted order ids to cancel). K2, K7."""
        level = validate_scope(scope)
        for k in self._kills.values():
            if k.scope == scope and k.state is not KillState.CLEARED:
                self._emit("engage_dup", k, actor, reason)
                return k.kill_id, []
        k = Kill(self._next_id, level, scope, actor, reason)
        self._next_id += 1
        self._kills[k.kill_id] = k
        cancels = sorted(o for o, p in self._working.items() if covers(scope, p))
        k.outstanding = set(cancels)
        if not cancels:
            k.state = KillState.KILLED
        self._emit("engage", k, actor, reason, tuple(cancels))
        return k.kill_id, cancels

    def cancel_on_disconnect(self, session_paths: list[Path]) -> list[tuple[int, list[int]]]:
        """K6: engage a STRATEGY kill for every strategy on a dropped session."""
        strategies = sorted({make_scope(Level.STRATEGY, p) for p in session_paths})
        return [self.engage(s, "system:cod", "cancel_on_disconnect") for s in strategies]

    # ---- block ----------------------------------------------------------
    def check(self, path: Path) -> str | None:
        """K1: None if allowed, else the reject reason naming the widest kill."""
        active = [
            k for k in self._kills.values() if k.state is not KillState.CLEARED and covers(k.scope, path)
        ]
        if not active:
            return None
        k = min(active, key=lambda x: (_DEPTH[x.level], x.kill_id))
        return f"KILLED:{k.level.value}:{'/'.join(k.scope)}"

    # ---- reset ----------------------------------------------------------
    def request_reset(self, kill_id: int, actor: str) -> bool:
        k = self._kills[kill_id]
        if k.state is not KillState.KILLED:
            self._emit("reset_rejected", k, actor, f"state={k.state.value}")
            return False
        k.state = KillState.RESET_PENDING
        k.reset_requester = actor
        self._emit("reset_requested", k, actor, "")
        return True

    def approve_reset(self, kill_id: int, actor: str) -> bool:
        """K4: the approver must differ from the requester."""
        k = self._kills[kill_id]
        if k.state is not KillState.RESET_PENDING:
            self._emit("reset_rejected", k, actor, f"state={k.state.value}")
            return False
        if actor == k.reset_requester:
            self._emit("reset_rejected", k, actor, "same_actor")
            return False
        k.state = KillState.CLEARED
        self._emit("cleared", k, actor, "")
        return True

    def state(self, kill_id: int) -> KillState:
        return self._kills[kill_id].state

    def _emit(self, ev: str, k: Kill, actor: str, reason: str, cancel: tuple[int, ...] = ()) -> None:
        self._seq += 1
        self.events.append(
            KillEvent(self._seq, ev, k.kill_id, k.level.value, k.scope, actor, k.state.value, cancel, reason)
        )


def replay(commands: list[dict[str, Any]]) -> list[str]:
    """Run a golden-vector command list and return the wire events.

    Commands: {"op": "working", "id", "path"}, {"op": "terminal", "id"},
    {"op": "engage", "scope", "actor", "reason"}, {"op": "check", "path"},
    {"op": "request_reset"|"approve_reset", "kill_id", "actor"},
    {"op": "disconnect", "paths"}. A "check" emits a synthetic event line.
    """
    ks = KillSwitch()
    out: list[str] = []
    for c in commands:
        before = len(ks.events)
        op = c["op"]
        if op == "working":
            ks.on_order_working(int(c["id"]), tuple(c["path"]))
        elif op == "terminal":
            ks.on_order_terminal(int(c["id"]))
        elif op == "engage":
            ks.engage(tuple(c["scope"]), str(c["actor"]), str(c.get("reason", "")))
        elif op == "disconnect":
            ks.cancel_on_disconnect([tuple(p) for p in c["paths"]])
        elif op == "request_reset":
            ks.request_reset(int(c["kill_id"]), str(c["actor"]))
        elif op == "approve_reset":
            ks.approve_reset(int(c["kill_id"]), str(c["actor"]))
        elif op == "check":
            r = ks.check(tuple(c["path"]))
            out.append(
                json.dumps({"check": list(c["path"]), "result": r}, sort_keys=True, separators=(",", ":"))
            )
            continue
        else:
            raise ValueError(f"unknown op {op!r}")
        out.extend(e.to_wire() for e in ks.events[before:])
    return out
