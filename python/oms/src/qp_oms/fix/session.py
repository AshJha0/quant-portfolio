"""FIX 4.4 session layer (W28b): logon, heartbeats, sequence numbers, resend.

Plain idea
----------
The session layer makes an unreliable byte pipe look like an ordered,
gap-free, at-least-once message stream. Each side numbers what it sends
(``34`` MsgSeqNum) and tracks the number it expects next. A gap makes the
receiver ask for a resend; the sender replays application messages with
``43=Y`` (PossDup) and replaces session messages with a SequenceReset in
gap-fill mode. Sequence numbers and sent messages are persisted *before*
the bytes leave (write-ahead), so a restart resumes where it stopped.

Formal statement (FIX 4.4 Vol. 2, "Session Protocol")
-----------------------------------------------------
Let ``E`` be the next expected inbound MsgSeqNum and ``s`` the received one.

* ``s = E``: process, ``E <- E + 1``.
* ``s > E``: queue the message, send ResendRequest(``7=E``, ``16=0``) once
  for the gap, and process queued messages in order once the gap closes.
  A SequenceReset-Reset (``123`` absent or ``N``) is applied whatever ``s``.
* ``s < E``: with ``43=Y`` ignore it (a duplicate); without, the stream is
  broken: send Logout and close ("MsgSeqNum too low").

On ResendRequest(``b``, ``e``) (``e = 0`` means "to the last sent"), stored
application messages ``b..e`` are re-sent with their original MsgSeqNum,
``43=Y`` and ``122`` OrigSendingTime; runs of session messages (Logon,
Logout, Heartbeat, TestRequest, ResendRequest, SequenceReset) become one
SequenceReset ``123=Y`` with ``36`` NewSeqNo = next number after the run.

Heartbeats: with ``108=H`` seconds, a side sends a Heartbeat after ``H``
seconds of silence on its outbound side; after ``H`` seconds plus a grace
(20%) of inbound silence it sends a TestRequest; if nothing arrives for a
further ``H`` seconds it closes the session.

Acceptor authentication: the acceptor refuses a Logon whose
SenderCompID/TargetCompID do not match its configuration or whose
Username (553)/Password (554) fail the configured check; it answers with a
Logout and closes, and never sends its own Logon. Passwords are redacted
before the Logon is persisted.

Application layer: ExecutionReports (35=8) are passed up at most once per
ExecID (17); the seen set is persisted with the sequence numbers, so a
resent or PossDup report after a restart is still recognised.

Worked example: see ``tests/test_session.py::test_gap_is_filled_by_resend``:
the initiator's message 3 is lost, the acceptor receives 4, asks for 3..0,
and gets 3 again with ``43=Y``, then processes 4.

Scope: one session per object, no network (an in-memory link stands in for
TCP); FIX 4.4 only; no encryption (``98=0``). QuickFIX interoperability is
the optional job listed in the W28 plan, not covered here.
"""

from __future__ import annotations

import enum
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from qp_core.eventlog import EventLog, FsyncPolicy

from qp_oms.fix.codec import FixDecodeError, FixMessage, decode, encode

__all__ = [
    "ADMIN_TYPES",
    "FixSession",
    "InMemoryLink",
    "Role",
    "SessionConfig",
    "SessionState",
    "SessionStore",
]

ADMIN_TYPES = frozenset({"0", "1", "2", "4", "5", "A"})  # gap-filled on resend; 3 (Reject) is resent
_GRACE = 1.2
_NS = 1_000_000_000


class Role(enum.Enum):
    INITIATOR = "initiator"
    ACCEPTOR = "acceptor"


class SessionState(enum.Enum):
    DISCONNECTED = "disconnected"
    LOGON_SENT = "logon_sent"
    ACTIVE = "active"
    LOGOUT_SENT = "logout_sent"
    CLOSED = "closed"


def _fmt_time(ns: int) -> str:
    import datetime as _dt

    t = _dt.datetime.fromtimestamp(ns // _NS, tz=_dt.UTC)
    return t.strftime("%Y%m%d-%H:%M:%S") + f".{(ns % _NS) // 1_000_000:03d}"


class SessionStore:
    """Persisted sequence numbers, sent messages and seen ExecIDs.

    Backed by an :class:`EventLog` (hash-chained, fsync per record by
    default). Records: ``fix.out`` {seq, raw}, ``fix.in`` {seq},
    ``fix.execid`` {id}, ``fix.reset`` {out, in}.
    """

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self.next_out = 1
        self.next_in = 1
        self.sent: dict[int, bytes] = {}
        self.exec_ids: set[str] = set()
        for rec in log:
            p = rec.payload
            if rec.kind == "fix.out":
                self.sent[int(p["seq"])] = str(p["raw"]).encode("latin-1")
                self.next_out = max(self.next_out, int(p["seq"]) + 1)
            elif rec.kind == "fix.in":
                self.next_in = int(p["seq"]) + 1
            elif rec.kind == "fix.execid":
                self.exec_ids.add(str(p["id"]))
            elif rec.kind == "fix.reset":
                self.next_out, self.next_in = int(p["out"]), int(p["in"])
                if p.get("clear"):
                    self.sent.clear()

    @classmethod
    def open(
        cls, path: str | os.PathLike[str], *, fsync: FsyncPolicy = FsyncPolicy.EVERY_RECORD
    ) -> SessionStore:
        return cls(EventLog(path, fsync=fsync, clock="session"))

    @classmethod
    def in_memory(cls) -> SessionStore:
        return cls(EventLog.in_memory())

    def record_out(self, seq: int, raw: bytes, ts_ns: int) -> None:
        self.log.append("fix.out", {"seq": seq, "raw": raw.decode("latin-1")}, ts_ns=ts_ns)
        self.sent[seq] = raw
        self.next_out = seq + 1

    def record_in(self, seq: int, ts_ns: int) -> None:
        self.log.append("fix.in", {"seq": seq}, ts_ns=ts_ns)
        self.next_in = seq + 1

    def record_exec_id(self, exec_id: str, ts_ns: int) -> None:
        self.log.append("fix.execid", {"id": exec_id}, ts_ns=ts_ns)
        self.exec_ids.add(exec_id)

    def set_next_in(self, seq: int, ts_ns: int) -> None:
        self.log.append("fix.reset", {"out": self.next_out, "in": seq}, ts_ns=ts_ns)
        self.next_in = seq

    def reset(self, ts_ns: int) -> None:
        self.log.append("fix.reset", {"out": 1, "in": 1, "clear": True}, ts_ns=ts_ns)
        self.next_out = self.next_in = 1
        self.sent.clear()


@dataclass
class SessionConfig:
    sender: str
    target: str
    role: Role
    heartbeat_s: int = 30
    username: str | None = None  # initiator: sent in 553
    password: str | None = None  # initiator: sent in 554 (redacted in the store)
    authenticate: Callable[[str | None, str | None], bool] | None = None  # acceptor check
    reset_on_logon: bool = False


@dataclass
class _Stats:
    garbled: int = 0
    duplicates_ignored: int = 0
    exec_dupes: int = 0
    resend_requests_sent: int = 0
    resend_requests_served: int = 0
    rejects_sent: int = 0
    messages: list[str] = field(default_factory=list)


class FixSession:
    """One side of a FIX 4.4 session over a ``send(bytes)`` callable."""

    def __init__(
        self,
        cfg: SessionConfig,
        store: SessionStore,
        send: Callable[[bytes], None],
        *,
        clock: Callable[[], int],
        on_app: Callable[[FixMessage], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self._send = send
        self._clock = clock
        self._on_app = on_app or (lambda m: None)
        self.state = SessionState.DISCONNECTED
        self.close_reason: str | None = None
        self.stats = _Stats()
        self._queue: dict[int, bytes] = {}
        self._resend_until: int | None = None
        now = clock()
        self._last_out = now
        self._last_in = now
        self._test_req_at: int | None = None
        self._test_req_n = 0

    # -- outbound -------------------------------------------------------
    def _header(self, seq: int, now: int) -> list[tuple[int, str]]:
        return [(49, self.cfg.sender), (56, self.cfg.target), (34, str(seq)), (52, _fmt_time(now))]

    def _emit(self, msg_type: str, fields: Iterable[tuple[int, str]]) -> int:
        now = self._clock()
        seq = self.store.next_out
        body = list(fields)
        raw = encode(msg_type, [*self._header(seq, now), *body])
        stored = raw
        if msg_type == "A" and any(t == 554 for t, _ in body):
            stored = encode(
                msg_type, [*self._header(seq, now), *((t, "***" if t == 554 else v) for t, v in body)]
            )
        self.store.record_out(seq, stored, now)  # write-ahead: persist, then send
        self._send(raw)
        self._last_out = now
        return seq

    def logon(self) -> None:
        """Initiator: send Logon (35=A)."""
        if self.cfg.role is not Role.INITIATOR:
            raise RuntimeError("only the initiator sends the first Logon")
        if self.state is not SessionState.DISCONNECTED:
            raise RuntimeError(f"logon from state {self.state.name}")
        if self.cfg.reset_on_logon:
            self.store.reset(self._clock())
        self._emit("A", self._logon_fields())
        self.state = SessionState.LOGON_SENT

    def _logon_fields(self) -> list[tuple[int, str]]:
        f = [(98, "0"), (108, str(self.cfg.heartbeat_s))]
        if self.cfg.reset_on_logon:
            f.append((141, "Y"))
        if self.cfg.username is not None:
            f.append((553, self.cfg.username))
        if self.cfg.password is not None:
            f.append((554, self.cfg.password))
        return f

    def send_app(self, msg_type: str, fields: Iterable[tuple[int, str]]) -> int:
        """Send an application message (35=D, G, F, 8, q, r, ...); returns its MsgSeqNum."""
        if msg_type in ADMIN_TYPES or msg_type == "3":
            raise ValueError(f"35={msg_type} is a session message")
        if self.state is not SessionState.ACTIVE:
            raise RuntimeError(f"cannot send application messages in state {self.state.name}")
        return self._emit(msg_type, fields)

    def logout(self, text: str = "") -> None:
        if self.state in (SessionState.CLOSED, SessionState.LOGOUT_SENT):
            return
        self._emit("5", [(58, text)] if text else [])
        self.state = SessionState.LOGOUT_SENT

    def _close(self, reason: str) -> None:
        self.state = SessionState.CLOSED
        self.close_reason = reason

    def _fatal(self, reason: str) -> None:
        if self.state is not SessionState.CLOSED:
            self._emit("5", [(58, reason)])
        self._close(reason)

    def disconnect(self, reason: str = "transport closed") -> None:
        """The transport went away (e.g. kill -9 of the peer)."""
        self._close(reason)

    # -- timer ----------------------------------------------------------
    def on_timer(self) -> None:
        """Call periodically: heartbeats, test requests, dead-peer detection."""
        if self.state is not SessionState.ACTIVE:
            return
        now = self._clock()
        hb = self.cfg.heartbeat_s * _NS
        if self._test_req_at is not None and now - self._test_req_at >= hb:
            self._fatal("no response to TestRequest")
            return
        if self._test_req_at is None and now - self._last_in >= int(hb * _GRACE):
            self._test_req_n += 1
            self._emit("1", [(112, f"TEST{self._test_req_n}")])
            self._test_req_at = now
            return
        if now - self._last_out >= hb:
            self._emit("0", [])

    # -- inbound --------------------------------------------------------
    def on_bytes(self, raw: bytes) -> None:
        """Handle one framed inbound message."""
        if self.state is SessionState.CLOSED:
            return
        try:
            msg = decode(raw)
            seq = int(msg.require(34))
            msg.require(49)
            msg.require(56)
        except (FixDecodeError, ValueError):
            self.stats.garbled += 1  # FIX: garbled messages are ignored, not answered
            return
        now = self._clock()
        self._last_in = now
        self._test_req_at = None
        self.stats.messages.append(msg.msg_type)

        if msg.get(49) != self.cfg.target or msg.get(56) != self.cfg.sender:
            self._fatal("CompID problem")
            return
        if self.state in (SessionState.DISCONNECTED, SessionState.LOGON_SENT) and msg.msg_type != "A":
            if msg.msg_type == "5":
                self._close(msg.get(58) or "logout before logon")
                return
            self._fatal("first message must be Logon")
            return
        if msg.msg_type == "A" and self.state is SessionState.DISCONNECTED:
            if not self._accept_logon(msg):
                return
            if msg.get(141) == "Y":
                self.store.reset(now)
        # SequenceReset-Reset ignores sequence checks.
        if msg.msg_type == "4" and msg.get(123) != "Y":
            self._apply_reset(int(msg.require(36)), now)
            return
        self._sequenced(seq, msg, raw, now)

    def _accept_logon(self, msg: FixMessage) -> bool:
        if self.cfg.role is Role.ACCEPTOR:
            auth = self.cfg.authenticate
            if auth is not None and not auth(msg.get(553), msg.get(554)):
                self._fatal("authentication failed")
                return False
        return True

    def _sequenced(self, seq: int, msg: FixMessage, raw: bytes, now: int) -> None:
        exp = self.store.next_in
        if seq < exp:
            if msg.get(43) == "Y":
                self.stats.duplicates_ignored += 1
                return
            self._fatal(f"MsgSeqNum too low, expecting {exp} but received {seq}")
            return
        if seq > exp:
            if msg.msg_type == "A":
                self._process(msg, now)  # complete the logon, then ask for the gap
            elif msg.msg_type == "5":
                self._process(msg, now)
                return
            else:
                self._queue[seq] = raw
            if self._resend_until is None or exp > self._resend_until:
                self._resend_until = seq
                self.stats.resend_requests_sent += 1
                self._emit("2", [(7, str(exp)), (16, "0")])
            return
        self._in_order(seq, msg, now)
        self._drain(now)

    def _in_order(self, seq: int, msg: FixMessage, now: int) -> None:
        """Handle the message whose MsgSeqNum equals the expected one."""
        exp = self.store.next_in
        if msg.msg_type == "4":  # gap fill
            new_seq = int(msg.require(36))
            if new_seq <= exp:
                self.store.record_in(seq, now)
                self._reject(seq, "NewSeqNo not above expected")
                return
            self.store.set_next_in(new_seq, now)
        else:
            self.store.record_in(seq, now)
            self._process(msg, now)

    def _drain(self, now: int) -> None:
        """Process queued messages that are now in order."""
        while self.state is not SessionState.CLOSED and self._queue:
            exp = self.store.next_in
            for s in [s for s in self._queue if s < exp]:
                del self._queue[s]
            raw = self._queue.pop(exp, None)
            if raw is None:
                break
            self._in_order(exp, decode(raw), now)
        if not self._queue:
            self._resend_until = None

    def _apply_reset(self, new_seq: int, now: int) -> None:
        if new_seq < self.store.next_in:
            self._reject(self.store.next_in, "SequenceReset to a lower number")
            return
        self.store.set_next_in(new_seq, now)
        self._drain(now)

    def _reject(self, ref_seq: int, text: str) -> None:
        self.stats.rejects_sent += 1
        self._emit("3", [(45, str(ref_seq)), (58, text)])

    def _process(self, msg: FixMessage, now: int) -> None:
        t = msg.msg_type
        if t == "A":
            if self.state is SessionState.DISCONNECTED:  # acceptor answers
                hb = msg.get(108)
                if hb is not None and hb.isdigit():
                    self.cfg.heartbeat_s = int(hb)
                fields = [(98, "0"), (108, str(self.cfg.heartbeat_s))]
                if msg.get(141) == "Y":
                    fields.append((141, "Y"))
                self._emit("A", fields)
            self.state = SessionState.ACTIVE
        elif t == "0":
            pass
        elif t == "1":
            self._emit("0", [(112, msg.require(112))])
        elif t == "2":
            self._serve_resend(int(msg.require(7)), int(msg.require(16)))
        elif t == "3":
            pass  # session reject: recorded by the store; the app may inspect stats.messages
        elif t == "5":
            if self.state is not SessionState.LOGOUT_SENT:
                self._emit("5", [])
            self._close(msg.get(58) or "peer logout")
        else:
            if t == "8":
                eid = msg.get(17)
                if eid is not None:
                    if eid in self.store.exec_ids:
                        self.stats.exec_dupes += 1
                        return
                    self.store.record_exec_id(eid, now)
            self._on_app(msg)

    def _serve_resend(self, begin: int, end: int) -> None:
        self.stats.resend_requests_served += 1
        last = self.store.next_out - 1
        end = last if end == 0 or end > last else end
        now = self._clock()
        gap_start: int | None = None

        def flush_gap(upto: int) -> None:
            nonlocal gap_start
            if gap_start is not None:
                raw = encode(
                    "4",
                    [
                        *self._header(gap_start, now),
                        (43, "Y"),
                        (122, _fmt_time(now)),
                        (123, "Y"),
                        (36, str(upto)),
                    ],
                )
                self._send(raw)  # not persisted: it reuses an already-stored number
                gap_start = None

        for s in range(begin, end + 1):
            stored = self.store.sent.get(s)
            if stored is None or decode(stored).msg_type in ADMIN_TYPES:
                if gap_start is None:
                    gap_start = s
                continue
            flush_gap(s)
            old = decode(stored)
            orig_time = old.get(52) or _fmt_time(now)
            body = [(t, v) for t, v in old.body() if t not in (49, 56, 34, 52, 43, 122)]
            raw = encode(old.msg_type, [*self._header(s, now), (43, "Y"), (122, orig_time), *body])
            self._send(raw)
        flush_gap(end + 1)
        self._last_out = now


class InMemoryLink:
    """Two-way in-memory transport with optional loss, for tests and demos.

    ``a_to_b`` / ``b_to_a`` hold bytes in flight; :meth:`pump` delivers
    until both are empty. ``drop`` is called on each message and returns
    True to lose it.
    """

    def __init__(self) -> None:
        self.a_to_b: list[bytes] = []
        self.b_to_a: list[bytes] = []
        self.a: FixSession | None = None
        self.b: FixSession | None = None
        self.drop: Callable[[str, bytes], bool] = lambda direction, raw: False

    def send_a(self, raw: bytes) -> None:
        if not self.drop("a->b", raw):
            self.a_to_b.append(raw)

    def send_b(self, raw: bytes) -> None:
        if not self.drop("b->a", raw):
            self.b_to_a.append(raw)

    def pump(self, max_steps: int = 10_000) -> None:
        for _ in range(max_steps):
            if self.a_to_b and self.b is not None:
                self.b.on_bytes(self.a_to_b.pop(0))
            elif self.b_to_a and self.a is not None:
                self.a.on_bytes(self.b_to_a.pop(0))
            else:
                return
        raise RuntimeError("link did not settle")
