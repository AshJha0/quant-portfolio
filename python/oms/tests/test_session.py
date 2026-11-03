"""FIX session tests: logon/auth, heartbeats, gaps, resend, PossDup, persistence."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qp_oms.fix import (
    FixMessage,
    FixSession,
    InMemoryLink,
    Role,
    SessionConfig,
    SessionState,
    SessionStore,
    decode,
    encode,
)

NS = 1_000_000_000


class Clock:
    def __init__(self) -> None:
        self.t = 1_700_000_000 * NS

    def __call__(self) -> int:
        return self.t

    def advance(self, s: float) -> None:
        self.t += int(s * NS)


def _pair(
    *,
    clock: Clock | None = None,
    a_store: SessionStore | None = None,
    b_store: SessionStore | None = None,
    password: str = "pw",
) -> tuple[FixSession, FixSession, InMemoryLink, list[FixMessage], list[FixMessage]]:
    clk = clock or Clock()
    link = InMemoryLink()
    a_app: list[FixMessage] = []
    b_app: list[FixMessage] = []
    a = FixSession(
        SessionConfig("BUY", "SELL", Role.INITIATOR, heartbeat_s=30, username="u", password=password),
        a_store or SessionStore.in_memory(),
        link.send_a,
        clock=clk,
        on_app=a_app.append,
    )
    b = FixSession(
        SessionConfig("SELL", "BUY", Role.ACCEPTOR, authenticate=lambda u, p: (u, p) == ("u", "pw")),
        b_store or SessionStore.in_memory(),
        link.send_b,
        clock=clk,
        on_app=b_app.append,
    )
    link.a, link.b = a, b
    return a, b, link, a_app, b_app


def _up(a: FixSession, link: InMemoryLink) -> None:
    a.logon()
    link.pump()


def _order(i: int) -> list[tuple[int, str]]:
    return [(11, f"C{i}"), (55, "XYZ"), (54, "1"), (38, "100"), (40, "2"), (44, "10.5")]


def test_logon_handshake() -> None:
    a, b, link, *_ = _pair()
    _up(a, link)
    assert a.state is SessionState.ACTIVE and b.state is SessionState.ACTIVE
    assert a.store.next_out == 2 and b.store.next_in == 2 and a.store.next_in == 2


def test_acceptor_rejects_bad_password_and_never_sends_logon() -> None:
    a, b, link, *_ = _pair(password="wrong")
    a.logon()
    link.pump()
    assert b.state is SessionState.CLOSED and b.close_reason == "authentication failed"
    assert a.state is SessionState.CLOSED
    sent_by_b = [decode(r).msg_type for r in b.store.sent.values()]
    assert sent_by_b == ["5"]


def test_acceptor_rejects_wrong_compid() -> None:
    clk = Clock()
    link = InMemoryLink()
    a = FixSession(
        SessionConfig("EVIL", "SELL", Role.INITIATOR), SessionStore.in_memory(), link.send_a, clock=clk
    )
    b = FixSession(
        SessionConfig("SELL", "BUY", Role.ACCEPTOR), SessionStore.in_memory(), link.send_b, clock=clk
    )
    link.a, link.b = a, b
    a.logon()
    link.pump()
    assert b.state is SessionState.CLOSED and b.close_reason == "CompID problem"


def test_password_is_redacted_in_store() -> None:
    a, _b, link, *_ = _pair()
    _up(a, link)
    assert b"554=***" in a.store.sent[1] and b"554=pw" not in a.store.sent[1]


def test_app_messages_flow_in_both_directions() -> None:
    a, b, link, a_app, b_app = _pair()
    _up(a, link)
    a.send_app("D", _order(1))
    b.send_app("8", [(37, "O1"), (11, "C1"), (17, "X1"), (150, "0"), (39, "0"), (55, "XYZ"), (54, "1")])
    link.pump()
    assert [m.msg_type for m in b_app] == ["D"] and [m.msg_type for m in a_app] == ["8"]


def test_cannot_send_app_before_logon_or_admin_via_send_app() -> None:
    a, _b, link, *_ = _pair()
    with pytest.raises(RuntimeError):
        a.send_app("D", _order(1))
    _up(a, link)
    with pytest.raises(ValueError):
        a.send_app("0", [])


def test_gap_is_filled_by_resend() -> None:
    a, b, link, _a_app, b_app = _pair()
    _up(a, link)
    link.drop = lambda d, raw: d == "a->b" and b"\x0111=C2\x01" in raw and b"43=Y" not in raw
    for i in (1, 2, 3):
        a.send_app("D", _order(i))
    link.pump()
    assert [m.get(11) for m in b_app] == ["C1", "C2", "C3"]
    resent = [m for m in b_app if m.get(43) == "Y"]
    assert [m.get(11) for m in resent] == ["C2"] and resent[0].get(122) is not None
    assert b.stats.resend_requests_sent == 1 and b.store.next_in == a.store.next_out


def test_resend_gap_fills_admin_messages() -> None:
    clk = Clock()
    a, b, link, _a_app, b_app = _pair(clock=clk)
    _up(a, link)
    link.drop = lambda d, raw: d == "a->b"  # lose everything for a while
    a.send_app("D", _order(1))  # seq 2
    clk.advance(31)
    a.on_timer()  # heartbeat, seq 3
    a.send_app("D", _order(2))  # seq 4
    link.drop = lambda d, raw: False
    a.send_app("D", _order(3))  # seq 5 arrives, gap 2..4
    link.pump()
    # in-order processing: queued 5 waits for the resend of 2..4
    assert [m.get(11) for m in b_app] == ["C1", "C2", "C3"]
    assert b.store.next_in == 6 and "4" in b.stats.messages  # the heartbeat came back as a gap fill


def test_sequence_too_low_without_possdup_is_fatal() -> None:
    a, b, link, *_ = _pair()
    _up(a, link)
    a.store.next_out = 1  # a peer that lost its sequence state re-uses number 1
    a.send_app("D", _order(1))
    link.pump()
    assert b.state is SessionState.CLOSED and "too low" in (b.close_reason or "")


def test_possdup_below_expected_is_ignored() -> None:
    a, b, link, *_ = _pair()
    _up(a, link)
    a.send_app("D", _order(1))
    link.pump()
    m = decode(a.store.sent[2])
    b.on_bytes(encode(m.msg_type, [*m.body(), (43, "Y")]))
    assert b.state is SessionState.ACTIVE and b.stats.duplicates_ignored == 1


def test_exec_id_dedupe_survives_restart(tmp_path: Path) -> None:
    store_path = tmp_path / "buy.jsonl"
    a, b, link, a_app, _ = _pair(a_store=SessionStore.open(store_path))
    _up(a, link)
    er = [(37, "O1"), (11, "C1"), (17, "X1"), (150, "F"), (39, "2"), (32, "100"), (31, "10")]
    b.send_app("8", er)
    link.pump()
    assert len(a_app) == 1
    a.store.log.close()
    # restart the buy side from disk; the sell side re-sends X1 as a new message
    clk = Clock()
    link2 = InMemoryLink()
    got: list[FixMessage] = []
    a2 = FixSession(
        SessionConfig("BUY", "SELL", Role.INITIATOR, username="u", password="pw"),
        SessionStore.open(store_path),
        link2.send_a,
        clock=clk,
        on_app=got.append,
    )
    b2 = FixSession(b.cfg, b.store, link2.send_b, clock=clk)
    link2.a, link2.b = a2, b2
    _up(a2, link2)
    assert a2.state is SessionState.ACTIVE
    b2.send_app("8", er)
    link2.pump()
    assert got == [] and a2.stats.exec_dupes == 1
    a2.store.log.close()


def test_sequence_numbers_persist_across_restart(tmp_path: Path) -> None:
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a, b, link, _, _ = _pair(a_store=SessionStore.open(pa), b_store=SessionStore.open(pb))
    _up(a, link)
    a.send_app("D", _order(1))
    link.pump()
    out_before = a.store.next_out
    a.store.log.close()
    b.store.log.close()
    a2, b2, link2, _, b_app2 = _pair(a_store=SessionStore.open(pa), b_store=SessionStore.open(pb))
    assert a2.store.next_out == out_before and b2.store.next_in == out_before
    _up(a2, link2)
    a2.send_app("D", _order(2))
    link2.pump()
    assert a2.state is SessionState.ACTIVE and [m.get(11) for m in b_app2] == ["C2"]
    assert b2.stats.resend_requests_sent == 0
    a2.store.log.close()
    b2.store.log.close()


def test_restart_after_crash_before_send_recovers_by_resend(tmp_path: Path) -> None:
    """Write-ahead: the message is in the store but never reached the wire."""
    pa = tmp_path / "a.jsonl"
    a, b, link, _, b_app = _pair(a_store=SessionStore.open(pa))
    _up(a, link)
    link.drop = lambda d, raw: d == "a->b"
    a.send_app("D", _order(1))  # persisted, then lost (process killed mid-send)
    a.store.log.close()
    clk = Clock()
    link2 = InMemoryLink()
    a2 = FixSession(a.cfg, SessionStore.open(pa), link2.send_a, clock=clk)
    b2 = FixSession(b.cfg, b.store, link2.send_b, clock=clk, on_app=b_app.append)
    link2.a, link2.b = a2, b2
    _up(a2, link2)
    assert [m.get(11) for m in b_app] == ["C1"] and b_app[0].get(43) == "Y"
    a2.store.log.close()


def test_heartbeat_and_test_request_and_dead_peer() -> None:
    clk = Clock()
    a, b, link, *_ = _pair(clock=clk)
    _up(a, link)
    clk.advance(30)
    a.on_timer()
    link.pump()
    assert b.stats.messages[-1] == "0"
    # b goes silent: a sends TestRequest after 36 s of inbound silence ...
    link.drop = lambda d, raw: d == "b->a"
    clk.advance(7)
    a.on_timer()
    assert decode(a.store.sent[a.store.next_out - 1]).msg_type == "1"
    # ... and closes after a further 30 s without an answer.
    clk.advance(30)
    a.on_timer()
    assert a.state is SessionState.CLOSED and a.close_reason == "no response to TestRequest"


def test_test_request_is_answered_with_its_id() -> None:
    clk = Clock()
    a, b, link, *_ = _pair(clock=clk)
    _up(a, link)
    clk.advance(37)
    a.on_timer()
    link.pump()
    hb = decode(b.store.sent[b.store.next_out - 1])
    assert hb.msg_type == "0" and hb.get(112) == "TEST1" and a.state is SessionState.ACTIVE


def test_logout_handshake() -> None:
    a, b, link, *_ = _pair()
    _up(a, link)
    a.logout("bye")
    link.pump()
    assert a.state is SessionState.CLOSED and b.state is SessionState.CLOSED


def test_sequence_reset_reset_mode() -> None:
    a, b, link, *_ = _pair()
    _up(a, link)
    raw = encode("4", [(49, "BUY"), (56, "SELL"), (34, "99"), (52, "20260101-00:00:00.000"), (36, "50")])
    b.on_bytes(raw)
    assert b.store.next_in == 50 and b.state is SessionState.ACTIVE


def test_garbled_message_is_ignored() -> None:
    a, b, link, *_ = _pair()
    _up(a, link)
    b.on_bytes(b"8=FIX.4.4\x019=5\x0135=0\x0110=000\x01")
    assert b.stats.garbled == 1 and b.state is SessionState.ACTIVE


@settings(max_examples=60, deadline=None)
@given(st.lists(st.booleans(), min_size=1, max_size=25))
def test_property_random_loss_still_delivers_in_order_once(lost: list[bool]) -> None:
    """Lose any subset of first transmissions; resend delivers each order exactly once, in order."""
    a, b, link, _, b_app = _pair()
    _up(a, link)
    plan = iter(lost)

    def drop(d: str, raw: bytes) -> bool:
        if d != "a->b" or b"\x0135=D\x01" not in raw or b"\x0143=Y\x01" in raw:
            return False
        return next(plan, False)

    link.drop = drop
    for i in range(len(lost)):
        a.send_app("D", _order(i))
        link.pump()
    a.send_app("D", _order(999))  # a final message exposes any trailing gap
    link.pump()
    assert [m.get(11) for m in b_app] == [f"C{i}" for i in range(len(lost))] + ["C999"]
    assert b.store.next_in == a.store.next_out
