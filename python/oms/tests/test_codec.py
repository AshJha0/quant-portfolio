"""FIX codec tests: a published-style vector, round trips and a decoder fuzz."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qp_oms.fix import SOH, FixDecodeError, decode, encode, split_stream


def _wire(s: str) -> bytes:
    return s.replace("|", "\x01").encode()


def test_hand_checked_vector() -> None:
    # BodyLength and CheckSum computed by hand: body "35=0|49=BUY|56=SELL|34=2|"
    # is 25 bytes; sum of "8=FIX.4.4|9=25|" + body bytes is 2080, 2080 mod 256 = 32.
    raw = _wire("8=FIX.4.4|9=25|35=0|49=BUY|56=SELL|34=2|10=032|")
    assert sum(raw[: raw.rfind(b"10=")]) == 2080
    assert encode("0", [(49, "BUY"), (56, "SELL"), (34, "2")]) == raw
    assert decode(raw).fields == ((35, "0"), (49, "BUY"), (56, "SELL"), (34, "2"))


@pytest.mark.parametrize(
    ("bad", "why"),
    [
        ("8=FIX.4.4|9=25|35=0|49=BUY|56=SELL|34=2|10=033|", "CheckSum"),
        ("8=FIX.4.4|9=24|35=0|49=BUY|56=SELL|34=2|10=031|", "BodyLength"),
        ("8=FIX.4.2|9=25|35=0|49=BUY|56=SELL|34=2|10=030|", "BeginString"),
        ("8=FIX.4.4|9=25|49=BUY|35=0|56=SELL|34=2|10=032|", "MsgType"),
        ("8=FIX.4.4|9=25|35=0|49=BUY|56=SELL|34=2|10=032", "SOH"),
        ("8=FIX.4.4|9=25|35=0|49=BUY|56=SELL|34=|10=032|", "malformed"),
        ("8=FIX.4.4|9=25|35=0|x9=BUY|56=SELL|34=2|10=032|", "malformed"),
    ],
)
def test_decoder_rejects(bad: str, why: str) -> None:
    with pytest.raises(FixDecodeError, match=why):
        decode(_wire(bad))


def test_encoder_refuses_framing_tags_and_soh() -> None:
    with pytest.raises(ValueError):
        encode("D", [(10, "1")])
    with pytest.raises(ValueError):
        encode("D", [(58, "a\x01b")])
    with pytest.raises(ValueError):
        encode("D", [(58, "")])


_val = st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=12)
_fields = st.lists(st.tuples(st.integers(11, 9999).filter(lambda t: t != 35), _val), max_size=12)


@given(st.sampled_from(["D", "G", "F", "8", "q", "r", "0", "A"]), _fields)
def test_round_trip(msg_type: str, fields: list[tuple[int, str]]) -> None:
    raw = encode(msg_type, fields)
    m = decode(raw)
    assert m.msg_type == msg_type and list(m.fields[1:]) == fields


@settings(max_examples=500)
@given(st.binary(max_size=200))
def test_fuzz_decoder_only_raises_decode_error(data: bytes) -> None:
    try:
        decode(data)
    except FixDecodeError:
        pass


@settings(max_examples=300)
@given(_fields, st.integers(0, 10_000), st.integers(1, 255))
def test_fuzz_single_byte_mutation_is_caught(fields: list[tuple[int, str]], pos: int, delta: int) -> None:
    raw = bytearray(encode("D", fields))
    i = pos % len(raw)
    raw[i] = (raw[i] + delta) % 256
    try:
        decode(bytes(raw))
    except FixDecodeError:
        return
    raise AssertionError("a one-byte change passed BodyLength and CheckSum")


@given(st.lists(_fields, min_size=1, max_size=5), st.integers(1, 30))
def test_split_stream_reassembles_chunks(msgs: list[list[tuple[int, str]]], chunk: int) -> None:
    raws = [encode("D", f) for f in msgs]
    stream = b"junk" + b"".join(raws)
    out: list[bytes] = []
    buf = b""
    for i in range(0, len(stream), chunk):
        got, buf = split_stream(buf + stream[i : i + chunk])
        out.extend(got)
    assert out == raws
    assert all(r.endswith(SOH) for r in out)
