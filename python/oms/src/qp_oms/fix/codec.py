"""FIX tag=value codec (W28b).

Plain idea
----------
A FIX message is a run of ``tag=value`` fields, each ended by the SOH byte
(``0x01``). Three fields frame it: ``8`` BeginString first, ``9``
BodyLength second (bytes from the field after it up to and including the
SOH before ``10``), and ``10`` CheckSum last (byte sum of everything before
it, mod 256, as three digits). ``35`` MsgType must be the third field.

Formal statement
----------------
For a message ``m = H || B || T`` with ``H = "8=FIX.4.4|9=<n>|"``,
body ``B`` and trailer ``T = "10=<ccc>|"``: ``n = len(B)`` in bytes and
``ccc = (sum of bytes of H || B) mod 256``, zero padded (FIX 4.4 Vol. 1,
"Message Format" and "CheckSum Calculation").

Worked example
--------------
>>> raw = encode("0", [(49, "BUY"), (56, "SELL"), (34, "2")])
>>> raw.replace(SOH, b"|")
b'8=FIX.4.4|9=25|35=0|49=BUY|56=SELL|34=2|10=032|'
>>> decode(raw).get(56)
'SELL'

The decoder is strict: any framing, length, checksum, tag or ordering
error raises :class:`FixDecodeError`, never another exception type (the
fuzz test in ``tests/test_codec.py`` holds this over random bytes).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = ["BEGIN_STRING", "SOH", "FixDecodeError", "FixMessage", "decode", "encode", "split_stream"]

SOH = b"\x01"
BEGIN_STRING = "FIX.4.4"


class FixDecodeError(ValueError):
    """The bytes are not one well-formed FIX message."""


@dataclass(frozen=True)
class FixMessage:
    """A decoded message: fields in wire order, excluding 8, 9 and 10."""

    fields: tuple[tuple[int, str], ...]

    @property
    def msg_type(self) -> str:
        return self.fields[0][1]

    def get(self, tag: int, default: str | None = None) -> str | None:
        for t, v in self.fields:
            if t == tag:
                return v
        return default

    def require(self, tag: int) -> str:
        v = self.get(tag)
        if v is None:
            raise FixDecodeError(f"required tag {tag} missing from 35={self.msg_type}")
        return v

    def body(self) -> list[tuple[int, str]]:
        """Fields after 35, for re-encoding."""
        return list(self.fields[1:])


def _checksum(data: bytes) -> str:
    return f"{sum(data) % 256:03d}"


def _field(tag: int, value: str) -> bytes:
    if not isinstance(tag, int) or tag <= 0:
        raise ValueError(f"bad tag {tag!r}")
    b = str(value).encode("ascii")
    if SOH in b or not b:
        raise ValueError(f"tag {tag}: value must be non-empty and contain no SOH")
    return f"{tag}=".encode("ascii") + b + SOH


def encode(msg_type: str, fields: Iterable[tuple[int, str]], *, begin: str = BEGIN_STRING) -> bytes:
    """Frame one message. ``fields`` must not contain tags 8, 9, 10 or 35."""
    body = _field(35, msg_type)
    for tag, value in fields:
        if tag in (8, 9, 10, 35):
            raise ValueError(f"tag {tag} is set by the codec")
        body += _field(tag, value)
    head = _field(8, begin) + _field(9, str(len(body)))
    return head + body + _field(10, _checksum(head + body))


def _parse_fields(raw: bytes) -> list[tuple[int, str]]:
    if not raw.endswith(SOH):
        raise FixDecodeError("message does not end with SOH")
    out: list[tuple[int, str]] = []
    for part in raw[:-1].split(SOH):
        tag_b, sep, val_b = part.partition(b"=")
        if not sep or not tag_b.isdigit() or not val_b or (len(tag_b) > 1 and tag_b[:1] == b"0"):
            raise FixDecodeError(f"malformed field {part[:20]!r}")
        try:
            out.append((int(tag_b), val_b.decode("ascii")))
        except UnicodeDecodeError as exc:
            raise FixDecodeError("non-ASCII value") from exc
    return out


def decode(raw: bytes, *, begin: str = BEGIN_STRING) -> FixMessage:
    """Validate framing, BodyLength and CheckSum; return the fields."""
    fields = _parse_fields(raw)
    if len(fields) < 4:
        raise FixDecodeError("too few fields")
    if fields[0] != (8, begin):
        raise FixDecodeError(f"BeginString must be 8={begin}")
    if fields[1][0] != 9 or not fields[1][1].isdigit():
        raise FixDecodeError("BodyLength (9) must be the second field")
    if fields[2][0] != 35:
        raise FixDecodeError("MsgType (35) must be the third field")
    if fields[-1][0] != 10:
        raise FixDecodeError("CheckSum (10) must be the last field")
    trailer_at = raw.rfind(SOH + b"10=") + 1
    head_len = len(f"8={begin}".encode()) + 1 + len(f"9={fields[1][1]}".encode()) + 1
    if int(fields[1][1]) != trailer_at - head_len:
        raise FixDecodeError(f"BodyLength {fields[1][1]} != {trailer_at - head_len}")
    if fields[-1][1] != _checksum(raw[:trailer_at]):
        raise FixDecodeError(f"CheckSum {fields[-1][1]} != {_checksum(raw[:trailer_at])}")
    middle = fields[2:-1]
    if any(t in (8, 9, 10) for t, _ in middle):
        raise FixDecodeError("framing tag repeated inside the body")
    return FixMessage(tuple(middle))


def split_stream(buf: bytes, *, begin: str = BEGIN_STRING) -> tuple[list[bytes], bytes]:
    """Cut complete messages off the front of a byte stream.

    Returns (messages, remainder). Garbage before a ``8=`` start is
    skipped; the messages are not validated (call :func:`decode`).
    """
    start_tag = f"8={begin}".encode() + SOH + b"9="
    msgs: list[bytes] = []
    while True:
        i = buf.find(start_tag)
        if i < 0:
            return msgs, buf[-(len(start_tag) - 1) :] if len(buf) >= len(start_tag) else buf
        buf = buf[i:]
        j = buf.find(SOH, len(start_tag))
        if j < 0:
            return msgs, buf
        n_txt = buf[len(start_tag) : j]
        if not n_txt.isdigit():
            buf = buf[1:]
            continue
        end_body = j + 1 + int(n_txt)
        end = end_body + 7  # "10=ccc" + SOH
        if len(buf) < end:
            return msgs, buf
        msgs.append(buf[:end])
        buf = buf[end:]


def fields_of(msg: FixMessage, tags: Sequence[int]) -> dict[int, str | None]:
    """Convenience: the values of ``tags`` (None when absent)."""
    return {t: msg.get(t) for t in tags}
