"""QPRC v1 recording format, as read by cpp/ull/feed/include/qp/ull/recorder.hpp.

Plain idea: keep every frame exactly as it came off the wire, with the
receive time and the line it came from, so any later parser can replay it.

Formal statement (all integers little-endian):

    header : b"QPRC", u16 version = 1, u16 reserved = 0
    record : u64 recv_ns, u8 line (0 = A, 1 = B, 255 = REST),
             u8 kind (0 = websocket frame, 1 = snapshot body), u32 len, bytes

The Rust crate (rust/ull/feed/src/recorder.rs) reads the same records behind
a different 6-byte magic, b"QPRR1\\n". :func:`to_qprr1` swaps the header; the
records are byte-identical.

Worked example: one frame at 1 ns on line B, payload ``{}``, is the header
followed by ``01 00 00 00 00 00 00 00 | 01 | 00 | 02 00 00 00 | 7b 7d``.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import BinaryIO

MAGIC = b"QPRC"
HEADER = MAGIC + struct.pack("<HH", 1, 0)
QPRR1_MAGIC = b"QPRR1\n"
REC = struct.Struct("<QBBI")

LINE_A, LINE_B, LINE_REST = 0, 1, 255
KIND_FRAME, KIND_SNAPSHOT = 0, 1


@dataclass(frozen=True)
class Record:
    recv_ns: int
    line: int
    kind: int
    data: bytes


class Writer:
    """Append records to a binary stream; flushes after every record so a
    crash loses at most the record being written."""

    def __init__(self, out: BinaryIO) -> None:
        self._out = out
        self.count = 0
        out.write(HEADER)
        out.flush()

    def write(self, recv_ns: int, line: int, kind: int, data: bytes) -> None:
        if line not in (LINE_A, LINE_B, LINE_REST) or kind not in (KIND_FRAME, KIND_SNAPSHOT):
            raise ValueError(f"bad line/kind {line}/{kind}")
        if len(data) > 0xFFFFFFFF:
            raise ValueError("frame too large")
        self._out.write(REC.pack(recv_ns, line, kind, len(data)) + data)
        self._out.flush()
        self.count += 1


def read(buf: bytes) -> tuple[list[Record], bool]:
    """All whole records, and whether the file ends inside a record."""
    if buf[:8] != HEADER:
        raise ValueError("not a QPRC v1 file")
    out: list[Record] = []
    pos = 8
    while pos < len(buf):
        if pos + REC.size > len(buf):
            return out, True
        ns, line, kind, n = REC.unpack_from(buf, pos)
        pos += REC.size
        if pos + n > len(buf):
            return out, True
        out.append(Record(ns, line, kind, buf[pos : pos + n]))
        pos += n
    return out, False


def iter_records(buf: bytes) -> Iterator[Record]:
    recs, _ = read(buf)
    yield from recs


def to_qprr1(buf: bytes) -> bytes:
    """Same records under the Rust crate's magic."""
    if buf[:8] != HEADER:
        raise ValueError("not a QPRC v1 file")
    return QPRR1_MAGIC + buf[8:]
