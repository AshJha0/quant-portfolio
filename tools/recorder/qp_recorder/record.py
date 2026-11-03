"""Record public, no-auth crypto market data into QPRC v1 files.

Venues and channels (all free, no account, no key):

- ``binance``: spot diff-depth ``<symbol>@depth@100ms`` on one or two
  independent websocket connections (lines A and B), plus the REST book
  ``GET /api/v3/depth`` fetched a few seconds after the stream starts and
  again after every reconnect.
- ``coinbase``: Coinbase Exchange ``level2_batch`` (served as ``level2_50``),
  whose first message is an in-band snapshot. Coinbase's ``full`` and
  ``level2`` channels need an authenticated connection since 2022-2023 (the
  server answers "level2, level3, and full channels now require
  authentication"); the recorder can ask for them with ``--channel`` but
  this project does not hold keys, so it stops with that message.

Every websocket text frame is written as received (UTF-8 bytes), with
``time.time_ns()`` at receipt. Nothing is parsed before it is written.

Usage (from the repository root)::

    python -m tools.recorder.qp_recorder.record binance --symbol BTCUSDT \
        --seconds 60 --lines 2 --out tools/recorder/recordings/x.qprc
    python -m tools.recorder.qp_recorder.record coinbase --product BTC-USD \
        --seconds 60 --out y.qprc

``--rotate-minutes N`` starts a new file every N minutes (name gets a UTC
timestamp suffix) for full-day captures; see tools/recorder/README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from .qprc import KIND_FRAME, KIND_SNAPSHOT, LINE_REST, Writer

BINANCE_WS = "wss://stream.binance.com:9443/ws/{sym}@depth@100ms"
BINANCE_REST = "https://api.binance.com/api/v3/depth?symbol={sym}&limit={limit}"
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
UA = {"User-Agent": "quant-portfolio-recorder/1 (public market data)"}


class RotatingSink:
    """One Writer at a time; a new file every ``rotate_s`` seconds."""

    def __init__(self, out: Path, rotate_s: float | None) -> None:
        self.base = out
        self.rotate_s = rotate_s
        self.paths: list[Path] = []
        self._fh: BinaryIO | None = None
        self._w: Writer | None = None
        self._opened = 0.0

    def _open(self) -> None:
        if self._fh:
            self._fh.close()
        if self.rotate_s:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            p = self.base.with_name(f"{self.base.stem}_{stamp}{self.base.suffix}")
        else:
            p = self.base
        p.parent.mkdir(parents=True, exist_ok=True)
        self._fh = p.open("wb")
        self._w = Writer(self._fh)
        self._opened = time.monotonic()
        self.paths.append(p)

    def write(self, line: int, kind: int, data: bytes) -> bool:
        """Write one record; True when this write rotated to a new file (the
        first file does not count), which then needs its own snapshot."""
        new = False
        if self._w is None:
            self._open()
        elif self.rotate_s and time.monotonic() - self._opened >= self.rotate_s:
            self._open()
            new = True
        assert self._w is not None
        self._w.write(time.time_ns(), line, kind, data)
        return new

    def close(self) -> None:
        if self._fh:
            self._fh.close()


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read()


class _Reconnect(Exception):
    """Raised by an on_rotate hook to force a fresh connection."""


async def _ws_line(
    url: str, line: int, sink: RotatingSink, stop: float, on_connect, on_rotate, subscribe=None
) -> None:
    import websockets  # PyPI, BSD-3; imported here so the offline parts need no install

    while time.monotonic() < stop:
        try:
            async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
                if subscribe is not None:
                    await ws.send(json.dumps(subscribe))
                await on_connect()
                while (left := stop - time.monotonic()) > 0:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=min(left, 30))
                    except TimeoutError:
                        continue
                    data = msg.encode() if isinstance(msg, str) else msg
                    if b'"type":"error"' in data[:40]:
                        sink.write(line, KIND_FRAME, data)
                        raise SystemExit(f"venue refused the subscription: {data[:300]!r}")
                    if sink.write(line, KIND_FRAME, data):
                        await on_rotate()  # new file: it needs its own snapshot
                return
        except _Reconnect:
            continue
        except (OSError, websockets.exceptions.ConnectionClosed) as e:
            print(f"line {line}: {e!r}; reconnecting in 1 s", file=sys.stderr)
            await asyncio.sleep(1)


async def record_binance(a: argparse.Namespace) -> list[Path]:
    sym = a.symbol.upper()
    sink = RotatingSink(Path(a.out), a.rotate_minutes * 60 if a.rotate_minutes else None)
    stop = time.monotonic() + a.seconds
    pending = asyncio.Lock()

    async def snapshot() -> None:
        # Binance: open the stream, buffer, then fetch the book (its
        # lastUpdateId must not be older than the first buffered event).
        async with pending:
            await asyncio.sleep(a.snapshot_delay)
            body = await asyncio.to_thread(_get, BINANCE_REST.format(sym=sym, limit=a.depth_limit))
            sink.write(LINE_REST, KIND_SNAPSHOT, body)

    tasks: set[asyncio.Task] = set()

    async def on_connect() -> None:
        t = asyncio.create_task(snapshot())
        tasks.add(t)

    async def nothing() -> None:
        return None

    url = BINANCE_WS.format(sym=sym.lower())
    # One snapshot per line-A (re)connect and per file rotation; line B only
    # adds frames. A second, unneeded snapshot is not harmless: a depth-limited
    # REST book loaded over a live book drops the levels beyond its depth.
    await asyncio.gather(
        *(
            _ws_line(url, ln, sink, stop, on_connect if ln == 0 else nothing, on_connect)
            for ln in range(a.lines)
        )
    )
    for t in tasks:
        if not t.done():
            t.cancel()
    sink.close()
    return sink.paths


async def record_coinbase(a: argparse.Namespace) -> list[Path]:
    sink = RotatingSink(Path(a.out), a.rotate_minutes * 60 if a.rotate_minutes else None)
    stop = time.monotonic() + a.seconds
    sub = {"type": "subscribe", "product_ids": [a.product], "channels": [a.channel]}

    async def nothing() -> None:
        return None

    # One line: level2 has no sequence numbers, so two lines cannot be
    # arbitrated by id (the C++ handler dedupes by payload, which only works
    # when both lines batch identically).
    async def reconnect() -> None:
        raise _Reconnect  # a fresh subscription starts with an in-band snapshot

    await _ws_line(COINBASE_WS, 0, sink, stop, nothing, reconnect, subscribe=sub)
    sink.close()
    return sink.paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="venue", required=True)
    b = sub.add_parser("binance")
    b.add_argument("--symbol", default="BTCUSDT")
    b.add_argument("--lines", type=int, choices=(1, 2), default=2)
    b.add_argument("--depth-limit", type=int, default=1000, choices=(5, 10, 20, 50, 100, 500, 1000, 5000))
    b.add_argument("--snapshot-delay", type=float, default=2.0)
    c = sub.add_parser("coinbase")
    c.add_argument("--product", default="BTC-USD")
    c.add_argument("--channel", default="level2_batch", choices=("level2_batch", "level2", "full"))
    for s in (b, c):
        s.add_argument("--seconds", type=float, required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--rotate-minutes", type=float, default=0)
    a = p.parse_args(argv)
    fn = record_binance if a.venue == "binance" else record_coinbase
    paths = asyncio.run(fn(a))
    for x in paths:
        print(x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
