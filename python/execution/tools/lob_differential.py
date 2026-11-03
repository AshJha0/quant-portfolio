"""Acceptance run: fast book == reference book after every one of N events.

    python tools/lob_differential.py N SEED WORKDIR REPLAY

REPLAY is the exec_core_replay path, or ``wsl:/path`` to run it through WSL.
Streams everything, so 1e7 events need little memory (about 1.3 GB of files
in WORKDIR). Prints the event count, mismatches and the first divergence.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from qp_exec.lob import ReferenceBook
from qp_exec.lob.events import apply_event, random_events, write_events


def main() -> int:
    n, seed, work, replay = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    work.mkdir(parents=True, exist_ok=True)
    ev, dg = work / f"ev_{seed}.bin", work / f"dg_{seed}.bin"
    t0 = time.time()
    write_events(ev, random_events(n, seed, auction_every=7919))
    print(f"generated {n} events in {time.time() - t0:.0f}s", flush=True)
    if replay.startswith("wsl:"):
        def w(p: Path) -> str:
            s = str(p.resolve()).replace("\\", "/")
            return f"/mnt/{s[0].lower()}{s[2:]}"
        cmd = ["wsl", "-e", replay[4:], w(ev), w(dg)]
    else:
        cmd = [replay, str(ev), str(dg)]
    print(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout, flush=True)
    fast = np.memmap(dg, dtype="<u8", mode="r")
    book = ReferenceBook()
    bad = 0
    first = -1
    fills_total = 0
    from qp_exec.lob.events import read_events
    for i, e in enumerate(read_events(ev)):
        fills = apply_event(book, e)
        fills_total += len(fills)
        if book.digest(fills) != int(fast[i]):
            bad += 1
            if first < 0:
                first = i
    print(f"events {n} fills {fills_total} mismatches {bad} first {first} "
          f"elapsed {time.time() - t0:.0f}s", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
