"""The C++ fast book equals the reference book after every event.

Needs the ``exec_core_replay`` binary from ``cpp/exec-core``. Point
``QP_EXEC_REPLAY`` at it (on Windows, ``wsl:/path/in/wsl`` runs it through
WSL). Without it the test is skipped; CI builds the binary first.
``QP_EXEC_DIFF_EVENTS`` sets the stream length (default 50000; the
acceptance run is 1e7, see the package README).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from qp_exec.lob.events import random_events, replay_digests, write_events


def _to_wsl(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}"


def run_replay(events_path: Path, out_path: Path) -> str:
    exe = os.environ.get("QP_EXEC_REPLAY")
    if not exe:
        pytest.skip("QP_EXEC_REPLAY not set")
    if exe.startswith("wsl:"):
        cmd = ["wsl", "-e", exe[4:], _to_wsl(events_path), _to_wsl(out_path)]
    else:
        cmd = [exe, str(events_path), str(out_path)]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


@pytest.mark.parametrize("seed", [11, 12])
def test_fast_book_matches_reference_every_event(tmp_path: Path, seed: int) -> None:
    n = int(os.environ.get("QP_EXEC_DIFF_EVENTS", "50000"))
    events = list(random_events(n, seed=seed, auction_every=7919))
    ev_path, out_path = tmp_path / "ev.bin", tmp_path / "dg.bin"
    write_events(ev_path, events)
    run_replay(ev_path, out_path)
    ref = np.array(replay_digests(events), dtype=np.uint64)
    fast = np.fromfile(out_path, dtype="<u8")
    assert fast.shape == ref.shape
    bad = np.flatnonzero(fast != ref)
    assert bad.size == 0, f"first divergence at event {bad[0]}: {events[bad[0]]}"
