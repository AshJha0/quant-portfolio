import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
from qp_oms.recovery import Journal, JournalCorrupt, SimClock, WallClock, read_journal, sim


def test_journal_roundtrip_and_clock_fields(tmp_path: Path) -> None:
    j = Journal(tmp_path / "j.log")
    j.append({"t": "a"}, SimClock().now())
    j.append({"t": "b"})
    j.close()
    recs, head = read_journal(tmp_path / "j.log")
    assert [r["t"] for r in recs] == ["a", "b"] and len(head) == 64
    assert recs[0]["ts_src"] == "sim" and recs[0]["ts_err_ns"] == 0


def test_torn_tail_truncated_and_appends_continue(tmp_path: Path) -> None:
    p = tmp_path / "j.log"
    j = Journal(p)
    j.append({"t": "a"})
    j.close()
    with open(p, "ab") as f:
        f.write(b'{"seq":2,"prev":"')
    j = Journal(p)
    assert len(j.records) == 1
    j.append({"t": "b"})
    j.close()
    assert [r["t"] for r in read_journal(p)[0]] == ["a", "b"]


def test_edit_in_the_middle_is_corruption_not_a_crash(tmp_path: Path) -> None:
    p = tmp_path / "j.log"
    j = Journal(p)
    for t in "abc":
        j.append({"t": t})
    j.close()
    p.write_bytes(p.read_bytes().replace(b'"t":"b"', b'"t":"B"'))
    with pytest.raises(JournalCorrupt):
        read_journal(p)


def test_wall_clock_is_monotonic_with_stated_bound() -> None:
    c = WallClock()
    a, b = c.now(), c.now()
    assert b.ns > a.ns and a.err_ns == 5_000_000 and a.source == "system/ntp"


def test_kill9_1000_seeded_runs_zero_unexplained_breaks(tmp_path: Path) -> None:
    """W28 acceptance: kill at random points in 1,000 seeded runs; restart reconciles."""
    n_never = n_rec = 0
    for seed in range(1000):
        r = sim.crash_run(tmp_path, seed)
        assert r.unexplained == [], (seed, r.unexplained)
        n_never += bool(r.never_sent)
        n_rec += bool(r.recovered_execs)
    # Both crash windows were actually exercised.
    assert n_never > 50 and n_rec > 50, (n_never, n_rec)


def test_send_before_log_is_caught(tmp_path: Path) -> None:
    """The harness has teeth: the wrong ordering produces unexplained breaks."""
    bad = [s for s in range(200) if sim.crash_run(tmp_path, s, log_first=False).unexplained]
    assert len(bad) > 10


@pytest.mark.parametrize("seed", range(5))
def test_real_process_kill(tmp_path: Path, seed: int) -> None:
    """TerminateProcess/SIGKILL on a real child, at a random moment."""
    j, v = tmp_path / "j.log", tmp_path / "v.log"
    child = Path(__file__).with_name("_child.py")
    env = dict(os.environ, PYTHONPATH=str(child.parent))
    p = subprocess.Popen(
        [sys.executable, str(child), str(seed), str(j), str(v)], stdout=subprocess.PIPE, env=env
    )
    assert p.stdout is not None and p.stdout.readline().strip() == b"ready"
    import time

    time.sleep(random.Random(seed).uniform(0.05, 0.4))
    p.kill()
    p.wait()
    acc = sim.Acceptor.load(seed, v)
    assert acc.orders, "child was killed before trading; raise the sleep"
    res = sim.recover(j, acc)
    assert res.unexplained == []
    # The venue log is never ahead of the journal by more than the one in-flight order.
    assert len([r for r in read_journal(j)[0] if r["t"] == "new"]) >= len(acc.orders)
    assert json.loads(v.read_text().splitlines()[0])[0] == "C0"
