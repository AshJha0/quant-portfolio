"""Crash-and-restart harness: a seeded trader, a simulated acceptor, recovery.

The acceptor is the venue. It outlives the trader and is the authority on
what was received and filled. ``run_trader`` journals then sends; a crash is
injected after any step. ``recover`` rebuilds the book from the journal,
asks the acceptor for its order and execution status (FIX 35=AF order mass
status plus a ResendRequest for executions), applies what the journal is
missing and reconciles. Zero unexplained breaks is the acceptance test.

Explained outcomes after a crash:

- NEVER_SENT: journaled intent the acceptor does not know (crash between the
  journal write and the send). The order is marked rejected locally.
- RECOVERED_EXEC: an execution the acceptor sent that the journal lacks
  (crash before the report was journaled). Applied on restart.

Anything else, in particular an order the acceptor knows and the journal does
not, breaks write-ahead ordering and is unexplained.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

from qp_oms.recon import Break, Fill, reconcile

from .journal import Journal, SimClock, read_journal


class Crash(Exception):
    pass


@dataclass
class Acceptor:
    """Seeded venue: each new order gets 0-3 partial fills at once."""

    seed: int
    persist: Path | None = None  # venue's own durable log (for the real-process kill test)
    orders: dict[str, float] = field(default_factory=dict)
    execs: dict[str, tuple[str, str, float, float]] = field(
        default_factory=dict
    )  # id -> cid, sym, signed qty, px

    def new_order(
        self, cid: str, symbol: str, signed_qty: float, px: float
    ) -> list[tuple[str, str, str, float, float]]:
        if cid in self.orders:
            return []  # PossDup resend of a known order: no new fills
        self.orders[cid] = signed_qty
        if self.persist is not None:
            with open(self.persist, "a", encoding="utf-8") as f:
                f.write(json.dumps([cid, symbol, signed_qty, px]) + "\n")
                f.flush()
                os.fsync(f.fileno())
        rng = random.Random(f"{self.seed}:{cid}")
        left = abs(signed_qty)
        out = []
        for k in range(rng.randint(0, 3)):
            q = float(rng.randint(1, int(left))) if left >= 1 else 0.0
            if q <= 0:
                break
            left -= q
            eid = f"X{cid}.{k}"
            sq = q if signed_qty > 0 else -q
            self.execs[eid] = (cid, symbol, sq, px)
            out.append((eid, cid, symbol, sq, px))
        return out

    @classmethod
    def load(cls, seed: int, persist: Path) -> Acceptor:
        """Rebuild the venue from its log (fills are a pure function of seed and order)."""
        acc = cls(seed)
        if persist.exists():
            for line in persist.read_text(encoding="utf-8").splitlines():
                if line.endswith("]"):
                    acc.new_order(*json.loads(line))
        acc.persist = persist
        return acc

    def positions(self) -> dict[tuple[str, str], float]:
        pos: dict[tuple[str, str], float] = {}
        for _cid, sym, q, _px in self.execs.values():
            pos[("T", sym)] = pos.get(("T", sym), 0.0) + q
        return pos


def run_trader(
    journal_path: Path,
    acc: Acceptor,
    seed: int,
    n_orders: int,
    crash_after: int | None,
    log_first: bool = True,
    fsync: bool = True,
) -> int:
    """Run the seeded trader; raise Crash after ``crash_after`` steps. Returns steps done.

    ``fsync=False`` is faithful for a process kill (the OS page cache survives
    it); it is not for power loss, which needs the default.

    ``log_first=False`` sends before journaling: the broken ordering, kept so
    the test can show the harness catches it.
    """
    j = Journal(journal_path, fsync=fsync)
    clock = SimClock()
    steps = 0
    start = sum(1 for r in j.records if r["t"] == "new")

    def step() -> None:
        nonlocal steps
        steps += 1
        if crash_after is not None and steps > crash_after:
            j.close()
            raise Crash

    try:
        for i in range(start, n_orders):
            cid = f"C{i}"
            rng = random.Random(f"{seed}:{i}")  # per order, so a restart regenerates the same day
            sym = rng.choice(["X", "Y"])
            q = float(rng.randint(1, 50)) * rng.choice([1, -1])
            px = 100.0 + rng.randint(-5, 5)
            new = {"t": "new", "cid": cid, "sym": sym, "q": q, "px": px}
            if log_first:
                step()
                j.append(new, clock.now())
            step()
            fills = acc.new_order(cid, sym, q, px)
            if not log_first:
                step()
                j.append(new, clock.now())
            for eid, c, s, sq, p in fills:
                step()
                j.append({"t": "exec", "eid": eid, "cid": c, "sym": s, "q": sq, "px": p}, clock.now())
    finally:
        if not j._f.closed:
            j.close()
    return steps


def tear_tail(path: Path, rng: random.Random) -> None:
    """Simulate a write cut short by the kill: a partial record with no newline."""
    junk = json.dumps({"seq": 10**6, "rec": {"t": "exec", "pad": "x" * 40}})
    with open(path, "ab") as f:
        f.write(junk[: rng.randint(1, len(junk) - 1)].encode())


@dataclass
class RecoveryResult:
    never_sent: list[str]
    recovered_execs: list[str]
    unexplained: list[str]
    breaks: list[Break]


def recover(journal_path: Path, acc: Acceptor, fsync: bool = True) -> RecoveryResult:
    j = Journal(journal_path, fsync=fsync)  # repairs a torn tail
    try:
        intents = {r["cid"]: r for r in j.records if r["t"] == "new"}
        fills: dict[str, Fill] = {
            r["eid"]: ("T", r["sym"], r["q"], r["px"]) for r in j.records if r["t"] == "exec"
        }
        never_sent = sorted(c for c in intents if c not in acc.orders)
        for c in never_sent:
            j.append({"t": "local_reject", "cid": c, "why": "NEVER_SENT"})
        unexplained = sorted(f"UNKNOWN_ORDER:{c}" for c in acc.orders if c not in intents)
        recovered = sorted(e for e in acc.execs if e not in fills)
        for e in recovered:
            cid, sym, q, px = acc.execs[e]
            j.append({"t": "exec", "eid": e, "cid": cid, "sym": sym, "q": q, "px": px, "recovered": True})
            fills[e] = ("T", sym, q, px)
        pos: dict[tuple[str, str], float] = {}
        for _a, s, q, _p in fills.values():
            pos[("T", s)] = pos.get(("T", s), 0.0) + q
        acc_fills: dict[str, Fill] = {e: ("T", v[1], v[2], v[3]) for e, v in acc.execs.items()}
        breaks = reconcile(fills, acc_fills, pos, acc.positions())
        unexplained += [f"{b.kind}:{b.key}" for b in breaks if not b.explained]
        return RecoveryResult(never_sent, recovered, unexplained, breaks)
    finally:
        j.close()


def crash_run(tmp: Path, seed: int, n_orders: int = 12, log_first: bool = True) -> RecoveryResult:
    """One seeded kill at a random step, an optional torn tail, restart, recover, finish."""
    rng = random.Random(seed)
    path = tmp / f"j{seed}.log"
    acc = Acceptor(seed)
    total = run_trader(tmp / f"dry{seed}.log", Acceptor(seed), seed, n_orders, None, log_first, fsync=False)
    try:
        run_trader(path, acc, seed, n_orders, rng.randint(0, total - 1), log_first, fsync=False)
    except Crash:
        pass
    if rng.random() < 0.5:
        tear_tail(path, rng)
    res = recover(path, acc, fsync=False)
    if res.unexplained:
        return res
    # Restart finishes the day; the end state must still reconcile.
    run_trader(path, acc, seed, n_orders, None, log_first, fsync=False)
    final = recover(path, acc, fsync=False)
    recs, _ = read_journal(path)
    assert sum(1 for r in recs if r["t"] == "new") == n_orders
    return RecoveryResult(res.never_sent, res.recovered_execs, final.unexplained, final.breaks)
