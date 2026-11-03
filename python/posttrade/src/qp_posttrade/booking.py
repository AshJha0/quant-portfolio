"""Booking, lifecycle events and confirmation matching.

Plain idea: a fill is an execution; a *trade* is the booked record a back
office settles. Booking groups fills by (account, symbol, side, trade date),
prices the trade at the quantity-weighted average, and fixes the value date.
After booking, a trade can only change through explicit lifecycle events,
each of which leaves the old version in the history (append-only), as in the
FINOS CDM ``BusinessEvent`` model: amend, cancel, novation (a new
counterparty steps in; economics unchanged).

Confirmation matching compares our booked trades with the counterparty's
file field by field, with a price tolerance, and returns matched, unmatched
and mismatched trades. The field set mirrors the core of a SWIFT MT300
(trade date, value date, amounts, rate) and a DTCC CTM equity confirm.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Literal

from qp_posttrade.model import Fill, Side
from qp_posttrade.settlement import fx_spot_date, us_equity_settle

AssetClass = Literal["US_EQUITY", "FX_SPOT"]
Status = Literal["LIVE", "CANCELLED"]


@dataclass(frozen=True)
class Trade:
    trade_id: str
    version: int
    account: str
    counterparty: str
    symbol: str
    side: Side
    qty: float
    avg_px: float
    trade_date: date
    value_date: date
    exec_ids: tuple[str, ...]
    status: Status = "LIVE"


def value_date(asset: AssetClass, symbol: str, trade_date: date) -> date:
    if asset == "US_EQUITY":
        return us_equity_settle(trade_date)
    return fx_spot_date(symbol, trade_date)


@dataclass
class Blotter:
    """Append-only trade store. ``history[trade_id]`` holds every version."""

    asset: AssetClass
    history: dict[str, list[Trade]] = field(default_factory=dict)
    _seen_exec: set[str] = field(default_factory=set)

    def book(self, fills: Iterable[Fill], counterparty: str) -> list[Trade]:
        groups: dict[tuple[str, str, str, date], list[Fill]] = defaultdict(list)
        for f in fills:
            if f.exec_id in self._seen_exec:
                continue  # drop-copy resend; booked once
            self._seen_exec.add(f.exec_id)
            groups[(f.account, f.symbol, f.side, f.ts.date())].append(f)
        out = []
        for (acct, sym, _side, td), fs in sorted(groups.items()):
            qty = sum(f.qty for f in fs)
            px = sum(f.qty * f.px for f in fs) / qty
            tid = f"T{len(self.history) + 1:06d}"
            ids = tuple(sorted(f.exec_id for f in fs))
            t = Trade(
                tid, 1, acct, counterparty, sym, fs[0].side, qty, px, td, value_date(self.asset, sym, td), ids
            )
            self.history[tid] = [t]
            out.append(t)
        return out

    def current(self, trade_id: str) -> Trade:
        return self.history[trade_id][-1]

    def _append(self, trade_id: str, **changes: object) -> Trade:
        cur = self.current(trade_id)
        if cur.status == "CANCELLED":
            raise ValueError(f"{trade_id} is cancelled; no further events")
        new = replace(cur, version=cur.version + 1, **changes)  # type: ignore[arg-type]
        self.history[trade_id].append(new)
        return new

    def amend(self, trade_id: str, *, qty: float | None = None, avg_px: float | None = None) -> Trade:
        ch: dict[str, object] = {}
        if qty is not None:
            ch["qty"] = qty
        if avg_px is not None:
            ch["avg_px"] = avg_px
        if not ch:
            raise ValueError("amend needs at least one field")
        return self._append(trade_id, **ch)

    def cancel(self, trade_id: str) -> Trade:
        return self._append(trade_id, status="CANCELLED")

    def novate(self, trade_id: str, new_counterparty: str) -> Trade:
        if new_counterparty == self.current(trade_id).counterparty:
            raise ValueError("novation must change the counterparty")
        return self._append(trade_id, counterparty=new_counterparty)

    def live(self) -> list[Trade]:
        return [h[-1] for h in self.history.values() if h[-1].status == "LIVE"]


@dataclass(frozen=True)
class Confirm:
    """One line of the counterparty's confirmation file."""

    ref: str
    account: str
    symbol: str
    side: Side
    qty: float
    px: float
    trade_date: date
    value_date: date


@dataclass(frozen=True)
class MatchResult:
    matched: list[tuple[str, str]]
    mismatched: list[tuple[str, str, tuple[str, ...]]]  # trade_id, ref, fields
    unmatched_ours: list[str]
    unmatched_theirs: list[str]


def match_confirms(trades: Iterable[Trade], confirms: Iterable[Confirm], px_tol: float = 1e-6) -> MatchResult:
    """Pair on the economic key (account, symbol, side, trade date), then compare."""
    ours: dict[tuple[str, str, str, date], list[Trade]] = defaultdict(list)
    for t in trades:
        if t.status == "LIVE":
            ours[(t.account, t.symbol, t.side, t.trade_date)].append(t)
    matched: list[tuple[str, str]] = []
    mism: list[tuple[str, str, tuple[str, ...]]] = []
    theirs_left: list[str] = []
    for c in confirms:
        cands = ours.get((c.account, c.symbol, c.side, c.trade_date), [])
        if not cands:
            theirs_left.append(c.ref)
            continue
        # prefer an exact quantity match, else the first candidate
        t = next((x for x in cands if abs(x.qty - c.qty) < 1e-9), cands[0])
        cands.remove(t)
        bad = []
        if abs(t.qty - c.qty) > 1e-9:
            bad.append("qty")
        if abs(t.avg_px - c.px) > px_tol * max(1.0, abs(t.avg_px)):
            bad.append("px")
        if t.value_date != c.value_date:
            bad.append("value_date")
        if bad:
            mism.append((t.trade_id, c.ref, tuple(bad)))
        else:
            matched.append((t.trade_id, c.ref))
    left = sorted(t.trade_id for v in ours.values() for t in v)
    return MatchResult(matched, mism, left, theirs_left)
