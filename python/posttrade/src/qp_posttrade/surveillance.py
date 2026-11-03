"""Market-abuse detectors over the attributable audit log.

Each detector reads ``OrderEvent`` lines (who, what, when) and returns
``Alert`` records carrying the evidence order ids, so a reviewer can replay
the case. Thresholds are parameters with stated defaults; the false-positive
rates at those defaults are measured on seeded benign flow in the tests and
quoted in the package README.

Behaviours, as described in EU MAR Art. 12(2) and Annex I, CFTC
anti-spoofing (CEA s.4c(a)(5)(C), added by Dodd-Frank s.747) and FINRA
Rule 5210/5310 guidance:

- **Spoofing / layering.** Large one-sided orders (layering: at several
  price levels) placed with no intent to trade, a fill on the *other* side,
  then the one-sided orders are cancelled. Formal test, per trader and
  symbol: an opposite-side fill at time t; within ``window`` seconds before
  t, the trader added one-sided quantity at least ``size_ratio`` times the
  fill, at least ``min_levels`` distinct prices (1 = spoofing, 3 =
  layering), the trader's own same-window quantity on the fill side is at
  most 1/``size_ratio`` of it, and at least ``cancel_frac`` of it is
  cancelled within ``cancel_within`` seconds after t without filling.
  ``size_ratio`` defaults to 5: at 3 the benign market makers (quote sizes
  100-300, so a 3:1 one-sided imbalance happens by chance) produced 144.5
  false alerts per session over 20 seeds; at 3.5 and 5 they produced none.
- **Wash trades.** A buy fill and a sell fill in the same symbol by the
  same beneficial owner, same quantity, within ``max_dt`` seconds and
  ``px_tol`` relative price: no change in ownership.
- **Marking the close.** In the last ``close_window`` seconds of the session
  a trader is at least ``min_share`` of all fills by volume, trades one
  side only, and those fills are at least ``min_own_frac`` of the trader's
  whole-session volume in that symbol.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from qp_posttrade import STAMP
from qp_posttrade.model import OrderEvent

AlertKind = Literal["SPOOFING", "LAYERING", "WASH", "MARKING_CLOSE"]


@dataclass(frozen=True)
class Alert:
    kind: AlertKind
    trader: str
    symbol: str
    ts: float
    evidence: tuple[str, ...]
    stamp: str = STAMP


def _by_trader_symbol(events: Iterable[OrderEvent]) -> dict[tuple[str, str], list[OrderEvent]]:
    g: dict[tuple[str, str], list[OrderEvent]] = defaultdict(list)
    for e in events:
        g[(e.trader, e.symbol)].append(e)
    for v in g.values():
        v.sort(key=lambda e: e.ts)
    return g


def detect_spoofing(
    events: Iterable[OrderEvent],
    *,
    window: float = 10.0,
    cancel_within: float = 5.0,
    size_ratio: float = 5.0,
    cancel_frac: float = 0.8,
    layering_levels: int = 3,
) -> list[Alert]:
    alerts: list[Alert] = []
    for (trader, sym), evs in _by_trader_symbol(events).items():
        news = [e for e in evs if e.kind == "NEW"]  # already time-sorted
        new_ts = [n.ts for n in news]
        cancel_t = {e.order_id: e.ts for e in evs if e.kind == "CANCEL"}
        filled = {e.order_id for e in evs if e.kind == "FILL"}
        flagged: set[str] = set()
        for f in (e for e in evs if e.kind == "FILL"):
            other = "SELL" if f.side == "BUY" else "BUY"
            pre = news[bisect_left(new_ts, f.ts - window) : bisect_right(new_ts, f.ts)]
            spoof = [n for n in pre if n.side == other and n.order_id not in filled]
            own = sum(n.qty for n in pre if n.side == f.side)
            sq = sum(n.qty for n in spoof)
            if not spoof or sq < size_ratio * f.qty or own * size_ratio > sq:
                continue
            cq = sum(n.qty for n in spoof if f.ts <= cancel_t.get(n.order_id, -1e18) <= f.ts + cancel_within)
            if cq < cancel_frac * sq:
                continue
            ids = tuple(sorted(n.order_id for n in spoof))
            if set(ids) <= flagged:
                continue
            flagged.update(ids)
            levels = len({n.px for n in spoof})
            kind: AlertKind = "LAYERING" if levels >= layering_levels else "SPOOFING"
            alerts.append(Alert(kind, trader, sym, f.ts, (f.order_id, *ids)))
    return alerts


def detect_wash(events: Iterable[OrderEvent], *, max_dt: float = 1.0, px_tol: float = 1e-4) -> list[Alert]:
    fills: dict[tuple[str, str], list[OrderEvent]] = defaultdict(list)
    for e in events:
        if e.kind == "FILL":
            fills[(e.owner, e.symbol)].append(e)
    alerts = []
    for (_owner, sym), fs in fills.items():
        fs.sort(key=lambda e: e.ts)
        used: set[int] = set()
        for i, a in enumerate(fs):
            if i in used:
                continue
            for j in range(i + 1, len(fs)):
                b = fs[j]
                if b.ts - a.ts > max_dt:
                    break
                if j in used or b.side == a.side or abs(b.qty - a.qty) > 1e-9:
                    continue
                if abs(b.px - a.px) > px_tol * a.px:
                    continue
                used.update((i, j))
                traders = "+".join(sorted({a.trader, b.trader}))
                alerts.append(Alert("WASH", traders, sym, b.ts, (a.order_id, b.order_id)))
                break
    return alerts


def detect_marking_close(
    events: Iterable[OrderEvent],
    close_ts: float,
    *,
    close_window: float = 300.0,
    min_share: float = 0.25,
    min_own_frac: float = 0.5,
) -> list[Alert]:
    fills = [e for e in events if e.kind == "FILL"]
    late = [e for e in fills if close_ts - close_window <= e.ts <= close_ts]
    mkt: dict[str, float] = defaultdict(float)
    for e in late:
        mkt[e.symbol] += e.qty
    alerts = []
    for (trader, sym), evs in _by_trader_symbol(fills).items():
        mine = [e for e in evs if close_ts - close_window <= e.ts <= close_ts]
        if not mine or len({e.side for e in mine}) != 1:
            continue
        q = sum(e.qty for e in mine)
        if q < min_share * mkt[sym] or q < min_own_frac * sum(e.qty for e in evs):
            continue
        alerts.append(Alert("MARKING_CLOSE", trader, sym, mine[-1].ts, tuple(e.order_id for e in mine)))
    return alerts
