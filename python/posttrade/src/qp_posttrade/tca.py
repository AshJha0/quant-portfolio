"""Transaction cost analysis from fills.

Plain idea: what did trading cost us compared with a fair reference price?

Formal definitions, per fill with sign s (+1 buy, -1 sell), quantity q,
price p, arrival mid a and mid at fill m (Perold 1988 for implementation
shortfall; SEC Rule 605 (2024) for effective spread):

- shortfall cost (currency) = s * (p - a) * q
- spread cost (currency)    = s * (p - m) * q   (half the effective spread
  times quantity; effective spread = 2 * s * (p - m))
- shortfall in bps          = 1e4 * shortfall cost / (a * q)

Worked example: buy 100 at 10.02 with arrival 10.00 and mid 10.01 costs
2.00 shortfall and 1.00 spread, i.e. 20 bps of shortfall.

>>> from datetime import datetime
>>> from qp_posttrade.model import Fill
>>> f = Fill("e1", "A", "XYZ", "BUY", 100, 10.02, datetime(2026, 11, 2, 15), "V1",
...          arrival_px=10.0, mid_at_fill=10.01)
>>> r = fill_costs(f)
>>> round(r.shortfall, 6), round(r.spread, 6), round(r.shortfall_bps, 6)
(2.0, 1.0, 20.0)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from qp_posttrade import STAMP
from qp_posttrade.model import Fill


@dataclass(frozen=True)
class FillCost:
    exec_id: str
    venue: str
    qty: float
    notional: float
    shortfall: float
    spread: float
    fee: float
    shortfall_bps: float


def fill_costs(f: Fill) -> FillCost:
    if f.arrival_px <= 0 or f.mid_at_fill <= 0:
        raise ValueError(f"{f.exec_id}: arrival_px and mid_at_fill are required for TCA")
    sf = f.sign * (f.px - f.arrival_px) * f.qty
    sp = f.sign * (f.px - f.mid_at_fill) * f.qty
    return FillCost(f.exec_id, f.venue, f.qty, f.px * f.qty, sf, sp, f.fee, 1e4 * sf / (f.arrival_px * f.qty))


@dataclass(frozen=True)
class TcaTotals:
    stamp: str
    fills: int
    qty: float
    notional: float
    shortfall: float
    spread: float
    fees: float

    @property
    def total_cost(self) -> float:
        return self.shortfall + self.fees


def _totals(cs: list[FillCost]) -> TcaTotals:
    return TcaTotals(
        STAMP,
        len(cs),
        sum(c.qty for c in cs),
        sum(c.notional for c in cs),
        sum(c.shortfall for c in cs),
        sum(c.spread for c in cs),
        sum(c.fee for c in cs),
    )


def tca(fills: Iterable[Fill]) -> tuple[TcaTotals, dict[str, TcaTotals]]:
    """Return the overall totals and the totals by venue."""
    cs = [fill_costs(f) for f in fills]
    by: dict[str, list[FillCost]] = defaultdict(list)
    for c in cs:
        by[c.venue].append(c)
    return _totals(cs), {v: _totals(x) for v, x in sorted(by.items())}
