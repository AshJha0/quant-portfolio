"""Flow toxicity: markout- and VPIN-based spread widening, and tier reports.

Plain idea
----------
A market maker earns the half-spread from uninformed flow and loses it, and
more, to flow that knows where the price is going. Two standard ways to see
the second kind coming:

* **Markouts.** After each fill, wait ``h`` steps and measure how far the mid
  moved against the maker: ``sign * (mid[t+h] - ref)`` per unit, in ticks
  (positive = the maker gained). With ``ref`` = the fill price this is the
  realised edge a desk reports per counterparty; with ``ref`` = the mid at
  the fill it is the adverse-selection part alone, which is what the widening
  should cover (the simulator feeds the estimator that one). A run of negative markouts means the flow
  hitting the quotes is informed. ``MarkoutToxicity`` keeps an exponentially
  weighted mean of matured markouts and widens both quotes by enough ticks to
  cover the expected adverse move.
* **VPIN.** Easley, Lopez de Prado and O'Hara (2012) group trades into equal
  volume buckets and measure the order imbalance ``|V_buy - V_sell| / V`` per
  bucket; the mean over the last ``n`` buckets estimates the probability of
  informed trading. ``VPIN`` does this with the aggressor side known (the
  simulator and the crypto feeds both publish it, so no bulk classification
  is needed). It widens once the estimate is above a threshold.

``tier_report`` splits fills by counterparty tier and reports, per tier, the
fill rate (maker fills / that tier's market orders), the mean markout with a
95% normal-approximation interval, and the total markout P&L.

Formal statement
----------------
Markout of fill ``i`` (side ``s_i`` = +1 maker bought, -1 maker sold, price
``p_i`` in ticks, filled at step ``t_i``)::

    M_i(h) = s_i * (mid[t_i + h] - p_i)

EWMA with weight ``lam`` on each new matured value:
``m <- (1 - lam) * m + lam * M_i``; on a step with nothing maturing,
``m <- (1 - decay) * m``. Widening, in whole ticks::

    w = min(max_widen, ceil(beta * max(0, -m) - 1e-9))

VPIN over the last ``n`` complete buckets of size ``V``::

    VPIN = (1 / (n V)) * sum_b |V_b^buy - V_b^sell|

and the widening is ``min(max_widen, floor(scale * max(0, VPIN - threshold)))``.
A trade that crosses a bucket boundary is split between the two buckets
(Easley et al. 2012, section 3).

Worked example
--------------
Three fills, horizon 1: maker buys at 99 with next mid 98 (markout -1), sells
at 101 with next mid 102 (-1), buys at 99 with next mid 100 (+1).

>>> t = MarkoutToxicity(horizon=1, lam=0.5, beta=1.0, max_widen=3)
>>> t.on_fill(step=0, side=1, price_ticks=99)
>>> t.on_mid(1, 98.0)
>>> t.mean, t.widen_ticks()
(-1.0, 1)

VPIN with 10-lot buckets: 8 bought / 2 sold, then 5 / 5.

>>> v = VPIN(bucket_volume=10, n_buckets=2)
>>> v.on_trade(1, 8); v.on_trade(-1, 2); v.on_trade(1, 5); v.on_trade(-1, 5)
>>> v.value()
0.3
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from qp_core.model import model


@model(id="EXEC-MM-TOXICITY")
class MarkoutToxicity:
    """EWMA of matured fill markouts, in ticks; widens when it is negative."""

    def __init__(self, horizon: int, lam: float = 0.1, beta: float = 1.0,
                 max_widen: int = 5, decay: float = 0.0) -> None:
        if (horizon < 1 or not 0 < lam <= 1 or beta < 0 or max_widen < 0
                or not 0 <= decay < 1):
            raise ValueError("horizon >= 1, 0 < lam <= 1, beta >= 0, max_widen >= 0, "
                             "0 <= decay < 1 required")
        self.h, self.lam, self.beta, self.max_widen = horizon, lam, beta, max_widen
        self.decay = decay
        self._pending: deque[tuple[int, int, float, int]] = deque()  # (mature_step, side, px, qty)
        self.mean = 0.0
        self.matured = 0

    def on_fill(self, step: int, side: int, price_ticks: float, qty: int = 1) -> None:
        if side not in (1, -1):
            raise ValueError("side must be +1 or -1")
        self._pending.append((step + self.h, side, float(price_ticks), qty))

    def on_mid(self, step: int, mid_ticks: float) -> None:
        """Mature every pending fill whose horizon has been reached at ``step``.

        With ``decay > 0`` a step on which nothing matures shrinks the estimate
        toward zero by that fraction, so a maker that widened itself out of all
        fills is not locked out for ever (there would be no new markouts).
        """
        if self.decay and not (self._pending and self._pending[0][0] <= step):
            self.mean *= 1 - self.decay
        while self._pending and self._pending[0][0] <= step:
            _, s, px, qty = self._pending.popleft()
            m = s * (mid_ticks - px)
            for _ in range(qty):  # each unit is one observation
                self.mean = m if self.matured == 0 else (1 - self.lam) * self.mean + self.lam * m
                self.matured += 1

    def widen_ticks(self) -> int:
        if self.matured == 0:
            return 0
        return min(self.max_widen, max(0, math.ceil(self.beta * max(0.0, -self.mean) - 1e-9)))


@model(id="EXEC-MM-TOXICITY")
class VPIN:
    """Volume-synchronised probability of informed trading, aggressor side known."""

    def __init__(self, bucket_volume: float, n_buckets: int = 50, threshold: float = 0.0,
                 scale: float = 0.0, max_widen: int = 5) -> None:
        if bucket_volume <= 0 or n_buckets < 1:
            raise ValueError("bucket_volume > 0 and n_buckets >= 1 required")
        self.V, self.n = float(bucket_volume), n_buckets
        self.threshold, self.scale, self.max_widen = threshold, scale, max_widen
        self._buckets: deque[float] = deque(maxlen=n_buckets)  # |imbalance| / V per bucket
        self._buy = 0.0
        self._sell = 0.0

    def on_trade(self, aggressor_side: int, qty: float) -> None:
        if aggressor_side not in (1, -1) or qty < 0:
            raise ValueError("aggressor_side must be +1/-1 and qty >= 0")
        left = float(qty)
        while left > 0:
            room = self.V - self._buy - self._sell
            take = min(room, left)
            if aggressor_side == 1:
                self._buy += take
            else:
                self._sell += take
            left -= take
            if self._buy + self._sell >= self.V - 1e-12:
                self._buckets.append(abs(self._buy - self._sell) / self.V)
                self._buy = self._sell = 0.0

    @property
    def complete_buckets(self) -> int:
        return len(self._buckets)

    def value(self) -> float | None:
        """Mean imbalance over the last ``n`` complete buckets; None before the first."""
        if not self._buckets:
            return None
        return round(sum(self._buckets) / len(self._buckets), 12)

    def widen_ticks(self) -> int:
        v = self.value()
        if v is None:
            return 0
        return min(self.max_widen, max(0, math.floor(self.scale * max(0.0, v - self.threshold))))


@dataclass(frozen=True, slots=True)
class TierStats:
    tier: str
    market_orders: int
    fills: int
    fill_rate: float
    mean_markout: float  # ticks per unit, maker's view
    ci95: tuple[float, float]
    markout_pnl: float  # ticks, summed over units


def tier_report(fills: list[tuple[str, float]], orders_by_tier: dict[str, int]) -> list[TierStats]:
    """Per-tier fill rate and markout.

    ``fills`` holds one ``(tier, markout_ticks)`` per filled unit whose horizon
    matured; ``orders_by_tier`` counts each tier's market orders (unit size).
    """
    out: list[TierStats] = []
    for tier in sorted(orders_by_tier):
        ms = [m for t, m in fills if t == tier]
        n = len(ms)
        mean = sum(ms) / n if n else math.nan
        if n > 1:
            sd = math.sqrt(sum((m - mean) ** 2 for m in ms) / (n - 1))
            half = 1.959963984540054 * sd / math.sqrt(n)
            ci = (mean - half, mean + half)
        else:
            ci = (math.nan, math.nan)
        orders = orders_by_tier[tier]
        out.append(TierStats(tier, orders, n, n / orders if orders else math.nan, mean, ci,
                             float(sum(ms))))
    return out
