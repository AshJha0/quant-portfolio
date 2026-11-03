"""Routing transaction-cost analysis and markout-learned venue scores.

Definitions (``d`` = +1 for a buy, -1 for a sell; prices and mids in ticks)
----------------------------------------------------------------------------
* Fill rate of a venue: lots filled there / lots sent there.
* Effective spread of a fill: ``2 d (p - m)`` with ``m`` the consolidated mid
  at the fill (SEC Rule 605 uses the same definition). Negative for passive
  fills (we earned spread), zero for midpoint fills.
* Markout at horizon ``tau``: ``d (m_{t+tau} - p)``. Positive means the price
  went our way after we traded; persistently negative markouts mean the
  venue's counterparties were better informed than we were (adverse
  selection). Reversion is the markout of an aggressive fill measured from the
  mid: ``d (m_{t+tau} - m_t)``.
* All-in cost per share: ``d (p - m_arrival) + fee``.

Venue score (W27b)
------------------
``VenueScores`` keeps an exponentially weighted mean of each venue's
markout at ``tau`` (weight per lot). The router charges each venue an
adverse-selection cost ``a_k = max(0, -score_k)`` in its allocation, so
venues whose fills keep going against us get less flow, which is the
feedback loop the plan asks for.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .market import ExecFill, Market, sgn


@dataclass(frozen=True)
class VenueTCA:
    venue: str
    sent: int
    filled: int
    fill_rate: float
    eff_spread: float  # lot-weighted mean, ticks
    markout: dict[int, tuple[float, float]]  # tau_us -> (mean, standard error), ticks
    fee: float  # lot-weighted mean fee, ticks per share


def markout(f: ExecFill, mkt: Market, tau_us: int) -> float:
    return sgn(f.side) * (mkt.mid_at(f.ts + tau_us) - f.price)


def venue_tca(fills: Iterable[ExecFill], mkt: Market, sent: dict[str, int],
              taus_us: tuple[int, ...] = (100_000, 1_000_000)) -> dict[str, VenueTCA]:
    by: dict[str, list[ExecFill]] = defaultdict(list)
    for f in fills:
        by[f.venue].append(f)
    out = {}
    for v in sorted(set(sent) | set(by)):
        fs = by.get(v, [])
        q = sum(f.qty for f in fs)
        es = sum(2 * sgn(f.side) * (f.price - f.mid) * f.qty for f in fs) / q if q else math.nan
        fee = sum(f.fee_ticks * f.qty for f in fs) / q if q else math.nan
        mk = {}
        for tau in taus_us:
            xs = [markout(f, mkt, tau) for f in fs for _ in range(f.qty)]
            if len(xs) >= 2:
                mu = sum(xs) / len(xs)
                var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
                mk[tau] = (mu, math.sqrt(var / len(xs)))
            else:
                mk[tau] = (xs[0] if xs else math.nan, math.nan)
        s = sent.get(v, 0)
        out[v] = VenueTCA(v, s, q, q / s if s else math.nan, es, mk, fee)
    return out


class VenueScores:
    """Online, lot-weighted EWMA of markouts at ``tau_us`` per venue."""

    def __init__(self, tau_us: int = 1_000_000, halflife_lots: float = 200.0,
                 prior: float = 0.0) -> None:
        self.tau = tau_us
        self.alpha = 1 - 0.5 ** (1 / halflife_lots)
        self.prior = prior
        self.score: dict[str, float] = {}
        self.lots: dict[str, int] = defaultdict(int)
        self._pending: list[ExecFill] = []

    def observe(self, f: ExecFill) -> None:
        self._pending.append(f)

    def update(self, mkt: Market) -> None:
        """Score every pending fill whose horizon has passed."""
        keep = []
        for f in self._pending:
            if f.ts + self.tau > mkt.t:
                keep.append(f)
                continue
            x = markout(f, mkt, self.tau)
            s = self.score.get(f.venue, self.prior)
            w = 1 - (1 - self.alpha) ** f.qty
            self.score[f.venue] = (1 - w) * s + w * x
            self.lots[f.venue] += f.qty
        self._pending = keep

    def toxicity(self, venue: str) -> float:
        """Adverse-selection charge in ticks per share (>= 0)."""
        return max(0.0, -self.score.get(venue, self.prior))
