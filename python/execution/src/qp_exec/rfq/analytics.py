"""RFQ analytics: win rate, markout and last-look reports.

Plain idea
----------
After the fact a dealer asks three questions. How often do we win, by
client tier? When we win, does the price then move against us (markout)?
And if we hold a last-look window, how long do we hold and how often do we
reject, and do we reject more when the market moved our way, which the FX
Global Code (2021) principle 17 says last look must not be used for?

Formal statement
----------------
Win rate per tier ``w = wins / requests`` with SE ``sqrt(w (1 - w) / n)``.
Markout at horizon ``h`` for a won trade with firm side ``s`` (+1 firm
buys), price ``p``, size ``x`` and later mid ``m_{t+h}``:

    markout_h = s (m_{t+h} - p) x,

positive when the trade made money for the firm by ``t + h``; reported as a
size-weighted mean per unit with SE. The last-look asymmetry statistic is
the reject rate when the market moved against the firm during the hold
minus the reject rate when it moved in the firm's favour, with a two-sample
SE; a large positive gap is the pattern principle 17 targets.

Worked example
--------------
>>> from qp_exec.rfq.analytics import RfqRecord, win_rates
>>> recs = [RfqRecord("A", 1, True, 100.0, 1.0, 0), RfqRecord("A", 1, False, 100.0, 1.0, 1)]
>>> win_rates(recs)["A"].rate
0.5
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "LastLookRecord",
    "LastLookReport",
    "MarkoutStat",
    "RateStat",
    "RfqRecord",
    "last_look_report",
    "markouts",
    "win_rates",
]


@dataclass(frozen=True)
class RfqRecord:
    """One request: tier, firm side if won (+1 buy / -1 sell), outcome, price, size, time index."""

    tier: str
    firm_side: int
    won: bool
    price: float
    size: float
    t: int


@dataclass(frozen=True)
class RateStat:
    rate: float
    se: float
    n: int


@dataclass(frozen=True)
class MarkoutStat:
    """Size-weighted mean markout per unit and its SE (from per-trade values)."""

    mean: float
    se: float
    n: int


def _rate(k: int, n: int) -> RateStat:
    r = k / n
    return RateStat(r, math.sqrt(r * (1.0 - r) / n), n)


def win_rates(records: Sequence[RfqRecord]) -> dict[str, RateStat]:
    """Win rate per tier plus ``"ALL"``."""
    by: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by[r.tier].append(r.won)
        by["ALL"].append(r.won)
    return {t: _rate(sum(v), len(v)) for t, v in by.items()}


def markouts(
    records: Sequence[RfqRecord], mids: Sequence[float], horizons: Sequence[int], by_tier: bool = True
) -> dict[tuple[str, int], MarkoutStat]:
    """Markout per unit for won trades at each horizon; trades past the end are dropped."""
    mid = np.asarray(mids, dtype=np.float64)
    out: dict[tuple[str, int], MarkoutStat] = {}
    for h in horizons:
        groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in records:
            if not r.won or r.t + h >= mid.size:
                continue
            per_unit = r.firm_side * (float(mid[r.t + h]) - r.price)
            groups["ALL"].append((per_unit, r.size))
            if by_tier:
                groups[r.tier].append((per_unit, r.size))
        for g, vals in groups.items():
            v = np.array([a for a, _ in vals])
            w = np.array([b for _, b in vals])
            mean = float((v * w).sum() / w.sum())
            n = v.size
            # SE of a ratio estimator, linearised: residuals w_i (v_i - mean).
            resid = w * (v - mean)
            se = float(math.sqrt((resid**2).sum()) / w.sum()) if n > 1 else float("nan")
            out[(g, h)] = MarkoutStat(mean, se, n)
    return out


@dataclass(frozen=True)
class LastLookRecord:
    """A trade request under last look: hold time and price move against the firm during it."""

    hold_ms: float
    accepted: bool
    move_against_firm: float  # > 0: market moved against the firm during the hold


@dataclass(frozen=True)
class LastLookReport:
    n: int
    reject_rate: RateStat
    hold_median_ms: float
    hold_p95_ms: float
    hold_max_ms: float
    reject_rate_adverse: RateStat | None
    reject_rate_favourable: RateStat | None
    asymmetry: float
    asymmetry_se: float


def last_look_report(records: Sequence[LastLookRecord], max_hold_ms: float | None = None) -> LastLookReport:
    """Hold-time distribution, reject rate, and the adverse-minus-favourable reject gap.

    Raises if any hold exceeds ``max_hold_ms`` (the disclosed maximum), since a
    breach is a control failure, not a statistic.
    """
    if not records:
        raise ValueError("no last-look records")
    holds = np.array([r.hold_ms for r in records])
    if max_hold_ms is not None and float(holds.max()) > max_hold_ms:
        raise ValueError(f"hold time {holds.max():.3f} ms exceeds disclosed maximum {max_hold_ms} ms")
    rej = [not r.accepted for r in records]
    adv = [not r.accepted for r in records if r.move_against_firm > 0]
    fav = [not r.accepted for r in records if r.move_against_firm < 0]
    ra = _rate(sum(adv), len(adv)) if adv else None
    rf = _rate(sum(fav), len(fav)) if fav else None
    if ra is not None and rf is not None:
        gap = ra.rate - rf.rate
        gap_se = math.sqrt(ra.se**2 + rf.se**2)
    else:
        gap, gap_se = float("nan"), float("nan")
    return LastLookReport(
        n=len(records),
        reject_rate=_rate(sum(rej), len(rej)),
        hold_median_ms=float(np.median(holds)),
        hold_p95_ms=float(np.quantile(holds, 0.95)),
        hold_max_ms=float(holds.max()),
        reject_rate_adverse=ra,
        reject_rate_favourable=rf,
        asymmetry=gap,
        asymmetry_se=gap_se,
    )


def tier_summary(stats: Mapping[str, RateStat]) -> str:
    """Plain-text table, one line per tier."""
    return "\n".join(f"{t:>4}  {s.rate:6.3f} +/- {s.se:5.3f}  n={s.n}" for t, s in sorted(stats.items()))
