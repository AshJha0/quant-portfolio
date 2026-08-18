"""Pair candidate generation and pre-screens (correlation on returns, SSD).

Screening order on a desk: cheap filters first (sector tag, return
correlation or SSD distance), the expensive statistical test (Engle-Granger)
last. This module implements the cheap filters; :mod:`eq_pairs.cointegration`
implements the test.

Why correlation is computed on RETURNS and not on PRICES
--------------------------------------------------------
Price levels of any two upward-drifting non-stationary series are almost
always highly correlated: correlation of two independent random walks does
not converge to zero as the sample grows (spurious correlation, Granger &
Newbold 1974). Two independent random walks routinely show |price
correlation| > 0.8 while their *returns* — which are stationary — have
correlation ~ 0. A price-level correlation screen therefore selects pairs
that merely share a drift, not pairs that co-move. We screen on daily log
returns; the (still imperfect, see METHODOLOGY.md) premise is that names
whose shocks are correlated are more likely to share a common trend.

The SSD screen (Gatev, Goetzmann & Rouwenkamp 2006 "distance method") is the
non-parametric alternative: normalise each price path to start at 1, and rank
pairs by the sum of squared differences of the normalised paths. Low SSD =
the two paths tracked each other over the formation window.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "candidate_pairs",
    "log_returns",
    "pair_correlations",
    "correlation_screen",
    "ssd_distances",
    "ssd_screen",
]

Pair = tuple[str, str]


def candidate_pairs(
    tickers: Sequence[str],
    sectors: Optional[Mapping[str, str]] = None,
    same_sector_only: bool = True,
) -> list[Pair]:
    """Enumerate candidate pairs, optionally restricted to same-sector names.

    Parameters
    ----------
    tickers : sequence of str
        Universe of tickers (must be unique).
    sectors : mapping ticker -> sector tag, optional
        Required when ``same_sector_only`` is True.
    same_sector_only : bool
        If True (default), only emit pairs whose two legs share a sector tag —
        the standard economic prior that cointegration needs a common driver.

    Returns
    -------
    list of (ticker_a, ticker_b) with a < b lexicographically.
    """
    tickers = list(tickers)
    if len(set(tickers)) != len(tickers):
        raise ValueError("tickers must be unique")
    if same_sector_only:
        if sectors is None:
            raise ValueError("sectors mapping required when same_sector_only=True")
        missing = [t for t in tickers if t not in sectors]
        if missing:
            raise ValueError(f"tickers missing sector tags: {missing}")
    pairs = []
    for a, b in combinations(sorted(tickers), 2):
        if same_sector_only and sectors[a] != sectors[b]:
            continue
        pairs.append((a, b))
    return pairs


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns ln(P_t / P_{t-1}); first row dropped.

    Raises ``ValueError`` on non-positive prices.
    """
    arr = prices.to_numpy(dtype=float)
    if np.any(~np.isfinite(arr) & ~np.isnan(arr)) or np.nanmin(arr) <= 0.0:
        raise ValueError("prices must be strictly positive and finite")
    return np.log(prices).diff().iloc[1:]


def pair_correlations(
    prices: pd.DataFrame,
    pairs: Sequence[Pair],
    on: str = "returns",
) -> pd.DataFrame:
    """Pearson correlation for each candidate pair.

    Parameters
    ----------
    prices : DataFrame
        Price panel (columns = tickers).
    pairs : sequence of (a, b)
        Candidate pairs.
    on : {"returns", "prices"}
        "returns" (default) correlates daily log returns — the statistically
        meaningful choice. "prices" correlates raw levels and exists only to
        demonstrate the spurious-correlation trap (see module docstring).

    Returns
    -------
    DataFrame indexed by pair with column ``corr``, sorted descending.
    Pairs with an undefined correlation (zero-variance leg) get NaN.
    """
    if on == "returns":
        data = log_returns(prices)
    elif on == "prices":
        data = prices
    else:
        raise ValueError(f"on must be 'returns' or 'prices', got {on!r}")
    rows = []
    for a, b in pairs:
        x = data[a].to_numpy(dtype=float)
        y = data[b].to_numpy(dtype=float)
        sx, sy = np.std(x), np.std(y)
        if sx == 0.0 or sy == 0.0 or len(x) < 3:
            c = np.nan
        else:
            c = float(np.corrcoef(x, y)[0, 1])
        rows.append({"pair": (a, b), "corr": c})
    out = pd.DataFrame(rows).set_index("pair")
    return out.sort_values("corr", ascending=False)


def correlation_screen(
    prices: pd.DataFrame,
    pairs: Sequence[Pair],
    min_corr: float = 0.60,
    on: str = "returns",
) -> pd.DataFrame:
    """Keep pairs whose correlation is >= ``min_corr``.

    NaN correlations (e.g. a zero-variance leg) are always rejected.

    Returns
    -------
    DataFrame (subset of :func:`pair_correlations` output) of survivors,
    sorted by correlation descending.
    """
    if not -1.0 <= min_corr <= 1.0:
        raise ValueError(f"min_corr must be in [-1, 1], got {min_corr}")
    corr = pair_correlations(prices, pairs, on=on)
    return corr[corr["corr"] >= min_corr]


def ssd_distances(prices: pd.DataFrame, pairs: Sequence[Pair]) -> pd.DataFrame:
    """Sum of squared differences between normalised price paths.

    Each leg is normalised to start at 1 (P_t / P_0); the distance is
    sum_t (p_a,t - p_b,t)^2. Lower = closer tracking.

    Returns
    -------
    DataFrame indexed by pair with column ``ssd``, sorted ascending.
    """
    first = prices.iloc[0]
    if (first <= 0).any():
        raise ValueError("first price must be positive for normalisation")
    norm = prices / first
    rows = []
    for a, b in pairs:
        d = norm[a].to_numpy(dtype=float) - norm[b].to_numpy(dtype=float)
        rows.append({"pair": (a, b), "ssd": float(np.sum(d * d))})
    out = pd.DataFrame(rows).set_index("pair")
    return out.sort_values("ssd", ascending=True)


def ssd_screen(
    prices: pd.DataFrame,
    pairs: Sequence[Pair],
    top_n: int = 20,
) -> pd.DataFrame:
    """Keep the ``top_n`` pairs with the smallest SSD distance."""
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    return ssd_distances(prices, pairs).head(top_n)
