"""Concentration diagnostics, fill lag and the pairs P&L numeraire (W6).

Concentration. Plain idea: a Sharpe ratio built on a handful of days is a
bet on those days recurring. The pairs study had skew 44 and kurtosis 2745,
so a few days carried it. ``concentration`` reports:

* ``top_k_share``: sum of the k largest absolute daily P&Ls over the total
  absolute P&L (k = 5, 10, 20);
* ``pnl_by_year`` and ``sharpe_without_year`` (default 2015, the year of the
  SNB floor removal on 2015-01-15);
* ``kurtosis`` (raw) and ``kurtosis_gate``: fails when raw kurtosis exceeds 30.
  The threshold is an ASSUMPTION; a Student-t with 4.2 degrees of freedom
  already has infinite-like sample kurtosis, so 30 flags series dominated by
  a few outliers rather than ordinary fat tails.

Fill lag. ``lag_fill`` shifts positions so a signal computed from the print of
day t trades at the print of day t+1 and earns from t+1 to t+2. Using one
print for both signal and fill assumes a trader can deal at the fixing they
just observed.

Pairs numeraire. ``leg_pnl_eur`` gives the P&L of each leg in EUR, stated.
ECB rates ``S`` are units of currency A per EUR. Holding ``N`` EUR worth of A
bought at ``S_t`` is worth ``N S_t / S_{t+1}`` EUR at t+1, so the leg P&L is
``N (S_t / S_{t+1} - 1)`` EUR. A short leg has negative ``N``.

Worked example (``test_diagnostics.py``): long 100 EUR of USD at 1.10 USD per
EUR, short 100 EUR of GBP at 0.85. Next day 1.09 and 0.86. USD leg
``100 (1.10/1.09 - 1) = +0.917431``; GBP leg ``-100 (0.85/0.86 - 1) = +1.162791``;
total +2.080222 EUR.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as st

KURTOSIS_GATE = 30.0


def concentration(daily: pd.Series, drop_year: int = 2015,
                  ks: tuple[int, ...] = (5, 10, 20)) -> dict:
    """Concentration statistics of a daily P&L or return series."""
    r = daily.dropna()
    a = np.sort(np.abs(r.to_numpy(dtype=float)))[::-1]
    tot = float(a.sum())
    out: dict = {"top_k_share": {str(k): (float(a[:k].sum()) / tot if tot > 0 else None)
                                 for k in ks}}
    years = pd.DatetimeIndex(r.index).year
    out["pnl_by_year"] = {str(y): float(v) for y, v in r.groupby(years).sum().items()}
    rest = r[years != drop_year]
    sd = float(rest.std(ddof=1)) if len(rest) > 1 else 0.0
    out["drop_year"] = drop_year
    out["sharpe_without_year"] = float(rest.mean() / sd * math.sqrt(252)) if sd > 0 else None
    k = float(st.kurtosis(r.to_numpy(dtype=float), fisher=False)) if len(r) > 3 else None
    out["kurtosis"] = k
    out["kurtosis_gate"] = "pass" if k is not None and k <= KURTOSIS_GATE else "fail"
    return out


def lag_fill(positions: pd.DataFrame | pd.Series, lag: int = 1) -> pd.DataFrame | pd.Series:
    """Positions as filled: the target decided at t is held from t+lag."""
    if lag < 0:
        raise ValueError("lag must be >= 0")
    return positions.shift(lag).fillna(0.0)


def leg_pnl_eur(rate_per_eur: pd.Series, notional_eur: pd.Series | float) -> pd.Series:
    """EUR P&L of holding ``notional_eur`` EUR worth of a currency quoted per EUR.

    Row t is the P&L from t-1 to t of the notional held at t-1 (row 0 is 0).
    """
    s = rate_per_eur.to_numpy(dtype=float)
    n = (notional_eur.to_numpy(dtype=float) if isinstance(notional_eur, pd.Series)
         else np.full(len(s), float(notional_eur)))
    out = np.zeros(len(s))
    out[1:] = n[:-1] * (s[:-1] / s[1:] - 1.0)
    return pd.Series(out, index=rate_per_eur.index, name="pnl_eur")
