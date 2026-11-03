"""Carry, value and momentum signals for the FX panel (W23 part 1).

Plain idea: three old ideas about which currencies earn more.

* Carry: currencies with higher interest rates earn more on average,
  because they tend to fall in bad times and investors want paying for that
  risk (Lustig, Roussanov and Verdelhan 2011, RFS 24(11)).
* Value: a currency that has become expensive in real terms tends to drift
  back (Asness, Moskowitz and Pedersen 2013, JF 68(3), section I.B).
* Momentum: currencies that earned more over the past months keep doing so
  for a while (Menkhoff, Sarno, Schmeling and Schrimpf 2012, JFE 106(3);
  Asness, Moskowitz and Pedersen 2013).

Who pays: carry and value are compensation for crash and recession risk
borne by the investor (LRV 2011; Menkhoff et al. 2012 JF on global FX
volatility). Momentum is usually read as slow information diffusion, i.e. it
is paid by slower traders; it is the weakest of the three after 2000.

Formal statement (month-end ``m``; ``s`` is the log of base per foreign;
every rate and price index is the value *published* by ``m``, see
``research/lib/excess_returns.as_known``):

* ``carry_m = i*_m - i_m`` (the CIP forward discount, annualised);
* ``value_m = -(q_m - q_{m-L})``, ``q = s + log P* - log P`` the log real
  exchange rate, ``L = 60`` months (AMP 2013 use the 5-year change);
* ``mom_m = sum_{k=skip}^{L-1} rx_{m-k}``: the excess return over the past
  ``L`` months, skipping the most recent ``skip``. The default is ``L = 12``,
  ``skip = 1`` (AMP 2013); Menkhoff et al. 2012 favour ``L = 1``, ``skip = 0``.
* ``monthly_excess_returns``: ``rx_{m+1} = Delta s_{m+1} + (i*_m tau* - i_m tau)``,
  with calendar-day accrual (``excess_returns.excess_returns``).

Worked example (``tests/test_panel_study.py``): spot 1.00 then 1.01 over a
31-day month, foreign 5% ACT/360, base 1% ACT/360:
``rx = log 1.01 + 0.04 * 31 / 360 = 0.0099503 + 0.0034444 = 0.0133947``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import excess_returns as xr


def month_ends(daily: pd.DataFrame) -> pd.DataFrame:
    """The last observation of each calendar month, stamped at that date."""
    d = daily.sort_index()
    return d.groupby(pd.DatetimeIndex(d.index).to_period("M")).tail(1)


def monthly_excess_returns(spot: pd.DataFrame, rates: pd.DataFrame, base: str) -> pd.DataFrame:
    """Month x currency log excess returns of long foreign against ``base``.

    ``spot`` (base per foreign) and ``rates`` (annualised decimals, already
    lagged to publication) share a month-end index; ``rates`` has a ``base``
    column. Row m is the return from month m-1 to month m.
    """
    if base not in rates.columns:
        raise ValueError(f"rates need the base currency column {base!r}")
    out = {}
    for c in spot.columns:
        if c == base or c not in rates.columns:
            continue
        out[c] = xr.excess_returns(spot[c], rates[c], rates[base], c, base)["excess"]
    return pd.DataFrame(out, index=spot.index)


def carry(rates: pd.DataFrame, base: str) -> pd.DataFrame:
    return rates.drop(columns=[base]).sub(rates[base], axis=0)


def value(spot: pd.DataFrame, cpi: pd.DataFrame, base: str, lookback: int = 60) -> pd.DataFrame:
    """Minus the ``lookback``-month change in the log real exchange rate."""
    cols = [c for c in spot.columns if c != base and c in cpi.columns]
    rel = cpi[cols].div(cpi[base], axis=0)
    q = (spot[cols] * rel).apply(np.log)
    return -(q - q.shift(lookback))


def momentum(rx: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    if not 0 <= skip < lookback:
        raise ValueError("need 0 <= skip < lookback")
    return rx.shift(skip).rolling(lookback - skip, min_periods=lookback - skip).sum()
