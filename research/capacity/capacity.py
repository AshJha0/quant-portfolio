"""Turnover, holding period and a capacity curve (W23 part 2).

Plain idea: a strategy that earns 0.6 Sharpe on paper earns less as it gets
bigger, because each trade moves the price against it more. The capacity
curve shows net Sharpe against assets under management (AUM); the headline
capacity is the AUM at which net Sharpe falls to half of its zero-size value.

Formal statement. Weights ``w_t`` (fractions of AUM) are rebalanced each
period. Trade in currency ``i`` at ``t`` is ``q_it = A |w_it - w_i,t-1|``
dollars. Its cost, as a fraction of AUM, is

    c_t(A) = sum_i |dw_it| (h_i + Y sigma_i sqrt(A |dw_it| / ADV_i))

with ``h_i`` the half-spread, ``sigma_i`` the daily volatility, ``ADV_i``
the average daily dollar volume and ``Y`` the impact coefficient: the
square-root law (Torre 1997, BARRA Market Impact Model Handbook; Almgren,
Thum, Hauptmann and Li 2005; Toth et al. 2011). Net return
``r_t(A) = g_t - c_t(A)``; net Sharpe ``SR(A)``; capacity ``A*`` solves
``SR(A*) = SR(0) / 2`` (bisection on log A; SR(A) is decreasing).

Evidence class: ASSUMPTION for ``Y``. Until W25 calibrates impact on our own
book data, ``Y = 1.0`` is the literature order of magnitude (Toth et al.
2011 report Y of order 1 across markets). ADV comes from the BIS Triennial
Survey 2022: total daily FX turnover USD 7.5 trillion, currency shares out
of 200 % in ``BIS_SHARE_2022`` (BIS 2022, Table 2). A currency's ADV
against USD is taken as ``total * share / 200 %``, which overstates the
liquidity of the USD leg of crosses: a deliberately loose, labelled figure.

Worked example (``tests/test_capacity.py``): one asset, gross return
``g`` with mean 0.01 and sd 0.02 per period, weights 0, 1, 0, 1, ... so |dw| = 1 in every scored period,
h = 0, Y sigma = 0.01, ADV = 1e9. Cost = ``0.01 sqrt(A / 1e9)``, constant,
so SR falls only through the mean: ``SR(A) = (0.01 - 0.01 sqrt(A/1e9)) / 0.02``
and halves at ``sqrt(A/1e9) = 0.5``, i.e. ``A* = 2.5e8``.

Turnover is reported one-way (``sum |dw| / 2``) per period; the holding
period is ``2 * mean gross / mean sum |dw|`` periods (``holding_period``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

BIS_TOTAL_2022_USD = 7.5e12  # BIS Triennial Survey 2022, daily average, April 2022
BIS_SHARE_2022: dict[str, float] = {  # percent of 200, BIS 2022 Table 2
    "USD": 88.5, "EUR": 30.5, "JPY": 16.7, "GBP": 12.9, "CNY": 7.0, "AUD": 6.4,
    "CAD": 6.2, "CHF": 5.2, "HKD": 2.6, "SGD": 2.4, "SEK": 2.2, "KRW": 1.9,
    "NOK": 1.7, "NZD": 1.7, "INR": 1.6, "MXN": 1.5,
}
IMPACT_Y = 1.0  # ASSUMPTION, literature order of magnitude, until W25


def adv_bis(ccy: str, total: float = BIS_TOTAL_2022_USD) -> float:
    """Daily dollar volume from the BIS share; KeyError for unlisted codes."""
    return total * BIS_SHARE_2022[ccy] / 200.0


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover per period, ``sum_i |dw_i| / 2`` (first build counted)."""
    dw = weights.fillna(0.0).diff()
    dw.iloc[0] = weights.fillna(0.0).iloc[0]
    return dw.abs().sum(axis=1) / 2.0


def holding_period(weights: pd.DataFrame) -> float:
    """Average holding period in periods: mean gross / mean two-sided turnover.

    A book of gross G that replaces all of it every k periods sells G and
    buys G once per k periods, so ``sum |dw|`` averages ``2G/k`` and the
    ratio ``2G / (2G/k)`` is k. Example: weights flipping +1/-1 each period have gross 1 and
    ``sum |dw| = 2``, holding period 1.
    """
    w = weights.fillna(0.0)
    gross = w.abs().sum(axis=1).iloc[1:].mean()
    traded = w.diff().abs().sum(axis=1).iloc[1:].mean()
    return float(2.0 * gross / traded) if traded > 0 else math.inf


@dataclass(frozen=True)
class CapacityModel:
    half_spread: Mapping[str, float]   # fraction of notional
    sigma: Mapping[str, float]         # per-period volatility used in the law
    adv: Mapping[str, float]           # dollars per period
    impact_y: float = IMPACT_Y


def cost_series(weights: pd.DataFrame, aum: float, model: CapacityModel) -> pd.Series:
    """Cost per period as a fraction of AUM under the square-root law."""
    w = weights.fillna(0.0)
    dw = w.diff()
    dw.iloc[0] = w.iloc[0]
    a = dw.abs()
    out = pd.Series(0.0, index=w.index)
    for c in w.columns:
        q = a[c]
        lin = q * model.half_spread.get(c, 0.0)
        imp = q * model.impact_y * model.sigma[c] * np.sqrt(aum * q / model.adv[c])
        out = out + lin + imp
    return out


def net_sharpe(gross: pd.Series, weights: pd.DataFrame, aum: float, model: CapacityModel,
               periods: int = 12) -> float:
    r = gross - cost_series(weights, aum, model).reindex(gross.index).fillna(0.0)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan")


def curve(gross: pd.Series, weights: pd.DataFrame, model: CapacityModel,
          aums: np.ndarray, periods: int = 12) -> pd.Series:
    return pd.Series([net_sharpe(gross, weights, float(a), model, periods) for a in aums],
                     index=pd.Index(aums, name="aum"), name="net_sharpe")


def capacity(gross: pd.Series, weights: pd.DataFrame, model: CapacityModel,
             periods: int = 12, lo: float = 1e3, hi: float = 1e13, tol: float = 1e-10) -> float:
    """AUM where net Sharpe is half its zero-size (spread-only) value.

    Returns NaN when the zero-size Sharpe is not positive (nothing to halve)
    and ``inf`` when even ``hi`` does not halve it.
    """
    s0 = net_sharpe(gross, weights, 0.0, model, periods)
    if not s0 > 0:
        return float("nan")
    target = s0 / 2.0

    def f(a: float) -> float:
        return net_sharpe(gross, weights, a, model, periods) - target

    if f(hi) > 0:
        return math.inf
    if f(lo) < 0:
        return lo
    a, b = math.log(lo), math.log(hi)
    while b - a > tol:
        mid = 0.5 * (a + b)
        if f(math.exp(mid)) > 0:
            a = mid
        else:
            b = mid
    return math.exp(0.5 * (a + b))
