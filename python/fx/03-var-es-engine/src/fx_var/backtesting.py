"""VaR/ES backtesting: Kupiec, Christoffersen, Basel traffic light, ES test.

Exception convention: day t is an exception when the realised **loss**
exceeds the VaR forecast made for day t: ``-pnl_t > var_t``.

Tests
-----
* Kupiec POF (unconditional coverage): LR test that the exception
  frequency equals ``1 - alpha``; chi2(1).
* Christoffersen independence: LR test against first-order Markov
  clustering of exceptions; chi2(1).  FX desks care because volatility
  clustering makes an unconditional method (plain HS, sample-cov
  parametric) fail *this* test long before it fails Kupiec.
* Conditional coverage: LR_cc = LR_uc + LR_ind; chi2(2).
* Basel traffic light: zones from the cumulative Binomial(n, 1-alpha)
  probability of the exception count - green < 95%, yellow < 99.99%, red
  >= 99.99% - which for the regulatory 250-day, 99% window gives exactly
  green 0-4 / yellow 5-9 / red 10+, with the capital multiplier add-ons
  of the 1996 Basel table.
* ES backtest: Acerbi-Szekely (2014) unconditional test statistic
  ``Z = (1/(n(1-alpha))) * sum(L_t 1{L_t > VaR_t} / ES_t) - 1``,
  p-value by seeded parametric simulation under H0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2, norm

from .book import Book, Market
from .common import validate_alpha

__all__ = [
    "kupiec_pof",
    "christoffersen_independence",
    "conditional_coverage",
    "TrafficLight",
    "basel_traffic_light",
    "BacktestResult",
    "evaluate_var_backtest",
    "rolling_backtest",
    "es_backtest_acerbi_szekely",
]

#: 1996 Basel yellow-zone capital multiplier add-ons by exception count
#: (250-day, 99% window); multiplier = 3.0 + add-on, capped at 4.0 in red.
BASEL_ADDONS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def _xlogy(x: float, y: float) -> float:
    """x*log(y) with the 0*log(0) = 0 convention."""
    if x == 0:
        return 0.0
    if y <= 0:
        return -np.inf
    return x * np.log(y)


def kupiec_pof(n_exceptions: int, n_obs: int, alpha: float = 0.99) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR test.

    Returns ``(LR_uc, p_value)`` where LR_uc ~ chi2(1) under H0 that the
    exception probability is ``p = 1 - alpha``.
    """
    validate_alpha(alpha)
    x, n = int(n_exceptions), int(n_obs)
    if n <= 0 or not (0 <= x <= n):
        raise ValueError(f"need 0 <= n_exceptions <= n_obs, got x={x}, n={n}")
    p = 1.0 - alpha
    pi_hat = x / n
    ll0 = _xlogy(n - x, 1.0 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1.0 - pi_hat) + _xlogy(x, pi_hat)
    lr = -2.0 * (ll0 - ll1)
    lr = max(lr, 0.0)
    return float(lr), float(chi2.sf(lr, df=1))


def christoffersen_independence(exceedances) -> tuple[float, float]:
    """Christoffersen (1998) independence LR test on a 0/1 exception series.

    Tests first-order Markov dependence: H0 says P(exception | exception
    yesterday) = P(exception | none yesterday).  Returns ``(LR_ind, p)``,
    chi2(1).  Degenerate series (no transitions of one kind) return LR=0.
    """
    e = np.asarray(exceedances).astype(int).ravel()
    if e.size < 2:
        raise ValueError("need at least 2 observations for the independence test")
    if not np.isin(e, (0, 1)).all():
        raise ValueError("exceedances must be a 0/1 (or boolean) series")
    prev, curr = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    ll1 = (_xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01)
           + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11))
    lr = -2.0 * (ll0 - ll1)
    lr = max(lr, 0.0) if np.isfinite(lr) else 0.0
    return float(lr), float(chi2.sf(lr, df=1))


def conditional_coverage(exceedances, alpha: float = 0.99) -> tuple[float, float]:
    """Christoffersen conditional coverage: LR_cc = LR_uc + LR_ind, chi2(2)."""
    e = np.asarray(exceedances).astype(int).ravel()
    lr_uc, _ = kupiec_pof(int(e.sum()), e.size, alpha)
    lr_ind, _ = christoffersen_independence(e)
    lr_cc = lr_uc + lr_ind
    return float(lr_cc), float(chi2.sf(lr_cc, df=2))


@dataclass(frozen=True)
class TrafficLight:
    """Basel traffic-light outcome for a backtest window."""

    zone: str  # "green" | "yellow" | "red"
    n_exceptions: int
    n_obs: int
    cumulative_prob: float  # P(X <= n_exceptions) under Binomial(n, 1-alpha)
    multiplier: float  # capital multiplier (3.0 .. 4.0)


def basel_traffic_light(
    n_exceptions: int, n_obs: int = 250, alpha: float = 0.99
) -> TrafficLight:
    """Basel traffic-light zone and capital multiplier.

    Zones follow the Basel definition on the cumulative binomial
    probability of observing <= x exceptions when the model is correct:
    green < 95%, yellow in [95%, 99.99%), red >= 99.99%.  For n=250,
    alpha=0.99 this reproduces the regulatory table exactly:
    green 0-4, yellow 5-9 (add-ons 0.40/0.50/0.65/0.75/0.85), red >= 10
    (multiplier 4.0).
    """
    validate_alpha(alpha)
    x, n = int(n_exceptions), int(n_obs)
    if n <= 0 or not (0 <= x <= n):
        raise ValueError(f"need 0 <= n_exceptions <= n_obs, got x={x}, n={n}")
    p = 1.0 - alpha
    cum = float(binom.cdf(x, n, p))
    if cum < 0.95:
        zone, mult = "green", 3.0
    elif cum < 0.9999:
        zone = "yellow"
        mult = 3.0 + BASEL_ADDONS.get(x, 0.85 if x > max(BASEL_ADDONS) else 0.40)
    else:
        zone, mult = "red", 4.0
    return TrafficLight(zone, x, n, cum, mult)


@dataclass(frozen=True)
class BacktestResult:
    """Full VaR backtest summary over a forecast/realisation window."""

    n_obs: int
    n_exceptions: int
    exception_rate: float
    expected_rate: float
    kupiec_lr: float
    kupiec_p: float
    independence_lr: float
    independence_p: float
    cc_lr: float
    cc_p: float
    traffic_light: TrafficLight
    exceedances: np.ndarray


def evaluate_var_backtest(pnl, var_forecasts, alpha: float = 0.99) -> BacktestResult:
    """Score a realised P&L series against out-of-sample VaR forecasts.

    Parameters
    ----------
    pnl : array_like
        Realised P&L (profit +) per day.
    var_forecasts : array_like
        Positive VaR forecast for the same day (made ex ante).
    alpha : float
        VaR confidence level.
    """
    validate_alpha(alpha)
    pnl = np.asarray(pnl, dtype=float).ravel()
    var = np.asarray(var_forecasts, dtype=float).ravel()
    if pnl.size != var.size:
        raise ValueError("pnl and var_forecasts must have equal length")
    if pnl.size < 2:
        raise ValueError("need at least 2 observations to backtest")
    if np.isnan(pnl).any() or np.isnan(var).any():
        raise ValueError("backtest inputs contain NaNs (NaN policy: refuse)")
    exc = (-pnl > var).astype(int)
    x = int(exc.sum())
    lr_uc, p_uc = kupiec_pof(x, exc.size, alpha)
    lr_ind, p_ind = christoffersen_independence(exc)
    lr_cc = lr_uc + lr_ind
    p_cc = float(chi2.sf(lr_cc, df=2))
    tl = basel_traffic_light(x, exc.size, alpha)
    return BacktestResult(exc.size, x, x / exc.size, 1.0 - alpha, lr_uc, p_uc,
                          lr_ind, p_ind, lr_cc, p_cc, tl, exc)


def rolling_backtest(
    book: Book,
    market: Market,
    returns: pd.DataFrame,
    var_fn: Callable[[Book, Market, pd.DataFrame], float],
    window: int = 250,
    alpha: float = 0.99,
    option_method: str = "full",
) -> pd.DataFrame:
    """Rolling out-of-sample VaR backtest on a factor-return history.

    For each day t >= window, ``var_fn(book, market, returns[t-window:t])``
    produces the ex-ante VaR, and the realised (hypothetical) P&L is the
    full revaluation of today's book under day t's factor returns - the
    standard static-book backtest P&L (see docs/DESK_GUIDE.md).

    Returns
    -------
    pandas.DataFrame
        Columns ``pnl``, ``var``, ``exceed`` indexed like ``returns``
        (first ``window`` days dropped).
    """
    if window < 30:
        raise ValueError("window must be >= 30")
    if len(returns) <= window:
        raise ValueError(
            f"need more than window={window} rows of history, got {len(returns)}"
        )
    factors = book.factors(market)
    rows = []
    idx = []
    pnl_all = np.asarray(
        book.pnl(market, returns[factors], option_method=option_method), dtype=float
    )
    for t in range(window, len(returns)):
        win = returns.iloc[t - window : t]
        v = float(var_fn(book, market, win))
        realised = float(pnl_all[t])
        rows.append((realised, v, float(-realised > v)))
        idx.append(returns.index[t])
    out = pd.DataFrame(rows, columns=["pnl", "var", "exceed"], index=idx)
    return out


def es_backtest_acerbi_szekely(
    pnl,
    var_forecasts,
    es_forecasts,
    alpha: float = 0.975,
    n_sim: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Acerbi-Szekely (2014) unconditional ES backtest (their test 2).

    ``Z = (1/(n(1-alpha))) * sum_t L_t 1{L_t > VaR_t} / ES_t - 1``.
    Under a correct model E[Z] ~ 0; Z > 0 signals ES underestimation.
    The p-value is obtained by seeded parametric simulation under H0 with
    normal losses matched to each day's forecast (sigma_t implied from
    ES_t), i.e. P(Z_sim >= Z_obs).

    Returns ``(z_stat, p_value)``.
    """
    validate_alpha(alpha)
    pnl = np.asarray(pnl, dtype=float).ravel()
    var = np.asarray(var_forecasts, dtype=float).ravel()
    es = np.asarray(es_forecasts, dtype=float).ravel()
    if not (pnl.size == var.size == es.size):
        raise ValueError("pnl, var_forecasts and es_forecasts must align")
    if np.any(es <= 0):
        raise ValueError("es_forecasts must be positive losses")
    n = pnl.size
    losses = -pnl
    z_obs = float(np.sum(losses * (losses > var) / es) / (n * (1.0 - alpha)) - 1.0)

    # H0 simulation: normal losses with sigma_t implied from the ES forecast
    z_a = norm.ppf(alpha)
    es_factor = norm.pdf(z_a) / (1.0 - alpha)
    sig = es / es_factor
    var0 = sig * z_a
    rng = np.random.default_rng(seed)
    sims = rng.standard_normal((n_sim, n)) * sig[None, :]
    z_sim = np.sum(sims * (sims > var0[None, :]) / es[None, :], axis=1) / (
        n * (1.0 - alpha)
    ) - 1.0
    p = float((np.sum(z_sim >= z_obs) + 1) / (n_sim + 1))
    return z_obs, p
