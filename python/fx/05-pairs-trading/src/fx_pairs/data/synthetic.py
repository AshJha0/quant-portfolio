"""Seeded synthetic FX data generators (all offline, all deterministic).

Every generator takes an explicit ``seed`` and returns pandas objects on a
business-day index.  USD-leg panels are quoted CCYUSD (USD per 1 unit of
currency); market pairs and crosses are built from them with
:mod:`fx_pairs.universe`, so triangular consistency holds by construction.

Generators
----------
* :func:`make_cointegrated_pair` — two currency pairs whose log rates share a
  random-walk common factor with a known hedge ratio and a known OU spread
  (ground truth returned for recovery tests).
* :func:`make_correlated_walks` — correlated but *independent* random walks:
  high return correlation, no cointegration (the classic spurious-regression
  trap the funnel must reject).
* :func:`make_two_block_panel` — risk-on/risk-off two-block correlation
  structure (AUD/NZD/CAD commodity block vs JPY/CHF safe havens) with
  occasional regime flips where the cross-block correlation turns sharply
  negative.
* :func:`make_floor_then_break` — SNB-style pegged spread: years of tiny-vol
  mean reversion around a floor, then a single-day catastrophic break and a
  new high-vol regime (EURCHF, 2011-2015).
* :func:`make_carry_flip_pair` — a pair with a persistent deposit-rate
  differential whose spot spread drifts at the forward premium: a spot-only
  signal is systematically wrong-carry and its P&L flips sign once carry is
  accounted for.
* :func:`make_deposit_rate_panel` — slowly varying deposit-rate curves.
* :func:`make_pegged_pair` — a hard peg (zero volatility) for edge-case tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "business_days",
    "simulate_ou",
    "make_cointegrated_pair",
    "make_correlated_walks",
    "make_two_block_panel",
    "make_floor_then_break",
    "make_carry_flip_pair",
    "make_deposit_rate_panel",
    "make_pegged_pair",
]

_DT = 1.0 / 252.0


def business_days(n: int, start: str = "2015-01-02") -> pd.DatetimeIndex:
    """Business-day index of length ``n`` starting at ``start``."""
    return pd.bdate_range(start=start, periods=n)


def simulate_ou(
    n: int,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float = _DT,
    x0: float | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Exact-discretisation OU path ``dx = kappa (theta - x) dt + sigma dW``.

    ``kappa`` and ``sigma`` are annualised; ``dt`` in years.  ``x0`` defaults
    to a draw from the stationary distribution.
    """
    if kappa <= 0 or sigma <= 0:
        raise ValueError("kappa and sigma must be positive")
    if rng is None:
        rng = np.random.default_rng(seed)
    phi = np.exp(-kappa * dt)
    eps_sd = sigma * np.sqrt((1.0 - phi**2) / (2.0 * kappa))
    stat_sd = sigma / np.sqrt(2.0 * kappa)
    x = np.empty(n)
    x[0] = theta + stat_sd * rng.standard_normal() if x0 is None else x0
    shocks = eps_sd * rng.standard_normal(n - 1)
    for t in range(1, n):
        x[t] = theta + (x[t - 1] - theta) * phi + shocks[t - 1]
    return x


def make_cointegrated_pair(
    n: int = 1500,
    beta: float = 1.0,
    alpha: float = 0.0,
    kappa: float = 20.0,
    sigma_ou: float = 0.05,
    sigma_rw: float = 0.10,
    p2_0: float = 0.65,
    seed: int = 0,
    start: str = "2015-01-02",
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Cointegrated pair of currency pairs with known ground truth.

    ``log p2`` is a driftless random walk (annualised vol ``sigma_rw``);
    ``log p1 = alpha + beta log p2 + s`` with ``s`` an OU spread
    (annualised ``kappa``, ``sigma_ou``).  Think AUDUSD vs NZDUSD: a common
    commodity/USD factor plus a stationary relative-value spread.

    Returns
    -------
    (p1, p2, truth)
        Price series and ``{"alpha", "beta", "kappa", "sigma_ou", ...}``.
    """
    rng = np.random.default_rng(seed)
    idx = business_days(n, start)
    lp2 = np.log(p2_0) + np.cumsum(
        np.concatenate([[0.0], sigma_rw * np.sqrt(_DT) * rng.standard_normal(n - 1)])
    )
    s = simulate_ou(n, kappa, 0.0, sigma_ou, rng=rng)
    lp1 = alpha + beta * lp2 + s
    p1 = pd.Series(np.exp(lp1), index=idx, name="P1")
    p2 = pd.Series(np.exp(lp2), index=idx, name="P2")
    truth = {"alpha": alpha, "beta": beta, "kappa": kappa,
             "sigma_ou": sigma_ou, "sigma_rw": sigma_rw,
             "half_life_days": float(np.log(2.0) / (kappa * _DT))}
    return p1, p2, truth


def make_correlated_walks(
    n: int = 1000,
    rho: float = 0.9,
    sigma: float = 0.10,
    seed: int = 1,
    start: str = "2015-01-02",
) -> tuple[pd.Series, pd.Series]:
    """Correlated but NOT cointegrated pair: independent unit roots.

    Daily log returns are jointly normal with correlation ``rho``, but the
    levels are two independent random walks — return correlation without any
    stationary linear combination.  The spurious-regression trap: levels OLS
    will 'fit', but Engle-Granger must (usually) fail to reject no
    cointegration.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((2, n - 1))
    r1 = z[0]
    r2 = rho * z[0] + np.sqrt(1.0 - rho**2) * z[1]
    idx = business_days(n, start)
    lp1 = np.cumsum(np.concatenate([[0.0], sigma * np.sqrt(_DT) * r1]))
    lp2 = np.cumsum(np.concatenate([[0.0], sigma * np.sqrt(_DT) * r2]))
    p1 = pd.Series(np.exp(lp1), index=idx, name="W1")
    p2 = pd.Series(np.exp(0.3 + lp2), index=idx, name="W2")
    return p1, p2


def make_two_block_panel(
    n: int = 1500,
    block_a: tuple[str, ...] = ("AUD", "NZD", "CAD"),
    block_b: tuple[str, ...] = ("JPY", "CHF"),
    rho_intra: float = 0.85,
    rho_cross_calm: float = 0.10,
    rho_cross_riskoff: float = -0.60,
    n_flips: int = 3,
    flip_len: int = 60,
    sigma: float = 0.10,
    seed: int = 2,
    start: str = "2015-01-02",
) -> tuple[pd.DataFrame, pd.Series]:
    """USD-leg panel with a two-block risk-on/risk-off correlation structure.

    Block A (commodity currencies) and block B (safe havens) each load on a
    block factor with intra-block return correlation ``rho_intra``.  The two
    block factors have correlation ``rho_cross_calm`` in the calm regime; in
    ``n_flips`` risk-off episodes of ``flip_len`` days (evenly spaced) the
    factor correlation flips to ``rho_cross_riskoff`` — commodity currencies
    sell off as safe havens rally.  Implied pairwise cross-block correlation
    is ``rho_intra * rho_cross`` in each regime.

    Returns
    -------
    (legs, regime)
        ``legs``: DataFrame of CCYUSD levels (all start at plausible G10
        levels).  ``regime``: 0 = calm, 1 = risk-off.
    """
    if not (0 < rho_intra < 1):
        raise ValueError("rho_intra must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = business_days(n, start)
    regime = np.zeros(n, dtype=int)
    if n_flips > 0:
        seg = n // (n_flips + 1)
        for j in range(1, n_flips + 1):
            s0 = j * seg
            regime[s0 : min(s0 + flip_len, n)] = 1

    rho_f = np.where(regime == 1, rho_cross_riskoff, rho_cross_calm)[1:]
    zf = rng.standard_normal((2, n - 1))
    f_a = zf[0]
    f_b = rho_f * zf[0] + np.sqrt(1.0 - rho_f**2) * zf[1]

    w = np.sqrt(rho_intra)
    e = np.sqrt(1.0 - rho_intra)
    levels = {"AUD": 0.75, "NZD": 0.68, "CAD": 0.78, "JPY": 0.0090,
              "CHF": 1.05, "EUR": 1.10, "GBP": 1.45, "SEK": 0.12, "NOK": 0.13}
    data = {}
    for ccy in block_a:
        idio = rng.standard_normal(n - 1)
        r = sigma * np.sqrt(_DT) * (w * f_a + e * idio)
        data[ccy] = levels.get(ccy, 1.0) * np.exp(np.cumsum(np.concatenate([[0.0], r])))
    for ccy in block_b:
        idio = rng.standard_normal(n - 1)
        r = sigma * np.sqrt(_DT) * (w * f_b + e * idio)
        data[ccy] = levels.get(ccy, 1.0) * np.exp(np.cumsum(np.concatenate([[0.0], r])))
    legs = pd.DataFrame(data, index=idx)
    return legs, pd.Series(regime, index=idx, name="regime")


def make_floor_then_break(
    n_pre: int = 750,
    n_post: int = 250,
    beta: float = 1.0,
    kappa: float = 60.0,
    sigma_floor: float = 0.02,
    sigma_post: float = 0.12,
    jump: float = -0.15,
    dip_len: int = 12,
    dip_z: float = -4.0,
    sigma_rw: float = 0.08,
    seed: int = 3,
    start: str = "2012-01-02",
) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    """SNB-style floor-then-break: a 'perfect' mean reverter that ends in a gap.

    Pre-break (``n_pre`` days): the spread is a tight, fast OU around 0 —
    exactly what a cointegration scan scores highest (EURCHF under the
    1.20 floor, 2011-2015).  In the final ``dip_len`` pre-break days the
    spread sits at ``dip_z`` stationary standard deviations below the mean, so
    a z-score strategy is *long the spread into the break* (long the pegged
    currency pair — the crowd's position at the floor).  On the break day the
    spread gaps by ``jump`` (log units, e.g. -0.15 = -15%), then follows a
    high-vol OU around the new level.

    Returns
    -------
    (p1, p2, meta)
        ``meta["break_idx"]`` is the integer location of the break day;
        ``meta["stat_sd"]`` the pre-break stationary spread s.d.
    """
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    idx = business_days(n, start)
    stat_sd = sigma_floor / np.sqrt(2.0 * kappa)

    s_pre = simulate_ou(n_pre, kappa, 0.0, sigma_floor, rng=rng, x0=0.0)
    # engineered dip: the spread pins below the mean just before the break
    s_pre[n_pre - dip_len:] = dip_z * stat_sd
    s_post = simulate_ou(n_post, kappa / 4.0, dip_z * stat_sd + jump,
                         sigma_post, rng=rng, x0=dip_z * stat_sd + jump)
    s = np.concatenate([s_pre, s_post])

    lp2 = np.log(1.20) + np.cumsum(
        np.concatenate([[0.0], sigma_rw * np.sqrt(_DT) * rng.standard_normal(n - 1)])
    )
    lp1 = beta * lp2 + s
    p1 = pd.Series(np.exp(lp1), index=idx, name="PEGGED")
    p2 = pd.Series(np.exp(lp2), index=idx, name="ANCHOR")
    meta = {"break_idx": n_pre, "stat_sd": float(stat_sd), "jump": jump,
            "beta": beta}
    return p1, p2, meta


def make_carry_flip_pair(
    n: int = 1250,
    r_high: float = 0.08,
    r_low: float = 0.01,
    r_usd: float = 0.01,
    kappa: float = 25.0,
    sigma_ou: float = 0.03,
    sigma_rw: float = 0.09,
    drift_fraction: float = 0.5,
    seed: int = 4,
    start: str = "2015-01-02",
) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    """Pair where ignoring carry flips the sign of strategy P&L.

    Pair 1 is a high-yield currency vs USD (deposit rate ``r_high`` vs
    ``r_usd``); pair 2 a low-yielder (``r_low``).  Pair 1's spot depreciates
    at ``drift_fraction`` times the rate differential — the empirical forward
    premium puzzle: high-yielders depreciate by *less* than the differential
    (UIP fails), so the carry more than compensates the drift.  Its *spot*
    spread vs pair 2 trends down persistently while its *total-return* spread
    (spot + carry) mean-reverts around a mild upward bias.

    A spot-only z-score signal keeps seeing the spread 'cheap' vs its
    formation mean and goes long — losing money on spot as the drift
    continues, while a long position earns exactly that differential in
    carry.  Hence: spot-only backtest negative, carry-inclusive positive.
    This is the EM/carry trap: on spot alone the mean-reversion signal is
    systematically wrong-carry (here the drift *pays* the long; sell signals
    on the same pair would be systematically wrong-carry in the adverse
    direction).

    Returns
    -------
    (p1, p2, meta)
        ``meta["rates"]`` is the rates dict for :func:`fx_pairs.backtest.run_backtest`.
    """
    rng = np.random.default_rng(seed)
    idx = business_days(n, start)
    # spot depreciation of the high-yielder, per year (partial UIP)
    drift = -(r_high - r_low) * drift_fraction
    t = np.arange(n) * _DT
    lp2 = np.log(0.65) + np.cumsum(
        np.concatenate([[0.0], sigma_rw * np.sqrt(_DT) * rng.standard_normal(n - 1)])
    )
    s = simulate_ou(n, kappa, 0.0, sigma_ou, rng=rng, x0=0.0)
    lp1 = np.log(0.30) + lp2 - np.log(0.65) + drift * t + s
    p1 = pd.Series(np.exp(lp1), index=idx, name="HYUSD")
    p2 = pd.Series(np.exp(lp2), index=idx, name="LYUSD")
    rates = {"rb1": r_high, "rq1": r_usd, "rb2": r_low, "rq2": r_usd}
    meta = {"rates": rates, "beta": 1.0, "drift": drift,
            "carry_per_day_long": (r_high - r_usd - (r_low - r_usd)) / 365.0}
    return p1, p2, meta


def make_deposit_rate_panel(
    index: pd.DatetimeIndex,
    levels: dict[str, float],
    vol: float = 0.002,
    kappa: float = 2.0,
    seed: int = 5,
) -> pd.DataFrame:
    """Slowly varying deposit-rate panel (annualised simple rates, ACT/365F).

    Each currency's rate follows a slow OU around its ``levels`` entry with
    annualised vol ``vol``, floored at -1% (post-2015 CHF/JPY style negative
    rates are allowed, hyper-negative are not).
    """
    rng = np.random.default_rng(seed)
    n = len(index)
    data = {}
    for ccy, lvl in levels.items():
        path = simulate_ou(n, kappa, lvl, vol, rng=rng, x0=lvl)
        data[ccy] = np.maximum(path, -0.01)
    return pd.DataFrame(data, index=index)


def make_pegged_pair(
    n: int = 500,
    level: float = 3.6725,
    start: str = "2015-01-02",
    name: str = "PEG",
) -> pd.Series:
    """Hard-pegged pair (e.g. AED-style): constant price, zero volatility."""
    return pd.Series(level, index=business_days(n, start), name=name)
