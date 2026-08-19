"""Seeded synthetic daily price generator with realistic stylised facts.

A two-regime (calm/stressed) Markov-switching model with Student-t shocks.
It exists so the whole project runs offline and deterministically while
still reproducing the stylised facts of real daily equity returns that the
analysis depends on:

- **fat tails** (excess kurtosis well above 0) -- so historical vs Gaussian
  VaR genuinely disagree at high confidence;
- **volatility clustering** -- stressed days arrive in runs, so rolling and
  EWMA volatility diverge materially from the full-sample figure;
- **occasional drawdowns** -- the stressed regime has a negative drift.

It is **not** real market data and no conclusion about a real asset should
be drawn from it; see ``README.md`` ("Data") and ``docs/VALIDATION.md``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gamma

__all__ = ["generate"]

#: Fixed final business day of every generated sample. Anchoring the *end*
#: keeps the default 10-year sample stable across machines and run dates
#: (determinism is a hard requirement of the test suite and docs).
_END_DATE = "2026-08-14"

# Regime parameters (annualised drift/vol; ACT/252 daily scaling).
_CALM_MU, _CALM_VOL = 0.12, 0.10
_STRESS_MU, _STRESS_VOL = -0.15, 0.30
_P_CALM_TO_STRESS = 0.02  # per-day switch probabilities -> stressed
_P_STRESS_TO_CALM = 0.10  # ~17% of days stressed in the stationary law
_T_DF = 6.0  # Student-t degrees of freedom for daily shocks
# Two-piece scaling of the (symmetric) t shock: down-moves are stretched and
# up-moves compressed, which is what produces the negative skew real equity
# returns show. The pair is chosen so the shock keeps ~unit variance.
_DOWN_SCALE, _UP_SCALE = 1.25, 0.80

_TRADING_DAYS = 252


def _skewed_t_shocks(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` i.i.d. negatively skewed shocks with mean 0 and variance 1.

    A Student-t(``_T_DF``) draw is standardised to unit variance, then
    scaled two-piece (``_DOWN_SCALE`` below zero, ``_UP_SCALE`` above) to
    introduce negative skew, then re-standardised using the **closed-form**
    mean and variance of that transform -- not sample moments -- so the
    shock distribution is exactly mean-0/unit-variance in expectation and
    the regime drift and volatility mean what the docstring says they do
    at every sample length.

    For a symmetric unit-variance ``X`` with ``E|X| = m1``, the two-piece
    transform ``Y = a X`` (``X < 0``) / ``b X`` (``X >= 0``) has
    ``E[Y] = (b - a) m1 / 2`` and ``E[Y^2] = (a^2 + b^2) / 2``.
    """
    x = rng.standard_t(_T_DF, size=n) * np.sqrt((_T_DF - 2.0) / _T_DF)
    y = np.where(x < 0.0, _DOWN_SCALE * x, _UP_SCALE * x)
    # E|t_nu| = 2 sqrt(nu) Gamma((nu+1)/2) / ((nu-1) sqrt(pi) Gamma(nu/2)),
    # rescaled by sqrt((nu-2)/nu) for the unit-variance standardisation.
    m1 = (
        2.0
        * np.sqrt(_T_DF)
        * gamma((_T_DF + 1.0) / 2.0)
        / ((_T_DF - 1.0) * np.sqrt(np.pi) * gamma(_T_DF / 2.0))
    ) * np.sqrt((_T_DF - 2.0) / _T_DF)
    mean = (_UP_SCALE - _DOWN_SCALE) * m1 / 2.0
    var = (_DOWN_SCALE**2 + _UP_SCALE**2) / 2.0 - mean**2
    return (y - mean) / np.sqrt(var)


def generate(
    n_days: int = 2520,
    start_price: float = 100.0,
    seed: int | np.random.Generator | None = 2,
) -> pd.DataFrame:
    """Generate a deterministic synthetic daily adjusted-close series.

    The model: a two-state Markov chain (calm/stressed) drives the daily
    drift and volatility; within a regime, the daily simple return is

    ``r_t = mu_regime / 252 + vol_regime / sqrt(252) * t_t``

    where ``t_t`` are i.i.d. Student-t(6) shocks standardised to unit
    variance and then scaled two-piece (down-moves x1.25, up-moves x0.80)
    so the shock distribution is negatively skewed, as real daily equity
    returns are. Calm regime: mu=+12%/yr, vol=10%/yr; stressed:
    mu=-15%/yr, vol=30%/yr; switch probabilities 2% (calm->stressed) and
    10% (stressed->calm) per day. Prices compound simple returns from
    ``start_price``.

    The date index is business days ending at a **fixed anchor date**
    (2026-08-14), so the same ``(n_days, seed)`` pair always yields the
    same frame, regardless of when it is generated.

    Parameters
    ----------
    n_days : int
        Number of business days (and rows) to generate, >= 1.
    start_price : float
        First price level, strictly positive. Units: currency per share.
    seed : int, numpy.random.Generator or None
        Seed (or an explicit generator) for reproducibility. ``None``
        draws fresh OS entropy -- fine for exploration, never used in
        tests. Default 2 is the seed all documented results use.

    Returns
    -------
    pandas.DataFrame
        Columns ``Date`` (datetime64, business days, ascending) and
        ``Adj Close`` (float, strictly positive), ``n_days`` rows.

    Raises
    ------
    ValueError
        If ``n_days < 1`` or ``start_price <= 0``.
    """
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")
    if not start_price > 0:
        raise ValueError(f"start_price must be > 0, got {start_price}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    # Regime path: 0 = calm, 1 = stressed, started in calm.
    switch_u = rng.uniform(size=n_days)
    regime = np.zeros(n_days, dtype=np.int64)
    for t in range(1, n_days):
        if regime[t - 1] == 0:
            regime[t] = 1 if switch_u[t] < _P_CALM_TO_STRESS else 0
        else:
            regime[t] = 0 if switch_u[t] < _P_STRESS_TO_CALM else 1

    shocks = _skewed_t_shocks(rng, n_days)

    mu = np.where(regime == 1, _STRESS_MU, _CALM_MU) / _TRADING_DAYS
    vol = np.where(regime == 1, _STRESS_VOL, _CALM_VOL) / np.sqrt(_TRADING_DAYS)
    daily_returns = mu + vol * shocks

    # A Student-t shock below -1 daily return would send the price
    # non-positive; clip at -60% (far beyond any plausible single day for
    # the vol levels used, so in practice this never binds).
    daily_returns = np.maximum(daily_returns, -0.60)

    # Row 0 is the starting level itself; rows 1..n-1 compound the returns.
    growth = np.concatenate(([1.0], 1.0 + daily_returns[1:]))
    prices = start_price * np.cumprod(growth)

    dates = pd.bdate_range(end=_END_DATE, periods=n_days)
    return pd.DataFrame({"Date": dates, "Adj Close": prices})
