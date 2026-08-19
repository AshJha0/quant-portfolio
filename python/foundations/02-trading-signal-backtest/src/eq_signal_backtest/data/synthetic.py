"""Seeded synthetic price generator: a two-regime trending/correcting market.

A trend-following backtest needs data with *trends* in it, or the exercise
is vacuous. This generator is a two-state Markov-switching model with
deliberately persistent regimes -- long calm uptrends punctuated by
shorter, sharper drawdown regimes -- because that is the shape of market
in which a moving-average crossover is supposed to earn its keep, and
having it in the bundled data is what makes the *failure* case (the
mean-reverting path used in ``tests/test_edge_cases.py``) informative by
contrast.

It is **not** real market data. Every documented number in this project is
a property of this generator plus a seed, not a claim about any real
asset; see ``README.md`` and ``docs/VALIDATION.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["generate"]

#: Fixed final business day of every generated sample, so the bundled
#: series (and therefore every number in the docs) never depends on the
#: date the generator happens to be run.
_END_DATE = "2026-08-14"

# Regime parameters (annualised drift/vol, ACT/252). Chosen so the
# *stationary* properties look like a single large-cap equity rather than
# a strategy-flattering fantasy: ~8%/yr long-run drift, ~21% annualised
# volatility, deep (30-50%) drawdowns during the stressed regime.
_UP_MU, _UP_VOL = 0.18, 0.17
_DOWN_MU, _DOWN_VOL = -0.40, 0.33
# Switch probabilities per day: uptrends last ~200 trading days on
# average, drawdown regimes ~40, giving a stationary stressed share of
# p_ud / (p_ud + p_du) = 17%. Persistent enough that a slow crossover can
# in principle detect the change, short enough that it often cannot.
_P_UP_TO_DOWN = 0.005
_P_DOWN_TO_UP = 0.025
_T_DF = 5.0  # Student-t shocks -> fat tails

#: Default seed. Any fixed seed is a single draw, and a 10-year path from
#: this model ranges from roughly -14% to +28% CAGR depending on the draw.
#: 32 is used because its path sits close to the model's central case
#: (~8.6% CAGR, ~21% vol, Sharpe ~0.5, -47% peak drawdown) rather than
#: being an unusually kind or unusually cruel sample -- the economics
#: live in the parameters above, and a demo dataset should not smuggle in
#: an extra tailwind through the seed.
_DEFAULT_SEED = 32

_TRADING_DAYS = 252


def generate(
    n_days: int = 2520,
    start_price: float = 100.0,
    seed: int | np.random.Generator | None = _DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate a deterministic synthetic daily close series.

    Two-state Markov chain (uptrend / drawdown) driving drift and
    volatility, with unit-variance Student-t(5) daily shocks:

    ``r_t = mu_regime / 252 + vol_regime / sqrt(252) * t_t``

    Uptrend: mu=+18%/yr, vol=17%/yr, 0.5% chance per day of flipping to
    the drawdown regime. Drawdown: mu=-40%/yr, vol=33%/yr, 2.5% chance
    per day of recovering. Long-run: ~17% of days stressed, ~8%/yr drift,
    ~21% annualised volatility. Prices compound the simple returns from
    ``start_price``.

    Parameters
    ----------
    n_days : int
        Number of business days (rows) to generate, >= 1.
    start_price : float
        Initial price level, strictly positive.
    seed : int, numpy.random.Generator or None
        Seed or explicit generator. The default (32) is the seed every
        documented result in this project uses; see ``_DEFAULT_SEED``
        above for why that particular draw.

    Returns
    -------
    pandas.DataFrame
        Columns ``Date`` (datetime64, ascending business days) and
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

    switch_u = rng.uniform(size=n_days)
    regime = np.zeros(n_days, dtype=np.int64)  # 0 = uptrend, 1 = drawdown
    for t in range(1, n_days):
        if regime[t - 1] == 0:
            regime[t] = 1 if switch_u[t] < _P_UP_TO_DOWN else 0
        else:
            regime[t] = 0 if switch_u[t] < _P_DOWN_TO_UP else 1

    shocks = rng.standard_t(_T_DF, size=n_days) * np.sqrt((_T_DF - 2.0) / _T_DF)
    mu = np.where(regime == 1, _DOWN_MU, _UP_MU) / _TRADING_DAYS
    vol = np.where(regime == 1, _DOWN_VOL, _UP_VOL) / np.sqrt(_TRADING_DAYS)
    daily_returns = mu + vol * shocks
    # Guard against a Student-t draw below -100% (never binds at these vol
    # levels, but a non-positive price would be a silent disaster).
    daily_returns = np.maximum(daily_returns, -0.60)

    growth = np.concatenate(([1.0], 1.0 + daily_returns[1:]))
    prices = start_price * np.cumprod(growth)

    dates = pd.bdate_range(end=_END_DATE, periods=n_days)
    return pd.DataFrame({"Date": dates, "Adj Close": prices})
