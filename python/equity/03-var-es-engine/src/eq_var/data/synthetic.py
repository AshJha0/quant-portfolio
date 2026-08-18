"""Seeded synthetic market-data generators and the canned demo portfolio.

All generators take an explicit ``seed`` / ``numpy.random.Generator`` and run
offline — the test suite depends only on this module, never on live data.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..monte_carlo_var import safe_cholesky, simulate_factor_returns
from ..portfolio import (
    EquityPosition,
    FuturePosition,
    OptionPosition,
    Portfolio,
    RiskFactor,
)

__all__ = [
    "default_covariance",
    "simulate_returns",
    "simulate_garch_returns",
    "demo_portfolio",
    "demo_covariance",
]


def _as_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def default_covariance(
    vols: np.ndarray | list[float], corr: np.ndarray | list[list[float]]
) -> np.ndarray:
    """Covariance from a vol vector (daily) and a correlation matrix."""
    v = np.asarray(vols, dtype=float).ravel()
    c = np.atleast_2d(np.asarray(corr, dtype=float))
    if c.shape != (v.size, v.size):
        raise ValueError(f"correlation shape {c.shape} does not match {v.size} vols")
    return np.outer(v, v) * c


def simulate_returns(
    n_days: int,
    cov: np.ndarray,
    dist: Literal["normal", "t", "garch"] = "normal",
    df: float = 6.0,
    seed: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Simulate a (n_days, n_factors) daily factor-return panel.

    ``dist="normal"``/``"t"`` draw i.i.d. days from ``simulate_factor_returns``
    (t is variance-matched to ``cov``).  ``dist="garch"`` runs a GARCH(1,1)
    volatility process per factor with constant cross-correlation (the
    unconditional covariance matches ``cov``); see
    ``simulate_garch_returns`` for the parameters.
    """
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")
    if dist in ("normal", "t"):
        return simulate_factor_returns(cov, n_days, dist=dist, df=df, seed=seed)
    if dist == "garch":
        return simulate_garch_returns(n_days, cov, seed=seed)
    raise ValueError(f"dist must be 'normal', 't' or 'garch', got {dist!r}")


def simulate_garch_returns(
    n_days: int,
    cov: np.ndarray,
    omega_frac: float = 0.05,
    alpha_g: float = 0.09,
    beta_g: float = 0.86,
    innovations: Literal["normal", "t"] = "t",
    df: float = 7.0,
    seed: int | np.random.Generator | None = 0,
    burn_in: int = 250,
) -> np.ndarray:
    """GARCH(1,1) returns with volatility clustering and constant correlation.

    Per factor i: ``h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}`` with
    ``omega = sigma_i^2 * (1 - alpha - beta)`` so the unconditional variance
    equals ``cov[i, i]``; cross-sectional correlation comes from correlated
    innovations (Cholesky of the correlation matrix); ``innovations="t"``
    (default, df=7) adds conditional fat tails on top of the clustering.
    ``omega_frac`` is unused when alpha+beta parametrisation is given but kept
    for signature stability.  A ``burn_in`` period removes the influence of
    the initial variance.
    """
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")
    if not 0.0 < alpha_g + beta_g < 1.0:
        raise ValueError(f"need 0 < alpha + beta < 1 for stationarity, got {alpha_g + beta_g}")
    rng = _as_rng(seed)
    sig = np.atleast_2d(np.asarray(cov, dtype=float))
    n = sig.shape[0]
    vols = np.sqrt(np.maximum(np.diag(sig), 1e-32))
    denom = np.outer(vols, vols)
    corr = sig / denom
    np.fill_diagonal(corr, 1.0)
    chol = safe_cholesky(corr)
    total = n_days + burn_in
    z = rng.standard_normal((total, n)) @ chol.T
    if innovations == "t":
        if df <= 2:
            raise ValueError(f"df must be > 2, got {df}")
        w = rng.chisquare(df, size=total) / df
        z = z / np.sqrt(w)[:, None] * np.sqrt((df - 2.0) / df)
    elif innovations != "normal":
        raise ValueError(f"innovations must be 'normal' or 't', got {innovations!r}")
    uncond_var = vols**2
    omega = uncond_var * (1.0 - alpha_g - beta_g)
    h = uncond_var.copy()
    out = np.empty((total, n))
    r_prev = np.zeros(n)
    for t in range(total):
        h = omega + alpha_g * r_prev**2 + beta_g * h
        r_prev = np.sqrt(h) * z[t]
        out[t] = r_prev
    return out[burn_in:]


# --------------------------------------------------------------------------- #
# Demo portfolio: 2 stocks + index future + index put + vol factor
# --------------------------------------------------------------------------- #
def demo_portfolio() -> Portfolio:
    """Canned demo book: long tech/bank equity, short index futures hedge,
    long index put protection.

    Factors: AAPL 190, JPM 200, SPX 5000, SPX_IV 18 %.  Positions:
    5 000 AAPL, 4 000 JPM, short 1 SPX future (x50), long 2 SPX 4750 puts
    (3M, x100).  ~$1.75m long equity delta, ~$0.5m hedged away by the
    future + puts, leaving ~$1.2m net long with long gamma/vega protection.
    """
    factors = {
        "AAPL": RiskFactor("AAPL", "equity", 190.0),
        "JPM": RiskFactor("JPM", "equity", 200.0),
        "SPX": RiskFactor("SPX", "index", 5000.0),
        "SPX_IV": RiskFactor("SPX_IV", "vol", 0.18),
    }
    positions = [
        EquityPosition(name="AAPL_stock", factor="AAPL", shares=5000.0),
        EquityPosition(name="JPM_stock", factor="JPM", shares=4000.0),
        FuturePosition(name="SPX_hedge", factor="SPX", contracts=-1.0, multiplier=50.0),
        OptionPosition(
            name="SPX_put_protection",
            underlier="SPX",
            vol_factor="SPX_IV",
            strike=4750.0,
            expiry=0.25,
            rate=0.03,
            div_yield=0.015,
            kind="put",
            contracts=2.0,
            multiplier=100.0,
        ),
    ]
    return Portfolio(positions=positions, factors=factors)


def demo_covariance() -> np.ndarray:
    """Daily factor covariance for the demo portfolio's four factors.

    Annualised vols AAPL 28 %, JPM 24 %, SPX 16 %, SPX_IV 1.5 vol-pt daily
    moves; equities correlate 0.5-0.75 with each other/the index and vol is
    negatively correlated with prices (leverage effect ~ -0.7 vs index).
    Daily vol = annual / sqrt(252); vol-factor 'returns' are absolute daily
    implied-vol changes.
    """
    ann = np.array([0.28, 0.24, 0.16])
    daily = ann / np.sqrt(252.0)
    vols = np.array([daily[0], daily[1], daily[2], 0.015])
    corr = np.array(
        [
            [1.00, 0.50, 0.70, -0.55],
            [0.50, 1.00, 0.65, -0.50],
            [0.70, 0.65, 1.00, -0.70],
            [-0.55, -0.50, -0.70, 1.00],
        ]
    )
    return default_covariance(vols, corr)
