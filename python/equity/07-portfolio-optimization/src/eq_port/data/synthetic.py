"""Seeded synthetic multi-asset return generator with known moments.

Returns are simple (arithmetic) per-period returns at a daily frequency
(252 periods per year). The generator uses a linear factor structure

    r_t = mu + beta * f_mkt,t + Gamma * f_sec,t + eps_t

with one market factor, ``n_sectors`` sector factors and idiosyncratic
noise, all mean-zero Gaussian and mutually independent, so the TRUE
covariance is known in closed form:

    Sigma = sigma_mkt^2 * beta beta' + Gamma diag(sigma_sec^2) Gamma' + D.

An optional crisis regime multiplies the market-factor volatility by
``crisis_vol_mult`` inside a contiguous crisis window. Because only the
common factor scales up, pairwise correlations jump toward 1 in the
crisis — the classic correlation-breakdown stylised fact.

Everything is driven by an explicit seed; no network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["SyntheticPanel", "generate_panel"]


@dataclass(frozen=True)
class SyntheticPanel:
    """A simulated return panel with its true data-generating moments.

    Attributes
    ----------
    returns : pd.DataFrame
        (T, N) simple per-period (daily) returns, columns = asset names.
    true_mean : np.ndarray
        (N,) true per-period expected returns (calm regime).
    true_cov : np.ndarray
        (N, N) true per-period covariance in the calm regime.
    crisis_cov : np.ndarray
        (N, N) true per-period covariance in the crisis regime
        (equals ``true_cov`` when ``regimes=False``).
    market_weights : np.ndarray
        (N,) strictly positive cap-style weights summing to 1, used for
        reverse optimization / Black-Litterman demos.
    crisis_mask : np.ndarray
        (T,) boolean, True on crisis days (all False when regimes off).
    asset_names : list[str]
        Column labels of ``returns``.
    """

    returns: pd.DataFrame
    true_mean: np.ndarray
    true_cov: np.ndarray
    crisis_cov: np.ndarray
    market_weights: np.ndarray
    crisis_mask: np.ndarray
    asset_names: list = field(default_factory=list)


def generate_panel(
    n_assets: int = 8,
    n_periods: int = 2000,
    seed: int = 0,
    regimes: bool = False,
    crisis_start_frac: float = 0.70,
    crisis_len_frac: float = 0.12,
    crisis_vol_mult: float = 3.0,
    crisis_mkt_drift: float = -0.40,
    n_sectors: int = 3,
) -> SyntheticPanel:
    """Simulate a multi-asset daily return panel with known mean/covariance.

    Parameters
    ----------
    n_assets : int
        Number of assets N (>= 1).
    n_periods : int
        Number of daily periods T.
    seed : int
        Seed for ``numpy.random.default_rng``; the panel is fully
        deterministic given the seed.
    regimes : bool
        If True, insert a contiguous crisis window where the market-factor
        volatility is multiplied by ``crisis_vol_mult`` (correlations jump).
    crisis_start_frac, crisis_len_frac : float
        Crisis window position/length as fractions of T.
    crisis_vol_mult : float
        Market-factor vol multiplier inside the crisis (> 1).
    crisis_mkt_drift : float
        ANNUALISED drift added to the market factor on crisis days
        (default -40%/yr: crises are high-vol, high-correlation
        drawdowns, not just noisy melt-ups). ``true_mean``/``true_cov``
        always refer to the CALM regime.
    n_sectors : int
        Number of sector factors (capped at ``n_assets``).

    Returns
    -------
    SyntheticPanel
        Panel plus true moments; ``true_mean``/``true_cov`` are per-period
        (daily) calm-regime moments.
    """
    if n_assets < 1:
        raise ValueError(f"n_assets must be >= 1, got {n_assets}")
    if n_periods < 1:
        raise ValueError(f"n_periods must be >= 1, got {n_periods}")

    rng = np.random.default_rng(seed)
    n_sectors = min(n_sectors, n_assets)

    # --- parameters (annualised, then scaled to daily) -------------------
    ppy = 252.0
    # Annual expected excess returns 3%..8% — deliberately similar across
    # assets so the cross-sectional dispersion of TRUE means is small
    # relative to estimation noise (the Merton problem is visible).
    ann_mu = np.linspace(0.03, 0.08, n_assets)
    ann_mu = rng.permutation(ann_mu)
    mu = ann_mu / ppy

    beta = rng.uniform(0.6, 1.4, size=n_assets)  # market betas
    sigma_mkt = 0.16 / np.sqrt(ppy)  # 16% annual market factor vol

    # sector loadings: each asset belongs to one sector
    sector_of = np.arange(n_assets) % n_sectors
    gamma_load = rng.uniform(0.4, 0.8, size=n_assets)
    Gamma = np.zeros((n_assets, n_sectors))
    Gamma[np.arange(n_assets), sector_of] = gamma_load
    sigma_sec = np.full(n_sectors, 0.08 / np.sqrt(ppy))  # 8% annual

    idio_ann = rng.uniform(0.10, 0.22, size=n_assets)  # 10-22% annual idio
    d = (idio_ann / np.sqrt(ppy)) ** 2

    true_cov = (
        sigma_mkt**2 * np.outer(beta, beta)
        + Gamma @ np.diag(sigma_sec**2) @ Gamma.T
        + np.diag(d)
    )
    crisis_sigma_mkt = sigma_mkt * (crisis_vol_mult if regimes else 1.0)
    crisis_cov = (
        crisis_sigma_mkt**2 * np.outer(beta, beta)
        + Gamma @ np.diag(sigma_sec**2) @ Gamma.T
        + np.diag(d)
    )

    # --- crisis mask -----------------------------------------------------
    crisis_mask = np.zeros(n_periods, dtype=bool)
    if regimes:
        c0 = int(crisis_start_frac * n_periods)
        c1 = min(n_periods, c0 + max(1, int(crisis_len_frac * n_periods)))
        crisis_mask[c0:c1] = True

    # --- simulate --------------------------------------------------------
    f_mkt = rng.standard_normal(n_periods) * sigma_mkt
    f_mkt[crisis_mask] *= crisis_vol_mult
    f_mkt[crisis_mask] += crisis_mkt_drift / ppy
    f_sec = rng.standard_normal((n_periods, n_sectors)) * sigma_sec
    eps = rng.standard_normal((n_periods, n_assets)) * np.sqrt(d)

    r = mu[None, :] + np.outer(f_mkt, beta) + f_sec @ Gamma.T + eps

    names = [f"A{i+1}" for i in range(n_assets)]
    idx = pd.bdate_range("2015-01-01", periods=n_periods)
    returns = pd.DataFrame(r, index=idx, columns=names)

    # cap-style market weights: strictly positive, decaying, sum to 1
    raw_caps = rng.uniform(0.5, 2.0, size=n_assets) * np.linspace(
        2.0, 0.8, n_assets
    )
    market_weights = raw_caps / raw_caps.sum()

    return SyntheticPanel(
        returns=returns,
        true_mean=mu,
        true_cov=true_cov,
        crisis_cov=crisis_cov,
        market_weights=market_weights,
        crisis_mask=crisis_mask,
        asset_names=names,
    )
