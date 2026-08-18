"""Seeded synthetic data generators for the regime-switching project.

Two generators:

* :func:`make_regime_panel` — a multi-asset daily-return panel driven by a
  hidden Markov chain with a KNOWN transition matrix and per-state
  mean / vol / correlation.  The bear state has negative drift, high vol and
  high cross-asset correlation (the classic "correlations go to one" crisis
  stylised fact).  Used for parameter-recovery and detection tests.
* :func:`make_gbm_panel` — a single-regime GBM null (constant drift/vol/corr).
  Used for false-positive tests: on this data a regime model should NOT
  systematically beat buy-and-hold, and BIC should prefer fewer states.

Conventions: returns are daily log-returns; vol parameters are quoted
annualised (ACT/252) and converted internally with ``sigma_d = sigma / sqrt(252)``.
All randomness flows through an explicit ``seed`` / ``numpy.random.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252

__all__ = [
    "RegimePanel",
    "default_regime_params",
    "make_regime_panel",
    "make_gbm_panel",
    "simulate_markov_chain",
]


@dataclass(frozen=True)
class RegimePanel:
    """A simulated regime-switching panel with its ground truth.

    Attributes
    ----------
    prices : pd.DataFrame
        (T x N) price levels, starting at 100.0, business-day index.
    returns : pd.DataFrame
        (T x N) daily log-returns (first row is day 1; ``prices`` has T+1 rows).
    states : np.ndarray
        (T,) integer hidden state per return observation (ground truth).
    transition : np.ndarray
        (K x K) true transition matrix, rows sum to 1.
    mu : np.ndarray
        (K,) true annualised drift per state.
    sigma : np.ndarray
        (K,) true annualised vol per state.
    corr : np.ndarray
        (K,) true pairwise correlation per state (equicorrelation).
    """

    prices: pd.DataFrame
    returns: pd.DataFrame
    states: np.ndarray
    transition: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    corr: np.ndarray

    @property
    def n_states(self) -> int:
        return len(self.mu)


def default_regime_params(n_states: int) -> dict[str, np.ndarray]:
    """Return the canonical 2- or 3-state parameter set.

    State ordering (by construction): 0 = low-vol bull, then increasingly
    stressed states; the last state is the high-vol, high-correlation bear
    with negative drift.

    Parameters
    ----------
    n_states : int
        2 or 3.

    Returns
    -------
    dict
        Keys ``transition`` (K x K), ``mu`` (K, annualised), ``sigma``
        (K, annualised), ``corr`` (K, equicorrelation in [0, 1)).
    """
    if n_states == 2:
        return {
            "transition": np.array([[0.99, 0.01], [0.03, 0.97]]),
            "mu": np.array([0.12, -0.25]),
            "sigma": np.array([0.10, 0.35]),
            "corr": np.array([0.25, 0.75]),
        }
    if n_states == 3:
        return {
            "transition": np.array(
                [
                    [0.985, 0.013, 0.002],
                    [0.030, 0.940, 0.030],
                    [0.005, 0.045, 0.950],
                ]
            ),
            "mu": np.array([0.15, 0.02, -0.30]),
            "sigma": np.array([0.10, 0.18, 0.38]),
            "corr": np.array([0.20, 0.45, 0.80]),
        }
    raise ValueError(f"default_regime_params supports n_states in {{2, 3}}, got {n_states}")


def simulate_markov_chain(
    transition: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
    initial_state: int = 0,
) -> np.ndarray:
    """Simulate a discrete Markov chain path.

    Parameters
    ----------
    transition : np.ndarray
        (K x K) row-stochastic transition matrix.
    n_steps : int
        Number of steps to simulate (must be >= 1).
    rng : np.random.Generator
        Source of randomness.
    initial_state : int
        State at step 0.

    Returns
    -------
    np.ndarray
        (n_steps,) integer state path.
    """
    transition = np.asarray(transition, dtype=float)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be a square matrix")
    if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("transition matrix rows must sum to 1")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    k = transition.shape[0]
    if not 0 <= initial_state < k:
        raise ValueError(f"initial_state {initial_state} out of range for {k} states")

    cdf = np.cumsum(transition, axis=1)
    states = np.empty(n_steps, dtype=int)
    states[0] = initial_state
    u = rng.random(n_steps)
    for t in range(1, n_steps):
        states[t] = int(np.searchsorted(cdf[states[t - 1]], u[t]))
    return states


def _equicorr_chol(n_assets: int, rho: float) -> np.ndarray:
    """Cholesky factor of an N x N equicorrelation matrix."""
    corr = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(corr, 1.0)
    return np.linalg.cholesky(corr)


def make_regime_panel(
    n_states: int = 3,
    n_assets: int = 8,
    n_days: int = 2520,
    seed: int = 7,
    params: dict[str, np.ndarray] | None = None,
    start: str = "2015-01-01",
) -> RegimePanel:
    """Generate a multi-asset regime-switching panel with known ground truth.

    Each day the hidden Markov chain picks a state; asset log-returns are
    jointly Gaussian with per-state annualised drift ``mu[k]``, vol
    ``sigma[k]`` and equicorrelation ``corr[k]``.  Small fixed cross-asset
    dispersion in drift/vol makes assets distinguishable without changing
    the regime structure.

    Parameters
    ----------
    n_states : int
        Number of hidden states (2 or 3 for the default parameter set).
    n_assets : int
        Number of assets in the panel.
    n_days : int
        Number of return observations.
    seed : int
        Seed for the ``numpy.random.default_rng`` generator.
    params : dict, optional
        Override of :func:`default_regime_params` (same keys/shapes).
    start : str
        First business date of the price index.

    Returns
    -------
    RegimePanel
        Prices, returns, hidden state path and true parameters.
    """
    if n_assets < 2:
        raise ValueError(f"n_assets must be >= 2, got {n_assets}")
    if n_days < 10:
        raise ValueError(f"n_days must be >= 10, got {n_days}")
    p = params if params is not None else default_regime_params(n_states)
    transition = np.asarray(p["transition"], dtype=float)
    mu = np.asarray(p["mu"], dtype=float)
    sigma = np.asarray(p["sigma"], dtype=float)
    corr = np.asarray(p["corr"], dtype=float)
    if not (len(mu) == len(sigma) == len(corr) == transition.shape[0] == n_states):
        raise ValueError("parameter shapes inconsistent with n_states")

    rng = np.random.default_rng(seed)
    states = simulate_markov_chain(transition, n_days, rng, initial_state=0)

    # Fixed per-asset tilts (deterministic in seed): mild dispersion.
    beta = 1.0 + 0.10 * rng.standard_normal(n_assets)  # vol tilt
    alpha = 0.02 * rng.standard_normal(n_assets)       # drift tilt (annualised)

    chols = [_equicorr_chol(n_assets, float(r)) for r in corr]
    z = rng.standard_normal((n_days, n_assets))

    rets = np.empty((n_days, n_assets))
    for k in range(n_states):
        mask = states == k
        if not mask.any():
            continue
        eps = z[mask] @ chols[k].T
        sig_d = sigma[k] * beta / np.sqrt(TRADING_DAYS)
        mu_d = (mu[k] + alpha) / TRADING_DAYS - 0.5 * sig_d**2
        rets[mask] = mu_d + sig_d * eps

    dates = pd.bdate_range(start, periods=n_days + 1)
    returns = pd.DataFrame(
        rets, index=dates[1:], columns=[f"A{i:02d}" for i in range(n_assets)]
    )
    prices = 100.0 * np.exp(returns.cumsum())
    prices = pd.concat(
        [pd.DataFrame(100.0, index=dates[:1], columns=returns.columns), prices]
    )
    return RegimePanel(
        prices=prices,
        returns=returns,
        states=states,
        transition=transition,
        mu=mu,
        sigma=sigma,
        corr=corr,
    )


def make_gbm_panel(
    n_assets: int = 8,
    n_days: int = 2520,
    seed: int = 11,
    mu: float = 0.07,
    sigma: float = 0.16,
    rho: float = 0.35,
    start: str = "2015-01-01",
) -> RegimePanel:
    """Generate a single-regime (no-regime) correlated GBM null panel.

    Constant drift, vol and correlation — there is nothing for a regime
    model to find.  Returned as a :class:`RegimePanel` with a single state
    so it can flow through the same pipeline; ``states`` is all zeros.

    Parameters
    ----------
    n_assets, n_days, seed, start
        As in :func:`make_regime_panel`.
    mu, sigma : float
        Annualised drift and vol (same for every day).
    rho : float
        Constant pairwise correlation.

    Returns
    -------
    RegimePanel
    """
    params = {
        "transition": np.array([[1.0]]),
        "mu": np.array([mu]),
        "sigma": np.array([sigma]),
        "corr": np.array([rho]),
    }
    return make_regime_panel(
        n_states=1, n_assets=n_assets, n_days=n_days, seed=seed, params=params, start=start
    )
