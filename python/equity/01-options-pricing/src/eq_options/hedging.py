"""Discrete delta-hedging simulator on GBM paths.

Setup: at t=0 we *sell* one European option at the Black-Scholes price
computed with hedge vol ``sigma_hedge``, then delta-hedge it by trading the
underlying at ``n_rebalance`` equally spaced times, financing at rate ``r``
and receiving the continuous dividend yield ``q`` on the stock held. At
expiry the option is cash-settled and the stock position unwound.

Key facts this module demonstrates (and the tests verify):

* Hedged at the true (realized) vol, the P&L distribution has mean ~ 0 and
  standard deviation shrinking like ``1/sqrt(n_rebalance)``.
* Hedged at a *wrong* vol, the expected P&L is the classic
  Carr/El Karoui result::

      E[P&L] ~ integral_0^T (sigma_hedge^2 - sigma_realized^2)/2
                             * S_t^2 * Gamma_hedge(S_t, t) dt

  so selling at implied above realized and delta-hedging earns the gamma-
  weighted vol spread — model risk made visible.
* Proportional transaction costs drag the mean P&L down and grow with
  rebalancing frequency.

Conventions: rates/yields continuously compounded, annualised (ACT/365F);
``T`` in years; vols annualised. All randomness via an explicit ``seed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .black_scholes import OptionType, bs_price, validate_inputs

__all__ = ["HedgeResult", "simulate_delta_hedge", "pnl_std_vs_frequency"]


@dataclass(frozen=True)
class HedgeResult:
    """Result of a discrete delta-hedging simulation (short one option).

    Attributes
    ----------
    pnl : numpy.ndarray
        Per-path terminal P&L in currency units (premium received, hedge
        traded, option cash-settled, stock unwound), *not* discounted.
    mean : float
        Sample mean of ``pnl``.
    std : float
        Sample standard deviation of ``pnl`` (ddof=1).
    mean_se : float
        Standard error of ``mean``.
    theory_pnl : float
        Monte Carlo estimate, on the same paths, of the misspecified-vol
        formula ``sum (sigma_h^2 - sigma_r^2)/2 * S^2 * Gamma_h * dt``;
        ~0 when hedging at the realized vol.
    n_paths, n_rebalance : int
        Simulation size.
    premium : float
        Option premium received at t=0 (priced at ``sigma_hedge``).
    """

    pnl: np.ndarray
    mean: float
    std: float
    mean_se: float
    theory_pnl: float
    n_paths: int
    n_rebalance: int
    premium: float


def _bs_delta_vec(
    s: np.ndarray, K: float, tau: np.ndarray | float, r: float, sigma: float,
    q: float, option_type: OptionType,
) -> np.ndarray:
    """Vectorised Black-Scholes delta for positive ``s`` and ``tau``."""
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(s / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    if option_type == "call":
        return np.exp(-q * tau) * norm.cdf(d1)
    return np.exp(-q * tau) * (norm.cdf(d1) - 1.0)


def _bs_gamma_vec(
    s: np.ndarray, K: float, tau: np.ndarray | float, r: float, sigma: float, q: float
) -> np.ndarray:
    """Vectorised Black-Scholes gamma for positive ``s`` and ``tau``."""
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(s / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    return np.exp(-q * tau) * norm.pdf(d1) / (s * sigma * sqrt_tau)


def simulate_delta_hedge(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_realized: float,
    sigma_hedge: float | None = None,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_rebalance: int = 52,
    n_paths: int = 2_000,
    mu: float | None = None,
    tc_rate: float = 0.0,
    seed: int | np.random.Generator | None = 7,
) -> HedgeResult:
    """Simulate discrete delta hedging of a short European option on GBM paths.

    Parameters
    ----------
    S0, K : float
        Initial spot and strike (currency units), strictly positive.
    T : float
        Time to expiry in years (ACT/365F), strictly positive.
    r : float
        Continuously compounded annualised financing rate.
    sigma_realized : float
        Annualised vol at which the underlying actually diffuses, > 0.
    sigma_hedge : float, optional
        Vol used both to price (premium received) and to compute hedge
        deltas. Defaults to ``sigma_realized`` (correctly specified hedge).
    q : float
        Continuous dividend yield received on the stock held.
    option_type : {"call", "put"}
        Option sold.
    n_rebalance : int
        Number of hedge rebalances over ``[0, T)`` (equally spaced), >= 1.
    n_paths : int
        Number of simulated paths, >= 2.
    mu : float, optional
        Real-world drift of the underlying. Defaults to ``r - q`` growth
        (risk-neutral) so the correctly hedged mean P&L is ~ 0 exactly.
    tc_rate : float
        Proportional transaction cost per unit of stock traded (e.g. 5e-4
        = 5bp of traded notional), charged on initial position, every
        rebalance and the final unwind.
    seed : int or numpy.random.Generator or None
        Explicit RNG seed for reproducibility.

    Returns
    -------
    HedgeResult
        P&L distribution, summary statistics, and the misspecified-vol
        theoretical P&L estimate on the same paths.

    Raises
    ------
    ValueError
        On non-positive ``S0``, ``K``, ``T``, ``sigma_realized`` or
        ``sigma_hedge``, negative ``tc_rate``, or too-small
        ``n_rebalance``/``n_paths``.
    """
    validate_inputs(S0, K, T, sigma_realized, option_type)
    sigma_h = sigma_realized if sigma_hedge is None else sigma_hedge
    if S0 <= 0 or K <= 0 or T <= 0 or sigma_realized <= 0 or sigma_h <= 0:
        raise ValueError("simulate_delta_hedge requires S0, K, T and vols > 0")
    if n_rebalance < 1:
        raise ValueError(f"n_rebalance must be >= 1, got {n_rebalance!r}")
    if n_paths < 2:
        raise ValueError(f"n_paths must be >= 2, got {n_paths!r}")
    if tc_rate < 0:
        raise ValueError(f"tc_rate must be >= 0, got {tc_rate!r}")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    drift = (r - q) if mu is None else mu
    n = n_rebalance
    dt = T / n
    growth = math.exp(r * dt)
    div_factor = math.exp(q * dt) - 1.0  # dividend yield collected per step

    # GBM paths under the chosen drift, exact scheme, shape (n_paths, n+1).
    z = rng.standard_normal((n_paths, n))
    log_steps = (drift - 0.5 * sigma_realized**2) * dt + sigma_realized * math.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)], axis=1
    )
    paths = S0 * np.exp(log_paths)

    premium = bs_price(S0, K, T, r, sigma_h, q, option_type)
    times = np.linspace(0.0, T, n + 1)

    delta = _bs_delta_vec(paths[:, 0], K, T, r, sigma_h, q, option_type)
    cash = premium - delta * paths[:, 0] - tc_rate * np.abs(delta) * paths[:, 0]
    theory_increments = np.zeros(n_paths)

    for i in range(1, n):
        s_i = paths[:, i]
        tau = T - times[i]
        # accrue financing and collect dividends on the stock held
        cash = cash * growth + delta * paths[:, i - 1] * div_factor
        new_delta = _bs_delta_vec(s_i, K, tau, r, sigma_h, q, option_type)
        trade = new_delta - delta
        cash -= trade * s_i + tc_rate * np.abs(trade) * s_i
        # accumulate the misspecified-vol theoretical P&L on the same grid
        gamma_h = _bs_gamma_vec(paths[:, i - 1], K, T - times[i - 1], r, sigma_h, q)
        theory_increments += (
            0.5 * (sigma_h**2 - sigma_realized**2)
            * paths[:, i - 1] ** 2 * gamma_h * dt
        )
        delta = new_delta

    s_T = paths[:, -1]
    cash = cash * growth + delta * paths[:, -2] * div_factor
    gamma_h = _bs_gamma_vec(paths[:, -2], K, T - times[-2], r, sigma_h, q)
    theory_increments += (
        0.5 * (sigma_h**2 - sigma_realized**2) * paths[:, -2] ** 2 * gamma_h * dt
    )
    sign = 1.0 if option_type == "call" else -1.0
    payoff = np.maximum(sign * (s_T - K), 0.0)
    pnl = cash + delta * s_T - tc_rate * np.abs(delta) * s_T - payoff

    mean = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1))
    return HedgeResult(
        pnl=pnl,
        mean=mean,
        std=std,
        mean_se=std / math.sqrt(n_paths),
        theory_pnl=float(np.mean(theory_increments)),
        n_paths=n_paths,
        n_rebalance=n,
        premium=premium,
    )


def pnl_std_vs_frequency(
    frequencies: list[int],
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_realized: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 2_000,
    seed: int = 7,
    tc_rate: float = 0.0,
) -> dict[int, float]:
    """Hedging P&L standard deviation as a function of rebalance frequency.

    Demonstrates the ``std ~ 1/sqrt(N)`` law of discrete delta hedging
    (each frequency uses an independent seeded stream).

    Parameters
    ----------
    frequencies : list of int
        Rebalance counts to test, each >= 1.
    S0, K, T, r, sigma_realized, q, option_type, n_paths, tc_rate
        As in :func:`simulate_delta_hedge` (hedged at the realized vol).
    seed : int
        Base seed; frequency ``i`` uses ``seed + i``-th spawned stream.

    Returns
    -------
    dict
        ``{n_rebalance: pnl_std}`` in currency units.
    """
    out: dict[int, float] = {}
    for i, n in enumerate(frequencies):
        res = simulate_delta_hedge(
            S0, K, T, r, sigma_realized,
            q=q, option_type=option_type, n_rebalance=n,
            n_paths=n_paths, seed=seed + 1000 * i, tc_rate=tc_rate,
        )
        out[n] = res.std
    return out
