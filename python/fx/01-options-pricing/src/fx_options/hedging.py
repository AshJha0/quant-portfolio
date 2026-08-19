"""Delta-hedging simulator for FX options with domestic-currency accounting.

A short-option delta hedge in FX differs from equities in one crucial way:
the hedge asset is *foreign cash*, which earns the foreign rate.  The
simulator therefore keeps two accounts:

* a **domestic cash account** accruing at ``r_d``;
* a **foreign currency position** of ``a`` units of base currency whose
  size grows at ``r_f`` (foreign interest reinvested into base ccy).

Between rebalances: ``cash *= e^{r_d dt}``, ``a *= e^{r_f dt}``.  At each
rebalance the position is traded to the model delta, paying transaction
costs quoted in **pips** (half-spread per unit of base currency traded:
cost = |units| * pips * pip_size, in domestic ccy).

Simulated spot paths follow GBM at the *true* vol with drift ``mu``
(default: the domestic risk-neutral drift ``r_d - r_f``, under which the
discounted hedged P&L has zero mean when hedging at the true vol).
Hedging can be run at a *wrong* vol (``sigma_hedge != sigma_true``) to
reproduce the classic Carr/El Karoui result: the P&L acquires a bias with
sign ``(sigma_hedge - sigma_true)`` times the path's gamma exposure, and
its dispersion no longer vanishes with rebalancing frequency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ._common import validate_inputs, validate_option_type
from .garman_kohlhagen import gk_price
from .greeks import analytic_greeks

__all__ = ["HedgeResult", "simulate_delta_hedge", "hedge_frequency_study"]


@dataclass(frozen=True)
class HedgeResult:
    """Outcome of a delta-hedge simulation (short one unit of the option).

    P&L is terminal domestic-currency P&L at expiry per unit foreign
    notional (premium and all financing included).
    """

    mean_pnl: float
    std_pnl: float
    option_premium: float
    n_rebalances: int
    n_paths: int
    total_transaction_costs: float
    pnl: np.ndarray = field(repr=False)


def simulate_delta_hedge(
    S0: float, K: float, T: float, r_d: float, r_f: float,
    sigma_true: float, option_type: str, *,
    sigma_hedge: float | None = None,
    n_rebalances: int = 50, n_paths: int = 2000,
    rng: np.random.Generator | int | None = 0,
    mu: float | None = None,
    transaction_cost_pips: float = 0.0, pip_size: float = 1e-4,
) -> HedgeResult:
    """Simulate a discretely rebalanced delta hedge of a short FX option.

    Parameters
    ----------
    S0, K, T, r_d, r_f : float
        Market inputs as in :func:`fx_options.garman_kohlhagen.gk_price`;
        requires T > 0.
    sigma_true : float
        Volatility realised by the simulated paths, > 0.
    option_type : str
        ``"call"`` or ``"put"``.
    sigma_hedge : float, optional
        Vol used for pricing the premium and computing hedge deltas.
        Defaults to ``sigma_true`` (hedging at the right vol).
    n_rebalances : int
        Number of hedge intervals (rebalance at each of the N-1 interior
        dates; initial hedge at t=0, unwind at expiry).
    n_paths : int
        Number of simulated paths.
    rng : numpy.random.Generator or int or None
        Explicit generator or seed.
    mu : float, optional
        Real-world drift of the spot; defaults to ``r_d - r_f``.
    transaction_cost_pips : float
        Half-spread cost per unit of base ccy traded, in pips.
    pip_size : float
        1e-4 for most pairs, 1e-2 for JPY-quoted pairs.

    Returns
    -------
    HedgeResult
    """
    phi = validate_option_type(option_type)
    validate_inputs(S0, K, T, r_d, r_f, sigma_true)
    if T <= 0.0 or sigma_true <= 0.0:
        raise ValueError("simulate_delta_hedge requires T > 0, sigma_true > 0")
    if sigma_hedge is None:
        sigma_hedge = sigma_true
    if not math.isfinite(sigma_hedge) or sigma_hedge <= 0.0:
        raise ValueError(
            f"sigma_hedge must be positive and finite, got {sigma_hedge!r}"
        )
    if mu is not None and not math.isfinite(mu):
        raise ValueError(f"mu must be finite, got {mu!r}")
    if not isinstance(n_rebalances, int) or n_rebalances < 1:
        raise ValueError(f"n_rebalances must be a positive int, got {n_rebalances!r}")
    if not isinstance(n_paths, int) or n_paths < 2:
        raise ValueError(f"n_paths must be an int >= 2, got {n_paths!r}")
    if not math.isfinite(transaction_cost_pips) or transaction_cost_pips < 0.0:
        raise ValueError(
            "transaction_cost_pips must be >= 0 and finite, got "
            f"{transaction_cost_pips!r}"
        )
    if not math.isfinite(pip_size) or pip_size <= 0.0:
        raise ValueError(
            f"pip_size must be > 0 and finite, got {pip_size!r}"
        )

    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    drift = (r_d - r_f) if mu is None else mu
    dt = T / n_rebalances
    grow_d = math.exp(r_d * dt)
    grow_f = math.exp(r_f * dt)
    cost_per_unit = transaction_cost_pips * pip_size

    # Simulate paths (n_paths, n_rebalances+1).
    z = gen.standard_normal((n_paths, n_rebalances))
    log_steps = (drift - 0.5 * sigma_true**2) * dt + sigma_true * math.sqrt(dt) * z
    log_paths = np.cumsum(log_steps, axis=1)
    spots = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))

    from scipy.stats import norm as _norm

    def hedge_delta(s: np.ndarray, t_remaining: float) -> np.ndarray:
        # Vectorised GK spot delta: phi e^{-r_f tau} N(phi d1).
        v = sigma_hedge * math.sqrt(t_remaining)
        d1 = (np.log(s / K)
              + (r_d - r_f + 0.5 * sigma_hedge**2) * t_remaining) / v
        return phi * math.exp(-r_f * t_remaining) * _norm.cdf(phi * d1)

    premium = gk_price(S0, K, T, r_d, r_f, sigma_hedge, option_type)
    delta0 = analytic_greeks(S0, K, T, r_d, r_f, sigma_hedge,
                             option_type).delta_spot
    cash = np.full(n_paths, premium - delta0 * S0 - abs(delta0) * cost_per_unit)
    a = np.full(n_paths, delta0)
    total_costs = np.full(n_paths, abs(delta0) * cost_per_unit)

    for step in range(1, n_rebalances):
        cash *= grow_d
        a *= grow_f
        t_remaining = T - step * dt
        s_now = spots[:, step]
        target = hedge_delta(s_now, t_remaining)
        trade = target - a
        cost = np.abs(trade) * cost_per_unit
        cash -= trade * s_now + cost
        total_costs += cost
        a = target

    # Final interval: accrue, unwind hedge, settle option.
    cash *= grow_d
    a *= grow_f
    s_final = spots[:, -1]
    cost = np.abs(a) * cost_per_unit
    cash += a * s_final - cost
    total_costs += cost
    payoff = np.maximum(phi * (s_final - K), 0.0)
    pnl = cash - payoff

    return HedgeResult(
        mean_pnl=float(pnl.mean()), std_pnl=float(pnl.std(ddof=1)),
        option_premium=premium, n_rebalances=n_rebalances, n_paths=n_paths,
        total_transaction_costs=float(total_costs.mean()), pnl=pnl,
    )


def hedge_frequency_study(
    S0: float, K: float, T: float, r_d: float, r_f: float,
    sigma_true: float, option_type: str, *,
    frequencies: tuple[int, ...] = (4, 12, 50, 100, 250),
    n_paths: int = 2000, rng: np.random.Generator | int | None = 0,
    sigma_hedge: float | None = None,
    transaction_cost_pips: float = 0.0, pip_size: float = 1e-4,
) -> list[dict[str, float]]:
    """P&L statistics versus rebalancing frequency.

    Returns one row per frequency: ``{"n_rebalances", "mean_pnl",
    "std_pnl", "mean_costs"}``.  With zero costs and correct vol the
    std shrinks like 1/sqrt(N); with costs there is a cost/variance
    trade-off (costs grow like sqrt(N)).
    """
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    rows = []
    for n in frequencies:
        res = simulate_delta_hedge(
            S0, K, T, r_d, r_f, sigma_true, option_type,
            sigma_hedge=sigma_hedge, n_rebalances=n, n_paths=n_paths,
            rng=gen, transaction_cost_pips=transaction_cost_pips,
            pip_size=pip_size)
        rows.append({"n_rebalances": float(n), "mean_pnl": res.mean_pnl,
                     "std_pnl": res.std_pnl,
                     "mean_costs": res.total_transaction_costs})
    return rows
