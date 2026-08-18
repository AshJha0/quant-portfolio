"""Stress testing: historical replay, hypothetical shocks, sensitivity
ladders and reverse stress tests.

Historical scenarios are encoded as factor-shock vectors keyed by factor
*kind* (``equity`` / ``index`` / ``vol``): equity and index factors get a
return shock, vol factors an absolute implied-vol shock.  The numbers are
**approximations of published market moves** (exact levels differ by index
and source) and are documented as such in docs/DESK_GUIDE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .monte_carlo_var import safe_cholesky
from .portfolio import Portfolio

__all__ = [
    "StressScenario",
    "HISTORICAL_SCENARIOS",
    "scenario_shock_vector",
    "apply_scenario",
    "scenario_table",
    "sensitivity_ladder",
    "reverse_stress_delta",
    "reverse_stress_delta_gamma",
]


@dataclass(frozen=True)
class StressScenario:
    """A named factor-shock scenario.

    ``shocks_by_kind`` maps factor kind -> shock (return for price factors,
    absolute vol change for vol factors); ``shocks_by_name`` overrides
    individual factors by name.
    """

    name: str
    description: str
    shocks_by_kind: dict[str, float] = field(default_factory=dict)
    shocks_by_name: dict[str, float] = field(default_factory=dict)


#: Approximate factor moves in famous stress episodes.  Sources (approximate,
#: see docs/DESK_GUIDE.md): 1987-10-19 S&P 500 -20.5 % in one day, implied
#: vol regime jumped from ~20 % to well above 100 % intraday (we shock a
#: conservative +60 vol pts); Lehman fortnight (Sep-Oct 2008) S&P ~ -25 %
#: with VIX 25 -> 80 (+55 pts); COVID crash (late Feb-Mar 2020) S&P -34 %
#: peak-to-trough with VIX 15 -> 83 (+68 pts); single stocks assumed to move
#: with beta ~ 1.1 of the index in each episode.
HISTORICAL_SCENARIOS: dict[str, StressScenario] = {
    "1987_black_monday": StressScenario(
        name="1987_black_monday",
        description="19 Oct 1987: -20.5% index in one day, implied vol explosion",
        shocks_by_kind={"index": -0.205, "equity": -0.225, "vol": 0.60},
    ),
    "2008_lehman": StressScenario(
        name="2008_lehman",
        description="Sep-Oct 2008 Lehman fortnight: ~-25% equities, VIX 25->80",
        shocks_by_kind={"index": -0.25, "equity": -0.28, "vol": 0.55},
    ),
    "2020_covid": StressScenario(
        name="2020_covid",
        description="Feb-Mar 2020 COVID crash: -34% peak-to-trough, VIX 15->83",
        shocks_by_kind={"index": -0.34, "equity": -0.37, "vol": 0.68},
    ),
    "rate_equity_vol_combo": StressScenario(
        name="rate_equity_vol_combo",
        description="Hypothetical: -15% equities with +25 vol pts (risk-off combo)",
        shocks_by_kind={"index": -0.15, "equity": -0.17, "vol": 0.25},
    ),
    "melt_up": StressScenario(
        name="melt_up",
        description="Hypothetical: +10% equities, -5 vol pts (short-gamma pain trade)",
        shocks_by_kind={"index": 0.10, "equity": 0.11, "vol": -0.05},
    ),
}


def scenario_shock_vector(portfolio: Portfolio, scenario: StressScenario) -> np.ndarray:
    """Map a scenario onto the portfolio's factor vector (factor_names order)."""
    shocks = np.zeros(portfolio.n_factors)
    for j, name in enumerate(portfolio.factor_names):
        kind = portfolio.factors[name].kind
        val = scenario.shocks_by_kind.get(kind, 0.0)
        if name in scenario.shocks_by_name:
            val = scenario.shocks_by_name[name]
        shocks[j] = val
    return shocks


def apply_scenario(
    portfolio: Portfolio, scenario: StressScenario, method: str = "full"
) -> float:
    """Portfolio P&L (currency units, loss < 0) under a stress scenario."""
    shocks = scenario_shock_vector(portfolio, scenario)
    return float(portfolio.pnl(shocks, method=method)[0])


def scenario_table(
    portfolio: Portfolio, scenarios: dict[str, StressScenario] | None = None
) -> pd.DataFrame:
    """Stress P&L table across scenarios, full reval vs delta-gamma."""
    scenarios = HISTORICAL_SCENARIOS if scenarios is None else scenarios
    rows = []
    for sc in scenarios.values():
        full = apply_scenario(portfolio, sc, "full")
        dg = apply_scenario(portfolio, sc, "delta_gamma")
        rows.append(
            {
                "scenario": sc.name,
                "description": sc.description,
                "pnl_full": full,
                "pnl_delta_gamma": dg,
                "approx_error": dg - full,
            }
        )
    return pd.DataFrame(rows)


def sensitivity_ladder(
    portfolio: Portfolio,
    factor: str,
    shocks: np.ndarray | None = None,
    method: str = "full",
) -> pd.DataFrame:
    """One-factor P&L ladder: shock a single factor, hold the rest flat."""
    if factor not in portfolio.factors:
        raise ValueError(f"unknown factor {factor!r}; portfolio has {portfolio.factor_names}")
    if shocks is None:
        if portfolio.factors[factor].kind == "vol":
            shocks = np.array([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20])
        else:
            shocks = np.array([-0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20])
    shocks = np.asarray(shocks, dtype=float)
    j = portfolio.factor_names.index(factor)
    scen = np.zeros((shocks.size, portfolio.n_factors))
    scen[:, j] = shocks
    pnl = portfolio.pnl(scen, method=method)
    return pd.DataFrame({"shock": shocks, "pnl": pnl})


# --------------------------------------------------------------------------- #
# Reverse stress testing
# --------------------------------------------------------------------------- #
def reverse_stress_delta(
    exposures: np.ndarray, cov: np.ndarray, radius: float = 3.0
) -> dict[str, object]:
    """Closed-form worst-case direction for a *linear* (delta) portfolio.

    Maximise loss ``-w'x`` over shock vectors ``x`` at fixed Mahalanobis
    radius ``x' Sigma^{-1} x = r^2`` (a plausibility constraint: r = 3 means
    'a 3-sigma joint move').  Lagrange conditions give the closed form

    ``x* = -r * Sigma w / sqrt(w' Sigma w)``,   loss ``= r * sqrt(w' Sigma w)``

    — the worst plausible scenario is a move along the portfolio's own
    covariance-weighted exposure direction, and the worst loss is exactly
    ``r`` portfolio standard deviations.

    Returns dict with ``shock`` (factor-move vector), ``loss`` (> 0) and
    ``radius``.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    w = np.asarray(exposures, dtype=float).ravel()
    sig = np.atleast_2d(np.asarray(cov, dtype=float))
    sw = sig @ w
    sigma_p = float(np.sqrt(max(w @ sw, 0.0)))
    if sigma_p == 0.0:
        return {"shock": np.zeros_like(w), "loss": 0.0, "radius": radius}
    shock = -radius * sw / sigma_p
    return {"shock": shock, "loss": float(radius * sigma_p), "radius": radius}


def reverse_stress_delta_gamma(
    exposures: np.ndarray,
    gamma: np.ndarray,
    cov: np.ndarray,
    radius: float = 3.0,
    n_starts: int = 8,
    seed: int | np.random.Generator | None = 0,
) -> dict[str, object]:
    """Numerical worst-case direction for a delta-gamma portfolio.

    Minimise quadratic P&L ``w'x + 0.5 x'G x`` on the Mahalanobis sphere
    ``x' Sigma^{-1} x = r^2``.  Parameterise ``x = r * L u / |u|`` with
    ``L`` the Cholesky factor of Sigma, which turns the constrained problem
    into an unconstrained one on the unit sphere; multi-start BFGS guards
    against local minima (the objective on the sphere can have several).

    Setting ``gamma = 0`` recovers the closed-form delta solution
    (unit-tested against ``reverse_stress_delta``).
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    w = np.asarray(exposures, dtype=float).ravel()
    g = np.atleast_2d(np.asarray(gamma, dtype=float))
    sig = np.atleast_2d(np.asarray(cov, dtype=float))
    n = w.size
    chol = safe_cholesky(sig)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    def x_of(u: np.ndarray) -> np.ndarray:
        norm_u = np.linalg.norm(u)
        if norm_u == 0.0:
            norm_u = 1.0
        return radius * (chol @ u) / norm_u

    def objective(u: np.ndarray) -> float:
        x = x_of(u)
        return float(w @ x + 0.5 * x @ g @ x)  # P&L; minimise = worst loss

    starts = [rng.standard_normal(n) for _ in range(n_starts)]
    # include the delta closed-form direction as a warm start
    delta_dir = reverse_stress_delta(w, sig, radius)["shock"]
    if np.linalg.norm(delta_dir) > 0:
        starts.append(np.linalg.solve(chol, delta_dir))
    best_x, best_pnl = np.zeros(n), np.inf
    for u0 in starts:
        res = minimize(objective, u0, method="BFGS")
        if res.fun < best_pnl:
            best_pnl = float(res.fun)
            best_x = x_of(res.x)
    return {"shock": best_x, "loss": float(-best_pnl), "radius": radius}
