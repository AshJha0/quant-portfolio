"""Multi-step variance forecasts and an out-of-sample forecasting harness.

Forecast conventions
--------------------
* ``forecast_*`` functions return **daily conditional variances** for horizons
  k = 1..h, conditioning on the full fitted sample (information set T).
* GARCH / GJR forecasts use the analytic recursion
  E_T[sigma2_{T+k}] = omega_eff + P * E_T[sigma2_{T+k-1}] with persistence P
  (alpha + beta, resp. alpha + gamma/2 + beta), which converges monotonically
  to the unconditional variance omega/(1-P).
* EGARCH forecasts are computed by **Monte Carlo simulation with a seeded
  Generator**. Why no closed form is used: the recursion is linear in
  ln sigma2, but the object of interest is E_T[sigma2_{T+k}] =
  E_T[exp(ln sigma2_{T+k})], which involves products of terms
  E[exp(a|z| + g z)] over intermediate shocks. For Gaussian z this has a
  semi-analytic expression in normal CDFs (Nelson 1991), but it does not
  extend to other innovation distributions and is numerically delicate for
  persistent beta; the simulation estimator is unbiased, general, and exactly
  reproducible via the explicit seed (package policy: every stochastic
  component is seeded).
* EWMA forecasts are flat at sigma2_{T+1} (IGARCH with zero intercept — no
  level to mean-revert to; see :mod:`eq_vol.ewma`).
* Historical forecasts are flat at the trailing ``window`` mean of squared
  returns (the implicit "model" of a rolling realized-vol desk estimate).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from ._results import VolatilityFitResult
from ._utils import TRADING_DAYS, as_generator, validate_returns
from .egarch import egarch_recursion, expected_abs_z, fit_egarch
from .ewma import ewma_forecast
from .garch import fit_garch, garch_recursion
from .gjr import fit_gjr, gjr_persistence, gjr_recursion
from ._utils import initial_variance

__all__ = [
    "forecast_garch",
    "forecast_gjr",
    "forecast_egarch",
    "forecast_historical",
    "forecast",
    "term_structure",
    "rolling_one_step_forecasts",
    "RollingForecastResult",
]


def _check_horizon(horizon: int) -> None:
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")


def _persistence_recursion(v1: float, omega: float, p: float, horizon: int) -> np.ndarray:
    """E_T[sigma2_{T+k}] for k=1..h via v_k = omega + p v_{k-1}.

    Monotone convergence to omega/(1-p) for p < 1; linear drift for p = 1
    (IGARCH); explosive for p > 1 — the recursion is exact in all cases.
    """
    out = np.empty(horizon)
    out[0] = v1
    for k in range(1, horizon):
        out[k] = omega + p * out[k - 1]
    return out


def forecast_garch(result: VolatilityFitResult, horizon: int = 10) -> np.ndarray:
    """Analytic multi-step GARCH(1,1) variance forecast.

    sigma2_{T+1} = omega + alpha r_T^2 + beta sigma2_T (exact, equals the
    in-sample recursion advanced one step), then
    E_T[sigma2_{T+k}] = sigma_bar^2 + (alpha+beta)^{k-1} (sigma2_{T+1} - sigma_bar^2).
    """
    _check_horizon(horizon)
    p = result.params
    v1 = p["omega"] + p["alpha"] * result.returns[-1] ** 2 + p["beta"] * result.sigma2[-1]
    return _persistence_recursion(v1, p["omega"], p["alpha"] + p["beta"], horizon)


def forecast_gjr(result: VolatilityFitResult, horizon: int = 10) -> np.ndarray:
    """Analytic multi-step GJR-GARCH(1,1) variance forecast.

    One step uses the realised indicator on r_T; further steps replace the
    indicator by its expectation 1/2 (symmetric innovations), giving the same
    persistence recursion with P = alpha + gamma/2 + beta.
    """
    _check_horizon(horizon)
    p = result.params
    a_eff = p["alpha"] + (p["gamma"] if result.returns[-1] < 0 else 0.0)
    v1 = p["omega"] + a_eff * result.returns[-1] ** 2 + p["beta"] * result.sigma2[-1]
    pers = gjr_persistence(p["alpha"], p["gamma"], p["beta"])
    return _persistence_recursion(v1, p["omega"], pers, horizon)


def forecast_egarch(
    result: VolatilityFitResult,
    horizon: int = 10,
    n_sims: int = 20_000,
    seed: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Simulation-based multi-step EGARCH(1,1) variance forecast.

    The one-step forecast is deterministic (all inputs are known at T);
    steps k >= 2 average sigma2 over ``n_sims`` simulated shock paths drawn
    from the fitted innovation distribution using a seeded Generator (see
    module docstring for why no closed form is used).

    MC error of the k-step forecast is O(sd(sigma2_{T+k}) / sqrt(n_sims));
    with the default 20k paths it is well under 1% of the forecast level for
    typical daily-equity parameters.
    """
    _check_horizon(horizon)
    rng = as_generator(seed)
    p = result.params
    omega, alpha, gamma, beta = p["omega"], p["alpha"], p["gamma"], p["beta"]
    nu = p.get("nu", np.nan)
    eaz = expected_abs_z(result.dist, nu)

    z_T = result.returns[-1] / np.sqrt(result.sigma2[-1])
    log_v1 = omega + beta * np.log(result.sigma2[-1]) + alpha * (abs(z_T) - eaz) + gamma * z_T
    out = np.empty(horizon)
    out[0] = np.exp(log_v1)
    if horizon == 1:
        return out
    log_s2 = np.full(n_sims, log_v1)
    for k in range(1, horizon):
        if result.dist == "t":
            z = rng.standard_t(nu, n_sims) * np.sqrt((nu - 2.0) / nu)
        else:
            z = rng.standard_normal(n_sims)
        log_s2 = omega + beta * log_s2 + alpha * (np.abs(z) - eaz) + gamma * z
        out[k] = float(np.mean(np.exp(log_s2)))
    return out


def forecast_historical(
    returns: np.ndarray, horizon: int = 10, window: int = 21
) -> np.ndarray:
    """Flat forecast at the trailing ``window``-day mean of squared returns.

    This is what an unconditional rolling-window estimate implies: no
    dynamics, so every horizon gets the same number.
    """
    _check_horizon(horizon)
    r = validate_returns(returns, min_obs=window)
    return np.full(horizon, float(np.mean(r[-window:] ** 2)))


def forecast(
    result: VolatilityFitResult,
    horizon: int = 10,
    n_sims: int = 20_000,
    seed: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Dispatch multi-step forecast on the fitted model type."""
    if result.model == "GARCH":
        return forecast_garch(result, horizon)
    if result.model == "GJR-GARCH":
        return forecast_gjr(result, horizon)
    if result.model == "EGARCH":
        return forecast_egarch(result, horizon, n_sims=n_sims, seed=seed)
    raise ValueError(f"unknown model {result.model!r}")


def term_structure(
    result: VolatilityFitResult,
    horizon: int = 252,
    n_sims: int = 20_000,
    seed: int | np.random.Generator | None = 0,
) -> pd.DataFrame:
    """Term structure of the volatility forecast.

    Returns
    -------
    pandas.DataFrame indexed by horizon k with:
        forward_vol_annual — sqrt(252 * E_T[sigma2_{T+k}]) (per-period);
        avg_vol_annual — sqrt(252 * mean(E_T[sigma2_{T+1..k}])), the constant
        vol over [T, T+k] consistent with the cumulative forecast variance
        (the number an option desk compares to implied vol of expiry k).
    """
    v = forecast(result, horizon, n_sims=n_sims, seed=seed)
    cum_avg = np.cumsum(v) / np.arange(1, horizon + 1)
    return pd.DataFrame(
        {
            "forward_vol_annual": np.sqrt(v * TRADING_DAYS),
            "avg_vol_annual": np.sqrt(cum_avg * TRADING_DAYS),
        },
        index=pd.RangeIndex(1, horizon + 1, name="horizon_days"),
    )


# ---------------------------------------------------------------------------
# rolling out-of-sample harness
# ---------------------------------------------------------------------------

class RollingForecastResult(NamedTuple):
    """Output of :func:`rolling_one_step_forecasts`.

    forecasts : daily variance forecasts, one per out-of-sample date
    test_index : positions t in the original series being forecast
    test_returns : realised returns at those positions
    n_refits : number of successful re-estimations
    n_failed_refits : refits that failed to converge (previous parameters
        were reused — surfaced here, never silent)
    """

    forecasts: np.ndarray
    test_index: np.ndarray
    test_returns: np.ndarray
    n_refits: int
    n_failed_refits: int


_FIT_FUNCS = {"garch": fit_garch, "egarch": fit_egarch, "gjr": fit_gjr}


def _one_step_variance(
    model: str,
    params: dict[str, float],
    hist: np.ndarray,
    lam: float,
    hist_window: int,
    init_method: str,
) -> float:
    """One-step-ahead variance forecast given fixed parameters and history."""
    if model == "historical":
        return float(np.mean(hist[-hist_window:] ** 2))
    if model == "ewma":
        return float(ewma_forecast(hist, horizon=1, lam=lam)[0])
    b = initial_variance(hist, init_method)
    if model == "garch":
        s2 = garch_recursion(hist, params["omega"], params["alpha"], params["beta"], b)
        return float(params["omega"] + params["alpha"] * hist[-1] ** 2 + params["beta"] * s2[-1])
    if model == "gjr":
        s2 = gjr_recursion(hist, params["omega"], params["alpha"], params["gamma"], params["beta"], b)
        a_eff = params["alpha"] + (params["gamma"] if hist[-1] < 0 else 0.0)
        return float(params["omega"] + a_eff * hist[-1] ** 2 + params["beta"] * s2[-1])
    if model == "egarch":
        eaz = expected_abs_z("normal")
        s2 = egarch_recursion(
            hist, params["omega"], params["alpha"], params["gamma"], params["beta"], b, eaz
        )
        z = hist[-1] / np.sqrt(s2[-1])
        log_v = (
            params["omega"]
            + params["beta"] * np.log(s2[-1])
            + params["alpha"] * (abs(z) - eaz)
            + params["gamma"] * z
        )
        return float(np.exp(log_v))
    raise ValueError(
        f"unknown model {model!r}; use 'historical', 'ewma', 'garch', 'egarch' or 'gjr'"
    )


def rolling_one_step_forecasts(
    returns: np.ndarray,
    model: str,
    min_train: int = 1000,
    scheme: str = "expanding",
    window: int = 1000,
    refit_every: int = 25,
    lam: float = 0.94,
    hist_window: int = 21,
    init_method: str = "backcast",
) -> RollingForecastResult:
    """Rolling out-of-sample 1-step-ahead variance forecasts.

    At each out-of-sample date t (t = min_train..n-1) the model forecasts
    sigma2_t using only returns[0..t-1] (expanding scheme) or the last
    ``window`` returns (rolling scheme). GARCH-family parameters are
    re-estimated every ``refit_every`` days (desk-realistic cadence: daily
    refits are unnecessary — parameters move slowly — while the conditional
    variance recursion *is* updated every day with the fixed parameters).

    Parameters
    ----------
    model : {"historical", "ewma", "garch", "egarch", "gjr"}
    scheme : {"expanding", "rolling"}
        Expanding uses all history; rolling truncates to the last ``window``
        observations (adapts faster after structural breaks, at an
        efficiency cost).
    refit_every : int
        Re-estimation frequency in days (ignored for historical/ewma, which
        have no estimated parameters).

    Raises
    ------
    ValueError
        On unknown model/scheme or if ``min_train`` leaves no test set.
    ConvergenceError
        If the *first* fit fails; later refit failures fall back to the last
        good parameters and are counted in ``n_failed_refits``.
    """
    r = validate_returns(returns, min_obs=min_train + 1)
    if scheme not in ("expanding", "rolling"):
        raise ValueError(f"unknown scheme {scheme!r}; use 'expanding' or 'rolling'")
    if model not in ("historical", "ewma", "garch", "egarch", "gjr"):
        raise ValueError(
            f"unknown model {model!r}; use 'historical', 'ewma', 'garch', "
            f"'egarch' or 'gjr'"
        )
    if refit_every < 1:
        raise ValueError(f"refit_every must be >= 1, got {refit_every}")
    n = r.size
    test_index = np.arange(min_train, n)
    forecasts = np.empty(test_index.size)
    params: dict[str, float] = {}
    n_refits = 0
    n_failed = 0
    needs_fit = model in _FIT_FUNCS
    for i, t in enumerate(test_index):
        lo = max(0, t - window) if scheme == "rolling" else 0
        hist = r[lo:t]
        if needs_fit and i % refit_every == 0:
            res = _FIT_FUNCS[model](hist, raise_on_failure=(i == 0))
            if res.converged:
                params = res.params
                n_refits += 1
            else:
                n_failed += 1  # keep previous params — surfaced in the result
        forecasts[i] = _one_step_variance(model, params, hist, lam, hist_window, init_method)
    return RollingForecastResult(forecasts, test_index, r[test_index], n_refits, n_failed)
