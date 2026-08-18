"""Multi-step variance forecasting and a rolling out-of-sample harness.

* GARCH / GARCH-X / GJR: analytic multi-step forecasts. With persistence
  ``p < 1`` the h-step variance mean-reverts geometrically to the
  unconditional level:

      E[sigma2_{T+h}] = sigma2_bar + p^{h-1} * (sigma2_{T+1} - sigma2_bar)

  (GJR uses ``p = alpha + gamma/2 + beta``, valid for symmetric innovations
  -- both Gaussian and standardized-t are symmetric, so ``P(z<0) = 1/2``.)
* EGARCH: no closed form for ``E[sigma2_{T+h}]`` (the expectation of the
  exponential of the recursion involves moment-generating functions of |z|),
  so the forecaster is simulation-based with an explicit seeded Generator.
* EWMA: flat forecast (persistence exactly 1) -- :func:`fx_vol.ewma.ewma_forecast`.

The rolling harness produces genuine one-step-ahead out-of-sample variance
forecasts with periodic re-fitting, for the model races in
docs/VALIDATION.md and the Diebold-Mariano tests in :mod:`fx_vol.evaluation`.
"""

from __future__ import annotations

from math import exp, log, sqrt
from typing import Sequence

import numpy as np

from ._mle import FitResult, student_t_abs_moment
from .egarch import GAUSSIAN_ABS_MOMENT, egarch_filter
from .ewma import ewma_variance
from .garch import fit_garch, garch_filter
from .gjr import fit_gjr, gjr_filter
from .egarch import fit_egarch

__all__ = [
    "forecast_variance",
    "forecast_egarch_simulated",
    "rolling_one_step",
]


def _garch_family_forecast(result: FitResult, horizon: int, x_future: np.ndarray | None) -> np.ndarray:
    p = result.params
    omega, alpha, beta = p["omega"], p["alpha"], p["beta"]
    gamma = p.get("gamma", 0.0)  # GJR asymmetry; 0 for plain GARCH
    r_T = result.returns[-1]
    s2_T = result.sigma2[-1]
    persistence = result.persistence

    gammas_x = np.array([v for k, v in p.items() if k.startswith("gamma_x")])
    if gammas_x.size and x_future is None:
        x_future = np.zeros((horizon, gammas_x.size))  # no scheduled events assumed
    if x_future is not None:
        x_future = np.atleast_2d(np.asarray(x_future, dtype=float))
        if x_future.shape[0] == 1 and horizon > 1 and x_future.shape[1] == horizon:
            x_future = x_future.T
        if x_future.shape != (horizon, max(gammas_x.size, 1)):
            raise ValueError(
                f"x_future must have shape ({horizon}, {max(gammas_x.size, 1)}), got {x_future.shape}"
            )
        if not gammas_x.size:
            raise ValueError("x_future supplied but the fitted model has no exogenous terms")

    out = np.empty(horizon)
    ind = 1.0 if r_T < 0 else 0.0
    s2 = omega + (alpha + gamma * ind) * r_T ** 2 + beta * s2_T
    if gammas_x.size:
        s2 += float(x_future[0] @ gammas_x)
    out[0] = s2
    for h in range(1, horizon):
        s2 = omega + persistence * out[h - 1]
        if gammas_x.size:
            s2 += float(x_future[h] @ gammas_x)
        out[h] = s2
    return out


def forecast_egarch_simulated(
    result: FitResult,
    horizon: int,
    n_paths: int = 5000,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Simulation-based EGARCH variance forecast ``E[sigma2_{T+h}]``, h=1..H.

    h=1 is deterministic (z_T observed); h>=2 averages the recursion over
    ``n_paths`` simulated innovation paths (Gaussian or standardized-t
    matching the fitted distribution).

    Parameters
    ----------
    result : FitResult
        A fitted EGARCH model (``fit_egarch``).
    horizon : int
        Number of steps ahead.
    n_paths : int
        Monte Carlo paths; standard error of the h-step forecast scales as
        ``1/sqrt(n_paths)``.
    rng : numpy Generator, int seed, or None
        Explicit randomness control (None -> fresh entropy).
    """
    if result.model != "egarch":
        raise ValueError(f"expected an EGARCH FitResult, got model={result.model!r}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    p = result.params
    omega, alpha, gamma, beta = p["omega"], p["alpha"], p["gamma"], p["beta"]
    am = result.extra.get("abs_moment", GAUSSIAN_ABS_MOMENT)
    nu = p.get("nu")

    z_T = result.returns[-1] / sqrt(result.sigma2[-1])
    ls2_next = omega + beta * log(result.sigma2[-1]) + alpha * (abs(z_T) - am) + gamma * z_T
    out = np.empty(horizon)
    out[0] = exp(ls2_next)
    if horizon == 1:
        return out

    ls2 = np.full(n_paths, ls2_next)
    for h in range(1, horizon):
        if nu is None:
            z = rng.standard_normal(n_paths)
        else:
            z = rng.standard_t(nu, n_paths) * sqrt((nu - 2.0) / nu)  # unit variance
        ls2 = omega + beta * ls2 + alpha * (np.abs(z) - am) + gamma * z
        np.clip(ls2, -60.0, 60.0, out=ls2)
        out[h] = float(np.mean(np.exp(ls2)))
    return out


def forecast_variance(
    result: FitResult,
    horizon: int,
    x_future: Sequence[float] | np.ndarray | None = None,
    n_paths: int = 5000,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Forecast conditional variance h = 1..horizon steps ahead.

    Dispatches on the fitted model: analytic recursion for GARCH/GARCH-X/GJR,
    simulation for EGARCH. Variance units match the fitted returns (daily
    decimal returns in -> daily decimal variance out; annualize vol with
    ``sqrt(252 * var)``).

    Parameters
    ----------
    result : FitResult
        Output of ``fit_garch`` / ``fit_gjr`` / ``fit_egarch``.
    horizon : int
        Steps ahead (>= 1).
    x_future : array-like, optional
        Future exogenous regressors, shape (horizon, k) -- e.g. the known
        FOMC/ECB calendar over the forecast window (GARCH-X only). Defaults
        to zeros (no scheduled events).
    n_paths, rng
        EGARCH simulation controls (ignored for analytic models).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if result.model in ("garch", "garch-x", "gjr"):
        return _garch_family_forecast(result, horizon, x_future)
    if result.model == "egarch":
        if x_future is not None:
            raise ValueError("x_future is only supported for GARCH-X")
        return forecast_egarch_simulated(result, horizon, n_paths=n_paths, rng=rng)
    raise ValueError(f"unknown model {result.model!r}")


def _filtered_sigma2(model: str, params: dict[str, float], seg: np.ndarray) -> np.ndarray:
    """One-step-ahead conditional variances over ``seg`` under fixed params."""
    if model in ("garch", "garch_t"):
        return garch_filter(seg, params["omega"], params["alpha"], params["beta"])
    if model in ("gjr", "gjr_t"):
        return gjr_filter(seg, params["omega"], params["alpha"], params["gamma"], params["beta"])
    if model in ("egarch", "egarch_t"):
        am = params.get("_abs_moment", GAUSSIAN_ABS_MOMENT)
        return egarch_filter(seg, params["omega"], params["alpha"], params["gamma"], params["beta"], abs_moment=am)
    raise ValueError(f"unknown model {model!r}")


def rolling_one_step(
    returns: Sequence[float] | np.ndarray,
    model: str,
    window: int = 1000,
    refit_every: int = 100,
    lam: float = 0.94,
    n_oos: int | None = None,
) -> dict:
    """Rolling out-of-sample one-step-ahead variance forecasts.

    For each OOS day t (t = window .. n-1, optionally capped at the last
    ``n_oos`` days), the forecast of Var(r_t | F_{t-1}) uses parameters
    fitted on the rolling window of ``window`` returns ending at the most
    recent refit date (refit every ``refit_every`` days), with the variance
    state filtered forward daily under those fixed parameters -- the standard
    desk setup (daily state update, periodic re-estimation).

    Parameters
    ----------
    returns : array-like
        Log returns.
    model : str
        'ewma', 'garch', 'garch_t', 'gjr', 'gjr_t', 'egarch', 'egarch_t'.
    window : int
        Rolling estimation window length.
    refit_every : int
        Refit frequency in days (ignored for 'ewma', which has no fit).
    lam : float
        EWMA decay (model='ewma' only).
    n_oos : int, optional
        Keep only the last ``n_oos`` out-of-sample days.

    Returns
    -------
    dict
        ``forecast`` (variance forecasts), ``realized`` (r_t^2 proxy),
        ``returns_oos``, ``start_index``, ``model``, ``refits`` (number of
        re-estimations performed).
    """
    r = np.asarray(returns, dtype=float)
    if not np.isfinite(r).all():
        raise ValueError("returns contain NaN or infinite values")
    n = r.size
    if window < 100:
        raise ValueError(f"window must be >= 100, got {window}")
    if n <= window:
        raise ValueError(f"need more than window={window} returns, got {n}")
    oos_start = window if n_oos is None else max(window, n - n_oos)

    forecasts = np.empty(n - oos_start)
    refits = 0

    if model == "ewma":
        init = float(np.mean(r[:window] ** 2))
        sigma2 = ewma_variance(r, lam=lam, init=init)
        forecasts[:] = sigma2[oos_start:]
    else:
        fitters = {
            "garch": lambda seg: fit_garch(seg, dist="gaussian"),
            "garch_t": lambda seg: fit_garch(seg, dist="t"),
            "gjr": lambda seg: fit_gjr(seg, dist="gaussian"),
            "gjr_t": lambda seg: fit_gjr(seg, dist="t"),
            "egarch": lambda seg: fit_egarch(seg, dist="gaussian"),
            "egarch_t": lambda seg: fit_egarch(seg, dist="t"),
        }
        if model not in fitters:
            raise ValueError(f"unknown model {model!r}; choose from {sorted(fitters) + ['ewma']}")
        for t0 in range(oos_start, n, refit_every):
            t1 = min(t0 + refit_every, n)
            fit = fitters[model](r[t0 - window : t0])
            refits += 1
            params = dict(fit.params)
            if model.startswith("egarch"):
                params["_abs_moment"] = fit.extra["abs_moment"]
            # filter over [t0 - window, t1): position window + j is the
            # one-step-ahead variance of day t0 + j given data through t0+j-1
            seg = r[t0 - window : t1]
            sig = _filtered_sigma2(model, params, seg)
            forecasts[t0 - oos_start : t1 - oos_start] = sig[window : window + (t1 - t0)]

    return {
        "forecast": forecasts,
        "realized": r[oos_start:] ** 2,
        "returns_oos": r[oos_start:],
        "start_index": oos_start,
        "model": model,
        "refits": refits,
    }
