"""Monte Carlo pricing of European options under GBM (exact scheme).

The terminal stock price is simulated exactly:
``S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)``, ``Z ~ N(0,1)``,
so there is *no* time-discretisation bias: only statistical error, which
is reported as a standard error and 95% confidence interval.

Variance reduction
------------------
* Antithetic variates: pairs ``(Z, -Z)``; the standard error is computed on
  the pair averages (the correct estimator for correlated pairs).
* Control variate: the discounted terminal stock ``exp(-rT) S_T`` with
  known mean ``S exp(-qT)`` (martingale property); the optimal coefficient
  is estimated from the sample covariance.

Greeks
------
* Pathwise (differentiate the payoff): delta and vega; unbiased, low
  variance, valid because the vanilla payoff is Lipschitz.
* Likelihood-ratio (differentiate the density): delta and vega; unbiased
  but higher variance, works for discontinuous payoffs too.
* Central finite differences with common random numbers as a generic
  fallback.

All stochastic entry points take an explicit ``seed`` (int or
``numpy.random.Generator``). Same seed => bit-identical results.

Conventions: continuously compounded annualised ``r``, ``q`` (ACT/365F),
``T`` in years, ``sigma`` annualised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from qp_core.model import model

from .black_scholes import OptionType, _check_rates, bs_price, validate_inputs

__all__ = [
    "MCResult",
    "mc_delta_lr",
    "mc_delta_pathwise",
    "mc_greek_fd",
    "mc_price",
    "mc_vega_lr",
    "mc_vega_pathwise",
]

_Z95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimate with statistical error bars.

    Attributes
    ----------
    value : float
        Point estimate (price in currency units, or a Greek in its
        natural units).
    std_error : float
        Standard error of the estimator.
    ci_low, ci_high : float
        Two-sided 95% confidence interval ``value ± 1.96 * std_error``.
    n_paths : int
        Number of simulated paths (antithetic pairs count as 2 paths).
    """

    value: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int

    def contains(self, x: float) -> bool:
        """Return True if ``x`` lies inside the 95% confidence interval."""
        return self.ci_low <= x <= self.ci_high


def _make_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a ``numpy.random.Generator`` from a seed or pass one through."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _summary(samples: np.ndarray, n_paths: int) -> MCResult:
    """Build an :class:`MCResult` from i.i.d. per-draw samples."""
    n = samples.size
    value = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    if not (math.isfinite(value) and math.isfinite(se)):
        raise ValueError("Monte Carlo estimate is not finite (simulated prices overflow)")
    return MCResult(
        value=value,
        std_error=se,
        ci_low=value - _Z95 * se,
        ci_high=value + _Z95 * se,
        n_paths=n_paths,
    )


def _terminal(
    S: float, T: float, r: float, q: float, sigma: float, z: np.ndarray
) -> np.ndarray:
    """Exact GBM terminal prices for standard normal draws ``z``."""
    drift = (r - q - 0.5 * sigma * sigma) * T
    with np.errstate(over="ignore"):
        s_t = S * np.exp(drift + sigma * math.sqrt(T) * z)
    if not np.all(np.isfinite(s_t)):
        raise ValueError("simulated terminal prices overflow a double; drift or vol too large")
    return s_t


def _check_mc_inputs(S: float, K: float, T: float, sigma: float,
                     option_type: str, n_paths: int, antithetic: bool = False,
                     control_variate: bool = False) -> None:
    validate_inputs(S, K, T, sigma, option_type)
    # A standard error needs two independent samples (paths, or antithetic
    # pairs) left after fitting: three when the control-variate coefficient
    # is fitted on the same samples, since two points fit a line exactly.
    units = (n_paths + 1) // 2 if antithetic else n_paths
    needed = 3 if control_variate else 2
    if units < needed:
        raise ValueError(
            f"n_paths={n_paths!r} gives {units} independent sample(s); at least "
            f"{needed} are needed for a standard error"
            f"{' with the control variate' if control_variate else ''}"
        )


@model(id="EQ-OPT-MC")
def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | np.random.Generator | None = 42,
) -> MCResult:
    """Monte Carlo price of a European option under exact-scheme GBM.

    Parameters
    ----------
    S, K : float
        Spot and strike (currency units), ``>= 0``.
    T : float
        Time to expiry in years (ACT/365F), ``>= 0``.
    r : float
        Continuously compounded annualised risk-free rate.
    sigma : float
        Annualised volatility, ``>= 0``.
    q : float
        Continuously compounded annualised dividend yield.
    option_type : {"call", "put"}
        Option payoff direction.
    n_paths : int
        Total number of paths (rounded up to even when ``antithetic``).
    antithetic : bool
        Use antithetic pairs ``(Z, -Z)``.
    control_variate : bool
        Use the discounted terminal stock as a control variate with the
        sample-optimal coefficient.
    seed : int or numpy.random.Generator or None
        Explicit seed for reproducibility. Same seed => identical result.

    Returns
    -------
    MCResult
        Price estimate with standard error and 95% CI.

    Raises
    ------
    ValueError
        On invalid inputs, fewer than two independent samples (three
        with the control variate; an antithetic pair is one sample), or a drift so large that simulated prices overflow.

    Notes
    -----
    ``T == 0`` or ``sigma == 0`` are deterministic; the exact value is
    returned with ``std_error = 0``.
    """
    _check_mc_inputs(S, K, T, sigma, option_type, n_paths, antithetic, control_variate)
    _check_rates(r=r, q=q)
    if T == 0.0 or sigma == 0.0:
        exact = bs_price(S, K, T, r, sigma, q, option_type)
        return MCResult(exact, 0.0, exact, exact, n_paths)

    rng = _make_rng(seed)
    disc = math.exp(-r * T)
    sign = 1.0 if option_type == "call" else -1.0

    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
        n_eff = 2 * half
    else:
        z = rng.standard_normal(n_paths)
        n_eff = n_paths

    s_t = _terminal(S, T, r, q, sigma, z)
    payoff = disc * np.maximum(sign * (s_t - K), 0.0)

    # Reduce to one value per independent draw first (the antithetic pair
    # average when antithetic), then fit the control-variate coefficient on
    # those same units. The optimal beta for the pair-average estimator is
    # Cov(pair-avg payoff, pair-avg control) / Var(pair-avg control);
    # fitting it on raw draws and pairing afterwards mis-sizes it.
    if antithetic:
        half = n_eff // 2
        samples = 0.5 * (payoff[:half] + payoff[half:])
    else:
        samples = payoff
    if control_variate:
        control = disc * s_t
        if antithetic:
            control = 0.5 * (control[:half] + control[half:])
        control_mean = S * math.exp(-q * T)
        dc = control - control.mean()
        var_c = float(np.dot(dc, dc))
        beta = float(np.dot(samples - samples.mean(), dc)) / var_c if var_c > 0.0 else 0.0
        samples = samples - beta * (control - control_mean)
    return _summary(samples, n_eff)


def mc_delta_pathwise(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    seed: int | np.random.Generator | None = 42,
) -> MCResult:
    """Pathwise Monte Carlo delta (dV/dS) of a European option.

    Uses ``dS_T/dS = S_T / S`` and differentiates through the payoff:
    call delta estimator ``exp(-rT) 1{S_T > K} S_T / S`` (puts analogous).
    Unbiased for vanilla payoffs (Lipschitz).

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type, n_paths, seed
        As in :func:`mc_price`; requires ``T > 0``, ``sigma > 0``, ``S > 0``.

    Returns
    -------
    MCResult
        Delta estimate (dimensionless) with standard error and 95% CI.

    Raises
    ------
    ValueError
        On invalid inputs or degenerate ``T``/``sigma``/``S``.
    """
    _check_mc_inputs(S, K, T, sigma, option_type, n_paths)
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        raise ValueError("pathwise delta requires S > 0, T > 0 and sigma > 0")
    rng = _make_rng(seed)
    z = rng.standard_normal(n_paths)
    s_t = _terminal(S, T, r, q, sigma, z)
    disc = math.exp(-r * T)
    if option_type == "call":
        samples = disc * (s_t > K) * s_t / S
    else:
        samples = -disc * (s_t < K) * s_t / S
    return _summary(samples, n_paths)


def mc_vega_pathwise(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    seed: int | np.random.Generator | None = 42,
) -> MCResult:
    """Pathwise Monte Carlo vega (dV/dsigma) of a European option.

    Uses ``dS_T/dsigma = S_T [ln(S_T/S) - (r - q + sigma^2/2) T] / sigma``.

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type, n_paths, seed
        As in :func:`mc_price`; requires ``T > 0``, ``sigma > 0``, ``S > 0``.

    Returns
    -------
    MCResult
        Vega estimate (per unit of annualised vol) with error bars.

    Raises
    ------
    ValueError
        On invalid inputs or degenerate ``T``/``sigma``/``S``.
    """
    _check_mc_inputs(S, K, T, sigma, option_type, n_paths)
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        raise ValueError("pathwise vega requires S > 0, T > 0 and sigma > 0")
    rng = _make_rng(seed)
    z = rng.standard_normal(n_paths)
    s_t = _terminal(S, T, r, q, sigma, z)
    disc = math.exp(-r * T)
    ds_dsigma = s_t * (np.log(s_t / S) - (r - q + 0.5 * sigma * sigma) * T) / sigma
    indicator = (s_t > K) if option_type == "call" else -(s_t < K).astype(float)
    samples = disc * indicator * ds_dsigma
    return _summary(samples, n_paths)


def mc_delta_lr(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    seed: int | np.random.Generator | None = 42,
) -> MCResult:
    """Likelihood-ratio Monte Carlo delta: payoff times score ``Z/(S sigma sqrt(T))``.

    Higher variance than pathwise but requires no payoff smoothness.

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type, n_paths, seed
        As in :func:`mc_price`; requires ``T > 0``, ``sigma > 0``, ``S > 0``.

    Returns
    -------
    MCResult
        Delta estimate (dimensionless) with error bars.

    Raises
    ------
    ValueError
        On invalid inputs or degenerate ``T``/``sigma``/``S``.
    """
    _check_mc_inputs(S, K, T, sigma, option_type, n_paths)
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        raise ValueError("LR delta requires S > 0, T > 0 and sigma > 0")
    rng = _make_rng(seed)
    z = rng.standard_normal(n_paths)
    s_t = _terminal(S, T, r, q, sigma, z)
    disc = math.exp(-r * T)
    sign = 1.0 if option_type == "call" else -1.0
    payoff = disc * np.maximum(sign * (s_t - K), 0.0)
    weight = z / (S * sigma * math.sqrt(T))
    return _summary(payoff * weight, n_paths)


def mc_vega_lr(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    seed: int | np.random.Generator | None = 42,
) -> MCResult:
    """Likelihood-ratio Monte Carlo vega: payoff times ``(Z^2 - 1)/sigma - Z sqrt(T)``.

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type, n_paths, seed
        As in :func:`mc_price`; requires ``T > 0``, ``sigma > 0``, ``S > 0``.

    Returns
    -------
    MCResult
        Vega estimate (per unit of annualised vol) with error bars.

    Raises
    ------
    ValueError
        On invalid inputs or degenerate ``T``/``sigma``/``S``.
    """
    _check_mc_inputs(S, K, T, sigma, option_type, n_paths)
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        raise ValueError("LR vega requires S > 0, T > 0 and sigma > 0")
    rng = _make_rng(seed)
    z = rng.standard_normal(n_paths)
    s_t = _terminal(S, T, r, q, sigma, z)
    disc = math.exp(-r * T)
    sign = 1.0 if option_type == "call" else -1.0
    payoff = disc * np.maximum(sign * (s_t - K), 0.0)
    weight = (z * z - 1.0) / sigma - z * math.sqrt(T)
    return _summary(payoff * weight, n_paths)


def mc_greek_fd(
    greek: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    rel_bump: float = 1e-2,
    seed: int | np.random.Generator | None = 42,
) -> float:
    """Central finite-difference MC Greek with common random numbers (fallback).

    Re-prices with the *same* normal draws at bumped inputs, so the noise
    largely cancels; a larger bump than for analytic pricers is used to
    keep the variance of the difference under control.

    Parameters
    ----------
    greek : {"delta", "vega", "rho", "theta"}
        Which first-order Greek to estimate.
    S, K, T, r, sigma, q, option_type, n_paths, seed
        As in :func:`mc_price`.
    rel_bump : float
        Relative bump size on the bumped input.

    Returns
    -------
    float
        Point estimate of the requested Greek (natural units, per year for
        theta).

    Raises
    ------
    ValueError
        For an unknown ``greek`` or invalid inputs.
    """
    _check_mc_inputs(S, K, T, sigma, option_type, n_paths)
    if greek not in ("delta", "vega", "rho", "theta"):
        raise ValueError(f"unknown greek {greek!r}")
    rng = _make_rng(seed)
    z = rng.standard_normal(n_paths)  # common random numbers
    sign = 1.0 if option_type == "call" else -1.0

    def price(s: float, t: float, rr: float, sig: float) -> float:
        s_t = _terminal(s, t, rr, q, sig, z)
        return float(np.mean(math.exp(-rr * t) * np.maximum(sign * (s_t - K), 0.0)))

    if greek == "delta":
        h = rel_bump * max(abs(S), 1.0)
        return (price(S + h, T, r, sigma) - price(S - h, T, r, sigma)) / (2 * h)
    if greek == "vega":
        h = rel_bump * max(abs(sigma), 1.0)
        return (price(S, T, r, sigma + h) - price(S, T, r, sigma - h)) / (2 * h)
    if greek == "rho":
        h = rel_bump * max(abs(r), 1.0)
        return (price(S, T, r + h, sigma) - price(S, T, r - h, sigma)) / (2 * h)
    h = rel_bump * max(abs(T), 1.0)
    if T - h <= 0.0:
        raise ValueError(f"T={T} too small for a central theta bump of {h}")
    return -(price(S, T + h, r, sigma) - price(S, T - h, r, sigma)) / (2 * h)
