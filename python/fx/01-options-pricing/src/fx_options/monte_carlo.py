"""Monte Carlo pricing of FX options under the domestic risk-neutral measure.

Under the domestic money-market numeraire the spot follows

    dS/S = (r_d - r_f) dt + sigma dW^d,

so terminal spot is sampled exactly:
``S_T = S exp((r_d - r_f - sigma^2/2) T + sigma sqrt(T) Z)``.

Variance reduction:

* **Antithetic variates** — pairs (Z, -Z), estimator averaged per pair.
* **Control variate** — the discounted terminal spot ``e^{-r_d T} S_T``
  is a martingale-adjusted quantity with known mean ``S e^{-r_f T}``
  (the value today of one unit of foreign currency delivered at T);
  the optimal coefficient is estimated from the sample covariance.

Measure care — the digital example
----------------------------------
This module prices **cash-or-nothing digitals in either currency** to make
the measure/numeraire choice explicit:

* a *domestic-cash* digital pays 1 unit of quote ccy if S_T > K:
  ``e^{-r_d T} N(phi d2)`` — plain expectation under the domestic measure;
* a *foreign-cash* digital pays 1 unit of base ccy, worth ``S_T`` domestic
  at expiry: ``e^{-r_d T} E^d[S_T 1{...}] = S e^{-r_f T} N(phi d1)``.
  Equivalently, changing numeraire to the foreign money market account
  turns it into ``S e^{-r_f T} N^f(...)`` with the *foreign* risk-neutral
  drift ``r_f - r_d`` *for the inverted pair* — the N(d1)/N(d2) split IS
  the two-measure split.  Getting this wrong (discounting the foreign
  cash at ``e^{-r_f T}`` without converting, or using the wrong drift) is
  the classic quanto-style error; the MC here simulates only under the
  domestic measure and converts payoffs at the simulated ``S_T``,
  matching the analytic values to 3 standard errors in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ._common import validate_inputs, validate_option_type
from .garman_kohlhagen import d1 as _d1_fn

__all__ = ["MCResult", "mc_price", "mc_digital_price", "digital_price"]


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimate with sampling-error diagnostics.

    Attributes
    ----------
    price : float
        Point estimate, domestic ccy per unit foreign notional.
    std_error : float
        Standard error of the estimate.
    ci_low, ci_high : float
        95% confidence interval (price +/- 1.96 SE).
    n_paths : int
        Number of underlying draws (antithetic pairs count as 2).
    method : str
        Description of variance-reduction techniques applied.
    """

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    method: str


def _resolve_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _terminal_spots(S: float, T: float, r_d: float, r_f: float, sigma: float,
                    n_paths: int, rng: np.random.Generator,
                    antithetic: bool) -> np.ndarray:
    drift = (r_d - r_f - 0.5 * sigma * sigma) * T
    vol = sigma * math.sqrt(T)
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])[:2 * half]
    else:
        z = rng.standard_normal(n_paths)
    return S * np.exp(drift + vol * z)


def _estimate(discounted: np.ndarray, antithetic: bool,
              control: np.ndarray | None, control_mean: float | None,
              method: str) -> MCResult:
    if control is not None:
        cov = np.cov(discounted, control, ddof=1)
        var_c = cov[1, 1]
        beta = cov[0, 1] / var_c if var_c > 0 else 0.0
        discounted = discounted - beta * (control - control_mean)
    if antithetic:
        # SE from independent pair averages.
        half = discounted.size // 2
        samples = 0.5 * (discounted[:half] + discounted[half:])
    else:
        samples = discounted
    price = float(samples.mean())
    se = float(samples.std(ddof=1) / math.sqrt(samples.size))
    return MCResult(price=price, std_error=se, ci_low=price - 1.96 * se,
                    ci_high=price + 1.96 * se, n_paths=int(discounted.size),
                    method=method)


def mc_price(S: float, K: float, T: float, r_d: float, r_f: float,
             sigma: float, option_type: str, n_paths: int = 100_000,
             rng: np.random.Generator | int | None = 0,
             antithetic: bool = True,
             control_variate: bool = True) -> MCResult:
    """Monte Carlo GK price of a European FX vanilla.

    Parameters
    ----------
    S, K, T, r_d, r_f, sigma : float
        As in :func:`fx_options.garman_kohlhagen.gk_price`; requires T > 0.
    option_type : str
        ``"call"`` or ``"put"``.
    n_paths : int
        Number of terminal draws (rounded up to even when antithetic).
    rng : numpy.random.Generator or int or None
        Generator or seed; every stochastic call is explicitly seeded.
    antithetic, control_variate : bool
        Variance-reduction switches.

    Returns
    -------
    MCResult
    """
    phi = validate_option_type(option_type)
    validate_inputs(S, K, T, r_d, r_f, sigma)
    if T <= 0.0:
        raise ValueError("mc_price requires T > 0")
    if not isinstance(n_paths, int) or n_paths < 2:
        raise ValueError(f"n_paths must be an integer >= 2, got {n_paths!r}")
    gen = _resolve_rng(rng)
    s_t = _terminal_spots(S, T, r_d, r_f, sigma, n_paths, gen, antithetic)
    df_d = math.exp(-r_d * T)
    payoff = df_d * np.maximum(phi * (s_t - K), 0.0)
    control = df_d * s_t if control_variate else None
    control_mean = S * math.exp(-r_f * T) if control_variate else None
    method = "antithetic+" if antithetic else ""
    method += "control_variate" if control_variate else "plain"
    return _estimate(payoff, antithetic, control, control_mean,
                     method.rstrip("+") or "plain")


def digital_price(S: float, K: float, T: float, r_d: float, r_f: float,
                  sigma: float, option_type: str,
                  payout_currency: str = "domestic") -> float:
    """Analytic cash-or-nothing FX digital (unit payout).

    ``payout_currency="domestic"``: pays 1 quote-ccy unit if in the money
    at expiry; value ``e^{-r_d T} N(phi d2)``.
    ``payout_currency="foreign"``: pays 1 base-ccy unit, value in domestic
    ccy ``S e^{-r_f T} N(phi d1)`` (asset-or-nothing divided by S_T at
    payout, i.e. the numeraire-changed probability).  See module docstring
    for the measure discussion.
    """
    phi = validate_option_type(option_type)
    validate_inputs(S, K, T, r_d, r_f, sigma)
    if payout_currency not in ("domestic", "foreign"):
        raise ValueError(
            f"payout_currency must be 'domestic' or 'foreign', "
            f"got {payout_currency!r}"
        )
    if T <= 0.0 or sigma <= 0.0:
        itm = phi * (S * math.exp((r_d - r_f) * max(T, 0.0)) - K) > 0
        if payout_currency == "domestic":
            return math.exp(-r_d * T) * float(itm)
        return math.exp(-r_d * T) * S * math.exp((r_d - r_f) * T) * float(itm)
    _d1 = _d1_fn(S, K, T, r_d, r_f, sigma)
    _d2 = _d1 - sigma * math.sqrt(T)
    if payout_currency == "domestic":
        return math.exp(-r_d * T) * norm.cdf(phi * _d2)
    return S * math.exp(-r_f * T) * norm.cdf(phi * _d1)


def mc_digital_price(S: float, K: float, T: float, r_d: float, r_f: float,
                     sigma: float, option_type: str,
                     payout_currency: str = "domestic",
                     n_paths: int = 200_000,
                     rng: np.random.Generator | int | None = 0,
                     antithetic: bool = True) -> MCResult:
    """Monte Carlo cash-or-nothing digital under the domestic measure.

    The foreign-cash digital's payoff is converted at the simulated
    ``S_T`` and discounted at ``r_d`` — the numeraire-consistent way to
    value a base-currency cash flow.  See module docstring.
    """
    phi = validate_option_type(option_type)
    validate_inputs(S, K, T, r_d, r_f, sigma)
    if T <= 0.0:
        raise ValueError("mc_digital_price requires T > 0")
    if payout_currency not in ("domestic", "foreign"):
        raise ValueError(
            f"payout_currency must be 'domestic' or 'foreign', "
            f"got {payout_currency!r}"
        )
    gen = _resolve_rng(rng)
    s_t = _terminal_spots(S, T, r_d, r_f, sigma, n_paths, gen, antithetic)
    itm = (phi * (s_t - K) > 0).astype(float)
    df_d = math.exp(-r_d * T)
    if payout_currency == "domestic":
        payoff = df_d * itm
    else:
        payoff = df_d * s_t * itm  # convert base-ccy unit at S_T
    method = ("antithetic" if antithetic else "plain") + f"|{payout_currency}"
    return _estimate(payoff, antithetic, None, None, method)
