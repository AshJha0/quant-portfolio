"""Black-76 pricing and Greeks for options on forwards/futures.

Use case: equity *index futures* options (and index options quoted off the
forward). Marking off the forward absorbs both the financing rate and the
(hard-to-observe) dividend yield into one observable input ``F``, which is
why index vol desks quote and risk-manage in Black-76 terms.

Conventions
-----------
* ``F`` is the forward/futures price for expiry ``T`` (years, ACT/365F).
* ``r`` is the continuously compounded annualised discount rate applied to
  the premium (for daily-margined futures options with no premium
  discounting, pass ``r = 0``).
* ``sigma`` is the annualised volatility of the forward's log-returns.
* Equivalence with Black-Scholes: with ``F = S exp((r - q) T)``,
  Black-76 reproduces the Black-Scholes-Merton price exactly.

Greeks returned are with respect to the forward ``F`` (delta, gamma) and
per unit of vol / per year / per unit of rate (vega, theta, rho). Rho here
is the sensitivity of the *discounting only* (``F`` held fixed):
``rho = -T * price``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from .black_scholes import OptionType, validate_inputs

__all__ = ["black76_price", "black76_greeks", "Black76Greeks", "b76_d1_d2"]

_SQRT_2PI = math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class Black76Greeks:
    """Analytic Black-76 Greeks (with respect to the forward ``F``).

    Attributes
    ----------
    price : float
        Present value, currency units.
    delta : float
        dV/dF, dimensionless.
    gamma : float
        d2V/dF2, per currency unit.
    vega : float
        dV/dsigma, currency units per unit of annualised vol (divide by 100
        for 'per vol point').
    theta : float
        dV/dt = -dV/dT, currency units per year (calendar decay with F fixed).
    rho : float
        dV/dr with F held fixed: pure discounting sensitivity ``-T * V``.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def b76_d1_d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    """Black-76 ``d1``/``d2``: ``d1 = [ln(F/K) + sigma^2 T/2]/(sigma sqrt(T))``.

    Parameters
    ----------
    F, K : float
        Forward and strike, strictly positive.
    T : float
        Time to expiry in years, strictly positive.
    sigma : float
        Annualised volatility, strictly positive.

    Returns
    -------
    tuple of float
        ``(d1, d2)`` with ``d2 = d1 - sigma sqrt(T)``.

    Raises
    ------
    ValueError
        If ``F``, ``K``, ``T`` or ``sigma`` is not strictly positive.
    """
    validate_inputs(F, K, T, sigma)
    if F <= 0.0 or K <= 0.0 or T <= 0.0 or sigma <= 0.0:
        raise ValueError(
            "b76_d1_d2 requires strictly positive F, K, T, sigma; "
            f"got F={F}, K={K}, T={T}, sigma={sigma}"
        )
    sqrt_t = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def black76_price(
    F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Black-76 present value of a European option on a forward/futures price.

    Parameters
    ----------
    F : float
        Forward/futures price for expiry ``T``, ``F >= 0``.
    K : float
        Strike, ``K >= 0``.
    T : float
        Time to expiry in years (ACT/365F), ``T >= 0``.
    r : float
        Continuously compounded annualised discount rate for the premium.
    sigma : float
        Annualised volatility of the forward, ``sigma >= 0``.
    option_type : {"call", "put"}
        Option payoff direction.

    Returns
    -------
    float
        Present value in currency units.

    Raises
    ------
    ValueError
        On negative/NaN inputs or invalid ``option_type``.

    Notes
    -----
    ``T == 0`` returns intrinsic; ``sigma == 0`` returns the discounted
    intrinsic ``exp(-rT) max(±(F - K), 0)``.
    """
    validate_inputs(F, K, T, sigma, option_type)
    sign = 1.0 if option_type == "call" else -1.0
    if T == 0.0:
        return max(sign * (F - K), 0.0)
    df = math.exp(-r * T)
    if sigma == 0.0 or K == 0.0 or F == 0.0:
        return df * max(sign * (F - K), 0.0)
    d1, d2 = b76_d1_d2(F, K, T, sigma)
    return df * sign * (F * norm.cdf(sign * d1) - K * norm.cdf(sign * d2))


def black76_greeks(
    F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> Black76Greeks:
    """Analytic Black-76 Greeks with respect to the forward.

    Parameters
    ----------
    F, K, T, r, sigma : float
        As in :func:`black76_price`; ``F``, ``K``, ``T``, ``sigma`` must be
        strictly positive for finite Greeks.
    option_type : {"call", "put"}
        Option payoff direction.

    Returns
    -------
    Black76Greeks
        Dataclass of price, delta, gamma, vega, theta, rho (units in the
        dataclass docstring).

    Raises
    ------
    ValueError
        If inputs are invalid or not strictly positive.
    """
    d1, d2 = b76_d1_d2(F, K, T, sigma)
    df = math.exp(-r * T)
    sqrt_t = math.sqrt(T)
    pdf_d1 = math.exp(-0.5 * d1 * d1) / _SQRT_2PI
    sign = 1.0 if option_type == "call" else -1.0

    price = df * sign * (F * norm.cdf(sign * d1) - K * norm.cdf(sign * d2))
    delta = df * sign * norm.cdf(sign * d1)
    gamma = df * pdf_d1 / (F * sigma * sqrt_t)
    vega = df * F * pdf_d1 * sqrt_t
    # theta = dV/dt at fixed F: r*V - decay of the time value.
    theta = r * price - df * F * pdf_d1 * sigma / (2.0 * sqrt_t)
    rho = -T * price
    return Black76Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
