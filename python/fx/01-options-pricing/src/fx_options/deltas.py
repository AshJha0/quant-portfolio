"""FX delta conventions: spot, forward, premium-adjusted, and inversions.

Why four deltas?  FX options are quoted interbank in (delta, vol) space, so
the *meaning* of delta is part of the quote.  Two independent choices:

1. **Spot vs forward hedge**: hedge with spot (delta includes the foreign
   discount factor) or with an outright forward.
2. **Premium adjustment**: if the premium is paid in the *base* (foreign)
   currency — standard for USDJPY and most EM/non-EURUSD pairs — the
   premium itself is a position in the underlying and must be subtracted
   from the hedge.

With ``phi`` = +1 call / -1 put, ``F = S e^{(r_d - r_f)T}``:

======================  =====================================
Convention              Delta
======================  =====================================
spot                    ``phi e^{-r_f T} N(phi d1)``
forward                 ``phi N(phi d1)``
spot premium-adj        ``phi e^{-r_f T} (K/F) N(phi d2)``
forward premium-adj     ``phi (K/F) N(phi d2)``
======================  =====================================

Relations: ``delta_forward = delta_spot * e^{r_f T}`` and
``delta_pa = delta_unadjusted - premium/S`` (spot form).

**Strike from delta**: analytic for the unadjusted conventions.  For
premium-adjusted *calls* the map K -> delta is **not monotone** (it rises
then falls; (K/F)N(d2) -> 0 both as K -> 0 and K -> inf), so the equation
has zero, one, or two solutions.  Market convention takes the solution on
the *right* (decreasing) branch, i.e. the larger strike — implemented here
by locating the peak of K N(d2) and Brent-solving on [K_peak, K_max].
Premium-adjusted put deltas are monotone in K and Brent-solve directly.

**ATM conventions**:

* ATM-forward: ``K = F``.
* ATM delta-neutral straddle (DNS): strike where call delta + put delta = 0
  under the pair's delta convention: ``K = F e^{+sigma^2 T/2}`` for
  unadjusted deltas (d1 = 0), ``K = F e^{-sigma^2 T/2}`` for
  premium-adjusted deltas (d2 = 0).
"""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm

from ._common import validate_inputs, validate_option_type
from .garman_kohlhagen import d1 as _d1_fn

__all__ = [
    "CONVENTIONS",
    "delta",
    "spot_to_forward_delta",
    "forward_to_spot_delta",
    "premium_adjust_spot_delta",
    "strike_from_delta",
    "atm_forward_strike",
    "atm_dns_strike",
]

CONVENTIONS = ("spot", "forward", "spot_pa", "forward_pa")


def _validate_convention(convention: str) -> str:
    if not isinstance(convention, str) or convention.lower() not in CONVENTIONS:
        raise ValueError(
            f"convention must be one of {CONVENTIONS}, got {convention!r}"
        )
    return convention.lower()


def delta(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float,
          option_type: str, convention: str = "spot") -> float:
    """FX option delta under a chosen quoting convention.

    Parameters
    ----------
    S, K, T, r_d, r_f, sigma : float
        As in :func:`fx_options.garman_kohlhagen.gk_price`; requires
        T > 0 and sigma > 0.
    option_type : str
        ``"call"`` or ``"put"``.
    convention : str
        One of ``"spot"``, ``"forward"``, ``"spot_pa"``, ``"forward_pa"``.
        Premium-adjusted (``_pa``) deltas assume the premium is paid in
        the base (foreign) currency.

    Returns
    -------
    float
        Delta in units of foreign notional (per 1 unit of foreign
        notional of the option).
    """
    phi = validate_option_type(option_type)
    conv = _validate_convention(convention)
    _d1 = _d1_fn(S, K, T, r_d, r_f, sigma)  # validates S,K,T,sigma
    _d2 = _d1 - sigma * math.sqrt(T)
    F = S * math.exp((r_d - r_f) * T)
    if conv == "spot":
        return phi * math.exp(-r_f * T) * norm.cdf(phi * _d1)
    if conv == "forward":
        return phi * norm.cdf(phi * _d1)
    if conv == "spot_pa":
        return phi * math.exp(-r_f * T) * (K / F) * norm.cdf(phi * _d2)
    return phi * (K / F) * norm.cdf(phi * _d2)  # forward_pa


def spot_to_forward_delta(delta_spot: float, T: float, r_f: float) -> float:
    """Convert spot delta to forward delta: ``delta_f = delta_s e^{r_f T}``.

    Holds for both plain and premium-adjusted forms.
    """
    validate_inputs(1.0, 1.0, T, 0.0, r_f, 0.0)
    if not math.isfinite(delta_spot):
        raise ValueError(f"delta_spot must be finite, got {delta_spot!r}")
    return delta_spot * math.exp(r_f * T)


def forward_to_spot_delta(delta_forward: float, T: float, r_f: float) -> float:
    """Convert forward delta to spot delta: ``delta_s = delta_f e^{-r_f T}``."""
    validate_inputs(1.0, 1.0, T, 0.0, r_f, 0.0)
    if not math.isfinite(delta_forward):
        raise ValueError(
            f"delta_forward must be finite, got {delta_forward!r}"
        )
    return delta_forward * math.exp(-r_f * T)


def premium_adjust_spot_delta(delta_spot: float, price: float,
                              S: float) -> float:
    """Premium-adjust a spot delta: ``delta_pa = delta_spot - V/S``.

    ``price`` is the domestic-currency premium; ``V/S`` is the premium
    converted to base currency, which is itself a long-base position the
    hedger already holds and therefore does not need to buy.
    """
    if not math.isfinite(S) or S <= 0:
        raise ValueError(f"Spot S must be positive and finite, got {S!r}")
    for name, value in (("delta_spot", delta_spot), ("price", price)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    return delta_spot - price / S


def atm_forward_strike(S: float, T: float, r_d: float, r_f: float) -> float:
    """ATM-forward strike: ``K = F = S e^{(r_d - r_f) T}``."""
    validate_inputs(S, S, T, r_d, r_f, 0.0)
    return S * math.exp((r_d - r_f) * T)


def atm_dns_strike(S: float, T: float, r_d: float, r_f: float, sigma: float,
                   convention: str = "spot") -> float:
    """ATM delta-neutral-straddle strike under a delta convention.

    ``K = F e^{+sigma^2 T/2}`` for unadjusted deltas (call + put spot or
    forward delta vanishes at d1 = 0), ``K = F e^{-sigma^2 T/2}`` for
    premium-adjusted deltas (vanishes at d2 = 0).  This is the strike at
    which the market quotes 'ATM' vol for most pairs.
    """
    conv = _validate_convention(convention)
    validate_inputs(S, S, T, r_d, r_f, sigma)
    F = S * math.exp((r_d - r_f) * T)
    sign = -1.0 if conv.endswith("_pa") else 1.0
    return F * math.exp(sign * 0.5 * sigma * sigma * T)


def _pa_peak_strike(F: float, T: float, sigma: float) -> float:
    """Strike maximising K*N(d2(K)) — the fold point of the PA call delta.

    Setting d/dK [K N(d2)] = 0 gives ``N(d2) sigma sqrt(T) = n(d2)``,
    a one-dimensional root in d2 (unique: LHS increasing, RHS
    log-concave with a single crossing), then
    ``K = F exp(-d2 sigma sqrt(T) - sigma^2 T / 2)``.
    """
    v = sigma * math.sqrt(T)

    def g(x: float) -> float:
        return norm.cdf(x) * v - norm.pdf(x)

    root = brentq(g, -20.0, 20.0, xtol=1e-14)
    return F * math.exp(-root * v - 0.5 * v * v)


def strike_from_delta(target_delta: float, S: float, T: float, r_d: float,
                      r_f: float, sigma: float, option_type: str,
                      convention: str = "spot") -> float:
    """Invert delta -> strike under any of the four conventions.

    Unadjusted conventions are analytic:
    ``K = F exp(-phi z sigma sqrt(T) + sigma^2 T / 2)`` with
    ``z = N^{-1}(phi delta e^{r_f T})`` (spot) or ``N^{-1}(phi delta)``
    (forward).  Premium-adjusted conventions are solved with Brent.

    For **premium-adjusted calls** the delta-to-strike map is not
    injective; per market convention the root on the decreasing branch
    (the larger strike, the one consistent with OTM quoting) is returned.
    A ``target_delta`` above the fold's maximum attainable PA delta
    raises ``ValueError``.

    Parameters
    ----------
    target_delta : float
        Desired delta, signed (calls in (0, upper), puts in (lower, 0)).
    S, T, r_d, r_f, sigma : float
        Market inputs; requires T > 0, sigma > 0.
    option_type : str
        ``"call"`` or ``"put"``.
    convention : str
        One of :data:`CONVENTIONS`.

    Returns
    -------
    float
        The strike reproducing ``target_delta``.

    Raises
    ------
    ValueError
        If the delta is out of the attainable range for the convention.
    """
    phi = validate_option_type(option_type)
    conv = _validate_convention(convention)
    validate_inputs(S, S, T, r_d, r_f, sigma)
    if T <= 0 or sigma <= 0:
        raise ValueError("strike_from_delta requires T > 0 and sigma > 0")
    if not math.isfinite(target_delta):
        raise ValueError(f"target_delta must be finite, got {target_delta!r}")
    if phi * target_delta <= 0.0:
        raise ValueError(
            f"{option_type} delta must have sign {int(phi)}, got {target_delta}"
        )

    F = S * math.exp((r_d - r_f) * T)
    v = sigma * math.sqrt(T)

    if conv in ("spot", "forward"):
        fwd_delta = (target_delta * math.exp(r_f * T)
                     if conv == "spot" else target_delta)
        if not 0.0 < phi * fwd_delta < 1.0:
            raise ValueError(
                f"forward-equivalent delta {fwd_delta} outside (0, 1) range"
            )
        z = norm.ppf(phi * fwd_delta)  # z = phi * d1
        return F * math.exp(-phi * z * v + 0.5 * v * v)

    # Premium-adjusted: solve phi (K/F) N(phi d2(K)) = forward-equivalent PA delta.
    fwd_delta = (target_delta * math.exp(r_f * T)
                 if conv == "spot_pa" else target_delta)

    def pa_fwd_delta(K: float) -> float:
        _d2 = (math.log(F / K) - 0.5 * v * v) / v
        return phi * (K / F) * norm.cdf(phi * _d2)

    if phi > 0:  # call: non-monotone; use decreasing branch [K_peak, inf)
        k_peak = _pa_peak_strike(F, T, sigma)
        max_delta = pa_fwd_delta(k_peak)
        if fwd_delta > max_delta + 1e-14:
            raise ValueError(
                f"premium-adjusted call delta {target_delta} exceeds the "
                f"maximum attainable {max_delta * (math.exp(-r_f * T) if conv == 'spot_pa' else 1.0):.6f} "
                "for these market inputs"
            )
        k_hi = k_peak
        while pa_fwd_delta(k_hi) > fwd_delta and k_hi < F * math.exp(30 * v):
            k_hi *= 2.0
        return float(brentq(lambda K: pa_fwd_delta(K) - fwd_delta,
                            k_peak, k_hi, xtol=1e-14, maxiter=200))

    # put: |PA delta| strictly increasing in K -> unique root
    if not -1.0 < fwd_delta < 0.0:
        # |put PA fwd delta| < (K/F)N(-d2) can exceed... it is bounded by
        # K/F which is unbounded, but for sensible quotes require (−1, 0).
        raise ValueError(
            f"premium-adjusted put forward delta {fwd_delta} outside (-1, 0)"
        )
    k_lo = F * math.exp(-30 * v)
    k_hi = F
    while pa_fwd_delta(k_hi) > fwd_delta and k_hi < F * math.exp(30 * v):
        k_hi *= 2.0
    return float(brentq(lambda K: pa_fwd_delta(K) - fwd_delta,
                        k_lo, k_hi, xtol=1e-14, maxiter=200))
