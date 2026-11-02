"""FX smile reconstruction from broker quotes and delta -> strike solving.

The OTC FX options market does not quote vols by strike.  Per expiry a
broker screen shows five numbers in *delta space*:

* ``ATM``   - the delta-neutral-straddle (DNS) vol,
* ``RR25``  - 25-delta risk reversal, ``sigma(25dC) - sigma(25dP)``,
* ``BF25``  - 25-delta butterfly,
  ``(sigma(25dC) + sigma(25dP))/2 - sigma_ATM``,
* ``RR10``, ``BF10`` - same at 10 delta.

Broker (market) strangles
-------------------------
:class:`SmileQuotes` carries *smile* butterflies, under which the map
{ATM, RR, BF} <-> five smile vols is linear:

    sigma_C = ATM + BF + RR/2,      sigma_P = ATM + BF - RR/2.

A broker screen quotes the *one-vol market strangle*: the single vol
``sigma_ms = ATM + BF_ms`` at which a strangle struck at the
``sigma_ms``-delta strikes has the same premium as the same two strikes
priced on the smile.  :func:`smile_quotes_from_broker` converts broker
quotes to smile quotes (a two-equation root solve on a five-point SVI
smile), and :func:`broker_strangle_from_smile` computes the broker
quote implied by a smile.  The two butterflies differ by a fraction of a
vol point at 25 delta for G10 and more at 10 delta for skewed pairs.

Delta -> strike
---------------
The strike behind "the 25-delta call" depends on the vol at that strike,
which depends on the strike: a fixed-point/root problem.  For unadjusted
deltas the inversion at a *given* vol is closed-form; premium-adjusted
(pa) deltas require a numerical solve, and the pa *call* delta is
non-monotone in strike - two strikes share one delta.  The market
standard is the strike on the right-hand, decreasing branch (the more
OTM candidate, beyond the delta maximum); see
:func:`strike_from_delta_pa_candidates`.

ATM DNS strikes are exact:
``K = F exp(+sigma^2 T / 2)`` unadjusted, ``K = F exp(-sigma^2 T / 2)``
premium-adjusted (JPY-style pairs quote pa deltas and the pa DNS ATM).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from .garman_kohlhagen import (
    DELTA_CONVENTIONS,
    _phi,
    gk_delta,
    gk_forward,
)

__all__ = [
    "PILLAR_LABELS",
    "SmileQuotes",
    "atm_dns_strike",
    "broker_strangle_from_smile",
    "pa_call_delta_max",
    "quotes_from_vols",
    "smile_quotes_from_broker",
    "solve_pillar_strikes",
    "strike_from_delta",
    "strike_from_delta_pa_candidates",
    "vols_from_quotes",
]

PILLAR_LABELS = ("10p", "25p", "atm", "25c", "10c")


@dataclass(frozen=True)
class SmileQuotes:
    """One expiry of broker quotes, all in vol decimals (0.01 = 1 vol pt).

    ``bf`` quotes use the *smile* butterfly definition.  Convert broker
    one-vol strangles first with :func:`smile_quotes_from_broker`.
    """

    atm: float
    rr25: float
    bf25: float
    rr10: float
    bf10: float

    def __post_init__(self) -> None:
        if self.atm <= 0.0:
            raise ValueError(f"ATM vol must be positive, got {self.atm}")


def vols_from_quotes(quotes: SmileQuotes) -> dict[str, float]:
    """Exact linear map {ATM, RR, BF} -> five smile vols.

    ``sigma_25c = atm + bf25 + rr25/2``, ``sigma_25p = atm + bf25 - rr25/2``
    (same at 10 delta).  A positive RR therefore makes calls richer than
    puts by construction.  Negative butterflies (concave smile centre)
    are unusual but not impossible on broker screens; they are accepted
    with a warning.  A quote set implying a non-positive wing vol raises.

    Returns
    -------
    dict
        Keys ``('10p', '25p', 'atm', '25c', '10c')``, vols in decimals.
    """
    if quotes.bf25 < 0.0 or quotes.bf10 < 0.0:
        warnings.warn(
            "negative butterfly quote encountered (concave smile centre); "
            "proceeding, but check for a broker-strangle vs smile-strangle "
            "convention mismatch",
            UserWarning,
            stacklevel=2,
        )
    vols = {
        "10p": quotes.atm + quotes.bf10 - 0.5 * quotes.rr10,
        "25p": quotes.atm + quotes.bf25 - 0.5 * quotes.rr25,
        "atm": quotes.atm,
        "25c": quotes.atm + quotes.bf25 + 0.5 * quotes.rr25,
        "10c": quotes.atm + quotes.bf10 + 0.5 * quotes.rr10,
    }
    for label, v in vols.items():
        if v <= 0.0:
            raise ValueError(
                f"quote set implies non-positive vol at pillar {label}: {v}"
            )
    return vols


def quotes_from_vols(vols: dict[str, float]) -> SmileQuotes:
    """Inverse of :func:`vols_from_quotes` (exact round trip)."""
    for label in PILLAR_LABELS:
        if label not in vols:
            raise ValueError(f"missing pillar {label!r} in vols dict")
    return SmileQuotes(
        atm=vols["atm"],
        rr25=vols["25c"] - vols["25p"],
        bf25=0.5 * (vols["25c"] + vols["25p"]) - vols["atm"],
        rr10=vols["10c"] - vols["10p"],
        bf10=0.5 * (vols["10c"] + vols["10p"]) - vols["atm"],
    )


def atm_dns_strike(F: float, sigma: float, T: float, premium_adjusted: bool = False) -> float:
    """Delta-neutral-straddle ATM strike.

    Unadjusted conventions: call and put (spot or forward) deltas cancel
    iff ``d1 = 0``, i.e. ``K = F exp(+sigma^2 T/2)``.  Premium-adjusted:
    they cancel iff ``d2 = 0``, i.e. ``K = F exp(-sigma^2 T/2)`` - the pa
    DNS ATM sits *below* the forward.
    """
    if not all(math.isfinite(v) for v in (F, sigma, T)):
        raise ValueError(f"F, sigma, T must be finite, got F={F}, sigma={sigma}, T={T}")
    if F <= 0.0 or sigma <= 0.0 or T <= 0.0:
        raise ValueError("F, sigma, T must be positive")
    sign = -1.0 if premium_adjusted else 1.0
    return F * math.exp(sign * 0.5 * sigma * sigma * T)


def _strike_from_delta_unadjusted(
    delta: float, cp: int, sigma: float, S: float, T: float,
    r_d: float, r_f: float, convention: str,
) -> float:
    """Closed-form inversion for unadjusted spot/forward delta."""
    F = gk_forward(S, T, r_d, r_f)
    sqT = math.sqrt(T)
    d_fwd = delta * math.exp(r_f * T) if convention == "spot" else delta
    if d_fwd >= 1.0:
        raise ValueError(
            f"target forward delta {d_fwd:.6f} >= 1 is unattainable "
            f"(spot delta {delta} with r_f={r_f}, T={T})"
        )
    d1 = ndtri(d_fwd) if cp == 1 else -ndtri(d_fwd)
    return F * math.exp(-sigma * sqT * d1 + 0.5 * sigma * sigma * T)


def pa_call_delta_max(
    sigma: float, S: float, T: float, r_d: float, r_f: float, convention: str,
) -> tuple[float, float]:
    """Location and value of the premium-adjusted call delta maximum.

    The pa call delta ``w DF (K/F) N(d2)`` vanishes at both K -> 0 and
    K -> inf, with a unique interior maximum at the strike where
    ``N(d2) sigma sqrt(T) = phi(d2)``.

    Returns
    -------
    (K_max, delta_max) : tuple of float
    """
    F = gk_forward(S, T, r_d, r_f)
    sqT = math.sqrt(T)
    v = sigma * sqT

    def h(d2: float) -> float:
        return ndtr(d2) * v - _phi(d2)

    # h is increasing for d2 > -v and h(-v) < 0 (Mills ratio), so the
    # root is unique in (-v, inf).
    hi = 1.0
    while h(hi) < 0.0:
        hi += 2.0
    d2_star = brentq(h, -v, hi, xtol=1e-14)
    K_max = F * math.exp(-d2_star * v - 0.5 * v * v)
    delta_max = gk_delta(S, K_max, T, r_d, r_f, sigma, +1, convention)
    return K_max, delta_max


def strike_from_delta_pa_candidates(
    delta: float, sigma: float, S: float, T: float,
    r_d: float, r_f: float, convention: str,
) -> tuple[float, float]:
    """Both strikes sharing a given premium-adjusted *call* delta.

    Returns ``(K_low, K_market)``: the low-strike candidate on the
    rising branch (rejected by the market) and the high-strike candidate
    on the falling branch (the market-standard choice - the OTM call).
    """
    if not 0.0 < delta:
        raise ValueError(f"pa call delta target must be positive, got {delta}")
    K_max, delta_max = pa_call_delta_max(sigma, S, T, r_d, r_f, convention)
    if delta >= delta_max:
        raise ValueError(
            f"pa call delta {delta:.6f} unattainable: maximum achievable "
            f"is {delta_max:.6f} at K={K_max:.6f}"
        )

    def g(K: float) -> float:
        return gk_delta(S, K, T, r_d, r_f, sigma, +1, convention) - delta

    F = gk_forward(S, T, r_d, r_f)
    v = sigma * math.sqrt(T)
    lo = K_max * math.exp(-10.0 * v - 10.0)
    hi = K_max * math.exp(10.0 * v + 10.0)
    K_low = brentq(g, lo, K_max, xtol=1e-14 * F)
    K_market = brentq(g, K_max, hi, xtol=1e-14 * F)
    return K_low, K_market


def strike_from_delta(
    delta: float,
    cp: int,
    sigma: float,
    S: float,
    T: float,
    r_d: float,
    r_f: float,
    convention: str = "spot",
) -> float:
    """Solve the strike with the given (unsigned) delta at a given vol.

    Parameters
    ----------
    delta : float
        Unsigned delta magnitude in (0, 1), e.g. 0.25 for both the
        25-delta call and the 25-delta put.
    cp : int
        +1 call, -1 put.
    sigma : float
        The vol *at that strike* (for pillar construction this is the
        pillar vol; smile-consistent solving iterates this function).
    convention : str
        One of ``spot | forward | spot_pa | forward_pa``.

    Notes
    -----
    Unadjusted conventions invert in closed form.  Premium-adjusted puts
    are monotone in strike (Brent).  Premium-adjusted calls are
    non-monotone: the market-standard high-strike branch is selected
    (see :func:`strike_from_delta_pa_candidates`).
    """
    if convention not in DELTA_CONVENTIONS:
        raise ValueError(
            f"unknown delta convention {convention!r}; expected one of {DELTA_CONVENTIONS}"
        )
    if cp not in (+1, -1):
        raise ValueError(f"cp must be +1 or -1, got {cp}")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError(f"delta magnitude must be in (0, 1), got {delta}")
    if not all(math.isfinite(v) for v in (sigma, S, T, r_d, r_f)):
        raise ValueError(
            f"sigma, S, T, r_d, r_f must be finite, got sigma={sigma}, S={S}, "
            f"T={T}, r_d={r_d}, r_f={r_f}"
        )
    if sigma <= 0.0 or S <= 0.0 or T <= 0.0:
        raise ValueError("sigma, S, T must be positive")

    if convention in ("spot", "forward"):
        return _strike_from_delta_unadjusted(delta, cp, sigma, S, T, r_d, r_f, convention)

    if cp == 1:
        _, K_market = strike_from_delta_pa_candidates(
            delta, sigma, S, T, r_d, r_f, convention
        )
        return K_market

    # pa put: |delta| = DF (K/F) N(-d2) is strictly increasing in K -> unique root.
    F = gk_forward(S, T, r_d, r_f)
    v = sigma * math.sqrt(T)

    def g(K: float) -> float:
        return -gk_delta(S, K, T, r_d, r_f, sigma, -1, convention) - delta

    lo = F * math.exp(-12.0 * v - 12.0)
    hi = F * math.exp(12.0 * v + 12.0)
    if g(hi) < 0.0:
        raise ValueError(
            f"pa put delta {delta} unattainable within bracket (T={T}, sigma={sigma})"
        )
    return brentq(g, lo, hi, xtol=1e-14 * F)


def solve_pillar_strikes(
    vols: dict[str, float],
    S: float,
    T: float,
    r_d: float,
    r_f: float,
    convention: str = "spot",
) -> dict[str, float]:
    """Strikes for the five pillars 10P/25P/ATM/25C/10C from their vols.

    ATM uses the delta-neutral-straddle strike consistent with the
    convention (pa conventions use the pa DNS ``K = F exp(-sigma^2 T/2)``).
    Wing pillars invert delta at the pillar vol.  Verifies the market
    ordering ``K10P < K25P < KATM < K25C < K10C`` and raises otherwise.

    Returns
    -------
    dict
        Strikes keyed by pillar label.
    """
    F = gk_forward(S, T, r_d, r_f)
    pa = convention.endswith("_pa")
    strikes = {
        "10p": strike_from_delta(0.10, -1, vols["10p"], S, T, r_d, r_f, convention),
        "25p": strike_from_delta(0.25, -1, vols["25p"], S, T, r_d, r_f, convention),
        "atm": atm_dns_strike(F, vols["atm"], T, premium_adjusted=pa),
        "25c": strike_from_delta(0.25, +1, vols["25c"], S, T, r_d, r_f, convention),
        "10c": strike_from_delta(0.10, +1, vols["10c"], S, T, r_d, r_f, convention),
    }
    ks = [strikes[label] for label in PILLAR_LABELS]
    if not all(a < b for a, b in zip(ks, ks[1:], strict=False)):
        raise ValueError(
            f"pillar strikes not strictly ordered K10P < K25P < KATM < K25C < K10C: "
            f"{dict(zip(PILLAR_LABELS, ks, strict=False))}"
        )
    return strikes


def _strangle_premium(
    K_put: float, K_call: float, sig_put: float, sig_call: float,
    S: float, T: float, r_d: float, r_f: float,
) -> float:
    from .garman_kohlhagen import gk_price
    return (gk_price(S, K_call, T, r_d, r_f, sig_call, +1)
            + gk_price(S, K_put, T, r_d, r_f, sig_put, -1))


def broker_strangle_from_smile(
    vol_fn, atm: float, delta: float, S: float, T: float,
    r_d: float, r_f: float, convention: str = "spot",
) -> float:
    """One-vol market strangle quote ``BF_ms`` implied by a smile.

    Solves for ``BF_ms`` such that the strangle struck at the
    ``delta``-call and ``delta``-put strikes computed at the single vol
    ``atm + BF_ms`` has the same premium at ``atm + BF_ms`` as it has
    when each leg is priced at its own smile vol ``vol_fn(K)``.
    """
    if not 0.0 < delta < 0.5:
        raise ValueError(f"strangle delta must be in (0, 0.5), got {delta}")
    if not (math.isfinite(atm) and atm > 0.0):
        raise ValueError(f"atm must be positive and finite, got {atm}")

    def f(bf: float) -> float:
        sig = atm + bf
        Kc = strike_from_delta(delta, +1, sig, S, T, r_d, r_f, convention)
        Kp = strike_from_delta(delta, -1, sig, S, T, r_d, r_f, convention)
        return (_strangle_premium(Kp, Kc, sig, sig, S, T, r_d, r_f)
                - _strangle_premium(Kp, Kc, float(vol_fn(Kp)), float(vol_fn(Kc)),
                                    S, T, r_d, r_f))

    lo, hi = -0.5 * atm, 0.5 * atm
    if f(lo) * f(hi) > 0.0:
        raise ValueError("market strangle not bracketed within +-50% of ATM vol")
    return brentq(f, lo, hi, xtol=1e-15, rtol=4.0 * np.finfo(float).eps)


def smile_quotes_from_broker(
    atm: float, rr25: float, bf25_ms: float, rr10: float, bf10_ms: float,
    S: float, T: float, r_d: float, r_f: float, convention: str = "spot",
    tol: float = 1e-12,
) -> SmileQuotes:
    """Convert broker quotes (one-vol market strangles) to smile quotes.

    Risk reversals and ATM are the same in both conventions.  The two
    smile butterflies are solved jointly so that the five-point SVI
    smile through the resulting pillar vols reprices both market
    strangles (25 and 10 delta).  Raises ``ValueError`` if the solve does
    not converge to ``tol`` in premium.

    The interpolating smile matters: the 25d market strangle strikes lie
    between pillars, so a different interpolant gives slightly different
    smile butterflies.  SVI is the interpolant used by
    :func:`fx_surface.surface.build_slice`.
    """
    from scipy.optimize import root

    from .smile import SVISmile

    for name, v in (("atm", atm), ("rr25", rr25), ("bf25_ms", bf25_ms),
                    ("rr10", rr10), ("bf10_ms", bf10_ms)):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v}")
    F = gk_forward(S, T, r_d, r_f)

    def smile_for(bf25: float, bf10: float):
        q = SmileQuotes(atm, rr25, bf25, rr10, bf10)
        vols = vols_from_quotes(q)
        ks = solve_pillar_strikes(vols, S, T, r_d, r_f, convention)
        return q, SVISmile.fit(np.array([ks[p] for p in PILLAR_LABELS]),
                               np.array([vols[p] for p in PILLAR_LABELS]), F, T)

    def resid(x: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, sm = smile_for(float(x[0]), float(x[1]))
        out = []
        for d, bf_ms in ((0.25, bf25_ms), (0.10, bf10_ms)):
            sig = atm + bf_ms
            Kc = strike_from_delta(d, +1, sig, S, T, r_d, r_f, convention)
            Kp = strike_from_delta(d, -1, sig, S, T, r_d, r_f, convention)
            vega_scale = S * math.sqrt(T)
            out.append((_strangle_premium(Kp, Kc, float(sm.vol(Kp)), float(sm.vol(Kc)),
                                          S, T, r_d, r_f)
                        - _strangle_premium(Kp, Kc, sig, sig, S, T, r_d, r_f)) / vega_scale)
        return np.array(out)

    sol = root(resid, np.array([bf25_ms, bf10_ms]), method="hybr", tol=1e-14)
    if not np.all(np.isfinite(sol.x)) or np.max(np.abs(resid(sol.x))) > tol:
        raise ValueError(
            f"broker strangle conversion did not converge: {sol.message} "
            f"(residual {np.max(np.abs(resid(sol.x))):.2e})"
        )
    return SmileQuotes(atm, rr25, float(sol.x[0]), rr10, float(sol.x[1]))
