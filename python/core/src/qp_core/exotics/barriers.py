"""Closed-form barrier options, touches and double barriers under Black-Scholes.

Plain idea
----------
A barrier option is a vanilla that is switched on (knock-in) or off
(knock-out) if the spot touches a level before expiry. Under constant
volatility the first-passage distribution of a drifted Brownian motion is
known, so these prices have closed forms built from a handful of normal
CDF terms. A *touch* pays cash when (one-touch) or if not (no-touch) the
level is hit. Double barriers have two levels; their prices are series
over the images of the reflection principle, which converge in a few terms.

All functions use the cost-of-carry form ``b`` (``b = r - q`` for a stock
with dividend yield ``q``, ``b = r_d - r_f`` for FX), continuous
monitoring, and a cash rebate.

Formal statement
----------------
Single barriers: Reiner and Rubinstein (1991), "Breaking down the
barriers", Risk 4(8), in the A-F form of Haug (2007), *The Complete Guide
to Option Pricing Formulas*, 2nd ed., s.4.17.1. With ``mu = (b -
sigma^2/2)/sigma^2`` and ``lambda = sqrt(mu^2 + 2 r / sigma^2)``, eight
types are combinations of the six terms ``A..F`` (see ``_rr_terms``).
Knock-in rebates are paid at expiry; knock-out rebates at the hit.

Touches: Reiner and Rubinstein (1991b), "Unscrambling the binary code",
Risk 4(9); Haug (2007) s.4.19, types 1-4 (cash at hit), 5-8 (cash at
expiry if hit) and 9-12 (cash at expiry if not hit).

Double knock-out: Ikeda and Kunitomo (1992), "Pricing options with curved
boundaries", Math. Finance 2(4); Haug (2007) s.4.17.3, flat boundaries.
Double no-touch: Hui (1996), "One-touch double barrier binary option
values", Applied Financial Economics 6(4); Haug (2007) s.4.19.5.

Worked example
--------------
Haug (2007) Table 4-13: ``S = 100, X = 90, H = 95, rebate 3, T = 0.5,
r = 0.08, b = 0.04, sigma = 0.25``; the down-and-out call is 9.0246.

>>> round(barrier_price("down_out", "call", 100, 90, 95, 0.5, 0.08, 0.04, 0.25, rebate=3.0), 4)
9.0246
"""

from __future__ import annotations

import math

from qp_core.model import model

from ..normal import norm_cdf as N

__all__ = [
    "BARRIER_TYPES",
    "barrier_price",
    "double_knockout_price",
    "double_no_touch",
    "touch_price",
    "vanilla_price",
]

BARRIER_TYPES = ("down_in", "down_out", "up_in", "up_out")


def _check(S: float, X: float, T: float, sigma: float) -> None:
    for name, val in (("S", S), ("X", X), ("T", T), ("sigma", sigma)):
        if not (math.isfinite(val) and val > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {val}")


def _phi_of(cp: str) -> float:
    if cp == "call":
        return 1.0
    if cp == "put":
        return -1.0
    raise ValueError(f"cp must be 'call' or 'put', got {cp!r}")


def vanilla_price(cp: str, S: float, X: float, T: float, r: float, b: float, sigma: float) -> float:
    """Generalised Black-Scholes (Haug 2007 s.1.1.6) with cost of carry ``b``."""
    _check(S, X, T, sigma)
    phi = _phi_of(cp)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / X) + (b + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return phi * (S * math.exp((b - r) * T) * N(phi * d1) - X * math.exp(-r * T) * N(phi * d2))


def _rr_terms(
    phi: float, eta: float, S: float, X: float, H: float, K: float,
    T: float, r: float, b: float, sigma: float,
) -> tuple[float, float, float, float, float, float]:
    v = sigma * math.sqrt(T)
    mu = (b - 0.5 * sigma * sigma) / (sigma * sigma)
    lam = math.sqrt(mu * mu + 2.0 * r / (sigma * sigma))
    x1 = math.log(S / X) / v + (1.0 + mu) * v
    x2 = math.log(S / H) / v + (1.0 + mu) * v
    y1 = math.log(H * H / (S * X)) / v + (1.0 + mu) * v
    y2 = math.log(H / S) / v + (1.0 + mu) * v
    z = math.log(H / S) / v + lam * v
    fs = S * math.exp((b - r) * T)
    fx = X * math.exp(-r * T)
    hs = H / S
    A = phi * fs * N(phi * x1) - phi * fx * N(phi * x1 - phi * v)
    B = phi * fs * N(phi * x2) - phi * fx * N(phi * x2 - phi * v)
    C = (phi * fs * hs ** (2.0 * (mu + 1.0)) * N(eta * y1)
         - phi * fx * hs ** (2.0 * mu) * N(eta * y1 - eta * v))
    D = (phi * fs * hs ** (2.0 * (mu + 1.0)) * N(eta * y2)
         - phi * fx * hs ** (2.0 * mu) * N(eta * y2 - eta * v))
    E = K * math.exp(-r * T) * (N(eta * x2 - eta * v) - hs ** (2.0 * mu) * N(eta * y2 - eta * v))
    F = K * (hs ** (mu + lam) * N(eta * z) + hs ** (mu - lam) * N(eta * z - 2.0 * eta * lam * v))
    return A, B, C, D, E, F


@model(id="CORE-EXO-BARRIER")
def barrier_price(
    kind: str, cp: str, S: float, X: float, H: float, T: float, r: float, b: float,
    sigma: float, rebate: float = 0.0,
) -> float:
    """Single-barrier option, continuous monitoring (Reiner-Rubinstein 1991).

    ``kind`` is one of ``down_in``, ``down_out``, ``up_in``, ``up_out``.
    If the spot is already at or beyond the barrier, a knock-in is the
    vanilla and a knock-out is worth its rebate, paid now.
    """
    _check(S, X, T, sigma)
    if not (math.isfinite(H) and H > 0.0):
        raise ValueError(f"H must be finite and > 0, got {H}")
    if rebate < 0.0 or not math.isfinite(rebate):
        raise ValueError(f"rebate must be finite and >= 0, got {rebate}")
    if kind not in BARRIER_TYPES:
        raise ValueError(f"kind must be one of {BARRIER_TYPES}, got {kind!r}")
    phi = _phi_of(cp)
    down = kind.startswith("down")
    knocked = S <= H if down else S >= H
    if knocked:
        return vanilla_price(cp, S, X, T, r, b, sigma) if kind.endswith("in") else rebate
    eta = 1.0 if down else -1.0
    A, B, C, D, E, F = _rr_terms(phi, eta, S, X, H, rebate, T, r, b, sigma)
    above = X > H  # Haug's case split is on X > H versus X < H; X == H uses X < H
    call = phi > 0
    table = {
        ("down_in", True): (C + E, A - B + D + E),
        ("up_in", True): (A + E, B - C + D + E),
        ("down_in", False): (B - C + D + E, A + E),
        ("up_in", False): (A - B + D + E, C + E),
        ("down_out", True): (A - C + F, B - D + F),
        ("up_out", True): (F, A - B + C - D + F),
        ("down_out", False): (A - B + C - D + F, F),
        ("up_out", False): (B - D + F, A - C + F),
    }
    hi, lo = table[(kind, call)]
    return hi if above else lo


@model(id="CORE-EXO-TOUCH")
def touch_price(
    kind: str, S: float, H: float, T: float, r: float, b: float, sigma: float,
    cash: float = 1.0, pay_at_hit: bool = False,
) -> float:
    """One-touch / no-touch paying ``cash`` (Reiner-Rubinstein 1991b; Haug 2007 s.4.19).

    ``kind`` is ``one_touch_down``, ``one_touch_up``, ``no_touch_down`` or
    ``no_touch_up``. One-touches pay at expiry by default; with
    ``pay_at_hit=True`` they pay at the first hit (Haug types 1-4).
    No-touches always pay at expiry.
    """
    _check(S, 1.0, T, sigma)
    if not (math.isfinite(H) and H > 0.0):
        raise ValueError(f"H must be finite and > 0, got {H}")
    if kind not in ("one_touch_down", "one_touch_up", "no_touch_down", "no_touch_up"):
        raise ValueError(f"unknown touch kind {kind!r}")
    down = kind.endswith("down")
    knocked = S <= H if down else S >= H
    df = math.exp(-r * T)
    one = kind.startswith("one")
    if knocked:
        return (cash if pay_at_hit else cash * df) if one else 0.0
    eta = 1.0 if down else -1.0
    v = sigma * math.sqrt(T)
    mu = (b - 0.5 * sigma * sigma) / (sigma * sigma)
    hs = H / S
    x2 = math.log(S / H) / v + (1.0 + mu) * v
    y2 = math.log(H / S) / v + (1.0 + mu) * v
    # Probability the barrier is NOT hit by T (risk-neutral), Haug B2 - B4 terms.
    p_no = N(eta * x2 - eta * v) - hs ** (2.0 * mu) * N(eta * y2 - eta * v)
    if not one:
        return float(cash * df * p_no)
    if not pay_at_hit:
        return float(cash * df * (1.0 - p_no))
    lam = math.sqrt(mu * mu + 2.0 * r / (sigma * sigma))
    z = math.log(H / S) / v + lam * v
    return float(cash * (hs ** (mu + lam) * N(eta * z) + hs ** (mu - lam) * N(eta * z - 2.0 * eta * lam * v)))


@model(id="CORE-EXO-BARRIER")
def double_knockout_price(
    cp: str, S: float, X: float, L: float, U: float, T: float, r: float, b: float,
    sigma: float, n_terms: int = 10,
) -> float:
    """Double knock-out call or put with flat barriers (Ikeda-Kunitomo 1992).

    The series runs over ``n = -n_terms..n_terms``; ten terms each side is
    far past convergence for any ``ln(U/L) > 0.05`` at ``sigma sqrt(T) < 2``
    (the test suite checks that 10 and 20 terms agree to 1e-14).
    """
    _check(S, X, T, sigma)
    if not (0.0 < L < U and math.isfinite(U)):
        raise ValueError(f"need 0 < L < U, got L={L}, U={U}")
    phi = _phi_of(cp)
    if S <= L or S >= U:
        return 0.0
    v = sigma * math.sqrt(T)
    s2 = sigma * sigma
    drift = (b + 0.5 * s2) * T
    mu1 = 2.0 * b / s2 + 1.0
    mu3 = mu1  # flat barriers: delta1 = delta2 = 0, so mu2 = 0 and mu3 = mu1
    E, F = L, U  # the curved barriers at expiry; flat here
    s1 = 0.0
    s2_ = 0.0
    for n in range(-n_terms, n_terms + 1):
        u2n = U ** (2 * n)
        l2n = L ** (2 * n)
        a1 = (U**n / L**n) ** mu1
        a1m = (U**n / L**n) ** (mu1 - 2.0)
        r3 = L ** (n + 1) / (U**n * S)
        a3 = r3**mu3
        a3m = r3 ** (mu3 - 2.0)
        if phi > 0:
            d1 = (math.log(S * u2n / (X * l2n)) + drift) / v
            d2 = (math.log(S * u2n / (F * l2n)) + drift) / v
            d3 = (math.log(L ** (2 * n + 2) / (X * S * u2n)) + drift) / v
            d4 = (math.log(L ** (2 * n + 2) / (F * S * u2n)) + drift) / v
        else:
            d1 = (math.log(S * u2n / (E * l2n)) + drift) / v
            d2 = (math.log(S * u2n / (X * l2n)) + drift) / v
            d3 = (math.log(L ** (2 * n + 2) / (E * S * u2n)) + drift) / v
            d4 = (math.log(L ** (2 * n + 2) / (X * S * u2n)) + drift) / v
        s1 += a1 * (N(d1) - N(d2)) - a3 * (N(d3) - N(d4))
        s2_ += a1m * (N(d1 - v) - N(d2 - v)) - a3m * (N(d3 - v) - N(d4 - v))
    fs = S * math.exp((b - r) * T)
    fx = X * math.exp(-r * T)
    if phi > 0:
        if X >= U:
            return 0.0
        return fs * s1 - fx * s2_
    if X <= L:
        return 0.0
    return fx * s2_ - fs * s1


@model(id="CORE-EXO-TOUCH")
def double_no_touch(
    S: float, L: float, U: float, T: float, r: float, b: float, sigma: float,
    cash: float = 1.0, n_terms: int = 50,
) -> float:
    """Double no-touch paying ``cash`` at expiry if neither barrier is hit (Hui 1996).

    The sine series converges like ``exp(-i^2 pi^2 sigma^2 T / (2 Z^2))``
    with ``Z = ln(U/L)``; fifty terms are ample unless ``sigma^2 T`` is tiny
    relative to ``Z^2``, in which case more are needed (and passed).
    """
    _check(S, 1.0, T, sigma)
    if not (0.0 < L < U and math.isfinite(U)):
        raise ValueError(f"need 0 < L < U, got L={L}, U={U}")
    if S <= L or S >= U:
        return 0.0
    s2 = sigma * sigma
    Z = math.log(U / L)
    alpha = -0.5 * (2.0 * b / s2 - 1.0)
    beta = -0.25 * (2.0 * b / s2 - 1.0) ** 2 - 2.0 * r / s2
    total = 0.0
    sl = (S / L) ** alpha
    su = (S / U) ** alpha
    for i in range(1, n_terms + 1):
        ip = i * math.pi / Z
        total += (
            2.0 * math.pi * i / (Z * Z)
            * (sl - (-1) ** i * su) / (alpha * alpha + ip * ip)
            * math.sin(ip * math.log(S / L))
            * math.exp(-0.5 * (ip * ip - beta) * s2 * T)
        )
    return cash * total
