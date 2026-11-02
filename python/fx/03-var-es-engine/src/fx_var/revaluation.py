"""Option revaluation under scenarios, Greek approximations and P&L explain.

Plain idea
----------
A VaR engine has to turn each scenario's factor moves into book P&L. For a
linear book that is a dot product. For options it is not: a delta-only
approximation misses the convexity, and a delta-gamma one misses the vol
move. *Full revaluation* reprices every option in every scenario; the
Taylor approximations are kept as a fast fallback and as the vocabulary of
*P&L explain*, which splits a day's actual P&L into delta, gamma, vega,
vanna, volga and theta pieces plus an unexplained residual.

Formal statement
----------------
Black-Scholes-Merton with a continuous carry yield ``b`` (dividend yield for
equity; the foreign rate for FX, which gives Garman-Kohlhagen 1983):

    V = s S e^{-b tau} N(s d1) - s K e^{-r tau} N(s d2),   s = +1 call, -1 put
    d1 = (ln(S/K) + (r - b + sigma^2/2) tau) / (sigma sqrt(tau)),  d2 = d1 - sigma sqrt(tau)

(Hull, *Options, Futures and Other Derivatives*, 11th ed., ch. 17 for the
price and ch. 19 for the Greeks). Under a scenario the spot moves by ``S -> S e^{x}`` (``x``
the factor log return), the vol by ``sigma -> sigma + dv``, and time by
``dt`` years. The second-order Taylor expansion used for the approximations
and the explain is

    dV ~ Delta dS + 1/2 Gamma dS^2 + Vega dv + Vanna dS dv + 1/2 Volga dv^2 + Theta dt

with ``Theta = -dV/dtau``. The residual of P&L explain is third order in
the moves, which is what the oracle test checks.

Vol scenario modes (``vol_mode`` of :func:`full_revaluation_pnl`)
-----------------------------------------------------------------
Plain idea: when spot moves, what happens to the implied vol of the
option you hold? Three standard answers (Derman 1999, "Regimes of
volatility", Risk 12(4), 55-59, for the first two):

* ``"sticky_strike"`` (default, the behaviour before round 4): the vol of
  a given strike does not move with spot. ``sigma_1 = sigma_0 + dv``.
* ``"sticky_delta"``: the smile moves with spot, so the vol depends on
  moneyness ``K / S`` only. With the leg's smile slope ``skew`` (the
  derivative of implied vol with respect to log-moneyness ``ln(K/S)`` at
  this strike, linear smile assumed), a spot move ``S -> S e^x`` changes
  the strike's log-moneyness by ``-x``, so
  ``sigma_1 = sigma_0 - skew * x + dv``. With ``skew = 0`` it equals
  sticky strike.
* ``"historical_atm"``: ``vol_shocks`` hold historical *log changes* of
  the ATM vol, ``dv = ln(sigma_ATM,t+h / sigma_ATM,t)``, and every strike's
  vol is scaled by the same factor, ``sigma_1 = sigma_0 e^{dv}``. This
  replays the proportional ATM moves of the history while keeping the
  smile shape (Alexander, *Market Risk Analysis* IV, 2008, s. 5.4, on
  relative vol shocks in historical simulation).

In every mode the shocked vol is floored at ``1e-12``. The compiled
engines (C++ and Rust, equity and FX) implement the same three formulas
and are checked against this module to 1e-12 relative.

Worked example: ``tests/test_revaluation.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

from . import _r3checks as chk

Kind = Literal["call", "put"]
VolMode = Literal["sticky_strike", "sticky_delta", "historical_atm"]
VOL_MODES: tuple[str, ...] = ("sticky_strike", "sticky_delta", "historical_atm")


@dataclass(frozen=True)
class OptionLeg:
    """European option on factor ``factor`` (a column of the scenario matrix).

    ``carry`` is the dividend yield (equity) or foreign rate (FX); all rates
    continuously compounded, ``tau`` in years, ``vol`` annualised. ``skew``
    is ``d sigma / d ln(K/S)`` at this strike, used only by
    ``vol_mode="sticky_delta"``.
    """

    quantity: float
    kind: Kind
    spot: float
    strike: float
    tau: float
    rate: float
    carry: float
    vol: float
    factor: int = 0
    skew: float = 0.0

    def __post_init__(self):
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")
        chk.scalar("quantity", self.quantity)
        chk.scalar("spot", self.spot, lo=0.0, strict=True)
        chk.scalar("strike", self.strike, lo=0.0, strict=True)
        chk.scalar("tau", self.tau, lo=0.0, strict=True)
        chk.scalar("rate", self.rate)
        chk.scalar("carry", self.carry)
        chk.scalar("vol", self.vol, lo=0.0, strict=True)
        chk.count("factor", self.factor, lo=0)
        chk.scalar("skew", self.skew)


@dataclass(frozen=True)
class LinearLeg:
    """Spot (or forward-like) position: value ``quantity * S``."""

    quantity: float
    spot: float
    factor: int = 0


Leg = OptionLeg | LinearLeg


def bsm_price(spot, strike, tau, rate, carry, vol, kind: Kind) -> np.ndarray:
    """Vectorised BSM price; ``spot``, ``vol`` and ``tau`` may be arrays.

    At ``tau <= 0`` returns intrinsic value.
    """
    s = 1.0 if kind == "call" else -1.0
    spot, vol, tau = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, dtype=float))
                                           for v in (spot, vol, tau)))
    out = np.maximum(s * (spot - strike), 0.0)
    live = tau > 0.0
    if live.any():
        S, v, t = spot[live], np.maximum(vol[live], 1e-12), tau[live]
        sq = v * np.sqrt(t)
        d1 = (np.log(S / strike) + (rate - carry + 0.5 * v * v) * t) / sq
        d2 = d1 - sq
        out[live] = s * (S * np.exp(-carry * t) * norm.cdf(s * d1)
                         - strike * np.exp(-rate * t) * norm.cdf(s * d2))
    return out


def bsm_greeks(leg: OptionLeg) -> dict[str, float]:
    """Closed-form BSM Greeks per unit quantity.

    ``vega``, ``vanna`` and ``volga`` are per 1.00 of vol (decimal); ``theta``
    is ``-dV/dtau`` per year.
    """
    S, K, t, r, b, v = leg.spot, leg.strike, leg.tau, leg.rate, leg.carry, leg.vol
    s = 1.0 if leg.kind == "call" else -1.0
    sq = v * np.sqrt(t)
    d1 = (np.log(S / K) + (r - b + 0.5 * v * v) * t) / sq
    d2 = d1 - sq
    eb, er, pdf = np.exp(-b * t), np.exp(-r * t), norm.pdf(d1)
    vega = S * eb * pdf * np.sqrt(t)
    theta = (-S * eb * pdf * v / (2.0 * np.sqrt(t))
             + s * b * S * eb * norm.cdf(s * d1) - s * r * K * er * norm.cdf(s * d2))
    return {
        "delta": float(s * eb * norm.cdf(s * d1)),
        "gamma": float(eb * pdf / (S * sq)),
        "vega": float(vega),
        "vanna": float(-eb * pdf * d2 / v),
        "volga": float(vega * d1 * d2 / v),
        "theta": float(theta),
    }


def _scenarios(legs: Sequence[Leg], spot_log_returns, vol_shocks):
    if not legs:
        raise ValueError("legs must not be empty")
    x = chk.array("spot_log_returns", spot_log_returns, ndim=2)
    nf = 1 + max(leg.factor for leg in legs)
    if x.shape[1] < nf:
        raise ValueError(f"scenario matrix has {x.shape[1]} factors, legs need {nf}")
    dv = np.zeros_like(x) if vol_shocks is None else chk.array("vol_shocks", vol_shocks, ndim=2)
    if dv.shape != x.shape:
        raise ValueError("vol_shocks must match spot_log_returns in shape")
    return x, dv


def shocked_vol(leg: OptionLeg, x, dv, vol_mode: VolMode = "sticky_strike") -> np.ndarray:
    """Scenario vol of ``leg`` given its factor's log return ``x`` and vol shock ``dv``.

    See the module note for the three modes; floored at ``1e-12``.
    """
    x = np.asarray(x, dtype=float)
    dv = np.asarray(dv, dtype=float)
    if vol_mode == "sticky_strike":
        v1 = leg.vol + dv
    elif vol_mode == "sticky_delta":
        v1 = leg.vol - leg.skew * x + dv
    elif vol_mode == "historical_atm":
        v1 = leg.vol * np.exp(dv)
    else:
        raise ValueError(f"vol_mode must be one of {VOL_MODES}, got {vol_mode!r}")
    return np.maximum(v1, 1e-12)


def full_revaluation_pnl(legs: Sequence[Leg], spot_log_returns, vol_shocks=None,
                         dt: float = 0.0, *, vol_mode: VolMode = "sticky_strike") -> np.ndarray:
    """Exact book P&L per scenario by repricing every leg.

    ``spot_log_returns`` and ``vol_shocks`` are ``(n_scenarios, n_factors)``;
    ``dt`` (years) ages every option, floored at expiry. ``vol_mode`` picks
    how the scenario vol is formed (module note); the default reproduces the
    pre-round-4 behaviour exactly.
    """
    if vol_mode not in VOL_MODES:
        raise ValueError(f"vol_mode must be one of {VOL_MODES}, got {vol_mode!r}")
    x, dv = _scenarios(legs, spot_log_returns, vol_shocks)
    dt = chk.scalar("dt", dt, lo=0.0)
    pnl = np.zeros(x.shape[0])
    for leg in legs:
        s1 = leg.spot * np.exp(x[:, leg.factor])
        if isinstance(leg, LinearLeg):
            pnl += leg.quantity * (s1 - leg.spot)
            continue
        v1 = shocked_vol(leg, x[:, leg.factor], dv[:, leg.factor], vol_mode)
        p0 = bsm_price(leg.spot, leg.strike, leg.tau, leg.rate, leg.carry, leg.vol, leg.kind)
        p1 = bsm_price(s1, leg.strike, max(leg.tau - dt, 0.0), leg.rate, leg.carry, v1, leg.kind)
        pnl += leg.quantity * (p1 - p0)
    return pnl


_TERMS = ("delta", "gamma", "vega", "vanna", "volga", "theta")


def _terms(leg: Leg, ds, dv, dt) -> dict[str, np.ndarray]:
    zero = np.zeros_like(ds)
    if isinstance(leg, LinearLeg):
        return {"delta": leg.quantity * ds, **{k: zero for k in _TERMS[1:]}}
    g = bsm_greeks(leg)
    q = leg.quantity
    return {
        "delta": q * g["delta"] * ds,
        "gamma": q * 0.5 * g["gamma"] * ds * ds,
        "vega": q * g["vega"] * dv,
        "vanna": q * g["vanna"] * ds * dv,
        "volga": q * 0.5 * g["volga"] * dv * dv,
        "theta": q * g["theta"] * dt + zero,
    }


def taylor_pnl(legs: Sequence[Leg], spot_log_returns, vol_shocks=None, dt: float = 0.0,
               *, order: Literal["delta", "delta_gamma", "full"] = "delta_gamma") -> np.ndarray:
    """Greek-approximated P&L per scenario.

    ``order="delta"`` keeps the delta term; ``"delta_gamma"`` adds gamma;
    ``"full"`` uses every second-order term plus theta.
    """
    keep = {"delta": _TERMS[:1], "delta_gamma": _TERMS[:2], "full": _TERMS}.get(order)
    if keep is None:
        raise ValueError(f"order must be 'delta', 'delta_gamma' or 'full', got {order!r}")
    x, dv = _scenarios(legs, spot_log_returns, vol_shocks)
    dt = chk.scalar("dt", dt, lo=0.0)
    pnl = np.zeros(x.shape[0])
    for leg in legs:
        ds = leg.spot * np.expm1(x[:, leg.factor])
        t = _terms(leg, ds, dv[:, leg.factor], dt)
        pnl += sum(t[k] for k in keep)
    return pnl


def pnl_explain(legs: Sequence[Leg], spot_log_returns, vol_shocks=None,
                dt: float = 0.0) -> dict[str, float]:
    """Split one day's actual P&L into Greek terms plus ``unexplained``.

    ``spot_log_returns`` and ``vol_shocks`` are one row per factor (1-D).
    Returns the six terms, ``explained`` (their sum), ``actual`` (full
    revaluation) and ``unexplained = actual - explained``.
    """
    x = np.atleast_2d(chk.array("spot_log_returns", spot_log_returns))
    dv = None if vol_shocks is None else np.atleast_2d(chk.array("vol_shocks", vol_shocks))
    x, dvm = _scenarios(legs, x, dv)
    dt = chk.scalar("dt", dt, lo=0.0)
    out = dict.fromkeys(_TERMS, 0.0)
    for leg in legs:
        ds = leg.spot * np.expm1(x[:, leg.factor])
        for k, val in _terms(leg, ds, dvm[:, leg.factor], dt).items():
            out[k] += float(val[0])
    out["explained"] = sum(out[k] for k in _TERMS)
    out["actual"] = float(full_revaluation_pnl(legs, x, dvm, dt)[0])
    out["unexplained"] = out["actual"] - out["explained"]
    return out
