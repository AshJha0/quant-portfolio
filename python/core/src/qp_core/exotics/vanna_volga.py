"""Vanna-volga smile adjustment for vanillas and barrier options.

Plain idea
----------
Price the option at the ATM vol, then add the cost of hedging its vega,
vanna and volga with three liquid vanillas (25-delta put, ATM, 25-delta
call) bought at their *market* vols. For a vanilla this reproduces the
three market prices exactly and interpolates the smile in between. For a
barrier the hedge is only needed while the option is alive, so desks scale
the hedge cost by the probability the barrier is not hit.

Formal statement
----------------
Castagna and Mercurio (2007), "The vanna-volga method for implied
volatilities", Risk, January, eqs. 4-6. With ``O`` the target, ``C_i`` the
three pillar calls and every Black-Scholes number evaluated at the ATM vol
``sigma_2``, choose weights ``x`` with

    [vega, vanna, volga](C_i) x = [vega, vanna, volga](O)

and set ``O_VV = O_BS + sum_i x_i (C_i^mkt - C_i^BS)``. For knock-outs,
``O_VV = O_BS + p * sum_i x_i (C_i^mkt - C_i^BS)`` where ``p`` is the
risk-neutral survival probability at the ATM vol (Wystup 2006, *FX
Options and Structured Products*, s.1.5.4; Bossens, Rayee, Skantzos and
Deelstra 2010, "Vanna-volga methods applied to FX derivatives", IJTAF
13(8), the "survival-probability weighted" variant). Knock-ins follow
from in-out parity with the vanna-volga vanilla, which keeps the parity
exact.

Greeks of the barrier are taken by central finite differences of the
closed form; for pillar vanillas they are analytic.

Worked example
--------------
With a flat smile the adjustment is zero:

>>> args = (1.30, 1.30, 1.20, 0.5, 0.02, 0.01)
>>> pk, pv = (1.25, 1.30, 1.35), (0.10, 0.10, 0.10)
>>> from qp_core.exotics.barriers import barrier_price
>>> bs = barrier_price("down_out", "call", 1.30, 1.30, 1.20, 0.5, 0.02, 0.01, 0.10)
>>> abs(vanna_volga_barrier("down_out", "call", *args, pk, pv) - bs) < 1e-12
True
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from ..normal import norm_pdf
from .barriers import barrier_price, touch_price, vanilla_price

__all__ = ["vanna_volga_barrier", "vanna_volga_vanilla", "vv_weights"]


def _bs_vvv(S: float, K: float, T: float, rd: float, rf: float, sig: float) -> np.ndarray:
    """Analytic vega, vanna (d vega / dS) and volga (d vega / d sigma) of a GK call."""
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (rd - rf + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    vega = S * math.exp(-rf * T) * norm_pdf(d1) * math.sqrt(T)
    vanna = -math.exp(-rf * T) * norm_pdf(d1) * d2 / sig
    volga = vega * d1 * d2 / sig
    return np.array([vega, vanna, volga])


def _fd_vvv(f: Callable[[float, float], float], S: float, sig: float) -> np.ndarray:
    hs = 1e-4 * S
    hv = 1e-4
    vega = (f(S, sig + hv) - f(S, sig - hv)) / (2 * hv)
    volga = (f(S, sig + hv) - 2 * f(S, sig) + f(S, sig - hv)) / (hv * hv)
    vanna = (f(S + hs, sig + hv) - f(S + hs, sig - hv) - f(S - hs, sig + hv) + f(S - hs, sig - hv)) / (
        4 * hs * hv
    )
    return np.array([vega, vanna, volga])


def vv_weights(
    target: np.ndarray, S: float, T: float, rd: float, rf: float,
    pillar_strikes: Sequence[float], sig_atm: float,
) -> np.ndarray:
    """Weights of the three pillar calls that match the target's vega, vanna and volga."""
    A = np.column_stack([_bs_vvv(S, K, T, rd, rf, sig_atm) for K in pillar_strikes])
    return np.linalg.solve(A, target)


def _check_pillars(pillar_strikes: Sequence[float], pillar_vols: Sequence[float]) -> None:
    if len(pillar_strikes) != 3 or len(pillar_vols) != 3:
        raise ValueError("vanna-volga needs exactly three pillar strikes and vols")
    k = list(pillar_strikes)
    if not (0 < k[0] < k[1] < k[2]) or min(pillar_vols) <= 0:
        raise ValueError("pillar strikes must be increasing and positive, vols positive")


def _hedge_cost(
    target: np.ndarray, S: float, T: float, rd: float, rf: float,
    pillar_strikes: Sequence[float], pillar_vols: Sequence[float],
) -> float:
    sig_atm = pillar_vols[1]
    x = vv_weights(target, S, T, rd, rf, pillar_strikes, sig_atm)
    cost = 0.0
    for xi, K, sv in zip(x, pillar_strikes, pillar_vols, strict=True):
        cost += xi * (vanilla_price("call", S, K, T, rd, rd - rf, sv)
                      - vanilla_price("call", S, K, T, rd, rd - rf, sig_atm))
    return float(cost)


def vanna_volga_vanilla(
    cp: str, S: float, X: float, T: float, rd: float, rf: float,
    pillar_strikes: Sequence[float], pillar_vols: Sequence[float],
) -> float:
    """Vanna-volga price of a vanilla (Castagna-Mercurio 2007)."""
    _check_pillars(pillar_strikes, pillar_vols)
    sig = pillar_vols[1]
    target = _bs_vvv(S, X, T, rd, rf, sig)  # put and call share vega, vanna, volga
    return vanilla_price(cp, S, X, T, rd, rd - rf, sig) + _hedge_cost(
        target, S, T, rd, rf, pillar_strikes, pillar_vols)


def vanna_volga_barrier(
    kind: str, cp: str, S: float, X: float, H: float, T: float, rd: float, rf: float,
    pillar_strikes: Sequence[float], pillar_vols: Sequence[float],
    rebate: float = 0.0, survival_weighted: bool = True,
) -> float:
    """Vanna-volga price of a single-barrier option.

    Knock-outs: BS price at the ATM vol plus the survival-weighted hedge
    cost (``survival_weighted=False`` gives the unweighted, full hedge
    cost, which overprices the smile effect near the barrier). Knock-ins:
    the vanna-volga vanilla minus the vanna-volga knock-out, plus the
    knock-in rebate leg at the ATM vol.
    """
    _check_pillars(pillar_strikes, pillar_vols)
    if kind.endswith("in"):
        ko = vanna_volga_barrier(kind.replace("in", "out"), cp, S, X, H, T, rd, rf,
                                 pillar_strikes, pillar_vols, 0.0, survival_weighted)
        reb = 0.0
        if rebate > 0.0:
            sfx = "down" if kind.startswith("down") else "up"
            reb = touch_price(f"no_touch_{sfx}", S, H, T, rd, rd - rf, pillar_vols[1], cash=rebate)
        return vanna_volga_vanilla(cp, S, X, T, rd, rf, pillar_strikes, pillar_vols) - ko + reb
    sig = pillar_vols[1]
    b = rd - rf

    def f(s: float, v: float) -> float:
        return barrier_price(kind, cp, s, X, H, T, rd, b, v, rebate)

    bs = f(S, sig)
    down = kind.startswith("down")
    if (down and S <= H) or ((not down) and S >= H):
        return bs
    target = _fd_vvv(f, S, sig)
    cost = _hedge_cost(target, S, T, rd, rf, pillar_strikes, pillar_vols)
    if survival_weighted:
        sfx = "down" if down else "up"
        p = touch_price(f"no_touch_{sfx}", S, H, T, rd, b, sig) * math.exp(rd * T)
        cost *= p
    return bs + cost

