"""Par-rate (instrument-space) risk of a bootstrapped curve (W14).

The plain idea
--------------
``keyrates.py`` and ``DiscountCurve.bumped_pillars`` bump zero rates at the
curve's pillars. A desk hedges with the instruments the curve was built
from, so it wants "PV change per 1bp move in the 5y swap rate", with the
curve rebuilt. Rebuilding once per quote is slow and noisy; the Jacobian of
the model quotes with respect to the pillar zero rates gives the same answer
from one set of pillar bumps.

The formal statement
--------------------
With pillar zero rates ``z`` and model quotes ``Q_i(z)`` (deposit simple
rate, FRA forward, par swap rate; see ``bootstrap._pv_error``), the
bootstrap solves ``Q(z) = q``. Then::

    J = dQ/dz                  (square: one pillar per instrument)
    dP/dq = J^{-T} dP/dz       (qp_core.risk.node_to_quote_risk)

and the par-rate DV01 of instrument ``i`` is ``dP/dq_i x 1e-4``. Andersen and
Piterbarg vol. 3 (2010) section 26.3. The reference is the "rebuild" route:
bump quote ``i`` by ``+-h``, bootstrap again and reprice
(:func:`par_rate_risk_by_rebuild`); tests require the two to agree to 1e-6
relative. Cross-gamma in quote space needs the curvature of the
bootstrap map as well as the node Hessian. Differentiating ``Q(z(q)) = q``
twice gives::

    Gamma_q = J^{-T} ( Gamma_z - sum_i g_q,i H_i ) J^{-1},   H_i = d^2 Q_i / dz dz

The tempting shortcut ``J^{-T} Gamma_z J^{-1}`` drops the ``H_i`` term. It
is not small: for the test book it is off by about 200% of the largest
entry, because the first-order risk ``g_q`` is large and par rates are
visibly non-linear in zero rates. The tests compare the full formula with
the Hessian of "bump two quotes, rebuild, reprice".

Worked example
--------------
A curve from a 1y deposit at 4% and a 2y annual par swap at 4.5%, and a
book that receives 4.5% fixed for 2y on 100: the par-rate DV01 to the 2y
swap rate is about ``-0.0188`` per bp (the receiver loses when the rate
rises), and to the 1y deposit it is zero to rounding (the book is the 2y
swap itself, which reprices at par whatever the 1y rate is). The test
``test_par_rate_risk_of_the_hedge_instrument_itself`` checks both.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Sequence

import numpy as np
from qp_core.risk import cross_gamma, node_to_quote_risk

from ..bootstrap import FRA, Deposit, Instrument, ParSwap, _validate, bootstrap_curve
from ..curve import DiscountCurve, ExtrapolationWarning

__all__ = [
    "model_quotes",
    "node_gamma",
    "node_risk",
    "par_rate_gamma",
    "par_rate_risk",
    "par_rate_risk_by_rebuild",
    "quote_jacobian",
    "with_rates",
]

PV = Callable[[DiscountCurve], float]


def with_rates(
    instruments: Sequence[Instrument], rates: Sequence[float] | np.ndarray
) -> list[Instrument]:
    """Copies of ``instruments`` (in the given order) quoting ``rates``."""
    out: list[Instrument] = []
    for ins, r in zip(instruments, rates, strict=True):
        if isinstance(ins, Deposit):
            out.append(Deposit(ins.maturity, float(r)))
        elif isinstance(ins, FRA):
            out.append(FRA(ins.start, ins.end, float(r)))
        else:
            out.append(ParSwap(ins.maturity, float(r), ins.frequency))
    return out


def model_quotes(instruments: Sequence[Instrument], curve: DiscountCurve) -> np.ndarray:
    """Model rate of each instrument off ``curve`` (same order as given).

    Same definitions as ``bootstrap._pv_error`` (a test pins the two to
    1e-15), with every discount factor read in one ``curve.df`` call: the
    risk functions call this a few hundred times per curve.
    """
    kinds, flat, starts, extra = _layout(tuple(instruments))
    dfs = np.ones_like(flat)
    pos = flat > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtrapolationWarning)
        dfs[pos] = np.asarray(curve.df(flat[pos]), dtype=float)
    ends = np.r_[starts[1:], flat.size] - 1
    first, last = dfs[starts], dfs[ends]
    sums = np.add.reduceat(dfs, starts)
    return np.where(
        kinds == 0, (1.0 / first - 1.0) / extra,
        np.where(kinds == 1, (first / last - 1.0) / extra, (1.0 - last) / (extra * sums)),
    )


@functools.lru_cache(maxsize=256)
def _layout(
    instruments: tuple[Instrument, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kinds (0 deposit, 1 FRA, 2 swap), the flat time grid, the start of
    each instrument's slice, and the scalar each formula divides by."""
    kinds, times, starts, extra = [], [], [], []
    k = 0
    for ins in instruments:
        if isinstance(ins, Deposit):
            t, kind, x = np.array([ins.maturity]), 0, ins.maturity
        elif isinstance(ins, FRA):
            t, kind, x = np.array([ins.start, ins.end]), 1, ins.end - ins.start
        else:
            t, kind, x = ins.payment_times(), 2, 1.0 / ins.frequency
        kinds.append(kind)
        starts.append(k)
        extra.append(x)
        times.append(t)
        k += t.size
    return (np.array(kinds), np.concatenate(times), np.array(starts), np.array(extra, dtype=float))


def _ordered(instruments: Sequence[Instrument], curve: DiscountCurve) -> list[Instrument]:
    ordered = _validate(list(instruments))
    pillars = np.array([i.pillar for i in ordered])
    if pillars.shape != curve.times.shape or not np.allclose(pillars, curve.times, rtol=0, atol=1e-12):
        raise ValueError("instrument pillars do not match the curve's pillar times")
    return ordered


def _check_calibrated(ordered: Sequence[Instrument], curve: DiscountCurve, tol: float) -> None:
    err = np.max(np.abs(model_quotes(ordered, curve) - np.array([i.rate for i in ordered])))
    if err > tol:
        raise ValueError(
            f"curve does not reprice its quotes (max error {err:.3g} > {tol:g}); "
            "the Jacobian identity dP/dq = J^-T dP/dz needs Q(z) = q. A "
            "sequential bootstrap with a non-local interpolation (pchip_zero, "
            "monotone_convex) moves earlier segments when later pillars are "
            "added; use par_rate_risk_by_rebuild or a global solve."
        )


def quote_jacobian(
    instruments: Sequence[Instrument], curve: DiscountCurve, h: float = 1e-6
) -> np.ndarray:
    """``J[i, j] = dQ_i / dz_j``, instruments sorted by pillar (as the
    bootstrap sorts them), central differences of pillar zero rates."""
    ordered = _ordered(instruments, curve)
    n = curve.times.size
    jac = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        up = model_quotes(ordered, curve.bumped_pillars(e))
        dn = model_quotes(ordered, curve.bumped_pillars(-e))
        jac[:, j] = (up - dn) / (2.0 * h)
    return jac


def node_risk(pv: PV, curve: DiscountCurve, h: float = 1e-6) -> np.ndarray:
    """``dP/dz_j`` per unit of pillar zero rate (central differences)."""
    n = curve.times.size
    out = np.empty(n)
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        out[j] = (pv(curve.bumped_pillars(e)) - pv(curve.bumped_pillars(-e))) / (2.0 * h)
    return out


def node_gamma(pv: PV, curve: DiscountCurve, h: float = 1e-4) -> np.ndarray:
    """Pillar zero-rate Hessian (cross-gamma), per unit rate squared."""
    return cross_gamma(lambda z: pv(curve.bumped_pillars(z)), np.zeros(curve.times.size), h)


def par_rate_risk(
    pv: PV, instruments: Sequence[Instrument], curve: DiscountCurve, h: float = 1e-6,
    calibration_tol: float = 1e-9, jacobian: np.ndarray | None = None,
) -> np.ndarray:
    """``dP/dq`` per unit quote, instruments sorted by pillar (Jacobian route).

    Multiply by ``1e-4`` for the par-rate DV01 per basis point. Raises
    ``ValueError`` if the curve does not reprice the quotes to
    ``calibration_tol`` (the identity then does not hold; measured on the
    test book: 0.7% error under pchip_zero, 1.8% under monotone_convex).
    """
    _check_calibrated(_ordered(instruments, curve), curve, calibration_tol)
    jac = quote_jacobian(instruments, curve, h) if jacobian is None else jacobian
    return node_to_quote_risk(node_risk(pv, curve, h), jac)


def _vector_hessian(
    f: Callable[[DiscountCurve], np.ndarray], curve: DiscountCurve, h: float
) -> np.ndarray:
    """``H[m, j, k] = d^2 f_m / dz_j dz_k`` over pillar zero bumps, with the
    stencil of ``qp_core.risk.cross_gamma`` (one curve per stencil point,
    shared by every component of ``f``)."""
    n = curve.times.size
    eye = np.eye(n) * h

    def at(z: np.ndarray) -> np.ndarray:
        return np.asarray(f(curve.bumped_pillars(z)), dtype=float)

    f0 = at(np.zeros(n))
    out = np.empty((f0.size, n, n))
    for i in range(n):
        out[:, i, i] = (at(eye[i]) - 2.0 * f0 + at(-eye[i])) / h**2
        for j in range(i + 1, n):
            v = (at(eye[i] + eye[j]) - at(eye[i] - eye[j]) - at(-eye[i] + eye[j])
                 + at(-eye[i] - eye[j])) / (4.0 * h**2)
            out[:, i, j] = out[:, j, i] = v
    return out


def par_rate_gamma(
    pv: PV, instruments: Sequence[Instrument], curve: DiscountCurve,
    h: float = 1e-4, jacobian: np.ndarray | None = None,
    quote_risk: np.ndarray | None = None,
) -> np.ndarray:
    """Quote-space cross-gamma ``J^{-T} (Gamma_z - sum_i g_q,i H_i) J^{-1}``."""
    ordered = _ordered(instruments, curve)
    _check_calibrated(ordered, curve, 1e-9)
    jac = quote_jacobian(ordered, curve) if jacobian is None else jacobian
    inv = np.linalg.inv(jac)
    g_q = node_to_quote_risk(node_risk(pv, curve), jac) if quote_risk is None else quote_risk
    hess = _vector_hessian(
        lambda c: np.concatenate(([pv(c)], model_quotes(ordered, c))), curve, h
    )
    correction = np.einsum("i,ijk->jk", g_q, hess[1:])
    return np.asarray(inv.T @ (hess[0] - correction) @ inv, dtype=float)


def par_rate_risk_by_rebuild(
    pv: PV, instruments: Sequence[Instrument], interpolation: str = "loglinear_df",
    h: float = 1e-6,
) -> np.ndarray:
    """Reference route: bump each quote, bootstrap again, reprice."""
    ordered = _validate(list(instruments))
    rates = np.array([i.rate for i in ordered])
    out = np.empty(rates.size)
    for i in range(rates.size):
        e = np.zeros(rates.size)
        e[i] = h
        up = pv(bootstrap_curve(with_rates(ordered, rates + e), interpolation))
        dn = pv(bootstrap_curve(with_rates(ordered, rates - e), interpolation))
        out[i] = (up - dn) / (2.0 * h)
    return out
