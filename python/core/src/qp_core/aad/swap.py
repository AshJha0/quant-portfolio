"""Swap PV off the dated bootstrap, with adjoint curve and quote Greeks.

Plain idea. A swap's PV depends on the curve's pillar discount factors, and
those depend on the market quotes the bootstrap reprices. The adjoint of the
PV gives ``dPV / d ln P_j`` for every pillar in one backward sweep. To get
``dPV / d quote_k`` without re-running the bootstrap once per quote, use the
fact that the built curve solves ``R(x, q) = 0`` (each quote's model rate
minus its quote): by the implicit function theorem ``dx/dq = J^{-1}`` with
``J = dR/dx``, so the quote Greeks are ``lambda = J^{-T} g`` where
``g = dPV/dx``. Each row of ``J`` is one adjoint sweep through one quote's
residual; the solve is a small dense system (Henrard 2014, *Interest Rate
Modelling in the Multi-Curve Framework*, ch. 4; Savine 2018, s. 14.4).

Formal statement. ``x_j = ln P(t_j)`` on the pillar days of
``fi_rates.bootstrap.DatedCurve`` (log-linear in ``ln P`` against ACT/365F
time, flat forward beyond the last pillar; the interpolation is copied
here so it can run on tape variables). The swap is a payer by default:
``PV = N (sum_float P(pay)(P(start)/P(end) - 1) - K sum_fixed tau P(pay))``.
For a self-discounted OIS curve the residual of a forward quote is
``(P(s)/P(e) - 1)/tau - rate`` and of a swap quote ``float/annuity - rate``,
the expressions of ``fi_rates.bootstrap.dated_quote_residual``. Since
``dR/dq = -I``, ``dPV/dq = g^T J^{-1}``.

Worked example: the 5Y payer swap at 4 % on the QuantLib USD OIS fixture;
pillar and quote Greeks match central bumps (bumped ``ln P`` and rebuilt
curves) to 1e-6 relative (``tests/test_aad_python.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from qp_core.model import model

from .tape import Real, Tape, Var, exp, value

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class Swap:
    """Fixed-against-float swap; periods duck-type ``fi_rates`` periods."""

    fixed: tuple[Any, ...]
    floating: tuple[Any, ...]
    rate: float
    notional: float = 1.0
    payer: bool = True


def log_df(days: Sequence[int], xs: Sequence[Real], day: float) -> Real:
    """``ln P(day)``: ``xs[j] = ln P(days[j])``; the node at day 0 is implicit."""
    t = day / DAYS_PER_YEAR
    ts = [0.0] + [d / DAYS_PER_YEAR for d in days]
    xx: list[Real] = [0.0, *xs]
    n = len(ts) - 1
    if t >= ts[n]:
        return xx[n] + (xx[n] - xx[n - 1]) * ((t - ts[n]) / (ts[n] - ts[n - 1]))
    lo, hi = 0, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ts[mid] <= t:
            lo = mid
        else:
            hi = mid
    w = (t - ts[lo]) / (ts[hi] - ts[lo])
    return xx[lo] + (xx[hi] - xx[lo]) * w


def df(days: Sequence[int], xs: Sequence[Real], day: float) -> Real:
    return exp(log_df(days, xs, day))


def swap_pv(days: Sequence[int], xs: Sequence[Real], swap: Swap) -> Real:
    flt: Real = 0.0
    for p in swap.floating:
        flt = flt + df(days, xs, p.pay) * (df(days, xs, p.start) / df(days, xs, p.end) - 1.0)
    ann: Real = 0.0
    for f in swap.fixed:
        ann = ann + df(days, xs, f.pay) * f.accrual
    pv = (flt - ann * swap.rate) * swap.notional
    return pv if swap.payer else -pv


def quote_residual(q: Any, days: Sequence[int], xs: Sequence[Real]) -> Real:
    """Model minus quoted rate on a self-discounted, self-projected curve."""
    if hasattr(q, "floating"):
        flt: Real = 0.0
        for p in q.floating:
            flt = flt + df(days, xs, p.pay) * (df(days, xs, p.start) / df(days, xs, p.end) - 1.0)
        ann: Real = 0.0
        for f in q.fixed:
            ann = ann + df(days, xs, f.pay) * f.accrual
        return flt / ann - float(q.rate)
    return (df(days, xs, q.start) / df(days, xs, q.end) - 1.0) / float(q.accrual) - float(q.rate)


class PillarGreeks(NamedTuple):
    pv: float
    d_log_df: tuple[float, ...]


@model(id="CORE-AAD-SWAP")
def pillar_greeks(days: Sequence[int], dfs: Sequence[float], swap: Swap) -> PillarGreeks:
    """PV and ``dPV / d ln P(days[j])`` from one backward sweep."""
    tape = Tape()
    xs = [tape.var(float(np.log(d))) for d in dfs]
    pv = swap_pv(days, xs, swap)
    if not isinstance(pv, Var):  # pragma: no cover - a swap always touches the curve
        return PillarGreeks(float(pv), tuple(0.0 for _ in xs))
    return PillarGreeks(pv.value, tuple(tape.gradient(pv, xs)))


def residual_jacobian(quotes: Sequence[Any], days: Sequence[int],
                      dfs: Sequence[float]) -> np.ndarray:
    """``J[k, j] = d R_k / d ln P(days[j])``, one adjoint sweep per quote."""
    tape = Tape()
    xs = [tape.var(float(np.log(d))) for d in dfs]
    mark = tape.mark()
    jac = np.zeros((len(quotes), len(xs)))
    for k, q in enumerate(quotes):
        r = quote_residual(q, days, xs)
        if isinstance(r, Var):
            jac[k] = tape.gradient(r, xs)
        tape.rewind(mark)
    return jac


class QuoteGreeks(NamedTuple):
    pv: float
    d_quote: tuple[float, ...]
    d_log_df: tuple[float, ...]


@model(id="CORE-AAD-SWAP")
def quote_greeks(quotes: Sequence[Any], days: Sequence[int], dfs: Sequence[float],
                 swap: Swap) -> QuoteGreeks:
    """``dPV / d quote_k`` (in quote order) through the implicit function theorem.

    ``days``/``dfs`` must be the curve the bootstrap built from ``quotes``
    (one pillar per quote).
    """
    if len(quotes) != len(days):
        raise ValueError("need one quote per pillar")
    pg = pillar_greeks(days, dfs, swap)
    jac = residual_jacobian(quotes, days, dfs)
    lam = np.linalg.solve(jac.T, np.asarray(pg.d_log_df))
    return QuoteGreeks(pg.pv, tuple(float(v) for v in lam), pg.d_log_df)


def pv_float(days: Sequence[int], dfs: Sequence[float], swap: Swap) -> float:
    """The plain-float PV (the denominator of the cost ratio)."""
    return value(swap_pv(days, [float(np.log(d)) for d in dfs], swap))
