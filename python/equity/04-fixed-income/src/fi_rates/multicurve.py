"""OIS discounting with a separate projection curve (multi-curve).

Since 2008 a collateralised swap is discounted on the overnight (OIS) curve
while its floating leg is projected on the curve of its own index tenor
(Ametrano and Bianchetti 2013, "Everything you always wanted to know about
multiple interest rate curve bootstrapping but were afraid to ask", SSRN
2219548, Sections 2-3).

Bootstrap, in this order:

1. OIS curve from OIS par rates. A compounded-overnight leg reset over
   ``[0, T]`` is worth ``1 - D(T)`` (ibid., eq. 3.6), so the par rate is
   ``R = (1 - D(T_n)) / sum_i tau_i D(T_i)``.
2. Projection curve ``P`` from par swap rates on the index, with the float
   leg projected on ``P`` and everything discounted on ``D``:

       sum_j tau_j D(t_j) (P(t_{j-1}) / P(t_j) - 1) = R * sum_i tau_i D(T_i).

Each new pillar is solved with Brent's method; dates between pillars use
log-linear discount factors (the ``DiscountCurve`` default). Accruals are
``1/frequency`` (the package's 30/360-style simplification).

These year-based functions are kept for the teaching notebooks. Dated
curves (real schedules, ACT/360 accruals, payment lags, futures, FRAs) use
:func:`bootstrap_projection_dated` here and
:func:`fi_rates.bootstrap.bootstrap_ois_dated`; those are checked against
QuantLib to 1e-12 in discount factor (``tests/test_bootstrap_dated.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq

from .bootstrap import DatedBootstrapResult, DatedQuote, DfSource, bootstrap_dated
from .curve import DiscountCurve


def _schedule(maturity: float, frequency: int) -> np.ndarray:
    n = maturity * frequency
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"maturity {maturity} * frequency {frequency} must be integral")
    return np.arange(1, int(round(n)) + 1, dtype=float) / frequency


def _check_quotes(maturities: Sequence[float], rates: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(maturities, dtype=float)
    r = np.asarray(rates, dtype=float)
    if m.shape != r.shape or m.size == 0:
        raise ValueError("maturities and rates must be non-empty and equal length")
    if np.any(np.diff(m) <= 0) or m[0] <= 0:
        raise ValueError("maturities must be positive and strictly increasing")
    if not np.all(np.isfinite(r)):
        raise ValueError("rates must be finite")
    return m, r


def _extend(times: list[float], dfs: list[float], t: float, x: float) -> DiscountCurve:
    return DiscountCurve(np.array([*times, t]), np.array([*dfs, x]))


def ois_par_rate(curve: DiscountCurve, maturity: float, frequency: int = 1) -> float:
    t = _schedule(maturity, frequency)
    d = np.asarray(curve.df(t))
    return float((1.0 - d[-1]) / (np.sum(d) / frequency))


def bootstrap_ois(
    maturities: Sequence[float], rates: Sequence[float], frequency: int = 1
) -> DiscountCurve:
    """OIS discount curve repricing every OIS par quote."""
    m, r = _check_quotes(maturities, rates)
    times: list[float] = []
    dfs: list[float] = []
    for T, R in zip(m, r, strict=False):
        def f(x: float, T: float = T, R: float = R) -> float:
            return ois_par_rate(_extend(times, dfs, T, x), T, frequency) - R
        dfs.append(brentq(f, 1e-6, 5.0, xtol=1e-15))
        times.append(T)
    return DiscountCurve(np.array(times), np.array(dfs))


def swap_par_rate(
    discount: DiscountCurve,
    projection: DiscountCurve,
    maturity: float,
    fixed_frequency: int = 1,
    float_frequency: int = 4,
) -> float:
    """Par fixed rate of a vanilla swap, float projected on ``projection``."""
    tf = _schedule(maturity, float_frequency)
    p = np.asarray(projection.df(tf))
    p_prev = np.concatenate(([1.0], p[:-1]))
    float_pv = float(np.sum(np.asarray(discount.df(tf)) * (p_prev / p - 1.0)))
    tx = _schedule(maturity, fixed_frequency)
    annuity = float(np.sum(np.asarray(discount.df(tx)))) / fixed_frequency
    return float_pv / annuity


def bootstrap_projection(
    discount: DiscountCurve,
    maturities: Sequence[float],
    rates: Sequence[float],
    fixed_frequency: int = 1,
    float_frequency: int = 4,
) -> DiscountCurve:
    """Projection (pseudo-discount) curve for one index tenor, OIS-discounted."""
    m, r = _check_quotes(maturities, rates)
    times: list[float] = []
    dfs: list[float] = []
    for T, R in zip(m, r, strict=False):
        def f(x: float, T: float = T, R: float = R) -> float:
            return swap_par_rate(discount, _extend(times, dfs, T, x), T,
                                 fixed_frequency, float_frequency) - R
        dfs.append(brentq(f, 1e-6, 5.0, xtol=1e-15))
        times.append(T)
    return DiscountCurve(np.array(times), np.array(dfs))


def ois_ibor_basis(discount: DiscountCurve, projection: DiscountCurve,
                   t1: float, t2: float) -> float:
    """Simple-forward spread of the index over OIS on ``[t1, t2]``."""
    return projection.simple_forward_rate(t1, t2) - discount.simple_forward_rate(t1, t2)


def bootstrap_projection_dated(
    quotes: Sequence[DatedQuote], discount: DfSource
) -> DatedBootstrapResult:
    """Dated projection curve for one index tenor, discounted on ``discount``.

    Plain idea: after 2008 a collateralised swap's cash flows are discounted
    on the overnight (OIS) curve, while a term index (3M, 6M) has its own,
    higher, forward curve. Formal statement: with the OIS curve ``D`` fixed,
    each pillar of the projection curve ``P`` solves
    ``sum_j D(p_j) (P(s_j)/P(e_j) - 1) = K sum_i tau_i D(p_i)`` for swaps
    and ``(P(s)/P(e) - 1)/tau = R`` for deposits and FRAs (Ametrano and
    Bianchetti 2013, sec. 3; Henrard 2014, ch. 3). Worked example: the
    ``term3m`` block of ``tests/data/dated_bootstrap_usd.txt``, where the
    10Y pillar lands on QuantLib's 0.674649332561 to 1e-12.
    """
    return bootstrap_dated(quotes, discount=discount)
