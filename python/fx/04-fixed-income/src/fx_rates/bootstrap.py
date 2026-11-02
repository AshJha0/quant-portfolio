"""Curve bootstrapping: deposits + par swaps per currency, and cross-currency
basis-adjusted foreign curves from FX forwards / basis-spread quotes.

Quote conventions
-----------------
- Deposits: simple (money-market) interest.  A deposit quote ``(tau, r)``
  with accrual fraction ``tau`` (years, already converted with the relevant
  day count, e.g. ACT/360) implies ``DF(tau) = 1 / (1 + r * tau)``.
- Par swaps: annual fixed leg with unit accrual fractions (documented
  simplification A3 in METHODOLOGY.md).  A par swap of integer maturity
  ``n`` with rate ``c`` satisfies ``1 = c * sum_{i=1..n} DF(i) + DF(n)``.
- Cross-currency basis spread ``s(T)`` (decimal, e.g. -0.0025 = -25 bp)
  is the par spread of a float-float cross-currency basis swap of
  maturity ``T``: the non-collateral leg pays its own floating rate plus
  ``s`` against the collateral (USD) leg flat, with notional exchanges at
  both ends. :func:`basis_discount_curve` bootstraps the non-collateral
  currency's discount curve so that every quoted swap prices at par, with
  floating rates projected off that currency's own curve and the USD leg
  worth par on the USD curve. Negative EURUSD basis (post-2008 norm) means
  EUR discount factors rise, market forwards sit above pure CIP forwards,
  and borrowing USD through the FX swap market is expensive.
- :func:`basis_adjusted_curve` keeps the older zero-spread shortcut
  ``z_adj = z_f + s``. Under it a parallel move of the foreign curve and of
  the basis are the same risk factor, so it is no longer the default.
"""


from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._validation import require_finite
from .curve import DiscountCurve

__all__ = [
    "basis_adjusted_curve",
    "basis_discount_curve",
    "basis_swap_par_spread",
    "bootstrap_curve",
    "curve_from_fx_forwards",
    "deposit_rate_from_df",
    "df_from_deposit",
    "implied_basis_from_forwards",
    "par_swap_rate",
    "reprice_deposits",
    "reprice_swaps",
]

Quote = tuple[float, float]  # (time in years, rate/price)


def df_from_deposit(rate: float, tau: float) -> float:
    """Discount factor implied by a simple-interest deposit.

    ``DF = 1 / (1 + r * tau)`` with ``tau`` the accrual fraction in years.
    """
    require_finite(rate=rate, tau=tau)
    if tau <= 0.0:
        raise ValueError(f"deposit accrual fraction must be > 0, got {tau}")
    df = 1.0 / (1.0 + rate * tau)
    if df <= 0.0:
        raise ValueError(f"deposit rate {rate} at tau {tau} implies non-positive DF")
    return df


def deposit_rate_from_df(df: float, tau: float) -> float:
    """Simple-interest deposit rate implied by a discount factor."""
    require_finite(df=df, tau=tau)
    if tau <= 0.0 or df <= 0.0:
        raise ValueError("require tau > 0 and df > 0")
    return (1.0 / df - 1.0) / tau


def par_swap_rate(curve: DiscountCurve, maturity_years: int) -> float:
    """Par rate of an annual fixed-for-float swap of integer maturity.

    ``c = (1 - DF(n)) / sum_{i=1..n} DF(i)``: the single-curve textbook
    identity (fixed leg PV = floating leg PV = 1 - DF(n)).
    """
    n = int(maturity_years)
    if n < 1 or n != maturity_years:
        raise ValueError(f"swap maturity must be a positive integer, got {maturity_years}")
    times = np.arange(1, n + 1, dtype=float)
    dfs = curve.df(times)
    return float((1.0 - dfs[-1]) / dfs.sum())


def bootstrap_curve(
    deposits: Sequence[Quote],
    swaps: Sequence[Quote],
    name: str = "",
) -> DiscountCurve:
    """Bootstrap a single-currency discount curve from deposits and par swaps.

    Parameters
    ----------
    deposits : sequence of (tau, rate)
        Money-market deposits, simple interest; must include a pillar at or
        beyond 1.0y so the swap annuity can start from a known DF(1).
    swaps : sequence of (maturity_years, par_rate)
        Annual par swap quotes at integer maturities >= 2. Each quote adds
        one pillar at its own maturity. Discount factors at unquoted integer
        years in between follow the curve's log-linear interpolation, and
        the new pillar is solved (Brent) so that the quoted swap prices at
        par on exactly that curve. No intermediate quotes are invented.

    Returns
    -------
    DiscountCurve
        Pillars at every deposit tenor plus every quoted swap maturity.
        Quoted instruments reprice to machine precision (tested).
    """
    from scipy.optimize import brentq

    deps = sorted((float(t), float(r)) for t, r in deposits)
    swps = sorted((float(t), float(r)) for t, r in swaps)
    if not deps:
        raise ValueError("at least one deposit quote is required")
    if len({t for t, _ in deps}) != len(deps):
        raise ValueError("duplicate deposit tenors")
    times = [t for t, _ in deps]
    dfs = [df_from_deposit(r, t) for t, r in deps]

    if swps:
        mats = [t for t, _ in swps]
        if any(m < 2 or abs(m - round(m)) > 1e-9 for m in mats):
            raise ValueError("swap maturities must be integers >= 2 (annual fixed leg)")
        if len(set(mats)) != len(mats):
            raise ValueError("duplicate swap maturities")
        if deps[-1][0] < 1.0:
            raise ValueError("need a deposit at tenor >= 1.0y to anchor the swap annuity")
        for m, c in swps:
            n = int(round(m))
            if n in times:
                raise ValueError(f"deposit and swap pillars clash at t = {n}")
            if n <= times[-1]:
                raise ValueError(f"swap maturity {n}y lies inside the deposit strip")
            require_finite(swap_rate=c)
            t_prev = times[-1]
            base = DiscountCurve(times, dfs, name=name)
            known = [k for k in range(1, n) if k <= t_prev]
            ann_known = float(np.sum(base.df(np.asarray(known, dtype=float)))) if known else 0.0
            ln_prev = float(np.log(base.df(t_prev)))
            gap = [k for k in range(1, n) if k > t_prev]

            def par_error(
                x: float,
                ln_prev: float = ln_prev,
                t_prev: float = t_prev,
                n: int = n,
                gap: list[int] = gap,
                c: float = c,
                ann_known: float = ann_known,
            ) -> float:
                lx = np.log(x)
                mid = sum(np.exp(ln_prev + (k - t_prev) / (n - t_prev) * (lx - ln_prev))
                          for k in gap)
                return c * (ann_known + mid + x) + x - 1.0

            lo, hi = 1e-8, 10.0
            if par_error(lo) * par_error(hi) > 0.0:
                raise ValueError(f"bootstrap cannot bracket DF at {n}y (rate {c})")
            times.append(float(n))
            dfs.append(float(brentq(par_error, lo, hi, xtol=1e-16, rtol=8.9e-16)))
    return DiscountCurve(times, dfs, name=name)


def reprice_deposits(curve: DiscountCurve, deposits: Sequence[Quote]) -> float:
    """Max absolute error (in rate) repricing deposit quotes off the curve."""
    errs = [
        abs(deposit_rate_from_df(curve.df(t), t) - r) for t, r in deposits
    ]
    return max(errs) if errs else 0.0


def reprice_swaps(curve: DiscountCurve, swaps: Sequence[Quote]) -> float:
    """Max absolute error (in rate) repricing par swap quotes off the curve."""
    errs = [abs(par_swap_rate(curve, int(round(t))) - r) for t, r in swaps]
    return max(errs) if errs else 0.0


# ---------------------------------------------------------------------- #
# cross-currency basis
# ---------------------------------------------------------------------- #
def basis_adjusted_curve(
    foreign_curve: DiscountCurve,
    basis_spreads: Sequence[Quote],
    name: str | None = None,
) -> DiscountCurve:
    """Foreign discount curve adjusted for the cross-currency basis.

    ``z_adj(t) = z_f(t) + s(t)`` with the spread ``s`` linearly interpolated
    in maturity (flat extrapolation at both ends).  Pillars are the union of
    the curve pillars and the spread tenors, so a zero spread reproduces the
    input curve *exactly* (tested identity).

    Parameters
    ----------
    basis_spreads : sequence of (T, s)
        Spread quotes in **decimal** (e.g. -0.0025 for -25 bp).
    """
    if not basis_spreads:
        return foreign_curve
    st = np.asarray([t for t, _ in basis_spreads], dtype=float)
    sv = np.asarray([s for _, s in basis_spreads], dtype=float)
    require_finite(basis_tenors=st, basis_spreads=sv)
    order = np.argsort(st)
    st, sv = st[order], sv[order]
    if np.any(st <= 0.0):
        raise ValueError("basis spread tenors must be > 0")
    if np.any(np.diff(st) <= 0.0):
        raise ValueError("duplicate basis spread tenors")
    times = np.unique(np.concatenate([foreign_curve.times, st]))
    z = foreign_curve.zero_rate(times) + np.interp(times, st, sv)
    label = name if name is not None else f"{foreign_curve.name}+basis"
    return DiscountCurve.from_zero_rates(times, z, name=label)


def _spread_at(basis_spreads: Sequence[Quote], times: np.ndarray) -> np.ndarray:
    st = np.asarray([t for t, _ in basis_spreads], dtype=float)
    sv = np.asarray([s for _, s in basis_spreads], dtype=float)
    require_finite(basis_tenors=st, basis_spreads=sv)
    order = np.argsort(st)
    st, sv = st[order], sv[order]
    if np.any(st <= 0.0):
        raise ValueError("basis spread tenors must be > 0")
    if np.any(np.diff(st) <= 0.0):
        raise ValueError("duplicate basis spread tenors")
    return np.interp(times, st, sv)


def basis_discount_curve(
    projection_curve: DiscountCurve,
    basis_spreads: Sequence[Quote],
    frequency: int = 4,
    name: str | None = None,
) -> DiscountCurve:
    """Discount curve of the non-collateral currency from basis-swap quotes.

    Each quote ``(T, s)`` is the par spread of a float-float cross-currency
    basis swap with ``frequency`` resets a year and notional exchanges at
    0 and ``T``. Per unit notional the non-collateral leg is worth

        PV(T) = sum_i (F_i + s) tau D(t_i) + D(T) - 1

    where ``F_i tau = P(t_{i-1}) / P(t_i) - 1`` is projected off
    ``projection_curve`` (``P``) and ``D`` is the discount curve being
    built. The collateral (USD) leg is a floater discounted on its own
    curve and is worth zero, so par means ``PV(T) = 0``. Spreads are
    linearly interpolated onto the reset grid (flat beyond the ends) and
    ``D`` is peeled off one grid point at a time:

        D(t_n) = (1 - sum_{i<n} (F_i + s_n) tau D(t_i)) / (1 + (F_n + s_n) tau).

    Between grid points ``ln(D / P)`` is linear in ``t``, so with zero
    spreads ``D == P`` at every time.
    """
    if not basis_spreads:
        return projection_curve
    if frequency < 1 or int(frequency) != frequency:
        raise ValueError(f"frequency must be a positive integer, got {frequency}")
    t_max = max(float(t) for t, _ in basis_spreads)
    n = int(np.ceil(t_max * frequency - 1e-9))
    grid = np.arange(1, n + 1, dtype=float) / frequency
    tau = 1.0 / frequency
    s = _spread_at(basis_spreads, grid)
    p = np.asarray(projection_curve.df(grid))
    p_prev = np.concatenate(([1.0], p[:-1]))
    fwd_tau = p_prev / p - 1.0
    d = np.empty_like(grid)
    for k in range(n):
        acc = float(np.sum((fwd_tau[:k] + s[k] * tau) * d[:k]))
        d[k] = (1.0 - acc) / (1.0 + fwd_tau[k] + s[k] * tau)
        if d[k] <= 0.0:
            raise ValueError(f"basis bootstrap produced non-positive DF at t={grid[k]}")
    log_ratio = np.log(d / p)
    times = np.unique(np.concatenate([projection_curve.times, grid]))
    lr = np.interp(times, np.concatenate(([0.0], grid)), np.concatenate(([0.0], log_ratio)))
    beyond = times > grid[-1]
    if np.any(beyond) and n >= 2:
        slope = (log_ratio[-1] - log_ratio[-2]) / tau
        lr = np.where(beyond, log_ratio[-1] + slope * (times - grid[-1]), lr)
    dfs = np.asarray(projection_curve.df(times)) * np.exp(lr)
    label = name if name is not None else f"{projection_curve.name}+xccy"
    return DiscountCurve(times, dfs, name=label)


def basis_swap_par_spread(
    projection_curve: DiscountCurve,
    discount_curve: DiscountCurve,
    maturity: float,
    frequency: int = 4,
) -> float:
    """Par spread of a float-float basis swap given both non-collateral curves.

    ``s = (1 - D(T) - sum_i F_i tau D(t_i)) / (tau sum_i D(t_i))``.
    """
    require_finite(maturity=maturity)
    n = maturity * frequency
    if maturity <= 0.0 or abs(n - round(n)) > 1e-9:
        raise ValueError("maturity * frequency must be a positive integer")
    grid = np.arange(1, int(round(n)) + 1, dtype=float) / frequency
    tau = 1.0 / frequency
    p = np.asarray(projection_curve.df(grid))
    p_prev = np.concatenate(([1.0], p[:-1]))
    d = np.asarray(discount_curve.df(grid))
    float_leg = float(np.sum((p_prev / p - 1.0) * d))
    return (1.0 - d[-1] - float_leg) / (tau * float(np.sum(d)))


def implied_basis_from_forwards(
    spot: float,
    domestic_curve: DiscountCurve,
    foreign_curve: DiscountCurve,
    forwards: Sequence[Quote],
) -> list[Quote]:
    """Zero-spread-equivalent basis implied by market FX forwards.

    This inverts the zero-spread definition. It differs from the float-float
    par spread (``basis_swap_par_spread``) by second-order annuity terms.

    From ``F_mkt(T) = S * DF_f(T) * exp(-s T) / DF_d(T)``:
    ``s(T) = -(1/T) * ln( F_mkt(T) * DF_d(T) / (S * DF_f(T)) )``.

    Returns a list of ``(T, s)`` in decimal, one per forward quote.
    """
    require_finite(spot=spot)
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot}")
    out: list[Quote] = []
    for t, f in sorted(forwards):
        require_finite(forward_tenor=t, forward_quote=f)
        if t <= 0.0 or f <= 0.0:
            raise ValueError(f"invalid forward quote (T={t}, F={f})")
        cip = spot * foreign_curve.df(t) / domestic_curve.df(t)
        out.append((float(t), float(-np.log(f / cip) / t)))
    return out


def curve_from_fx_forwards(
    spot: float,
    domestic_curve: DiscountCurve,
    forwards: Sequence[Quote],
    name: str = "implied-foreign",
) -> DiscountCurve:
    """Basis-adjusted foreign discount curve bootstrapped from FX forwards.

    Inverts covered interest parity quote by quote:
    ``DF_f_adj(T) = F_mkt(T) * DF_d(T) / S``.
    This is the curve a desk actually discounts foreign FX-linked cashflows
    on, it embeds the cross-currency basis by construction.
    """
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot}")
    quotes = sorted(forwards)
    times = [t for t, _ in quotes]
    dfs = [f * domestic_curve.df(t) / spot for t, f in quotes]
    return DiscountCurve(times, dfs, name=name)
