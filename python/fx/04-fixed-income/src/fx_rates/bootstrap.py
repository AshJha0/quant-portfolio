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
- Cross-currency basis spread ``s(T)`` (decimal, e.g. -0.0025 = -25 bp):
  the *foreign* (base-currency) discount curve used for FX-linked cashflows
  is ``DF_f_adj(T) = DF_f(T) * exp(-s(T) * T)``, i.e. adjusted zero
  ``z_adj = z_f + s``.  Negative EURUSD basis (post-2008 norm) means
  ``s < 0``: EUR discount factors *rise*, market forwards sit above pure
  CIP forwards, and borrowing USD through the FX swap market is expensive.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .curve import DiscountCurve

__all__ = [
    "df_from_deposit",
    "deposit_rate_from_df",
    "par_swap_rate",
    "bootstrap_curve",
    "reprice_deposits",
    "reprice_swaps",
    "basis_adjusted_curve",
    "implied_basis_from_forwards",
    "curve_from_fx_forwards",
]

Quote = tuple[float, float]  # (time in years, rate/price)


def df_from_deposit(rate: float, tau: float) -> float:
    """Discount factor implied by a simple-interest deposit.

    ``DF = 1 / (1 + r * tau)`` with ``tau`` the accrual fraction in years.
    """
    if tau <= 0.0:
        raise ValueError(f"deposit accrual fraction must be > 0, got {tau}")
    df = 1.0 / (1.0 + rate * tau)
    if df <= 0.0:
        raise ValueError(f"deposit rate {rate} at tau {tau} implies non-positive DF")
    return df


def deposit_rate_from_df(df: float, tau: float) -> float:
    """Simple-interest deposit rate implied by a discount factor."""
    if tau <= 0.0 or df <= 0.0:
        raise ValueError("require tau > 0 and df > 0")
    return (1.0 / df - 1.0) / tau


def par_swap_rate(curve: DiscountCurve, maturity_years: int) -> float:
    """Par rate of an annual fixed-for-float swap of integer maturity.

    ``c = (1 - DF(n)) / sum_{i=1..n} DF(i)`` — the single-curve textbook
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
        Annual par swap quotes at integer maturities >= 2.  Par rates are
        linearly interpolated onto every integer year up to the longest
        quoted maturity, then discount factors are peeled off recursively:
        ``DF(n) = (1 - c_n * A_{n-1}) / (1 + c_n)``.

    Returns
    -------
    DiscountCurve
        Pillars at every deposit tenor plus every integer year 2..N.
        Quoted instruments reprice to machine precision (tested).
    """
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
        n_max = int(round(mats[-1]))
        grid = np.arange(2, n_max + 1, dtype=float)
        rates = np.interp(grid, mats, [r for _, r in swps])
        # DF(1): interpolate the deposit segment at t = 1 (exact if a 1y
        # deposit is quoted, which the synthetic generators always do).
        dep_curve = DiscountCurve(times, dfs, name=name)
        annuity = float(dep_curve.df(1.0))
        boot_t: list[float] = []
        boot_df: list[float] = []
        for n, c in zip(grid, rates):
            df_n = (1.0 - c * annuity) / (1.0 + c)
            if df_n <= 0.0:
                raise ValueError(f"bootstrap produced non-positive DF at {n}y (rate {c})")
            annuity += df_n
            boot_t.append(float(n))
            boot_df.append(df_n)
        # merge, dropping deposit pillars beyond the 2y point that would clash
        for t, d in zip(boot_t, boot_df):
            if t in times:
                raise ValueError(f"deposit and swap pillars clash at t = {t}")
        times = times + boot_t
        dfs = dfs + boot_df
        order = np.argsort(times)
        times = list(np.asarray(times)[order])
        dfs = list(np.asarray(dfs)[order])
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


def implied_basis_from_forwards(
    spot: float,
    domestic_curve: DiscountCurve,
    foreign_curve: DiscountCurve,
    forwards: Sequence[Quote],
) -> list[Quote]:
    """Back out the basis spread curve implied by market FX forwards.

    From ``F_mkt(T) = S * DF_f(T) * exp(-s T) / DF_d(T)``:
    ``s(T) = -(1/T) * ln( F_mkt(T) * DF_d(T) / (S * DF_f(T)) )``.

    Returns a list of ``(T, s)`` in decimal, one per forward quote.
    """
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot}")
    out: list[Quote] = []
    for t, f in sorted(forwards):
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
    on — it embeds the cross-currency basis by construction.
    """
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot}")
    quotes = sorted(forwards)
    times = [t for t, _ in quotes]
    dfs = [f * domestic_curve.df(t) / spot for t, f in quotes]
    return DiscountCurve(times, dfs, name=name)
