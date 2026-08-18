"""Bond pricing: fixed coupon, zero coupon, FRN, annuities; YTM and z-spread.

Conventions
-----------
* Prices are quoted **per 100 face** unless stated otherwise; PV functions
  taking a ``face`` return currency amounts.
* ``dirty = clean + accrued``.  Accrued interest uses the bond's own day
  count within the current coupon period.
* YTM is the **street convention** periodic yield compounded ``frequency``
  times per year: cashflow ``k`` periods after settlement (fractional first
  period ``w``) is discounted by ``(1 + y/m)^-(w + k - 1)`` where
  ``w = accrual fraction of the current period remaining``.
  At an exact coupon date, a par bond therefore has YTM == coupon exactly.
* Curve pricing maps payment dates to times via ACT/365F from settlement and
  applies a **continuously compounded z-spread**:
  ``PV = sum cf_i * P(t_i) * exp(-z * t_i)``.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .curve import DiscountCurve, ExtrapolationWarning
from .daycount import (
    SUPPORTED_FREQUENCIES,
    generate_schedule,
    year_fraction,
)

__all__ = [
    "FixedRateBond",
    "ZeroCouponBond",
    "bond_cashflows",
    "accrued_interest",
    "clean_price_from_curve",
    "dirty_price_from_curve",
    "price_from_ytm",
    "ytm_from_price",
    "z_spread_from_price",
    "zcb_price_from_curve",
    "frn_price_from_curve",
    "annuity_pv",
    "curve_time",
]

_CURVE_DCC = "ACT/365F"  # date -> curve-time mapping convention (documented)


def curve_time(settlement: dt.date, date: dt.date) -> float:
    """Map a calendar date to curve time: ACT/365F years from settlement."""
    return year_fraction(settlement, date, _CURVE_DCC)


@dataclass(frozen=True)
class FixedRateBond:
    """Fixed-coupon bullet bond.

    Parameters
    ----------
    effective : datetime.date
        Interest accrual start (dated date).
    maturity : datetime.date
        Redemption date.
    coupon : float
        Annual coupon rate, decimal (0.05 = 5%), paid ``frequency`` times/yr.
    frequency : int
        1, 2 or 4 coupons per year.
    daycount : str
        Accrual convention for coupons/accrued interest (default 30/360US,
        the US corporate/agency convention; use ACT/ACT-ISDA for Treasuries).
    face : float
        Redemption amount per bond (default 100).
    """

    effective: dt.date
    maturity: dt.date
    coupon: float
    frequency: int = 2
    daycount: str = "30/360US"
    face: float = 100.0

    def __post_init__(self) -> None:
        if self.frequency not in SUPPORTED_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {SUPPORTED_FREQUENCIES}, "
                f"got {self.frequency}"
            )
        if self.maturity <= self.effective:
            raise ValueError(
                f"maturity {self.maturity} must be after effective "
                f"{self.effective}"
            )
        if self.face <= 0:
            raise ValueError(f"face must be > 0, got {self.face}")
        if self.coupon < 0:
            raise ValueError(f"coupon must be >= 0, got {self.coupon}")

    def schedule(self) -> list[dt.date]:
        """Coupon payment dates in ``(effective, maturity]``."""
        return generate_schedule(self.effective, self.maturity, self.frequency)


@dataclass(frozen=True)
class ZeroCouponBond:
    """Zero-coupon bond redeeming ``face`` at ``maturity``."""

    maturity: dt.date
    face: float = 100.0

    def __post_init__(self) -> None:
        if self.face <= 0:
            raise ValueError(f"face must be > 0, got {self.face}")


def _check_settlement(settlement: dt.date, maturity: dt.date) -> None:
    if settlement > maturity:
        raise ValueError(
            f"settlement {settlement} is after maturity {maturity}: "
            "the bond has expired"
        )


def bond_cashflows(
    bond: FixedRateBond, settlement: dt.date
) -> list[tuple[dt.date, float]]:
    """Remaining cashflows strictly after ``settlement``.

    Coupon amounts use the bond's day count over each (unadjusted) coupon
    period; redemption is added to the final coupon.  Returns an empty list
    when ``settlement == maturity`` (all cashflows have been paid).
    """
    _check_settlement(settlement, bond.maturity)
    dates = bond.schedule()
    period_starts = [bond.effective] + dates[:-1]
    flows: list[tuple[dt.date, float]] = []
    for start, end in zip(period_starts, dates):
        if end <= settlement:
            continue
        accr = year_fraction(start, end, bond.daycount)
        amount = bond.face * bond.coupon * accr
        if end == bond.maturity:
            amount += bond.face
        flows.append((end, amount))
    return flows


def accrued_interest(bond: FixedRateBond, settlement: dt.date) -> float:
    """Accrued interest per bond at settlement (0 at maturity / coupon dates)."""
    _check_settlement(settlement, bond.maturity)
    if settlement <= bond.effective or settlement == bond.maturity:
        return 0.0
    dates = bond.schedule()
    period_starts = [bond.effective] + dates[:-1]
    for start, end in zip(period_starts, dates):
        if start < settlement <= end:
            if settlement == end:
                return 0.0
            accr = year_fraction(start, settlement, bond.daycount)
            return bond.face * bond.coupon * accr
    return 0.0


# --------------------------------------------------------------------- curve
def dirty_price_from_curve(
    bond: FixedRateBond,
    settlement: dt.date,
    curve: DiscountCurve,
    z_spread: float = 0.0,
) -> float:
    """Dirty price per bond off the curve with a continuous z-spread."""
    flows = bond_cashflows(bond, settlement)
    if not flows:
        return 0.0
    t = np.array([curve_time(settlement, d) for d, _ in flows])
    cf = np.array([a for _, a in flows])
    dfs = np.asarray(curve.df(t)) * np.exp(-z_spread * t)
    return float(np.sum(cf * dfs))


def clean_price_from_curve(
    bond: FixedRateBond,
    settlement: dt.date,
    curve: DiscountCurve,
    z_spread: float = 0.0,
) -> float:
    """Clean price = dirty price - accrued interest."""
    return dirty_price_from_curve(bond, settlement, curve, z_spread) - accrued_interest(
        bond, settlement
    )


def zcb_price_from_curve(
    zcb: ZeroCouponBond, settlement: dt.date, curve: DiscountCurve
) -> float:
    """Zero-coupon bond price = face * P(T) exactly."""
    _check_settlement(settlement, zcb.maturity)
    t = curve_time(settlement, zcb.maturity)
    return zcb.face * float(np.asarray(curve.df(t)))


# ----------------------------------------------------------------------- YTM
def _period_exponents(
    bond: FixedRateBond, settlement: dt.date
) -> tuple[np.ndarray, np.ndarray]:
    """Street-convention discount exponents (in periods) and cashflows."""
    flows = bond_cashflows(bond, settlement)
    if not flows:
        raise ValueError(
            f"no remaining cashflows at settlement {settlement} "
            f"(maturity {bond.maturity}): YTM undefined"
        )
    dates = bond.schedule()
    period_starts = [bond.effective] + dates[:-1]
    # Fraction remaining of the period containing the *next* cashflow.  When
    # settlement is exactly a coupon date, the next period is complete: w = 1.
    first_date = flows[0][0]
    w = 1.0
    for start, end in zip(period_starts, dates):
        if end == first_date:
            if settlement > start:
                full = year_fraction(start, end, bond.daycount)
                if full > 0:
                    w = year_fraction(settlement, end, bond.daycount) / full
            break
    exps = w + np.arange(len(flows))
    cf = np.array([a for _, a in flows])
    return exps, cf


def price_from_ytm(bond: FixedRateBond, settlement: dt.date, ytm: float) -> float:
    """Dirty price from street-convention YTM (compounded ``frequency``/yr).

    Handles negative and very large yields; requires ``1 + y/m > 0``.
    """
    m = bond.frequency
    if 1.0 + ytm / m <= 0.0:
        raise ValueError(f"require 1 + ytm/frequency > 0, got ytm={ytm}, m={m}")
    exps, cf = _period_exponents(bond, settlement)
    disc = (1.0 + ytm / m) ** (-exps)
    return float(np.sum(cf * disc))


def ytm_from_price(
    bond: FixedRateBond,
    settlement: dt.date,
    clean_price: float,
    tol: float = 1e-12,
) -> float:
    """Street-convention YTM from a clean price (Brent root solve).

    Round-trips price -> ytm -> price to better than 1e-10 (tested).
    """
    if clean_price <= 0:
        raise ValueError(f"clean price must be > 0, got {clean_price}")
    dirty_target = clean_price + accrued_interest(bond, settlement)

    def f(y: float) -> float:
        return price_from_ytm(bond, settlement, y) - dirty_target

    m = bond.frequency
    lo = -0.999 * m  # keeps 1 + y/m > 0
    hi = 1.0
    while f(hi) > 0:  # price decreasing in yield; expand until bracketed
        hi *= 2.0
        if hi > 1e4:
            raise ValueError(
                f"could not bracket YTM for clean price {clean_price}; "
                "price may be below the minimum attainable value"
            )
    if f(lo) < 0:
        raise ValueError(
            f"clean price {clean_price} exceeds the maximum attainable "
            "price at the yield lower bound"
        )
    return float(brentq(f, lo, hi, xtol=tol, rtol=8.9e-16))


# ------------------------------------------------------------------ z-spread
def z_spread_from_price(
    bond: FixedRateBond,
    settlement: dt.date,
    curve: DiscountCurve,
    clean_price: float,
) -> float:
    """Continuously compounded z-spread matching the quoted clean price.

    Solves ``sum cf_i P(t_i) exp(-z t_i) = clean + accrued`` for ``z``.
    """
    if clean_price <= 0:
        raise ValueError(f"clean price must be > 0, got {clean_price}")
    dirty_target = clean_price + accrued_interest(bond, settlement)

    def f(z: float) -> float:
        return dirty_price_from_curve(bond, settlement, curve, z) - dirty_target

    lo, hi = -1.0, 1.0
    for _ in range(60):
        if f(lo) > 0 > f(hi):
            break
        lo *= 2.0
        hi *= 2.0
    else:
        raise ValueError(
            f"could not bracket z-spread for clean price {clean_price}"
        )
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=8.9e-16))


# ----------------------------------------------------------------------- FRN
def frn_price_from_curve(
    settlement: dt.date,
    maturity: dt.date,
    curve: DiscountCurve,
    frequency: int = 4,
    quoted_margin: float = 0.0,
    face: float = 100.0,
) -> float:
    """Floating-rate note priced off the (single, self-discounting) curve.

    Assumes a reset **on the settlement date** (start of a coupon period):
    projected coupons are the curve's simple forwards plus ``quoted_margin``
    over each period, discounted off the same curve.  With zero margin the
    telescoping identity gives price == par exactly (tested):

    ``sum_i (P(t_{i-1}) - P(t_i)) + P(T) = P(t_0) = 1``.
    """
    _check_settlement(settlement, maturity)
    if settlement == maturity:
        return 0.0
    if frequency not in SUPPORTED_FREQUENCIES:
        raise ValueError(
            f"frequency must be one of {SUPPORTED_FREQUENCIES}, got {frequency}"
        )
    dates = generate_schedule(settlement, maturity, frequency)
    t = np.array([curve_time(settlement, d) for d in dates])
    t_prev = np.concatenate(([0.0], t[:-1]))
    alpha = t - t_prev
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtrapolationWarning)
        p = np.asarray(curve.df(t))
        p_prev = np.asarray(curve.df(t_prev))
    fwd = (p_prev / p - 1.0) / alpha
    coupons = (fwd + quoted_margin) * alpha * p
    return face * float(np.sum(coupons) + p[-1])


# ------------------------------------------------------------------- annuity
def annuity_pv(
    payment: float,
    settlement: dt.date,
    maturity: dt.date,
    curve: DiscountCurve,
    frequency: int = 1,
) -> float:
    """PV of a level annuity paying ``payment`` each period until maturity."""
    _check_settlement(settlement, maturity)
    if settlement == maturity:
        return 0.0
    dates = generate_schedule(settlement, maturity, frequency)
    t = np.array([curve_time(settlement, d) for d in dates])
    return payment * float(np.sum(np.asarray(curve.df(t))))
