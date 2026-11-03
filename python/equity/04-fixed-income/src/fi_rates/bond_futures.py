"""Government bond futures: conversion factors, basis, implied repo, CTD and
a basic delivery (switch) option.

Plain idea
----------
A bond future lets the short deliver any bond from a basket. To make the
bonds comparable each has a conversion factor (CF): roughly its price per 1
of face at the contract's notional yield (6% for CME Treasuries, 6% Eurex,
4% for the Long Gilt since 2011, 7% before). The short receives
``F * CF + accrued`` and picks the bond that is cheapest to deliver (CTD):
the one with the highest implied repo rate, equivalently the lowest net
basis. Because the CTD can change before delivery, the short holds an
option, and the futures price sits below the forward of today's CTD by
roughly its value.

Formal statement
----------------
* CME CF (CME Group, "Calculating U.S. Treasury Futures Conversion
  Factors", IR232): coupon ``c`` rounded to the nearest 1/8% (ties up);
  ``n`` whole years and ``z`` whole months from the first day of the
  delivery month to maturity (or first call), ``z`` rounded down to a
  quarter for TY, TN, US and UB, to the month for TU, Z3N and FV;
  ``v = z if z < 7 else (3 for TY/US, z - 6 for TU/FV)``;
  ``a = 1.03^{-v/6}``, ``b = (c/2)(6 - v)/6``,
  ``k = 1.03^{-2n}`` if ``z < 7`` else ``1.03^{-(2n+1)}``, ``d = (c/0.06)(1 - k)``,
  ``CF = round4(a (c/2 + k + d) - b)``.
* Yield CF (Eurex, ICE Long Gilt): clean price per 1 face at the notional
  yield on the first delivery day, street convention, rounded to
  ``digits`` (Eurex 6, ICE 7).
* Gross basis ``B = P - F CF``. Invoice ``I = F CF + AI(T_d)``.
* Implied repo with one coupon ``C`` paid at ``t_c`` before delivery
  (Burghardt, *The Treasury Bond Basis*, 3rd ed., ch. 2):
  ``r = (I + C - D) / (D tau - C tau_c)`` with ``D = P + AI(t_0)``,
  ``tau = (T_d - t_0)/B``, ``tau_c = (T_d - t_c)/B``, ``B`` = 360 (US
  money market) or 365.
* Net basis at repo ``r``: forward clean price minus ``F CF`` where
  forward dirty ``= D (1 + r tau) - C (1 + r tau_c)``.
* Delivery option (basic, Burghardt ch. 5; Hull 2022 s. 6.2): a single
  parallel shift ``x ~ N(0, sigma^2 T)`` moves every bond's forward yield;
  the fair futures price is ``E[min_i P_i(y_i + x)/CF_i]`` and the switch
  option is ``min_i E[P_i/CF_i] - E[min_i P_i/CF_i] >= 0`` (Jensen), so a
  one-bond basket has none. The expectation uses a
  fixed 801-point trapezoid on ``[-8, 8]`` standard deviations so the
  Rust and C++ ports give the same numbers.

The scalar kernels (:func:`cme_cf_kernel`, :func:`implied_repo_kernel`,
:func:`net_basis_kernel`, :func:`delivery_option_kernel`) are repeated
line for line in ``rust/core/src/bond_futures.rs`` and
``cpp/core/include/qp/bond_futures.hpp`` (plan decision D1).

Worked example
--------------
CME's example 4: the 3-3/4s of 15 Nov 2018 into December 2008 TY. From 1
Dec 2008 the remaining term is 9y 11m 14d, so ``n = 9``, ``z = 9`` is cut
to a quarter (``z = 9``), ``v = 3``; ``a = 0.985329``, ``b = 0.009375``,
``k = 0.570286``, ``d = 0.268571`` and ``CF = 0.835651 -> 0.8357``.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass

from qp_core.holidays import Calendar, is_business_day

from .bond import FixedRateBond, _period_exponents, accrued_interest, price_from_ytm, ytm_from_price

__all__ = [
    "CME_CONTRACTS",
    "Deliverable",
    "cheapest_to_deliver",
    "cme_cf_kernel",
    "cme_conversion_factor",
    "delivery_option",
    "delivery_option_kernel",
    "gross_basis",
    "implied_repo",
    "implied_repo_kernel",
    "net_basis",
    "net_basis_kernel",
    "yield_conversion_factor",
]

# contract -> z rounded to whole quarters?
CME_CONTRACTS: dict[str, bool] = {
    "TU": False, "Z3N": False, "FV": False,
    "TY": True, "TN": True, "US": True, "UB": True,
}


def _round_half_up(x: float, digits: int) -> float:
    s = 10.0 ** digits
    return math.floor(x * s + 0.5) / s


def cme_cf_kernel(coupon: float, n: int, z: int, quarterly: bool) -> float:
    """Unrounded CME conversion factor from rounded coupon, ``n`` years and ``z`` months."""
    if quarterly:
        z = 3 * (z // 3)
        v = z if z < 7 else 3
    else:
        v = z if z < 7 else z - 6
    a = 1.0 / 1.03 ** (v / 6.0)
    b = coupon / 2.0 * (6 - v) / 6.0
    k = 1.0 / 1.03 ** (2 * n) if z < 7 else 1.0 / 1.03 ** (2 * n + 1)
    d = coupon / 0.06 * (1.0 - k)
    return a * (coupon / 2.0 + k + d) - b


def _whole_months(start: dt.date, end: dt.date) -> int:
    m = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        m -= 1
    return m


def cme_conversion_factor(coupon: float, maturity: dt.date, delivery_month: dt.date,
                          contract: str = "TY") -> float:
    """CME Treasury CF, rounded to 4 dp. ``maturity`` is the first call date for
    callable bonds; ``delivery_month`` any day of the delivery month."""
    if contract not in CME_CONTRACTS:
        raise ValueError(f"unknown contract {contract!r}; one of {sorted(CME_CONTRACTS)}")
    if not math.isfinite(coupon) or coupon < 0.0:
        raise ValueError("coupon must be finite and non-negative")
    first = delivery_month.replace(day=1)
    months = _whole_months(first, maturity)
    if months < 0:
        raise ValueError("bond matures before the delivery month")
    c = _round_half_up(coupon * 800.0, 0) / 800.0  # nearest 1/8 of 1%, ties up
    return _round_half_up(cme_cf_kernel(c, months // 12, months % 12, CME_CONTRACTS[contract]), 4)


def yield_conversion_factor(bond: FixedRateBond, first_delivery_day: dt.date,
                            notional_yield: float, digits: int = 6,
                            ex_dividend_days: int = 0, calendar: Calendar = None) -> float:
    """Clean price per 1 face at ``notional_yield`` (Eurex: 6%, 6 dp; ICE Long Gilt 7 dp).

    ``ex_dividend_days`` business days before a coupon the bond trades ex
    (gilts: 7, on ``"GBP"``): the next coupon is dropped from the dirty
    price and accrued interest turns negative, which raises the clean price
    by ``C (1 - v^w)``. ICE's 5-3/4% 2009 gilt into June 2000 needs it
    (0.9142255 with, 0.9141930 without).
    """
    dirty = price_from_ytm(bond, first_delivery_day, notional_yield)
    accrued = accrued_interest(bond, first_delivery_day)
    if ex_dividend_days > 0:
        nxt = next(d for d in bond.schedule() if d > first_delivery_day)
        ex = nxt
        for _ in range(ex_dividend_days):
            ex -= dt.timedelta(days=1)
            while not is_business_day(ex, calendar):
                ex -= dt.timedelta(days=1)
        if first_delivery_day >= ex:
            exps, _ = _period_exponents(bond, first_delivery_day)
            c = bond.face * bond.coupon / bond.frequency
            dirty -= c * (1.0 + notional_yield / bond.frequency) ** (-float(exps[0]))
            accrued -= c
    return _round_half_up((dirty - accrued) / bond.face, digits)


# ------------------------------------------------------------- basis, repo
def gross_basis(clean_price: float, futures_price: float, cf: float) -> float:
    return clean_price - futures_price * cf


def implied_repo_kernel(dirty_now: float, invoice: float, coupon_cash: float,
                        tau: float, tau_c: float) -> float:
    """Simple-interest repo rate equating carry to the invoice (one interim coupon)."""
    den = dirty_now * tau - coupon_cash * tau_c
    if not den > 0.0:
        raise ValueError("delivery must be after settlement")
    return (invoice + coupon_cash - dirty_now) / den


def net_basis_kernel(dirty_now: float, coupon_cash: float, tau: float, tau_c: float,
                     repo: float, accrued_delivery: float, futures_x_cf: float) -> float:
    fwd_dirty = dirty_now * (1.0 + repo * tau) - coupon_cash * (1.0 + repo * tau_c)
    return fwd_dirty - accrued_delivery - futures_x_cf


def _carry_terms(bond: FixedRateBond, settle: dt.date, delivery: dt.date,
                 basis_days: float) -> tuple[float, float, float, float]:
    """(coupon cash in (settle, delivery], tau, tau_c, accrued at delivery)."""
    if delivery <= settle:
        raise ValueError("delivery must be after settlement")
    cash, tau_c = 0.0, 0.0
    flows = [d for d in bond.schedule() if settle < d <= delivery]
    if len(flows) > 1:
        raise ValueError("more than one coupon before delivery is not supported")
    if flows:
        cash = bond.face * bond.coupon / bond.frequency
        tau_c = (delivery - flows[0]).days / basis_days
    return cash, (delivery - settle).days / basis_days, tau_c, accrued_interest(bond, delivery)


def implied_repo(bond: FixedRateBond, clean_price: float, settle: dt.date, delivery: dt.date,
                 futures_price: float, cf: float, basis_days: float = 360.0) -> float:
    cash, tau, tau_c, ai_d = _carry_terms(bond, settle, delivery, basis_days)
    dirty = clean_price + accrued_interest(bond, settle)
    invoice = futures_price * cf + ai_d
    return implied_repo_kernel(dirty, invoice, cash, tau, tau_c)


def net_basis(bond: FixedRateBond, clean_price: float, settle: dt.date, delivery: dt.date,
              futures_price: float, cf: float, repo: float, basis_days: float = 360.0) -> float:
    cash, tau, tau_c, ai_d = _carry_terms(bond, settle, delivery, basis_days)
    dirty = clean_price + accrued_interest(bond, settle)
    return net_basis_kernel(dirty, cash, tau, tau_c, repo, ai_d, futures_price * cf)


@dataclass(frozen=True)
class Deliverable:
    bond: FixedRateBond
    clean_price: float
    cf: float


def cheapest_to_deliver(basket: Sequence[Deliverable], settle: dt.date, delivery: dt.date,
                        futures_price: float, basis_days: float = 360.0
                        ) -> list[tuple[int, float]]:
    """Basket indices with implied repo, best (highest repo, the CTD) first."""
    if not basket:
        raise ValueError("empty basket")
    repos = [(i, implied_repo(d.bond, d.clean_price, settle, delivery, futures_price, d.cf,
                              basis_days)) for i, d in enumerate(basket)]
    return sorted(repos, key=lambda x: -x[1])


# ---------------------------------------------------------- delivery option
_NODES = 801
_WIDTH = 8.0


def delivery_option_kernel(exps: Sequence[Sequence[float]], flows: Sequence[Sequence[float]],
                           accrued: Sequence[float], freq: Sequence[int],
                           fwd_yield: Sequence[float], cf: Sequence[float],
                           sd: float) -> tuple[float, float]:
    """(fair futures price, delivery-option value) under one parallel yield shift.

    Bond ``i`` at delivery has street-convention discount exponents
    ``exps[i]`` and cash flows ``flows[i]``; its clean price at yield ``y`` is
    ``sum flows (1 + y/m)^-exps - accrued``.
    """
    if not sd >= 0.0:
        raise ValueError("sd must be non-negative")

    def ratio(i: int, y: float) -> float:
        m = freq[i]
        p = 0.0
        for e, f in zip(exps[i], flows[i], strict=True):
            p += f * (1.0 + y / m) ** (-e)
        return (p - accrued[i]) / cf[i]

    nb = len(cf)
    if sd == 0.0:
        return min(ratio(i, fwd_yield[i]) for i in range(nb)), 0.0
    h = 2.0 * _WIDTH / (_NODES - 1)
    den = 0.0
    num = 0.0
    each = [0.0] * nb
    for j in range(_NODES):
        x = -_WIDTH + j * h
        w = math.exp(-0.5 * x * x) * (0.5 if j in (0, _NODES - 1) else 1.0)
        best = math.inf
        for i in range(nb):
            r = ratio(i, fwd_yield[i] + sd * x)
            each[i] += w * r
            best = min(best, r)
        num += w * best
        den += w
    fair = num / den
    return fair, min(e / den for e in each) - fair


def delivery_option(basket: Sequence[Deliverable], settle: dt.date, delivery: dt.date,
                    repo: float, yield_vol: float, basis_days: float = 360.0
                    ) -> dict[str, float]:
    """Fair futures price and switch-option value (basic one-factor model).

    Each bond's forward clean price at ``repo`` gives its forward yield;
    ``yield_vol`` is the normal (absolute) annual vol of the common shift.
    """
    if not yield_vol >= 0.0:
        raise ValueError("yield_vol must be non-negative")
    exps, flows, acc, freq, fy, cfs = [], [], [], [], [], []
    for d in basket:
        cash, tau, tau_c, ai_d = _carry_terms(d.bond, settle, delivery, basis_days)
        dirty = d.clean_price + accrued_interest(d.bond, settle)
        fwd_clean = dirty * (1.0 + repo * tau) - cash * (1.0 + repo * tau_c) - ai_d
        e, cf_ = _period_exponents(d.bond, delivery)
        exps.append([float(x) for x in e])
        flows.append([float(x) for x in cf_])
        acc.append(ai_d)
        freq.append(d.bond.frequency)
        fy.append(ytm_from_price(d.bond, delivery, fwd_clean))
        cfs.append(d.cf)
    t = (delivery - settle).days / 365.0
    fair, opt = delivery_option_kernel(exps, flows, acc, freq, fy, cfs, yield_vol * math.sqrt(t))
    return {"fair_futures": fair, "option_value": opt}
