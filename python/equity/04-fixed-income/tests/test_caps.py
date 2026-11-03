"""fi_rates.caps: dated caplets, payment lag, SABR smile, cap stripping."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pytest
from fi_rates.caps import (
    CapletPeriod,
    StrippedCapletVols,
    cap_price,
    caplet_price,
    periods_from_schedule,
    sabr_caplet_vol,
    strip_caplet_vols,
)
from fi_rates.curve import DiscountCurve
from fi_rates.rate_options import black76
from fi_rates.rate_options import cap_price as old_cap_price
from qp_core.vol.sabr import sabr_vol
from scipy.optimize import brentq

T = np.arange(1, 121) * 0.125
FLAT = DiscountCurve(T, np.exp(-0.03 * T))
PROJ = DiscountCurve(T, np.exp(-0.032 * T))


def _quarterly(maturity: float, lag: float = 0.0) -> list[CapletPeriod]:
    t = np.arange(1, int(maturity * 4)) / 4.0  # the first period (fixed today) is skipped
    return [CapletPeriod(s, s, s + 0.25, s + 0.25 + lag, 0.25) for s in t]


def test_reduces_to_old_cap_without_lag():
    ours = cap_price(FLAT, PROJ, _quarterly(5.0), 0.03, 0.2)
    assert ours == pytest.approx(old_cap_price(FLAT, PROJ, 0.03, 5.0, 0.2), rel=1e-13)


def test_payment_lag_discounts_further():
    # Failing-first: the old pricer discounts to the accrual end, so it
    # cannot see a lag at all. On a flat 3% curve every caplet is worth
    # exactly e^{-0.03 lag} times its unlagged value.
    lag = 2.0 / 365.0
    no = cap_price(FLAT, FLAT, _quarterly(5.0), 0.03, 0.2)
    yes = cap_price(FLAT, FLAT, _quarterly(5.0, lag), 0.03, 0.2)
    assert yes == pytest.approx(no * np.exp(-0.03 * lag), rel=1e-12)
    assert yes < no


def test_caplet_formula_by_hand():
    p = CapletPeriod(1.0, 1.0, 1.5, 1.51, 0.5)
    F = (np.exp(-0.032) / np.exp(-0.032 * 1.5) - 1) / 0.5
    want = 0.5 * np.exp(-0.03 * 1.51) * black76(F, 0.031, 1.0, 0.25)
    assert caplet_price(FLAT, PROJ, p, 0.031, 0.25) == pytest.approx(want, rel=1e-14)


@pytest.mark.parametrize(("model", "vol", "shift"), [("black", 0.2, 0.0), ("shifted_black", 0.15, 0.01),
                                                     ("normal", 0.008, 0.0)])
def test_cap_minus_floor_is_swap(model, vol, shift):
    ps = _quarterly(4.0, 1 / 365)
    K = 0.025
    swap = sum(p.accrual * float(FLAT.df(p.payment))
               * ((float(PROJ.df(p.start)) / float(PROJ.df(p.end)) - 1) / p.accrual - K) for p in ps)
    cap = cap_price(FLAT, PROJ, ps, K, vol, model, shift)
    flo = cap_price(FLAT, PROJ, ps, K, vol, model, shift, floor=True)
    assert cap - flo == pytest.approx(swap, abs=1e-15)


def test_stripping_reprices_every_cap_and_recovers_a_term_structure():
    ps = _quarterly(10.0)
    mats = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    true = [0.30, 0.27, 0.25, 0.22, 0.20, 0.19]
    cut = tuple(max(p.fixing for p in ps if p.end <= m and (i == 0 or p.end > mats[i - 1]))
                for i, m in enumerate(mats))
    truth = StrippedCapletVols(cut, tuple(true), "black", 0.0)
    flat = []
    for m in mats:
        cap = [p for p in ps if p.end <= m]
        target = cap_price(FLAT, PROJ, cap, 0.03, truth)
        flat.append(brentq(lambda v, cap=cap, target=target: cap_price(FLAT, PROJ, cap, 0.03, v) - target,
                           0.01, 2.0, xtol=1e-15))
    got = strip_caplet_vols(FLAT, PROJ, ps, mats, flat, 0.03)
    assert got.vols == pytest.approx(true, abs=1e-10)
    for m, s in zip(mats, flat, strict=True):
        cap = [p for p in ps if p.end <= m]
        assert cap_price(FLAT, PROJ, cap, 0.03, got) == pytest.approx(
            cap_price(FLAT, PROJ, cap, 0.03, s), abs=1e-14)


def test_stripping_rejects_calendar_arbitrage_and_empty_buckets():
    ps = _quarterly(3.0)
    with pytest.raises(ValueError, match="arbitrage"):
        strip_caplet_vols(FLAT, PROJ, ps, [1.0, 2.0], [0.40, 0.05], 0.03)
    with pytest.raises(ValueError, match="adds no caplet"):
        strip_caplet_vols(FLAT, PROJ, ps, [1.0, 1.1], [0.3, 0.3], 0.03)


def test_sabr_caplet_vol_uses_caplet_forward():
    p = CapletPeriod(2.0, 2.0, 2.25, 2.25, 0.25)
    F = (float(PROJ.df(2.0)) / float(PROJ.df(2.25)) - 1) / 0.25
    v = sabr_caplet_vol(lambda t: (0.02, 0.5, -0.3, 0.4), PROJ, p, 0.025, shift=0.01)
    assert v == sabr_vol(F, 0.025, 2.0, 0.02, 0.5, -0.3, 0.4, 0.01)


@dataclass
class _P:  # stand-in for a W13a schedule period
    accrual_start: dt.date
    accrual_end: dt.date
    payment_date: dt.date


def test_periods_from_schedule_drops_fixed_period_and_uses_day_counts():
    v = dt.date(2026, 1, 15)
    sched = [_P(dt.date(2026, 1, 15), dt.date(2026, 4, 15), dt.date(2026, 4, 17)),
             _P(dt.date(2026, 4, 15), dt.date(2026, 7, 15), dt.date(2026, 7, 17))]
    ps = periods_from_schedule(sched, v)
    assert len(ps) == 1
    p = ps[0]
    assert p.accrual == pytest.approx(91 / 360)
    assert p.start == pytest.approx(90 / 365)
    assert p.payment == pytest.approx(183 / 365)


def test_period_validation():
    with pytest.raises(ValueError):
        CapletPeriod(1.0, 1.0, 0.5, 1.0, 0.25)
    with pytest.raises(ValueError):
        CapletPeriod(1.1, 1.0, 1.25, 1.25, 0.25)
    with pytest.raises(ValueError):
        cap_price(FLAT, PROJ, _quarterly(2.0), 0.03, [0.2])
    with pytest.raises(ValueError):
        cap_price(FLAT, PROJ, _quarterly(2.0), 0.03, 0.2, model="sabr")
