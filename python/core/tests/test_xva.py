"""xVA: Hull-White closed forms against MC, IRS CVA against the swaption strip."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qp_core.errors import InputError
from qp_core.market import DiscountCurve
from qp_core.xva import (
    HazardCurve,
    HullWhite,
    SwapSpec,
    cva,
    dva,
    eepe,
    fva,
    pfe_index,
    profile,
    swap_exposure,
    xva,
)

CURVE = DiscountCurve(
    (0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    tuple(
        math.exp(-r * t)
        for r, t in [(0.025, 0.5), (0.027, 1.0), (0.03, 2.0), (0.033, 5.0), (0.035, 10.0), (0.036, 30.0)]
    ),
)
HW = HullWhite(CURVE, a=0.05, sigma=0.01)


def par_rate(dates: list[float]) -> float:
    ann = sum((b - a) * CURVE.df(b) for a, b in zip(dates, dates[1:], strict=False))
    p0 = 1.0 if dates[0] == 0.0 else CURVE.df(dates[0])
    return (p0 - CURVE.df(dates[-1])) / ann


def test_zcb_reprices_curve_and_mc_martingale() -> None:
    for T in (0.5, 3.0, 12.0):
        assert HW.zcb(0.0, T, 0.0) == pytest.approx(CURVE.df(T), rel=1e-14)
    paths = HW.simulate([0.0, 1.0, 3.0, 7.0], 200_000, seed=7)
    for k, t in enumerate(paths.times[1:], start=1):
        assert paths.disc[:, k].mean() == pytest.approx(CURVE.df(t), rel=2e-4)
        # E[D(0,t) P(t,T)] = P(0,T)
        T = t + 3.0
        pt = np.exp(HW.log_a(t, T) - HW.b(t, T) * paths.x[:, k])
        assert (paths.disc[:, k] * pt).mean() == pytest.approx(CURVE.df(T), rel=3e-4)


@pytest.mark.parametrize("payer", [True, False])
def test_jamshidian_matches_mc(payer: bool) -> None:
    pay = [2.0, 3.0, 4.0, 5.0, 6.0]
    acc = [1.0] * 5
    k = par_rate([1.0, *pay])
    analytic = HW.swaption(1.0, pay, acc, k, payer=payer)
    paths = HW.simulate([0.0, 1.0], 400_000, seed=11)
    x = paths.x[:, 1]
    bond = sum(a * k * np.exp(HW.log_a(1.0, t) - HW.b(1.0, t) * x) for a, t in zip(acc, pay, strict=True))
    bond = bond + np.exp(HW.log_a(1.0, pay[-1]) - HW.b(1.0, pay[-1]) * x)
    pay_val = 1.0 - bond
    payoff = np.maximum(pay_val if payer else -pay_val, 0.0) * paths.disc[:, 1]
    se = payoff.std(ddof=1) / math.sqrt(payoff.size)
    assert abs(payoff.mean() - analytic) < 4.0 * se
    # put-call parity on the coupon bond: payer - receiver = forward swap value
    other = HW.swaption(1.0, pay, acc, k, payer=not payer)
    fwd = CURVE.df(1.0) - CURVE.df(6.0) - k * sum(CURVE.df(t) for t in pay)
    assert (analytic - other) * (1 if payer else -1) == pytest.approx(fwd, abs=1e-14)


def test_irs_cva_matches_swaption_strip() -> None:
    """W17 acceptance: CVA of one IRS from MC exposure equals the semi-analytic
    swaption-strip CVA to within Monte Carlo error."""
    dates = [0.0] + [0.5 * i for i in range(1, 21)]  # 10y semi-annual
    k = par_rate(dates)
    spec = SwapSpec(tuple(dates), k, payer=True, notional=1e8)
    prof, _, _ = swap_exposure(HW, spec, 100_000, seed=2026)
    hc = HazardCurve.from_spread(0.012, 0.4)
    grid = list(prof.times)
    strip_ee = [0.0]
    strip_ene = [0.0]
    for j, t in enumerate(dates[1:-1], start=1):
        rest = dates[j + 1 :]
        accs = [b - a for a, b in zip(dates[j:], rest, strict=False)]
        strip_ee.append(1e8 * HW.swaption(t, rest, accs, k, payer=True))
        strip_ene.append(-1e8 * HW.swaption(t, rest, accs, k, payer=False))
    assert grid == dates[:-1]
    c_mc = cva(grid, prof.ee_disc, hc, 0.4)
    c_an = cva(grid, strip_ee, hc, 0.4)
    # Error bar: CVA is linear in EE*, bound by the sum of per-date SEs.
    lgd_pd = [0.6 * (hc.survival(grid[i - 1]) - hc.survival(grid[i])) for i in range(1, len(grid))]
    bound = 4.0 * sum(w * s for w, s in zip(lgd_pd, prof.ee_disc_se[1:], strict=True))
    assert abs(c_mc - c_an) < bound
    assert abs(c_mc - c_an) / c_an < 0.01
    for k_ in range(1, len(grid)):
        assert abs(prof.ee_disc[k_] - strip_ee[k_]) < 5.0 * prof.ee_disc_se[k_] + 1e-9
    d_mc = dva(grid, prof.ene_disc, hc, 0.4)
    d_an = dva(grid, strip_ene, hc, 0.4)
    assert d_mc == pytest.approx(d_an, rel=0.01)


def test_profile_and_pfe_definition() -> None:
    v = np.arange(1, 101, dtype=float).reshape(100, 1) - 50.0
    v = np.hstack([np.zeros((100, 1)), v])
    p = profile(np.array([0.0, 1.0]), v, q=0.95)
    assert pfe_index(100, 0.95) == 94
    assert p.pfe[1] == 45.0  # 95th smallest of -49..50
    assert p.ee[1] == pytest.approx(sum(range(1, 51)) / 100)
    assert p.ene[1] == pytest.approx(-sum(range(0, 50)) / 100)


def test_eepe_non_decreasing_and_horizon() -> None:
    assert eepe([0.0, 0.25, 0.5, 0.75, 1.0], [0, 4, 1, 1, 1], 5.0) == pytest.approx(4 * 0.25 * 4)
    assert eepe([0.0, 0.25, 0.5], [0, 2, 2], 0.5) == pytest.approx(2.0)


def test_cva_closed_form_flat() -> None:
    h = HazardCurve.flat(0.03)
    t = list(np.linspace(0, 5, 11))
    assert cva(t, [2.0] * 11, h, 0.4) == pytest.approx(2.0 * 0.6 * (1 - math.exp(-0.15)), rel=1e-14)
    # first-to-default: weight by bank survival
    b = HazardCurve.flat(0.01)
    bil = cva(t, [2.0] * 11, h, 0.4, bank=b)
    assert bil < cva(t, [2.0] * 11, h, 0.4)


def test_fva_and_total_sign() -> None:
    t = [0.0, 1.0, 2.0]
    h0 = HazardCurve.flat(0.0)
    fca, fba = fva(t, [0, 1, 1], [0, -2, -2], 0.01, 0.005, h0, h0)
    assert (fca, fba) == pytest.approx((0.02, 0.02))
    r = xva(
        t,
        [0, 1, 1],
        [0, -2, -2],
        HazardCurve.flat(0.02),
        HazardCurve.flat(0.01),
        borrow_spread=0.01,
        lend_spread=0.005,
    )
    assert r.total == pytest.approx(-r.cva + r.dva - r.fca + r.fba)
    assert r.cva > 0 and r.dva > 0


def test_piecewise_hazard_survival() -> None:
    h = HazardCurve((1.0, 3.0), (0.01, 0.03))
    assert h.survival(2.0) == pytest.approx(math.exp(-(0.01 + 0.03)))
    assert h.survival(5.0) == pytest.approx(math.exp(-(0.01 + 0.06 + 0.06)))


def test_bad_inputs() -> None:
    with pytest.raises(InputError):
        HullWhite(CURVE, a=0.0, sigma=0.01)
    with pytest.raises(InputError):
        HW.simulate([0.5, 1.0], 10, 1)
    with pytest.raises(InputError):
        SwapSpec((1.0, 1.0), 0.03)
    with pytest.raises(InputError):
        cva([0.0, 1.0], [1.0], HazardCurve.flat(0.01), 0.4)
    with pytest.raises(InputError):
        HazardCurve.from_spread(0.01, 1.0)
