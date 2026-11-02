"""Float-float cross-currency swaps: constant notional and MtM reset."""

import pytest
from fx_rates.curve import DiscountCurve
from fx_rates.fxforward import MarketState
from fx_rates.xccy import BasisSwap
from fx_rates.xccy_float import FloatFloatXccy, XccyCurves, par_base_spread

TIMES = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


def _curve(z0, slope):
    return DiscountCurve.from_zero_rates(TIMES, [z0 + slope * t for t in TIMES])


EUR_PROJ, EUR_DISC = _curve(0.025, 0.001), _curve(0.022, 0.0012)
USD_PROJ, USD_DISC = _curve(0.045, -0.001), _curve(0.043, -0.0009)
C = XccyCurves(1.085, EUR_PROJ, EUR_DISC, USD_PROJ, USD_DISC)


def test_single_curve_zero_spread_legs_are_par():
    """Projection = discount and zero spread -> each leg (with exchanges) is worth 0."""
    c = XccyCurves(1.1, EUR_DISC, EUR_DISC, USD_DISC, USD_DISC)
    for mtm in (False, True):
        s = FloatFloatXccy(1e6, 5.0, 4, mtm_reset=mtm)
        assert s.base_leg_pv(c) == pytest.approx(0.0, abs=1e-8)
        assert s.quote_leg_pv(c) == pytest.approx(0.0, abs=1e-8)


def test_constant_notional_matches_existing_basis_swap():
    """Independent code path: fx_rates.xccy.BasisSwap on a basis-quoted market."""
    m = MarketState(1.085, USD_PROJ, EUR_PROJ, basis_spreads=((2.0, -0.002), (5.0, -0.0025),
                                                               (10.0, -0.003)),
                    pair=("EUR", "USD"))
    c = XccyCurves(m.spot, m.foreign_curve, m.foreign_curve_adjusted,
                   m.domestic_curve, m.domestic_curve_adjusted)
    for s in (-0.003, 0.0, 0.001):
        ours = FloatFloatXccy(1e6, 5.0, 4, spread_base=s).value(c)
        ref = BasisSwap(1e6, 1e6 * m.spot, s, 5.0, 4, include_initial_exchange=True).value(m)
        assert ours == pytest.approx(ref, abs=1e-6)


def test_one_period_mtm_equals_constant():
    a = FloatFloatXccy(1e6, 0.25, 4, spread_base=0.001, mtm_reset=False).value(C)
    b = FloatFloatXccy(1e6, 0.25, 4, spread_base=0.001, mtm_reset=True).value(C)
    assert a == pytest.approx(b, abs=1e-9)


def test_mtm_at_par_has_no_fx_delta():
    """MtM value is proportional to spot, so at par its FX delta is zero; the
    constant-notional swap at its own par keeps an FX delta."""
    def bumped(spot):
        return XccyCurves(spot, EUR_PROJ, EUR_DISC, USD_PROJ, USD_DISC)

    for mtm, expect_zero in ((True, True), (False, False)):
        s0 = FloatFloatXccy(1e6, 5.0, 4, mtm_reset=mtm, notional_quote=1e6 * C.spot)
        sp = par_base_spread(s0, C)
        s = FloatFloatXccy(1e6, 5.0, 4, spread_base=sp, mtm_reset=mtm,
                           notional_quote=1e6 * C.spot)
        assert s.value(C) == pytest.approx(0.0, abs=1e-7)
        delta = (s.value(bumped(1.095)) - s.value(bumped(1.075))) / 0.02
        if expect_zero:
            assert abs(delta) < 1e-6
        else:
            assert abs(delta) > 1.0


def test_receive_pay_symmetry_and_validation():
    a = FloatFloatXccy(1e6, 3.0, 2, 0.001, 0.0, True, True).value(C)
    b = FloatFloatXccy(1e6, 3.0, 2, 0.001, 0.0, True, False).value(C)
    assert a == pytest.approx(-b, abs=1e-12)
    with pytest.raises(ValueError):
        FloatFloatXccy(1e6, 1.1, 4)
    with pytest.raises(ValueError):
        XccyCurves(-1.0, EUR_PROJ, EUR_DISC, USD_PROJ, USD_DISC)


# ------------------------------------------------ MtM convexity, seasoning
import numpy as np  # noqa: E402
from fx_rates.xccy_float import MtmConvexity  # noqa: E402

CVX = MtmConvexity(fx_vol=0.15, rate_vol=0.012, mean_reversion=0.05, correlation=0.8)


def test_convexity_only_touches_the_spread():
    """Single curve, zero spread: the MtM leg is worth 0 whatever the vols."""
    c = XccyCurves(1.1, EUR_DISC, EUR_DISC, USD_DISC, USD_DISC)
    s = FloatFloatXccy(1e6, 5.0, 4, mtm_reset=True, convexity=CVX)
    assert s.quote_leg_pv(c) == pytest.approx(0.0, abs=1e-8)
    a = FloatFloatXccy(1e6, 5.0, 4, spread_quote=0.002, mtm_reset=True)
    b = FloatFloatXccy(1e6, 5.0, 4, spread_quote=0.002, mtm_reset=True, convexity=CVX)
    # positive correlation lowers the value of the spread received later
    assert b.quote_leg_pv(C) < a.quote_leg_pv(C) - 1.0


def test_convexity_factor_limits_and_validation():
    assert MtmConvexity(0.1, 0.0, 0.05, 0.5).factor(5.0, 0.25) == 1.0
    assert CVX.factor(0.0, 0.25) == 1.0
    up = MtmConvexity(0.15, 0.012, 0.05, 0.8).factor(5.0, 1.0)
    dn = MtmConvexity(0.15, 0.012, 0.05, -0.8).factor(5.0, 1.0)
    assert up < 1.0 < dn
    # small-a limit: ln c -> -tau (rho sS sr t + sr^2 t^2 / 2)
    sm = MtmConvexity(0.15, 0.012, 1e-6, 0.8).factor(5.0, 1.0)
    ref = np.exp(-1.0 * (0.8 * 0.15 * 0.012 * 5.0 + 0.012**2 * 25.0 / 2))
    assert sm == pytest.approx(ref, rel=1e-6)
    with pytest.raises(ValueError):
        MtmConvexity(0.1, 0.01, 0.0, 0.5)
    with pytest.raises(ValueError):
        MtmConvexity(0.1, 0.01, 0.05, 1.5)


def test_convexity_factor_monte_carlo_oracle():
    """Quote-measure MC: Hull-White quote rate, lognormal spot, deterministic
    base rate. E[D_q(0,t2) S(t1)] = c F(0,t1) D_q(t2), estimated with the
    exact control E[D_q(0,t1) S(t1)] = S_0 D_b(t1)."""
    rng = np.random.default_rng(2026)
    a, sr, ss, rho = 0.05, 0.012, 0.15, 0.8
    f0, rb, S0 = 0.04, 0.02, 1.1
    t1, tau = 5.0, 1.0
    n_steps, n = 500, 200_000
    dt = t1 / n_steps
    e = np.exp(-a * dt)
    sd_x = sr * np.sqrt((1 - e * e) / (2 * a))

    def alpha(t):
        return f0 + sr**2 / (2 * a**2) * (1 - np.exp(-a * t)) ** 2

    x = np.zeros(n)
    integ = np.zeros(n)
    lnS = np.full(n, np.log(S0))
    r_prev = x + alpha(0.0)
    for i in range(1, n_steps + 1):
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(1 - rho * rho) * rng.standard_normal(n)
        x = x * e + sd_x * z1
        r = x + alpha(i * dt)
        step = 0.5 * (r_prev + r) * dt
        lnS += step - rb * dt - 0.5 * ss**2 * dt + ss * np.sqrt(dt) * z2
        integ += step
        r_prev = r
    B = (1 - np.exp(-a * tau)) / a
    lnA = B * f0 - sr**2 / (4 * a) * (1 - np.exp(-2 * a * t1)) * B * B - f0 * tau
    P12 = np.exp(lnA - B * r_prev)
    Z = np.exp(-integ + lnS)
    p_fwd = np.exp(-f0 * tau)
    est = Z * (P12 - p_fwd)
    mean = est.mean() + S0 * np.exp(-rb * t1) * p_fwd
    se = est.std(ddof=1) / np.sqrt(n)
    F_D = S0 * np.exp(-rb * t1) * p_fwd  # F(0,t1) D_q(t2)
    c = MtmConvexity(ss, sr, a, rho).factor(t1, tau)
    assert abs(mean - c * F_D) < 4 * se + 2e-5 * F_D, (mean, c * F_D, se)
    assert abs(mean - F_D) > 10 * se  # the adjustment is visible


def _seasoned(elapsed, **kw):
    base = dict(notional_base=1e6, maturity=5.0, frequency=4, spread_base=0.001,
                spread_quote=0.0005, notional_quote=1.1e6)
    base.update(kw)
    return FloatFloatXccy(elapsed=elapsed, **base)


def test_seasoned_trade_needs_fixings():
    with pytest.raises(ValueError):
        _seasoned(1.1)
    with pytest.raises(ValueError):
        _seasoned(1.1, mtm_reset=True, base_fixing=0.02, quote_fixing=0.04)
    with pytest.raises(ValueError):
        _seasoned(5.0)
    _seasoned(1.0)  # on a reset date no fixing is needed


def test_seasoned_mtm_current_period_is_a_known_amount():
    """Moving the FX fixing changes the value by exactly the fixed flow."""
    kw = dict(mtm_reset=True, base_fixing=0.025, quote_fixing=0.045)
    a = _seasoned(1.1, fx_fixing=1.05, **kw).value(C)
    b = _seasoned(1.1, fx_fixing=1.07, **kw).value(C)
    t1 = 1.25 - 1.1
    flow = 1e6 * 0.02 * float(USD_DISC.df(t1)) * (1 + 0.25 * (0.045 + 0.0005))
    assert a - b == pytest.approx(flow, rel=1e-12)


def test_seasoned_constant_leg_matches_hand_value():
    s = _seasoned(1.1, base_fixing=0.025, quote_fixing=0.045)
    t = np.arange(5, 21) / 4.0 - 1.1
    p, d = EUR_PROJ.df(t), EUR_DISC.df(t)
    cpn = d[0] * 0.25 * (0.025 + 0.001) + np.sum(d[1:] * (p[:-1] / p[1:] - 1 + 0.25 * 0.001))
    assert s.base_leg_pv(C) == pytest.approx(C.spot * 1e6 * (cpn + d[-1]), rel=1e-12)


def test_seasoned_value_is_continuous_through_a_reset():
    """On the reset date (today's flows settled) vs an instant later with the
    fixings equal to today's spot and forwards."""
    for mtm in (False, True):
        on = _seasoned(1.0, mtm_reset=mtm, convexity=CVX).value(C)
        eps = 1e-7
        lb = (EUR_PROJ.df(1e-12) / EUR_PROJ.df(0.25) - 1) / 0.25
        lq = (USD_PROJ.df(1e-12) / USD_PROJ.df(0.25) - 1) / 0.25
        later = _seasoned(1.0 + eps, mtm_reset=mtm, convexity=CVX, base_fixing=float(lb),
                          quote_fixing=float(lq), fx_fixing=C.spot).value(C)
        assert later == pytest.approx(on, abs=0.05)
