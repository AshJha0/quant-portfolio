"""Float-float cross-currency basis: bootstrap, collateral currency, risk separation."""

import numpy as np
import pytest

from fx_rates.bootstrap import basis_discount_curve, basis_swap_par_spread
from fx_rates.curve import DiscountCurve
from fx_rates.fxforward import MarketState
from fx_rates.risk import basis_dv01, dv01
from fx_rates.xccy import BasisSwap

TIMES = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


def _curve(z0, slope, name):
    return DiscountCurve.from_zero_rates(TIMES, [z0 + slope * t for t in TIMES], name=name)


@pytest.fixture
def usdjpy():
    return MarketState(150.0, _curve(0.005, 0.0008, "JPY"), _curve(0.045, -0.001, "USD"),
                       ((1.0, -0.0040), (5.0, -0.0060), (10.0, -0.0055)), ("USD", "JPY"))


class TestBootstrap:
    def test_zero_spreads_reproduce_projection_curve(self):
        proj = _curve(0.03, 0.001, "EUR")
        disc = basis_discount_curve(proj, ((2.0, 0.0), (10.0, 0.0)))
        t = np.linspace(0.01, 12.0, 301)
        assert np.max(np.abs(disc.df(t) / proj.df(t) - 1.0)) < 1e-14

    @pytest.mark.parametrize("freq", [1, 2, 4])
    def test_quoted_swaps_reprice_at_par(self, freq):
        proj = _curve(0.03, 0.001, "EUR")
        quotes = ((1.0, -0.0015), (3.0, -0.0022), (5.0, -0.0025), (10.0, -0.0030))
        disc = basis_discount_curve(proj, quotes, frequency=freq)
        for t, s in quotes:
            assert basis_swap_par_spread(proj, disc, t, freq) == pytest.approx(s, abs=1e-14)

    def test_negative_basis_raises_discount_factors(self):
        proj = _curve(0.03, 0.001, "EUR")
        disc = basis_discount_curve(proj, ((5.0, -0.0025),))
        assert disc.df(5.0) > proj.df(5.0)

    def test_bad_inputs(self):
        proj = _curve(0.03, 0.0, "EUR")
        with pytest.raises(ValueError, match="frequency"):
            basis_discount_curve(proj, ((5.0, -0.001),), frequency=0)
        with pytest.raises(ValueError, match="duplicate"):
            basis_discount_curve(proj, ((5.0, -0.001), (5.0, -0.002)))
        with pytest.raises(ValueError):
            basis_discount_curve(proj, ((5.0, np.nan),))
        with pytest.raises(ValueError, match="integer"):
            basis_swap_par_spread(proj, proj, 1.1)


class TestCollateralCurrency:
    def test_eurusd_basis_on_eur_curve(self, market):
        assert market.collateral_ccy == "USD" and market.basis_ccy == "EUR"
        assert market.domestic_curve_adjusted is market.domestic_curve
        assert market.foreign_curve_adjusted is not market.foreign_curve

    def test_usdjpy_basis_on_jpy_not_usd(self, usdjpy):
        # the old model shifted the base (USD) curve, the collateral leg
        assert usdjpy.basis_ccy == "JPY"
        assert usdjpy.foreign_curve_adjusted is usdjpy.foreign_curve
        assert usdjpy.domestic_curve_adjusted.df(5.0) > usdjpy.domestic_curve.df(5.0)

    def test_collateral_must_be_in_pair(self, market):
        with pytest.raises(ValueError, match="collateral_ccy"):
            market.replace(collateral_ccy="GBP")

    def test_zero_spread_model_is_deprecated(self, market):
        with pytest.warns(DeprecationWarning, match="zero_spread"):
            market.replace(basis_model="zero_spread")
        with pytest.raises(ValueError, match="basis_model"):
            market.replace(basis_model="other")


def _par_swap(m, t, notional=100e6):
    s = dict(m.basis_spreads)[t]
    return BasisSwap(notional, notional * m.spot, s, t, pair=m.pair)


class TestBasisSwapRisk:
    @pytest.mark.parametrize("t", [1.0, 5.0, 10.0])
    def test_par_swap_has_zero_pv(self, market, usdjpy, t):
        for m in (market, usdjpy):
            if t in dict(m.basis_spreads):
                sw = _par_swap(m, t)
                assert abs(sw.value(m)) < 1e-12 * sw.notional_quote

    def test_basis_dv01_is_minus_spread_annuity(self, market):
        """Exact identity: with all quotes up by d, a swap at the old par
        spread loses N*S*d*tau*sum D'(t_i), D' the re-bootstrapped curve."""
        t = 5.0
        sw = _par_swap(market, t)
        times = sw.payment_times()
        ann = []
        for bp in (+1.0, -1.0):
            m = market.replace(basis_spreads=tuple((x, s + bp * 1e-4)
                                                   for x, s in market.basis_spreads))
            ann.append(float(np.sum(m.foreign_curve_adjusted.df(times))) / sw.frequency)
        expected = -sw.notional_base * market.spot * 1e-4 * 0.5 * (ann[0] + ann[1])
        assert basis_dv01(sw, market) == pytest.approx(expected, rel=1e-9)

    def test_par_swap_has_no_rate_dv01(self, market):
        sw = _par_swap(market, 5.0)
        bdv = abs(basis_dv01(sw, market))
        assert abs(dv01(sw, market, "EUR")) < 1e-6 * bdv
        assert abs(dv01(sw, market, "USD")) < 1e-6 * bdv

    def test_usdjpy_basis_risk_sits_on_jpy_leg(self, usdjpy):
        sw = _par_swap(usdjpy, 5.0)
        assert basis_dv01(sw, usdjpy) != 0.0
        assert abs(dv01(sw, usdjpy, "JPY")) < 1e-6 * abs(basis_dv01(sw, usdjpy))

    def test_validation(self):
        with pytest.raises(ValueError):
            BasisSwap(1e6, 1e6, 0.0, 5.1, frequency=4)
        with pytest.raises(ValueError):
            BasisSwap(-1e6, 1e6, 0.0, 5.0)
        with pytest.raises(ValueError):
            BasisSwap(1e6, 1e6, np.inf, 5.0)


class TestConventionFixes:
    def test_bootstrap_adds_pillars_only_at_quoted_maturities(self, quotes, market):
        # the old bootstrap linearly interpolated par rates onto 4y, 6y, ...
        expected = sorted({t for t, _ in quotes.domestic_deposits}
                          | {float(t) for t, _ in quotes.domestic_swaps})
        assert list(market.domestic_curve.times) == expected

    def test_jpy_quoted_pairs_use_two_decimal_points(self):
        from fx_rates.fxforward import _point_factor
        assert _point_factor(("GBP", "JPY")) == 1e2
        assert _point_factor(("GBP", "USD")) == 1e4

    def test_cip_detector_uses_each_currency_accrual(self):
        from fx_rates.arbitrage import CIPQuotes, no_arb_bounds
        common = dict(spot_bid=1.27, spot_ask=1.27, fwd_bid=1.27, fwd_ask=1.27,
                      dom_rate_bid=0.05, dom_rate_ask=0.05, for_rate_bid=0.04,
                      for_rate_ask=0.04)
        # GBPUSD 182 days: USD ACT/360, GBP ACT/365F
        q = CIPQuotes(**common, tau=182 / 360, tau_dom=182 / 360, tau_for=182 / 365)
        lo, hi = no_arb_bounds(q)
        assert lo == pytest.approx(1.27 * (1 + 0.05 * 182 / 360) / (1 + 0.04 * 182 / 365),
                                   rel=1e-15)
        assert CIPQuotes(**common, tau=0.5).tau_for == 0.5
