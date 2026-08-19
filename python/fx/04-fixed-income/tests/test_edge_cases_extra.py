"""Further edge cases: MarketState immutability, curve extrapolation limits,
CIP-violation detection, pegged/managed pairs, deep-negative rates, tiny and
degenerate inputs, and basis-sign economics.

Complements tests/test_edge_cases.py; documented in docs/VALIDATION.md.
"""

import dataclasses
import datetime as dt

import numpy as np
import pytest

from fx_rates.arbitrage import CIPQuotes, detect_cip_arbitrage, no_arb_bounds
from fx_rates.bootstrap import (
    basis_adjusted_curve,
    bootstrap_curve,
    curve_from_fx_forwards,
    df_from_deposit,
    implied_basis_from_forwards,
    par_swap_rate,
)
from fx_rates.curve import DiscountCurve
from fx_rates.daycount import tenor_to_years, year_fraction
from fx_rates.fxforward import (
    FXForward,
    FXSwap,
    MarketState,
    cip_forward,
    forward_points,
    forward_points_table,
    market_forward,
)
from fx_rates.risk import basis_dv01, book_risk_report, book_value, dv01, fx_delta
from fx_rates.scenarios import Scenario, apply_scenario, forward_carry


def _curve(zeros, name=""):
    return DiscountCurve.from_zero_rates(
        [0.25, 0.5, 1.0, 2.0, 5.0, 10.0], zeros, name=name
    )


@pytest.fixture(scope="module")
def simple_market():
    d = _curve([0.053, 0.052, 0.051, 0.048, 0.044, 0.042], "USD")
    f = _curve([0.039, 0.038, 0.037, 0.034, 0.030, 0.028], "EUR")
    return MarketState(1.0850, d, f, ((1.0, -0.0015), (5.0, -0.0025)),
                       ("EUR", "USD"))


# --------------------------------------------------------------------------
# Immutability of the cached basis curve
# --------------------------------------------------------------------------
class TestMarketStateImmutability:
    """``foreign_curve_adjusted`` is cached; a mutable MarketState would let
    the cache go stale and silently price off the OLD basis curve."""

    def test_in_place_mutation_is_rejected(self, simple_market):
        for field, value in (("spot", 1.20),
                             ("basis_spreads", ((1.0, -0.05),)),
                             ("foreign_curve", _curve([0.01] * 6))):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(simple_market, field, value)

    def test_replace_rebuilds_the_cached_basis_curve(self, simple_market):
        f0 = market_forward(simple_market, 5.0)
        wide = simple_market.replace(basis_spreads=((1.0, -0.05), (5.0, -0.05)))
        f1 = market_forward(wide, 5.0)
        # -50bp of basis over 5y must move the 5y forward materially...
        assert f1 / f0 == pytest.approx(np.exp(0.0475 * 5.0), rel=1e-9)
        # ...and must not disturb the original object.
        assert market_forward(simple_market, 5.0) == pytest.approx(f0, abs=0.0)

    def test_replace_rebuilds_cache_after_foreign_curve_change(self,
                                                              simple_market):
        base = market_forward(simple_market, 5.0)
        bumped = simple_market.replace(
            foreign_curve=simple_market.foreign_curve.parallel_shift(100.0))
        assert market_forward(bumped, 5.0) < base  # higher r_f -> lower F

    def test_risk_bumps_do_not_leak_between_calls(self, simple_market):
        fwd = FXForward(10e6, market_forward(simple_market, 2.0), 2.0)
        v_before = fwd.value(simple_market)
        dv01(fwd, simple_market, "USD")
        dv01(fwd, simple_market, "EUR")
        basis_dv01(fwd, simple_market)
        assert fwd.value(simple_market) == pytest.approx(v_before, abs=0.0)


# --------------------------------------------------------------------------
# Curve behaviour at and beyond the pillars
# --------------------------------------------------------------------------
class TestCurveLimits:
    def test_df_at_zero_is_one(self):
        c = _curve([0.05] * 6)
        assert c.df(0.0) == pytest.approx(1.0, abs=1e-15)

    def test_zero_rate_at_zero_rejected(self):
        c = _curve([0.05] * 6)
        with pytest.raises(ValueError, match="t > 0"):
            c.zero_rate(0.0)

    def test_negative_time_rejected(self):
        c = _curve([0.05] * 6)
        with pytest.raises(ValueError, match="negative time"):
            c.df(-0.5)

    def test_flat_forward_extrapolation_beyond_last_pillar(self):
        # Log-linear DF interpolation => piecewise-constant instantaneous
        # forwards; beyond the last pillar the last segment's forward is
        # extrapolated flat, so f(10,20) == f(5,10) exactly.
        c = _curve([0.03, 0.032, 0.034, 0.036, 0.038, 0.040])
        assert c.forward_rate(10.0, 20.0) == pytest.approx(
            c.forward_rate(5.0, 10.0), rel=1e-12)

    def test_single_pillar_curve_is_flat(self):
        c = DiscountCurve([1.0], [np.exp(-0.04)], "flat")
        for t in (0.1, 1.0, 7.5):
            assert c.zero_rate(t) == pytest.approx(0.04, rel=1e-12)

    def test_df_above_one_allowed_for_negative_rates(self):
        c = DiscountCurve.from_zero_rates([1.0, 5.0], [-0.006, -0.004], "CHF")
        assert c.df(1.0) > 1.0 and c.df(5.0) > 1.0
        assert c.zero_rate(3.0) < 0.0

    @pytest.mark.parametrize("bad", [
        dict(times=[1.0, 1.0], dfs=[0.99, 0.98]),      # duplicate pillars
        dict(times=[2.0, 1.0], dfs=[0.98, 0.99]),      # not increasing
        dict(times=[0.0, 1.0], dfs=[1.0, 0.99]),       # zero first pillar
        dict(times=[1.0], dfs=[0.0]),                  # zero DF
        dict(times=[1.0], dfs=[-0.5]),                 # negative DF
        dict(times=[1.0, 2.0], dfs=[0.99]),            # length mismatch
        dict(times=[], dfs=[]),                        # empty
        dict(times=[1.0], dfs=[np.nan]),               # NaN
        dict(times=[np.inf], dfs=[0.99]),              # Inf
    ])
    def test_invalid_curve_inputs_rejected(self, bad):
        with pytest.raises(ValueError):
            DiscountCurve(**bad)

    def test_pillar_shift_is_local(self):
        c = _curve([0.03, 0.032, 0.034, 0.036, 0.038, 0.040])
        shifted = c.pillar_shift(3, 25.0)  # the 2.0y pillar
        assert shifted.df(0.25) == pytest.approx(c.df(0.25), rel=1e-14)
        assert shifted.df(2.0) != pytest.approx(c.df(2.0), rel=1e-14)
        assert shifted.df(10.0) == pytest.approx(c.df(10.0), rel=1e-14)

    def test_pillar_shift_does_not_mutate_the_original(self):
        c = _curve([0.03, 0.032, 0.034, 0.036, 0.038, 0.040])
        z_before = c.zero_rates.copy()
        c.pillar_shift(2, 100.0)
        np.testing.assert_allclose(c.zero_rates, z_before, atol=0.0)

    def test_pillar_index_out_of_range_rejected(self):
        c = _curve([0.03] * 6)
        for bad in (-1, 6, 99):
            with pytest.raises(ValueError, match="out of range"):
                c.pillar_shift(bad, 1.0)


# --------------------------------------------------------------------------
# CIP violations and the arbitrage band
# --------------------------------------------------------------------------
class TestCIPViolations:
    BASE = dict(spot_bid=1.0849, spot_ask=1.0851,
                dom_rate_bid=0.0525, dom_rate_ask=0.0535,
                for_rate_bid=0.0375, for_rate_ask=0.0385, tau=0.25)

    def _quotes(self, fwd_mid, half_spread=0.00005):
        return CIPQuotes(fwd_bid=fwd_mid - half_spread,
                         fwd_ask=fwd_mid + half_spread, **self.BASE)

    def test_forward_inside_the_band_is_not_arbitrage(self):
        lo, hi = no_arb_bounds(self._quotes(1.0890))
        mid = 0.5 * (lo + hi)
        res = detect_cip_arbitrage(self._quotes(mid))
        assert not res.is_arbitrage and res.direction == "none"
        assert res.pnl == 0.0

    def test_forward_far_above_the_band_is_sell_forward_arbitrage(self):
        _, hi = no_arb_bounds(self._quotes(1.0890))
        res = detect_cip_arbitrage(self._quotes(hi * 1.01))
        assert res.is_arbitrage and res.direction == "sell_forward"
        assert res.pnl > 0.0

    def test_forward_far_below_the_band_is_buy_forward_arbitrage(self):
        lo, _ = no_arb_bounds(self._quotes(1.0890))
        res = detect_cip_arbitrage(self._quotes(lo * 0.99))
        assert res.is_arbitrage and res.direction == "buy_forward"
        assert res.pnl > 0.0

    def test_band_is_non_empty_and_ordered(self):
        lo, hi = no_arb_bounds(self._quotes(1.0890))
        assert 0.0 < lo < hi

    def test_wider_spreads_widen_the_band(self):
        narrow = no_arb_bounds(self._quotes(1.089))
        wide_q = CIPQuotes(
            spot_bid=1.0840, spot_ask=1.0860, fwd_bid=1.0885, fwd_ask=1.0895,
            dom_rate_bid=0.0500, dom_rate_ask=0.0560,
            for_rate_bid=0.0350, for_rate_ask=0.0410, tau=0.25)
        wide = no_arb_bounds(wide_q)
        assert wide[0] < narrow[0] and wide[1] > narrow[1]

    def test_min_pnl_threshold_suppresses_marginal_signals(self):
        _, hi = no_arb_bounds(self._quotes(1.0890))
        q = self._quotes(hi * 1.0005)
        assert detect_cip_arbitrage(q).is_arbitrage
        assert not detect_cip_arbitrage(q, min_pnl=1.0).is_arbitrage

    def test_negative_deposit_rates_are_legal(self):
        q = CIPQuotes(spot_bid=1.0849, spot_ask=1.0851,
                      fwd_bid=1.0900, fwd_ask=1.0902,
                      dom_rate_bid=0.0020, dom_rate_ask=0.0030,
                      for_rate_bid=-0.0060, for_rate_ask=-0.0050, tau=0.25)
        lo, hi = no_arb_bounds(q)
        assert lo > 0 and hi > lo

    def test_crossed_market_rejected(self):
        with pytest.raises(ValueError, match="bid .* exceeds ask"):
            CIPQuotes(spot_bid=1.0851, spot_ask=1.0849,
                      fwd_bid=1.089, fwd_ask=1.090,
                      dom_rate_bid=0.05, dom_rate_ask=0.055,
                      for_rate_bid=0.03, for_rate_ask=0.035, tau=0.25)

    def test_nonpositive_tenor_rejected(self):
        with pytest.raises(ValueError, match="tau"):
            CIPQuotes(spot_bid=1.0849, spot_ask=1.0851, fwd_bid=1.089,
                      fwd_ask=1.090, dom_rate_bid=0.05, dom_rate_ask=0.055,
                      for_rate_bid=0.03, for_rate_ask=0.035, tau=0.0)


# --------------------------------------------------------------------------
# Basis economics and round-trips
# --------------------------------------------------------------------------
class TestBasisEconomics:
    def test_zero_basis_reproduces_the_input_curve_exactly(self):
        f = _curve([0.039, 0.038, 0.037, 0.034, 0.030, 0.028], "EUR")
        adj = basis_adjusted_curve(f, ((1.0, 0.0), (5.0, 0.0)))
        for t in (0.1, 1.0, 3.3, 5.0, 12.0):
            assert adj.df(t) == pytest.approx(f.df(t), rel=1e-14)

    def test_empty_basis_returns_the_same_object(self):
        f = _curve([0.03] * 6, "EUR")
        assert basis_adjusted_curve(f, ()) is f

    def test_negative_basis_lifts_forwards_above_pure_cip(self, simple_market):
        # EURUSD basis is negative post-2008: EUR DFs rise, so the market
        # forward sits ABOVE the pure-CIP forward.
        for t in (1.0, 5.0):
            f_cip = cip_forward(simple_market.spot,
                                simple_market.domestic_curve,
                                simple_market.foreign_curve, t)
            assert market_forward(simple_market, t) > f_cip

    def test_implied_basis_round_trips_through_the_forwards(self,
                                                            simple_market):
        tenors = (0.5, 1.0, 2.0, 5.0)
        fwds = [(t, market_forward(simple_market, t)) for t in tenors]
        implied = implied_basis_from_forwards(
            simple_market.spot, simple_market.domestic_curve,
            simple_market.foreign_curve, fwds)
        for t, s in implied:
            expected = np.interp(t, [1.0, 5.0], [-0.0015, -0.0025])
            assert s == pytest.approx(float(expected), abs=1e-12)

    def test_curve_from_fx_forwards_reprices_the_forwards(self, simple_market):
        tenors = (0.5, 1.0, 2.0, 5.0)
        fwds = [(t, market_forward(simple_market, t)) for t in tenors]
        implied_curve = curve_from_fx_forwards(
            simple_market.spot, simple_market.domestic_curve, fwds)
        for t, f in fwds:
            assert cip_forward(simple_market.spot,
                               simple_market.domestic_curve,
                               implied_curve, t) == pytest.approx(f, rel=1e-14)

    def test_basis_table_recovers_the_quoted_spread(self, simple_market):
        tab = forward_points_table(simple_market, [1.0, 5.0])
        assert tab.loc[0, "basis_spread_bp"] == pytest.approx(-15.0, abs=1e-8)
        assert tab.loc[1, "basis_spread_bp"] == pytest.approx(-25.0, abs=1e-8)
        # Negative basis => market points above CIP points.
        assert (tab["basis_points_effect"] > 0).all()

    def test_basis_dv01_is_defined_even_without_quoted_spreads(self):
        d = _curve([0.05] * 6, "USD")
        f = _curve([0.03] * 6, "EUR")
        m = MarketState(1.0850, d, f, (), ("EUR", "USD"))
        fwd = FXForward(10e6, market_forward(m, 2.0), 2.0)
        b = basis_dv01(fwd, m)
        assert np.isfinite(b) and b != 0.0

    def test_basis_dv01_sign_flips_with_the_position(self, simple_market):
        k = market_forward(simple_market, 3.0)
        long_ = FXForward(10e6, k, 3.0)
        short = FXForward(-10e6, k, 3.0)
        assert basis_dv01(long_, simple_market) == pytest.approx(
            -basis_dv01(short, simple_market), rel=1e-10)


# --------------------------------------------------------------------------
# Pegged / managed pairs and deep-negative rates
# --------------------------------------------------------------------------
class TestPeggedAndNegativeRegimes:
    def test_pegged_pair_forward_is_driven_purely_by_the_rate_gap(self):
        # USDHKD-style: spot glued to 7.80, but HKD and USD rates diverge,
        # so the forward points are pure carry -- and can be large even
        # though realised spot vol is ~zero.
        d = _curve([0.053, 0.052, 0.051, 0.048, 0.044, 0.042], "HKD")
        f = _curve([0.055, 0.055, 0.054, 0.050, 0.046, 0.044], "USD")
        m = MarketState(7.80, d, f, (), ("USD", "HKD"))
        pts = forward_points(m, 1.0, point_factor=1e4)
        assert pts < 0  # USD yields more -> forward discount on USD
        assert abs(pts) > 100  # over 100 pips: economically material

    def test_deep_negative_rates_keep_dfs_positive_and_above_one(self):
        # CHF/JPY-era: -0.75% out to 10y.
        f = DiscountCurve.from_zero_rates(
            [0.25, 1.0, 5.0, 10.0], [-0.0075, -0.0075, -0.0070, -0.0060],
            "CHF")
        d = _curve([0.053, 0.052, 0.051, 0.048, 0.044, 0.042], "USD")
        m = MarketState(1.1200, d, f, (), ("CHF", "USD"))
        assert (f.dfs > 1.0).all()
        for t in (0.25, 1.0, 5.0, 10.0):
            assert market_forward(m, t) > m.spot  # r_d > r_f => premium

    def test_both_curves_negative_still_prices_and_risks(self):
        d = DiscountCurve.from_zero_rates([1.0, 5.0], [-0.0050, -0.0040],
                                          "EUR")
        f = DiscountCurve.from_zero_rates([1.0, 5.0], [-0.0075, -0.0070],
                                          "CHF")
        m = MarketState(0.9400, d, f, (), ("CHF", "EUR"))
        fwd = FXForward(10e6, market_forward(m, 3.0), 3.0, ("CHF", "EUR"))
        assert fwd.value(m) == pytest.approx(0.0, abs=1e-6)
        assert np.isfinite(dv01(fwd, m, "EUR"))
        assert np.isfinite(dv01(fwd, m, "CHF"))
        assert fx_delta(fwd, m) > 0

    def test_zero_rate_gap_gives_flat_forward_curve(self):
        c = _curve([0.03] * 6, "X")
        m = MarketState(1.25, c, _curve([0.03] * 6, "Y"), (), ("EUR", "USD"))
        for t in (0.25, 1.0, 5.0, 10.0):
            assert market_forward(m, t) == pytest.approx(1.25, rel=1e-13)
            assert forward_points(m, t) == pytest.approx(0.0, abs=1e-8)


# --------------------------------------------------------------------------
# Degenerate positions, books and inputs
# --------------------------------------------------------------------------
class TestDegenerateInputs:
    def test_zero_notional_forward_has_zero_pv_and_zero_risk(self,
                                                             simple_market):
        fwd = FXForward(0.0, 1.10, 2.0)
        assert fwd.value(simple_market) == 0.0
        assert fx_delta(fwd, simple_market) == 0.0
        assert dv01(fwd, simple_market, "USD") == 0.0
        assert basis_dv01(fwd, simple_market) == 0.0

    def test_empty_book_reports_a_zero_total_row(self, simple_market):
        rep = book_risk_report([], simple_market)
        assert list(rep.index) == ["TOTAL"]
        assert (rep.loc["TOTAL"].to_numpy() == 0.0).all()
        assert book_value([], simple_market) == 0.0

    @pytest.mark.parametrize("kwargs", [
        dict(notional_base=1e6, strike=0.0, expiry=1.0),
        dict(notional_base=1e6, strike=-1.10, expiry=1.0),
        dict(notional_base=1e6, strike=1.10, expiry=0.0),
        dict(notional_base=1e6, strike=1.10, expiry=-1.0),
    ])
    def test_invalid_forward_inputs_rejected(self, kwargs):
        with pytest.raises(ValueError):
            FXForward(**kwargs)

    def test_same_currency_cross_rejected_everywhere(self):
        c = _curve([0.03] * 6)
        with pytest.raises(ValueError, match="same-currency"):
            MarketState(1.0, c, c, (), ("USD", "USD"))
        with pytest.raises(ValueError, match="same-currency"):
            FXForward(1e6, 1.0, 1.0, ("EUR", "EUR"))

    def test_fx_swap_requires_ordered_legs(self):
        with pytest.raises(ValueError, match="near_expiry < far_expiry"):
            FXSwap(1e6, 1.09, 1.0, 1.10, 0.5)
        with pytest.raises(ValueError, match="near_expiry < far_expiry"):
            FXSwap(1e6, 1.09, 1.0, 1.10, 1.0)

    def test_market_pair_mismatch_rejected(self, simple_market):
        fwd = FXForward(1e6, 1.10, 1.0, ("GBP", "USD"))
        with pytest.raises(ValueError, match="does not match"):
            fwd.value(simple_market)

    def test_nonpositive_spot_rejected(self):
        c = _curve([0.03] * 6)
        for bad in (0.0, -1.08):
            with pytest.raises(ValueError, match="spot must be > 0"):
                MarketState(bad, c, c, (), ("EUR", "USD"))

    def test_bootstrap_requires_deposits(self):
        with pytest.raises(ValueError, match="at least one deposit"):
            bootstrap_curve([], [(2.0, 0.04)])

    def test_bootstrap_rejects_duplicate_and_fractional_quotes(self):
        deps = [(0.25, 0.05), (1.0, 0.05)]
        with pytest.raises(ValueError, match="duplicate deposit"):
            bootstrap_curve([(1.0, 0.05), (1.0, 0.05)], [])
        with pytest.raises(ValueError, match="integers >= 2"):
            bootstrap_curve(deps, [(2.5, 0.04)])
        with pytest.raises(ValueError, match="duplicate swap"):
            bootstrap_curve(deps, [(2.0, 0.04), (2.0, 0.041)])

    def test_deposit_df_identity_and_negative_rates(self):
        assert df_from_deposit(0.05, 0.25) == pytest.approx(1 / 1.0125)
        assert df_from_deposit(-0.006, 1.0) > 1.0
        with pytest.raises(ValueError, match="accrual fraction must be > 0"):
            df_from_deposit(0.05, 0.0)

    def test_quoted_swaps_reprice_to_machine_precision(self):
        deps = [(0.25, 0.053), (0.5, 0.052), (1.0, 0.051)]
        swaps = [(2.0, 0.048), (5.0, 0.044), (10.0, 0.042)]
        c = bootstrap_curve(deps, swaps, "USD")
        for n, r in swaps:
            assert par_swap_rate(c, int(n)) == pytest.approx(r, abs=1e-12)

    @pytest.mark.parametrize("bad", ["", "X", "3", "M", "3X", "-2Y", "0Y",
                                     "abc"])
    def test_unparseable_tenors_rejected(self, bad):
        with pytest.raises(ValueError):
            tenor_to_years(bad)

    def test_year_fraction_zero_and_reversed(self):
        d = dt.date(2024, 3, 15)
        assert year_fraction(d, d, "ACT/365F") == 0.0
        with pytest.raises(ValueError, match="precedes"):
            year_fraction(d, dt.date(2024, 3, 14))
        with pytest.raises(ValueError, match="Unknown day-count"):
            year_fraction(d, dt.date(2024, 6, 15), "ACT/ACT")


# --------------------------------------------------------------------------
# Carry / roll economics
# --------------------------------------------------------------------------
class TestCarryEconomics:
    def test_long_low_yielder_rolls_down(self, simple_market):
        # EUR yields less than USD -> EURUSD forward points are positive,
        # so a long EUR forward rolls DOWN the points curve: negative carry.
        k = market_forward(simple_market, 1.0)
        fwd = FXForward(10e6, k, 1.0)
        out = forward_carry(fwd, simple_market, 0.5)
        assert out["forward_end"] < out["forward_start"]
        assert out["points_roll"] < 0.0
        assert out["carry_pnl"] < 0.0

    def test_short_position_carry_is_the_mirror_image(self, simple_market):
        k = market_forward(simple_market, 1.0)
        long_ = forward_carry(FXForward(10e6, k, 1.0), simple_market, 0.5)
        short = forward_carry(FXForward(-10e6, k, 1.0), simple_market, 0.5)
        assert long_["carry_pnl"] == pytest.approx(-short["carry_pnl"],
                                                   rel=1e-12)

    def test_horizon_must_lie_inside_the_remaining_life(self, simple_market):
        fwd = FXForward(10e6, 1.10, 1.0)
        for bad in (0.0, -0.25, 1.0, 2.0):
            with pytest.raises(ValueError, match="horizon"):
                forward_carry(fwd, simple_market, bad)

    def test_scenario_leaves_the_base_market_untouched(self, simple_market):
        f0 = market_forward(simple_market, 5.0)
        sc = Scenario("shock", spot_pct=-10.0, domestic_bp=100.0,
                      foreign_bp=-50.0, basis_bp=-75.0)
        shocked = apply_scenario(simple_market, sc)
        assert market_forward(simple_market, 5.0) == pytest.approx(f0, abs=0.0)
        assert shocked.spot == pytest.approx(simple_market.spot * 0.90)
        assert market_forward(shocked, 5.0) != pytest.approx(f0)

    def test_zero_scenario_is_a_no_op(self, simple_market):
        shocked = apply_scenario(simple_market, Scenario("flat"))
        for t in (0.5, 2.0, 10.0):
            assert market_forward(shocked, t) == pytest.approx(
                market_forward(simple_market, t), rel=1e-14)
