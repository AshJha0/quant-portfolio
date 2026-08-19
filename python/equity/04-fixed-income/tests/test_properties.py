"""Property-based invariants for the curve, bond and risk layers.

These pin the *shape* of the pricing functions rather than individual values,
so they hold for any correct implementation and catch sign errors, wrong
compounding, or broken interpolation immediately:

* curve identities — P(0)=1, forward/zero/df consistency, pillar exactness,
  par-rate reprices par, partition-of-unity of the key-rate triangles;
* bond monotonicity and convexity in yield, pull-to-par, duration ordering;
* analytic vs numerical duration/convexity;
* scale/linearity of the portfolio aggregates;
* zero-net-value books, negative rates, zero coupons and other edge cases.
"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import pytest

import fi_rates as fr
from fi_rates.curve import ExtrapolationWarning
from fi_rates.risk import ZeroNetValueWarning

D = dt.date
PILLARS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]


# --------------------------------------------------------------------------- #
# curve identities
# --------------------------------------------------------------------------- #
class TestCurveIdentities:
    @pytest.mark.parametrize("interp", fr.INTERPOLATIONS)
    def test_df_at_time_zero_is_one(self, interp):
        c = fr.DiscountCurve.from_zero_rates(PILLARS, [0.03] * len(PILLARS), interp)
        assert c.df(0.0) == pytest.approx(1.0, abs=1e-15)
        assert c.df(-1.0) == pytest.approx(1.0, abs=1e-15)

    @pytest.mark.parametrize("interp", fr.INTERPOLATIONS)
    def test_pillars_are_reproduced_exactly(self, interp):
        zeros = [0.020, 0.024, 0.028, 0.031, 0.034, 0.036, 0.037]
        c = fr.DiscountCurve.from_zero_rates(PILLARS, zeros, interp)
        np.testing.assert_allclose(c.df(np.array(PILLARS)),
                                   np.exp(-np.array(zeros) * np.array(PILLARS)),
                                   rtol=1e-12)
        np.testing.assert_allclose(c.zero_rate(np.array(PILLARS)), zeros, rtol=1e-12)

    @pytest.mark.parametrize("interp", fr.INTERPOLATIONS)
    def test_df_zero_rate_round_trip(self, interp):
        zeros = [0.020, 0.024, 0.028, 0.031, 0.034, 0.036, 0.037]
        c = fr.DiscountCurve.from_zero_rates(PILLARS, zeros, interp)
        t = np.array([0.3, 0.9, 3.0, 7.5, 20.0])
        z = np.asarray(c.zero_rate(t))
        np.testing.assert_allclose(np.asarray(c.df(t)), np.exp(-z * t), rtol=1e-12)

    @pytest.mark.parametrize("interp", fr.INTERPOLATIONS)
    def test_forward_rate_reproduces_discount_factors(self, interp):
        c = fr.DiscountCurve.from_zero_rates(
            PILLARS, [0.02, 0.024, 0.028, 0.031, 0.034, 0.036, 0.037], interp
        )
        for t1, t2 in [(0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0)]:
            f = c.forward_rate(t1, t2)
            assert float(np.asarray(c.df(t2))) == pytest.approx(
                float(np.asarray(c.df(t1))) * np.exp(-f * (t2 - t1)), rel=1e-12
            )

    def test_simple_and_continuous_forwards_are_consistent(self):
        c = fr.DiscountCurve.from_zero_rates(PILLARS, [0.03] * len(PILLARS))
        t1, t2 = 1.0, 2.0
        f_c = c.forward_rate(t1, t2)
        f_s = c.simple_forward_rate(t1, t2)
        assert f_s == pytest.approx((np.exp(f_c * (t2 - t1)) - 1.0) / (t2 - t1), rel=1e-12)
        assert f_s > f_c  # simple compounding quotes higher for positive rates

    def test_flat_curve_has_flat_forwards_equal_to_the_zero(self):
        c = fr.DiscountCurve.from_zero_rates(PILLARS, [0.035] * len(PILLARS))
        for t1, t2 in [(0.25, 0.5), (1.0, 2.0), (5.0, 10.0)]:
            assert c.forward_rate(t1, t2) == pytest.approx(0.035, abs=1e-12)

    def test_positive_rates_give_decreasing_discount_factors(self):
        c = fr.DiscountCurve.from_zero_rates(PILLARS, [0.03] * len(PILLARS))
        t = np.linspace(0.01, 30.0, 400)
        p = np.asarray(c.df(t))
        assert np.all(np.diff(p) < 0)
        assert np.all(p > 0)

    def test_negative_rates_give_discount_factors_above_one(self):
        c = fr.DiscountCurve.from_zero_rates(PILLARS, [-0.005] * len(PILLARS))
        t = np.linspace(0.01, 30.0, 200)
        p = np.asarray(c.df(t))
        assert np.all(p > 1.0)
        assert np.all(np.diff(p) > 0)  # rising with maturity

    def test_par_rate_reprices_a_par_bond(self):
        c = fr.DiscountCurve.from_zero_rates(
            PILLARS, [0.02, 0.024, 0.028, 0.031, 0.034, 0.036, 0.037]
        )
        for maturity, freq in [(2.0, 1), (5.0, 2), (10.0, 2), (10.0, 4)]:
            par = c.par_rate(maturity, freq)
            n = int(round(maturity * freq))
            t = np.arange(1, n + 1) / freq
            dfs = np.asarray(c.df(t))
            pv = float(np.sum(par / freq * dfs) + dfs[-1])
            assert pv == pytest.approx(1.0, abs=1e-12)

    def test_flat_curve_par_rate_matches_the_equivalent_compounded_rate(self):
        z, freq = 0.04, 2
        c = fr.DiscountCurve.from_zero_rates(PILLARS, [z] * len(PILLARS))
        expected = freq * (np.exp(z / freq) - 1.0)
        assert c.par_rate(10.0, freq) == pytest.approx(expected, rel=1e-12)

    def test_extrapolation_beyond_last_pillar_warns_and_stays_flat(self):
        c = fr.DiscountCurve.from_zero_rates([1.0, 5.0, 10.0], [0.03, 0.035, 0.04])
        with pytest.warns(ExtrapolationWarning):
            z_far = c.zero_rate(50.0)
        assert z_far == pytest.approx(0.04, rel=1e-12)
        with pytest.warns(ExtrapolationWarning):
            p_far = float(np.asarray(c.df(50.0)))
        assert p_far == pytest.approx(np.exp(-0.04 * 50.0), rel=1e-12)

    def test_short_end_is_silent_and_consistent_across_schemes(self):
        zeros = [0.02, 0.03, 0.04]
        vals = []
        for interp in fr.INTERPOLATIONS:
            c = fr.DiscountCurve.from_zero_rates([1.0, 5.0, 10.0], zeros, interp)
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # must NOT warn on the short end
                vals.append(float(np.asarray(c.df(0.1))))
        assert all(v == pytest.approx(vals[0], rel=1e-12) for v in vals)

    def test_parallel_bump_shifts_every_pillar_zero_exactly(self):
        c = fr.DiscountCurve.from_zero_rates(PILLARS, [0.03] * len(PILLARS))
        bumped = c.bumped_parallel(1e-4)
        np.testing.assert_allclose(bumped.zero_rates, c.zero_rates + 1e-4, rtol=1e-12)

    def test_curve_construction_validation(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            fr.DiscountCurve([1.0, 1.0, 2.0], [0.99, 0.98, 0.97])
        with pytest.raises(ValueError, match="must be > 0"):
            fr.DiscountCurve([0.0, 1.0], [1.0, 0.99])
        with pytest.raises(ValueError, match="discount factors"):
            fr.DiscountCurve([1.0, 2.0], [0.99, -0.1])
        with pytest.raises(ValueError, match="equal length"):
            fr.DiscountCurve([1.0, 2.0], [0.99])
        with pytest.raises(ValueError, match="unknown interpolation"):
            fr.DiscountCurve([1.0], [0.99], "cubic_spline")


# --------------------------------------------------------------------------- #
# bond price/yield shape
# --------------------------------------------------------------------------- #
class TestBondShape:
    def test_price_strictly_decreasing_in_yield(self, settlement, bond_5y):
        ys = [-0.02, 0.0, 0.02, 0.04, 0.06, 0.10, 0.20]
        ps = [fr.price_from_ytm(bond_5y, settlement, y) for y in ys]
        assert all(b < a for a, b in zip(ps, ps[1:]))

    def test_price_convex_in_yield(self, settlement, bond_5y):
        """Positive convexity: the second difference in yield is positive."""
        ys = np.linspace(0.0, 0.10, 11)
        ps = np.array([fr.price_from_ytm(bond_5y, settlement, y) for y in ys])
        assert np.all(np.diff(ps, 2) > 0)

    def test_par_bond_at_coupon_date_has_ytm_equal_to_coupon(self, settlement):
        for freq in (1, 2, 4):
            bond = fr.FixedRateBond(
                effective=settlement, maturity=D(2031, 8, 18), coupon=0.05,
                frequency=freq, daycount="30/360US",
            )
            price = fr.price_from_ytm(bond, settlement, 0.05)
            assert price == pytest.approx(100.0, abs=1e-9)
            assert fr.ytm_from_price(bond, settlement, 100.0) == pytest.approx(
                0.05, abs=1e-10
            )

    def test_premium_and_discount_bonds_are_labelled_correctly(self, settlement):
        bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2031, 8, 18), coupon=0.05, frequency=2
        )
        assert fr.price_from_ytm(bond, settlement, 0.03) > 100.0  # premium
        assert fr.price_from_ytm(bond, settlement, 0.07) < 100.0  # discount

    def test_ytm_round_trip_across_a_wide_yield_range(self, settlement, bond_5y):
        for y in (-0.02, -0.005, 0.0, 0.01, 0.05, 0.15, 0.5, 2.0):
            p = fr.price_from_ytm(bond_5y, settlement, y)
            clean = p - fr.accrued_interest(bond_5y, settlement)
            assert fr.ytm_from_price(bond_5y, settlement, clean) == pytest.approx(
                y, abs=1e-10
            )

    def test_zero_coupon_fixed_rate_bond_behaves_like_a_zcb(self, settlement):
        bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2031, 8, 18), coupon=0.0, frequency=2
        )
        y = 0.04
        p = fr.price_from_ytm(bond, settlement, y)
        n_periods = 5 * 2
        assert p == pytest.approx(100.0 * (1 + y / 2) ** (-n_periods), rel=1e-12)
        assert fr.accrued_interest(bond, settlement + dt.timedelta(days=60)) == 0.0
        # a zero-coupon bond's Macaulay duration is exactly its maturity
        assert fr.macaulay_duration(bond, settlement, y) == pytest.approx(5.0, rel=1e-12)

    def test_price_pulls_to_par_as_maturity_approaches(self, settlement):
        """A discount bond's price rises toward par as time passes."""
        curve = fr.DiscountCurve.from_zero_rates(PILLARS, [0.06] * len(PILLARS))
        bond = fr.FixedRateBond(
            effective=D(2020, 8, 18), maturity=D(2031, 8, 18), coupon=0.02, frequency=2
        )
        cleans = []
        for years_out in (5, 3, 1):
            s = D(2031 - years_out, 8, 18)
            cleans.append(fr.clean_price_from_curve(bond, s, curve))
        assert all(b > a for a, b in zip(cleans, cleans[1:]))
        assert all(c < 100.0 for c in cleans)

    def test_longer_maturity_has_longer_duration(self, settlement):
        durations = []
        for year in (2028, 2031, 2036, 2046):
            bond = fr.FixedRateBond(
                effective=settlement, maturity=D(year, 8, 18), coupon=0.04, frequency=2
            )
            durations.append(fr.modified_duration(bond, settlement, 0.04))
        assert all(b > a for a, b in zip(durations, durations[1:]))

    def test_lower_coupon_has_longer_duration_at_the_same_maturity(self, settlement):
        durations = []
        for coupon in (0.08, 0.05, 0.02, 0.0):
            bond = fr.FixedRateBond(
                effective=settlement, maturity=D(2046, 8, 18), coupon=coupon, frequency=2
            )
            durations.append(fr.modified_duration(bond, settlement, 0.04))
        assert all(b > a for a, b in zip(durations, durations[1:]))

    def test_macaulay_exceeds_modified_duration_for_positive_yield(self, settlement, bond_5y):
        mac = fr.macaulay_duration(bond_5y, settlement, 0.04)
        mod = fr.modified_duration(bond_5y, settlement, 0.04)
        assert mac > mod
        assert mod == pytest.approx(mac / (1 + 0.04 / 2), rel=1e-12)

    def test_duration_below_maturity_for_a_coupon_bond(self, settlement, bond_5y):
        assert 0 < fr.macaulay_duration(bond_5y, settlement, 0.04) < 5.0


# --------------------------------------------------------------------------- #
# analytic vs numerical risk
# --------------------------------------------------------------------------- #
class TestRiskConsistency:
    @pytest.mark.parametrize("ytm", [-0.01, 0.0, 0.02, 0.05, 0.12])
    def test_analytic_duration_matches_finite_difference(self, settlement, bond_5y, ytm):
        assert fr.modified_duration(bond_5y, settlement, ytm) == pytest.approx(
            fr.numerical_modified_duration(bond_5y, settlement, ytm), rel=1e-6
        )

    @pytest.mark.parametrize("ytm", [0.0, 0.02, 0.05, 0.12])
    def test_analytic_convexity_matches_finite_difference(self, settlement, bond_5y, ytm):
        assert fr.convexity(bond_5y, settlement, ytm) == pytest.approx(
            fr.numerical_convexity(bond_5y, settlement, ytm), rel=1e-5
        )

    def test_convexity_is_positive_for_a_vanilla_bond(self, settlement, bond_5y):
        for y in (-0.01, 0.0, 0.04, 0.10):
            assert fr.convexity(bond_5y, settlement, y) > 0

    def test_dv01_is_positive_and_matches_duration_times_price(self, settlement, bond_5y):
        y = 0.04
        p = fr.price_from_ytm(bond_5y, settlement, y)
        d = fr.modified_duration(bond_5y, settlement, y)
        assert fr.dv01(bond_5y, settlement, y) == pytest.approx(d * p * 1e-4, rel=1e-12)
        assert fr.dv01(bond_5y, settlement, y) > 0

    def test_curve_dv01_close_to_ytm_dv01_on_a_flat_curve(self, settlement, flat_curve):
        """Flat continuous 4% curve: the two DV01 definitions must broadly
        agree. They are *not* identical — the curve DV01 bumps a continuously
        compounded zero while the YTM DV01 bumps a semiannual street yield, so
        a ~2% gap at 10y is expected and is a convention difference, not an
        error. Anything much larger would signal a real bug."""
        bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2036, 8, 18), coupon=0.04, frequency=2
        )
        clean = fr.clean_price_from_curve(bond, settlement, flat_curve)
        y = fr.ytm_from_price(bond, settlement, clean)
        assert fr.dv01_curve(bond, settlement, flat_curve) == pytest.approx(
            fr.dv01(bond, settlement, y), rel=0.05
        )

    def test_duration_convexity_beats_duration_alone(self, settlement, bond_5y):
        tbl = fr.pnl_approximation_table(bond_5y, settlement, 0.04)
        assert np.all(tbl["err_dur_conv"].abs() < tbl["err_duration"].abs())

    def test_duration_only_always_understates_the_price_rise(self, settlement, bond_5y):
        """Positive convexity ⇒ the linear estimate is below the true price
        for a move in either direction."""
        tbl = fr.pnl_approximation_table(bond_5y, settlement, 0.04)
        assert np.all(tbl["err_duration"] < 0)

    def test_taylor_error_grows_with_shock_size(self, settlement, bond_5y):
        tbl = fr.pnl_approximation_table(
            bond_5y, settlement, 0.04, shocks_bp=(25, 50, 100, 200, 400)
        )
        errs = tbl["err_duration"].abs().to_numpy()
        assert all(b > a for a, b in zip(errs, errs[1:]))


# --------------------------------------------------------------------------- #
# key-rate durations
# --------------------------------------------------------------------------- #
class TestKeyRateProperties:
    def test_triangle_weights_are_a_partition_of_unity(self):
        t = np.linspace(0.01, 40.0, 500)
        w = fr.triangle_weights(fr.DEFAULT_KEY_TENORS, t)
        np.testing.assert_allclose(w.sum(axis=0), 1.0, rtol=1e-12)
        assert np.all(w >= 0.0)

    def test_triangle_weight_is_one_at_its_own_tenor(self):
        ks = fr.DEFAULT_KEY_TENORS
        w = fr.triangle_weights(ks, np.array(ks))
        np.testing.assert_allclose(w, np.eye(len(ks)), atol=1e-12)

    def test_triangle_weights_validation(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            fr.triangle_weights([5.0, 2.0], np.array([1.0]))
        with pytest.raises(ValueError, match="non-empty"):
            fr.triangle_weights([], np.array([1.0]))

    @pytest.mark.parametrize("interp", ["loglinear_df", "linear_zero"])
    def test_krdv01s_sum_to_the_parallel_dv01_for_local_schemes(self, settlement, interp):
        """Partition-of-unity ⇒ the key-rate bumps add up to a parallel bump.
        For *local* interpolation the only residual is the second-order
        cross-gamma of the finite differences: measured ~3-4e-7 relative."""
        c = fr.DiscountCurve.from_zero_rates(
            PILLARS, [0.02, 0.024, 0.028, 0.031, 0.034, 0.036, 0.037], interp
        )
        bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2046, 8, 18), coupon=0.04, frequency=2
        )
        krdv = fr.key_rate_dv01s(bond, settlement, c)
        assert krdv.sum() == pytest.approx(fr.dv01_curve(bond, settlement, c), rel=1e-6)

    def test_krdv01_additivity_is_looser_under_pchip(self, settlement):
        """PCHIP is non-local: a pillar bump reshapes neighbouring segments, so
        the key-rate bumps no longer compose into an exact parallel bump. The
        residual is ~2e-4 relative — three orders of magnitude worse than the
        local schemes, and the reason the desk convention is a local
        interpolator for risk production (docs/METHODOLOGY.md)."""
        c = fr.DiscountCurve.from_zero_rates(
            PILLARS, [0.02, 0.024, 0.028, 0.031, 0.034, 0.036, 0.037], "pchip_zero"
        )
        bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2046, 8, 18), coupon=0.04, frequency=2
        )
        krdv = fr.key_rate_dv01s(bond, settlement, c)
        dv = fr.dv01_curve(bond, settlement, c)
        rel = abs(krdv.sum() / dv - 1.0)
        assert rel > 1e-6      # materially worse than the local schemes ...
        assert rel < 1e-3      # ... but still well inside a risk tolerance

    def test_krd_concentrates_at_the_bonds_own_maturity(self, settlement, curve):
        """A 10y bullet's key-rate exposure must peak at the 10y bucket."""
        bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2036, 8, 18), coupon=0.04, frequency=2
        )
        krdv = fr.key_rate_dv01s(bond, settlement, curve, key_tenors=(2.0, 5.0, 10.0, 30.0))
        assert int(np.argmax(krdv)) == 2  # the 10y bucket
        assert krdv[2] > 0.5 * krdv.sum()


# --------------------------------------------------------------------------- #
# bootstrap round trip
# --------------------------------------------------------------------------- #
class TestBootstrapProperties:
    @pytest.mark.parametrize("interp", ["loglinear_df", "linear_zero"])
    def test_local_schemes_reprice_every_instrument_exactly(self, interp):
        """For *local* interpolation (log-linear DF, linear zero) the
        sequential bootstrap is exact: adding a pillar cannot move the curve
        before it, so every quote reprices to machine precision."""
        from fi_rates.data import market_quotes

        quotes = market_quotes("upward", noise_bp=0.0)
        c = fr.bootstrap_curve(quotes, interpolation=interp)
        for _, err in fr.reprice_instruments(quotes, c):
            assert abs(err) < 1e-12

    def test_pchip_bootstrap_is_only_approximately_exact(self):
        """PCHIP is *non-local*: appending a pillar reshapes the spline over
        earlier segments, so the sequential bootstrap no longer reprices the
        already-solved instruments exactly. The residual is small but real
        (~4e-6 in rate terms here) — documented in the bootstrap module rather
        than papered over, because a desk quoting off a spline curve needs to
        know its own quotes are not reproduced to machine precision."""
        from fi_rates.data import market_quotes

        quotes = market_quotes("upward", noise_bp=0.0)
        c = fr.bootstrap_curve(quotes, interpolation="pchip_zero")
        errs = [abs(e) for _, e in fr.reprice_instruments(quotes, c)]
        assert max(errs) > 1e-12          # not exact ...
        assert max(errs) < 1e-4           # ... but well inside a basis point

    def test_bootstrap_is_independent_of_input_order(self):
        from fi_rates.data import market_quotes

        quotes = market_quotes("upward", noise_bp=0.0)
        a = fr.bootstrap_curve(quotes)
        b = fr.bootstrap_curve(list(reversed(quotes)))
        np.testing.assert_allclose(a.times, b.times, rtol=1e-12)
        np.testing.assert_allclose(a.dfs, b.dfs, rtol=1e-10)

    def test_duplicate_pillars_rejected(self):
        with pytest.raises(ValueError, match="duplicate pillar"):
            fr.bootstrap_curve([fr.Deposit(1.0, 0.03), fr.Deposit(1.0, 0.04)])

    def test_empty_instrument_list_rejected(self):
        with pytest.raises(ValueError, match="no instruments"):
            fr.bootstrap_curve([])

    def test_flat_deposit_quotes_give_a_flat_simple_curve(self):
        rate = 0.03
        quotes = [fr.Deposit(t, rate) for t in (0.25, 0.5, 1.0, 2.0)]
        c = fr.bootstrap_curve(quotes)
        for t in (0.25, 0.5, 1.0, 2.0):
            assert float(np.asarray(c.df(t))) == pytest.approx(
                1.0 / (1.0 + rate * t), rel=1e-12
            )


# --------------------------------------------------------------------------- #
# FRN and annuity identities
# --------------------------------------------------------------------------- #
class TestFRNAndAnnuity:
    @pytest.mark.parametrize("freq", [1, 2, 4])
    def test_zero_margin_frn_prices_at_par(self, settlement, curve, freq):
        """Telescoping identity — independent of the curve's shape."""
        p = fr.frn_price_from_curve(
            settlement, D(2031, 8, 18), curve, frequency=freq, quoted_margin=0.0
        )
        assert p == pytest.approx(100.0, abs=1e-9)

    def test_positive_margin_lifts_the_frn_above_par(self, settlement, curve):
        p = fr.frn_price_from_curve(
            settlement, D(2031, 8, 18), curve, frequency=4, quoted_margin=0.005
        )
        assert p > 100.0

    def test_annuity_pv_equals_sum_of_discount_factors(self, settlement, curve):
        pv = fr.annuity_pv(1.0, settlement, D(2031, 8, 18), curve, frequency=1)
        dates = fr.generate_schedule(settlement, D(2031, 8, 18), 1)
        t = np.array([fr.curve_time(settlement, d) for d in dates])
        assert pv == pytest.approx(float(np.sum(np.asarray(curve.df(t)))), rel=1e-12)

    def test_annuity_pv_linear_in_payment(self, settlement, curve):
        base = fr.annuity_pv(1.0, settlement, D(2031, 8, 18), curve)
        for c in (2.5, 100.0):
            assert fr.annuity_pv(c, settlement, D(2031, 8, 18), curve) == pytest.approx(
                c * base, rel=1e-12
            )


# --------------------------------------------------------------------------- #
# portfolio aggregation
# --------------------------------------------------------------------------- #
class TestPortfolioAggregation:
    def test_portfolio_value_linear_in_quantity(self, settlement, curve, bond_5y):
        base = fr.portfolio_value([fr.Position(bond_5y, 1.0)], settlement, curve)
        for q in (2.0, -3.0, 1000.0):
            assert fr.portfolio_value(
                [fr.Position(bond_5y, q)], settlement, curve
            ) == pytest.approx(q * base, rel=1e-12)

    def test_portfolio_dv01_is_additive(self, settlement, curve):
        b1 = fr.FixedRateBond(settlement, D(2031, 8, 18), 0.04, 2)
        b2 = fr.FixedRateBond(settlement, D(2046, 8, 18), 0.03, 2)
        pos = [fr.Position(b1, 10.0), fr.Position(b2, -4.0)]
        rpt = fr.portfolio_risk(pos, settlement, curve)
        expected = (
            10.0 * fr.dv01_curve(b1, settlement, curve)
            - 4.0 * fr.dv01_curve(b2, settlement, curve)
        )
        assert float(rpt.loc["TOTAL", "dv01"]) == pytest.approx(expected, rel=1e-10)

    def test_weights_sum_to_one_for_a_long_only_book(self, settlement, curve):
        b1 = fr.FixedRateBond(settlement, D(2031, 8, 18), 0.04, 2)
        b2 = fr.FixedRateBond(settlement, D(2046, 8, 18), 0.03, 2)
        rpt = fr.portfolio_risk(
            [fr.Position(b1, 10.0), fr.Position(b2, 5.0)], settlement, curve
        )
        assert rpt.loc[rpt.index != "TOTAL", "weight"].sum() == pytest.approx(1.0, rel=1e-12)
        assert float(rpt.loc["TOTAL", "weight"]) == 1.0

    def test_zero_net_value_book_warns_and_reports_nan_weights(self, settlement, curve):
        """Regression test: a duration-neutral long/short book used to emit
        silent ±inf weights and NaN aggregates. It must now warn explicitly,
        report NaN for the undefined weighted quantities, and keep mv/dv01
        exact — DV01 is the aggregate that actually means something here."""
        b1 = fr.FixedRateBond(settlement, D(2031, 8, 18), 0.05, 2)
        b2 = fr.FixedRateBond(settlement, D(2036, 8, 18), 0.04, 2)
        v1 = fr.portfolio_value([fr.Position(b1, 1.0)], settlement, curve)
        v2 = fr.portfolio_value([fr.Position(b2, 1.0)], settlement, curve)
        pos = [fr.Position(b1, 1.0, label="long"), fr.Position(b2, -v1 / v2, label="short")]
        with pytest.warns(ZeroNetValueWarning, match="market-value weights are undefined"):
            rpt = fr.portfolio_risk(pos, settlement, curve)
        assert float(rpt.loc["TOTAL", "mv"]) == pytest.approx(0.0, abs=1e-9)
        assert np.isnan(rpt.loc["TOTAL", "mod_duration"])
        assert np.isnan(rpt.loc["TOTAL", "convexity"])
        assert rpt.loc[rpt.index != "TOTAL", "weight"].isna().all()
        assert np.isfinite(float(rpt.loc["TOTAL", "dv01"]))
        # no infinities anywhere (the actual defect)
        assert not np.isinf(rpt.select_dtypes("number").to_numpy()).any()

    def test_normal_book_does_not_warn(self, settlement, curve, bond_5y):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ZeroNetValueWarning)
            fr.portfolio_risk([fr.Position(bond_5y, 10.0)], settlement, curve)


# --------------------------------------------------------------------------- #
# scenarios, carry and roll-down
# --------------------------------------------------------------------------- #
class TestScenarioProperties:
    def test_zero_scenario_is_a_no_op(self, settlement, curve, bond_5y):
        shocked = fr.apply_scenario(curve, fr.parallel_scenario(0.0))
        np.testing.assert_allclose(shocked.zero_rates, curve.zero_rates, atol=1e-15)

    def test_parallel_scenario_pnl_sign_and_ordering(self, settlement, curve, bond_5y):
        pos = [fr.Position(bond_5y, 100.0)]
        tbl = fr.scenario_pnl_table(
            pos, settlement, curve,
            [fr.parallel_scenario(bp) for bp in (-100, -50, 50, 100)],
        )
        pnl = tbl["pnl_full"].to_numpy()
        assert pnl[0] > pnl[1] > 0 > pnl[2] > pnl[3]  # rates down = gain

    def test_steepener_and_flattener_are_mirror_images(self, settlement, curve):
        pos = [fr.Position(fr.FixedRateBond(settlement, D(2046, 8, 18), 0.04, 2), 100.0)]
        steep = fr.scenario_pnl_table(
            pos, settlement, curve, [fr.steepener_scenario(-50, 50)]
        )["pnl_full"].iloc[0]
        flat = fr.scenario_pnl_table(
            pos, settlement, curve, [fr.steepener_scenario(50, -50)]
        )["pnl_full"].iloc[0]
        assert steep * flat < 0  # opposite signs
        # Not exactly equal and opposite: positive convexity makes the gain
        # from falling rates exceed the loss from rising rates (~6% here).
        assert steep == pytest.approx(-flat, rel=0.12)
        assert abs(flat) > abs(steep)

    def test_historical_scenarios_have_the_documented_direction(self, settlement, curve):
        pos = [fr.Position(fr.FixedRateBond(settlement, D(2036, 8, 18), 0.04, 2), 100.0)]
        tbl = fr.scenario_pnl_table(
            pos, settlement, curve, list(fr.HISTORICAL_SCENARIOS.values())
        )
        assert tbl.loc["taper_tantrum_2013", "pnl_full"] < 0  # rates up -> loss
        assert tbl.loc["hiking_2022", "pnl_full"] < 0
        assert tbl.loc["gfc_2008", "pnl_full"] > 0  # rates down -> gain
        # 2022 was the bigger rate move, so the bigger loss
        assert (
            tbl.loc["hiking_2022", "pnl_full"]
            < tbl.loc["taper_tantrum_2013", "pnl_full"]
        )

    def test_scenario_validation(self):
        with pytest.raises(ValueError, match="equal length"):
            fr.Scenario("bad", (1.0, 2.0), (10.0,))
        with pytest.raises(ValueError, match="at least one tenor"):
            fr.Scenario("bad", (), ())
        with pytest.raises(ValueError, match="strictly increasing"):
            fr.Scenario("bad", (5.0, 2.0), (10.0, 20.0))
        with pytest.raises(ValueError, match="pivot"):
            fr.steepener_scenario(-50, 50, pivot=1.0)

    def test_carry_plus_rolldown_equals_total_static_pnl(self, settlement, curve):
        bond = fr.FixedRateBond(D(2024, 8, 18), D(2036, 8, 18), 0.04, 2)
        for months in (3, 6, 12, 24):
            horizon = fr.add_months(settlement, months)
            out = fr.carry_rolldown(bond, settlement, horizon, curve)
            assert out["total"] == pytest.approx(out["carry"] + out["rolldown"], rel=1e-12)

    def test_carry_is_positive_for_a_coupon_bond_on_an_upward_curve(self, settlement, curve):
        bond = fr.FixedRateBond(D(2024, 8, 18), D(2036, 8, 18), 0.04, 2)
        out = fr.carry_rolldown(bond, settlement, fr.add_months(settlement, 12), curve)
        assert out["carry"] > 0

    def test_carry_rolldown_validation(self, settlement, curve, bond_5y):
        with pytest.raises(ValueError, match="settlement < horizon"):
            fr.carry_rolldown(bond_5y, settlement, settlement, curve)
        with pytest.raises(ValueError, match="settlement < horizon"):
            fr.carry_rolldown(
                bond_5y, settlement, bond_5y.maturity + dt.timedelta(days=1), curve
            )


# --------------------------------------------------------------------------- #
# day count properties
# --------------------------------------------------------------------------- #
class TestDayCountProperties:
    @pytest.mark.parametrize("conv", fr.SUPPORTED_CONVENTIONS)
    def test_year_fraction_is_zero_on_identical_dates(self, conv):
        assert fr.year_fraction(D(2026, 3, 15), D(2026, 3, 15), conv) == 0.0

    @pytest.mark.parametrize("conv", fr.SUPPORTED_CONVENTIONS)
    def test_year_fraction_is_antisymmetric(self, conv):
        a, b = D(2025, 2, 10), D(2026, 7, 4)
        assert fr.year_fraction(a, b, conv) == pytest.approx(
            -fr.year_fraction(b, a, conv), rel=1e-12
        )

    @pytest.mark.parametrize("conv", fr.SUPPORTED_CONVENTIONS)
    def test_year_fraction_is_additive_across_a_split_date(self, conv):
        a, mid, b = D(2025, 2, 10), D(2025, 9, 30), D(2026, 7, 4)
        assert fr.year_fraction(a, b, conv) == pytest.approx(
            fr.year_fraction(a, mid, conv) + fr.year_fraction(mid, b, conv), rel=1e-12
        )

    @pytest.mark.parametrize("conv", fr.SUPPORTED_CONVENTIONS)
    def test_year_fraction_increases_with_the_end_date(self, conv):
        a = D(2025, 1, 1)
        fracs = [fr.year_fraction(a, D(2025 + k, 1, 1), conv) for k in (1, 2, 5, 10)]
        assert all(b > a_ for a_, b in zip(fracs, fracs[1:]))

    def test_act365f_exact_day_counts(self):
        assert fr.year_fraction(D(2025, 1, 1), D(2026, 1, 1), "ACT/365F") == pytest.approx(
            365 / 365
        )
        # 2024 is a leap year: ACT/365F over-counts by design
        assert fr.year_fraction(D(2024, 1, 1), D(2025, 1, 1), "ACT/365F") == pytest.approx(
            366 / 365
        )

    def test_act_act_isda_handles_the_leap_year_exactly(self):
        assert fr.year_fraction(
            D(2024, 1, 1), D(2025, 1, 1), "ACT/ACT-ISDA"
        ) == pytest.approx(1.0, rel=1e-12)
        assert fr.year_fraction(
            D(2025, 1, 1), D(2026, 1, 1), "ACT/ACT-ISDA"
        ) == pytest.approx(1.0, rel=1e-12)

    def test_thirty360_whole_years_are_exact(self):
        for k in (1, 2, 5, 10):
            assert fr.year_fraction(
                D(2025, 3, 15), D(2025 + k, 3, 15), "30/360US"
            ) == pytest.approx(float(k), rel=1e-12)

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError, match="Unknown day count"):
            fr.year_fraction(D(2025, 1, 1), D(2026, 1, 1), "ACT/364")

    def test_add_months_clamps_to_month_end(self):
        assert fr.add_months(D(2024, 1, 31), 1) == D(2024, 2, 29)  # leap
        assert fr.add_months(D(2025, 1, 31), 1) == D(2025, 2, 28)
        assert fr.add_months(D(2025, 3, 31), -1) == D(2025, 2, 28)
        assert fr.add_months(D(2025, 6, 15), 12) == D(2026, 6, 15)

    @pytest.mark.parametrize("freq", [1, 2, 4])
    def test_schedule_ends_at_maturity_and_is_increasing(self, freq):
        dates = fr.generate_schedule(D(2025, 3, 10), D(2031, 3, 10), freq)
        assert dates[-1] == D(2031, 3, 10)
        assert all(b > a for a, b in zip(dates, dates[1:]))
        assert all(d > D(2025, 3, 10) for d in dates)
        assert len(dates) == 6 * freq

    def test_schedule_produces_a_short_front_stub(self):
        dates = fr.generate_schedule(D(2025, 5, 1), D(2031, 3, 10), 2)
        assert dates[0] > D(2025, 5, 1)
        assert (dates[0] - D(2025, 5, 1)).days < 183

    def test_schedule_validation(self):
        with pytest.raises(ValueError, match="frequency"):
            fr.generate_schedule(D(2025, 1, 1), D(2030, 1, 1), 3)
        with pytest.raises(ValueError, match="after"):
            fr.generate_schedule(D(2030, 1, 1), D(2025, 1, 1), 2)
