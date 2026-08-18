"""Curve interpolation, derived rates, identities and extrapolation policy."""

from __future__ import annotations

import numpy as np
import pytest

from fi_rates.curve import INTERPOLATIONS, DiscountCurve, ExtrapolationWarning

TIMES = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
ZEROS = [0.030, 0.032, 0.035, 0.039, 0.042, 0.045]


@pytest.fixture(params=INTERPOLATIONS)
def any_curve(request) -> DiscountCurve:
    return DiscountCurve.from_zero_rates(TIMES, ZEROS, request.param)


class TestPillarRoundTrip:
    def test_dfs_reproduced_exactly(self, any_curve):
        expected = np.exp(-np.array(ZEROS) * np.array(TIMES))
        got = np.asarray(any_curve.df(np.array(TIMES)))
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)

    def test_zeros_reproduced_exactly(self, any_curve):
        got = np.asarray(any_curve.zero_rate(np.array(TIMES)))
        np.testing.assert_allclose(got, ZEROS, rtol=0, atol=1e-14)

    def test_df_at_zero_is_one(self, any_curve):
        assert any_curve.df(0.0) == 1.0

    def test_scalar_and_array_queries_agree(self, any_curve):
        arr = np.asarray(any_curve.df(np.array([1.7, 3.3])))
        assert arr[0] == pytest.approx(any_curve.df(1.7), abs=1e-16)
        assert arr[1] == pytest.approx(any_curve.df(3.3), abs=1e-16)


class TestLogLinearForwards:
    def test_piecewise_constant_forward_identity(self):
        """Log-linear DF <=> instantaneous forwards constant on each segment."""
        c = DiscountCurve.from_zero_rates(TIMES, ZEROS, "loglinear_df")
        # inside the [2, 5] segment, all sub-forwards equal the segment forward
        f_seg = c.forward_rate(2.0, 5.0)
        for a, b in [(2.0, 3.0), (2.5, 3.5), (4.0, 5.0), (2.2, 2.3)]:
            assert c.forward_rate(a, b) == pytest.approx(f_seg, abs=1e-14)

    def test_forward_differs_across_segments(self):
        c = DiscountCurve.from_zero_rates(TIMES, ZEROS, "loglinear_df")
        assert c.forward_rate(1.0, 2.0) != pytest.approx(
            c.forward_rate(2.0, 5.0), abs=1e-6
        )

    def test_forwards_positive_for_reasonable_curve(self, any_curve):
        grid = np.linspace(0.5, 29.5, 100)
        fwds = [any_curve.forward_rate(t, t + 0.5) for t in grid]
        assert min(fwds) > 0.0


class TestRateIdentities:
    def test_forward_reconstructs_df(self, any_curve):
        """P(t2) = P(t1) * exp(-f(t1,t2) (t2-t1)) for any t1 < t2."""
        for t1, t2 in [(0.5, 1.0), (1.3, 4.7), (2.0, 10.0)]:
            f = any_curve.forward_rate(t1, t2)
            lhs = any_curve.df(t2)
            rhs = any_curve.df(t1) * np.exp(-f * (t2 - t1))
            assert lhs == pytest.approx(rhs, rel=1e-13)

    def test_one_period_par_rate_is_simple_rate(self, any_curve):
        """par(1y, annual) == (1/P(1) - 1) since the annuity is P(1)."""
        p1 = any_curve.df(1.0)
        assert any_curve.par_rate(1.0, 1) == pytest.approx(1.0 / p1 - 1.0, abs=1e-14)

    def test_par_rate_between_zero_extremes(self, any_curve):
        """Par rate is a weighted average of forwards: within [min, max] zero."""
        par = any_curve.par_rate(10.0, 1)
        zmin, zmax = min(ZEROS), max(ZEROS)
        # simple-compounding par vs continuous zeros: allow small slack
        assert zmin * 0.9 < par < zmax * 1.15

    def test_simple_forward_vs_continuous(self, any_curve):
        """Simple forward > continuous forward (convexity of exp) for f > 0."""
        f_cont = any_curve.forward_rate(2.0, 3.0)
        f_simple = any_curve.simple_forward_rate(2.0, 3.0)
        assert f_simple > f_cont
        assert f_simple == pytest.approx(np.expm1(f_cont), rel=1e-12)


class TestBumps:
    def test_parallel_bump_shifts_zeros(self, any_curve):
        b = any_curve.bumped_parallel(1e-4)
        np.testing.assert_allclose(
            b.zero_rates, any_curve.zero_rates + 1e-4, atol=1e-15
        )

    def test_pillar_bump_selective(self, any_curve):
        shifts = np.zeros(len(TIMES))
        shifts[2] = 5e-4
        b = any_curve.bumped_pillars(shifts)
        assert b.zero_rates[2] == pytest.approx(ZEROS[2] + 5e-4, abs=1e-15)
        assert b.zero_rates[0] == pytest.approx(ZEROS[0], abs=1e-15)

    def test_pillar_bump_wrong_length_raises(self, any_curve):
        with pytest.raises(ValueError, match="shifts"):
            any_curve.bumped_pillars([1e-4, 2e-4])


class TestExtrapolation:
    def test_long_end_warns(self, any_curve):
        with pytest.warns(ExtrapolationWarning):
            any_curve.df(35.0)

    def test_long_end_flat_zero(self, any_curve):
        with pytest.warns(ExtrapolationWarning):
            z = any_curve.zero_rate(40.0)
        assert z == pytest.approx(ZEROS[-1], abs=1e-14)

    def test_short_end_silent_and_flat(self, any_curve):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", ExtrapolationWarning)
            z = any_curve.zero_rate(0.1)
        assert z == pytest.approx(ZEROS[0], abs=1e-12)


class TestValidation:
    def test_single_pillar_curve_works(self):
        c = DiscountCurve([5.0], [0.8])
        assert c.df(5.0) == pytest.approx(0.8, abs=1e-15)
        assert c.zero_rate(2.0) == pytest.approx(-np.log(0.8) / 5.0, abs=1e-14)

    def test_non_increasing_times_raise(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            DiscountCurve([1.0, 1.0, 2.0], [0.99, 0.98, 0.97])

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            DiscountCurve([-1.0, 2.0], [0.99, 0.98])

    def test_negative_df_raises(self):
        with pytest.raises(ValueError, match="discount factors"):
            DiscountCurve([1.0, 2.0], [0.99, -0.5])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            DiscountCurve([1.0, 2.0], [0.99])

    def test_unknown_interpolation_raises(self):
        with pytest.raises(ValueError, match="interpolation"):
            DiscountCurve([1.0], [0.99], "cubic_spline")

    def test_forward_requires_t2_gt_t1(self, any_curve):
        with pytest.raises(ValueError, match="t2 > t1"):
            any_curve.forward_rate(2.0, 2.0)

    def test_par_rate_fractional_maturity_raises(self, any_curve):
        with pytest.raises(ValueError, match="whole number"):
            any_curve.par_rate(2.3, 1)

    def test_df_above_one_allowed_negative_rates(self):
        c = DiscountCurve.from_zero_rates([1.0, 5.0], [-0.005, -0.002])
        assert c.df(1.0) > 1.0
