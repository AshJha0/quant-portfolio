"""DiscountCurve identities: interpolation, zeros/forwards, bumping."""

import numpy as np
import pytest

from fx_rates.curve import DiscountCurve

TIMES = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
ZEROS = np.array([0.035, 0.037, 0.040, 0.042, 0.044, 0.045])


@pytest.fixture
def curve():
    return DiscountCurve.from_zero_rates(TIMES, ZEROS, name="TEST")


class TestConstruction:
    def test_pillars_reproduced_exactly(self, curve):
        assert np.allclose(curve.df(TIMES), np.exp(-ZEROS * TIMES), atol=1e-15)
        assert np.allclose(curve.zero_rates, ZEROS, atol=1e-15)

    def test_df_at_zero_is_one(self, curve):
        assert curve.df(0.0) == pytest.approx(1.0, abs=1e-15)

    def test_scalar_in_scalar_out(self, curve):
        assert isinstance(curve.df(1.0), float)
        assert isinstance(curve.zero_rate(1.0), float)

    def test_negative_time_raises(self, curve):
        with pytest.raises(ValueError, match="negative time"):
            curve.df(-0.1)

    def test_zero_rate_at_zero_raises(self, curve):
        with pytest.raises(ValueError, match="t > 0"):
            curve.zero_rate(0.0)

    def test_non_increasing_times_raise(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            DiscountCurve([1.0, 1.0], [0.9, 0.8])

    def test_nonpositive_df_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            DiscountCurve([1.0, 2.0], [0.9, -0.1])

    def test_first_pillar_at_zero_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            DiscountCurve([0.0, 1.0], [1.0, 0.9])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            DiscountCurve([], [])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            DiscountCurve([1.0, 2.0], [0.9])

    def test_negative_rates_allowed_df_above_one(self):
        c = DiscountCurve.from_zero_rates([1.0, 5.0], [-0.005, -0.002])
        assert c.df(1.0) > 1.0  # negative rates => DF > 1 is legitimate


class TestInterpolation:
    def test_log_linear_between_pillars(self, curve):
        # log DF linear => DF(t) = DF(t1)^(1-w) * DF(t2)^w
        t1, t2, t = 1.0, 2.0, 1.4
        w = (t - t1) / (t2 - t1)
        expected = curve.df(t1) ** (1 - w) * curve.df(t2) ** w
        assert curve.df(t) == pytest.approx(expected, abs=1e-15)

    def test_flat_forward_within_segment(self, curve):
        # piecewise-constant instantaneous forwards: any sub-interval of a
        # segment has the same forward rate
        f1 = curve.forward_rate(1.1, 1.5)
        f2 = curve.forward_rate(1.2, 1.9)
        assert f1 == pytest.approx(f2, abs=1e-12)

    def test_extrapolation_flat_forward_beyond_last_pillar(self, curve):
        f_last = curve.forward_rate(5.0, 10.0)
        f_extra = curve.forward_rate(10.0, 12.0)
        assert f_extra == pytest.approx(f_last, abs=1e-12)

    def test_df_multiplicative_identity(self, curve):
        # DF(0,t2) = DF(0,t1) * exp(-f(t1,t2)*(t2-t1))
        t1, t2 = 0.7, 3.3
        f = curve.forward_rate(t1, t2)
        assert curve.df(t2) == pytest.approx(
            curve.df(t1) * np.exp(-f * (t2 - t1)), abs=1e-14
        )

    def test_forward_rate_bad_order_raises(self, curve):
        with pytest.raises(ValueError, match="t1 < t2"):
            curve.forward_rate(2.0, 1.0)

    def test_zero_forward_consistency(self, curve):
        # z(t)*t = integral of forwards = -log DF(t)
        t = 3.7
        assert curve.zero_rate(t) * t == pytest.approx(
            -np.log(curve.df(t)), abs=1e-14
        )


class TestBumping:
    def test_parallel_shift_moves_all_zeros(self, curve):
        up = curve.parallel_shift(10.0)
        assert np.allclose(up.zero_rates - curve.zero_rates, 1e-3, atol=1e-15)

    def test_parallel_shift_roundtrip(self, curve):
        back = curve.parallel_shift(25.0).parallel_shift(-25.0)
        assert np.allclose(back.dfs, curve.dfs, atol=1e-14)

    def test_pillar_shift_only_moves_one_zero(self, curve):
        up = curve.pillar_shift(2, 1.0)
        diff = up.zero_rates - curve.zero_rates
        assert diff[2] == pytest.approx(1e-4, abs=1e-15)
        assert np.all(diff[[0, 1, 3, 4, 5]] == 0.0)

    def test_pillar_shift_locality_in_df(self, curve):
        # bumping pillar i leaves DF unchanged outside (t_{i-1}, t_{i+1})
        up = curve.pillar_shift(3, 5.0)  # pillar at t=2.0; neighbours 1.0, 5.0
        for t in [0.25, 0.6, 1.0, 5.0, 7.0, 10.0]:
            assert up.df(t) == pytest.approx(curve.df(t), abs=1e-15)
        assert up.df(2.0) != pytest.approx(curve.df(2.0), abs=1e-9)

    def test_pillar_shift_bad_index_raises(self, curve):
        with pytest.raises(ValueError, match="out of range"):
            curve.pillar_shift(99, 1.0)
