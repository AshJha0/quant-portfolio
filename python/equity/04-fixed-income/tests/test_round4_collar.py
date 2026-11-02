"""Round 4: Hagan-West (2006) Section 5 positivity collar on monotone_convex."""

import numpy as np
import pytest
from fi_rates.curve import DiscountCurve

# Discrete forwards 0.1 %, 5 %, 4.5 %, 0.2 %, 4 %: all positive, but the plain
# scheme's node forwards f_0 = fd_1 - (f_1 - fd_1)/2 and the dip at 5y go
# below zero.
T = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
FD = np.array([0.001, 0.05, 0.045, 0.002, 0.04])
Y = np.cumsum(FD * np.diff(np.concatenate(([0.0], T))))
GRID = np.linspace(1e-6, 7.0, 7001)


def _curve(collar):
    return DiscountCurve(T, np.exp(-Y), "monotone_convex", positive_forwards=collar)


def test_plain_scheme_goes_negative_on_this_curve():
    assert np.min(_curve(False).instantaneous_forward(GRID)) < 0.0


def test_collar_keeps_forward_non_negative():
    f = np.asarray(_curve(True).instantaneous_forward(GRID))
    assert f.min() >= 0.0


def test_collar_still_reprices_every_pillar():
    c = _curve(True)
    np.testing.assert_allclose(c.df(T), np.exp(-Y), rtol=0, atol=1e-15)


def test_collar_forward_continuous_and_integrates_to_minus_log_df():
    c = _curve(True)
    for t in T[:-1]:
        assert c.instantaneous_forward(t - 1e-10) == pytest.approx(
            c.instantaneous_forward(t + 1e-10), abs=1e-8)
    h = 1e-6
    t = np.linspace(0.01, 6.9, 500)
    fd = -(np.log(c.df(t + h)) - np.log(c.df(t - h))) / (2 * h)
    np.testing.assert_allclose(c.instantaneous_forward(t), fd, atol=1e-7)


def test_collar_is_a_no_op_on_a_benign_curve():
    t = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]
    z = [0.040, 0.041, 0.043, 0.046, 0.048, 0.050, 0.051, 0.0515, 0.050]
    a = DiscountCurve.from_zero_rates(t, z, "monotone_convex")
    b = DiscountCurve.from_zero_rates(t, z, "monotone_convex", positive_forwards=True)
    g = np.linspace(0.01, 20.0, 999)
    np.testing.assert_array_equal(a.instantaneous_forward(g), b.instantaneous_forward(g))


def test_collar_survives_bumps():
    c = _curve(True)
    assert c.bumped_parallel(1e-4).positive_forwards
    assert c.bumped_pillars(np.full(T.size, 1e-4)).positive_forwards


def test_collar_rejects_other_schemes_and_negative_forwards():
    with pytest.raises(ValueError, match="monotone_convex only"):
        DiscountCurve(T, np.exp(-Y), "loglinear_df", positive_forwards=True)
    yneg = np.cumsum(np.array([0.01, -0.005, 0.02, 0.02, 0.02]) * np.diff(np.concatenate(([0.0], T))))
    with pytest.raises(ValueError, match="discrete forward"):
        DiscountCurve(T, np.exp(-yneg), "monotone_convex", positive_forwards=True)


def test_collar_single_pillar():
    c = DiscountCurve([5.0], [0.8], "monotone_convex", positive_forwards=True)
    assert c.df(5.0) == pytest.approx(0.8, abs=1e-15)
    assert c.instantaneous_forward(2.0) == pytest.approx(-np.log(0.8) / 5.0)
