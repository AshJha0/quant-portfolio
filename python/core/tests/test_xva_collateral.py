"""Collateralised exposure (VM with MPOR, threshold, MTA, IM) against closed
forms for a Brownian position and against exact limiting cases.

The Monte Carlo checks use a fixed seed and a 4-standard-error bound, the
same acceptance rule as test_xva.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qp_core.errors import InputError
from qp_core.xva import collateral as K
from qp_core.xva import profile

SIGMA = 1e6  # value vol per sqrt(year)
MPOR = 10 / 250


def _brownian(times: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = np.diff(times, prepend=times[0])
    inc = rng.standard_normal((n, times.size)) * SIGMA * np.sqrt(dt)
    return np.cumsum(inc, axis=1)


def _within(mean: float, se: float, want: float) -> None:
    assert abs(mean - want) <= 4 * se, (mean, se, want)


def test_worked_example_and_target() -> None:
    v = np.array([[0.0, 10.0, 4.0, -8.0]])
    csa = K.Csa(threshold_cpty=3.0, threshold_bank=5.0)
    assert K.target_vm(v, csa).tolist() == [[0.0, 7.0, 1.0, -3.0]]
    e = K.collateralised_exposure(np.arange(4.0), v, csa)
    assert e.exposure.tolist() == [[0.0, 3.0, 3.0, 0.0]]
    assert e.negative_exposure.tolist() == [[0.0, 0.0, 0.0, -5.0]]


def test_infinite_threshold_is_uncollateralised() -> None:
    t = np.linspace(0.0, 1.0, 51)
    v = _brownian(t, 2000, 1)
    csa = K.Csa(threshold_cpty=math.inf, threshold_bank=math.inf, mpor=MPOR, mta=1e5)
    ce = K.collateralised_exposure(t, v, csa)
    assert np.array_equal(ce.exposure, np.maximum(v, 0.0))
    assert np.array_equal(ce.negative_exposure, np.minimum(v, 0.0))


def test_perfect_csa_no_lag_is_zero() -> None:
    t = np.linspace(0.0, 1.0, 51)
    v = _brownian(t, 500, 2)
    ce = K.collateralised_exposure(t, v, K.Csa())
    assert np.all(ce.exposure == 0.0) and np.all(ce.negative_exposure == 0.0)


def test_mta_bounds_exposure_pathwise() -> None:
    t = np.linspace(0.0, 1.0, 251)
    v = _brownian(t, 500, 3)
    mta = 50_000.0
    ce = K.collateralised_exposure(t, v, K.Csa(mta=mta))
    assert ce.exposure.max() < mta
    assert ce.negative_exposure.min() > -mta
    assert ce.exposure.max() > 0.5 * mta  # the MTA does bind
    # MTA = 0 is the same as no MTA.
    a = K.collateral_balance(v, K.Csa(mta=0.0))
    assert np.array_equal(a, K.target_vm(v, K.Csa()))


def test_mta_infinite_keeps_initial_balance() -> None:
    t = np.linspace(0.0, 1.0, 11)
    v = _brownian(t, 50, 4)
    b = K.collateral_balance(v, K.Csa(mta=math.inf, initial=7.0))
    assert np.all(b == 7.0)


def test_mpor_closed_form() -> None:
    # Perfect CSA with a 10-business-day MPOR: EE(t) = sigma sqrt(MPOR) phi(0)
    # for every t >= MPOR, the standard remargining result.
    base = np.linspace(0.0, 1.0, 11)
    t = K.lagged_grid(base, MPOR)
    v = _brownian(t, 100_000, 5)
    ce = K.collateralised_exposure(t, v, K.Csa(mpor=MPOR))
    want = K.brownian_ee_mpor(SIGMA, MPOR)
    assert want == pytest.approx(SIGMA * math.sqrt(MPOR / (2 * math.pi)), rel=1e-15)
    for k, tk in enumerate(t):
        col = ce.exposure[:, k]
        se = col.std(ddof=1) / math.sqrt(col.size)
        if not np.any(np.isclose(base, tk)):
            continue  # an added lag date: its own lag falls between grid dates
        if tk >= MPOR - 1e-12:
            _within(col.mean(), se, want)
            # By symmetry ENE = -EE.
            neg = ce.negative_exposure[:, k]
            _within(-neg.mean(), neg.std(ddof=1) / math.sqrt(col.size), want)
        else:
            # Before the first call can be lagged, C = initial = 0: uncollateralised.
            assert np.array_equal(col, np.maximum(v[:, k], 0.0))


def test_mpor_with_initial_margin_closed_form() -> None:
    t = K.lagged_grid(np.array([0.0, 0.5, 1.0]), MPOR)
    v = _brownian(t, 200_000, 6)
    s = SIGMA * math.sqrt(MPOR)
    for k_mult in (0.5, 1.0, 2.326):  # 2.326 s ~ the 99% one-sided IM
        im = k_mult * s
        ce = K.collateralised_exposure(t, v, K.Csa(mpor=MPOR, im_received=im))
        want = K.brownian_ee_im(SIGMA, MPOR, im)
        col = ce.exposure[:, -1]
        _within(col.mean(), col.std(ddof=1) / math.sqrt(col.size), want)
    # IM = 0 reduces to the plain MPOR formula exactly.
    assert K.brownian_ee_im(SIGMA, MPOR, 0.0) == pytest.approx(K.brownian_ee_mpor(SIGMA, MPOR), rel=1e-15)
    # A 99% IM cuts EE by roughly two orders of magnitude.
    assert K.brownian_ee_im(SIGMA, MPOR, 2.326 * s) / K.brownian_ee_mpor(SIGMA, MPOR) < 0.01


def test_threshold_no_lag_closed_form() -> None:
    t = np.array([0.0, 1.0])
    v = _brownian(t, 200_000, 7)
    for h in (2e5, 1e6, 3e6):
        ce = K.collateralised_exposure(t, v, K.Csa(threshold_cpty=h, threshold_bank=h))
        col = ce.exposure[:, 1]
        _within(col.mean(), col.std(ddof=1) / math.sqrt(col.size), K.brownian_ee_threshold(SIGMA, h))
    # Limits of the closed form: H -> 0 gives 0, H -> inf gives v phi(0).
    assert K.brownian_ee_threshold(SIGMA, 1e-9) == pytest.approx(0.0, abs=1e-6)
    assert K.brownian_ee_threshold(SIGMA, 1e12) == pytest.approx(SIGMA / math.sqrt(2 * math.pi), rel=1e-12)


def test_threshold_closed_form_by_quadrature() -> None:
    # E[min(V+, H)] = int_0^H P(V > x) dx, done by the trapezoid rule.
    v_sd, h = 1.0, 0.7
    x = np.linspace(0.0, h, 200_001)
    tail = 0.5 * np.array([math.erfc(xi / math.sqrt(2)) for xi in x])
    quad = float(np.sum((tail[1:] + tail[:-1]) * 0.5 * np.diff(x)))
    assert K.brownian_ee_threshold(v_sd, h) == pytest.approx(quad, rel=1e-10)


def test_profile_feeds_cva_code() -> None:
    t = K.lagged_grid(np.linspace(0.0, 1.0, 5), MPOR)
    v = _brownian(t, 20_000, 8)
    csa = K.Csa(mpor=MPOR, im_received=1e4, im_posted=2e4)
    ce = K.collateralised_exposure(t, v, csa)
    prof = K.collateralised_profile(t, v, csa)
    assert np.allclose(prof.ee, ce.exposure.mean(axis=0), rtol=0, atol=1e-9)
    assert np.allclose(prof.ene, ce.negative_exposure.mean(axis=0), rtol=0, atol=1e-9)
    # Collateral can only reduce exposure here (zero thresholds, no MTA).
    raw = profile(t, v)
    assert np.all(prof.ee <= raw.ee + 1e-9)


def test_lagged_grid_and_bad_inputs() -> None:
    assert K.lagged_grid([0.0, 0.5], 1.0).tolist() == [0.0, 0.5]
    with pytest.raises(InputError):
        K.Csa(mta=-1.0)
    with pytest.raises(InputError):
        K.collateralised_exposure(np.array([0.0, 0.0]), np.zeros((1, 2)), K.Csa())
    with pytest.raises(InputError):
        K.collateralised_exposure(np.array([0.0, 1.0]), np.zeros((1, 2)), K.Csa(im_received=-1.0))
