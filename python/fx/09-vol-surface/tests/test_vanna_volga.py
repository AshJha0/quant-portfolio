"""Vanna-volga: exact replication weights, pillar reproduction, continuity."""

import math

import numpy as np
import pytest

from fx_surface import (
    VannaVolgaSmile,
    gk_forward,
    gk_price,
    gk_vanna,
    gk_vega,
    gk_volga,
    smile_digital,
)

S, T, RD, RF = 1.10, 0.5, 0.045, 0.033
KS = np.array([1.055, 1.105, 1.150])  # 25P, ATM, 25C
VOLS = np.array([0.0925, 0.0880, 0.0895])


@pytest.fixture(scope="module")
def vv():
    return VannaVolgaSmile(S, T, RD, RF, KS, VOLS)


def test_pillar_vols_reproduced_exactly(vv):
    for K, sig in zip(KS, VOLS):
        assert vv.vol(K) == pytest.approx(sig, abs=1e-10)


def test_pillar_weights_are_unit_vectors(vv):
    for i, K in enumerate(KS):
        x = vv.weights(K)
        expected = np.zeros(3)
        expected[i] = 1.0
        np.testing.assert_allclose(x, expected, atol=1e-10)


@pytest.mark.parametrize("K", [1.02, 1.08, 1.12, 1.18, 1.22])
def test_weights_solve_replication_system_exactly(vv, K):
    x = vv.weights(K)
    ref = vv.sigma_ref
    target = np.array(
        [
            gk_vega(S, K, T, RD, RF, ref),
            gk_vanna(S, K, T, RD, RF, ref),
            gk_volga(S, K, T, RD, RF, ref),
        ]
    )
    achieved = vv._A @ x
    np.testing.assert_allclose(achieved, target, atol=1e-12)


def test_vv_price_respects_no_arbitrage_bounds(vv):
    F = gk_forward(S, T, RD, RF)
    for K in np.linspace(1.03, 1.20, 25):
        c = vv.price(float(K), +1)
        lower = max(math.exp(-RD * T) * (F - K), 0.0)
        assert lower - 1e-12 <= c <= S * math.exp(-RF * T)


def test_vv_adjustment_zero_at_reference_vol_pillars(vv):
    """With all pillar vols equal to the reference, VV price == flat BS."""
    flat = VannaVolgaSmile(S, T, RD, RF, KS, np.full(3, 0.088))
    for K in (1.02, 1.10, 1.21):
        assert flat.price(K, +1) == pytest.approx(
            gk_price(S, K, T, RD, RF, 0.088, +1), abs=1e-14
        )
        assert flat.vol(K) == pytest.approx(0.088, abs=1e-10)


def test_vv_put_call_parity(vv):
    F = gk_forward(S, T, RD, RF)
    for K in (1.04, 1.10, 1.19):
        c, p = vv.price(K, +1), vv.price(K, -1)
        assert c - p == pytest.approx(math.exp(-RD * T) * (F - K), abs=1e-14)


def test_vv_smile_continuous_and_convex_body(vv):
    Ks = np.linspace(1.02, 1.21, 200)
    vols = vv.vol(Ks)
    assert not np.any(np.isnan(vols))
    # continuity: neighbouring vols differ by < 15bp on this fine grid
    assert np.max(np.abs(np.diff(vols))) < 0.0015
    # smile shape: minimum lies strictly inside the pillar range
    i_min = int(np.argmin(vols))
    assert 0 < i_min < len(Ks) - 1


def test_vv_smile_interpolates_between_pillar_vols(vv):
    """Between adjacent pillars the VV vol stays within a small margin of
    the pillar vol envelope (no oscillation)."""
    lo, hi = float(np.min(VOLS)), float(np.max(VOLS))
    for K in np.linspace(KS[0], KS[2], 60):
        v = float(vv.vol(float(K)))
        assert lo - 0.002 < v < hi + 0.002


def test_vv_digital_includes_skew_correction(vv):
    """Smile-consistent digital differs from the flat-GK digital in the
    direction implied by the local smile slope at K."""
    from fx_surface import gk_digital

    K = 1.08
    flat = gk_digital(S, K, T, RD, RF, float(vv.vol(K)), 1)
    smile_dig = smile_digital(vv, K, S, T, RD, RF, 1)
    h = 1e-4
    slope = (float(vv.vol(K + h)) - float(vv.vol(K - h))) / (2 * h)
    vega_dig = gk_vega(S, K, T, RD, RF, float(vv.vol(K)))  # ~ digital vega scale
    assert (smile_dig - flat) * (-slope) > 0  # sign of -vega*dsigma/dK
    assert 0.0 < smile_dig < math.exp(-RD * T)


def test_vv_invalid_pillars_raise():
    with pytest.raises(ValueError, match="exactly 3"):
        VannaVolgaSmile(S, T, RD, RF, KS[:2], VOLS[:2])
    with pytest.raises(ValueError, match="increasing"):
        VannaVolgaSmile(S, T, RD, RF, KS[::-1], VOLS)
    with pytest.raises(ValueError, match="positive"):
        VannaVolgaSmile(S, T, RD, RF, KS, np.array([0.09, -0.01, 0.09]))
