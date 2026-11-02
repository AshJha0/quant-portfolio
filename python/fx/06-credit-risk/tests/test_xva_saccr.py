"""Round 3: bilateral CVA/DVA, CSA with MPoR, wrong-way risk, SA-CCR (FX)."""

import numpy as np
import pytest
from fx_credit.saccr import FxTrade, fx_addon, maturity_factor, pfe_multiplier, saccr_ead
from fx_credit.xva import (
    bilateral_cva,
    collateralised_exposure,
    cva_wrong_way,
    epe_mpor_closed_form,
    flat_survival,
    hazard_from_spread,
    simulate_forward_values,
)
from scipy.stats import norm

T = np.linspace(0.0, 5.0, 61)


def test_unilateral_cva_constant_exposure_closed_form():
    lam = hazard_from_spread(0.012, 0.4)
    assert lam == pytest.approx(0.02)
    res = bilateral_cva(
        T, np.full(T.size, 1e6), np.zeros(T.size), flat_survival(T, lam), np.ones(T.size), 0.4
    )
    assert res.cva == pytest.approx(0.6 * 1e6 * (1.0 - np.exp(-0.02 * 5.0)), rel=1e-13)
    assert res.dva == 0.0


def test_cva_dva_mirror_symmetry():
    rng = np.random.default_rng(0)
    epe, ene = rng.uniform(0, 1, T.size), rng.uniform(0, 1, T.size)
    qa, qb = flat_survival(T, 0.03), flat_survival(T, 0.01)
    ab = bilateral_cva(T, epe, ene, qa, qb, 0.4, 0.3, 0.02)
    ba = bilateral_cva(T, ene, epe, qb, qa, 0.3, 0.4, 0.02)
    assert ab.cva == pytest.approx(ba.dva, rel=1e-14)
    assert ab.dva == pytest.approx(ba.cva, rel=1e-14)
    assert ab.bcva == pytest.approx(-ba.bcva, rel=1e-14)


def test_uncollateralised_epe_matches_black():
    s0, k, vol, n = 1.10, 1.10, 0.10, 1e6
    t, v = simulate_forward_values(s0, k, n, 1.0, vol, 0.0, 0.0, 12, 200_000, seed=1)
    ee = np.maximum(v, 0.0).mean(axis=0)
    for i in (3, 6, 12):
        tau = t[i]
        d1 = (np.log(s0 / k) + 0.5 * vol**2 * tau) / (vol * np.sqrt(tau))
        bs = n * (s0 * norm.cdf(d1) - k * norm.cdf(d1 - vol * np.sqrt(tau)))
        se = np.maximum(v[:, i], 0).std() / np.sqrt(v.shape[0])
        assert abs(ee[i] - bs) < 4 * se


def test_mpor_epe_matches_closed_form():
    s0, vol, n = 1.25, 0.12, 1e6
    steps_per_year, mpor_steps = 250, 10
    _, v = simulate_forward_values(s0, 1.30, n, 0.5, vol, 0.0, 0.0, 125, 100_000, seed=2)
    e = collateralised_exposure(v, mpor_steps)
    target = epe_mpor_closed_form(s0, n, vol, mpor_steps / steps_per_year)
    for i in (20, 60, 125):
        se = e[:, i].std() / np.sqrt(e.shape[0])
        assert abs(e[:, i].mean() - target) < 4 * se


def test_threshold_increases_exposure_and_mta_zeroes_small_calls():
    _, v = simulate_forward_values(1.1, 1.1, 1e6, 1.0, 0.1, 0.0, 0.0, 50, 2000, seed=3)
    e0 = collateralised_exposure(v, 5).mean()
    e_h = collateralised_exposure(v, 5, threshold=20_000.0).mean()
    e_unc = np.maximum(v, 0).mean()
    assert e0 < e_h < e_unc
    assert np.array_equal(collateralised_exposure(v, 5, mta=1e12), np.maximum(v, 0.0))


def test_wwr_b_zero_equals_independent_cva():
    t, v = simulate_forward_values(1.1, 1.1, 1e6, 2.0, 0.1, 0.01, 0.0, 24, 4000, seed=4)
    q = flat_survival(t, 0.02)
    ind = bilateral_cva(
        t, np.maximum(v, 0).mean(axis=0), np.zeros(t.size), q, np.ones(t.size), 0.4, 0.4, 0.01
    )
    assert cva_wrong_way(t, v, q, b=0.0, recovery=0.4, rate=0.01) == pytest.approx(ind.cva, rel=1e-9)


def test_wwr_direction():
    t, v = simulate_forward_values(1.1, 1.1, 1e6, 2.0, 0.1, 0.0, 0.0, 24, 4000, seed=5)
    q = flat_survival(t, 0.02)
    base = cva_wrong_way(t, v, q, 0.0)
    assert cva_wrong_way(t, v, q, 1e-5) > base > cva_wrong_way(t, v, q, -1e-5)


def test_saccr_unmargined_hand_value():
    trades = [
        FxTrade("EURUSD", 10e6, True, 0.5),
        FxTrade("EURUSD", 4e6, False, 2.0),
        FxTrade("USDJPY", 5e6, True, 0.02),
    ]
    addon = fx_addon(trades)
    hand = 0.04 * (abs(10e6 * np.sqrt(0.5) - 4e6 * 1.0) + 5e6 * np.sqrt(10 / 250))
    assert addon == pytest.approx(hand, rel=1e-14)
    res = saccr_ead(trades, mtm=150_000.0)
    assert res.multiplier == 1.0
    assert res.ead == pytest.approx(1.4 * (150_000.0 + hand), rel=1e-14)


def test_saccr_multiplier_and_margined_rc():
    assert pfe_multiplier(-1e9, 1.0) == pytest.approx(0.05)
    m = pfe_multiplier(-100.0, 1000.0)
    assert m == pytest.approx(0.05 + 0.95 * np.exp(-100.0 / (2 * 0.95 * 1000.0)), rel=1e-14)
    assert maturity_factor(3.0, margined=True, mpor_days=10) == pytest.approx(1.5 * np.sqrt(10 / 250))
    trades = [FxTrade("GBPUSD", 20e6, True, 1.0)]
    res = saccr_ead(trades, mtm=0.0, collateral=0.0, margined=True, threshold=1e5, mta=5e4)
    assert res.rc == pytest.approx(1.5e5)
    assert res.ead <= saccr_ead(trades, 0.0).ead
