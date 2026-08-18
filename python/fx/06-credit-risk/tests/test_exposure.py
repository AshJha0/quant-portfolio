"""EE/PFE simulation for FX forwards, netting sets, CVA — shape and hand checks."""

import numpy as np
import pytest

from fx_credit.exposure import (
    FXForward,
    cva,
    cva_for_forward,
    exposure_profile,
    forward_mtm,
    hazard_from_pd1y,
    netting_set_profile,
    pd_term_structure,
    simulate_fx_paths,
)

SPOT, VOL, RD, RF = 1.08, 0.12, 0.03, 0.02
FWD = FXForward("EURUSD", 10_000_000, 1.08, 1.0, buy_base=True)


@pytest.fixture(scope="module")
def profile():
    return exposure_profile(FWD, SPOT, VOL, RD, RF, n_steps=24, n_paths=100_000, seed=1)


def test_paths_shape_seeded_positive():
    t = np.linspace(0.1, 1.0, 10)
    p1 = simulate_fx_paths(SPOT, VOL, RD, RF, t, 500, seed=3)
    p2 = simulate_fx_paths(SPOT, VOL, RD, RF, t, 500, seed=3)
    p3 = simulate_fx_paths(SPOT, VOL, RD, RF, t, 500, seed=4)
    assert p1.shape == (500, 10)
    assert np.array_equal(p1, p2) and not np.array_equal(p1, p3)
    assert np.all(p1 > 0)


def test_paths_forward_martingale():
    """E[S_T] = S0 exp((rd-rf) T) within 3 standard errors (exact GBM scheme)."""
    t = np.array([1.0])
    p = simulate_fx_paths(SPOT, VOL, RD, RF, t, 200_000, seed=5)[:, 0]
    target = SPOT * np.exp((RD - RF) * 1.0)
    se = p.std() / np.sqrt(p.size)
    assert abs(p.mean() - target) < 3 * se


def test_paths_invalid_inputs_raise():
    with pytest.raises(ValueError, match="spot"):
        simulate_fx_paths(-1.0, VOL, RD, RF, np.array([1.0]), 10)
    with pytest.raises(ValueError, match="times"):
        simulate_fx_paths(SPOT, VOL, RD, RF, np.array([0.5, 0.2]), 10)


def test_forward_mtm_hand_computed_t0():
    """V_0 = N (F_0 - K) e^{-rd T} with F_0 = S e^{(rd-rf)T}."""
    f0 = SPOT * np.exp((RD - RF) * 1.0)
    expected = 10e6 * (f0 - 1.08) * np.exp(-RD * 1.0)
    assert forward_mtm(FWD, SPOT, 0.0, RD, RF) == pytest.approx(expected, rel=1e-12)


def test_forward_mtm_sell_side_sign():
    sell = FXForward("EURUSD", 10e6, 1.08, 1.0, buy_base=False)
    assert forward_mtm(sell, 1.20, 0.5, RD, RF) == pytest.approx(
        -forward_mtm(FWD, 1.20, 0.5, RD, RF), rel=1e-12
    )


def test_forward_mtm_at_maturity_is_payoff():
    assert forward_mtm(FWD, 1.20, 1.0, RD, RF) == pytest.approx(10e6 * (1.20 - 1.08), rel=1e-12)


def test_forward_mtm_after_maturity_zero():
    assert forward_mtm(FWD, 1.20, 1.5, RD, RF) == 0.0


def test_matured_forward_zero_profile():
    dead = FXForward("EURUSD", 10e6, 1.08, 0.0, True)
    prof = exposure_profile(dead, SPOT, VOL, RD, RF, n_paths=100, seed=0)
    assert prof.times.size == 0 and prof.peak_pfe(0.99) == 0.0


def test_zero_notional_zero_profile():
    z = FXForward("EURUSD", 0.0, 1.08, 1.0, True)
    prof = exposure_profile(z, SPOT, VOL, RD, RF, n_paths=100, seed=0)
    assert np.all(prof.ee == 0.0) and np.all(prof.pfe[0.99] == 0.0)


def test_pfe_monotone_increasing_to_maturity(profile):
    """Single forward: no interim cashflows, so PFE GROWS to maturity (~sqrt t).

    This is the correct shape for an outright forward — the 'mid-life hump'
    belongs to amortising products like swaps, not to a bullet forward.
    """
    assert np.all(np.diff(profile.pfe[0.99]) > 0)
    assert np.all(np.diff(profile.pfe[0.95]) > 0)


def test_pfe_concave_sqrt_t_shape(profile):
    """PFE increments shrink with t (concavity, sqrt-t diffusion) on a quarterly grid."""
    q = profile.pfe[0.99][[5, 11, 17, 23]]  # t = 0.25, 0.5, 0.75, 1.0
    increments = np.diff(np.r_[0.0, q])
    assert np.all(np.diff(increments) < 0)


def test_pfe_correlates_with_sqrt_t(profile):
    corr = np.corrcoef(profile.pfe[0.99], np.sqrt(profile.times))[0, 1]
    assert corr > 0.99


def test_pfe99_above_pfe95(profile):
    assert np.all(profile.pfe[0.99] >= profile.pfe[0.95])
    assert profile.peak_pfe(0.99) > profile.peak_pfe(0.95)


def test_ee_below_pfe95_and_positive(profile):
    assert np.all(profile.ee >= 0.0)
    assert np.all(profile.ee <= profile.pfe[0.95] + 1e-9)
    assert profile.ee[-1] > 0.0


def test_netting_reduces_pfe_for_offsetting_trades():
    sell = FXForward("EURUSD", 10e6, 1.08, 1.0, buy_base=False)
    kw = dict(n_steps=12, n_paths=20_000, seed=2)
    net = netting_set_profile([FWD, sell], SPOT, VOL, RD, RF, netting=True, **kw)
    gross = netting_set_profile([FWD, sell], SPOT, VOL, RD, RF, netting=False, **kw)
    assert net.peak_pfe(0.99) == pytest.approx(0.0, abs=1e-9)  # exact offset
    assert gross.peak_pfe(0.99) > 0.0


def test_netting_equals_gross_for_same_direction_trades():
    """Perfectly correlated same-direction trades: netting gives no benefit (upper bound)."""
    kw = dict(n_steps=12, n_paths=20_000, seed=2)
    net = netting_set_profile([FWD, FWD], SPOT, VOL, RD, RF, netting=True, **kw)
    gross = netting_set_profile([FWD, FWD], SPOT, VOL, RD, RF, netting=False, **kw)
    assert np.allclose(net.pfe[0.99], gross.pfe[0.99], rtol=1e-12)
    assert np.allclose(net.ee, gross.ee, rtol=1e-12)


def test_netting_bounds_mixed_book():
    """max(sum V,0) <= sum max(V,0) path-by-path => netted PFE <= gross PFE always."""
    other = FXForward("EURUSD", 6e6, 1.12, 0.8, buy_base=False)
    kw = dict(n_steps=12, n_paths=20_000, seed=3)
    net = netting_set_profile([FWD, other], SPOT, VOL, RD, RF, netting=True, **kw)
    gross = netting_set_profile([FWD, other], SPOT, VOL, RD, RF, netting=False, **kw)
    assert np.all(net.pfe[0.99] <= gross.pfe[0.99] + 1e-9)
    assert np.all(net.ee <= gross.ee + 1e-9)


def test_netting_set_validation():
    with pytest.raises(ValueError, match="empty"):
        netting_set_profile([], SPOT, VOL, RD, RF)
    with pytest.raises(ValueError, match="one pair"):
        netting_set_profile(
            [FWD, FXForward("GBPUSD", 1e6, 1.27, 1.0, True)], SPOT, VOL, RD, RF
        )


def test_hazard_round_trip():
    pd1 = 0.02
    h = hazard_from_pd1y(pd1)
    assert 1.0 - np.exp(-h * 1.0) == pytest.approx(pd1, abs=1e-15)
    assert pd_term_structure(pd1, np.array([1.0]))[0] == pytest.approx(pd1, abs=1e-15)


def test_hazard_invalid_raises():
    with pytest.raises(ValueError, match="pd_1y"):
        hazard_from_pd1y(1.0)


def test_pd_term_structure_monotone():
    q = pd_term_structure(0.05, np.linspace(0.1, 5.0, 50))
    assert np.all(np.diff(q) > 0) and np.all((q > 0) & (q < 1))


def test_cva_hand_computed_two_period():
    """LGD 0.6, EE = [10, 8], cumPD = [0.01, 0.03], r = 0.

    CVA = 0.6 * (10*0.01 + 8*0.02) = 0.156 exactly.
    """
    val = cva(np.array([0.5, 1.0]), np.array([10.0, 8.0]), np.array([0.01, 0.03]), 0.6)
    assert val == pytest.approx(0.156, abs=1e-15)


def test_cva_hand_computed_with_discounting():
    r = 0.05
    expected = 0.6 * (10 * 0.01 * np.exp(-r * 0.5) + 8 * 0.02 * np.exp(-r * 1.0))
    val = cva(np.array([0.5, 1.0]), np.array([10.0, 8.0]), np.array([0.01, 0.03]), 0.6, r_d=r)
    assert val == pytest.approx(expected, abs=1e-15)


def test_cva_zero_pd_counterparty_is_zero():
    val, prof = cva_for_forward(FWD, SPOT, VOL, RD, RF, pd_1y=0.0, lgd=0.6,
                                n_steps=6, n_paths=2000, seed=1)
    assert val == 0.0 and prof.times.size > 0


def test_cva_increases_with_pd():
    kw = dict(n_steps=6, n_paths=5000, seed=1)
    lo, _ = cva_for_forward(FWD, SPOT, VOL, RD, RF, pd_1y=0.01, lgd=0.6, **kw)
    hi, _ = cva_for_forward(FWD, SPOT, VOL, RD, RF, pd_1y=0.05, lgd=0.6, **kw)
    assert 0.0 < lo < hi


def test_cva_matured_forward_zero():
    dead = FXForward("EURUSD", 10e6, 1.08, 0.0, True)
    val, _ = cva_for_forward(dead, SPOT, VOL, RD, RF, pd_1y=0.02, lgd=0.6,
                             n_steps=6, n_paths=1000, seed=1)
    assert val == 0.0


def test_cva_validation_errors():
    t = np.array([0.5, 1.0])
    e = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="lgd"):
        cva(t, e, np.array([0.01, 0.02]), lgd=1.5)
    with pytest.raises(ValueError, match="nondecreasing"):
        cva(t, e, np.array([0.03, 0.01]), lgd=0.5)
    with pytest.raises(ValueError, match="shape"):
        cva(t, e[:1], np.array([0.01, 0.02]), lgd=0.5)
