"""Edge-case and property tests added in the review pass.

Focus: sovereign/counterparty credit edge cases, pegged-pair (zero-vol)
exposures, distressed-sovereign PD limits, NaN/Inf rejection at every public
entry point, tiny/constant samples, and settlement-book degenerate cases.
"""

import numpy as np
import pandas as pd
import pytest

from fx_credit import (
    FXForward,
    FXTrade,
    assign_rating,
    auc,
    bootstrap_auc_ci,
    cva,
    cva_for_forward,
    expected_loss,
    exposure_profile,
    gross_settlement_exposure,
    hazard_from_pd1y,
    hosmer_lemeshow,
    ks_statistic,
    net_settlement_exposure,
    pd_term_structure,
    psi,
    settlement_exposure,
    simulate_fx_paths,
    vasicek_capital,
    vasicek_conditional_pd,
    woe_table,
)

RATES = {"USD": 1.0, "JPY": 1 / 148.0, "EUR": 1.08, "GBP": 1.27}


# ---------------------------------------------------------------------------
# Pegged pair: zero volatility
# ---------------------------------------------------------------------------

def test_pegged_pair_zero_vol_paths_deterministic():
    """A pegged pair (vol=0) produces deterministic forward-drift paths."""
    t = np.array([0.25, 0.5, 1.0])
    p = simulate_fx_paths(spot=7.85, vol=0.0, r_d=0.05, r_f=0.04, times=t, n_paths=100, seed=3)
    expected = 7.85 * np.exp((0.05 - 0.04) * t)
    assert np.allclose(p, expected[None, :], atol=1e-12)
    assert np.allclose(p.std(axis=0), 0.0)


def test_pegged_pair_zero_vol_exposure_pfe_equals_ee():
    """With vol=0 exposure is deterministic: PFE at any quantile equals EE."""
    fwd = FXForward("USDHKD", 10e6, strike=7.80, maturity=1.0)
    prof = exposure_profile(fwd, spot=7.85, vol=0.0, r_d=0.05, r_f=0.04,
                            n_steps=8, n_paths=500, seed=1)
    assert np.allclose(prof.ee, prof.pfe[0.95], atol=1e-9)
    assert np.allclose(prof.ee, prof.pfe[0.99], atol=1e-9)
    # deterministic MTM is positive here (spot above strike, positive carry)
    assert np.all(prof.ee > 0)


# ---------------------------------------------------------------------------
# Distressed sovereign: PD near 1, CVA bounds
# ---------------------------------------------------------------------------

def test_hazard_explodes_as_pd_to_one():
    assert hazard_from_pd1y(0.999999) > 10.0
    with pytest.raises(ValueError):
        hazard_from_pd1y(1.0)


def test_cva_bounded_by_lgd_times_peak_ee():
    """CVA <= LGD * max EE * total default mass (undiscounted upper bound)."""
    fwd = FXForward("EURUSD", 5e6, strike=1.10, maturity=2.0)
    value, prof = cva_for_forward(fwd, spot=1.10, vol=0.12, r_d=0.03, r_f=0.02,
                                  pd_1y=0.60, lgd=0.75, n_steps=12, n_paths=20_000, seed=5)
    assert 0.0 < value <= 0.75 * prof.ee.max() * 1.0 + 1e-9


def test_cva_monotone_in_lgd():
    fwd = FXForward("EURUSD", 5e6, strike=1.10, maturity=1.0)
    vals = []
    for lgd in (0.25, 0.50, 0.75):
        v, _ = cva_for_forward(fwd, 1.10, 0.12, 0.03, 0.02, pd_1y=0.05, lgd=lgd,
                               n_steps=8, n_paths=10_000, seed=9)
        vals.append(v)
    assert vals[0] < vals[1] < vals[2]
    assert vals[1] == pytest.approx(2 * vals[0], rel=1e-12)


def test_pd_term_structure_saturates_for_distressed_sovereign():
    t = np.array([1.0, 5.0, 30.0])
    q = pd_term_structure(0.60, t)
    assert q[0] == pytest.approx(0.60, abs=1e-12)
    assert q[-1] > 0.999999
    assert np.all(np.diff(q) > 0)


def test_capital_vanishes_at_both_pd_extremes_high_rho():
    """Vasicek K -> 0 as PD -> 0 and PD -> 1 even at sovereign rho."""
    tiny = vasicek_capital(1e-9, 0.45, 0.30)
    sure = vasicek_capital(1.0 - 1e-12, 0.45, 0.30)
    mid = vasicek_capital(0.02, 0.45, 0.30)
    assert tiny < 1e-6 and sure < 1e-6
    assert mid > 10 * max(tiny, sure)


# ---------------------------------------------------------------------------
# NaN / Inf rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_simulate_fx_paths_rejects_nonfinite_spot_vol_rates(bad):
    t = np.array([0.5, 1.0])
    with pytest.raises(ValueError):
        simulate_fx_paths(bad, 0.1, 0.02, 0.01, t, 10)
    with pytest.raises(ValueError):
        simulate_fx_paths(1.1, bad, 0.02, 0.01, t, 10)
    with pytest.raises(ValueError):
        simulate_fx_paths(1.1, 0.1, bad, 0.01, t, 10)
    with pytest.raises(ValueError):
        simulate_fx_paths(1.1, 0.1, 0.02, bad, t, 10)


def test_simulate_fx_paths_rejects_nan_times():
    with pytest.raises(ValueError):
        simulate_fx_paths(1.1, 0.1, 0.02, 0.01, np.array([0.5, np.nan]), 10)


def test_forward_dataclass_rejects_nonfinite():
    with pytest.raises(ValueError):
        FXForward("EURUSD", np.nan, 1.1, 1.0)
    with pytest.raises(ValueError):
        FXForward("EURUSD", 1e6, np.inf, 1.0)
    with pytest.raises(ValueError):
        FXForward("EURUSD", 1e6, 1.1, np.nan)


def test_fx_trade_rejects_nonfinite():
    with pytest.raises(ValueError):
        FXTrade("T", "C", "EURUSD", np.nan, 1.08, True)
    with pytest.raises(ValueError):
        FXTrade("T", "C", "EURUSD", 1e6, np.inf, True)


def test_expected_loss_rejects_nan_inf():
    with pytest.raises(ValueError):
        expected_loss(np.nan, 0.5, 100.0)
    with pytest.raises(ValueError):
        expected_loss(0.01, np.nan, 100.0)
    with pytest.raises(ValueError):
        expected_loss(0.01, 0.5, np.inf)


def test_vasicek_rejects_nan_pd_and_factor():
    with pytest.raises(ValueError):
        vasicek_conditional_pd(np.nan, 0.3, 2.0)
    with pytest.raises(ValueError):
        vasicek_conditional_pd(0.02, 0.3, np.inf)
    with pytest.raises(ValueError):
        vasicek_capital(0.02, np.nan, 0.3)


def test_cva_rejects_nonfinite_and_negative_ee():
    t = np.array([1.0])
    with pytest.raises(ValueError):
        cva(t, np.array([np.nan]), np.array([0.01]), 0.5)
    with pytest.raises(ValueError):
        cva(t, np.array([np.inf]), np.array([0.01]), 0.5)
    with pytest.raises(ValueError):
        cva(t, np.array([-1.0]), np.array([0.01]), 0.5)
    with pytest.raises(ValueError):
        cva(t, np.array([1.0]), np.array([np.nan]), 0.5)
    with pytest.raises(ValueError):
        cva(t, np.array([1.0]), np.array([1.5]), 0.5)


def test_auc_ks_reject_nan_scores():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    s = np.array([0.1, np.nan, 0.2, 0.9])
    with pytest.raises(ValueError):
        auc(y, s)
    with pytest.raises(ValueError):
        ks_statistic(y, s)


def test_psi_rejects_nonfinite():
    with pytest.raises(ValueError):
        psi(np.array([1.0, 2.0, np.inf]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        psi(np.array([1.0, 2.0, 3.0]), np.array([np.nan, 2.0, 3.0]))


def test_hosmer_lemeshow_rejects_bad_probabilities():
    y = np.array([0.0, 1.0] * 20)
    with pytest.raises(ValueError):
        hosmer_lemeshow(y, np.full(40, np.nan))
    with pytest.raises(ValueError):
        hosmer_lemeshow(y, np.full(40, 1.5))


def test_settlement_rejects_nonpositive_usd_rate():
    # buy USD sell JPY: pay JPY early, receive USD late -> exposed in USD
    tr = FXTrade("T", "C", "USDJPY", 1e6, 148.0, we_buy_base=True)
    with pytest.raises(ValueError):
        settlement_exposure(tr, {"USD": np.nan, "JPY": 1 / 148.0})
    with pytest.raises(ValueError):
        settlement_exposure(tr, {"USD": -1.0, "JPY": 1 / 148.0})


def test_assign_rating_rejects_nan():
    with pytest.raises(ValueError):
        assign_rating(float("nan"))


# ---------------------------------------------------------------------------
# Tiny / constant samples
# ---------------------------------------------------------------------------

def test_auc_minimal_two_row_sample():
    assert auc(np.array([0.0, 1.0]), np.array([0.1, 0.9])) == 1.0
    assert auc(np.array([1.0, 0.0]), np.array([0.1, 0.9])) == 0.0


def test_auc_constant_score_is_half():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert auc(y, np.full(4, 0.5)) == pytest.approx(0.5)


def test_ks_constant_score_is_zero():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert ks_statistic(y, np.full(4, 0.5)) == pytest.approx(0.0)


def test_bootstrap_ci_tiny_sample_degenerate_but_finite():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    point, lo, hi = bootstrap_auc_ci(y, s, n_boot=200, seed=2)
    assert 0.0 <= lo <= point <= hi <= 1.0


def test_woe_single_row_per_class():
    """Tiny 2-row sample: WOE table still builds with smoothing."""
    t = woe_table(np.array([1.0, 2.0]), np.array([0.0, 1.0]), n_bins=2, smoothing=0.5)
    assert t.bad_total == 1 and t.good_total == 1
    assert np.isfinite(t.iv)


def test_netting_all_cls_book_zero_exposure():
    trades = [
        FXTrade("T1", "C", "USDJPY", 1e6, 148.0, True, cls_settled=True),
        FXTrade("T2", "C", "EURUSD", 2e6, 1.08, False, cls_settled=True),
    ]
    assert gross_settlement_exposure(trades, RATES) == 0.0
    assert net_settlement_exposure(trades, RATES) == 0.0


def test_empty_trade_book_zero_exposure():
    assert gross_settlement_exposure([], RATES) == 0.0
    assert net_settlement_exposure([], RATES) == 0.0
