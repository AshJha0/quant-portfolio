"""Strategy tests: hand-computed carry accrual, vol targeting, pip costs."""

import numpy as np
import pandas as pd
import pytest

from fx_regime import (
    PIP_FRACTION,
    StrategyConfig,
    base_weights,
    carry_accrual,
    regime_weights,
    transaction_cost,
    vol_target_scale,
)

CCYS = ["AUD", "NZD", "CAD", "EUR", "JPY", "CHF", "MXN", "ZAR"]
RATES = pd.Series(
    {
        "AUD": 0.045, "NZD": 0.055, "CAD": 0.035, "EUR": 0.015,
        "JPY": 0.001, "CHF": -0.005, "MXN": 0.09, "ZAR": 0.075,
        "USD": 0.02,
    }
)
CFG = StrategyConfig()


def test_risk_on_weights_long_high_yield_short_low_yield():
    w = base_weights("risk_on", CCYS, RATES, CFG)
    assert np.isclose(w.sum(), 0.0, atol=1e-12)  # dollar-neutral
    for c in ("MXN", "ZAR", "NZD"):  # top-3 yielders long
        assert w[c] == pytest.approx(1.0 / 3.0)
    for c in ("CHF", "JPY", "EUR"):  # bottom-3 short
        assert w[c] == pytest.approx(-1.0 / 3.0)
    assert w["AUD"] == 0.0 and w["CAD"] == 0.0


def test_risk_off_weights_havens_vs_risk():
    w = base_weights("risk_off", CCYS, RATES, CFG)
    assert w["JPY"] == pytest.approx(0.5)
    assert w["CHF"] == pytest.approx(0.5)
    # risk block present in this universe: AUD, NZD, MXN, ZAR
    for c in ("AUD", "NZD", "MXN", "ZAR"):
        assert w[c] == pytest.approx(-0.25)
    assert np.isclose(w.sum(), 0.0, atol=1e-12)
    assert (w[["EUR", "CAD"]] == 0.0).all()


def test_usd_squeeze_weights_short_everything():
    w = base_weights("usd_squeeze", CCYS, RATES, CFG)
    assert (w == -1.0 / len(CCYS)).all()
    assert np.isclose(w.sum(), -1.0, atol=1e-12)  # net long USD


def test_unknown_regime_raises():
    with pytest.raises(ValueError, match="unknown regime"):
        base_weights("sideways", CCYS, RATES, CFG)


def test_carry_accrual_exact_hand_computed():
    w = pd.Series({"AUD": 0.5, "JPY": -0.5})
    # 0.5*(0.045-0.02) + (-0.5)*(0.001-0.02) = 0.0125 + 0.0095 = 0.022 p.a.
    expected = 0.022 / 252.0
    assert carry_accrual(w, RATES, 0.02) == pytest.approx(expected, abs=1e-15)


def test_carry_accrual_zero_for_flat_book():
    w = pd.Series(0.0, index=CCYS)
    assert carry_accrual(w, RATES, 0.02) == 0.0


def test_vol_target_hit_exactly_ex_ante():
    """With a known covariance the scaled book's ex-ante vol IS the target."""
    w = pd.Series({"A": 1.0, "B": -1.0})
    sigma_d = 0.01
    cov = np.eye(2) * sigma_d**2  # daily
    scale = vol_target_scale(w, cov, target_vol=0.10, max_leverage=10.0)
    ws = w * scale
    realised = np.sqrt(float(ws @ cov @ ws) * 252.0)
    assert realised == pytest.approx(0.10, abs=1e-12)


def test_vol_target_leverage_cap():
    w = pd.Series({"A": 1.0})
    cov = np.eye(1) * (1e-8) ** 2  # near-zero vol asset
    scale = vol_target_scale(w, cov, target_vol=0.10, max_leverage=4.0,
                             vol_floor=0.01)
    assert scale == pytest.approx(4.0)


def test_vol_floor_guards_pegged_book():
    """Zero covariance (pegged currency) must not produce inf leverage."""
    w = pd.Series({"PEG": 1.0})
    cov = np.zeros((1, 1))
    scale = vol_target_scale(w, cov, target_vol=0.10, max_leverage=50.0,
                             vol_floor=0.01)
    assert scale == pytest.approx(min(0.10 / 0.01, 50.0))
    assert np.isfinite(scale)


def test_regime_weights_scaled_book():
    cov = np.eye(len(CCYS)) * (0.006) ** 2
    w = regime_weights("risk_off", CCYS, RATES, cov, CFG)
    realised = np.sqrt(float(w @ cov @ w) * 252.0)
    assert realised == pytest.approx(CFG.target_vol, rel=1e-10)


def test_transaction_cost_hand_computed():
    old = pd.Series({"AUD": 0.0, "JPY": 0.0})
    new = pd.Series({"AUD": 0.5, "JPY": -0.5})
    # AUD: 0.5 * 1.0 pip, JPY: 0.5 * 0.8 pip  (half-spreads, PIP=1e-4)
    expected = (0.5 * 1.0 + 0.5 * 0.8) * PIP_FRACTION
    assert transaction_cost(new, old) == pytest.approx(expected, abs=1e-15)


def test_transaction_cost_zero_without_trades():
    w = pd.Series({"AUD": 0.3, "JPY": -0.3})
    assert transaction_cost(w, w.copy()) == 0.0


def test_transaction_cost_unknown_currency_default():
    old = pd.Series({"XXX": 0.0})
    new = pd.Series({"XXX": 1.0})
    assert transaction_cost(new, old) == pytest.approx(2.0 * PIP_FRACTION)


def test_config_validation():
    with pytest.raises(ValueError):
        StrategyConfig(target_vol=-0.1)
    with pytest.raises(ValueError):
        StrategyConfig(max_leverage=0.0)
    with pytest.raises(ValueError):
        StrategyConfig(cov_window=5)
