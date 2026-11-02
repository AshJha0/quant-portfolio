"""HL degrees of freedom by simulation, per-grade back-test, BCBS settlement window."""

from __future__ import annotations

import math

import fx_credit as fc
import numpy as np
import pytest
from fx_credit.settlement import FXTrade


def test_hl_out_of_sample_size_is_nominal():
    # Known PDs: the 5% test should reject 5% of 1000 trials (MC SE 0.0069).
    # The old n_groups - 2 reference rejected about 11%.
    rng = np.random.default_rng(31)
    rej = 0
    for _ in range(1000):
        p = rng.uniform(0.01, 0.3, 2000)
        y = (rng.random(2000) < p).astype(float)
        rej += fc.hosmer_lemeshow(y, p, out_of_sample=True).p_value < 0.05
    assert 0.03 <= rej / 1000 <= 0.07


def test_hl_dof_follows_sample_flag():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.2, 500)
    y = (rng.random(500) < p).astype(float)
    assert fc.hosmer_lemeshow(y, p, out_of_sample=True).dof == 10
    assert fc.hosmer_lemeshow(y, p, out_of_sample=False).dof == 8
    with pytest.warns(DeprecationWarning):
        assert fc.hosmer_lemeshow(y, p).dof == 8


def test_by_grade_hand_values():
    y = np.r_[np.ones(6), np.zeros(94)]
    t = fc.pd_backtest_by_grade(y, np.full(100, 0.02), ["BB"] * 100)
    p_bin = sum(math.comb(100, k) * 0.02**k * 0.98 ** (100 - k) for k in range(6, 101))
    assert t.loc[0, "binomial_p"] == pytest.approx(p_bin, rel=1e-10)
    assert t.loc[0, "light"] == "amber"


def test_bcbs_window_is_never_shorter_than_operating_hours():
    for s in ("JPY", "EUR", "GBP", "USD"):
        for b in ("JPY", "EUR", "GBP", "USD"):
            assert fc.bcbs_window_hours(s, b) >= fc.at_risk_window_hours(s, b)
    # USD -> JPY: zero under operating hours, positive under BCBS
    assert fc.at_risk_window_hours("USD", "JPY") == 0.0
    # open(USD) 13.5 - 24 lead = -10.5; close(JPY) 8 + 12 lag = 20 -> 30.5 h
    assert fc.bcbs_window_hours("USD", "JPY") == pytest.approx(30.5)
    assert fc.bcbs_window_hours("USD", "JPY", extra_days=2) == pytest.approx(78.5)
    assert fc.bcbs_window_hours("USD", "JPY", 0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        fc.bcbs_window_hours("USD", "JPY", cancellation_lead_hours=-1.0)


def test_bcbs_window_creates_exposure_on_receive_before_pay_trade():
    t = FXTrade("T1", "Bank", "USDJPY", 10e6, 150.0, we_buy_base=False)  # sell USD, buy JPY
    rates = {"USD": 1.0, "JPY": 1 / 150.0}
    assert fc.settlement_exposure(t, rates).exposure_usd == 0.0
    e = fc.settlement_exposure(t, rates, cancellation_lead_hours=24.0,
                               reconciliation_lag_hours=12.0)
    assert e.exposure_usd == pytest.approx(10e6)
    assert fc.net_settlement_exposure([t], rates, 24.0, 12.0) == pytest.approx(10e6)
