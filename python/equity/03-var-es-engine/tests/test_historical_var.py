"""Historical simulation VaR: plain, age-weighted (BRW), filtered (FHS)."""

import numpy as np
import pytest

from eq_var import (
    age_weighted_var,
    brw_weights,
    christoffersen_independence,
    ewma_volatility,
    filtered_historical_var,
    historical_var,
    overlapping_horizon_pnl,
    rolling_var_backtest,
    scale_var_sqrt_time,
)
from eq_var.data import demo_covariance, demo_portfolio, simulate_garch_returns


class TestPlainHistorical:
    def test_exact_quantile_known_array(self):
        # pnl = -100, -99, ..., -1 (n=100).  Type-7 quantile at alpha=0.05:
        # h = 99*0.05 = 4.95 -> between sorted[4]=-96 and sorted[5]=-95:
        # q = -96 + 0.95*(1) = -95.05 -> VaR = 95.05
        pnl = np.arange(-100.0, 0.0)
        assert historical_var(pnl, 0.05) == pytest.approx(95.05, abs=1e-12)

    def test_exact_quantile_alpha_01(self):
        pnl = np.arange(-100.0, 0.0)
        # h = 99*0.01 = 0.99 -> -100 + 0.99 = -99.01 -> VaR = 99.01
        assert historical_var(pnl, 0.01) == pytest.approx(99.01, abs=1e-12)

    def test_order_invariance(self):
        rng = np.random.default_rng(3)
        pnl = rng.normal(0, 100, 500)
        shuffled = rng.permutation(pnl)
        assert historical_var(pnl, 0.01) == pytest.approx(historical_var(shuffled, 0.01))

    def test_positive_loss_convention(self):
        rng = np.random.default_rng(4)
        pnl = rng.normal(0, 1, 1000)
        assert historical_var(pnl, 0.01) > 0

    def test_insufficient_history_raises(self):
        with pytest.raises(ValueError, match="at least 50"):
            historical_var(np.zeros(10), 0.01)

    def test_nan_raises(self):
        pnl = np.zeros(100)
        pnl[3] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            historical_var(pnl, 0.01)

    def test_invalid_alpha_raises(self):
        pnl = np.random.default_rng(0).normal(size=100)
        for bad in (0.0, 0.5, 1.0, -0.01):
            with pytest.raises(ValueError, match="alpha"):
                historical_var(pnl, bad)


class TestAgeWeighted:
    def test_weights_sum_to_one(self):
        for n, lam in ((10, 0.9), (250, 0.98), (1000, 0.995)):
            assert brw_weights(n, lam).sum() == pytest.approx(1.0, abs=1e-12)

    def test_weights_increase_with_recency(self):
        w = brw_weights(100, 0.97)
        assert np.all(np.diff(w) > 0)  # oldest first, most recent last

    def test_lambda_to_one_recovers_equal_weights(self):
        pnl = np.arange(-100.0, 0.0)
        v = age_weighted_var(pnl, 0.05, lam=1 - 1e-9)
        # equal-weight step-CDF inversion: cumulative weight of the 5 worst
        # is 0.05 - eps (oldest weights are smallest), so the inverse CDF
        # lands on the 6th-worst = -95 -> VaR 95; within one order statistic
        # of plain (interpolated) HS at 95.05.
        assert v == pytest.approx(95.0, abs=1e-6)
        assert abs(v - historical_var(pnl, 0.05)) < 1.0

    def test_recent_losses_dominate(self):
        # same P&L values, but big losses either old or recent
        base = np.full(200, 1.0)
        old_losses, recent_losses = base.copy(), base.copy()
        old_losses[:5] = -50.0
        recent_losses[-5:] = -50.0
        lam = 0.97
        v_old = age_weighted_var(old_losses, 0.01, lam)
        v_recent = age_weighted_var(recent_losses, 0.01, lam)
        assert v_recent == pytest.approx(50.0)
        assert v_old < v_recent  # old losses have decayed out of the tail

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError, match="lam"):
            age_weighted_var(np.zeros(100), 0.01, lam=1.5)


class TestEwmaVolatility:
    def test_constant_vol_on_constant_variance_series(self):
        x = np.array([1.0, -1.0] * 200)
        sig = ewma_volatility(x, lam=0.94)
        np.testing.assert_allclose(sig, 1.0, rtol=1e-10)

    def test_no_lookahead(self):
        rng = np.random.default_rng(11)
        x = rng.normal(0, 1, 300)
        sig = ewma_volatility(x, lam=0.94)
        x2 = x.copy()
        x2[-1] = 100.0  # changing today's return must not change today's forecast
        sig2 = ewma_volatility(x2, lam=0.94, init="first")
        sig1 = ewma_volatility(x, lam=0.94, init="first")
        np.testing.assert_allclose(sig1, sig2)

    def test_vol_rises_after_large_returns(self):
        x = np.concatenate([np.full(100, 0.1), np.full(20, 5.0)])
        sig = ewma_volatility(x, lam=0.94)
        assert sig[-1] > 5 * sig[99]

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ewma_volatility(np.array([1.0]))


class TestFilteredHistorical:
    def test_fhs_close_to_plain_hs_on_iid_data(self):
        rng = np.random.default_rng(21)
        pnl = rng.normal(0, 100, 2000)
        hs = historical_var(pnl, 0.01)
        fhs = filtered_historical_var(pnl, 0.01)
        assert fhs == pytest.approx(hs, rel=0.25)  # no regime -> similar answer

    def test_fhs_scales_up_after_vol_regime_switch(self):
        rng = np.random.default_rng(7)
        low = rng.normal(0, 1.0, 450)
        high = rng.normal(0, 3.0, 50)  # vol tripled recently
        pnl = np.concatenate([low, high])
        hs = historical_var(pnl, 0.01)
        fhs = filtered_historical_var(pnl, 0.01)
        assert fhs > 1.5 * hs  # FHS tracks the new regime, HS is diluted

    def test_fhs_scales_down_after_calm_regime(self):
        rng = np.random.default_rng(8)
        high = rng.normal(0, 3.0, 450)
        low = rng.normal(0, 1.0, 50)
        pnl = np.concatenate([high, low])
        assert filtered_historical_var(pnl, 0.01) < historical_var(pnl, 0.01)

    def test_hs_fails_clustering_test_while_fhs_passes_on_garch(self):
        # Statistical test, fixed seed (verified): plain HS exceptions cluster
        # on GARCH data (Christoffersen independence rejects at 5 %), FHS
        # de-clusters them.
        pf = demo_portfolio()
        cov = demo_covariance()
        rets = simulate_garch_returns(750, cov, alpha_g=0.13, beta_g=0.85, df=5.0, seed=10)
        pnl = pf.pnl(rets, method="delta_gamma")
        bt_hs = rolling_var_backtest(pnl, historical_var, window=250, alpha=0.01, name="hs")
        bt_fhs = rolling_var_backtest(pnl, filtered_historical_var, window=250, alpha=0.01, name="fhs")
        p_hs = christoffersen_independence(bt_hs.exceptions)["pvalue"]
        p_fhs = christoffersen_independence(bt_fhs.exceptions)["pvalue"]
        assert p_hs < 0.05
        assert p_fhs > 0.10


class TestHorizonScaling:
    def test_sqrt_time_value(self):
        assert scale_var_sqrt_time(10.0, 10) == pytest.approx(10.0 * np.sqrt(10.0))

    def test_sqrt_time_one_day_identity(self):
        assert scale_var_sqrt_time(7.3, 1) == pytest.approx(7.3)

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon"):
            scale_var_sqrt_time(10.0, 0)

    def test_overlapping_sums_hand_computed(self):
        arr = np.arange(1.0, 61.0)  # 60 obs, h=10 -> 51 windows
        ov = overlapping_horizon_pnl(arr, 10)
        assert ov.size == 51
        assert ov[0] == pytest.approx(55.0)  # 1+...+10
        assert ov[-1] == pytest.approx(sum(range(51, 61)))

    def test_overlapping_too_short_raises(self):
        with pytest.raises(ValueError, match="observations"):
            overlapping_horizon_pnl(np.zeros(30), 10)

    def test_sqrt_time_matches_iid_normal_overlapping_roughly(self):
        rng = np.random.default_rng(30)
        pnl = rng.normal(0, 1, 5000)
        v1 = historical_var(pnl, 0.05)
        v10_sqrt = scale_var_sqrt_time(v1, 10)
        v10_overlap = historical_var(overlapping_horizon_pnl(pnl, 10), 0.05)
        assert v10_overlap == pytest.approx(v10_sqrt, rel=0.15)  # iid: both valid
