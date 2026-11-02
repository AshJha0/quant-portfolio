"""Historical simulation VaR: plain, age-weighted (BRW), filtered (FHS)."""

import numpy as np
import pytest
from eq_var import (
    age_weighted_var,
    brw_weights,
    ewma_forecast,
    ewma_volatility,
    factor_filtered_historical_var,
    filtered_historical_scenarios,
    filtered_historical_var,
    historical_var,
    overlapping_horizon_pnl,
    scale_var_sqrt_time,
)
from eq_var.data import demo_covariance, demo_portfolio, simulate_garch_returns


class TestPlainHistorical:
    def test_exact_quantile_known_array(self):
        # pnl = -100, -99, ..., -1 (n=100). Spec S2: losses 100, 99, ...;
        # k = 100 * 0.05 = 5, m = 5, VaR = L(5) = 96 (type-7 gave 95.05).
        pnl = np.arange(-100.0, 0.0)
        assert historical_var(pnl, 0.05) == 96.0

    def test_exact_quantile_alpha_01(self):
        pnl = np.arange(-100.0, 0.0)
        # k = 1, m = 1: the largest loss (type-7 gave 99.01).
        assert historical_var(pnl, 0.01) == 100.0

    def test_fractional_k_takes_the_ceiling_rank(self):
        # n = 250, alpha = 0.01: k = 2.5, m = 3, VaR = third largest loss.
        pnl = -np.arange(1.0, 251.0)
        assert historical_var(pnl, 0.01) == 248.0

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
        # lands on the 6th-worst = -95 -> VaR 95; one order statistic from
        # plain HS, which is L(5) = 96 under spec S2.
        assert v == pytest.approx(95.0, abs=1e-6)
        assert abs(v - historical_var(pnl, 0.05)) <= 1.0

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
        sig2 = ewma_volatility(x2, lam=0.94)
        np.testing.assert_array_equal(sig, sig2)

    def test_seed_is_mean_square_of_first_30(self):
        """Spec S8: the seed uses the first min(30, n) observations only.
        The old default seeded with the full-sample variance (look-ahead):
        changing a late observation moved sigma[0]."""
        x = np.random.default_rng(12).normal(0, 1, 100)
        sig = ewma_volatility(x, lam=0.94)
        assert sig[0] == pytest.approx(np.sqrt(np.mean(x[:30] ** 2)), rel=1e-15)
        x2 = x.copy()
        x2[80] = 50.0
        assert ewma_volatility(x2, lam=0.94)[0] == sig[0]

    def test_init_argument_is_deprecated(self):
        x = np.random.default_rng(13).normal(0, 1, 50)
        with pytest.deprecated_call():
            old = ewma_volatility(x, lam=0.94, init="sample")
        assert old[0] == pytest.approx(np.std(x), rel=1e-15)

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

    def test_factor_fhs_reacts_to_a_vol_spike_and_hs_does_not(self):
        """Spec S9: per-factor filtering. Tripling every factor move over the
        last 10 of 250 days roughly triples the per-factor FHS VaR of the
        demo book (the EWMA forecast tracks the new vol), while plain HS on
        the same revalued history moves by less than half."""
        pf = demo_portfolio()
        rets = simulate_garch_returns(250, demo_covariance(), seed=10)
        spiked = rets.copy()
        spiked[-10:] *= 3.0
        fhs_ratio = factor_filtered_historical_var(pf, spiked, 0.01) / factor_filtered_historical_var(pf, rets, 0.01)
        hs_ratio = historical_var(pf.pnl(spiked), 0.01) / historical_var(pf.pnl(rets), 0.01)
        assert fhs_ratio > 2.5
        assert hs_ratio < 1.5

    def test_factor_scenarios_are_rescaled_per_factor(self):
        """r*_{t,j} = r_{t,j} / sigma_{t,j} * sigma_{T+1,j}, each column on its own."""
        r = np.random.default_rng(14).normal(0, [0.01, 0.03], size=(120, 2))
        scen = filtered_historical_scenarios(r, 0.94)
        for j in range(2):
            sig = ewma_volatility(r[:, j], 0.94)
            nxt = ewma_forecast(r[:, j], 0.94)
            np.testing.assert_allclose(scen[:, j], r[:, j] / sig * nxt, rtol=1e-14)

    def test_pnl_level_fhs_is_deprecated(self):
        with pytest.deprecated_call():
            filtered_historical_var(np.random.default_rng(15).normal(size=300), 0.01)


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


def test_pnl_fhs_uses_order_statistic_like_plain_hs():
    """Deprecated P&L-level FHS reads its tail with the S2 order statistic.

    Rebuild the rescaled scenarios by hand and take the ceil(n alpha)-th
    largest loss; type-7 interpolation would give a different number here.
    """
    import warnings

    import numpy as np

    from eq_var.historical_var import filtered_historical_var

    pnl = np.random.default_rng(0).standard_normal(1000)
    lam, alpha = 0.94, 0.01
    s2 = np.empty(pnl.size)
    s2[0] = np.var(pnl)
    for t in range(1, pnl.size):
        s2[t] = lam * s2[t - 1] + (1 - lam) * pnl[t - 1] ** 2
    sigma = np.sqrt(s2)
    nxt = np.sqrt(lam * s2[-1] + (1 - lam) * pnl[-1] ** 2)
    losses = np.sort(-(pnl / sigma * nxt))[::-1]
    expected = losses[int(np.ceil(pnl.size * alpha)) - 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        got = filtered_historical_var(pnl, alpha, lam)
    assert got == pytest.approx(expected, rel=1e-12)
    assert got != pytest.approx(-np.quantile(-losses, alpha), rel=1e-6)
