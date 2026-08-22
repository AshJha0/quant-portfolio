"""Backtesting: Kupiec, Christoffersen, Basel traffic light, Acerbi-Szekely."""

import numpy as np
import pytest
from scipy.stats import chi2, norm

from eq_var import (
    acerbi_szekely_z2,
    basel_traffic_light,
    basel_zone_probabilities,
    christoffersen_cc,
    christoffersen_independence,
    exception_cluster_table,
    exceptions_from_pnl,
    kupiec_pof,
    normal_es,
    rolling_var_backtest,
)


def hand_kupiec(t: int, x: int, p: float) -> float:
    """Independent hand computation of the Kupiec LR statistic."""
    pihat = x / t
    ll_null = (t - x) * np.log(1 - p) + (x * np.log(p) if x else 0.0)
    ll_alt = ((t - x) * np.log(1 - pihat) if t - x else 0.0) + (
        x * np.log(pihat) if x else 0.0
    )
    return -2.0 * (ll_null - ll_alt)


class TestKupiec:
    def test_lr_matches_hand_computed(self):
        for t, x in ((250, 5), (250, 0), (250, 10), (500, 14), (1000, 3)):
            res = kupiec_pof(t, x, 0.01)
            assert res["lr"] == pytest.approx(hand_kupiec(t, x, 0.01), abs=1e-10)

    def test_zero_exceptions_known_value(self):
        # LR = -2 * 250 * ln(0.99) = 5.02517...
        res = kupiec_pof(250, 0, 0.01)
        assert res["lr"] == pytest.approx(-2 * 250 * np.log(0.99), abs=1e-12)

    def test_exact_coverage_gives_zero_lr(self):
        res = kupiec_pof(1000, 10, 0.01)  # observed rate == alpha exactly
        assert res["lr"] == pytest.approx(0.0, abs=1e-12)
        assert res["pvalue"] == pytest.approx(1.0)

    def test_pvalue_is_chi2_1df(self):
        res = kupiec_pof(250, 9, 0.01)
        assert res["pvalue"] == pytest.approx(float(chi2.sf(res["lr"], 1)), abs=1e-14)

    def test_rejects_gross_undercoverage(self):
        assert kupiec_pof(250, 12, 0.01)["pvalue"] < 0.001

    def test_expected_and_rate_fields(self):
        res = kupiec_pof(250, 5, 0.01)
        assert res["expected"] == pytest.approx(2.5)
        assert res["rate"] == pytest.approx(0.02)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="n_obs"):
            kupiec_pof(0, 0, 0.01)
        with pytest.raises(ValueError, match="n_exceptions"):
            kupiec_pof(250, 300, 0.01)

    @staticmethod
    def _exact_test_size(n_obs: int, alpha: float, nominal: float = 0.05) -> float:
        """Exact rejection probability of the chi2(1) Kupiec test under H0.

        No simulation needed: LR(x) depends only on the exception count
        ``x``, and ``x ~ Binomial(n_obs, alpha)`` under a correctly
        calibrated model, so the true test size is the exact binomial
        probability mass on the rejection region ``{x : LR(x) > crit}``.
        """
        from scipy.stats import binom

        crit = chi2.ppf(1.0 - nominal, df=1)
        xs = np.arange(0, n_obs + 1)
        lrs = np.array([kupiec_pof(n_obs, int(x), alpha)["lr"] for x in xs])
        return float(binom.pmf(xs[lrs > crit], n_obs, alpha).sum())

    def test_asymptotic_chi2_reference_is_oversized_at_the_regulatory_window(self):
        """The 250-day/99% Basel window is small-sample territory for the
        chi2(1) asymptotic reference: a nominally 5%-size test actually
        rejects a *correctly calibrated* model ~9.5% of the time here (the
        expected exception count is only 2.5). This is a property of the
        likelihood-ratio chi2 approximation for a rare (low-p) binomial,
        not a bug in the LR formula -- documented in docs/VALIDATION.md.
        It attenuates quickly as the expected count grows: by n_obs=1000
        (expected count 10) the exact size is within a point of nominal.
        """
        size_250 = self._exact_test_size(250, 0.01)
        size_1000 = self._exact_test_size(1000, 0.01)
        assert size_250 == pytest.approx(0.0948, abs=0.005)
        assert size_1000 == pytest.approx(0.0551, abs=0.005)
        assert size_250 > 1.8 * 0.05  # materially oversized at the regulatory window
        assert size_1000 < 1.2 * 0.05  # much closer to nominal at n=1000


class TestChristoffersen:
    def test_detects_planted_clustered_exceptions(self):
        ex = np.zeros(500, dtype=bool)
        ex[100:105] = True  # two tight runs
        ex[300:305] = True
        res = christoffersen_independence(ex)
        assert res["n11"] == 8
        assert res["pvalue"] < 1e-6

    def test_accepts_evenly_spread_exceptions(self):
        ex = np.zeros(500, dtype=bool)
        ex[::50] = True  # 10 isolated exceptions
        res = christoffersen_independence(ex)
        assert res["n11"] == 0
        assert res["pvalue"] > 0.10

    def test_no_exceptions_degenerate_case(self):
        res = christoffersen_independence(np.zeros(250, dtype=bool))
        assert res["lr"] == pytest.approx(0.0, abs=1e-12)

    def test_cc_equals_uc_plus_ind(self):
        rng = np.random.default_rng(1)
        ex = rng.random(500) < 0.02
        uc = kupiec_pof(500, int(ex.sum()), 0.01)
        ind = christoffersen_independence(ex)
        cc = christoffersen_cc(ex, 0.01)
        assert cc["lr"] == pytest.approx(uc["lr"] + ind["lr"], abs=1e-10)
        assert cc["pvalue"] == pytest.approx(float(chi2.sf(cc["lr"], 2)), abs=1e-14)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            christoffersen_independence(np.array([True]))


class TestBaselTrafficLight:
    def test_green_yellow_boundary_exact_at_4_5(self):
        assert basel_traffic_light(4)["zone"] == "green"
        assert basel_traffic_light(5)["zone"] == "yellow"

    def test_yellow_red_boundary_exact_at_9_10(self):
        assert basel_traffic_light(9)["zone"] == "yellow"
        assert basel_traffic_light(10)["zone"] == "red"

    def test_multiplier_schedule(self):
        assert basel_traffic_light(0)["multiplier"] == pytest.approx(3.0)
        assert basel_traffic_light(4)["multiplier"] == pytest.approx(3.0)
        assert basel_traffic_light(5)["multiplier"] == pytest.approx(3.40)
        assert basel_traffic_light(6)["multiplier"] == pytest.approx(3.50)
        assert basel_traffic_light(7)["multiplier"] == pytest.approx(3.65)
        assert basel_traffic_light(8)["multiplier"] == pytest.approx(3.75)
        assert basel_traffic_light(9)["multiplier"] == pytest.approx(3.85)
        assert basel_traffic_light(10)["multiplier"] == pytest.approx(4.0)
        assert basel_traffic_light(25)["multiplier"] == pytest.approx(4.0)

    def test_zone_probabilities_binomial(self):
        from scipy.stats import binom

        table = basel_zone_probabilities()
        green = table[table.zone == "green"]["prob_exact"].sum()
        assert green == pytest.approx(float(binom.cdf(4, 250, 0.01)), abs=1e-12)
        assert green == pytest.approx(0.8922, abs=1e-4)  # the documented ~89 %
        red = 1.0 - float(binom.cdf(9, 250, 0.01))
        assert red == pytest.approx(2.5019e-4, rel=1e-3)  # ~0.03 %: red is
        # essentially impossible for a correct model

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match="n_exceptions"):
            basel_traffic_light(-1)

    def test_calibration_true_var_mostly_green(self):
        """Simulated calibration: correct 99 % VaR on iid data -> ~1 %
        exceptions and green zone in ~89 % of 300 replications."""
        rng = np.random.default_rng(6)
        true_var = -norm.ppf(0.01)
        zones, rates = [], []
        for _ in range(300):
            pnl = rng.standard_normal(250)
            ex = exceptions_from_pnl(pnl, true_var)
            zones.append(basel_traffic_light(int(ex.sum()))["zone"])
            rates.append(ex.mean())
        green_frac = np.mean([z == "green" for z in zones])
        assert 0.80 < green_frac <= 1.0
        assert 0.007 < float(np.mean(rates)) < 0.013  # ~1 % unconditional


class TestAcerbiSzekely:
    def test_z2_near_zero_under_correct_model(self):
        rng = np.random.default_rng(2)
        pnl = rng.standard_normal(200_000)
        alpha = 0.025
        var = float(-norm.ppf(alpha))
        es = normal_es(1.0, alpha)
        res = acerbi_szekely_z2(pnl, var, es, alpha)
        assert abs(res["z2"]) < 0.05
        assert not res["reject"]

    def test_z2_negative_when_model_understates_tail(self):
        rng = np.random.default_rng(3)
        pnl = 1.5 * rng.standard_normal(100_000)  # true sigma 1.5x the model's
        alpha = 0.025
        var = float(-norm.ppf(alpha))  # model assumes sigma = 1
        es = normal_es(1.0, alpha)
        res = acerbi_szekely_z2(pnl, var, es, alpha)
        # analytic value: E[X 1(X<-1.96)]/(alpha*ES)+1 ~ -3.36
        assert res["z2"] < -2.0
        assert res["reject"]

    def test_no_exceptions_gives_plus_one(self):
        pnl = np.zeros(100)
        res = acerbi_szekely_z2(pnl, 5.0, 6.0, 0.025)
        assert res["z2"] == pytest.approx(1.0)
        assert res["n_exceptions"] == 0

    def test_es_below_var_raises(self):
        with pytest.raises(ValueError, match="ES must be >= VaR"):
            acerbi_szekely_z2(np.zeros(10), 5.0, 4.0, 0.025)

    def test_nonpositive_es_raises(self):
        with pytest.raises(ValueError, match="positive"):
            acerbi_szekely_z2(np.zeros(10), 1.0, 0.0, 0.025)


class TestRollingBacktest:
    def test_exception_indicator_sign_convention(self):
        pnl = np.array([-10.0, -4.9, -5.1, 3.0])
        ex = exceptions_from_pnl(pnl, 5.0)
        np.testing.assert_array_equal(ex, [True, False, True, False])

    def test_negative_var_raises(self):
        with pytest.raises(ValueError, match="positive loss"):
            exceptions_from_pnl(np.zeros(3), -1.0)

    def test_rolling_backtest_shapes_and_summary(self):
        rng = np.random.default_rng(4)
        pnl = rng.standard_normal(400)

        def fn(hist, alpha):
            return float(-norm.ppf(alpha) * np.std(hist, ddof=1))

        bt = rolling_var_backtest(pnl, fn, window=250, alpha=0.05, name="test")
        assert bt.n_obs == 150
        assert bt.var_series.shape == (150,)
        s = bt.summary()
        assert s["method"] == "test"
        assert 0.0 <= s["kupiec_p"] <= 1.0

    def test_rolling_backtest_too_short_raises(self):
        with pytest.raises(ValueError, match="window"):
            rolling_var_backtest(np.zeros(250), lambda h, a: 1.0, window=250)

    def test_exception_cluster_table_structure(self):
        ex = np.zeros(300, dtype=bool)
        ex[[10, 12, 200]] = True
        table = exception_cluster_table(ex)
        assert list(table["day"]) == [10, 12, 200]
        assert bool(table["clustered"].iloc[1]) is True  # gap of 2 days
        assert bool(table["clustered"].iloc[2]) is False
