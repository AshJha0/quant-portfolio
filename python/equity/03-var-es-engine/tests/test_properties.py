"""Property-based invariants and additional edge cases for the VaR/ES engine.

These constrain the *shape* of the risk measures rather than individual
numbers, so they hold for any correct implementation and would immediately
catch a sign flip, a wrong tail convention, or a broken quantile scheme.
The properties checked here are the ones a model-validation team asks for:

* coherence — ES subadditivity, positive homogeneity, translation invariance;
* ordering — ES >= VaR, VaR monotone in alpha, t-tail >= normal tail;
* scale/shift equivariance of every VaR family;
* backtest statistics: monotone in exception count, correct degenerate limits;
* NaN/Inf rejection and crisis-regime (vol-clustering) behaviour.

Golden-vector note: every check here is an *inequality or invariant*, never a
hard-coded value, so it constrains the engine without pinning any number that
the C++/Rust engines cross-validate against.
"""

import numpy as np
import pytest
from scipy.stats import norm

from eq_var import (
    EquityPosition,
    Portfolio,
    RiskFactor,
    acerbi_szekely_z2,
    age_weighted_var,
    basel_traffic_light,
    christoffersen_independence,
    cornish_fisher_var,
    exceptions_from_pnl,
    expected_shortfall,
    filtered_historical_var,
    historical_var,
    kupiec_pof,
    monte_carlo_var,
    normal_es,
    parametric_var,
    portfolio_sigma,
    sample_covariance,
    student_t_es,
)
from eq_var.data import synthetic as syn
from eq_var.historical_var import (
    brw_weights,
    ewma_volatility,
    overlapping_horizon_pnl,
    scale_var_sqrt_time,
)
from eq_var.monte_carlo_var import var_confidence_interval, var_standard_error_bootstrap

ALPHAS = [0.001, 0.01, 0.025, 0.05, 0.10]


@pytest.fixture(scope="module")
def pnl():
    """Fat-tailed daily P&L history, deterministic."""
    rng = np.random.default_rng(4242)
    return 10_000.0 * rng.standard_t(6, 2000) / np.sqrt(6 / 4)


# --------------------------------------------------------------------------- #
# ordering / consistency between the risk measures
# --------------------------------------------------------------------------- #
class TestOrdering:
    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_es_at_least_var_for_every_family(self, pnl, alpha):
        """ES is an average over the tail beyond VaR, so ES >= VaR always."""
        assert expected_shortfall(pnl, alpha) >= historical_var(pnl, alpha) - 1e-9
        sigma = float(np.std(pnl, ddof=1))
        assert normal_es(sigma, alpha) >= -norm.ppf(alpha) * sigma - 1e-9

    @pytest.mark.parametrize(
        "fn",
        [historical_var, expected_shortfall,
         lambda p, a: age_weighted_var(p, a),
         lambda p, a: filtered_historical_var(p, a)],
    )
    def test_risk_measure_decreasing_in_alpha(self, pnl, fn):
        """A deeper tail (smaller alpha) can never give a smaller risk number."""
        vals = [fn(pnl, a) for a in ALPHAS]
        assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))

    @pytest.mark.parametrize("alpha", [0.001, 0.01, 0.025])
    def test_student_t_tail_fatter_than_normal_in_the_deep_tail(self, alpha):
        """Variance-matched t charges more than the normal in the deep tail."""
        sigma = 1_000.0
        w, cov = np.array([1.0]), np.array([[sigma**2]])
        assert parametric_var(w, cov, alpha, dist="t", df=5.0) > parametric_var(
            w, cov, alpha, dist="normal"
        )
        assert student_t_es(sigma, alpha, df=5.0) > normal_es(sigma, alpha)

    @pytest.mark.parametrize("alpha", [0.05, 0.10])
    def test_variance_matched_t_is_thinner_than_normal_in_the_shoulder(self, alpha):
        """The other side of variance matching, and a real trap: because the
        two distributions have the *same* variance, the t's fat tail is paid
        for with a thinner shoulder. A t(5) charges LESS than the normal at
        95 % — so switching to a t model to 'be conservative' can lower a
        95 % VaR. The crossover for df=5 sits near alpha = 0.028.
        """
        sigma = 1_000.0
        w, cov = np.array([1.0]), np.array([[sigma**2]])
        assert parametric_var(w, cov, alpha, dist="t", df=5.0) < parametric_var(
            w, cov, alpha, dist="normal"
        )
        # ES, which averages the whole tail, is not fooled: it stays higher
        assert student_t_es(sigma, alpha, df=5.0) > normal_es(sigma, alpha)

    def test_t_vs_normal_var_crossover_is_between_2_5_and_5_percent(self):
        sigma = 1_000.0
        w, cov = np.array([1.0]), np.array([[sigma**2]])

        def diff(a):
            return parametric_var(w, cov, a, dist="t", df=5.0) - parametric_var(
                w, cov, a, dist="normal"
            )

        assert diff(0.025) > 0 and diff(0.05) < 0

    def test_student_t_converges_to_normal_as_df_grows(self):
        sigma, alpha = 1_000.0, 0.01
        gaps = [
            student_t_es(sigma, alpha, df=df) - normal_es(sigma, alpha)
            for df in (5.0, 15.0, 50.0, 500.0)
        ]
        assert all(b < a for a, b in zip(gaps, gaps[1:]))
        assert gaps[-1] == pytest.approx(0.0, abs=0.01 * sigma)

    def test_es_strictly_above_var_for_a_continuous_distribution(self, pnl):
        """With no atom at the quantile the inequality is strict."""
        for alpha in (0.01, 0.05):
            assert expected_shortfall(pnl, alpha) > historical_var(pnl, alpha)


# --------------------------------------------------------------------------- #
# coherence properties of ES (the reason FRTB moved to ES)
# --------------------------------------------------------------------------- #
class TestCoherence:
    @pytest.mark.parametrize("alpha", [0.01, 0.05])
    def test_es_subadditive_on_random_splits(self, alpha):
        """ES(X+Y) <= ES(X) + ES(Y) — the property VaR lacks."""
        rng = np.random.default_rng(99)
        for _ in range(25):
            x = rng.standard_t(5, 1000)
            y = rng.standard_t(5, 1000)
            es_sum = expected_shortfall(x + y, alpha)
            assert es_sum <= expected_shortfall(x, alpha) + expected_shortfall(y, alpha) + 1e-9

    @pytest.mark.parametrize("c", [0.5, 1.0, 3.0, 100.0])
    def test_positive_homogeneity(self, pnl, c):
        """Scaling the book scales VaR and ES by the same factor."""
        for fn in (historical_var, expected_shortfall, age_weighted_var,
                   filtered_historical_var):
            assert fn(c * pnl, 0.01) == pytest.approx(c * fn(pnl, 0.01), rel=1e-10)

    @pytest.mark.parametrize("shift", [-5_000.0, 0.0, 2_500.0])
    def test_translation_invariance(self, pnl, shift):
        """Adding a certain cash amount m reduces the risk number by exactly m."""
        for fn in (historical_var, expected_shortfall, age_weighted_var):
            assert fn(pnl + shift, 0.01) == pytest.approx(
                fn(pnl, 0.01) - shift, rel=1e-9, abs=1e-6
            )

    def test_var_can_violate_subadditivity(self):
        """Documented counter-example: VaR is *not* coherent. Two independent
        defaultable bonds, each with a 4% loss probability, at 95% VaR."""
        rng = np.random.default_rng(7)
        n = 200_000
        a = np.where(rng.random(n) < 0.04, -100.0, 2.0)
        b = np.where(rng.random(n) < 0.04, -100.0, 2.0)
        var_a = historical_var(a, 0.05)
        var_b = historical_var(b, 0.05)
        var_sum = historical_var(a + b, 0.05)
        assert var_sum > var_a + var_b  # superadditive: diversification "hurts"
        # ES, being coherent, does not misbehave here
        assert expected_shortfall(a + b, 0.05) <= (
            expected_shortfall(a, 0.05) + expected_shortfall(b, 0.05) + 1e-9
        )


# --------------------------------------------------------------------------- #
# scale / shift equivariance of the parametric family
# --------------------------------------------------------------------------- #
class TestParametricEquivariance:
    def test_var_linear_in_exposures(self):
        cov = syn.demo_covariance()
        w = np.array([1e6, 5e5, -2e5, 3e4])
        for c in (0.5, 2.0, 10.0):
            assert parametric_var(c * w, cov, 0.01) == pytest.approx(
                c * parametric_var(w, cov, 0.01), rel=1e-12
            )

    def test_sigma_scales_with_sqrt_of_covariance(self):
        cov = syn.demo_covariance()
        w = np.array([1e6, 5e5, -2e5, 3e4])
        for c in (0.25, 4.0):
            assert portfolio_sigma(w, c * cov) == pytest.approx(
                np.sqrt(c) * portfolio_sigma(w, cov), rel=1e-12
            )

    def test_sign_flip_of_exposures_leaves_sigma_unchanged(self):
        cov = syn.demo_covariance()
        w = np.array([1e6, 5e5, -2e5, 3e4])
        assert portfolio_sigma(-w, cov) == pytest.approx(portfolio_sigma(w, cov), rel=1e-12)

    def test_horizon_scaling_is_sqrt_time_at_zero_mean(self):
        cov = np.array([[4e-4]])
        w = np.array([1e6])
        base = parametric_var(w, cov, 0.01, horizon_days=1)
        for h in (4, 9, 10):
            assert parametric_var(w, cov, 0.01, horizon_days=h) == pytest.approx(
                base * np.sqrt(h), rel=1e-12
            )
            assert scale_var_sqrt_time(base, h) == pytest.approx(base * np.sqrt(h), rel=1e-12)

    def test_nonzero_mean_breaks_pure_sqrt_scaling(self):
        """Drift scales linearly while sigma scales as sqrt(h): the two must
        not be conflated (a classic multi-day VaR error)."""
        cov, w = np.array([[4e-4]]), np.array([1e6])
        v1 = parametric_var(w, cov, 0.01, mean=500.0, horizon_days=1)
        v10 = parametric_var(w, cov, 0.01, mean=500.0, horizon_days=10)
        assert v10 < v1 * np.sqrt(10)  # positive drift eats into the loss

    def test_cornish_fisher_reduces_to_normal_at_zero_moments(self):
        sigma, alpha = 1e5, 0.01
        assert cornish_fisher_var(sigma, alpha, skew=0.0, excess_kurt=0.0) == pytest.approx(
            -norm.ppf(alpha) * sigma, rel=1e-12
        )

    def test_cornish_fisher_charges_more_for_negative_skew_fat_tails(self):
        sigma, alpha = 1e5, 0.01
        base = cornish_fisher_var(sigma, alpha, 0.0, 0.0)
        assert cornish_fisher_var(sigma, alpha, skew=-0.5, excess_kurt=1.0) > base

    def test_cornish_fisher_rejects_its_own_invalid_domain(self):
        with pytest.raises(ValueError, match="non-monotone"):
            cornish_fisher_var(1e5, 0.01, skew=-3.0, excess_kurt=12.0)


# --------------------------------------------------------------------------- #
# weighting schemes
# --------------------------------------------------------------------------- #
class TestWeightingSchemes:
    @pytest.mark.parametrize("lam", [0.90, 0.98, 0.995])
    def test_brw_weights_sum_to_one_and_favour_recent(self, lam):
        w = brw_weights(500, lam)
        assert w.sum() == pytest.approx(1.0, rel=1e-12)
        assert np.all(np.diff(w) > 0)  # index 0 = oldest gets least weight
        assert w[-1] > w[0]

    def test_age_weighting_reacts_to_a_recent_vol_regime_change(self):
        """Recent turmoil after a calm history: BRW must charge more than
        equally weighted historical simulation."""
        rng = np.random.default_rng(31)
        calm = rng.normal(0, 1.0, 700)
        stressed = rng.normal(0, 4.0, 100)
        series = np.concatenate([calm, stressed])
        assert age_weighted_var(series, 0.05, lam=0.97) > historical_var(series, 0.05)

    def test_age_weighting_converges_to_plain_hs_as_lambda_goes_to_one(self):
        rng = np.random.default_rng(32)
        series = rng.normal(0, 1.0, 3000)
        gaps = [
            abs(age_weighted_var(series, 0.05, lam=lam) - historical_var(series, 0.05))
            for lam in (0.95, 0.99, 0.9999)
        ]
        assert gaps[-1] < gaps[0]

    def test_fhs_tracks_the_current_vol_regime(self):
        """The whole point of filtering: after a vol spike FHS reprices even
        though the unconditional sample is unchanged."""
        rng = np.random.default_rng(33)
        calm = rng.normal(0, 1.0, 700)
        stressed = rng.normal(0, 5.0, 60)
        quiet_end = np.concatenate([stressed, calm])   # turmoil long ago
        recent_end = np.concatenate([calm, stressed])  # turmoil right now
        assert filtered_historical_var(recent_end, 0.01) > filtered_historical_var(
            quiet_end, 0.01
        )

    def test_ewma_volatility_is_causal_and_positive(self):
        """sigma[t] must not depend on x[t] (no look-ahead in the filter)."""
        rng = np.random.default_rng(34)
        x = rng.normal(0, 1.0, 300)
        sig = ewma_volatility(x, init="first")
        y = x.copy()
        y[-1] = 500.0  # blow up only the last observation
        sig_y = ewma_volatility(y, init="first")
        np.testing.assert_allclose(sig[:-1], sig_y[:-1], rtol=1e-12)
        assert np.all(sig > 0)


# --------------------------------------------------------------------------- #
# backtesting statistics
# --------------------------------------------------------------------------- #
class TestBacktestProperties:
    def test_kupiec_lr_is_zero_at_the_expected_rate_and_grows_away_from_it(self):
        n, alpha = 1000, 0.01
        assert kupiec_pof(n, 10, alpha)["lr"] == pytest.approx(0.0, abs=1e-9)
        lrs = [kupiec_pof(n, x, alpha)["lr"] for x in (10, 15, 20, 30, 50)]
        assert all(b > a for a, b in zip(lrs, lrs[1:]))

    def test_kupiec_pvalue_decreasing_in_exception_count_above_expectation(self):
        n, alpha = 250, 0.01
        ps = [kupiec_pof(n, x, alpha)["pvalue"] for x in (3, 6, 10, 15)]
        assert all(b < a for a, b in zip(ps, ps[1:]))

    def test_kupiec_degenerate_counts_are_finite(self):
        for x in (0, 250):
            out = kupiec_pof(250, x, 0.01)
            assert np.isfinite(out["lr"]) and 0.0 <= out["pvalue"] <= 1.0

    def test_kupiec_rejects_invalid_counts(self):
        with pytest.raises(ValueError, match="n_exceptions"):
            kupiec_pof(100, 101, 0.01)
        with pytest.raises(ValueError, match="n_exceptions"):
            kupiec_pof(100, -1, 0.01)

    def test_christoffersen_detects_clustering_and_passes_iid(self):
        rng = np.random.default_rng(55)
        iid = rng.random(2000) < 0.01
        assert christoffersen_independence(iid)["pvalue"] > 0.01
        clustered = np.zeros(2000, dtype=bool)
        for start in range(0, 2000, 200):
            clustered[start:start + 5] = True  # exceptions arrive in runs
        assert christoffersen_independence(clustered)["pvalue"] < 0.01

    def test_christoffersen_zero_exceptions_is_degenerate_but_finite(self):
        out = christoffersen_independence(np.zeros(500, dtype=bool))
        assert out["lr"] == pytest.approx(0.0, abs=1e-12)
        assert 0.0 <= out["pvalue"] <= 1.0

    def test_basel_zone_boundaries_are_exactly_at_5_and_10(self):
        assert basel_traffic_light(4)["zone"] == "green"
        assert basel_traffic_light(5)["zone"] == "yellow"
        assert basel_traffic_light(9)["zone"] == "yellow"
        assert basel_traffic_light(10)["zone"] == "red"
        mults = [basel_traffic_light(x)["multiplier"] for x in range(0, 12)]
        assert all(b >= a for a, b in zip(mults, mults[1:]))  # never decreases
        assert mults[0] == 3.0 and mults[-1] == 4.0

    def test_acerbi_szekely_z2_flags_an_understated_es(self):
        """Feed real tail losses far beyond the model's ES: Z2 must go
        materially negative and trip the rejection flag."""
        rng = np.random.default_rng(66)
        p = rng.normal(0, 1.0, 2000)
        var = np.full(2000, 2.0)
        honest_es = np.full(2000, 2.4)
        good = acerbi_szekely_z2(p, var, honest_es, alpha=0.025)
        understated_es = np.full(2000, 2.01)
        bad = acerbi_szekely_z2(p * 4.0, var, understated_es, alpha=0.025)
        assert bad["z2"] < good["z2"]
        assert bad["reject"] is True

    def test_acerbi_szekely_rejects_es_below_var(self):
        p = np.random.default_rng(67).normal(size=100)
        with pytest.raises(ValueError, match="ES must be >= VaR"):
            acerbi_szekely_z2(p, np.full(100, 3.0), np.full(100, 1.0))

    def test_exceptions_definition_is_strict_inequality(self):
        pnl = np.array([-100.0, -99.9999, -100.0001, 0.0])
        ex = exceptions_from_pnl(pnl, 100.0)
        np.testing.assert_array_equal(ex, [False, False, True, False])

    def test_exceptions_reject_negative_var(self):
        with pytest.raises(ValueError, match="positive loss"):
            exceptions_from_pnl(np.zeros(10), -5.0)


# --------------------------------------------------------------------------- #
# non-finite input rejection
# --------------------------------------------------------------------------- #
class TestNonFiniteRejection:
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize(
        "fn",
        [historical_var, expected_shortfall, age_weighted_var, filtered_historical_var],
    )
    def test_var_family_rejects_non_finite_pnl(self, fn, bad):
        p = np.random.default_rng(77).normal(size=200)
        p[13] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            fn(p, 0.01)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_ewma_volatility_rejects_non_finite(self, bad):
        """The EWMA recursion is multiplicative in its own past, so a single
        bad point would silently poison every later forecast."""
        p = np.random.default_rng(78).normal(size=200)
        p[13] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            ewma_volatility(p)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_overlapping_horizon_rejects_non_finite(self, bad):
        p = np.random.default_rng(79).normal(size=200)
        p[13] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            overlapping_horizon_pnl(p, 10)

    def test_ewma_volatility_rejects_unknown_init(self):
        with pytest.raises(ValueError, match="init must be"):
            ewma_volatility(np.zeros(10), init="backcast")


# --------------------------------------------------------------------------- #
# multi-day horizon behaviour
# --------------------------------------------------------------------------- #
class TestHorizonScaling:
    def test_overlapping_sums_have_the_right_shape_and_values(self):
        p = np.arange(100.0)
        out = overlapping_horizon_pnl(p, 10)
        assert out.size == 100 - 10 + 1
        assert out[0] == pytest.approx(p[:10].sum())
        assert out[-1] == pytest.approx(p[-10:].sum())

    def test_temporal_aggregation_thins_the_tail_of_garch_returns(self):
        """Aggregation pulls fat-tailed daily returns toward normality (a CLT
        effect that vol clustering slows but does not stop): the excess
        kurtosis of 10-day sums is materially below the 1-day value."""
        from scipy.stats import kurtosis

        cov = np.array([[(0.20 / np.sqrt(252)) ** 2]])
        r = syn.simulate_garch_returns(20_000, cov, seed=88)[:, 0]
        agg = overlapping_horizon_pnl(r, 10)
        k1, k10 = kurtosis(r), kurtosis(agg)
        assert k1 > 3.0
        assert k10 < 0.6 * k1

    def test_sqrt_time_overstates_the_unconditional_10day_var_under_garch(self):
        """Consequence of the previous test, and the direction the pipeline
        table reports: scaling an *unconditional* fat-tailed 1-day VaR by
        sqrt(10) lands above the directly estimated 10-day quantile, because
        the 10-day distribution has already lost much of its excess kurtosis.
        (The opposite sign applies to a *conditional* VaR measured in a calm
        state — see docs/METHODOLOGY.md §6.)"""
        cov = np.array([[(0.20 / np.sqrt(252)) ** 2]])
        r = syn.simulate_garch_returns(20_000, cov, seed=88)[:, 0]
        var_1d = historical_var(r, 0.01)
        var_10d_direct = historical_var(overlapping_horizon_pnl(r, 10), 0.01)
        assert scale_var_sqrt_time(var_1d, 10) > var_10d_direct
        # but they remain the same order of magnitude
        assert scale_var_sqrt_time(var_1d, 10) < 1.5 * var_10d_direct

    def test_sqrt_time_understates_from_a_calm_conditional_state(self):
        """The conditional case: after a quiet stretch, today's vol is below
        the long-run level, so variance mean-reverts *upward* and the true
        10-day risk exceeds sqrt(10) x the current 1-day VaR."""
        rng = np.random.default_rng(881)
        calm = rng.normal(0, 1.0, 500)
        # 10-day P&L that begins calm but reverts to a higher long-run vol
        blocks = [
            np.concatenate([rng.normal(0, 1.0, 2), rng.normal(0, 2.5, 8)]).sum()
            for _ in range(20_000)
        ]
        var_1d_calm = historical_var(calm, 0.01)
        assert historical_var(np.array(blocks), 0.01) > scale_var_sqrt_time(
            var_1d_calm, 10
        )

    def test_sqrt_time_is_accurate_for_iid_normal_returns(self):
        """The same scaling is fine when its assumption actually holds."""
        rng = np.random.default_rng(89)
        r = rng.normal(0, 1.0, 60_000)
        var_1d = historical_var(r, 0.05)
        var_10d = historical_var(overlapping_horizon_pnl(r, 10), 0.05)
        assert var_10d == pytest.approx(scale_var_sqrt_time(var_1d, 10), rel=0.05)


# --------------------------------------------------------------------------- #
# estimation error
# --------------------------------------------------------------------------- #
class TestEstimationError:
    def test_var_standard_error_shrinks_with_sample_size(self):
        rng = np.random.default_rng(90)
        ses = [
            var_standard_error_bootstrap(rng.normal(size=n), 0.01, n_boot=300, seed=1)
            for n in (250, 1000, 4000)
        ]
        assert all(b < a for a, b in zip(ses, ses[1:]))

    def test_confidence_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(91)
        p = rng.normal(size=2000)
        lo, hi = var_confidence_interval(p, 0.01, conf=0.95)
        assert lo <= historical_var(p, 0.01) <= hi
        assert lo < hi

    def test_confidence_interval_widens_for_deeper_tails(self):
        rng = np.random.default_rng(92)
        p = rng.normal(size=2000)
        widths = []
        for alpha in (0.10, 0.05, 0.01):
            lo, hi = var_confidence_interval(p, alpha, conf=0.95)
            widths.append(hi - lo)
        assert widths[-1] > widths[0]

    def test_mc_var_converges_to_the_parametric_answer(self):
        """MC with more paths must land closer to the closed form."""
        factors = {"A": RiskFactor("A", "equity", 100.0)}
        pf = Portfolio([EquityPosition(name="a", factor="A", shares=1000.0)], factors)
        cov = np.array([[4e-4]])
        closed = parametric_var(pf.delta_exposures(), cov, 0.01)
        errs = [
            abs(monte_carlo_var(pf, cov, 0.01, n_paths=n, seed=5) - closed)
            for n in (2_000, 40_000, 800_000)
        ]
        assert errs[-1] < errs[0]
        assert errs[-1] < 0.02 * closed


# --------------------------------------------------------------------------- #
# crisis regime
# --------------------------------------------------------------------------- #
def test_crisis_regime_all_families_reprice_upward():
    """A 5x vol regime must lift every VaR family; the conditional ones
    (FHS, age-weighted) must react more than plain historical simulation."""
    rng = np.random.default_rng(101)
    calm = rng.normal(0, 1.0, 900)
    crisis = np.concatenate([calm, rng.normal(0, 5.0, 100)])
    plain_up = historical_var(crisis, 0.01) / historical_var(calm, 0.01)
    fhs_up = filtered_historical_var(crisis, 0.01) / filtered_historical_var(calm, 0.01)
    brw_up = age_weighted_var(crisis, 0.01, lam=0.97) / age_weighted_var(calm, 0.01, lam=0.97)
    assert plain_up > 1.0 and fhs_up > 1.0 and brw_up > 1.0
    assert fhs_up > plain_up
    assert brw_up > plain_up


def test_sample_covariance_is_symmetric_psd():
    r = syn.simulate_returns(500, syn.demo_covariance(), seed=102)
    cov = sample_covariance(r)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-12)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-12)
