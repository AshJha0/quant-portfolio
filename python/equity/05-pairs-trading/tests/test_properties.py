"""Property-based invariants across the pairs stack.

These constrain the *structure* of the library rather than point values, so
they hold for any correct implementation and catch sign flips, scale bugs and
look-ahead leaks that point checks miss:

* metric invariances — Sharpe/Sortino scale-invariant, drawdown homogeneous;
* OU/spread algebra — affine equivariance, OLS/MLE agreement, half-life
  recovery, kappa monotonicity;
* cointegration — size and power, MacKinnon critical-value ordering;
* signal state machine — sign symmetry, threshold monotonicity, arming;
* backtest identities — cost decomposition, P&L homogeneity in gross,
  direction antisymmetry, and the no-look-ahead guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq_pairs.backtest import ZERO_COSTS, CostModel, backtest_pair
from eq_pairs.cointegration import adf_test, engle_granger, hedge_ratio, mackinnon_crit
from eq_pairs.data import business_index
from eq_pairs.data import synthetic as syn
from eq_pairs.metrics import (
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    sharpe_se,
    sortino_ratio,
)
from eq_pairs.signals import SignalRules, generate_signals, time_stop_bars, size_positions
from eq_pairs.spread import compute_spread, fit_ou_ols, fit_ou_mle, half_life_from_kappa


@pytest.fixture(scope="module")
def returns():
    """Deterministic daily return stream for the metric invariance checks."""
    return np.random.default_rng(11).standard_normal(750) * 0.01 + 0.0003


# --------------------------------------------------------------------------- #
# metric invariances
# --------------------------------------------------------------------------- #
class TestMetricInvariances:
    @pytest.mark.parametrize("c", [0.5, 2.0, 100.0])
    def test_sharpe_is_scale_invariant(self, returns, c):
        """Sharpe is a ratio of first to second moment: leverage cancels."""
        assert sharpe_ratio(c * returns) == pytest.approx(sharpe_ratio(returns), rel=1e-12)

    @pytest.mark.parametrize("c", [0.5, 2.0, 100.0])
    def test_sortino_is_scale_invariant(self, returns, c):
        assert sortino_ratio(c * returns) == pytest.approx(
            sortino_ratio(returns), rel=1e-12
        )

    def test_sharpe_flips_sign_with_the_returns(self, returns):
        assert sharpe_ratio(-returns) == pytest.approx(-sharpe_ratio(returns), rel=1e-12)

    def test_sharpe_annualisation_scales_as_sqrt_periods(self, returns):
        daily = sharpe_ratio(returns, periods_per_year=252)
        weekly = sharpe_ratio(returns, periods_per_year=52)
        assert daily / weekly == pytest.approx(np.sqrt(252 / 52), rel=1e-12)

    @pytest.mark.parametrize("c", [0.5, 3.0])
    def test_max_drawdown_is_homogeneous_of_degree_one(self, c):
        eq = np.cumsum(np.random.default_rng(12).standard_normal(500)) + 100.0
        assert max_drawdown(c * eq) == pytest.approx(c * max_drawdown(eq), rel=1e-12)

    def test_max_drawdown_is_never_negative_zero(self):
        """A monotone-rising curve must report exactly +0.0, not -0.0 (which
        prints as a negative drawdown in a risk report)."""
        v = max_drawdown(np.linspace(1.0, 2.0, 50))
        assert v == 0.0
        assert not np.signbit(v)

    def test_drawdown_series_is_nonpositive_and_zero_at_new_peaks(self):
        eq = np.array([100.0, 105.0, 102.0, 110.0, 108.0])
        dd = drawdown_series(eq)
        assert np.all(dd <= 0.0)
        assert dd[0] == 0.0 and dd[1] == 0.0 and dd[3] == 0.0
        assert dd[2] == pytest.approx(-3.0)

    def test_max_drawdown_shift_invariant(self):
        """Adding a constant to the equity curve does not change the depth."""
        eq = np.cumsum(np.random.default_rng(13).standard_normal(300)) + 100.0
        assert max_drawdown(eq + 500.0) == pytest.approx(max_drawdown(eq), rel=1e-12)

    def test_constant_streams_give_nan_not_infinity(self):
        for value in (0.0, 0.001, -0.002):
            r = np.full(200, value)
            assert np.isnan(sharpe_ratio(r))
            assert np.isnan(sharpe_se(r))

    def test_lo_adjusted_se_exceeds_iid_under_positive_autocorrelation(self):
        """Mean-reversion books have autocorrelated P&L; the iid SE then
        understates uncertainty, which is precisely when it matters."""
        rng = np.random.default_rng(14)
        e = rng.standard_normal(1500)
        r = np.zeros(1500)
        for t in range(1, 1500):
            r[t] = 0.6 * r[t - 1] + e[t]  # strongly positively autocorrelated
        assert sharpe_se(r, lo_adjust=True) > sharpe_se(r, lo_adjust=False)

    def test_lo_adjusted_se_below_iid_under_negative_autocorrelation(self):
        alt = np.where(np.arange(400) % 2 == 0, 1.0, -1.0) * 0.01 + 1e-4
        assert sharpe_se(alt, lo_adjust=True) < sharpe_se(alt, lo_adjust=False)

    def test_metrics_reject_non_finite_input(self):
        r = np.random.default_rng(15).standard_normal(100)
        r[10] = np.nan
        for fn in (sharpe_ratio, sortino_ratio):
            with pytest.raises(ValueError, match="NaN"):
                fn(r)


# --------------------------------------------------------------------------- #
# spread / OU algebra
# --------------------------------------------------------------------------- #
class TestSpreadAlgebra:
    def test_compute_spread_is_linear_in_beta_and_alpha(self):
        rng = np.random.default_rng(21)
        y, x = rng.standard_normal(100) + 100, rng.standard_normal(100) + 50
        s1 = compute_spread(y, x, beta=1.5, alpha=2.0)
        np.testing.assert_allclose(s1, y - 1.5 * x - 2.0, rtol=1e-12)
        # shifting alpha shifts the spread by exactly that amount
        s2 = compute_spread(y, x, beta=1.5, alpha=5.0)
        np.testing.assert_allclose(s1 - s2, 3.0, rtol=1e-12)

    def test_ou_fit_is_equivariant_under_a_location_shift(self):
        """s -> s + d leaves kappa/sigma alone and shifts mu by exactly d."""
        s = syn.simulate_ou(2000, kappa=0.05, sigma=1.0, mu=0.0, seed=22)
        base = fit_ou_ols(s)
        shifted = fit_ou_ols(s + 7.5)
        assert shifted.kappa == pytest.approx(base.kappa, rel=1e-9)
        assert shifted.sigma == pytest.approx(base.sigma, rel=1e-9)
        assert shifted.mu == pytest.approx(base.mu + 7.5, abs=1e-8)
        assert shifted.half_life == pytest.approx(base.half_life, rel=1e-9)

    @pytest.mark.parametrize("c", [0.5, 4.0])
    def test_ou_fit_is_equivariant_under_a_scale_change(self, c):
        """s -> c*s scales mu and sigma by c and leaves kappa unchanged."""
        s = syn.simulate_ou(2000, kappa=0.05, sigma=1.0, mu=3.0, seed=23)
        base = fit_ou_ols(s)
        scaled = fit_ou_ols(c * s)
        assert scaled.kappa == pytest.approx(base.kappa, rel=1e-9)
        assert scaled.mu == pytest.approx(c * base.mu, rel=1e-8)
        assert scaled.sigma == pytest.approx(c * base.sigma, rel=1e-9)

    def test_ols_and_mle_agree_on_a_long_sample(self):
        s = syn.simulate_ou(4000, kappa=0.04, sigma=1.0, mu=1.0, seed=24)
        a, b = fit_ou_ols(s), fit_ou_mle(s)
        assert b.kappa == pytest.approx(a.kappa, rel=0.05)
        assert b.mu == pytest.approx(a.mu, abs=0.15)
        assert b.sigma == pytest.approx(a.sigma, rel=0.05)

    @pytest.mark.parametrize("kappa", [0.03, 0.08, 0.20])
    def test_half_life_recovered_from_simulated_ou(self, kappa):
        s = syn.simulate_ou(12_000, kappa=kappa, sigma=1.0, mu=0.0, seed=25)
        fit = fit_ou_ols(s)
        assert fit.mean_reverting
        assert fit.kappa == pytest.approx(kappa, rel=0.15)
        assert fit.half_life == pytest.approx(np.log(2) / kappa, rel=0.15)

    def test_ols_kappa_is_biased_downward_and_worse_for_slow_reversion(self):
        """The Dickey-Fuller/Kendall small-sample bias: OLS on the AR(1)
        under-estimates kappa, so estimated half-lives are too LONG. The bias
        grows as reversion slows. Measured at n=12,000:

            true kappa   0.01    0.03    0.08    0.20
            rel. bias   -20.5%  -10.5%  -5.9%   -3.5%

        A desk must know the sign: a time stop set at k x the *estimated*
        half-life is systematically too loose, letting losers run.
        """
        biases = []
        for k in (0.01, 0.03, 0.08, 0.20):
            fit = fit_ou_ols(syn.simulate_ou(12_000, kappa=k, sigma=1.0, seed=25))
            biases.append(fit.kappa / k - 1.0)
        assert all(b < 0 for b in biases)                  # always downward
        assert all(a < b for a, b in zip(biases, biases[1:]))  # worse when slower
        assert biases[0] < -0.10                            # materially so at k=0.01

    def test_faster_mean_reversion_gives_shorter_half_life(self):
        hls = [
            fit_ou_ols(syn.simulate_ou(8000, kappa=k, sigma=1.0, seed=26)).half_life
            for k in (0.01, 0.03, 0.10, 0.30)
        ]
        assert all(b < a for a, b in zip(hls, hls[1:]))

    def test_half_life_formula_and_degenerate_kappa(self):
        assert half_life_from_kappa(np.log(2)) == pytest.approx(1.0, rel=1e-12)
        assert half_life_from_kappa(0.0) == np.inf
        assert half_life_from_kappa(-0.1) == np.inf

    def test_exact_unit_root_is_flagged_not_mean_reverting(self):
        """Only b >= 1 trips the flag; then no half-life is fabricated."""
        from eq_pairs.spread import _ou_from_ar1

        fit = _ou_from_ar1(c=0.0, b=1.0, resid_var=1.0, dt=1.0, method="ols")
        assert not fit.mean_reverting
        assert fit.half_life == np.inf
        assert fit.kappa == 0.0
        assert fit.stationary_std == np.inf

    def test_an_ou_fit_alone_cannot_reject_a_random_walk(self):
        """Critical caveat, and the reason cointegration testing is the gate
        rather than the OU half-life: on a *pure random walk* OLS returns
        b slightly below 1 (finite-sample bias ~ -1.5/n), so the fit reports
        `mean_reverting=True` with a spurious half-life of several hundred
        days. Nothing in the OU fit flags this — `engle_granger` must.
        """
        spurious = []
        for k in range(5):
            rw = np.cumsum(np.random.default_rng(27 + k).standard_normal(3000))
            fit = fit_ou_ols(rw)
            assert fit.mean_reverting        # the flag does NOT save you
            assert fit.b < 1.0
            spurious.append(fit.half_life)
        # the "half-lives" are absurdly long - the tell a desk must screen on
        assert min(spurious) > 100.0
        # and the honest gate does reject them as tradeable
        rw_a = np.cumsum(np.random.default_rng(500).standard_normal(1200))
        rw_b = np.cumsum(np.random.default_rng(501).standard_normal(1200))
        assert not engle_granger(rw_a, rw_b).cointegrated("5%")

    def test_stationary_std_matches_the_simulated_dispersion(self):
        kappa, sigma = 0.05, 1.0
        s = syn.simulate_ou(30_000, kappa=kappa, sigma=sigma, mu=0.0, seed=28)
        fit = fit_ou_ols(s)
        assert fit.stationary_std == pytest.approx(np.std(s, ddof=1), rel=0.10)
        assert fit.stationary_std == pytest.approx(sigma / np.sqrt(2 * kappa), rel=0.15)


# --------------------------------------------------------------------------- #
# cointegration
# --------------------------------------------------------------------------- #
class TestCointegrationProperties:
    def test_mackinnon_values_are_ordered_and_n2_is_stricter(self):
        """The whole point of the N=2 surface: it is more negative than plain
        ADF, so using ADF values for a residual-based test over-rejects."""
        c1 = mackinnon_crit(n_series=1, nobs=500)
        c2 = mackinnon_crit(n_series=2, nobs=500)
        for level in ("1%", "5%", "10%"):
            assert c2[level] < c1[level]
        # stricter levels are more negative
        assert c2["10%"] > c2["5%"] > c2["1%"]
        assert c1["10%"] > c1["5%"] > c1["1%"]
        # the size distortion this prevents is ~0.5 of a t-unit at 5%
        assert c1["5%"] - c2["5%"] > 0.4

    def test_power_rejects_a_genuinely_cointegrated_pair(self):
        prices, truth = syn.cointegrated_pair(n=1500, beta=1.5, kappa=0.05, seed=31)
        res = engle_granger(prices.iloc[:, 0], prices.iloc[:, 1])
        assert res.cointegrated("5%")
        assert res.beta == pytest.approx(truth.beta, rel=0.05)

    def test_size_does_not_reject_independent_random_walks_too_often(self):
        """The trap case: correlated but NOT cointegrated. A correctly sized
        5% test should reject on only a small minority of samples."""
        rejects = 0
        trials = 40
        for k in range(trials):
            prices, _ = syn.correlated_random_walks(n=800, rho=0.9, seed=100 + k)
            res = engle_granger(prices.iloc[:, 0], prices.iloc[:, 1])
            rejects += bool(res.cointegrated("5%"))
        assert rejects <= trials * 0.25  # generous bound, but far from 100%

    def test_hedge_ratio_recovers_the_true_beta(self):
        prices, truth = syn.cointegrated_pair(n=2000, beta=2.25, seed=32)
        beta, alpha, resid = hedge_ratio(
            prices.iloc[:, 0].to_numpy(), prices.iloc[:, 1].to_numpy()
        )
        assert beta == pytest.approx(truth.beta, rel=0.05)
        assert len(resid) == 2000
        # residuals of the cointegrating regression are exactly mean-zero
        assert float(np.mean(resid)) == pytest.approx(0.0, abs=1e-9)

    def test_adf_more_negative_for_a_stationary_series(self):
        ou = syn.simulate_ou(1500, kappa=0.10, sigma=1.0, seed=33)
        rw = np.cumsum(np.random.default_rng(34).standard_normal(1500))
        assert adf_test(ou).stat < adf_test(rw).stat

    def test_engle_granger_is_direction_dependent(self):
        """Regressing y on x and x on y are different tests — a documented
        asymmetry of the two-step procedure (unlike Johansen)."""
        prices, _ = syn.cointegrated_pair(n=1200, beta=1.5, seed=35)
        y, x = prices.iloc[:, 0], prices.iloc[:, 1]
        a = engle_granger(y, x)
        b = engle_granger(x, y)
        assert a.beta != pytest.approx(1.0 / b.beta, rel=1e-6)


# --------------------------------------------------------------------------- #
# signal state machine
# --------------------------------------------------------------------------- #
class TestSignalProperties:
    @staticmethod
    def _z(vals):
        return pd.Series(np.asarray(vals, float), index=business_index(len(vals)))

    def test_positions_are_always_in_the_ternary_set(self):
        rng = np.random.default_rng(41)
        z = self._z(rng.standard_normal(2000) * 2.5)
        pos = generate_signals(z)["position"].to_numpy()
        assert set(np.unique(pos)).issubset({-1, 0, 1})

    def test_sign_symmetry_of_the_state_machine(self):
        """Negating the z-score must negate every position exactly (the rules
        are symmetric in z)."""
        rng = np.random.default_rng(42)
        z = self._z(rng.standard_normal(1500) * 2.5)
        a = generate_signals(z)["position"].to_numpy()
        b = generate_signals(-z)["position"].to_numpy()
        np.testing.assert_array_equal(a, -b)

    def test_higher_entry_threshold_never_trades_more(self):
        rng = np.random.default_rng(43)
        z = self._z(rng.standard_normal(3000) * 2.0)
        counts = []
        for entry in (1.0, 1.5, 2.0, 3.0):
            sig = generate_signals(z, SignalRules(entry_z=entry, stop_z=entry + 3.0))
            counts.append(int((sig["event"].str.startswith("entry")).sum()))
        assert all(b <= a for a, b in zip(counts, counts[1:]))

    def test_no_position_while_z_stays_inside_the_entry_band(self):
        z = self._z(np.linspace(-1.9, 1.9, 200))
        assert (generate_signals(z)["position"] == 0).all()

    def test_position_direction_opposes_the_z_sign(self):
        """z rich (>0) => short the spread; z cheap (<0) => long."""
        assert generate_signals(self._z([0.0, 3.0]))["position"].iloc[-1] == -1
        assert generate_signals(self._z([0.0, -3.0]))["position"].iloc[-1] == 1

    def test_stop_exits_and_disarms_until_z_re_enters_the_band(self):
        """After a stop, an immediately-still-extreme z must NOT re-enter."""
        z = self._z([0.0, 2.5, 4.5, 4.2, 3.0, 1.0, 2.5])
        out = generate_signals(z, SignalRules(entry_z=2.0, stop_z=4.0))
        assert out["event"].iloc[2] == "exit_stop"
        assert list(out["position"].iloc[3:5]) == [0, 0]   # disarmed
        assert out["position"].iloc[6] == -1               # re-armed after |z|<2

    def test_time_stop_closes_the_position(self):
        z = self._z([0.0] + [2.5] * 10)
        out = generate_signals(z, SignalRules(entry_z=2.0, stop_z=6.0, max_holding=3))
        assert "exit_time" in set(out["event"])
        assert out["position"].iloc[-1] == 0

    def test_nan_forces_flat_and_never_propagates(self):
        z = self._z([0.0, 3.0, np.nan, 3.0])
        out = generate_signals(z)
        assert out["position"].iloc[2] == 0
        assert out["event"].iloc[2] == "exit_nan"
        assert out["position"].dtype.kind in "iu"

    def test_time_stop_bars_helper(self):
        assert time_stop_bars(10.0, k=3.0) == 30
        assert time_stop_bars(10.0, k=3.0, cap=20) == 20
        assert time_stop_bars(np.inf) == 252
        assert time_stop_bars(0.0) == 252
        with pytest.raises(ValueError, match="k must be positive"):
            time_stop_bars(10.0, k=0.0)

    def test_signal_rules_validation(self):
        with pytest.raises(ValueError, match="entry_z"):
            SignalRules(entry_z=0.0)
        with pytest.raises(ValueError, match="exit_z"):
            SignalRules(entry_z=2.0, exit_z=2.5)
        with pytest.raises(ValueError, match="stop_z"):
            SignalRules(entry_z=2.0, stop_z=1.5)
        with pytest.raises(ValueError, match="max_holding"):
            SignalRules(max_holding=0)


# --------------------------------------------------------------------------- #
# position sizing
# --------------------------------------------------------------------------- #
class TestSizingProperties:
    def test_dollar_mode_is_dollar_neutral_and_hits_the_gross_target(self):
        qy, qx = size_positions(100.0, 40.0, direction=1, beta=2.0, gross=1e6,
                                mode="dollar")
        assert qy * 100.0 == pytest.approx(-qx * 40.0, rel=1e-12)
        assert abs(qy) * 100.0 + abs(qx) * 40.0 == pytest.approx(1e6, rel=1e-12)

    def test_beta_mode_uses_the_cointegrating_share_ratio(self):
        beta = 2.0
        qy, qx = size_positions(100.0, 40.0, direction=1, beta=beta, gross=1e6,
                                mode="beta")
        assert qx == pytest.approx(-beta * qy, rel=1e-12)

    @pytest.mark.parametrize("mode", ["dollar", "beta"])
    def test_direction_flips_both_legs(self, mode):
        a = size_positions(100.0, 40.0, 1, 2.0, 1e6, mode)
        b = size_positions(100.0, 40.0, -1, 2.0, 1e6, mode)
        assert a[0] == pytest.approx(-b[0], rel=1e-12)
        assert a[1] == pytest.approx(-b[1], rel=1e-12)

    @pytest.mark.parametrize("mode", ["dollar", "beta"])
    def test_sizes_are_linear_in_gross(self, mode):
        base = size_positions(100.0, 40.0, 1, 2.0, 1e6, mode)
        big = size_positions(100.0, 40.0, 1, 2.0, 5e6, mode)
        assert big[0] == pytest.approx(5 * base[0], rel=1e-12)
        assert big[1] == pytest.approx(5 * base[1], rel=1e-12)

    def test_flat_direction_is_exactly_zero(self):
        assert size_positions(100.0, 40.0, 0, 2.0) == (0.0, 0.0)

    def test_sizing_validation(self):
        with pytest.raises(ValueError, match="direction"):
            size_positions(100.0, 40.0, 2, 2.0)
        with pytest.raises(ValueError, match="prices"):
            size_positions(-1.0, 40.0, 1, 2.0)
        with pytest.raises(ValueError, match="gross"):
            size_positions(100.0, 40.0, 1, 2.0, gross=0.0)
        with pytest.raises(ValueError, match="beta"):
            size_positions(100.0, 40.0, 1, -2.0, mode="beta")
        with pytest.raises(ValueError, match="mode"):
            size_positions(100.0, 40.0, 1, 2.0, mode="vol")


# --------------------------------------------------------------------------- #
# backtest identities
# --------------------------------------------------------------------------- #
class TestBacktestIdentities:
    @staticmethod
    def _pair(n=300, seed=51):
        prices, _ = syn.cointegrated_pair(n=n, beta=1.5, kappa=0.06, seed=seed)
        y, x = prices.iloc[:, 0], prices.iloc[:, 1]
        s = compute_spread(y, x, beta=1.5)
        z = (s - s.mean()) / s.std(ddof=1)
        target = pd.Series(
            np.select([z > 1.5, z < -1.5], [-1, 1], 0), index=y.index
        )
        return y, x, target

    def test_net_equals_gross_minus_every_cost_component(self):
        y, x, target = self._pair()
        res = backtest_pair(y, x, target, beta=1.5,
                            costs=CostModel(5.0, 2.0, 50.0))
        d = res.daily
        np.testing.assert_allclose(
            d["net_pnl"].to_numpy(),
            (d["gross_pnl"] - d["commission"] - d["slippage"] - d["borrow"]).to_numpy(),
            atol=1e-9,
        )

    def test_zero_costs_make_net_equal_gross(self):
        y, x, target = self._pair()
        res = backtest_pair(y, x, target, beta=1.5, costs=ZERO_COSTS)
        assert res.net_pnl == pytest.approx(res.gross_pnl, abs=1e-9)
        assert res.total_costs == pytest.approx(0.0, abs=1e-12)

    def test_costs_are_monotone_in_the_cost_parameters(self):
        y, x, target = self._pair()
        nets = []
        for bps in (0.0, 2.0, 5.0, 20.0):
            res = backtest_pair(y, x, target, beta=1.5,
                                costs=CostModel(bps, 0.0, 0.0))
            nets.append(res.net_pnl)
        assert all(b <= a + 1e-9 for a, b in zip(nets, nets[1:]))

    def test_gross_pnl_is_homogeneous_in_the_gross_target(self):
        """Doubling the book size doubles every P&L line exactly."""
        y, x, target = self._pair()
        a = backtest_pair(y, x, target, beta=1.5, gross=1e6, costs=ZERO_COSTS)
        b = backtest_pair(y, x, target, beta=1.5, gross=2e6, costs=ZERO_COSTS)
        assert b.gross_pnl == pytest.approx(2.0 * a.gross_pnl, rel=1e-9)

    def test_flipping_every_signal_flips_the_gross_pnl(self):
        """With no costs the strategy is antisymmetric in the target."""
        y, x, target = self._pair()
        a = backtest_pair(y, x, target, beta=1.5, costs=ZERO_COSTS)
        b = backtest_pair(y, x, -target, beta=1.5, costs=ZERO_COSTS)
        assert b.gross_pnl == pytest.approx(-a.gross_pnl, abs=1e-6)

    def test_trade_pnls_reconcile_with_the_daily_net(self):
        y, x, target = self._pair()
        res = backtest_pair(y, x, target, beta=1.5, costs=CostModel(3.0, 1.0, 20.0))
        assert float(res.trades["pnl"].sum()) == pytest.approx(res.net_pnl, abs=1e-6)

    def test_no_lookahead_shifting_the_target_later_changes_nothing_before(self):
        """The engine may only use target_{t-1} at t: appending future signal
        values must not alter any earlier day's P&L."""
        y, x, target = self._pair()
        full = backtest_pair(y, x, target, beta=1.5, costs=ZERO_COSTS)
        cut = 200
        trunc = backtest_pair(
            y.iloc[:cut], x.iloc[:cut], target.iloc[:cut], beta=1.5, costs=ZERO_COSTS
        )
        # compare the overlapping region, excluding the forced close-out bar
        np.testing.assert_allclose(
            full.daily["gross_pnl"].to_numpy()[: cut - 1],
            trunc.daily["gross_pnl"].to_numpy()[: cut - 1],
            atol=1e-9,
        )

    def test_first_bar_never_carries_a_position(self):
        y, x, target = self._pair()
        target.iloc[0] = 1  # even if the signal says trade immediately
        res = backtest_pair(y, x, target, beta=1.5, costs=ZERO_COSTS)
        assert res.daily["position"].iloc[0] == 0
        assert res.daily["gross_pnl"].iloc[0] == 0.0

    def test_borrow_is_nonnegative_and_zero_without_a_short(self):
        y, x, target = self._pair()
        flat = pd.Series(np.zeros(len(y)), index=y.index)
        res = backtest_pair(y, x, flat, beta=1.5, costs=CostModel(0.0, 0.0, 200.0))
        assert float(res.daily["borrow"].sum()) == pytest.approx(0.0, abs=1e-12)
        live = backtest_pair(y, x, target, beta=1.5, costs=CostModel(0.0, 0.0, 200.0))
        assert np.all(live.daily["borrow"].to_numpy() >= -1e-12)
        assert float(live.daily["borrow"].sum()) > 0.0

    def test_exposures_are_consistent_with_the_position(self):
        y, x, target = self._pair()
        res = backtest_pair(y, x, target, beta=1.5, costs=ZERO_COSTS)
        d = res.daily
        flat_days = d["position"] == 0
        np.testing.assert_allclose(d.loc[flat_days, "gross_exposure"], 0.0, atol=1e-9)
        assert np.all(d["gross_exposure"].to_numpy() >= -1e-9)
