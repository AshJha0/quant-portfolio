"""Non-finite inputs must raise, not silently mute the strategy.

Two distinct defect patterns are pinned here, both of which produced a
*plausible-looking* wrong answer rather than an exception.

**Pattern 1 — inequality-only guards.** ``if sigma <= 0: raise``,
``if stop <= entry: raise``, ``if notional <= 0: raise``. Every comparison
against NaN is False, so NaN passed. The consequences are not cosmetic:

* a NaN ``sigma`` in :func:`fx_pairs.zscore` gives an all-NaN z-score, the
  state machine stays flat for the whole sample and the backtest reports a
  clean zero-P&L run — a strategy that never traded, reported as a result;
* a NaN ``stop`` in :func:`fx_pairs.generate_positions` **disables the hard
  stop**, because ``state * z <= -nan`` is always False. That stop is the
  control that exists to survive a regime break (the SNB floor case study in
  docs/VALIDATION.md), and it would have been switched off silently;
* a NaN ``min_abs_corr`` in :func:`fx_pairs.correlation_screen` makes every
  ``abs(rho) >= min_abs_corr`` False and returns an empty candidate table,
  indistinguishable from "the universe contains no pairs".

**Pattern 2 — ``isnan``-only guards on series.** `adf_test`,
`engle_granger`, `_validate_spread` and `run_backtest` all tested
``np.isnan(...)`` and therefore accepted **±Inf**. Inf is the realistic
corruption in this package, not NaN: a zero or missing price becomes
``-inf`` the instant it passes through ``log``. Inf then poisons the OLS
design matrix, `lstsq` returns NaN coefficients, the ADF tau is NaN, and
``bool(nan < critical_value)`` is False — so `engle_granger` reported
**"not cointegrated"** on data it could not test at all. A false negative
that looks exactly like a genuine one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fx_pairs as fp
from fx_pairs.data import synthetic as syn

NON_FINITE = [float("nan"), float("inf"), float("-inf")]
INFS = [float("inf"), float("-inf")]


@pytest.fixture(scope="module")
def pair():
    p1, p2, truth = syn.make_cointegrated_pair(n=800, beta=1.0, kappa=20.0,
                                               sigma_ou=0.05, seed=5)
    return p1, p2, truth


@pytest.fixture(scope="module")
def spread(pair):
    p1, p2, truth = pair
    return fp.log_spread(p1, p2, beta=1.0)


class TestInfinityInSeriesValidators:
    """`isnan` guards accepted Inf; a logged zero price is exactly Inf."""

    @pytest.mark.parametrize("bad", INFS)
    def test_adf_rejects_infinity(self, spread, bad) -> None:
        y = spread.to_numpy().copy()
        y[100] = bad
        with pytest.raises(ValueError, match="infinite"):
            fp.adf_test(y)

    @pytest.mark.parametrize("bad", INFS)
    def test_engle_granger_rejects_infinity(self, pair, bad) -> None:
        p1, p2, _ = pair
        y = np.log(p1.to_numpy()).copy()
        x = np.log(p2.to_numpy())
        y[50] = bad
        with pytest.raises(ValueError, match="infinite"):
            fp.engle_granger(y, x)

    @pytest.mark.parametrize("bad", INFS)
    def test_ou_fitters_reject_infinity(self, spread, bad) -> None:
        s = spread.to_numpy().copy()
        s[10] = bad
        for fitter in (fp.fit_ou_ols, fp.fit_ou_mle):
            with pytest.raises(ValueError, match="infinite"):
                fitter(s)

    def test_a_zero_price_is_rejected_rather_than_logged_to_minus_inf(
            self, pair) -> None:
        # This is how Inf actually arrives: a missing/zero fixing.
        p1, p2, _ = pair
        bad = p1.copy()
        bad.iloc[7] = 0.0
        with pytest.raises(ValueError, match="positive"):
            fp.log_spread(bad, p2, beta=1.0)

    def test_engle_granger_still_detects_the_planted_relation(
            self, pair) -> None:
        # Guard against over-rejection: the clean pair is still cointegrated.
        p1, p2, _ = pair
        eg = fp.engle_granger(np.log(p1.to_numpy()), np.log(p2.to_numpy()))
        assert eg.cointegrated is True
        assert eg.degenerate is False
        assert np.isfinite(eg.stat)
        assert eg.beta == pytest.approx(1.0, abs=0.1)


class TestSignalThresholdGuards:
    """The hard stop must never be disabled by an unnoticed NaN."""

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_stop_rejected(self, bad) -> None:
        z = pd.Series(np.linspace(-3.0, 3.0, 200))
        with pytest.raises(ValueError, match="stop"):
            fp.generate_positions(z, entry=2.0, exit_=0.5, stop=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize("field", ["entry", "exit_"])
    def test_non_finite_thresholds_rejected(self, field, bad) -> None:
        z = pd.Series(np.linspace(-3.0, 3.0, 200))
        kwargs = {"entry": 2.0, "exit_": 0.5, "stop": 4.0}
        kwargs[field] = bad
        with pytest.raises(ValueError, match=field.rstrip("_")):
            fp.generate_positions(z, **kwargs)

    def test_a_finite_stop_still_fires_on_a_regime_break(self) -> None:
        # z walks out to -6: a long entered at -2 must be stopped, not held.
        z = pd.Series(np.concatenate([np.zeros(5),
                                      np.linspace(-2.0, -6.0, 20)]))
        pos, trades = fp.generate_positions(z, entry=2.0, exit_=0.5, stop=4.0)
        assert any(t.exit_reason == "stop" for t in trades)
        # And the position is flat after the stop.
        stop_trade = next(t for t in trades if t.exit_reason == "stop")
        assert pos[stop_trade.exit] == 0.0

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_zscore_rejects_non_finite_frozen_sigma(self, spread, bad) -> None:
        with pytest.raises(ValueError, match="sigma"):
            fp.zscore(spread, mu=0.0, sigma=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_zscore_rejects_non_finite_frozen_mu(self, spread, bad) -> None:
        with pytest.raises(ValueError, match="mu"):
            fp.zscore(spread, mu=bad, sigma=0.05)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_carry_veto_rejects_non_finite_sigma_spread(self, bad) -> None:
        z = pd.Series(np.linspace(-3.0, 3.0, 100))
        with pytest.raises(ValueError, match="sigma_spread"):
            fp.carry_entry_veto(z, sigma_spread=bad, carry_per_day=1e-5,
                                half_life=10.0)

    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize(
        "field", ["target_vol", "ann_factor", "max_leverage"])
    def test_vol_target_rejects_non_finite_parameters(self, spread, field,
                                                      bad) -> None:
        kwargs = {"target_vol": 0.10, "ann_factor": 252.0,
                  "max_leverage": 10.0}
        kwargs[field] = bad
        with pytest.raises(ValueError, match=field):
            fp.vol_target_scale(spread, lookback=63, **kwargs)

    def test_vol_target_cap_still_binds_on_a_quiet_spread(self) -> None:
        # A near-pegged spread has tiny realised vol; without the cap the
        # scale would explode right before a break (the SNB lesson).
        quiet = pd.Series(np.zeros(200) + 1e-9 * np.arange(200))
        scale = fp.vol_target_scale(quiet, target_vol=0.10, lookback=63,
                                    max_leverage=10.0)
        assert scale.dropna().max() <= 10.0 + 1e-12


class TestOuAndHedgeGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_log_spread_rejects_non_finite_beta(self, pair, bad) -> None:
        p1, p2, _ = pair
        with pytest.raises(ValueError, match="beta"):
            fp.log_spread(p1, p2, beta=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_log_spread_rejects_non_finite_alpha(self, pair, bad) -> None:
        p1, p2, _ = pair
        with pytest.raises(ValueError, match="alpha"):
            fp.log_spread(p1, p2, beta=1.0, alpha=bad)

    def test_half_life_rejects_nan_kappa(self) -> None:
        with pytest.raises(ValueError, match="kappa"):
            fp.half_life_days(float("nan"))

    @pytest.mark.parametrize("bad", NON_FINITE + [0.0, -1.0])
    def test_half_life_rejects_bad_dt(self, bad) -> None:
        with pytest.raises(ValueError, match="dt"):
            fp.half_life_days(5.0, dt=bad)

    def test_half_life_semantics_unchanged_for_valid_inputs(self) -> None:
        # kappa <= 0 still means "no mean reversion" -> infinite half-life.
        assert fp.half_life_days(0.0) == float("inf")
        assert fp.half_life_days(-1.0) == float("inf")
        # ln2 / (kappa * dt) with the default business-daily step.
        assert fp.half_life_days(20.0) == pytest.approx(
            np.log(2.0) / (20.0 / 252.0), rel=1e-12)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_rls_rejects_non_finite_observation(self, bad) -> None:
        rls = fp.RLSHedge(lam=0.99)
        rls.update(0.1, 0.2)
        with pytest.raises(ValueError, match="log_p1|log_p2"):
            rls.update(bad, 0.2)

    def test_rls_state_survives_a_rejected_update(self) -> None:
        # The guard must reject *before* mutating theta/P, otherwise the
        # filter is permanently poisoned by one bad tick.
        rls = fp.RLSHedge(lam=0.99)
        for _ in range(50):
            rls.update(0.1, 0.2)
        before = (rls.alpha, rls.beta, rls.n_obs)
        with pytest.raises(ValueError):
            rls.update(float("nan"), 0.2)
        assert (rls.alpha, rls.beta, rls.n_obs) == before
        assert np.isfinite(rls.alpha) and np.isfinite(rls.beta)

    @pytest.mark.parametrize("bad", NON_FINITE + [0.0, -1.0])
    def test_rls_rejects_bad_delta(self, bad) -> None:
        with pytest.raises(ValueError, match="delta"):
            fp.RLSHedge(lam=0.99, delta=bad)

    def test_rls_converges_to_batch_ols_at_lam_one(self, pair) -> None:
        p1, p2, _ = pair
        rls = fp.RLSHedge(lam=1.0, delta=1e6)
        path = rls.fit_path(p2, p1)
        eg = fp.engle_granger(np.log(p1.to_numpy()), np.log(p2.to_numpy()))
        assert path["beta"].iloc[-1] == pytest.approx(eg.beta, abs=1e-6)


class TestBacktestGuards:
    @pytest.fixture
    def simple(self):
        idx = pd.bdate_range("2021-01-04", periods=40)
        rng = np.random.default_rng(0)
        p1 = pd.Series(1.20 * np.exp(np.cumsum(rng.standard_normal(40) * 2e-3)),
                       index=idx)
        p2 = pd.Series(0.75 * np.exp(np.cumsum(rng.standard_normal(40) * 2e-3)),
                       index=idx)
        pos = pd.Series(np.tile([0.0, 1.0, 1.0, -1.0], 10), index=idx)
        return p1, p2, pos

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_position_rejected(self, simple, bad) -> None:
        p1, p2, pos = simple
        pos = pos.copy()
        pos.iloc[5] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            fp.run_backtest(p1, p2, pos, beta=0.8)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_beta_rejected(self, simple, bad) -> None:
        p1, p2, pos = simple
        with pytest.raises(ValueError, match="beta"):
            fp.run_backtest(p1, p2, pos, beta=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize(
        "field", ["pip_spread_1", "pip_spread_2", "notional", "basis"])
    def test_non_finite_engine_parameters_rejected(self, simple, field,
                                                   bad) -> None:
        p1, p2, pos = simple
        kwargs = {"pip_spread_1": 1.0, "pip_spread_2": 1.0, "notional": 1e6,
                  "basis": 365.0}
        kwargs[field] = bad
        with pytest.raises(ValueError, match=field):
            fp.run_backtest(p1, p2, pos, beta=0.8, **kwargs)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_deposit_rate_rejected(self, simple, bad) -> None:
        p1, p2, pos = simple
        rates = {"rb1": 0.05, "rq1": 0.01, "rb2": 0.02, "rq2": bad}
        with pytest.raises(ValueError, match="rq2"):
            fp.run_backtest(p1, p2, pos, beta=0.8, rates=rates)

    def test_non_positive_price_rejected(self, simple) -> None:
        p1, p2, pos = simple
        p1 = p1.copy()
        p1.iloc[3] = 0.0
        with pytest.raises(ValueError, match="positive"):
            fp.run_backtest(p1, p2, pos, beta=0.8)

    def test_clean_run_still_decomposes_exactly(self, simple) -> None:
        p1, p2, pos = simple
        rates = {"rb1": 0.05, "rq1": 0.01, "rb2": 0.02, "rq2": 0.01}
        res = fp.run_backtest(p1, p2, pos, beta=0.8, pip_spread_1=1.0,
                              pip_spread_2=1.0, rates=rates, notional=1e6)
        dec = res.decomposition()
        assert dec["total"] == pytest.approx(
            dec["spot"] + dec["carry"] + dec["costs"], rel=1e-12)
        assert np.isfinite(res.total_pnl.to_numpy()).all()
        assert dec["costs"] <= 0.0


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    """Two highly correlated dollar pairs plus an uncorrelated third."""
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2021-01-04", periods=300)
    common = np.cumsum(rng.standard_normal(300) * 3e-3)
    return pd.DataFrame({
        "EURUSD": 1.10 * np.exp(common + np.cumsum(
            rng.standard_normal(300) * 1e-3)),
        "GBPUSD": 1.27 * np.exp(common + np.cumsum(
            rng.standard_normal(300) * 1e-3)),
        "USDJPY": 130.0 * np.exp(np.cumsum(
            rng.standard_normal(300) * 4e-3)),
    }, index=idx)


class TestScreenAndMetricGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_screen_rejects_non_finite_threshold(self, prices, bad) -> None:
        with pytest.raises(ValueError, match="min_abs_corr"):
            fp.correlation_screen(prices, min_abs_corr=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_screen_rejects_non_finite_vol_tol(self, prices, bad) -> None:
        with pytest.raises(ValueError, match="vol_tol"):
            fp.correlation_screen(prices, vol_tol=bad)

    def test_screen_still_finds_the_correlated_block(self, prices) -> None:
        out = fp.correlation_screen(prices, min_abs_corr=0.5)
        assert len(out) >= 1
        top = out.iloc[0]
        assert {top["pair_1"], top["pair_2"]} == {"EURUSD", "GBPUSD"}
        assert abs(top["corr"]) >= 0.5

    @pytest.mark.parametrize("bad", NON_FINITE + [0.0, -252.0])
    def test_metrics_reject_bad_annualisation_factor(self, bad) -> None:
        from fx_pairs.metrics import (
            sharpe_ratio,
            sharpe_se_lo,
            sortino_ratio,
            turnover,
        )

        r = np.random.default_rng(4).standard_normal(300) * 1e-3
        for fn in (sharpe_ratio, sharpe_se_lo, sortino_ratio):
            with pytest.raises(ValueError, match="ann_factor"):
                fn(r, ann_factor=bad)
        with pytest.raises(ValueError, match="ann_factor"):
            turnover(np.zeros(10), ann_factor=bad)

    def test_turnover_rejects_non_finite_positions(self) -> None:
        from fx_pairs.metrics import turnover

        with pytest.raises(ValueError, match="NaN or infinite"):
            turnover(np.array([0.0, 1.0, float("nan"), 0.0]))

    def test_metrics_still_tolerate_warmup_nans_in_pnl(self) -> None:
        # Deliberate asymmetry: P&L series legitimately start with NaN
        # warmup, so the metric layer drops them rather than raising.
        from fx_pairs.metrics import max_drawdown, sharpe_ratio

        r = np.concatenate([np.full(20, np.nan),
                            np.random.default_rng(6).standard_normal(200) * 1e-3])
        assert np.isfinite(sharpe_ratio(r))
        assert np.isfinite(max_drawdown(r))
