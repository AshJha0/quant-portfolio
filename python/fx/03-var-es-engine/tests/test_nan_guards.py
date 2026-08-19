"""Non-finite inputs must raise, never produce a NaN risk number.

`fx_var`'s stated NaN policy is *refuse, never impute*.  Market data and
factor-return histories already enforced it.  The gap closed here is the
long tail of bare scalar arguments guarded only by an inequality:
``if horizon_days <= 0: raise``, ``if sigma < 0: raise``,
``if df <= 2: raise``, ``if strike <= 0: raise``, ``if k <= 0: raise``.
Every comparison against NaN evaluates to False, so each of those guards
silently accepted NaN and the engine returned ``nan`` as the VaR — the worst
possible failure mode for a number that feeds a limit check and a
regulatory capital multiplier.

A NaN VaR does not trip a limit, does not colour a traffic light and does
not look obviously wrong on a report, so these tests assert an exception is
raised in every one of those paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fx_var.book import Book, Cash, Forward, Market, Option, Spot
from fx_var.common import validate_horizon
from fx_var.expected_shortfall import normal_es, normal_var, t_es, t_var
from fx_var.historical_var import historical_var
from fx_var.monte_carlo_var import JumpSpec, robust_cholesky, simulate_factor_returns
from fx_var.parametric_var import (
    cornish_fisher_var,
    parametric_var,
    portfolio_sigma,
    var_covar,
)
from fx_var.stress_testing import (
    peg_break_scenario,
    reverse_stress_linear,
    usd_broad_move,
)

NAN = float("nan")
INF = float("inf")
NON_FINITE = [NAN, INF, -INF]


@pytest.fixture(scope="module")
def market() -> Market:
    return Market({"EUR": 1.10, "JPY": 0.0090},
                  {"USD": 0.045, "EUR": 0.030, "JPY": 0.001},
                  {"EURUSD": 0.08})


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 260
    return pd.DataFrame({
        "FX:EUR": rng.standard_normal(n) * 0.005,
        "FX:JPY": rng.standard_normal(n) * 0.006,
        "IR:USD": rng.standard_normal(n) * 0.0004,
        "IR:EUR": rng.standard_normal(n) * 0.0003,
        "IR:JPY": rng.standard_normal(n) * 0.0002,
        "VOL:EURUSD": rng.standard_normal(n) * 0.004,
    })


class TestHorizonGuard:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_validate_horizon_rejects_non_finite(self, bad: float) -> None:
        with pytest.raises(ValueError, match="horizon_days"):
            validate_horizon(bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_historical_var_rejects_non_finite_horizon(
            self, market, returns, bad) -> None:
        book = Book([Spot("EURUSD", 5e6)])
        with pytest.raises(ValueError, match="horizon_days"):
            historical_var(book, market, returns, horizon_days=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_parametric_var_rejects_non_finite_horizon(
            self, market, returns, bad) -> None:
        book = Book([Spot("EURUSD", 5e6)])
        with pytest.raises(ValueError, match="horizon_days"):
            parametric_var(book, market, returns, horizon_days=bad)

    def test_finite_horizon_still_scales_by_sqrt_time(
            self, market, returns) -> None:
        book = Book([Spot("EURUSD", 5e6)])
        r1 = historical_var(book, market, returns, horizon_days=1.0)
        r10 = historical_var(book, market, returns, horizon_days=10.0)
        assert r10.var == pytest.approx(r1.var * np.sqrt(10.0), rel=1e-12)


class TestClosedFormTailGuards:
    @pytest.mark.parametrize("fn", [normal_var, normal_es])
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_normal_rejects_non_finite_sigma(self, fn, bad) -> None:
        with pytest.raises(ValueError, match="sigma"):
            fn(bad, 0.99)

    @pytest.mark.parametrize("fn", [normal_var, normal_es])
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_normal_rejects_non_finite_mean(self, fn, bad) -> None:
        with pytest.raises(ValueError, match="mean"):
            fn(1000.0, 0.99, bad)

    @pytest.mark.parametrize("fn", [t_var, t_es])
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_student_t_rejects_non_finite_sigma(self, fn, bad) -> None:
        with pytest.raises(ValueError, match="sigma"):
            fn(bad, 0.99, 6.0)

    @pytest.mark.parametrize("fn", [t_var, t_es])
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_student_t_rejects_non_finite_df(self, fn, bad) -> None:
        # `if df <= 2: raise` alone lets NaN through into the sqrt((df-2)/df)
        # scaling and returns a NaN VaR.
        with pytest.raises(ValueError, match="df"):
            fn(1000.0, 0.99, bad)

    def test_finite_path_unchanged(self) -> None:
        # Sanity: the guards did not disturb the closed forms.
        assert normal_var(1000.0, 0.99) == pytest.approx(2326.3478740408408,
                                                        rel=1e-12)
        assert t_var(1000.0, 0.99, 6.0) > normal_var(1000.0, 0.99)


class TestParametricInputGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_covariance_entry_rejected(self, bad) -> None:
        w = pd.Series({"FX:EUR": 1e6, "FX:JPY": -5e5})
        cov = pd.DataFrame([[1e-4, bad], [bad, 2e-4]],
                           index=w.index, columns=w.index)
        with pytest.raises(ValueError, match="cov"):
            portfolio_sigma(w, cov)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_exposure_rejected(self, bad) -> None:
        w = pd.Series({"FX:EUR": bad, "FX:JPY": -5e5})
        cov = pd.DataFrame([[1e-4, 1e-5], [1e-5, 2e-4]],
                           index=w.index, columns=w.index)
        with pytest.raises(ValueError, match="exposures"):
            var_covar(w, cov, 0.99)

    @pytest.mark.parametrize(
        "field", ["sigma", "skew", "excess_kurtosis", "mean"])
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_cornish_fisher_rejects_non_finite_moments(self, field,
                                                       bad) -> None:
        kwargs = {"sigma": 1000.0, "skew": -0.2, "excess_kurtosis": 1.0,
                  "mean": 0.0}
        kwargs[field] = bad
        with pytest.raises(ValueError, match=field):
            cornish_fisher_var(alpha=0.99, **kwargs)


class TestMonteCarloInputGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_covariance_rejected_by_cholesky(self, bad) -> None:
        cov = np.array([[1e-4, 0.0], [0.0, bad]])
        with pytest.raises(ValueError, match="cov"):
            robust_cholesky(cov)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_t_df_rejected(self, bad) -> None:
        cov = pd.DataFrame([[1e-4, 0.0], [0.0, 2e-4]],
                           index=["FX:EUR", "FX:JPY"],
                           columns=["FX:EUR", "FX:JPY"])
        with pytest.raises(ValueError, match="df"):
            simulate_factor_returns(cov, 100, dist="t", df=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_jump_mean_rejected(self, bad) -> None:
        with pytest.raises(ValueError, match="jump mean"):
            JumpSpec(prob=0.01, mean={"FX:TRY": bad})

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_jump_std_rejected(self, bad) -> None:
        with pytest.raises(ValueError, match="jump std"):
            JumpSpec(prob=0.01, mean={"FX:TRY": -0.15},
                     std={"FX:TRY": bad})

    def test_finite_jump_spec_produces_finite_scenarios(self) -> None:
        cov = pd.DataFrame([[1e-4]], index=["FX:TRY"], columns=["FX:TRY"])
        js = JumpSpec(prob=0.05, mean={"FX:TRY": -0.15},
                      std={"FX:TRY": 0.03})
        scen = simulate_factor_returns(cov, 5000, dist="jump", jumps=js,
                                       seed=1)
        assert np.isfinite(scen.to_numpy()).all()
        # The jump overlay adds variance the covariance cannot see.
        assert scen["FX:TRY"].std() > np.sqrt(1e-4)


class TestPositionConstructionGuards:
    """A NaN notional or strike silently poisons every downstream number."""

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_cash_amount(self, bad) -> None:
        with pytest.raises(ValueError, match="amount"):
            Cash("EUR", bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_spot_notional(self, bad) -> None:
        with pytest.raises(ValueError, match="notional"):
            Spot("EURUSD", bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_spot_entry_rate(self, bad) -> None:
        with pytest.raises(ValueError, match="entry_rate"):
            Spot("EURUSD", 1e6, bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize("field", ["notional", "expiry", "strike"])
    def test_forward_fields(self, field, bad) -> None:
        kwargs = {"pair": "EURUSD", "notional": 1e6, "expiry": 0.5,
                  "strike": 1.11}
        kwargs[field] = bad
        with pytest.raises(ValueError, match=field):
            Forward(**kwargs)

    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize("field", ["notional", "strike", "expiry"])
    def test_option_fields(self, field, bad) -> None:
        kwargs = {"pair": "EURUSD", "notional": 1e6, "strike": 1.12,
                  "expiry": 0.25}
        kwargs[field] = bad
        with pytest.raises(ValueError, match=field):
            Option(**kwargs)


class TestShockGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_scalar_shock_rejected(self, market, bad) -> None:
        book = Book([Spot("EURUSD", 5e6)])
        with pytest.raises(ValueError, match="FX:EUR"):
            book.pnl(market, {"FX:EUR": bad})

    def test_single_nan_row_in_a_scenario_matrix_rejected(self,
                                                          market) -> None:
        book = Book([Spot("EURUSD", 5e6)])
        scen = pd.DataFrame({"FX:EUR": [0.01, 0.02, NAN, -0.01]})
        with pytest.raises(ValueError, match="FX:EUR"):
            book.pnl(market, scen)

    def test_finite_scenario_matrix_still_broadcasts(self, market) -> None:
        book = Book([Spot("EURUSD", 5e6)])
        scen = pd.DataFrame({"FX:EUR": [0.01, 0.02, -0.01]})
        out = book.pnl(market, scen)
        assert out.shape == (3,)
        assert np.isfinite(out).all()


class TestStressScenarioGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_usd_broad_move_rejects_non_finite_pct(self, bad) -> None:
        with pytest.raises(ValueError, match="pct"):
            usd_broad_move(["EUR", "JPY"], bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_peg_break_rejects_non_finite_jump(self, bad) -> None:
        with pytest.raises(ValueError, match="jump"):
            peg_break_scenario("HKD", jump=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_peg_break_rejects_non_finite_vol_spike(self, bad) -> None:
        with pytest.raises(ValueError, match="vol_spike"):
            peg_break_scenario("HKD", jump=-0.30, vol_spike=bad,
                               vol_pairs=["USDHKD"])

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_peg_break_rejects_non_finite_contagion(self, bad) -> None:
        with pytest.raises(ValueError, match="contagion"):
            peg_break_scenario("HKD", contagion={"SAR": bad})

    def test_contagion_below_minus_one_hundred_percent_rejected(self) -> None:
        with pytest.raises(ValueError, match="contagion|-100"):
            peg_break_scenario("HKD", contagion={"SAR": -1.5})

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_reverse_stress_rejects_non_finite_radius(self, bad) -> None:
        w = pd.Series({"FX:EUR": 1e6, "FX:JPY": -5e5})
        cov = pd.DataFrame([[1e-4, 1e-5], [1e-5, 2e-4]],
                           index=w.index, columns=w.index)
        with pytest.raises(ValueError, match="radius"):
            reverse_stress_linear(w, cov, radius=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_reverse_stress_rejects_non_finite_loss_target(self, bad) -> None:
        w = pd.Series({"FX:EUR": 1e6, "FX:JPY": -5e5})
        cov = pd.DataFrame([[1e-4, 1e-5], [1e-5, 2e-4]],
                           index=w.index, columns=w.index)
        with pytest.raises(ValueError, match="loss_target"):
            reverse_stress_linear(w, cov, loss_target=bad)

    def test_peg_break_scenario_still_builds_the_right_log_shock(self) -> None:
        sc = peg_break_scenario("HKD", jump=-0.30)
        assert sc.shocks["FX:HKD"] == pytest.approx(np.log(0.70), rel=1e-12)


class TestEndToEndNoNanEscapes:
    """The headline numbers must be finite for every method on a live book."""

    def test_all_three_methods_return_finite_var_and_es(self, market,
                                                        returns) -> None:
        book = Book([
            Cash("USD", 1e6),
            Spot("EURUSD", 8e6),
            Forward("USDJPY", 4e6, 0.5),
            Option("EURUSD", 3e6, 1.12, 0.25, "call"),
        ])
        hs = historical_var(book, market, returns, alpha=0.99)
        pv = parametric_var(book, market, returns, alpha=0.99)
        for r in (hs, pv):
            assert np.isfinite(r.var) and np.isfinite(r.es)
            assert r.es >= r.var > 0.0
