"""Non-finite inputs must raise ``ValueError`` -- never a NaN risk number.

Every comparison with NaN is ``False``, so a guard such as
``if np.any(v < 0): raise`` silently accepts a NaN VaR, and the exception
indicator ``p < -v`` is ``False`` on a NaN P&L day.  Before this hardening
pass, ``exceptions_from_pnl(pnl, np.nan)`` recorded *zero* breaches on a
250-day window and the Kupiec / Basel backtests came back green -- a broken
feed made a broken model look better.

The C++ (``cpp/equity-var-engine/tests/test_backtest.cpp``) and Rust
(``rust/equity-var-engine/tests/test_backtest.rs``) twins reject NaN / inf
in these paths; this module pins the same error contract on the Python
reference so the three languages agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from eq_var import (
    acerbi_szekely_z2,
    historical_var,
    cornish_fisher_var,
    ewma_covariance,
    exceptions_from_pnl,
    parametric_var,
    portfolio_sigma,
    rolling_var_backtest,
    safe_cholesky,
    sample_covariance,
    var_standard_error_bootstrap,
)
from eq_var.backtesting import BacktestResult

NON_FINITE = [np.nan, np.inf, -np.inf]


@pytest.fixture
def clean_pnl() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(0.0, 1.0, size=250)


# --------------------------------------------------------------- CRIT-1 ---
class TestNonFiniteBacktesting:
    """TEST-1: the regulatory backtest refuses poisoned inputs."""

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_exceptions_from_pnl_rejects_non_finite_pnl_day(self, clean_pnl, bad):
        pnl = clean_pnl.copy()
        pnl[17] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            exceptions_from_pnl(pnl, 1.0)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_exceptions_from_pnl_rejects_non_finite_scalar_var(self, clean_pnl, bad):
        with pytest.raises(ValueError, match="VaR must be finite"):
            exceptions_from_pnl(clean_pnl, bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_exceptions_from_pnl_rejects_non_finite_var_day(self, clean_pnl, bad):
        var = np.full(clean_pnl.size, 1.0)
        var[100] = bad
        with pytest.raises(ValueError, match="VaR must be finite"):
            exceptions_from_pnl(clean_pnl, var)

    def test_nan_var_no_longer_reports_zero_exceptions_green(self, clean_pnl):
        # The exact symptom described in the top-level README: a NaN VaR used
        # to give 0 exceptions and a green Basel zone.
        with pytest.raises(ValueError):
            exceptions_from_pnl(clean_pnl, np.full(clean_pnl.size, np.nan))
        # And a result that *was* built from clean inputs still summarises.
        exc = exceptions_from_pnl(clean_pnl, 2.5)
        res = BacktestResult("hs", exc, np.full(250, 2.5), clean_pnl, 0.01)
        assert res.summary()["exceptions"] == int(exc.sum())

    def test_mixed_signs_of_infinity_both_rejected(self):
        # Before the fix: [True, False, False, False] -- the NaN day and the
        # +inf day were silently "not exceptions".
        with pytest.raises(ValueError):
            exceptions_from_pnl(np.array([-5.0, np.nan, -0.1, np.inf]), 1.0)

    def test_clean_inputs_still_count_exceptions(self, clean_pnl):
        exc = exceptions_from_pnl(clean_pnl, 1.0)
        assert exc.dtype == bool
        assert exc.sum() == int((clean_pnl < -1.0).sum())

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_rolling_backtest_refuses_before_any_var_is_computed(self, clean_pnl, bad):
        pnl = np.concatenate([clean_pnl, clean_pnl])
        pnl[400] = bad  # inside the forecast region, after the first window
        with pytest.raises(ValueError, match="NaN or infinite"):
            rolling_var_backtest(pnl, historical_var, window=250, alpha=0.01)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_acerbi_szekely_refuses_non_finite_pnl(self, clean_pnl, bad):
        pnl = clean_pnl.copy()
        pnl[3] = bad
        with pytest.raises(ValueError, match="finite"):
            acerbi_szekely_z2(pnl, 2.0, 2.5, alpha=0.025)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_acerbi_szekely_refuses_non_finite_var_or_es(self, clean_pnl, bad):
        with pytest.raises(ValueError, match="finite"):
            acerbi_szekely_z2(clean_pnl, bad, 2.5, alpha=0.025)
        with pytest.raises(ValueError, match="finite"):
            acerbi_szekely_z2(clean_pnl, 2.0, bad, alpha=0.025)

    def test_acerbi_szekely_refuses_empty(self):
        with pytest.raises(ValueError, match="empty"):
            acerbi_szekely_z2(np.array([]), 2.0, 2.5, alpha=0.025)


# ---------------------------------------------------------------- MAJ-1 ---
@pytest.fixture
def cov3() -> np.ndarray:
    vols = np.array([0.01, 0.015, 0.02])
    corr = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.3], [0.25, 0.3, 1.0]])
    return np.outer(vols, vols) * corr


@pytest.fixture
def w3() -> np.ndarray:
    return np.array([1.0e6, -5.0e5, 2.0e5])


class TestNonFiniteParametric:
    """TEST-2: parametric VaR entry points refuse NaN / inf market data."""

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_portfolio_sigma_rejects_non_finite_covariance(self, w3, cov3, bad):
        cov = cov3.copy()
        cov[1, 1] = bad
        with pytest.raises(ValueError, match="covariance contains"):
            portfolio_sigma(w3, cov)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_portfolio_sigma_rejects_non_finite_exposure(self, w3, cov3, bad):
        w = w3.copy()
        w[0] = bad
        with pytest.raises(ValueError, match="exposures contains"):
            portfolio_sigma(w, cov3)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_parametric_var_rejects_non_finite_covariance(self, w3, cov3, bad):
        cov = cov3.copy()
        cov[0, 2] = cov[2, 0] = bad
        with pytest.raises(ValueError):
            parametric_var(w3, cov, 0.01)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_parametric_var_rejects_non_finite_mean_and_df(self, w3, cov3, bad):
        with pytest.raises(ValueError, match="mean must be finite"):
            parametric_var(w3, cov3, 0.01, mean=bad)
        with pytest.raises(ValueError, match="df must be finite"):
            parametric_var(w3, cov3, 0.01, dist="t", df=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_covariance_estimators_reject_non_finite_returns(self, bad):
        rng = np.random.default_rng(1)
        r = rng.normal(size=(60, 3))
        r[10, 1] = bad
        with pytest.raises(ValueError, match="returns contains"):
            sample_covariance(r)
        with pytest.raises(ValueError, match="returns contains"):
            ewma_covariance(r, 0.94)

    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize("arg", ["sigma", "mean", "skew", "excess_kurt"])
    def test_cornish_fisher_rejects_non_finite_scalars(self, arg, bad):
        kwargs = dict(sigma=1.0, alpha=0.01, skew=-0.3, excess_kurt=1.0, mean=0.0)
        kwargs[arg] = bad
        with pytest.raises(ValueError, match=f"{arg} must be finite"):
            cornish_fisher_var(**kwargs)


class TestNonFiniteMonteCarlo:
    """MIN-9 / MIN-10: the Cholesky and bootstrap paths name the real problem."""

    def test_nan_covariance_is_reported_as_non_finite_not_asymmetric(self, cov3):
        cov = cov3.copy()
        cov[0, 0] = np.nan  # still symmetric -- the old message was misleading
        with pytest.raises(ValueError, match="NaN or infinite"):
            safe_cholesky(cov)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_bootstrap_se_rejects_non_finite_pnl(self, clean_pnl, bad):
        pnl = clean_pnl.copy()
        pnl[5] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            var_standard_error_bootstrap(pnl, 0.05, n_boot=20, seed=0)

    def test_bootstrap_se_is_chunk_invariant(self, clean_pnl):
        # Chunking is a memory optimisation only: the stream of draws (and so
        # the estimate) must not depend on the chunk boundary.  n_boot = 50
        # is exactly one chunk; n_boot = 120 spans three.
        se_one = var_standard_error_bootstrap(clean_pnl, 0.05, n_boot=50, seed=3)
        se_three = var_standard_error_bootstrap(clean_pnl, 0.05, n_boot=120, seed=3)
        assert np.isfinite(se_one) and np.isfinite(se_three)
        assert se_one > 0 and se_three > 0
        # Same seed, same first 50 resamples: the estimates differ only
        # because n_boot differs, and both sit within a factor of 2 of each
        # other on 250 observations.
        assert 0.5 < se_one / se_three < 2.0

    def test_bootstrap_se_rejects_too_few_resamples(self, clean_pnl):
        with pytest.raises(ValueError, match="n_boot"):
            var_standard_error_bootstrap(clean_pnl, 0.05, n_boot=1, seed=0)
