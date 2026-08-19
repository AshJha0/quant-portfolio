"""Non-finite *parameter* rejection in the variance filters and estimators.

The return-series entry points already reject NaN/Inf data
(:func:`fx_vol._mle.validate_returns`, :func:`fx_vol.returns._as_1d_array`).
The gap this module closes is on the *parameter* side: the filters guarded
their coefficients with inequalities like ``if omega <= 0 or alpha < 0 or
beta < 0 or beta >= 1: raise``.  Every comparison against NaN is False, so a
NaN ``omega`` sailed straight through and ``garch_filter`` returned an
array of NaNs.  Downstream that becomes a NaN conditional variance, a NaN
volatility forecast and a NaN VaR — with no exception anywhere on the path.

Each test below feeds a non-finite parameter to a filter or estimator and
asserts a ``ValueError`` names the offending parameter, plus companion
tests that the *finite* path still produces strictly positive, finite
variance so the guards did not over-reject.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fx_vol.egarch import egarch_filter
from fx_vol.ewma import ewma_variance
from fx_vol.garch import garch_filter
from fx_vol.gjr import gjr_filter
from fx_vol.historical import (
    close_to_close_vol,
    garman_klass_vol,
    parkinson_vol,
    rolling_close_vol,
)
from fx_vol.returns import cross_volatility
from fx_vol.vol_premium import realized_vol_forward, variance_swap_pnl

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.fixture(scope="module")
def rets() -> np.ndarray:
    """A short, deterministic EURUSD-like daily log-return series."""
    rng = np.random.default_rng(11)
    return rng.standard_normal(250) * 0.005


class TestGarchFilterParameterGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize("name", ["omega", "alpha", "beta"])
    def test_non_finite_coefficient_raises(self, rets, name, bad) -> None:
        kwargs = {"omega": 1e-6, "alpha": 0.05, "beta": 0.9}
        kwargs[name] = bad
        with pytest.raises(ValueError, match=name):
            garch_filter(rets, **kwargs)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_initial_variance_raises(self, rets, bad) -> None:
        with pytest.raises(ValueError, match="initial_variance"):
            garch_filter(rets, 1e-6, 0.05, 0.9, initial_variance=bad)

    @pytest.mark.parametrize("bad", [0.0, -1e-8])
    def test_non_positive_initial_variance_raises(self, rets, bad) -> None:
        with pytest.raises(ValueError, match="initial_variance"):
            garch_filter(rets, 1e-6, 0.05, 0.9, initial_variance=bad)

    def test_finite_parameters_give_positive_finite_variance(self, rets) -> None:
        s2 = garch_filter(rets, 1e-6, 0.05, 0.9)
        assert s2.shape == rets.shape
        assert np.all(np.isfinite(s2)) and np.all(s2 > 0.0)


class TestGjrFilterParameterGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize("name", ["omega", "alpha", "gamma", "beta"])
    def test_non_finite_coefficient_raises(self, rets, name, bad) -> None:
        kwargs = {"omega": 1e-6, "alpha": 0.03, "gamma": 0.04, "beta": 0.9}
        kwargs[name] = bad
        with pytest.raises(ValueError, match=name):
            gjr_filter(rets, **kwargs)

    @pytest.mark.parametrize("bad", NON_FINITE + [0.0, -1.0])
    def test_bad_initial_variance_raises(self, rets, bad) -> None:
        with pytest.raises(ValueError, match="initial_variance"):
            gjr_filter(rets, 1e-6, 0.03, 0.04, 0.9, initial_variance=bad)

    def test_finite_parameters_give_positive_finite_variance(self, rets) -> None:
        s2 = gjr_filter(rets, 1e-6, 0.03, 0.04, 0.9)
        assert np.all(np.isfinite(s2)) and np.all(s2 > 0.0)


class TestEgarchFilterParameterGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize("name", ["omega", "alpha", "gamma"])
    def test_non_finite_coefficient_raises(self, rets, name, bad) -> None:
        kwargs = {"omega": -0.2, "alpha": 0.1, "gamma": -0.05, "beta": 0.97}
        kwargs[name] = bad
        with pytest.raises(ValueError, match=name):
            egarch_filter(rets, **kwargs)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_abs_moment_raises(self, rets, bad) -> None:
        with pytest.raises(ValueError, match="abs_moment"):
            egarch_filter(rets, -0.2, 0.1, -0.05, 0.97, abs_moment=bad)

    @pytest.mark.parametrize("bad", NON_FINITE + [0.0, -1.0])
    def test_bad_initial_variance_raises(self, rets, bad) -> None:
        with pytest.raises(ValueError, match="initial|variance"):
            egarch_filter(rets, -0.2, 0.1, -0.05, 0.97, initial_variance=bad)

    def test_finite_parameters_give_positive_finite_variance(self, rets) -> None:
        s2 = egarch_filter(rets, -0.2, 0.1, -0.05, 0.97)
        assert np.all(np.isfinite(s2)) and np.all(s2 > 0.0)


class TestEwmaSeedGuard:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_seed_rejected(self, rets, bad) -> None:
        with pytest.raises(ValueError, match="init"):
            ewma_variance(rets, lam=0.94, init=bad)

    def test_finite_seed_gives_finite_path(self, rets) -> None:
        s2 = ewma_variance(rets, lam=0.94, init=1e-5)
        assert np.all(np.isfinite(s2)) and np.all(s2 > 0.0)


class TestAnnualisationFactorGuard:
    """``periods_per_year`` multiplies a variance then gets square-rooted."""

    @pytest.mark.parametrize("bad", NON_FINITE + [0, -252])
    def test_close_to_close_rejects_bad_ppy(self, rets, bad) -> None:
        with pytest.raises(ValueError, match="periods_per_year"):
            close_to_close_vol(rets, periods_per_year=bad)

    @pytest.mark.parametrize("bad", NON_FINITE + [0, -252])
    def test_rolling_rejects_bad_ppy(self, rets, bad) -> None:
        with pytest.raises(ValueError, match="periods_per_year"):
            rolling_close_vol(pd.Series(rets), window=20, periods_per_year=bad)

    @pytest.mark.parametrize("bad", NON_FINITE + [0, -252])
    def test_parkinson_rejects_bad_ppy(self, bad) -> None:
        high = np.array([1.101, 1.104, 1.099, 1.102])
        low = np.array([1.098, 1.100, 1.095, 1.097])
        with pytest.raises(ValueError, match="periods_per_year"):
            parkinson_vol(high, low, periods_per_year=bad)

    @pytest.mark.parametrize("bad", NON_FINITE + [0, -252])
    def test_garman_klass_rejects_bad_ppy(self, bad) -> None:
        o = np.array([1.100, 1.101, 1.097, 1.100])
        h = np.array([1.101, 1.104, 1.099, 1.102])
        low = np.array([1.098, 1.100, 1.095, 1.097])
        c = np.array([1.101, 1.102, 1.098, 1.101])
        with pytest.raises(ValueError, match="periods_per_year"):
            garman_klass_vol(o, h, low, c, periods_per_year=bad)

    @pytest.mark.parametrize("bad", NON_FINITE + [0, -252])
    def test_realized_vol_forward_rejects_bad_ppy(self, rets, bad) -> None:
        with pytest.raises(ValueError, match="periods_per_year"):
            realized_vol_forward(rets, window=20, periods_per_year=bad)

    def test_260_vs_252_convention_ratio_is_exact(self, rets) -> None:
        # The FX 24h/5d convention (260) vs the equity default (252) is a
        # pure constant factor; a desk comparing RV to an implied quote on
        # the other convention is off by ~1.6% of vol.
        v252 = close_to_close_vol(rets, periods_per_year=252)
        v260 = close_to_close_vol(rets, periods_per_year=260)
        assert v260 / v252 == pytest.approx(np.sqrt(260.0 / 252.0), rel=1e-12)


class TestCrossVolatilityGuards:
    """Triangulated cross vol takes three bare scalars."""

    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize("name", ["vol1", "vol2", "corr"])
    def test_non_finite_input_rejected(self, name, bad) -> None:
        kwargs = {"vol1": 0.08, "vol2": 0.09, "corr": 0.3, "sign_product": 1}
        kwargs[name] = bad
        with pytest.raises(ValueError, match=f"{name}|correlation"):
            cross_volatility(**kwargs)

    def test_eurjpy_from_eurusd_and_usdjpy_is_finite_and_sane(self) -> None:
        # Common-currency legs quoted the same way -> sign product +1, so
        # positive correlation widens the cross.
        v = cross_volatility(0.07, 0.09, 0.35, pair1="EURUSD",
                             pair2="USDJPY", cross="EURJPY")
        assert np.isfinite(v)
        assert v > max(0.07, 0.09)
        # Uncorrelated legs give the Pythagorean sum.
        v0 = cross_volatility(0.07, 0.09, 0.0, sign_product=1)
        assert v0 == pytest.approx(np.hypot(0.07, 0.09), rel=1e-12)


class TestVarianceSwapNotionalGuard:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_vega_notional_rejected(self, bad) -> None:
        with pytest.raises(ValueError, match="vega_notional"):
            variance_swap_pnl(np.array([0.10, 0.11]), np.array([0.08, 0.09]),
                              vega_notional=bad)

    def test_short_variance_pnl_sign_follows_the_premium(self) -> None:
        iv = np.array([0.10, 0.10])
        pnl_rich = variance_swap_pnl(iv, np.array([0.08, 0.06]))
        pnl_poor = variance_swap_pnl(iv, np.array([0.12, 0.20]))
        assert np.all(pnl_rich > 0.0)   # sold vol above realized
        assert np.all(pnl_poor < 0.0)   # realized overshot the strike
        # Convexity: the loss when realized doubles the strike is far bigger
        # than the gain when realized halves it.
        gain = variance_swap_pnl(np.array([0.10]), np.array([0.05]))[0]
        loss = -variance_swap_pnl(np.array([0.10]), np.array([0.20]))[0]
        assert loss > 3.0 * gain
