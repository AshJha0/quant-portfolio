"""Historical estimators: recovery on constant-vol data, range estimators,
annualization conventions, day-of-week seasonality."""

import numpy as np
import pandas as pd
import pytest

from fx_vol import (
    close_to_close_vol,
    day_of_week_vol_factors,
    garman_klass_vol,
    parkinson_vol,
    rolling_close_vol,
)
from fx_vol.data import synthetic as syn


def _intraday_ohlc(n_days: int, daily_vol: float, m: int, seed: int):
    """Simulate driftless intraday GBM and aggregate to daily OHLC."""
    rng = np.random.default_rng(seed)
    steps = daily_vol / np.sqrt(m) * rng.standard_normal((n_days, m))
    log_paths = np.cumsum(steps, axis=1)
    log_open = np.concatenate([[0.0], np.cumsum(log_paths[:-1, -1])])[:n_days]
    lp = log_open[:, None] + np.concatenate([np.zeros((n_days, 1)), log_paths], axis=1)
    o = np.exp(lp[:, 0]); h = np.exp(lp.max(axis=1)); l = np.exp(lp.min(axis=1)); c = np.exp(lp[:, -1])
    return o, h, l, c


class TestCloseToClose:
    def test_recovers_constant_vol(self):
        vol = 0.006
        r = syn.simulate_constant_vol(100_000, vol, seed=21)
        est = close_to_close_vol(r, periods_per_year=252)
        assert est == pytest.approx(vol * np.sqrt(252), rel=0.01)

    def test_annualization_252_vs_260(self):
        r = syn.simulate_constant_vol(1000, 0.006, seed=22)
        ratio = close_to_close_vol(r, 260) / close_to_close_vol(r, 252)
        assert ratio == pytest.approx(np.sqrt(260 / 252), abs=1e-12)

    def test_rejects_nan_and_short(self):
        with pytest.raises(ValueError, match="NaN"):
            close_to_close_vol([0.001, np.nan, 0.002])
        with pytest.raises(ValueError, match="at least"):
            close_to_close_vol([0.001])
        with pytest.raises(ValueError, match="periods_per_year"):
            close_to_close_vol([0.001, 0.002], periods_per_year=0)

    def test_rolling_matches_pandas(self):
        r = pd.Series(syn.simulate_constant_vol(300, 0.006, seed=23))
        rv = rolling_close_vol(r, window=50)
        expected = r.rolling(50).std() * np.sqrt(252)
        pd.testing.assert_series_equal(rv, expected)

    def test_rolling_validation(self):
        r = pd.Series(syn.simulate_constant_vol(100, 0.006, seed=24))
        with pytest.raises(ValueError, match="window"):
            rolling_close_vol(r, window=1)
        with pytest.raises(ValueError, match="exceeds"):
            rolling_close_vol(r, window=101)


class TestRangeEstimators:
    def test_parkinson_recovers_true_vol(self):
        daily_vol = 0.006
        o, h, l, c = _intraday_ohlc(252, daily_vol, m=780, seed=25)
        est = parkinson_vol(h, l, periods_per_year=252)
        # discrete monitoring biases the range slightly low; 780 steps ~ few %
        assert est == pytest.approx(daily_vol * np.sqrt(252), rel=0.07)

    def test_garman_klass_recovers_true_vol(self):
        daily_vol = 0.006
        o, h, l, c = _intraday_ohlc(252, daily_vol, m=780, seed=26)
        est = garman_klass_vol(o, h, l, c, periods_per_year=252)
        assert est == pytest.approx(daily_vol * np.sqrt(252), rel=0.07)

    def test_range_estimators_less_noisy_than_close(self):
        """Efficiency: across replications the range estimators have smaller
        dispersion around truth than close-to-close on the same days."""
        daily_vol = 0.006
        cc, pk = [], []
        for k in range(30):
            o, h, l, c = _intraday_ohlc(60, daily_vol, m=390, seed=100 + k)
            r = np.diff(np.log(c))
            cc.append(close_to_close_vol(r))
            pk.append(parkinson_vol(h, l))
        assert np.std(pk) < np.std(cc)

    def test_high_below_low_rejected(self):
        with pytest.raises(ValueError, match="high < low"):
            parkinson_vol([1.0, 1.0], [1.1, 0.9])
        with pytest.raises(ValueError, match="high < low"):
            garman_klass_vol([1.0, 1.0], [1.0, 1.0], [1.1, 0.9], [1.0, 1.0])

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="equal length"):
            parkinson_vol([1.1, 1.1], [1.0])


class TestDayOfWeekSeasonality:
    def test_recovers_injected_weekly_pattern(self):
        factors = {0: 0.85, 1: 0.95, 2: 1.15, 3: 1.0, 4: 1.10}
        r = syn.simulate_seasonal_returns(5000, weekday_factors=factors, seed=27)
        est = day_of_week_vol_factors(r)
        # ratio Wednesday / Monday should recover 1.15/0.85
        assert est["Wednesday"] / est["Monday"] == pytest.approx(1.15 / 0.85, rel=0.08)
        # normalization: mean squared factor = 1
        assert np.mean(est.to_numpy() ** 2) == pytest.approx(1.0, abs=1e-10)

    def test_flat_when_no_seasonality(self):
        flat = {d: 1.0 for d in range(5)}
        r = syn.simulate_seasonal_returns(5000, weekday_factors=flat, seed=28)
        est = day_of_week_vol_factors(r)
        assert np.allclose(est.to_numpy(), 1.0, atol=0.05)

    def test_requires_datetime_index(self):
        with pytest.raises(ValueError, match="DatetimeIndex"):
            day_of_week_vol_factors(pd.Series([0.01, -0.01]))

    def test_rejects_weekend_stamps(self):
        idx = pd.date_range("2024-01-06", periods=10, freq="D")  # includes weekends
        r = pd.Series(0.001 * np.arange(10), index=idx)
        with pytest.raises(ValueError, match="weekend"):
            day_of_week_vol_factors(r)

    def test_min_obs_enforced(self):
        r = syn.simulate_seasonal_returns(8, seed=29)
        with pytest.raises(ValueError, match="need >="):
            day_of_week_vol_factors(r)
