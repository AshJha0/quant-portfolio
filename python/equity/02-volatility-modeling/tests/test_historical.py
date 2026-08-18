"""Historical & range-based volatility estimators."""

import numpy as np
import pytest

from eq_vol.data import synthetic as syn
from eq_vol.historical import (
    close_to_close_var,
    garman_klass_var,
    parkinson_var,
    range_vol,
    realized_vol,
    rogers_satchell_var,
    window_sensitivity,
)

SIGMA = 0.20


class TestCloseToClose:
    def test_recovers_true_sigma(self):
        # SE of the vol estimate ~ sigma / sqrt(2n); allow 3 SE
        n = 4000
        r = syn.simulate_gbm_returns(n, sigma_annual=SIGMA, seed=10)
        vol = realized_vol(r, window=n)[-1]
        tol = 3.0 * SIGMA / np.sqrt(2.0 * n)
        assert abs(vol - SIGMA) < tol

    def test_annualization_factor(self):
        r = syn.simulate_gbm_returns(500, sigma_annual=SIGMA, seed=11)
        v252 = realized_vol(r, window=100, annualization=252)
        v1 = realized_vol(r, window=100, annualization=1)
        mask = ~np.isnan(v252)
        np.testing.assert_allclose(v252[mask], v1[mask] * np.sqrt(252.0), rtol=1e-12)

    def test_rolling_matches_manual_mean_of_squares(self):
        r = syn.simulate_gbm_returns(100, seed=12)
        w = 21
        vol = realized_vol(r, window=w)
        for t in [w - 1, 50, 99]:
            manual = np.sqrt(np.mean(r[t - w + 1 : t + 1] ** 2) * 252.0)
            assert vol[t] == pytest.approx(manual, rel=1e-12)

    def test_nan_prefix_length(self):
        r = syn.simulate_gbm_returns(60, seed=13)
        vol = realized_vol(r, window=21)
        assert np.all(np.isnan(vol[:20])) and np.all(np.isfinite(vol[20:]))

    def test_demean_reduces_drift_bias(self):
        # large drift inflates the zero-mean estimator; demeaning removes it
        r = syn.simulate_gbm_returns(6000, sigma_annual=SIGMA, mu_annual=3.0, seed=14)
        v_raw = realized_vol(r, window=6000)[-1]
        v_dm = realized_vol(r, window=6000, demean=True)[-1]
        assert abs(v_dm - SIGMA) < abs(v_raw - SIGMA)


@pytest.fixture(scope="module")
def ohlc():
    return syn.simulate_gbm_ohlc(1500, sigma_annual=SIGMA, steps_per_day=500, seed=20)


class TestRangeEstimators:
    @pytest.mark.parametrize("estimator", ["parkinson", "garman_klass", "rogers_satchell"])
    def test_recovers_true_sigma(self, ohlc, estimator):
        vol = range_vol(ohlc, estimator=estimator, window=len(ohlc))[-1]
        # 5% tolerance: covers sampling error plus the documented downward
        # discrete-monitoring bias (O(1/sqrt(steps_per_day)))
        assert vol == pytest.approx(SIGMA, rel=0.05)

    def test_lower_sampling_variance_than_close_to_close(self):
        # many replications: dispersion of the range-based estimates across
        # reps must be materially below close-to-close (theoretical
        # efficiency gains ~5-7x => sd ratio ~0.4-0.45)
        reps, days = 120, 120
        big = syn.simulate_gbm_ohlc(reps * days, sigma_annual=SIGMA, steps_per_day=150, seed=21)
        o, h, l, c = (big[k].to_numpy() for k in ("open", "high", "low", "close"))
        r_cc = np.log(c / o)  # no overnight gap: open = previous close
        cc = (r_cc**2).reshape(reps, days).mean(axis=1)
        pk = parkinson_var(h, l).reshape(reps, days).mean(axis=1)
        gk = garman_klass_var(o, h, l, c).reshape(reps, days).mean(axis=1)
        rs = rogers_satchell_var(o, h, l, c).reshape(reps, days).mean(axis=1)
        assert pk.std() < 0.7 * cc.std()
        assert gk.std() < 0.7 * cc.std()
        assert rs.std() < 0.7 * cc.std()

    def test_rogers_satchell_drift_robust(self):
        # under strong drift, RS stays centred on sigma while the raw
        # close-to-close estimate is inflated by the drift component
        ohlc = syn.simulate_gbm_ohlc(2000, sigma_annual=SIGMA, mu_annual=2.0,
                                     steps_per_day=300, seed=22)
        rs = range_vol(ohlc, estimator="rogers_satchell", window=len(ohlc))[-1]
        r_cc = np.log(ohlc["close"].to_numpy() / ohlc["open"].to_numpy())
        cc = np.sqrt(np.mean(r_cc**2) * 252.0)
        assert abs(rs - SIGMA) < abs(cc - SIGMA)

    def test_estimators_positive(self, ohlc):
        o, h, l, c = (ohlc[k].to_numpy() for k in ("open", "high", "low", "close"))
        assert np.all(parkinson_var(h, l) >= 0)
        assert np.all(garman_klass_var(o, h, l, c) > -1e-12)
        # RS can be slightly negative on individual days; rolling means are positive
        vol = range_vol(ohlc, estimator="rogers_satchell", window=21)
        assert np.nanmin(vol) > 0

    def test_inconsistent_ohlc_raises(self):
        with pytest.raises(ValueError, match="inconsistent OHLC"):
            garman_klass_var([100.0], [99.0], [98.0], [100.5])

    def test_nonpositive_price_raises(self):
        with pytest.raises(ValueError, match="non-positive"):
            parkinson_var([100.0, -1.0], [99.0, -2.0])

    def test_unknown_estimator_raises(self, ohlc):
        with pytest.raises(ValueError, match="unknown estimator"):
            range_vol(ohlc, estimator="yang_zhang")


class TestWindowSensitivity:
    def test_frame_contents(self):
        r = syn.simulate_gbm_returns(1000, sigma_annual=SIGMA, seed=30)
        df = window_sensitivity(r, windows=(10, 21, 63))
        assert list(df.index) == [10, 21, 63]
        assert set(df.columns) >= {"latest_vol", "mean_vol", "std_of_estimate", "approx_sampling_std"}
        # theoretical sampling noise decreases with window length
        assert df["approx_sampling_std"].is_monotonic_decreasing
        # empirical dispersion should too (constant-vol data)
        assert df["std_of_estimate"].iloc[0] > df["std_of_estimate"].iloc[-1]

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="at least"):
            window_sensitivity(np.zeros(50), windows=(10, 100))

    def test_close_to_close_var_demean(self):
        r = np.array([0.01, -0.02, 0.03, 0.0])
        np.testing.assert_allclose(close_to_close_var(r), r**2)
        np.testing.assert_allclose(close_to_close_var(r, demean=True), (r - r.mean()) ** 2)
