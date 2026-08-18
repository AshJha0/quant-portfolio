"""Feature engineering tests — point-in-time discipline is the core contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq_regime.features import (
    average_pairwise_correlation,
    build_features,
    credit_proxy_spread,
    drawdown_depth,
    expanding_zscore,
    realized_vol,
    return_dispersion,
    term_proxy,
    trend_strength,
)


def _dates(n):
    return pd.bdate_range("2020-01-01", periods=n)


class TestExpandingZScore:
    def test_pit_mutation_future_changes_leave_past_identical(self, panel3):
        """THE no-lookahead test: mutating future prices must leave every
        past feature value bit-identical."""
        prices = panel3.prices
        cutoff = prices.index[1000]
        mutated = prices.copy()
        mutated.iloc[1100:] *= 1.5  # violent future shock
        f_base = build_features(prices)
        f_mut = build_features(mutated)
        a = f_base.loc[:cutoff]
        b = f_mut.loc[:cutoff]
        pd.testing.assert_frame_equal(a, b, check_exact=True)

    def test_matches_manual_computation(self):
        rng = np.random.default_rng(1)
        s = pd.Series(rng.standard_normal(100), index=_dates(100))
        z = expanding_zscore(s, min_periods=10)
        for t in (10, 40, 99):
            past = s.iloc[: t + 1]
            expected = (s.iloc[t] - past.mean()) / past.std(ddof=1)
            assert z.iloc[t] == pytest.approx(expected, abs=1e-12)

    def test_warmup_is_nan(self):
        s = pd.Series(np.arange(20.0), index=_dates(20))
        z = expanding_zscore(s, min_periods=10)
        assert z.iloc[:9].isna().all()
        assert z.iloc[10:].notna().all()

    def test_constant_series_yields_nan_not_inf(self):
        s = pd.Series(np.ones(30), index=_dates(30))
        z = expanding_zscore(s, min_periods=5)
        assert not np.isinf(z.to_numpy()).any()
        assert z.iloc[10:].isna().all()

    def test_min_periods_validation(self):
        s = pd.Series(np.arange(10.0), index=_dates(10))
        with pytest.raises(ValueError, match="min_periods"):
            expanding_zscore(s, min_periods=1)


class TestHandChecked:
    def test_dispersion_two_assets(self):
        """std of 2 points (ddof=1) is |a-b|/sqrt(2)."""
        r = pd.DataFrame({"a": [0.01, -0.02], "b": [0.03, 0.02]}, index=_dates(2))
        d = return_dispersion(r, smooth=1)
        assert d.iloc[0] == pytest.approx(0.02 / np.sqrt(2))
        assert d.iloc[1] == pytest.approx(0.04 / np.sqrt(2))

    def test_avg_pairwise_correlation_tiny_panel(self):
        """Assets a == a2 (corr 1) and b orthogonal to both -> mean 1/3."""
        x = np.array([1.0, -1.0, 1.0, -1.0] * 3)
        y = np.array([1.0, 1.0, -1.0, -1.0] * 3)
        r = pd.DataFrame({"a": x, "a2": x, "b": y}, index=_dates(12))
        c = average_pairwise_correlation(r, window=12)
        assert c.iloc[-1] == pytest.approx(1.0 / 3.0, abs=1e-12)

    def test_avg_pairwise_correlation_perfect(self):
        rng = np.random.default_rng(2)
        base = rng.standard_normal(30)
        r = pd.DataFrame({"a": base, "b": 2 * base, "c": 0.5 * base}, index=_dates(30))
        c = average_pairwise_correlation(r, window=20)
        assert c.iloc[-1] == pytest.approx(1.0, abs=1e-10)

    def test_realized_vol_matches_numpy(self):
        rng = np.random.default_rng(3)
        r = pd.DataFrame(rng.standard_normal((60, 3)) * 0.01, index=_dates(60))
        v = realized_vol(r, windows=(21,))
        idx = r.mean(axis=1)
        expected = idx.iloc[-21:].std(ddof=1) * np.sqrt(252)
        assert v["vol_21d"].iloc[-1] == pytest.approx(expected, abs=1e-14)

    def test_drawdown_depth_constructed(self):
        p = pd.DataFrame({"a": [100, 110, 121, 96.8, 108.9], "b": [100, 110, 121, 96.8, 108.9]},
                         index=_dates(5), dtype=float)
        dd = drawdown_depth(p)
        assert dd.iloc[2] == pytest.approx(0.0)
        assert dd.iloc[3] == pytest.approx(1 - 96.8 / 121)

    def test_trend_strength_signs(self):
        up = pd.DataFrame({"a": np.linspace(100, 200, 50), "b": np.linspace(100, 200, 50)},
                          index=_dates(50))
        t = trend_strength(up, ma_window=10)
        assert (t.iloc[10:] > 0).all()
        down = pd.DataFrame({"a": np.linspace(200, 100, 50), "b": np.linspace(200, 100, 50)},
                            index=_dates(50))
        t2 = trend_strength(down, ma_window=10)
        assert (t2.iloc[10:] < 0).all()


class TestProxiesAndTable:
    def test_credit_proxy_higher_in_bear(self, panel3):
        cp = credit_proxy_spread(panel3.returns)
        states = pd.Series(panel3.states, index=panel3.returns.index)
        valid = cp.dropna().index
        bear = panel3.n_states - 1
        cpv, sv = cp.loc[valid], states.loc[valid]
        assert cpv[sv == bear].mean() > cpv[sv == 0].mean()

    def test_term_proxy_negative_in_bear(self, panel3):
        tp = term_proxy(panel3.returns)
        states = pd.Series(panel3.states, index=panel3.returns.index)
        valid = tp.dropna().index
        bear = panel3.n_states - 1
        tpv, sv = tp.loc[valid], states.loc[valid]
        # short vol spikes above long vol in the bear state
        assert tpv[sv == bear].mean() < tpv[sv == 0].mean()

    def test_no_nan_after_warmup(self, features3):
        assert features3.notna().all().all()
        assert len(features3) > 1000

    def test_expected_columns(self, features3):
        for col in ["vol_10d", "vol_21d", "vol_63d", "dispersion", "avg_corr",
                    "drawdown", "trend", "credit_proxy", "term_proxy"]:
            assert col in features3.columns

    def test_short_history_raises(self):
        p = pd.DataFrame(100.0, index=_dates(50), columns=["a", "b"])
        with pytest.raises(ValueError, match="too short"):
            build_features(p)

    def test_single_asset_raises(self):
        p = pd.DataFrame(100.0, index=_dates(600), columns=["a"])
        with pytest.raises(ValueError, match="2 assets"):
            build_features(p)
