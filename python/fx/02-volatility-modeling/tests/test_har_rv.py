"""HAR-RV for FX: 17:00 New York day construction and Corsi (2009) oracles."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fx_vol.har_rv import (
    fit_har,
    fx_daily_realized_variance,
    har_design,
    realized_variance,
    simulate_har,
)

TRUE = np.array([1e-6, 0.35, 0.30, 0.20])


def test_realized_variance_is_sum_of_squares():
    r = np.array([[0.001, -0.002, 0.0005], [0.0, 0.003, -0.001]])
    np.testing.assert_allclose(realized_variance(r), [5.25e-6, 1e-5], rtol=1e-14)
    with pytest.raises(ValueError):
        realized_variance(np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        realized_variance(np.array([[0.1, np.nan]]))


def _ticks(start_utc: str, periods: int, freq: str, seed: int = 0) -> pd.Series:
    idx = pd.date_range(start_utc, periods=periods, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 1e-4, periods))), index=idx)


def test_fx_day_cut_follows_new_york_dst():
    """17:00 New York is 21:00 UTC in summer and 22:00 UTC in winter."""
    for day, cut_utc in (("2024-07-10", "21:00"), ("2024-01-10", "22:00")):
        t = pd.Timestamp(f"{day} {cut_utc}", tz="UTC")
        idx = pd.DatetimeIndex([t - pd.Timedelta(minutes=10), t - pd.Timedelta(minutes=5), t,
                                t + pd.Timedelta(minutes=5), t + pd.Timedelta(minutes=10)])
        p = pd.Series([1.0, 1.01, 1.0201, 1.0, 1.02], index=idx)
        rv = fx_daily_realized_variance(p, min_obs=1)
        d = pd.Timestamp(day)
        # the tick at the cut closes day D; the next two returns are day D+1
        assert list(rv.index) == [d, d + pd.Timedelta(days=1)]
        assert rv.iloc[0] == pytest.approx(2 * np.log(1.01) ** 2, rel=1e-12)
        assert rv.iloc[1] == pytest.approx(np.log(1.02) ** 2, rel=1e-12)


def test_fx_day_drops_boundary_return_and_weekend():
    # Friday 2024-03-08 through Monday 2024-03-11, a tick every 30 minutes
    p = _ticks("2024-03-07 23:00", 4 * 48, "30min", seed=3)
    rv = fx_daily_realized_variance(p)
    assert all(d.dayofweek < 5 for d in rv.index)
    lr = np.diff(np.log(p.to_numpy()))
    # rebuild one full day by hand: Friday 08 = (Thu 17:00, Fri 17:00] New York
    wall = p.index.tz_convert("America/New_York").tz_localize(None)
    lo, hi = pd.Timestamp("2024-03-07 17:00"), pd.Timestamp("2024-03-08 17:00")
    inside = (wall[1:] > lo) & (wall[1:] <= hi) & (wall[:-1] > lo)
    assert rv.loc[pd.Timestamp("2024-03-08")] == pytest.approx(np.sum(lr[inside] ** 2), rel=1e-12)


def test_fx_day_input_guards():
    p = _ticks("2024-03-04", 10, "1h")
    with pytest.raises(ValueError, match="timezone"):
        fx_daily_realized_variance(p.tz_localize(None))
    with pytest.raises(TypeError):
        fx_daily_realized_variance(p.to_numpy())
    with pytest.raises(ValueError, match="positive"):
        fx_daily_realized_variance(-p)
    with pytest.raises(ValueError, match="increasing"):
        fx_daily_realized_variance(p.iloc[::-1])


def test_noiseless_series_recovers_exact_params():
    rng = np.random.default_rng(2)
    rv = list(rng.random(22) * 1e-5)
    for _ in range(300):
        rv.append(TRUE[0] + TRUE[1] * rv[-1] + TRUE[2] * np.mean(rv[-5:])
                  + TRUE[3] * np.mean(rv[-22:]))
    res = fit_har(np.array(rv))
    np.testing.assert_allclose(res.params, TRUE, rtol=1e-6, atol=1e-13)


def test_ols_consistency_and_forecast_limit():
    rv = simulate_har(TRUE, 20_000, noise_sd=2e-6, seed=4)
    res = fit_har(rv)
    np.testing.assert_allclose(res.params[1:], TRUE[1:], atol=4 * res.se_hac[1:].max() + 0.02)
    b0, bd, bw, bm = res.params
    assert res.forecast(2_000)[-1] == pytest.approx(b0 / (1 - bd - bw - bm), rel=1e-6)
    f1 = res.forecast(1)[0]
    assert res.forecast_vol(1) == pytest.approx(np.sqrt(f1 * 260), rel=1e-12)
    with pytest.raises(ValueError):
        res.forecast(0)


def test_design_guards():
    with pytest.raises(ValueError):
        har_design(np.ones(20))
    with pytest.raises(ValueError):
        har_design(np.r_[np.ones(40), -1.0])
    with pytest.raises(ValueError):
        simulate_har(np.array([1e-6, 0.5, 0.3, 0.3]), 10, 1e-6)


def test_matches_equity_twin():
    src = Path(__file__).resolve().parents[3] / "equity" / "02-volatility-modeling" / "src"
    if not src.is_dir():
        pytest.skip("equity package not in this checkout")
    sys.path.insert(0, str(src))
    eq = pytest.importorskip("eq_vol.har_rv")
    rv = simulate_har(TRUE, 3_000, noise_sd=2e-6, seed=9)
    a, b = fit_har(rv), eq.fit_har(rv)
    np.testing.assert_allclose(a.params, b.params, rtol=1e-14, atol=0)
    np.testing.assert_allclose(a.se_hac, b.se_hac, rtol=1e-12)
    np.testing.assert_allclose(a.forecast(10), b.forecast(10), rtol=1e-14)


def test_end_to_end_ticks_to_forecast():
    """Synthetic 5-minute EURUSD path with known daily variance."""
    rng = np.random.default_rng(11)
    days = pd.bdate_range("2024-01-02", periods=80)
    sig_d = 0.006 * np.exp(0.3 * np.sin(np.arange(80) / 7.0))
    ticks, vals, x = [], [], 0.0
    for d, s in zip(days, sig_d, strict=True):
        # localise the wall-clock start, so windows stay 17:05-16:55 across the DST change
        start = (d - pd.Timedelta(days=1) + pd.Timedelta(hours=17, minutes=5)).tz_localize(
            "America/New_York")
        idx = pd.date_range(start, periods=287, freq="5min")
        x_day = x + np.cumsum(rng.normal(0, s / np.sqrt(287), 287))
        ticks.append(idx.tz_convert("UTC"))
        vals.append(x_day)
        x = x_day[-1]
    p = pd.Series(1.1 * np.exp(np.concatenate(vals)), index=ticks[0].append(ticks[1:]))
    rv = fx_daily_realized_variance(p.tz_convert("UTC"))
    assert len(rv) == 80
    np.testing.assert_allclose(np.sqrt(rv.mean()), np.sqrt(np.mean(sig_d**2)), rtol=0.05)
    res = fit_har(rv)
    assert res.forecast(5).shape == (5,)
