"""Synthetic generators: determinism, planted alpha, panel sanity."""

import numpy as np
import pandas as pd
import pytest

from fx_algo import (
    EURUSD,
    build_bars,
    generate_daily_panel,
    generate_ticks,
    information_coefficient,
    momentum,
)


def test_ticks_shape_and_columns():
    t = generate_ticks(n_days=2, ticks_per_hour=12, seed=0)
    assert list(t.columns) == ["time_hours", "mid", "bid", "ask"]
    assert len(t) == 2 * 24 * 12


def test_ticks_seeded_reproducible():
    a = generate_ticks(n_days=2, seed=123)
    b = generate_ticks(n_days=2, seed=123)
    pd.testing.assert_frame_equal(a, b)
    c = generate_ticks(n_days=2, seed=124)
    assert not np.allclose(a["mid"], c["mid"])


def test_ticks_bid_ask_straddle_mid_with_session_spread():
    t = generate_ticks(n_days=1, seed=0)
    pip = EURUSD.pip_size
    half = 0.5 * EURUSD.spread_pips_at(t["time_hours"].to_numpy()) * pip
    assert np.allclose(t["ask"] - t["mid"], half)
    assert np.allclose(t["mid"] - t["bid"], half)


def test_planted_alpha_ic_positive_and_significant():
    ticks = generate_ticks(n_days=30, phi=0.25, seed=0)
    bars = build_bars(ticks, 1.0)
    fwd = bars["close"].pct_change().shift(-1)
    ic, t = information_coefficient(momentum(bars, 1), fwd)
    assert ic > 0.10
    assert t > 2.0


def test_noise_ic_insignificant():
    for seed in (1, 2, 3):
        ticks = generate_ticks(n_days=30, phi=0.0, seed=seed)
        bars = build_bars(ticks, 1.0)
        fwd = bars["close"].pct_change().shift(-1)
        ic, t = information_coefficient(momentum(bars, 1), fwd)
        assert abs(t) < 2.5
        assert abs(ic) < 0.1


def test_session_vol_profile_realised():
    ticks = generate_ticks(n_days=40, phi=0.0, seed=7)
    bars = build_bars(ticks, 1.0)
    r = bars["close"].diff() / EURUSD.pip_size  # pips per hour
    overlap = r[(bars["hour"] > 12) & (bars["hour"] <= 17)].std()
    asia = r[(bars["hour"] > 0) & (bars["hour"] <= 7)].std()
    # configured: overlap 2.2 vs asia 0.9 pips/sqrt-min
    assert overlap > 1.5 * asia


def test_invalid_phi_raises():
    with pytest.raises(ValueError):
        generate_ticks(n_days=1, phi=1.0)
    with pytest.raises(ValueError):
        generate_ticks(n_days=0)


def test_daily_panel_shape_and_carry_identity():
    p = generate_daily_panel(n_days=15, r_base=0.03, r_quote=0.05, seed=1)
    assert len(p) == 15
    assert np.allclose(p["carry"], p["r_base"] - p["r_quote"])
    assert (p["spot"] > 0).all()


def test_daily_panel_seeded():
    a = generate_daily_panel(5, seed=9)
    b = generate_daily_panel(5, seed=9)
    pd.testing.assert_frame_equal(a, b)
