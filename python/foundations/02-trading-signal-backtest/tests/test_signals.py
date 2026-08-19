"""ma_crossover_signal: hand-computed crossover points and input validation."""

import pandas as pd
import pytest

from eq_signal_backtest.signals import ma_crossover_signal


def test_fast_must_be_shorter_than_slow():
    prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="fast window must be shorter"):
        ma_crossover_signal(prices, fast=5, slow=5)
    with pytest.raises(ValueError, match="fast window must be shorter"):
        ma_crossover_signal(prices, fast=6, slow=5)


def test_known_crossover_points_hand_computed():
    """A step-up-then-step-down price path with a hand-verified fast(2)/
    slow(4) crossover: bullish (1) while the fast MA leads the slow MA up
    through the plateau, flat (0) once the slow MA catches up -- computed
    by hand below, independent of the implementation."""
    prices = pd.Series(
        [10, 10, 10, 10, 20, 20, 20, 20, 10, 10, 10, 10],
        dtype=float,
        index=pd.bdate_range("2024-01-01", periods=12),
    )
    sig = ma_crossover_signal(prices, fast=2, slow=4)
    expected = pd.Series(
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float, index=prices.index
    )
    pd.testing.assert_series_equal(sig, expected)


def test_warmup_period_is_flat_not_nan():
    """The first slow-1 observations, where the slow MA is undefined, must
    be 0.0 (flat), never NaN -- NaN would silently propagate into
    run_backtest's position/cost arithmetic."""
    prices = pd.Series(
        range(1, 21), dtype=float, index=pd.bdate_range("2024-01-01", periods=20)
    )
    sig = ma_crossover_signal(prices, fast=3, slow=8)
    assert not sig.iloc[:7].isna().any()
    assert (sig.iloc[:7] == 0.0).all()


def test_signal_is_binary():
    prices = pd.Series(
        [100, 102, 101, 105, 103, 108, 107, 110, 109, 112],
        dtype=float,
        index=pd.bdate_range("2024-01-01", periods=10),
    )
    sig = ma_crossover_signal(prices, fast=2, slow=5)
    assert set(sig.unique().tolist()) <= {0.0, 1.0}


def test_persistent_uptrend_ends_bullish():
    prices = pd.Series(
        [100.0 * (1.01**i) for i in range(30)],
        index=pd.bdate_range("2024-01-01", periods=30),
    )
    sig = ma_crossover_signal(prices, fast=3, slow=10)
    assert sig.iloc[-1] == 1.0


def test_persistent_downtrend_ends_flat():
    prices = pd.Series(
        [100.0 * (0.99**i) for i in range(30)],
        index=pd.bdate_range("2024-01-01", periods=30),
    )
    sig = ma_crossover_signal(prices, fast=3, slow=10)
    assert sig.iloc[-1] == 0.0
