"""eq_signal_backtest.data.synthetic.generate: determinism and shape."""

import numpy as np
import pandas as pd

from eq_signal_backtest.data.synthetic import generate


def test_deterministic_given_same_seed():
    a = generate(n_days=300, seed=42)
    b = generate(n_days=300, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = generate(n_days=300, seed=1)
    b = generate(n_days=300, seed=2)
    assert not a["Adj Close"].equals(b["Adj Close"])


def test_shape_and_columns():
    df = generate(n_days=500, seed=7)
    assert list(df.columns) == ["Date", "Adj Close"]
    assert len(df) == 500


def test_prices_strictly_positive():
    df = generate(n_days=1000, seed=9)
    assert (df["Adj Close"] > 0).all()


def test_dates_are_ascending_business_days():
    df = generate(n_days=50, seed=3)
    assert df["Date"].is_monotonic_increasing
    assert df["Date"].dt.dayofweek.isin([0, 1, 2, 3, 4]).all()


def test_generator_accepts_explicit_rng():
    rng = np.random.default_rng(123)
    a = generate(n_days=100, seed=rng)
    rng2 = np.random.default_rng(123)
    b = generate(n_days=100, seed=rng2)
    pd.testing.assert_frame_equal(a, b)


def test_default_args_reproduce_bundled_series_length():
    df = generate()
    assert len(df) == 2520
