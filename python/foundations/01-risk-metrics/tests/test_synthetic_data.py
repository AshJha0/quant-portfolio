"""Tests for eq_risk_metrics.data.synthetic: deterministic offline data gen."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq_risk_metrics.data import generate
from eq_risk_metrics.data.live import HAS_YFINANCE


def test_generate_is_deterministic_given_same_seed() -> None:
    df1 = generate(n_days=252, seed=42)
    df2 = generate(n_days=252, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_different_seeds_differ() -> None:
    df1 = generate(n_days=252, seed=1)
    df2 = generate(n_days=252, seed=2)
    assert not df1["Adj Close"].equals(df2["Adj Close"])


def test_generate_shape_and_columns() -> None:
    df = generate(n_days=100, seed=0)
    assert list(df.columns) == ["Date", "Adj Close"]
    assert len(df) == 100
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])


def test_generate_prices_positive_and_sorted_by_date() -> None:
    df = generate(n_days=500, seed=5)
    assert (df["Adj Close"] > 0).all()
    assert df["Date"].is_monotonic_increasing


def test_generate_accepts_a_numpy_generator_directly() -> None:
    rng = np.random.default_rng(99)
    df = generate(n_days=50, seed=rng)
    assert len(df) == 50


def test_generate_invalid_n_days_raises() -> None:
    with pytest.raises(ValueError):
        generate(n_days=0)


def test_generate_invalid_start_price_raises() -> None:
    with pytest.raises(ValueError):
        generate(n_days=10, start_price=0.0)


def test_generate_reproduces_stylised_facts_fat_tails() -> None:
    """The whole point of the two-regime + Student-t generator: the
    resulting daily returns should show excess kurtosis, like real
    equity returns (this is what justifies using it for demoing VaR
    methodology differences)."""
    from scipy import stats

    df = generate(n_days=2520, seed=2)
    prices = df.set_index("Date")["Adj Close"]
    returns = prices.pct_change().dropna()
    assert stats.kurtosis(returns) > 0.5


def test_no_network_access_required_to_import_live_module() -> None:
    """Importing eq_risk_metrics.data.live must never require yfinance or
    network access -- HAS_YFINANCE just reports whether the optional
    extra is installed."""
    assert isinstance(HAS_YFINANCE, bool)
