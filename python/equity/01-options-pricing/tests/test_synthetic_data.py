"""Synthetic data generators: determinism, GBM moments, chain arbitrage sanity."""

import math

import numpy as np
import pytest

from eq_options import bs_price, implied_vol
from eq_options.data import gbm_paths, skew_vol, synthetic_chain


def test_gbm_paths_shape_and_start() -> None:
    paths = gbm_paths(100.0, 0.05, 0.2, 1.0, 12, 50, seed=0)
    assert paths.shape == (50, 13)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0)


def test_gbm_paths_deterministic_given_seed() -> None:
    a = gbm_paths(100.0, 0.05, 0.2, 1.0, 52, 20, seed=123)
    b = gbm_paths(100.0, 0.05, 0.2, 1.0, 52, 20, seed=123)
    c = gbm_paths(100.0, 0.05, 0.2, 1.0, 52, 20, seed=124)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_gbm_terminal_moments_match_lognormal() -> None:
    S0, mu, sigma, T = 100.0, 0.07, 0.25, 1.0
    paths = gbm_paths(S0, mu, sigma, T, 1, 400_000, seed=42)
    s_t = paths[:, -1]
    assert float(np.mean(s_t)) == pytest.approx(S0 * math.exp(mu * T), rel=5e-3)
    log_ret = np.log(s_t / S0)
    assert float(np.mean(log_ret)) == pytest.approx(
        (mu - 0.5 * sigma**2) * T, abs=3e-3
    )
    assert float(np.std(log_ret)) == pytest.approx(sigma * math.sqrt(T), rel=1e-2)


def test_gbm_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        gbm_paths(-1.0, 0.05, 0.2, 1.0, 10, 10)
    with pytest.raises(ValueError):
        gbm_paths(100.0, 0.05, 0.2, 1.0, 0, 10)


def test_skew_vol_shape_and_floor() -> None:
    """Negative skew: low strikes richer than high strikes; floored > 0."""
    F, T = 100.0, 0.5
    low = skew_vol(70.0, F, T)
    atm = skew_vol(100.0, F, T)
    high = skew_vol(130.0, F, T)
    assert low > atm  # puts rich (classic equity skew)
    assert skew_vol(1e6, F, T) >= 0.05  # floor active far OTM


def test_synthetic_chain_prices_consistent_and_iv_round_trips() -> None:
    chain = synthetic_chain(S0=100.0, r=0.03, q=0.01)
    assert {"expiry", "strike", "type", "iv", "price", "forward"} <= set(chain.columns)
    assert (chain["iv"] > 0).all()
    assert (chain["price"] >= 0).all()
    # every quote round-trips through implied vol at 1e-8
    sample = chain.sample(n=12, random_state=0)
    for _, row in sample.iterrows():
        recovered = implied_vol(
            row["price"], 100.0, row["strike"], row["expiry"], 0.03, 0.01, row["type"]
        )
        assert recovered == pytest.approx(row["iv"], abs=1e-8)


def test_synthetic_chain_parity_within_expiry() -> None:
    chain = synthetic_chain(S0=100.0, r=0.03, q=0.01, expiries=(0.5,))
    calls = chain[chain["type"] == "call"].set_index("strike")["price"]
    puts = chain[chain["type"] == "put"].set_index("strike")["price"]
    for K in calls.index:
        expected = 100.0 * math.exp(-0.01 * 0.5) - K * math.exp(-0.03 * 0.5)
        assert calls[K] - puts[K] == pytest.approx(expected, abs=1e-10)


def test_synthetic_chain_prices_match_bs_at_quoted_iv() -> None:
    chain = synthetic_chain(S0=100.0, r=0.02, q=0.0, expiries=(0.25,), n_strikes=5)
    for _, row in chain.iterrows():
        assert row["price"] == pytest.approx(
            bs_price(100.0, row["strike"], 0.25, 0.02, row["iv"], 0.0, row["type"]),
            abs=1e-12,
        )
