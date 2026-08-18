"""Synthetic chain generator: determinism, ground truth, realism constraints."""

from __future__ import annotations

import numpy as np
import pytest

import eq_surface as es
from eq_surface.data import DEFAULT_TRUE_HESTON, generate_chain
from eq_surface.data.synthetic import default_svi_slices
from eq_surface.heston import heston_call_gl
from eq_surface.smile import check_butterfly, svi_total_variance


def test_deterministic_given_seed():
    a = generate_chain(mode="heston", seed=5, bid_ask=True)
    b = generate_chain(mode="heston", seed=5, bid_ask=True)
    assert a.df.equals(b.df)


def test_different_seed_changes_noisy_quotes():
    a = generate_chain(mode="heston", seed=5, bid_ask=True)
    b = generate_chain(mode="heston", seed=6, bid_ask=True)
    assert not np.allclose(a.df.call_mid.values, b.df.call_mid.values)


def test_heston_mode_mids_match_fourier_prices():
    chain = generate_chain(mode="heston", seed=0)
    for T in sorted(chain.df.expiry.unique()):
        sl = chain.slice(T)
        prices = np.asarray(heston_call_gl(chain.spot, sl.strike.values, float(T),
                                           chain.rate, chain.div_yield, chain.true_heston))
        assert np.allclose(sl.call_mid.values, prices, atol=1e-10)


def test_svi_mode_ivs_match_slices():
    chain = generate_chain(mode="svi", seed=0)
    T = 0.5
    sl = chain.slice(T)
    ivs = es.implied_vol_vector(sl.call_mid.values, chain.spot, sl.strike.values,
                                T, chain.rate, chain.div_yield)
    w_true = np.asarray(svi_total_variance(sl.log_moneyness.values, chain.true_svi[T]))
    assert np.allclose(ivs, np.sqrt(w_true / T), atol=1e-7)


def test_ground_truth_attached_per_mode():
    h = generate_chain(mode="heston", seed=0)
    assert h.true_heston == DEFAULT_TRUE_HESTON and h.true_svi is None
    s = generate_chain(mode="svi", seed=0)
    assert s.true_heston is None
    assert np.allclose(sorted(s.true_svi), sorted(s.df.expiry.unique()))


def test_strike_coverage_narrows_for_short_expiries():
    chain = generate_chain(mode="heston", seed=0)
    widths = {}
    for T in sorted(chain.df.expiry.unique()):
        sl = chain.slice(T)
        widths[T] = sl.log_moneyness.max() - sl.log_moneyness.min()
        assert len(sl) >= 5  # always enough quotes for an SVI fit
    Ts = sorted(widths)
    assert widths[Ts[0]] < widths[Ts[-1]]  # weekly narrower than 2y


def test_moneyness_within_configured_range():
    chain = generate_chain(mode="heston", seed=0, moneyness_lo=0.5, moneyness_hi=1.5)
    # strikes are rounded to 4dp, so allow the corresponding moneyness slack
    assert chain.df.moneyness.min() >= 0.5 - 1e-5
    assert chain.df.moneyness.max() <= 1.5 + 1e-5


def test_bid_ask_ordering_and_positivity():
    chain = generate_chain(mode="heston", seed=3, bid_ask=True)
    df = chain.df
    assert np.all(df.call_bid.values >= 0.0)
    assert np.all(df.call_bid.values <= df.call_mid.values + 1e-12)
    assert np.all(df.call_mid.values <= df.call_ask.values + 1e-12)


def test_default_svi_slices_are_arbitrage_free():
    """Generator ground truth is itself butterfly- and calendar-free."""
    Ts = np.array([1 / 52, 1 / 12, 0.25, 0.5, 1.0, 2.0])
    slices = default_svi_slices(Ts)
    k_grid = np.linspace(-0.7, 0.7, 141)
    prev_w = None
    for T in Ts:
        p = slices[float(T)]
        ok, min_g, _ = check_butterfly(p, -0.7, 0.7)
        assert ok, (T, min_g)
        w = np.asarray(svi_total_variance(k_grid, p))
        if prev_w is not None:
            assert np.all(w >= prev_w - 1e-12), T  # calendar-free
        prev_w = w


def test_slice_accessor_and_invalid_inputs():
    chain = generate_chain(mode="heston", seed=0)
    with pytest.raises(KeyError, match="no expiry"):
        chain.slice(0.123)
    with pytest.raises(ValueError, match="mode"):
        generate_chain(mode="local_vol")
    with pytest.raises(ValueError, match="moneyness"):
        generate_chain(moneyness_lo=-0.5)
    with pytest.raises(ValueError, match="expiries"):
        generate_chain(expiries=(0.0, 1.0))
