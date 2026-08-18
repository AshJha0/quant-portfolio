"""Ground-truth generator tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq_regime.data import make_gbm_panel, make_regime_panel, simulate_markov_chain


def test_shapes_and_alignment(panel3):
    assert panel3.prices.shape == (1501, 6)
    assert panel3.returns.shape == (1500, 6)
    assert panel3.states.shape == (1500,)
    assert (panel3.prices.iloc[0] == 100.0).all()
    # prices are the compounded returns
    rebuilt = 100.0 * np.exp(panel3.returns.cumsum())
    pd.testing.assert_frame_equal(panel3.prices.iloc[1:], rebuilt)


def test_bear_state_stylized_facts(panel3):
    """Bear state: negative mean, higher vol, higher cross-correlation."""
    rets = panel3.returns.to_numpy()
    states = panel3.states
    bull, bear = 0, panel3.n_states - 1
    r_bull, r_bear = rets[states == bull], rets[states == bear]
    assert r_bear.mean() < 0 < r_bull.mean()
    assert r_bear.std() > 2.0 * r_bull.std()
    corr_bull = np.corrcoef(r_bull.T)
    corr_bear = np.corrcoef(r_bear.T)
    iu = np.triu_indices(6, k=1)
    assert corr_bear[iu].mean() > corr_bull[iu].mean() + 0.3


def test_empirical_transition_matrix_close_to_truth(panel3):
    states = panel3.states
    k = panel3.n_states
    counts = np.zeros((k, k))
    for a, b in zip(states[:-1], states[1:]):
        counts[a, b] += 1
    emp = counts / counts.sum(axis=1, keepdims=True)
    assert np.abs(np.diag(emp) - np.diag(panel3.transition)).max() < 0.05


def test_seed_determinism():
    a = make_regime_panel(3, n_assets=4, n_days=200, seed=5)
    b = make_regime_panel(3, n_assets=4, n_days=200, seed=5)
    c = make_regime_panel(3, n_assets=4, n_days=200, seed=6)
    pd.testing.assert_frame_equal(a.prices, b.prices)
    assert not a.prices.equals(c.prices)


def test_gbm_panel_is_single_state(gbm_panel):
    assert gbm_panel.n_states == 1
    assert (gbm_panel.states == 0).all()
    assert gbm_panel.transition.shape == (1, 1)


def test_markov_chain_validation():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="rows must sum"):
        simulate_markov_chain(np.array([[0.5, 0.4], [0.1, 0.9]]), 10, rng)
    with pytest.raises(ValueError, match="square"):
        simulate_markov_chain(np.ones((2, 3)) / 3, 10, rng)
    with pytest.raises(ValueError, match="n_steps"):
        simulate_markov_chain(np.eye(2), 0, rng)


def test_generator_input_validation():
    with pytest.raises(ValueError, match="n_states"):
        make_regime_panel(n_states=4, n_days=100)
    with pytest.raises(ValueError, match="n_assets"):
        make_regime_panel(n_states=2, n_assets=1, n_days=100)
    with pytest.raises(ValueError, match="n_days"):
        make_regime_panel(n_states=2, n_days=5)
