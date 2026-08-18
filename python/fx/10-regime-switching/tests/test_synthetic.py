"""Tests for the RORO synthetic generator: known params must be recoverable."""

import numpy as np
import pandas as pd
import pytest

from fx_regime import (
    CURRENCIES,
    EM,
    G10_CARRY,
    HAVENS,
    MEAN_DEPOSIT_RATES,
    TRANSITION_2,
    TRANSITION_3,
    generate_null_gbm_panel,
    generate_roro_panel,
    simulate_markov_chain,
    state_correlation,
    state_moments,
    stationary_distribution,
)


def test_shapes_and_index():
    panel = generate_roro_panel(300, n_states=2, seed=0)
    assert panel.returns.shape == (300, len(CURRENCIES))
    assert panel.deposit_rates.shape == (300, len(CURRENCIES) + 1)
    assert len(panel.states) == 300
    assert (panel.returns.index == panel.deposit_rates.index).all()
    assert panel.transition.shape == (2, 2)


def test_seed_determinism():
    a = generate_roro_panel(200, n_states=3, seed=5)
    b = generate_roro_panel(200, n_states=3, seed=5)
    pd.testing.assert_frame_equal(a.returns, b.returns)
    assert (a.states == b.states).all()
    c = generate_roro_panel(200, n_states=3, seed=6)
    assert not np.allclose(a.returns.to_numpy(), c.returns.to_numpy())


def test_state_frequencies_near_stationary():
    panel = generate_roro_panel(6000, n_states=2, seed=1)
    freq = np.bincount(panel.states, minlength=2) / len(panel.states)
    pi = stationary_distribution(TRANSITION_2)
    assert np.abs(freq - pi).max() < 0.08


def test_per_state_moments_match_spec():
    panel = generate_roro_panel(8000, n_states=2, seed=2)
    for s, name in enumerate(panel.state_names):
        mask = panel.states == s
        sample = panel.returns.to_numpy()[mask]
        mu_true, cov_true = state_moments(name)
        mu_hat = sample.mean(axis=0) * 252
        vol_hat = sample.std(axis=0, ddof=1) * np.sqrt(252)
        vol_true = np.sqrt(np.diag(cov_true))
        # drift tolerance ~3 standard errors of an annualised-mean
        # estimate on the risk_off subsample (vol 30% over ~5 years)
        assert np.abs(mu_hat - mu_true).max() < 0.12
        assert np.abs(vol_hat / vol_true - 1.0).max() < 0.10


def test_risk_off_is_high_vol_high_corr():
    panel = generate_roro_panel(6000, n_states=2, seed=3)
    rets = panel.returns[list(G10_CARRY)].to_numpy()
    on, off = panel.states == 0, panel.states == 1
    vol_on = rets[on].std(axis=0).mean()
    vol_off = rets[off].std(axis=0).mean()
    assert vol_off > 1.5 * vol_on
    corr_on = np.corrcoef(rets[on].T)
    corr_off = np.corrcoef(rets[off].T)
    iu = np.triu_indices_from(corr_on, k=1)
    assert corr_off[iu].mean() > corr_on[iu].mean() + 0.2


def test_havens_rally_in_risk_off_and_fall_in_squeeze():
    panel = generate_roro_panel(8000, n_states=3, seed=4)
    hav = panel.returns[list(HAVENS)].mean(axis=1).to_numpy()
    assert hav[panel.states == 1].mean() > 0  # risk_off: haven bid
    assert hav[panel.states == 2].mean() < 0  # squeeze: USD beats havens
    carry = panel.returns[list(G10_CARRY) + list(EM)].mean(axis=1).to_numpy()
    assert carry[panel.states == 1].mean() < 0
    assert carry[panel.states == 2].mean() < 0


def test_deposit_rates_persistent_differentials():
    panel = generate_roro_panel(2000, n_states=2, seed=5)
    r = panel.deposit_rates
    # differentials keep their sign the whole sample (persistence)
    assert ((r["NZD"] - r["USD"]) > 0).all()
    assert ((r["JPY"] - r["USD"]) < 0).all()
    assert np.isclose(r["AUD"].mean(), MEAN_DEPOSIT_RATES["AUD"], atol=0.01)


def test_planted_flip_forces_state():
    panel = generate_roro_panel(
        500, n_states=3, seed=6, plant_flip_at=300, plant_flip_len=40,
        plant_flip_state=1,
    )
    assert (panel.states[300:340] == 1).all()


def test_transition_matrices_are_stochastic():
    for P in (TRANSITION_2, TRANSITION_3):
        assert np.allclose(P.sum(axis=1), 1.0)
        assert (P >= 0).all()


def test_markov_chain_transition_recovery():
    rng = np.random.default_rng(0)
    P = np.array([[0.9, 0.1], [0.3, 0.7]])
    states = simulate_markov_chain(P, 20000, rng)
    counts = np.zeros((2, 2))
    for a, b in zip(states[:-1], states[1:]):
        counts[a, b] += 1
    est = counts / counts.sum(axis=1, keepdims=True)
    assert np.abs(est - P).max() < 0.02


def test_null_gbm_has_no_regime_structure():
    panel = generate_null_gbm_panel(500, seed=0)
    assert (panel.states == 0).all()
    assert panel.transition.shape == (1, 1)
    # zero drift by construction
    assert abs(panel.returns.mean().mean()) * 252 < 0.05


def test_state_correlation_positive_definite():
    for name in ("risk_on", "risk_off", "usd_squeeze"):
        corr = state_correlation(name)
        w = np.linalg.eigvalsh(corr)
        assert w.min() > 0
        assert np.allclose(np.diag(corr), 1.0)


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        generate_roro_panel(10)  # too short
    with pytest.raises(ValueError):
        generate_roro_panel(100, n_states=4)
    with pytest.raises(ValueError):
        generate_roro_panel(100, n_states=2, plant_flip_at=500)
    with pytest.raises(ValueError):
        simulate_markov_chain(
            np.array([[0.5, 0.4], [0.5, 0.5]]), 10, np.random.default_rng(0)
        )
