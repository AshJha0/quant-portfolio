"""Shared fixtures: small, seeded, session-scoped panels and fits."""

from __future__ import annotations

import numpy as np
import pytest

from eq_regime.data import make_gbm_panel, make_regime_panel
from eq_regime.features import build_features
from eq_regime.hmm import fit_hmm


@pytest.fixture(scope="session")
def panel3():
    """3-state regime panel, 6 assets, ~6y of days."""
    return make_regime_panel(n_states=3, n_assets=6, n_days=1500, seed=7)


@pytest.fixture(scope="session")
def panel2():
    """2-state regime panel."""
    return make_regime_panel(n_states=2, n_assets=6, n_days=1500, seed=3)


@pytest.fixture(scope="session")
def gbm_panel():
    """No-regime GBM null panel."""
    return make_gbm_panel(n_assets=6, n_days=1200, seed=11)


@pytest.fixture(scope="session")
def features3(panel3):
    """Point-in-time standardized feature table of the 3-state panel."""
    return build_features(panel3.prices)


@pytest.fixture(scope="session")
def hmm2_fit(panel2):
    """2-state HMM fitted on the 2-state panel's index returns (1-D)."""
    r = panel2.returns.mean(axis=1).to_numpy()
    return fit_hmm(r, 2, seed=0, n_init=2, max_iter=150)


@pytest.fixture(scope="session")
def index_returns2(panel2):
    return panel2.returns.mean(axis=1).to_numpy()


@pytest.fixture(scope="session")
def wellsep_2state():
    """Very well-separated 2-state 1-D HMM sample with known truth."""
    from eq_regime.data.synthetic import simulate_markov_chain

    rng = np.random.default_rng(42)
    transition = np.array([[0.97, 0.03], [0.05, 0.95]])
    n = 3000
    states = simulate_markov_chain(transition, n, rng)
    means = np.array([-2.0, 2.0])
    sigmas = np.array([0.7, 0.7])
    x = means[states] + sigmas[states] * rng.standard_normal(n)
    return x, states, transition, means, sigmas
