"""Shared session-scoped fixtures: expensive simulate-and-fit pairs reused
across test modules (keeps the suite well under the runtime budget)."""

import numpy as np
import pytest

from eq_vol.data import synthetic as syn
from eq_vol.egarch import fit_egarch
from eq_vol.garch import fit_garch
from eq_vol.gjr import fit_gjr
from true_params import EGARCH_TRUE, GARCH_TRUE, GJR_TRUE  # noqa: F401


@pytest.fixture(scope="session")
def garch_sim():
    return syn.simulate_garch(20_000, seed=1, **GARCH_TRUE)


@pytest.fixture(scope="session")
def garch_fit(garch_sim):
    return fit_garch(garch_sim.returns)


@pytest.fixture(scope="session")
def gjr_sim():
    return syn.simulate_gjr(15_000, seed=2, **GJR_TRUE)


@pytest.fixture(scope="session")
def gjr_fit(gjr_sim):
    return fit_gjr(gjr_sim.returns)


@pytest.fixture(scope="session")
def egarch_sim():
    return syn.simulate_egarch(8_000, seed=3, **EGARCH_TRUE)


@pytest.fixture(scope="session")
def egarch_fit(egarch_sim):
    return fit_egarch(egarch_sim.returns)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(12345)
