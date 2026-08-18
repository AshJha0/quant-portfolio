"""Shared fixtures: synthetic markets, surfaces, and calibrations.

Session scope so the (cheap but not free) surface builds and Heston
calibrations run once for the whole suite.  Everything is seeded and
offline.
"""

from __future__ import annotations

import pytest

from fx_surface import HestonParams, calibrate_heston
from fx_surface.data import (
    calibration_slices,
    em_high_vol_market,
    eurusd_market,
    market_from_heston,
    usdjpy_market,
)
from fx_surface.surface import build_surface

TRUE_HESTON = HestonParams(v0=0.0064, kappa=1.8, theta=0.008, xi=0.45, rho=-0.35)


@pytest.fixture(scope="session")
def eurusd():
    return eurusd_market()


@pytest.fixture(scope="session")
def usdjpy():
    return usdjpy_market()


@pytest.fixture(scope="session")
def em_market():
    return em_high_vol_market()


@pytest.fixture(scope="session")
def eurusd_surface(eurusd):
    return build_surface(eurusd, smile_model="svi")


@pytest.fixture(scope="session")
def usdjpy_surface(usdjpy):
    return build_surface(usdjpy, smile_model="svi")


@pytest.fixture(scope="session")
def ground_truth_market():
    return market_from_heston(TRUE_HESTON)


@pytest.fixture(scope="session")
def ground_truth_calibration(ground_truth_market):
    m = ground_truth_market
    return calibrate_heston(m.S, calibration_slices(m))


@pytest.fixture(scope="session")
def eurusd_calibration(eurusd):
    return calibrate_heston(eurusd.S, calibration_slices(eurusd))


@pytest.fixture(scope="session")
def usdjpy_calibration(usdjpy):
    return calibrate_heston(usdjpy.S, calibration_slices(usdjpy))
