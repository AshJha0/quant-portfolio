"""Convenience re-exports of the seeded synthetic simulators.

The implementations live in :mod:`eq_vol.data.synthetic` (per CONVENTIONS.md,
tests use deterministic synthetic generators under ``data/``); this module
exists so ``from eq_vol import simulate`` reads naturally in scripts.
"""

from .data.synthetic import (  # noqa: F401
    SimulatedSeries,
    simulate_crisis,
    simulate_egarch,
    simulate_garch,
    simulate_gbm_ohlc,
    simulate_gbm_returns,
    simulate_gjr,
)

__all__ = [
    "SimulatedSeries",
    "simulate_crisis",
    "simulate_egarch",
    "simulate_garch",
    "simulate_gbm_ohlc",
    "simulate_gbm_returns",
    "simulate_gjr",
]
