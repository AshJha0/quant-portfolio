"""Stress scenarios (W14c). See :mod:`qp_core.scenarios.historical`."""

from __future__ import annotations

from .historical import (
    DESCRIPTIONS,
    SERIES,
    SOURCE_SHA256,
    HistoricalScenario,
    SeriesSpec,
    apply_to_quotes,
    apply_to_snapshot,
    reverse_stress_scale,
    scenario,
    scenario_names,
)

__all__ = [
    "DESCRIPTIONS",
    "SERIES",
    "SOURCE_SHA256",
    "HistoricalScenario",
    "SeriesSpec",
    "apply_to_quotes",
    "apply_to_snapshot",
    "reverse_stress_scale",
    "scenario",
    "scenario_names",
]
