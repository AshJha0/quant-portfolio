"""Data generators (synthetic, deterministic) and optional live loaders."""

from .synthetic import (
    PairTruth,
    PanelTruth,
    business_index,
    cointegrated_pair,
    correlated_random_walks,
    make_rng,
    mixed_panel,
    regime_break_pair,
    simulate_ou,
)

__all__ = [
    "PairTruth",
    "PanelTruth",
    "business_index",
    "cointegrated_pair",
    "correlated_random_walks",
    "make_rng",
    "mixed_panel",
    "regime_break_pair",
    "simulate_ou",
]
