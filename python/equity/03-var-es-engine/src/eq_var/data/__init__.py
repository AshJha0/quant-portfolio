"""Data generators: deterministic synthetic (tests) and guarded live loaders."""

from .synthetic import (
    default_covariance,
    demo_covariance,
    demo_portfolio,
    simulate_garch_returns,
    simulate_returns,
)

__all__ = [
    "default_covariance",
    "demo_covariance",
    "demo_portfolio",
    "simulate_garch_returns",
    "simulate_returns",
]
