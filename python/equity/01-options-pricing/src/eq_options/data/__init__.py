"""Synthetic (deterministic, offline) and optional live market data."""

from .synthetic import gbm_paths, skew_vol, synthetic_chain

__all__ = ["gbm_paths", "skew_vol", "synthetic_chain"]
