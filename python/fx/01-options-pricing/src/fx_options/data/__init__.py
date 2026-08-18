"""Data providers: deterministic synthetic generators (tests) and an
optional live ECB/Frankfurter loader (never used by tests)."""

from .synthetic import gbm_fx_paths, synthetic_vol_quotes

__all__ = ["gbm_fx_paths", "synthetic_vol_quotes"]
