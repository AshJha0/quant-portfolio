"""Data sources: deterministic synthetic generators (used by tests) and an
optional import-guarded live FRED loader (never used by tests)."""

from .synthetic import CURVE_VARIANTS, market_quotes, sample_portfolio

__all__ = ["CURVE_VARIANTS", "market_quotes", "sample_portfolio"]
