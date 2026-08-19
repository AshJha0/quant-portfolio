"""Data layer: deterministic synthetic generator + import-guarded live loader.

Per the portfolio conventions (``CONVENTIONS.md``), tests and examples use
only :mod:`eq_signal_backtest.data.synthetic`, which is seeded and fully
offline. :mod:`eq_signal_backtest.data.live` (yfinance) is import-guarded,
so importing this package never requires network access.
"""

from .synthetic import generate

__all__ = ["generate"]
