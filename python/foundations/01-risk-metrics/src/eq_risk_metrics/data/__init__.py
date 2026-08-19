"""Data layer: deterministic synthetic generator + import-guarded live loader.

Per the portfolio conventions (``CONVENTIONS.md``): tests and examples
depend only on :mod:`eq_risk_metrics.data.synthetic`, which is seeded and
fully offline. The optional live loader (:mod:`eq_risk_metrics.data.live`,
yfinance) is import-guarded so importing this package never requires
network access or the ``[live]`` extra.
"""

from .synthetic import generate

__all__ = ["generate"]
