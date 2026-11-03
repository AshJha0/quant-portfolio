"""Shared code for the quant-portfolio Python projects.

Modules
-------
stats       Sharpe ratio, Lo / Newey-West Sharpe standard error, deflated
            Sharpe ratio, PBO by CSCV, block bootstrap, seed distributions.
execution   ExecutionConfig and the t+1 fill rule.
calendars   Partial business-day calendars and roll conventions.
daycount    Day count fractions, month arithmetic, coupon schedules.
validation  Finite-input guards.
metrics     Drawdown, downside deviation, annualisation.
normal      Standard normal cdf and pdf for scalar floats.
black_scholes  Black-Scholes in cost-of-carry form (b = r - q = r_d - r_f):
            prices, analytic Greeks and NumPy array forms shared by the
            equity and FX option packages.
iv_core     Implied volatility from normalised Black (safeguarded Halley).
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
