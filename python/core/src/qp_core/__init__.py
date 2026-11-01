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
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
