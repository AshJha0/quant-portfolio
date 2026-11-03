"""Shared code for the quant-portfolio Python projects.

Modules
-------
stats       Sharpe ratio, Lo / Newey-West Sharpe standard error, deflated
            Sharpe ratio, PBO by CSCV, block bootstrap, seed distributions.
execution   ExecutionConfig and the t+1 fill rule.
holidays    Rule-based settlement calendars 2000-2099, roll conventions and
            FX spot dates (calendars is a deprecated shim over it).
daycount    Day count fractions, month arithmetic, coupon schedules.
validation  Finite-input guards.
metrics     Drawdown, downside deviation, annualisation.
normal      Standard normal cdf and pdf for scalar floats.
black_scholes  Black-Scholes in cost-of-carry form (b = r - q = r_d - r_f):
            prices, analytic Greeks and NumPy array forms shared by the
            equity and FX option packages.
iv_core     Implied volatility from normalised Black (safeguarded Halley).
errors      QPError and its subclasses InputError, ConvergenceError and
            DataError, raised by every package and both bindings.
logconfig   JSON logging with run context for entry points only.

The version is the repository version in the top-level ``VERSION`` file
(docs/VERSIONING.md); a test fails if they differ.

Library code never configures logging handlers. ``qp_core`` attaches one
``NullHandler`` to the ``qp_core`` logger so an unconfigured program stays
silent; entry points call ``qp_core.logconfig.configure_logging``.
"""

from __future__ import annotations

import logging

__version__ = "3.0.0.dev0"

__all__ = ["__version__"]

logging.getLogger(__name__).addHandler(logging.NullHandler())
