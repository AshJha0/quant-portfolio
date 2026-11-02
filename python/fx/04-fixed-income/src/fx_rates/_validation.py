"""Shared non-finite input rejection for :mod:`fx_rates`.

Why this module exists
----------------------
Most validation in this package is written as an inequality::

    if spot <= 0.0:
        raise ValueError(...)

Every comparison against NaN is ``False``, so that guard *accepts* NaN and
the NaN then flows through the curve, the forward, the mark-to-market and
the risk report without a single exception being raised.  In a rates/FX
context that is worse than a crash: a NaN forward point or a NaN DV01 does
not breach a limit, does not fail a reconciliation threshold and reads like
a display bug rather than a pricing failure.  The CIP arbitrage detector was
the sharpest example, with a NaN quote every comparison in the detector
returned ``False`` and it reported a confident "no arbitrage".

Call :func:`require_finite` *before* the inequality guards so NaN and Inf
raise a named, informative ``ValueError``. The guard lives in
:mod:`qp_core.validation`.
"""

from __future__ import annotations

from qp_core.validation import require_finite

__all__ = ["require_finite"]
