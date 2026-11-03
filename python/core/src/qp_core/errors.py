"""Typed errors shared by every quant-portfolio package and both bindings.

The idea: a caller who wants to treat "my inputs were bad" differently from
"the solver gave up" or "the data feed is broken" should not have to parse
message strings. Every library failure is a :class:`QPError`, and one of
three subclasses says which kind it is.

=========================  =======================================  ================
Class                      Raised when                              Also a
=========================  =======================================  ================
:class:`InputError`        an argument is outside the model's       ``ValueError``
                           domain (negative vol, ``T <= 0``, NaN,
                           wrong shape, unknown option type)
:class:`ConvergenceError`  a numerical method failed on input that  ``RuntimeError``,
                           passed validation (implied vol did not   ``ValueError``
                           converge, Brent lost its bracket, a
                           covariance stayed indefinite past the
                           Cholesky jitter cap)
:class:`DataError`         market or reference data is missing,     ``RuntimeError``
                           stale, malformed or fails a schema check
=========================  =======================================  ================

Each subclass also derives from the built-in exception the Python packages
raised for that kind of failure before this module existed (``ValueError``
for bad arguments, ``RuntimeError`` for failed solvers and empty data
downloads), so existing ``except`` clauses keep working when a package
switches to these classes (PEP 387). ``ConvergenceError`` is also a
``ValueError`` because the bindings and several kernels raised
``ValueError`` for solver failures; dropping that base would break
callers silently. New code should catch the
``qp_core.errors`` class.

The C++ (``qp_cpp``) and Rust (``qp_rust``) bindings raise these same
classes when ``qp_core`` is importable, and fall back to ``ValueError``
when it is not; see ``docs/VERSIONING.md``.

Library code raises; it does not log and re-raise. Logging happens once,
at the application boundary (:mod:`qp_core.logconfig`).

>>> from qp_core.errors import InputError, QPError
>>> try:
...     raise InputError("sigma must be > 0, got -0.2")
... except ValueError as exc:
...     isinstance(exc, QPError)
True
"""

from __future__ import annotations

__all__ = ["ConvergenceError", "DataError", "InputError", "QPError"]


class QPError(Exception):
    """Base class of every error the quant-portfolio libraries raise on purpose."""


class InputError(QPError, ValueError):
    """An argument is outside the domain the model or kernel accepts."""


class ConvergenceError(QPError, RuntimeError, ValueError):
    """A numerical method failed on input that passed validation.

    Typical causes: a root finder that did not meet its tolerance, a lost
    bracket, or a covariance matrix that is still indefinite after the
    bounded diagonal jitter. The inputs were well formed; the method could
    not produce an answer from them.
    """


class DataError(QPError, RuntimeError):
    """Market or reference data is missing, stale, malformed or inconsistent."""
