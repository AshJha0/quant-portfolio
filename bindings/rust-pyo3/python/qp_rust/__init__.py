"""PyO3 bindings over the quant-portfolio Rust engines.

Functions are prefixed by engine: ``eq_`` (equity options and equity VaR)
and ``fx_`` (FX options and FX VaR). Scalar functions take floats; ``*_batch``
functions take equal-length 1-D float64 arrays (used in place when
C-contiguous) and return new arrays. Engine errors raise
``qp_core.errors.InputError`` (a ``ValueError``) for bad arguments and
``qp_core.errors.ConvergenceError`` (a ``RuntimeError`` and a ``ValueError``) when a numerical
method fails on well-formed input, with the engine's message. Without
``qp_core`` installed both fall back to ``ValueError``.
"""

from __future__ import annotations

try:
    from ._core import *  # noqa: F403
    from ._core import __doc__ as _core_doc  # noqa: F401

    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - only when the build is broken
    _IMPORT_ERROR = exc

__version__ = "3.0.0.dev0"


def available() -> bool:
    """True when the compiled extension imported."""
    return _IMPORT_ERROR is None
