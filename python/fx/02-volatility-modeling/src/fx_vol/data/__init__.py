"""Data subpackage: deterministic synthetic generators (test data source) and
an import-guarded live ECB/Frankfurter loader (never used by tests)."""

from . import synthetic  # noqa: F401

__all__ = ["synthetic"]
