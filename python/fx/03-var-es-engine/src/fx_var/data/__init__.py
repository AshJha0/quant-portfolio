"""Data layer: deterministic synthetic FX generators (tests, examples) and
an import-guarded live loader (Frankfurter/ECB) that never runs in tests."""

from . import synthetic  # noqa: F401

__all__ = ["synthetic"]
