"""Data layer: deterministic synthetic FX generators (tests, examples) and
an import-guarded live loader (Frankfurter/ECB) that never runs in tests."""

from . import synthetic

__all__ = ["synthetic"]
