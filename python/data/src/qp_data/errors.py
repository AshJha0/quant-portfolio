"""Exceptions raised by :mod:`qp_data`."""

from __future__ import annotations

__all__ = [
    "DataError",
    "FetchError",
    "NetworkDisabled",
    "NoVintage",
    "SchemaError",
    "SourceNotPermitted",
    "UnknownReleaseRule",
]


class DataError(RuntimeError):
    """Base class for every data-layer failure."""


class NetworkDisabled(DataError):
    """Network access was not opted into (the offline default)."""


class FetchError(DataError):
    """A download failed after every retry, or failed with a permanent error."""


class SchemaError(DataError, ValueError):
    """A response did not have the layout the parser expects."""


class SourceNotPermitted(DataError):
    """The source needs a login, a key, payment details or terms the owner has not accepted."""


class UnknownReleaseRule(DataError, KeyError):
    """No publication-lag rule is known for a series, so its release dates cannot be checked."""


class NoVintage(DataError):
    """No stored vintage was retrieved on or before the requested as-of time."""
