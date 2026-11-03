"""Helpers for the deprecated per-project ``data/live.py`` fetchers.

The eight projects that had their own fetchers (fx_options, fx_vol, fx_var,
fx_rates, fx_pairs, fx_credit, fx_port, fi_rates) now delegate here, so every
download goes through :func:`qp_data.download` (offline guard, retries,
backoff) and the same parsers as the store. Their old names keep working
until v4.0.0 (docs/VERSIONING.md) and warn when called.
"""

from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd

from . import fetch_frame

__all__ = ["REMOVAL", "deprecated", "frankfurter", "fred_series"]

REMOVAL = "v4.0.0"


def deprecated(old: str, new: str) -> None:
    """Emit the standard ``DeprecationWarning`` for a ``live.py`` function."""
    warnings.warn(
        f"{old} is deprecated and will be removed in {REMOVAL}; use {new} (qp_data).",
        DeprecationWarning,
        stacklevel=3,
    )


def _date(x: str | dt.date | None) -> dt.date | None:
    return None if x is None else pd.Timestamp(x).date()


def frankfurter(
    base: str,
    symbols: list[str] | tuple[str, ...],
    start: str | dt.date | None = None,
    end: str | dt.date | None = None,
    *,
    timeout: float = 30.0,
    allow_network: bool | None = True,
) -> pd.DataFrame:
    """ECB reference rates via Frankfurter: units of each symbol per 1 ``base``.

    ``start=None`` returns the latest fixing only.
    """
    key = f"frankfurter:{base.upper()}:{','.join(s.upper() for s in symbols)}"
    frame, _, _ = fetch_frame(
        key, _date(start), _date(end), allow_network=allow_network, timeout=timeout
    )
    return frame


def fred_series(
    series: str,
    start: str | dt.date | None = None,
    *,
    timeout: float = 30.0,
    allow_network: bool | None = True,
) -> pd.Series:
    """One FRED series through the keyless ``fredgraph.csv`` download."""
    frame, _, _ = fetch_frame(
        f"fred:{series}", _date(start), None, allow_network=allow_network, timeout=timeout
    )
    out = frame["value"].rename(series)
    out.index.name = None
    return out
