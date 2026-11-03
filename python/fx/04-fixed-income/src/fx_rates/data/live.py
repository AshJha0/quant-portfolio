"""Guarded live market-data loaders (Frankfurter FX, FRED rates).

Deprecated: use ``qp_data.fetch`` / ``qp_data.get`` (for example
``"frankfurter:EUR:USD"``, ``"fred:DGS5"``, ``"nyfed:SOFR"``). Both loaders
now delegate to qp_data and are removed in v4.0.0.

Network access is **opt-in**: set the environment variable
``FX_RATES_ALLOW_NETWORK=1`` before calling any loader, otherwise a
``RuntimeError`` is raised immediately. Nothing in the library or the test
suite imports these functions on a code path that executes them, tests are
fully offline (CONVENTIONS.md).

FRED is read through the keyless ``fredgraph.csv`` download (owner decision
D6): no API key is needed and ``api_key`` is ignored.

These loaders return *raw* observations; mapping them into the quote
structures expected by :mod:`fx_rates.bootstrap` (deposit/swap par quotes,
basis spreads) is left to the caller because free sources do not publish a
complete, consistent multi-currency quote set.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

__all__ = ["load_frankfurter_spot", "load_fred_series", "network_allowed"]

_ENV_FLAG = "FX_RATES_ALLOW_NETWORK"


def network_allowed() -> bool:
    """True if the user has explicitly opted in to network access."""
    return os.environ.get(_ENV_FLAG, "") == "1"


def _require_network() -> None:
    if not network_allowed():
        raise RuntimeError(
            f"Network access is disabled. Set {_ENV_FLAG}=1 to enable live data "
            "loaders (tests never do this)."
        )


def _legacy() -> Any:
    try:
        from qp_data import legacy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("live data now comes from qp_data: pip install -e python/data") from exc
    return legacy


def load_frankfurter_spot(base: str = "EUR", quote: str = "USD",
                          timeout: float = 10.0) -> float:
    """Latest ECB reference spot for ``base``/``quote`` from Frankfurter.

    Returns the spot as quote-currency units per 1 base unit.
    """
    _require_network()
    legacy = _legacy()  # pragma: no cover
    legacy.deprecated("fx_rates.data.live.load_frankfurter_spot", 'qp_data.fetch("frankfurter:...")')
    frame = legacy.frankfurter(base, [quote], timeout=timeout)  # pragma: no cover
    return float(frame[quote].iloc[-1])  # pragma: no cover


def load_fred_series(series_id: str, api_key: str | None = None,
                     timeout: float = 10.0) -> list[tuple[str, float]]:
    """The 100 most recent observations of a FRED series (e.g. ``DGS5``, ``SOFR``),
    newest first, as ``[(date_iso, value), ...]`` with missing values dropped.

    No API key is needed; ``api_key`` is accepted for compatibility and ignored.
    """
    _require_network()
    legacy = _legacy()  # pragma: no cover
    legacy.deprecated("fx_rates.data.live.load_fred_series", 'qp_data.fetch("fred:<ID>")')
    if api_key is not None:  # pragma: no cover
        warnings.warn("api_key is ignored: FRED is read keyless", DeprecationWarning, stacklevel=2)
    s = legacy.fred_series(series_id, timeout=timeout).dropna()  # pragma: no cover
    tail = s.iloc[::-1].iloc[:100]  # pragma: no cover
    return [(d.date().isoformat(), float(v)) for d, v in tail.items()]  # pragma: no cover
