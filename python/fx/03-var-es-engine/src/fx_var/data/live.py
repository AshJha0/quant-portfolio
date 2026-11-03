"""Guarded live FX data loader (Frankfurter / ECB reference rates).

Deprecated: use ``qp_data.fetch("frankfurter:USD:EUR,JPY", start, end)`` and
invert. Both functions here now delegate to qp_data (retries, backoff, one
parser) and are removed in v4.0.0.

Network access is **opt-in only**: nothing in this module runs at import
time, no test imports it with the guard open, and every call raises unless
``allow_network=True`` is passed explicitly or the environment variable
``FX_VAR_ALLOW_NETWORK=1`` is set. Library code paths used by tests rely
exclusively on :mod:`fx_var.data.synthetic` (see CONVENTIONS.md).

Frankfurter serves ECB reference rates - daily fixings only, quoted as
QUOTE units per 1 unit of the requested base. For the engine's USD-factor
convention we request base=USD and invert: CCYUSD = 1 / (CCY per USD).
"""

from __future__ import annotations

import os
from typing import cast

import numpy as np
import pandas as pd

from ..common import fx_factor

__all__ = ["frankfurter_factor_returns", "load_frankfurter"]


def _check_guard(allow_network: bool | None) -> None:
    if allow_network is None:
        allow_network = os.environ.get("FX_VAR_ALLOW_NETWORK", "") == "1"
    if not allow_network:
        raise RuntimeError(
            "Network access is disabled by default (offline test policy). "
            "Pass allow_network=True or set FX_VAR_ALLOW_NETWORK=1 to fetch "
            "live ECB reference rates from Frankfurter."
        )


def load_frankfurter(
    symbols: list[str],
    start: str = "2023-01-01",
    end: str | None = None,
    timeout: float = 15.0,
    allow_network: bool | None = None,
) -> pd.DataFrame:
    """Fetch daily CCYUSD spot levels from Frankfurter (ECB fixings).

    Parameters
    ----------
    symbols : list of str
        Currencies, e.g. ``["EUR", "JPY", "GBP"]`` (USD excluded).
    start, end : str
        ISO dates; ``end=None`` means latest.
    allow_network : bool, optional
        Explicit opt-in; falls back to ``FX_VAR_ALLOW_NETWORK=1``.

    Returns
    -------
    pandas.DataFrame
        Business-day indexed USD price of 1 unit of each currency
        (columns = currency codes).
    """
    _check_guard(allow_network)
    symbols = [s.upper() for s in symbols if s.upper() != "USD"]
    if not symbols:
        raise ValueError("symbols must contain at least one non-USD currency")
    try:
        from qp_data import legacy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("live data now comes from qp_data: pip install -e python/data") from exc
    legacy.deprecated("fx_var.data.live.load_frankfurter", 'qp_data.fetch("frankfurter:USD:...")')
    frame = legacy.frankfurter("USD", symbols, start, end, timeout=timeout)  # pragma: no cover
    frame.index.name = None  # pragma: no cover
    # Frankfurter gives CCY per USD; invert to USD per CCY (engine convention)
    return (1.0 / frame[symbols]).astype(float)  # pragma: no cover


def frankfurter_factor_returns(
    symbols: list[str],
    start: str = "2023-01-01",
    end: str | None = None,
    allow_network: bool | None = None,
) -> pd.DataFrame:
    """Daily ``FX:CCY`` log returns from Frankfurter fixings.

    Convenience wrapper producing columns named per the engine's factor
    convention, ready for :func:`fx_var.historical_var.historical_var`.
    Note ECB fixings are once-a-day snapshots (14:15 CET) - fine for a demo,
    not a substitute for a desk's official closing rates (see
    docs/DESK_GUIDE.md on rate sourcing across time zones).
    """
    spots = load_frankfurter(symbols, start, end, allow_network=allow_network)
    rets = cast(pd.DataFrame, np.log(spots)).diff().dropna()
    rets.columns = [fx_factor(c) for c in rets.columns]
    return rets
