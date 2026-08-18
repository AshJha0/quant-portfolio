"""Guarded live FX data loader (Frankfurter API).

Network access lives ONLY here.  Nothing in the library or the test suite
imports this module's network functions at import time; tests run fully
offline against :mod:`fx_port.data.synthetic`.

The Frankfurter API (https://frankfurter.dev) serves ECB reference rates —
spot only, no deposit rates — so live panels support spot/momentum/value work
but carry requires a separate rates source.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pandas as pd

_API = "https://api.frankfurter.dev/v1"


def fetch_spots(
    currencies: list[str],
    start: str,
    end: str,
    base: str = "USD",
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch daily ECB reference spots from Frankfurter, quoted CCYUSD.

    Parameters
    ----------
    currencies : list of str
        ISO codes to fetch (e.g. ``["EUR", "JPY", "GBP"]``).
    start, end : str
        ISO dates bounding the sample.
    base : str
        Quote currency; default USD so columns are USD per 1 CCY (repo
        convention).
    timeout : float
        Socket timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Spot levels, index = dates, columns = currencies, quoted CCYUSD.

    Raises
    ------
    RuntimeError
        If the network is unavailable or the API response is malformed.
        Callers should catch this and fall back to
        :func:`fx_port.data.synthetic.make_panel`.
    """
    url = f"{_API}/{start}..{end}?base={base}&symbols={','.join(currencies)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Frankfurter fetch failed ({exc}); use fx_port.data.synthetic "
            "for offline work."
        ) from exc
    try:
        frame = pd.DataFrame(payload["rates"]).T.sort_index()
        frame.index = pd.to_datetime(frame.index)
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"malformed Frankfurter response: {exc}") from exc
    # API returns CCY per 1 USD; invert to USD per 1 CCY (CCYUSD).
    return (1.0 / frame).astype(float)
