"""Optional live market-data loader (FRED Treasury CMT yields).

NEVER imported by tests or the example pipeline — network access is
import-guarded per the portfolio conventions.  Usage::

    from fi_rates.data.live import load_fred_cmt
    df = load_fred_cmt(api_key="...")   # or FRED_API_KEY env var

The loader pulls constant-maturity Treasury (CMT) par yields from the FRED
API (series DGS3MO, DGS6MO, DGS1 ... DGS30).  CMT yields are *par* yields on
a semiannual bond basis; to build a discount curve from them, treat each as a
``ParSwap(maturity, rate/100, frequency=2)`` quote and bootstrap — an
approximation documented in docs/METHODOLOGY.md.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

try:  # pragma: no cover - optional dependency path, never exercised in tests
    import pandas as pd
except ImportError as _exc:  # pragma: no cover
    raise ImportError("fi_rates.data.live requires pandas") from _exc

__all__ = ["FRED_CMT_SERIES", "load_fred_cmt"]

#: FRED series id -> maturity in years.
FRED_CMT_SERIES: dict[str, float] = {
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def load_fred_cmt(
    api_key: str | None = None,
    start: str = "2020-01-01",
    timeout: float = 30.0,
) -> "pd.DataFrame":  # pragma: no cover - network path, never run in tests
    """Load Treasury CMT par yields from FRED.

    Parameters
    ----------
    api_key : str, optional
        FRED API key; falls back to the ``FRED_API_KEY`` environment variable.
    start : str
        Observation start date (ISO).
    timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    pandas.DataFrame
        Dates x maturities (columns are maturities in years), yields in
        percent as published by FRED.
    """
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "A FRED API key is required: pass api_key= or set FRED_API_KEY. "
            "This loader is optional and is never used by tests."
        )
    frames = {}
    for series, tenor in FRED_CMT_SERIES.items():
        params = urllib.parse.urlencode(
            {
                "series_id": series,
                "api_key": key,
                "file_type": "json",
                "observation_start": start,
            }
        )
        with urllib.request.urlopen(f"{_FRED_URL}?{params}", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        obs = payload.get("observations", [])
        s = pd.Series(
            {o["date"]: float(o["value"]) for o in obs if o["value"] != "."},
            name=tenor,
        )
        frames[tenor] = s
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()
