"""Guarded loader for real World Bank / IMF-style macro data (never used in tests).

Network access is disabled by default: ``fetch_worldbank_indicators`` refuses
to run unless the environment variable ``FX_CREDIT_ALLOW_NETWORK=1`` is set.
Offline, ``load_worldbank_csv`` validates a local CSV against the expected
schema so the rest of the pipeline can consume real data with zero changes.
"""

from __future__ import annotations

import os

import pandas as pd

__all__ = ["WORLDBANK_SCHEMA", "load_worldbank_csv", "fetch_worldbank_indicators"]

#: Expected long-format schema for a World Bank-style extract.
WORLDBANK_SCHEMA: dict[str, str] = {
    "country": "str",       # ISO3 or name
    "year": "int",          # calendar year
    "indicator": "str",     # e.g. FI.RES.TOTL.MO (reserves in months of imports)
    "value": "float",
}


def load_worldbank_csv(path: str) -> pd.DataFrame:
    """Load and validate a local long-format World Bank-style CSV.

    Parameters
    ----------
    path : str
        Path to a CSV with columns ``country, year, indicator, value``.

    Returns
    -------
    pandas.DataFrame
        Validated frame with coerced dtypes, sorted by (country, year).

    Raises
    ------
    ValueError
        If required columns are missing or dtypes cannot be coerced.
    """
    df = pd.read_csv(path)
    missing = [c for c in WORLDBANK_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    try:
        df["year"] = df["year"].astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"schema coercion failed: {exc}") from exc
    return df.sort_values(["country", "year"]).reset_index(drop=True)


def fetch_worldbank_indicators(
    indicators: list[str],
    countries: list[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch indicators from the World Bank API (guarded; requires opt-in).

    Raises
    ------
    RuntimeError
        Always, unless ``FX_CREDIT_ALLOW_NETWORK=1`` is set in the
        environment.  Tests and library code paths never set it.
    """
    if os.environ.get("FX_CREDIT_ALLOW_NETWORK") != "1":
        raise RuntimeError(
            "Network access is disabled. Set FX_CREDIT_ALLOW_NETWORK=1 to enable, "
            "or use fx_credit.data.synthetic generators / load_worldbank_csv."
        )
    raise NotImplementedError(
        "Live World Bank download is intentionally not implemented in this "
        "portfolio project; use load_worldbank_csv on a manual extract."
    )
