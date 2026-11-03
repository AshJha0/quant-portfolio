"""Optional live market-data loader (FRED Treasury CMT yields).

Deprecated: use ``qp_data.fetch("fred:DGS10")`` (or ``"ust:par"`` for the
whole Treasury curve) and ``qp_data.get(..., as_of=...)``, which also applies
the H.15 publication lag. :func:`load_fred_cmt` now delegates to qp_data and
is removed in v4.0.0.

FRED is read through the keyless ``fredgraph.csv`` download (owner decision
D6): no API key is needed. Network access is opt-in: pass
``allow_network=True`` or set ``QP_DATA_ALLOW_NETWORK=1``. Passing the old
``api_key=`` also counts as opting in (the key itself is ignored).

NEVER imported by tests or the example pipeline. Usage::

    from fi_rates.data.live import load_fred_cmt
    df = load_fred_cmt(allow_network=True)

The loader pulls constant-maturity Treasury (CMT) par yields (series DGS3MO,
DGS6MO, DGS1 ... DGS30). CMT yields are *par* yields on a semiannual bond
basis; to build a discount curve from them, treat each as a
``ParSwap(maturity, rate/100, frequency=2)`` quote and bootstrap, an
approximation documented in docs/METHODOLOGY.md.
"""

from __future__ import annotations

import os

import pandas as pd

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


def load_fred_cmt(
    api_key: str | None = None,
    start: str = "2020-01-01",
    timeout: float = 30.0,
    allow_network: bool | None = None,
) -> pd.DataFrame:
    """Load Treasury CMT par yields from FRED.

    Parameters
    ----------
    api_key : str, optional
        Ignored (FRED is read keyless); passing it counts as opting in to
        network access, as the old key-based call did.
    start : str
        Observation start date (ISO).
    timeout : float
        Per-request timeout in seconds.
    allow_network : bool, optional
        Explicit opt-in; falls back to ``QP_DATA_ALLOW_NETWORK=1``.

    Returns
    -------
    pandas.DataFrame
        Dates x maturities (columns are maturities in years), yields in
        percent as published by FRED.

    Raises
    ------
    RuntimeError
        Before any network access, if the caller did not opt in.
    """
    if allow_network is None:
        allow_network = api_key is not None or os.environ.get("QP_DATA_ALLOW_NETWORK", "") == "1"
    if not allow_network:
        raise RuntimeError(
            "Network access is disabled. No FRED API key is needed any more "
            "(keyless CSV); pass allow_network=True or set QP_DATA_ALLOW_NETWORK=1. "
            "This loader is optional and is never used by tests."
        )
    try:  # pragma: no cover - network path, never run in tests
        from qp_data import legacy
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("live data now comes from qp_data: pip install -e python/data") from exc
    legacy.deprecated("fi_rates.data.live.load_fred_cmt", 'qp_data.fetch("fred:DGS10")')  # pragma: no cover
    frames = {  # pragma: no cover
        tenor: legacy.fred_series(series, start, timeout=timeout)
        for series, tenor in FRED_CMT_SERIES.items()
    }
    df = pd.DataFrame(frames)  # pragma: no cover
    df.index = pd.to_datetime(df.index)  # pragma: no cover
    return df.sort_index()  # pragma: no cover
