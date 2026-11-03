"""Live FX data loader (Frankfurter API).

Deprecated: use ``qp_data.fetch("frankfurter:USD:EUR,JPY,...", start, end)``
and invert. :func:`fetch_spots` now delegates to qp_data (retries, backoff,
one parser) and is removed in v4.0.0.

Nothing in the library or the test suite calls this module's network
function; tests run fully offline against :mod:`fx_port.data.synthetic`.

The Frankfurter API (https://frankfurter.dev) serves ECB reference rates:
spot only, no deposit rates, so live panels support spot/momentum/value work
but carry requires a separate rates source (``qp_data``: ``ecb:ESTR``,
``nyfed:SOFR``, ``boe:IUDSOIA``).
"""

from __future__ import annotations

import pandas as pd


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
        If qp_data is missing, the network is unavailable or the response is
        malformed. Callers should catch this and fall back to
        :func:`fx_port.data.synthetic.make_panel`.
    """
    try:
        from qp_data import legacy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("live data now comes from qp_data: pip install -e python/data") from exc
    legacy.deprecated("fx_port.data.live.fetch_spots", 'qp_data.fetch("frankfurter:USD:...")')
    try:
        frame = legacy.frankfurter(base, currencies, start, end, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(
            f"Frankfurter fetch failed ({exc}); use fx_port.data.synthetic for offline work."
        ) from exc
    frame.index.name = None
    # API returns CCY per 1 USD; invert to USD per 1 CCY (CCYUSD).
    return (1.0 / frame).astype(float)
