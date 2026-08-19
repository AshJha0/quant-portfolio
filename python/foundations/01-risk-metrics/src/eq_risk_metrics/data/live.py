"""Optional live market-data loader (yfinance) -- import-guarded.

Importing this module never requires yfinance or a network connection:
the dependency is only imported inside :func:`load_prices`. Install with
``pip install -e ".[live]"``.

Command-line usage (writes a CSV the pipeline can consume offline)::

    python -m eq_risk_metrics.data.live SPY 2016-01-01   # -> data/SPY.csv
    python examples/run_pipeline.py --csv data/SPY.csv
"""
from __future__ import annotations

import importlib.util

import pandas as pd

__all__ = ["HAS_YFINANCE", "load_prices"]

#: True when the optional ``yfinance`` extra is installed. Checking this
#: never imports yfinance itself (spec lookup only), so it is safe offline.
HAS_YFINANCE: bool = importlib.util.find_spec("yfinance") is not None


def load_prices(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Download daily adjusted close prices for ``ticker`` via yfinance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. ``"SPY"``.
    start : str, optional
        ISO start date (``"2016-01-01"``); ``None`` fetches max history.

    Returns
    -------
    pandas.DataFrame
        Columns ``Date`` (datetime64) and ``Adj Close`` (float), ascending
        by date -- the same schema as the synthetic generator.

    Raises
    ------
    ImportError
        If yfinance is not installed (actionable message included).
    RuntimeError
        If the download returns no rows (offline, bad ticker).
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "yfinance is required for live data: pip install 'eq-risk-metrics[live]'"
        ) from exc
    data = yf.download(ticker, start=start, interval="1d", progress=False, auto_adjust=True)
    if data is None or len(data) == 0:  # pragma: no cover - network dependent
        raise RuntimeError(f"yfinance returned no data for {ticker!r} (offline or bad ticker)")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # multi-ticker column shape
        close = close.iloc[:, 0]
    out = close.rename("Adj Close").reset_index()
    out.columns = ["Date", "Adj Close"]
    return out


def _main() -> None:  # pragma: no cover - thin CLI, network dependent
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Download prices to data/<TICKER>.csv")
    parser.add_argument("ticker")
    parser.add_argument("start", nargs="?", default=None)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    df = load_prices(args.ticker, args.start)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.ticker}.csv")
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")


if __name__ == "__main__":  # pragma: no cover
    _main()
