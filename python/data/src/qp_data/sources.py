"""Adapters for the free, keyless sources the portfolio uses.

Each source has a URL builder and a parser. The parser takes the raw bytes
and returns a frame indexed by observation date (``obs_date``) with float
columns, and raises :class:`~qp_data.errors.SchemaError` on any other layout.
Parsers never touch the network, so tests run them on recorded responses in
``python/data/tests/fixtures``.

| Key | Source | Licence / access |
|---|---|---|
| ``fred:<ID>`` | FRED ``fredgraph.csv`` (no key, no ALFRED; D6) | public domain for Fed/BLS series; allow-list, others refused |
| ``ecb:EXR:<CCY>`` | ECB Data Portal, EXR dataflow | free reuse with attribution to the ECB |
| ``ecb:ESTR`` | ECB Data Portal, EST dataflow (volume-weighted rate) | as above |
| ``frankfurter:<BASE>`` | api.frankfurter.dev (republishes ECB reference rates) | open, no key |
| ``nyfed:SOFR`` | NY Fed Markets Data API | open, no key |
| ``boe:IUDSOIA`` | Bank of England IADB CSV (SONIA) | open, no key |
| ``ust:par`` | US Treasury daily par yield curve CSV | public domain |
| ``cboe:VIX`` | CBOE VIX history CSV | free download |
| ``coinbase:<PRODUCT>:1d`` | Coinbase Exchange public REST candles | public endpoint, no authentication |
| ``binance:<SYMBOL>:1d`` | Binance public market-data REST (data-api.binance.vision) | public endpoint, no authentication |
| ``french:<DATASET>`` | Kenneth R. French Data Library zip files (``qp_data.french``) | free with citation of the library |

Websocket recording of the crypto order books belongs to W25/W30 and is not here.
"""

from __future__ import annotations

import datetime as dt
import io
import json
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from .errors import SchemaError, SourceNotPermitted

__all__ = [
    "FRED_PERMITTED_PREFIXES",
    "SOURCES",
    "Source",
    "build_url",
    "fred_permitted",
    "parse",
    "parse_binance_klines",
    "parse_boe_csv",
    "parse_cboe_vix",
    "parse_coinbase_candles",
    "parse_ecb_csv",
    "parse_frankfurter",
    "parse_fred_csv",
    "parse_nyfed_sofr",
    "parse_ust_par",
]

# FRED series cleared for use: Federal Reserve and BLS series, which are
# public domain. FRED also hosts third-party series under copyright (equity
# indices, for example), so this is an allow-list: a new series is checked
# on its FRED page, then added here.
FRED_PERMITTED_PREFIXES = ("DEX", "DGS", "DTB", "DFF", "SOFR", "FEDFUNDS", "CPIAUCSL", "IR3TIB01")
# OECD Main Economic Indicators republished on FRED: 3-month interbank rates
# (``IR3TIB01<CC>M156N``) and CPI all items (``<ISO3>CPIALLMINMEI``). OECD
# data are licensed CC BY 4.0 (OECD terms and conditions, July 2024), so
# they may be stored and redistributed with attribution "Source: OECD".
FRED_PERMITTED_SUFFIXES = ("CPIALLMINMEI",)


def fred_permitted(series: str) -> bool:
    u = series.upper()
    return u.startswith(FRED_PERMITTED_PREFIXES) or u.endswith(FRED_PERMITTED_SUFFIXES)


def _frame(index: pd.Index | pd.Series, cols: dict[str, object]) -> pd.DataFrame:
    f = pd.DataFrame(cols, index=pd.DatetimeIndex(index, name="obs_date"))
    try:
        f = f.astype(float)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"non-numeric values: {exc}") from exc
    return f.sort_index()


def _need(cols: object, need: set[str], what: str) -> None:
    have = set(map(str, cols))  # type: ignore[call-overload]
    if not need <= have:
        raise SchemaError(f"unexpected {what} layout: missing {sorted(need - have)}")


# ------------------------------------------------------------------ FRED --
def parse_fred_csv(payload: bytes, series: str | None = None) -> pd.DataFrame:
    """``observation_date,<ID>`` (older files: ``DATE,<ID>``). FRED marks a
    missing day with ``.`` (older files) or an empty cell (current files);
    both become NaN and are dropped."""
    df = pd.read_csv(io.BytesIO(payload), na_values=["."])
    if df.shape[1] != 2 or df.columns[0] not in ("observation_date", "DATE"):
        raise SchemaError(f"unexpected FRED layout: columns {list(df.columns)}")
    name = series or str(df.columns[1])
    if series is not None and str(df.columns[1]) != series:
        raise SchemaError(f"FRED file is for {df.columns[1]}, not {series}")
    f = _frame(pd.to_datetime(df.iloc[:, 0]), {"value": df.iloc[:, 1].to_numpy()})
    f.attrs["series"] = name
    return f.dropna()


# ------------------------------------------------------------------- ECB --
def parse_ecb_csv(payload: bytes) -> pd.DataFrame:
    """ECB SDMX ``csvdata``. EXR gives one column per currency (foreign units
    per EUR); other dataflows give a single ``value`` column."""
    df = pd.read_csv(io.BytesIO(payload))
    _need(df.columns, {"TIME_PERIOD", "OBS_VALUE"}, "ECB")
    if "CURRENCY" in df.columns:
        wide = df.pivot_table(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE", aggfunc="last")
        return _frame(pd.to_datetime(wide.index), {str(c): wide[c].to_numpy() for c in wide.columns})
    return _frame(pd.to_datetime(df["TIME_PERIOD"]), {"value": df["OBS_VALUE"].to_numpy()})


# ----------------------------------------------------------- Frankfurter --
def parse_frankfurter(payload: bytes) -> pd.DataFrame:
    """A range ``{"rates": {date: {CCY: units per base}}}`` or the latest
    fixing ``{"date": d, "rates": {CCY: units per base}}``."""
    try:
        body = json.loads(payload)
        rates = body["rates"]
        if isinstance(rates, dict) and rates and not isinstance(next(iter(rates.values())), dict):
            rates = {body["date"]: rates}
    except (ValueError, KeyError, TypeError) as exc:
        raise SchemaError(f"unexpected Frankfurter layout: {exc}") from exc
    if not isinstance(rates, dict) or not rates:
        raise SchemaError("Frankfurter returned no rates")
    wide = pd.DataFrame(rates).T
    return _frame(pd.to_datetime(wide.index), {str(c): wide[c].to_numpy() for c in wide.columns})


# ---------------------------------------------------------------- NY Fed --
def parse_nyfed_sofr(payload: bytes) -> pd.DataFrame:
    """NY Fed ``refRates``: rate and volume, in percent and USD billions."""
    try:
        rows = json.loads(payload)["refRates"]
        df = pd.DataFrame(rows)
        _need(df.columns, {"effectiveDate", "percentRate", "volumeInBillions"}, "NY Fed")
    except (ValueError, KeyError, TypeError) as exc:
        raise SchemaError(f"unexpected NY Fed layout: {exc}") from exc
    return _frame(
        pd.to_datetime(df["effectiveDate"]),
        {"value": df["percentRate"].to_numpy(), "volume_bn": df["volumeInBillions"].to_numpy()},
    )


# ------------------------------------------------------------------- BoE --
def parse_boe_csv(payload: bytes) -> pd.DataFrame:
    """IADB CSV ``DATE,<CODE>`` with dates like ``01 Jul 2024``."""
    df = pd.read_csv(io.BytesIO(payload))
    if df.shape[1] != 2 or df.columns[0] != "DATE":
        raise SchemaError(f"unexpected BoE layout: columns {list(df.columns)}")
    return _frame(pd.to_datetime(df["DATE"], format="%d %b %Y"), {"value": df.iloc[:, 1].to_numpy()})


# ------------------------------------------------------------- Treasury --
def parse_ust_par(payload: bytes) -> pd.DataFrame:
    """Daily par yield curve, one column per tenor in years, yields in percent."""
    df = pd.read_csv(io.BytesIO(payload))
    if df.columns[0] != "Date":
        raise SchemaError(f"unexpected Treasury layout: columns {list(df.columns)}")
    tenors: dict[str, object] = {}
    for c in df.columns[1:]:
        n, unit = str(c).split()
        years = float(n) / 12.0 if unit == "Mo" else float(n) if unit == "Yr" else None
        if years is None:
            raise SchemaError(f"unknown Treasury tenor {c!r}")
        tenors[f"{years:g}"] = df[c].to_numpy()
    return _frame(pd.to_datetime(df["Date"], format="%m/%d/%Y"), tenors)


# ------------------------------------------------------------------ CBOE --
def parse_cboe_vix(payload: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(payload))
    _need(df.columns, {"DATE", "OPEN", "HIGH", "LOW", "CLOSE"}, "CBOE")
    return _frame(
        pd.to_datetime(df["DATE"], format="%m/%d/%Y"),
        {c.lower(): df[c].to_numpy() for c in ("OPEN", "HIGH", "LOW", "CLOSE")},
    )


# ---------------------------------------------------------------- crypto --
def parse_coinbase_candles(payload: bytes) -> pd.DataFrame:
    """Coinbase Exchange candles ``[time, low, high, open, close, volume]``,
    newest first, ``time`` = bucket start in Unix seconds (UTC)."""
    rows = json.loads(payload)
    if not isinstance(rows, list) or any(not isinstance(r, list) or len(r) != 6 for r in rows):
        raise SchemaError("unexpected Coinbase candle layout")
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    idx = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_localize(None)
    return _frame(idx, {c: df[c].to_numpy() for c in ("open", "high", "low", "close", "volume")})


def parse_binance_klines(payload: bytes) -> pd.DataFrame:
    """Binance klines: ``[open_time_ms, open, high, low, close, volume,
    close_time_ms, ...]`` with prices as strings."""
    rows = json.loads(payload)
    if not isinstance(rows, list) or any(not isinstance(r, list) or len(r) < 7 for r in rows):
        raise SchemaError("unexpected Binance kline layout")
    df = pd.DataFrame([r[:6] for r in rows], columns=["t", "open", "high", "low", "close", "volume"])
    idx = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
    return _frame(idx, {c: df[c].to_numpy() for c in ("open", "high", "low", "close", "volume")})


# --------------------------------------------------------------- registry --
@dataclass(frozen=True)
class Source:
    name: str
    url: Callable[[str, dt.date | None, dt.date | None], str]
    parse: Callable[[bytes], pd.DataFrame]


def _d(x: dt.date | None, default: str = "") -> str:
    return x.isoformat() if x else default


def _fred_url(key: str, start: dt.date | None, end: dt.date | None) -> str:
    sid = key.split(":", 1)[1]
    if not fred_permitted(sid):
        raise SourceNotPermitted(f"FRED series {sid} is not on the public-domain allow-list")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    if start:
        url += f"&cosd={start.isoformat()}"
    if end:
        url += f"&coed={end.isoformat()}"
    return url


def _ecb_url(key: str, start: dt.date | None, end: dt.date | None) -> str:
    parts = key.split(":")
    if parts[1] == "EXR":
        flow = f"EXR/D.{parts[2]}.EUR.SP00.A"
    elif parts[1] == "ESTR":
        flow = "EST/B.EU000A2X2A25.WT"
    else:
        raise KeyError(key)
    return (
        f"https://data-api.ecb.europa.eu/service/data/{flow}?format=csvdata"
        f"&startPeriod={_d(start, '1999-01-01')}&endPeriod={_d(end, dt.date.today().isoformat())}"
    )


def _frank_url(key: str, start: dt.date | None, end: dt.date | None) -> str:
    _, base, *syms = key.split(":")
    q = f"?base={base}" + (f"&symbols={syms[0]}" if syms else "")
    if start is None:
        return f"https://api.frankfurter.dev/v1/latest{q}"
    return f"https://api.frankfurter.dev/v1/{start.isoformat()}..{_d(end)}{q}"


def _nyfed_url(key: str, start: dt.date | None, end: dt.date | None) -> str:
    return (
        "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
        f"?startDate={_d(start, '2018-04-02')}&endDate={_d(end, dt.date.today().isoformat())}"
    )


def _boe_url(key: str, start: dt.date | None, end: dt.date | None) -> str:
    code = key.split(":", 1)[1]
    s = (start or dt.date(1997, 1, 1)).strftime("%d/%b/%Y")
    e = (end or dt.date.today()).strftime("%d/%b/%Y")
    return (
        "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes"
        f"&Datefrom={s}&Dateto={e}&SeriesCodes={code}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
    )


def _ust_url(key: str, start: dt.date | None, end: dt.date | None) -> str:
    year = (end or start or dt.date.today()).year
    return (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        f"daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
        f"&field_tdr_date_value={year}&page&_format=csv"
    )


def _cboe_url(key: str, start: dt.date | None, end: dt.date | None) -> str:
    return "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def _cb_url(key: str, start: dt.date | None, end: dt.date | None) -> str:
    _, product, gran = key.split(":")
    if gran != "1d":
        raise KeyError(f"only daily candles are wired: {key}")
    url = f"https://api.exchange.coinbase.com/products/{product}/candles?granularity=86400"
    if start:
        url += f"&start={start.isoformat()}T00:00:00Z&end={_d(end, dt.date.today().isoformat())}T00:00:00Z"
    return url


def _bn_url(key: str, start: dt.date | None, end: dt.date | None) -> str:
    _, symbol, gran = key.split(":")
    if gran != "1d":
        raise KeyError(f"only daily klines are wired: {key}")
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1d&limit=1000"
    if start:
        url += f"&startTime={int(pd.Timestamp(start, tz='UTC').timestamp() * 1000)}"
    if end:
        url += f"&endTime={int(pd.Timestamp(end, tz='UTC').timestamp() * 1000)}"
    return url


SOURCES: dict[str, Source] = {
    "fred": Source("fred", _fred_url, parse_fred_csv),
    "ecb": Source("ecb", _ecb_url, parse_ecb_csv),
    "frankfurter": Source("frankfurter", _frank_url, parse_frankfurter),
    "nyfed": Source("nyfed", _nyfed_url, parse_nyfed_sofr),
    "boe": Source("boe", _boe_url, parse_boe_csv),
    "ust": Source("ust", _ust_url, parse_ust_par),
    "cboe": Source("cboe", _cboe_url, parse_cboe_vix),
    "coinbase": Source("coinbase", _cb_url, parse_coinbase_candles),
    "binance": Source("binance", _bn_url, parse_binance_klines),
}


def _source(key: str) -> Source:
    try:
        return SOURCES[key.split(":", 1)[0]]
    except KeyError:
        raise KeyError(f"unknown source in {key!r}; known: {sorted(SOURCES)}") from None


def build_url(key: str, start: dt.date | None = None, end: dt.date | None = None) -> str:
    return _source(key).url(key, start, end)


def parse(key: str, payload: bytes) -> pd.DataFrame:
    return _source(key).parse(payload)


# Registered last: qp_data.french imports Source from this module, so the
# import has to come after Source and SOURCES exist.
from . import french as _french  # noqa: E402

SOURCES["french"] = _french.SOURCE
