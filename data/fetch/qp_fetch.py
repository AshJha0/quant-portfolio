"""Free-data fetchers for the Phase 4 research programme.

Sources and their status (checked 2026-11-02):

- FRED (St. Louis Fed) ``fredgraph.csv``: no key, no login. Series from
  the Fed H.10 release (``DEX*``) are public domain. Some FRED series are
  third-party copyright (for example ``SP500``); this module refuses them.
- ECB Data Portal SDMX API (``data-api.ecb.europa.eu``): no key, no login;
  reuse is allowed with attribution to the ECB.
- Nasdaq Data Link: every call needs an API key, so an account and its
  terms of use. Not fetched. ``nasdaq_data_link`` raises until the owner
  decides.
- LOBSTER sample files: the download page sits behind LOBSTER's terms. Not
  downloaded. ``read_lobster_sample`` parses files the owner places by hand.

Downloads land in ``data/raw/`` (git-ignored). Each file gets a
``.sha256`` sidecar so a study can record exactly which bytes it used.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "raw"

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
ECB_URL = (
    "https://data-api.ecb.europa.eu/service/data/EXR/D.{ccys}.EUR.SP00.A"
    "?format=csvdata&startPeriod={start}&endPeriod={end}"
)

# FRED series known to carry third-party copyright. Not exhaustive: a new
# series must be checked on its FRED page before it is added to a study.
FRED_RESTRICTED = frozenset({"SP500", "NASDAQCOM", "DJIA", "VIXCLS", "WILL5000IND"})

ECB_G10 = ("USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK")


class SourceNotPermitted(RuntimeError):
    """The source needs a login, a key or terms the owner has not accepted."""


def _download(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "quant-portfolio-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https hosts)
        return resp.read()


def _store(name: str, payload: bytes, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / name
    path.write_bytes(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n")
    return path


def parse_fred_csv(text: str, series: str) -> pd.Series:
    """Parse a ``fredgraph.csv`` body. FRED writes ``.`` for a missing day."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    if df.shape[1] != 2:
        raise ValueError(f"unexpected FRED layout: columns {list(df.columns)}")
    s = pd.Series(df.iloc[:, 1].to_numpy(float), index=pd.to_datetime(df.iloc[:, 0]), name=series)
    return s.dropna()


def fetch_fred(series: str, raw_dir: Path = RAW_DIR) -> pd.Series:
    if series.upper() in FRED_RESTRICTED:
        raise SourceNotPermitted(f"FRED series {series} is third-party copyright")
    payload = _download(FRED_URL.format(series=series))
    _store(f"fred_{series}.csv", payload, raw_dir)
    return parse_fred_csv(payload.decode("utf-8"), series)


def parse_ecb_csv(text: str) -> pd.DataFrame:
    """Parse an ECB SDMX ``csvdata`` body into a date x currency frame of
    foreign-currency units per one EUR."""
    df = pd.read_csv(io.StringIO(text))
    need = {"CURRENCY", "TIME_PERIOD", "OBS_VALUE"}
    if not need.issubset(df.columns):
        raise ValueError(f"unexpected ECB layout: missing {sorted(need - set(df.columns))}")
    wide = df.pivot_table(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    return wide.sort_index().astype(float)


def fetch_ecb_fx(
    ccys: tuple[str, ...] = ECB_G10,
    start: str = "1999-01-01",
    end: str = "2021-12-31",
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    payload = _download(ECB_URL.format(ccys="+".join(ccys), start=start, end=end))
    _store(f"ecb_exr_{start}_{end}.csv", payload, raw_dir)
    return parse_ecb_csv(payload.decode("utf-8"))


def nasdaq_data_link(*_args, **_kwargs):
    raise SourceNotPermitted(
        "Nasdaq Data Link needs an API key (account and terms). Owner decision required."
    )


LOBSTER_MESSAGE_COLUMNS = ("time", "type", "order_id", "size", "price", "direction")


def read_lobster_sample(message_csv: Path, orderbook_csv: Path, levels: int = 1) -> pd.DataFrame:
    """Join a LOBSTER message file with its order-book file (level ``levels``).

    LOBSTER prices are integer dollars x 10,000; time is seconds after
    midnight. Reference: LOBSTER data structure note, lobsterdata.com.
    """
    msg = pd.read_csv(message_csv, header=None).iloc[:, :6]
    msg.columns = list(LOBSTER_MESSAGE_COLUMNS)
    book = pd.read_csv(orderbook_csv, header=None)
    if len(book) != len(msg):
        raise ValueError("message and order-book files have different row counts")
    if book.shape[1] < 4 * levels:
        raise ValueError(f"order book has fewer than {levels} levels")
    cols = []
    for lvl in range(1, levels + 1):
        cols += [f"ask_px_{lvl}", f"ask_sz_{lvl}", f"bid_px_{lvl}", f"bid_sz_{lvl}"]
    book = book.iloc[:, : 4 * levels]
    book.columns = cols
    out = pd.concat([msg, book], axis=1)
    for c in ["price"] + [c for c in cols if "_px_" in c]:
        out[c] = out[c] / 10_000.0
    return out


if __name__ == "__main__":  # pragma: no cover
    import sys

    what = sys.argv[1] if len(sys.argv) > 1 else "ecb"
    if what == "ecb":
        print(fetch_ecb_fx().tail())
    else:
        print(fetch_fred(what).tail())
