"""Kenneth French data library adapter (keyless, W23 and the W9 extension).

Plain idea: Professor Kenneth French publishes the Fama-French equity factor
returns (market excess return, size, value, profitability, investment,
momentum) as free zip files, each holding one CSV. No account or key is
needed. This module builds the download URL and turns the CSV into a frame of
decimal returns indexed by observation date. It never touches the network
itself; ``qp_data.net.download`` fetches the bytes.

Formal statement. A library CSV has free-text preamble lines, then one or
more tables. Each table starts with a header line whose first field is empty
(``,Mkt-RF,SMB,HML,RF``) and ends at the first blank line. The first table
holds the monthly (``YYYYMM``) or daily (``YYYYMMDD``) rows; a later table
titled "Annual Factors" holds ``YYYY`` rows. Values are percent; the
missing-value codes -99.99 and -999 become NaN. :func:`parse_french` returns
the first table (or the annual one with ``table="annual"``) divided by 100.
Monthly rows are stamped at month end, annual rows at year end.

Worked example (``tests/test_french.py``, file built from the 202609 CRSP
database): the row ``192607,   2.89,  -2.42,  -2.75,   0.22`` becomes
1926-07-31 with Mkt-RF 0.0289, SMB -0.0242, HML -0.0275 and RF 0.0022.
Past values change when the library rebuilds from a new CRSP vintage, so a
study records the file's SHA-256 with its results.

Licence and access: the files are free to download and use with citation of
the library ("Kenneth R. French, Data Library",
mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Only the
datasets in :data:`DATASETS` are wired; a new one is checked on the library
page and added there.

Integration note: ``qp_data.sources.SOURCES`` registers sources by key
prefix. Add ``"french": french.SOURCE`` there to make ``french:<dataset>``
usable through ``sources.build_url`` and ``sources.parse``.
"""

from __future__ import annotations

import datetime as dt
import io
import re
import zipfile

import numpy as np
import pandas as pd

from .errors import SchemaError, SourceNotPermitted
from .sources import Source

__all__ = ["BASE_URL", "DATASETS", "SOURCE", "french_url", "parse_french", "parse_french_csv"]

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

#: Datasets cleared for use (file stem on the library's ftp page).
DATASETS = frozenset({
    "F-F_Research_Data_Factors",
    "F-F_Research_Data_Factors_daily",
    "F-F_Research_Data_5_Factors_2x3",
    "F-F_Research_Data_5_Factors_2x3_daily",
    "F-F_Momentum_Factor",
    "F-F_Momentum_Factor_daily",
    "Developed_3_Factors",
    "Developed_Mom_Factor",
})

_MISSING = (-99.99, -999.0)
_ROW = re.compile(r"^\s*(\d{4}|\d{6}|\d{8})\s*,")


def french_url(key: str, start: dt.date | None = None, end: dt.date | None = None) -> str:
    """URL of ``french:<dataset>``. The library serves whole files, so
    ``start`` and ``end`` are ignored (slice after parsing)."""
    name = key.split(":", 1)[1] if ":" in key else key
    if name not in DATASETS:
        raise SourceNotPermitted(f"French dataset {name!r} is not in the cleared list")
    return f"{BASE_URL}{name}_CSV.zip"


def _stamp(code: str) -> pd.Timestamp:
    if len(code) == 8:
        return pd.Timestamp(f"{code[:4]}-{code[4:6]}-{code[6:]}")
    if len(code) == 6:
        return pd.Timestamp(f"{code[:4]}-{code[4:]}-01") + pd.offsets.MonthEnd(0)
    return pd.Timestamp(f"{code}-12-31")


def _tables(text: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """(title, columns, rows) for each table in the file."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[tuple[str, list[str], list[list[str]]]] = []
    title = ""
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.lstrip().startswith(",") and ln.count(",") >= 1:
            cols = [c.strip() for c in ln.split(",")[1:]]
            rows: list[list[str]] = []
            i += 1
            while i < len(lines) and _ROW.match(lines[i]):
                rows.append([c.strip() for c in lines[i].split(",")])
                i += 1
            out.append((title, cols, rows))
            continue
        if ln.strip():
            title = ln.strip()
        i += 1
    return out


def parse_french_csv(text: str, table: str = "first") -> pd.DataFrame:
    """Parse the CSV text of a library file; see the module docstring."""
    tabs = _tables(text)
    if not tabs:
        raise SchemaError("no French-library table found (expected a ',<factor>,...' header)")
    if table == "first":
        title, cols, rows = tabs[0]
    elif table == "annual":
        hits = [t for t in tabs if t[0].lower().startswith("annual factors")]
        if not hits:
            raise SchemaError("no 'Annual Factors' table in this file")
        title, cols, rows = hits[0]
    else:
        raise ValueError(f"table must be 'first' or 'annual', got {table!r}")
    if not rows:
        raise SchemaError(f"empty French-library table {title!r}")
    widths = {len(r) for r in rows}
    if widths != {len(cols) + 1}:
        raise SchemaError(f"rows have {sorted(widths)} fields, header has {len(cols) + 1}")
    lengths = {len(r[0]) for r in rows}
    if len(lengths) != 1:
        raise SchemaError("mixed date codes in one table")
    idx = pd.DatetimeIndex([_stamp(r[0]) for r in rows], name="obs_date")
    try:
        vals = np.array([[float(x) for x in r[1:]] for r in rows], dtype=float)
    except ValueError as exc:
        raise SchemaError(f"non-numeric value: {exc}") from exc
    for m in _MISSING:
        vals[np.isclose(vals, m)] = np.nan
    f = pd.DataFrame(vals / 100.0, index=idx, columns=cols)
    if not f.index.is_monotonic_increasing or f.index.has_duplicates:
        raise SchemaError("dates are not strictly increasing")
    return f


def parse_french(payload: bytes, table: str = "first") -> pd.DataFrame:
    """Parse a library download: the zip as served, or the bare CSV bytes."""
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if len(names) != 1:
                raise SchemaError(f"expected one CSV in the zip, found {names}")
            raw = z.read(names[0])
    else:
        raw = payload
    return parse_french_csv(raw.decode("latin-1"), table=table)


SOURCE = Source("french", french_url, parse_french)
