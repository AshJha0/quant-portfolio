"""Content-addressed cache, manifest and bitemporal as-of queries.

Plain idea
----------
Every download is kept byte for byte under its SHA-256, so a study can name
exactly which bytes it used. Each download of a series is a *vintage*: what
the source said at the time it was retrieved. Asking for a series "as of"
some moment returns only observations already released by then, with the
values from the latest vintage retrieved by then. Two downloads that
disagree about the same date are a revision, and the as-of query keeps them
apart.

Formal statement
----------------
A vintage ``v`` has retrieval time ``k_v`` (knowledge time) and rows
``(t, x_v(t), r(t))``: observation date, value, release date. For a query
time ``a``,

    get(s, a) = { (t, x_{v*}(t)) : r(t) <= date(a) },
    v* = argmax_{v : k_v <= a} k_v.

If no vintage was retrieved by ``a`` (data first fetched later than the
moment being simulated), ``strict=True`` raises :class:`NoVintage`; the
default uses the earliest vintage and sets ``frame.attrs["revision_risk"] =
True``, because a value retrieved later may include revisions that were not
known at ``a``. This is the bitemporal model of Snodgrass (1999), with valid
time = observation date and transaction time = retrieval time.

Layout on disk (``root`` defaults to ``data/store`` in the repository, git-ignored)::

    objects/ab/abcdef...            raw response bytes, named by SHA-256
    series/<key>/<sha>.<ext>        parsed vintage (parquet if pyarrow, else csv)
    manifest.jsonl                  one JSON line per vintage

Worked example (no network)::

    >>> import datetime as dt, tempfile, pandas as pd
    >>> st = Store(tempfile.mkdtemp())
    >>> def frame(x):
    ...     return pd.DataFrame({"value": [x]}, index=pd.DatetimeIndex(["2024-03-01"], name="obs_date"))
    >>> _ = st.put("fred:CPIAUCSL", "fred", "u", b"v1", frame(312.2), retrieved_at="2024-04-30T00:00:00+00:00")
    >>> _ = st.put("fred:CPIAUCSL", "fred", "u", b"v2", frame(312.3), retrieved_at="2024-05-31T00:00:00+00:00")
    >>> float(st.get("fred:CPIAUCSL", as_of="2024-05-15")["value"].iloc[0])
    312.2
    >>> float(st.get("fred:CPIAUCSL", as_of="2024-06-01")["value"].iloc[0])
    312.3
    >>> len(st.get("fred:CPIAUCSL", as_of="2024-04-30T12:00:00+00:00", strict=True))  # 2024-04-22 release
    1
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import NoVintage, SchemaError
from .release import release_date

__all__ = ["DEFAULT_ROOT", "SCHEMA_VERSION", "ManifestEntry", "Store", "sha256"]

SCHEMA_VERSION = 1
DEFAULT_ROOT = Path(__file__).resolve().parents[4] / "data" / "store"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc(ts: str | dt.datetime | dt.date | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _as_of_ts(as_of: str | dt.datetime | dt.date | pd.Timestamp) -> pd.Timestamp:
    """A bare date means the end of that day (UTC): everything known that day."""
    t = pd.Timestamp(as_of)
    if (isinstance(as_of, dt.date) and not isinstance(as_of, dt.datetime)) or (
        isinstance(as_of, str) and len(as_of) == 10
    ):
        t = t + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return _utc(t)


def _parquet_ok() -> bool:
    return importlib.util.find_spec("pyarrow") is not None


@dataclass(frozen=True)
class ManifestEntry:
    """One vintage. ``as_of`` is the knowledge time (= ``retrieved_at``);
    ``release_date`` is the release date of the last observation in it."""

    series: str
    source: str
    url: str
    sha256: str
    retrieved_at: str
    as_of: str
    release_date: str | None
    first_obs: str | None
    last_obs: str | None
    n_rows: int
    schema_version: int
    path: str


class Store:
    """A directory holding raw bytes, parsed vintages and the manifest."""

    def __init__(self, root: str | Path = DEFAULT_ROOT, *, fmt: str | None = None) -> None:
        self.root = Path(root)
        self.fmt = fmt or ("parquet" if _parquet_ok() else "csv")
        if self.fmt not in ("parquet", "csv"):
            raise ValueError("fmt must be 'parquet' or 'csv'")

    # ------------------------------------------------------------ raw bytes --
    def object_path(self, digest: str) -> Path:
        return self.root / "objects" / digest[:2] / digest

    def put_bytes(self, payload: bytes) -> str:
        digest = sha256(payload)
        p = self.object_path(digest)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(payload)
        return digest

    def get_bytes(self, digest: str) -> bytes:
        payload = self.object_path(digest).read_bytes()
        if sha256(payload) != digest:
            raise SchemaError(f"object {digest} is corrupt (hash mismatch)")
        return payload

    # ------------------------------------------------------------- vintages --
    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    def manifest(self, series: str | None = None) -> list[ManifestEntry]:
        if not self.manifest_path.exists():
            return []
        out = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = ManifestEntry(**json.loads(line))
                if series is None or e.series == series:
                    out.append(e)
        return out

    def put(
        self,
        series: str,
        source: str,
        url: str,
        payload: bytes,
        frame: pd.DataFrame,
        *,
        retrieved_at: str | dt.datetime | None = None,
    ) -> ManifestEntry:
        """Store one vintage: raw bytes, the parsed frame and a manifest line.

        ``frame`` must have a DatetimeIndex of observation dates and float
        columns. A ``release_date`` column is added from the series' rule.
        Storing the same bytes twice for a series adds no second vintage.
        """
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise SchemaError("frame must be indexed by observation date")
        digest = self.put_bytes(payload)
        for e in self.manifest(series):
            if e.sha256 == digest:
                return e
        k = _utc(retrieved_at or dt.datetime.now(dt.UTC))
        f = frame.sort_index().copy()
        idx = pd.DatetimeIndex(f.index)
        f.index = idx.tz_localize(None) if idx.tz is not None else idx
        f.index.name = "obs_date"
        rel = [release_date(series, d.date()) for d in f.index]
        f["release_date"] = pd.to_datetime(rel)
        sdir = self.root / "series" / _SAFE.sub("_", series)
        sdir.mkdir(parents=True, exist_ok=True)
        path = sdir / f"{digest}.{self.fmt}"
        if self.fmt == "parquet":
            f.to_parquet(path)
        else:
            f.to_csv(path)
        entry = ManifestEntry(
            series=series,
            source=source,
            url=url,
            sha256=digest,
            retrieved_at=k.isoformat(),
            as_of=k.isoformat(),
            release_date=str(max(rel)) if rel else None,
            first_obs=str(f.index.min().date()) if len(f) else None,
            last_obs=str(f.index.max().date()) if len(f) else None,
            n_rows=len(f),
            schema_version=SCHEMA_VERSION,
            path=path.relative_to(self.root).as_posix(),
        )
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
        return entry

    def load(self, entry: ManifestEntry) -> pd.DataFrame:
        path = self.root / entry.path
        if path.suffix == ".parquet":
            f = pd.read_parquet(path)
        else:
            f = pd.read_csv(path, index_col="obs_date", parse_dates=["obs_date", "release_date"])
        if entry.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"{entry.path}: schema {entry.schema_version}, expected {SCHEMA_VERSION}")
        return f

    def get(
        self,
        series: str,
        as_of: str | dt.datetime | dt.date | pd.Timestamp | None = None,
        *,
        strict: bool = False,
    ) -> pd.DataFrame:
        """Observations of ``series`` known at ``as_of`` (latest if ``None``).

        A bare date means the end of that day. See the module docstring.
        """
        entries = sorted(self.manifest(series), key=lambda e: e.retrieved_at)
        if not entries:
            raise NoVintage(f"nothing stored for {series!r}")
        attrs: dict[str, Any] = {"series": series, "revision_risk": False}
        if as_of is None:
            chosen = entries[-1]
            f = self.load(chosen)
        else:
            a = _as_of_ts(as_of)
            known = [e for e in entries if _utc(e.retrieved_at) <= a]
            if known:
                chosen = known[-1]
            elif strict:
                raise NoVintage(
                    f"{series}: first vintage retrieved {entries[0].retrieved_at}, after as_of {a}"
                )
            else:
                chosen = entries[0]
                attrs["revision_risk"] = True
            f = self.load(chosen)
            f = f[f["release_date"] <= pd.Timestamp(a.date())]
        attrs["sha256"] = chosen.sha256
        attrs["retrieved_at"] = chosen.retrieved_at
        f = f.copy()
        f.attrs.update(attrs)
        return f
