"""SQL pack: every reference query returns what an independent pandas computation does."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

PACK = Path(__file__).resolve().parents[1] / "sql"
QUERIES = dict(re.findall(r"^### (S\d\d)\n```sql\n(.*?)```",
                          (PACK / "keys" / "answers.md").read_text("utf-8"), re.M | re.S))


@pytest.fixture(scope="module")
def db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript((PACK / "fixtures.sql").read_text("utf-8"))
    return con


@pytest.fixture(scope="module")
def frames(db: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    return {n: pd.read_sql(f"SELECT * FROM {n}", db) for n in ("trades", "quotes", "closes")}


def _rows(obj: Any) -> list[Any]:
    """Rows as ``Any``: pandas-stubs types cell values too narrowly for these checks."""
    return list(obj.itertuples() if hasattr(obj, "itertuples") else obj)


def _signed(tr: pd.DataFrame) -> pd.Series:
    return tr["qty"].where(tr["side"] == "B", -tr["qty"])


def _prevailing(tr: pd.DataFrame, qu: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in tr.iterrows():
        q = qu[(qu["sym"] == r["sym"]) & (qu["t"] <= r["t"])]
        if q.empty:
            rows.append((r["id"], None, None))
        else:
            last = q.loc[q["t"].idxmax()]
            rows.append((r["id"], last["bid"], last["ask"]))
    return pd.DataFrame(rows, columns=["id", "bid", "ask"])


def _s01(f: dict[str, pd.DataFrame]) -> list[tuple[object, ...]]:
    t = f["trades"].assign(n=lambda x: x.px * x.qty).groupby("sym")[["qty", "n"]].sum()
    return [(s, int(r.qty), round(r.n / r.qty, 4)) for s, r in _rows(t.iterrows())]


def _s02(f: dict[str, pd.DataFrame]) -> list[tuple[object, ...]]:
    t = f["trades"].assign(s=_signed(f["trades"])).groupby("sym")["s"].sum()
    return [(s, int(v)) for s, v in _rows(t.items())]


def _s03(f: dict[str, pd.DataFrame]) -> list[tuple[object, ...]]:
    tr = f["trades"].sort_values(["t", "id"]).assign(s=lambda x: _signed(x))
    tr["pos"] = tr.groupby("sym")["s"].cumsum()
    return [(int(r.id), r.sym, int(r.pos)) for r in _rows(tr.sort_values("id"))]


def _s04(f: dict[str, pd.DataFrame]) -> list[tuple[object, ...]]:
    p = _prevailing(f["trades"], f["quotes"]).sort_values("id")
    return [(int(r.id), r.bid, r.ask) for r in _rows(p)]


def _s05(f: dict[str, pd.DataFrame]) -> list[tuple[object, ...]]:
    p = _prevailing(f["trades"], f["quotes"]).merge(f["trades"], on="id").dropna(subset=["bid"])
    out: list[tuple[object, ...]] = []
    for r in _rows(p.sort_values("id")):
        mid = (r.bid + r.ask) / 2
        sgn = 1 if r.side == "B" else -1
        out.append((int(r.id), round(sgn * (r.px - mid) / mid * 10000, 2)))
    return out


def _s06(f: dict[str, pd.DataFrame]) -> list[tuple[object, ...]]:
    best = f["trades"].sort_values(["qty", "t"], ascending=[False, True]).groupby("sym").head(1)
    return [(r.sym, int(r.id)) for r in _rows(best.sort_values("sym"))]


def _s07(f: dict[str, pd.DataFrame]) -> list[tuple[object, ...]]:
    c = f["closes"].sort_values(["sym", "d"]).copy()
    c["ret"] = c.groupby("sym")["close"].pct_change()
    return [(r.sym, r.d, round(r.ret, 6)) for r in _rows(c.dropna())]


def _s08(f: dict[str, pd.DataFrame]) -> list[tuple[object, ...]]:
    c = f["closes"]
    grid = pd.MultiIndex.from_product([sorted(c.sym.unique()), sorted(c.d.unique())])
    have = set(zip(c.sym, c.d, strict=True))
    return [g for g in grid if g not in have]


def _s09(f: dict[str, pd.DataFrame]) -> list[tuple[object, ...]]:
    tr = f["trades"].sort_values(["t", "id"]).copy()
    tr["ma3"] = tr.groupby("sym")["px"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return [(int(r.id), round(r.ma3, 4)) for r in _rows(tr.sort_values("id"))]


def _s10(f: dict[str, pd.DataFrame]) -> list[tuple[object, ...]]:
    q = f["quotes"].assign(b=lambda x: np.floor_divide(x.t, 30))
    g = q.groupby(["sym", "b"]).size()
    return [(s, int(b), int(n)) for (s, b), n in _rows(g.items())]


EXPECTED: dict[str, Callable[[dict[str, pd.DataFrame]], list[tuple[object, ...]]]] = {
    "S01": _s01, "S02": _s02, "S03": _s03, "S04": _s04, "S05": _s05,
    "S06": _s06, "S07": _s07, "S08": _s08, "S09": _s09, "S10": _s10,
}


def test_ten_tasks_each_with_a_query() -> None:
    q = re.findall(r"^- \*\*(S\d\d)\*\*", (PACK / "questions.md").read_text("utf-8"), re.M)
    assert len(q) == 10 and set(q) == set(QUERIES) == set(EXPECTED)


def _norm(v: object) -> object:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, float):
        return round(v, 6)
    return v


@pytest.mark.parametrize("sid", sorted(EXPECTED))
def test_query_matches_pandas(sid: str, db: sqlite3.Connection,
                              frames: dict[str, pd.DataFrame]) -> None:
    got = [tuple(_norm(v) for v in row) for row in db.execute(QUERIES[sid]).fetchall()]
    want = [tuple(_norm(v) for v in row) for row in EXPECTED[sid](frames)]
    assert got == want


def test_asof_join_does_not_look_ahead(db: sqlite3.Connection) -> None:
    rows = dict((i, (b, a)) for i, b, a in db.execute(QUERIES["S04"]).fetchall())
    assert rows[9] == (None, None)  # CCC trades at 50, first quote at 55
