"""Point-in-time FX universe (W6c), read from ``research/universe/fx.csv``.

Plain idea: a study that ranks currencies in 2005 must only rank currencies
that existed and were quoted in 2005. Using today's list drops the currencies
that later joined the euro or stopped being quoted, which is survivorship.

Formal statement. Each row gives ``first`` (or ``auto``: the first non-missing
observation in the price file), ``last`` (empty: still live), a ``regime``
(``float`` or ``peg``) and a sourced ``reason``. ``members(as_of)`` is the set
with ``first <= as_of <= last``; studies call it at each rebalance.
``assert_live`` raises if a weight is non-zero outside a currency's window.
``include_pegs=False`` drops the ``peg`` rows, so every result can be reported
with and without them.

Worked example: ``members("2008-06-30")`` holds SKK (euro on 2009-01-01) but
not SIT (euro on 2007-01-01).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "universe" / "fx.csv"


class UniverseViolation(ValueError):
    """A study held a currency outside its live window."""


@dataclass(frozen=True)
class Universe:
    table: pd.DataFrame  # index ccy; columns first, last (Timestamp or NaT), regime, reason

    @classmethod
    def load(cls, prices: pd.DataFrame | None = None, path: Path = CSV) -> Universe:
        t = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("ccy")
        firsts: list[pd.Timestamp | None] = []
        for ccy, v in t["first"].items():
            if v == "auto":
                if prices is None or ccy not in prices.columns:
                    firsts.append(None)
                else:
                    fv = prices[ccy].first_valid_index()
                    firsts.append(None if fv is None else pd.Timestamp(str(fv)))
            else:
                firsts.append(pd.Timestamp(v))
        t["first"] = pd.to_datetime(pd.Series(firsts, index=t.index))
        t["last"] = pd.to_datetime(t["last"].replace("", None))
        return cls(t)

    def members(self, as_of: str | pd.Timestamp, include_pegs: bool = True) -> list[str]:
        d = pd.Timestamp(as_of)
        t = self.table
        ok = t["first"].notna() & (t["first"] <= d) & (t["last"].isna() | (d <= t["last"]))
        if not include_pegs:
            ok &= t["regime"] != "peg"
        return sorted(t.index[ok])

    def mask(self, index: pd.DatetimeIndex, columns: list[str],
             include_pegs: bool = True) -> pd.DataFrame:
        """Boolean dates x currencies: True where the currency is a member."""
        out = pd.DataFrame(False, index=index, columns=columns)
        for c in columns:
            if c not in self.table.index:
                continue
            row = self.table.loc[c]
            if not include_pegs and row["regime"] == "peg":
                continue
            if pd.isna(row["first"]):
                continue
            live = index >= row["first"]
            if pd.notna(row["last"]):
                live &= index <= row["last"]
            out[c] = live
        return out

    def assert_live(self, weights: pd.DataFrame) -> None:
        """Raise :class:`UniverseViolation` if any non-zero weight is outside its window."""
        m = self.mask(pd.DatetimeIndex(weights.index), list(weights.columns))
        bad = (weights.fillna(0.0) != 0.0) & ~m
        if bad.to_numpy().any():
            where = bad.stack()
            hits = where[where.to_numpy(dtype=bool)]
            first = hits.index[0]
            raise UniverseViolation(f"{first[1]} held on {first[0].date()} outside its live window")
