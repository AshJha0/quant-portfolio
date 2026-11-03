"""SOFR 2024 check: compounding NY Fed SOFR fixings against the NY Fed SOFR Index.

Plain idea: the New York Fed publishes both the daily SOFR and a SOFR Index
that compounds those same fixings since 2 April 2018. Over any period, the
compounded-in-arrears rate built from the daily fixings must equal the rate
implied by the ratio of the two index values, up to the index's rounding
(8 decimals). If our compounding (or the W13a overnight leg) disagrees, it
is wrong.

Formal statement (ARRC 2020 conventions, no lookback, ACT/360). For a period
``[d_0, d_n)`` with SOFR publication dates ``d_0 < ... < d_{n-1}``, weights
``n_i = d_{i+1} - d_i`` calendar days (``d_n`` = period end):

    R = (prod_i (1 + r_i n_i / 360) - 1) * 360 / (d_n - d_0)
    R_index = (I(d_n) / I(d_0) - 1) * 360 / (d_n - d_0)

The realised 1-year OIS par rate for a swap starting 2024-01-02 with one
annual payment is the ``R`` of that year (zero NPV ex post on the fixings).

Worked example: the calendar months of 2024, the full year, and the swap
year are each checked; ``|R - R_index|`` is reported in basis points.

Seam: if ``fi_rates.legs.compounded_rate`` (W13a, lane r09-l1-rates) is
importable, the same periods are priced through it too and compared.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import qp_data

HERE = Path(__file__).resolve().parent
INDEX_URL = (
    "https://markets.newyorkfed.org/api/rates/secured/sofrai/search.json"
    "?startDate=2023-12-01&endDate=2025-01-31"
)


def sofr_index() -> tuple[pd.Series, str]:
    with urllib.request.urlopen(INDEX_URL, timeout=120) as r:
        payload = r.read()
    rows = json.loads(payload)["refRates"]
    s = pd.Series({pd.Timestamp(x["effectiveDate"]): float(x["index"]) for x in rows}).sort_index()
    return s, hashlib.sha256(payload).hexdigest()


def compound(fix: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """``fix``: decimal SOFR indexed by publication (effective) date."""
    d = fix.loc[(fix.index >= start) & (fix.index < end)]
    if d.index[0] != start:
        raise ValueError(f"{start.date()} is not a SOFR date")
    bounds = [*d.index[1:], end]
    g = 1.0
    for (_, r), b, a in zip(d.items(), bounds, d.index, strict=True):
        g *= 1.0 + r * (b - a).days / 360.0
    return (g - 1.0) * 360.0 / (end - start).days


def periods(dates: pd.DatetimeIndex) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    out = []
    for m in range(1, 13):
        s = dates[dates >= pd.Timestamp(2024, m, 1)][0]
        e = dates[dates >= (pd.Timestamp(2025, 1, 1) if m == 12 else pd.Timestamp(2024, m + 1, 1))][0]
        out.append((f"2024-{m:02d}", s, e))
    out.append(
        (
            "calendar 2024",
            dates[dates >= pd.Timestamp("2024-01-01")][0],
            dates[dates >= pd.Timestamp("2025-01-01")][0],
        )
    )
    out.append(("1Y OIS 2024-01-02 to 2025-01-02", pd.Timestamp("2024-01-02"), pd.Timestamp("2025-01-02")))
    return out


def w13a(fix: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    try:
        from fi_rates import fixings as fx  # type: ignore[import-not-found]
        from fi_rates.legs import compounded_rate  # type: ignore[import-not-found]
    except ImportError:
        return None
    store = fx.FixingStore()
    store.add_many("SOFR", {pd.Timestamp(str(d)).date(): float(r) for d, r in fix.items()})
    r, share = compounded_rate(fx.SOFR, start.date(), end.date(), None, store, dt.date(2025, 6, 30))
    assert share == 1.0
    return float(r)


def main() -> dict:
    f = qp_data.get("nyfed:SOFR")
    fix = (f["value"] / 100.0).sort_index()
    fix.index = pd.DatetimeIndex(fix.index)
    idx, idx_sha = sofr_index()
    rows = []
    for name, s, e in periods(pd.DatetimeIndex(idx.index)):
        r = compound(fix, s, e)
        ri = (idx[e] / idx[s] - 1.0) * 360.0 / (e - s).days
        w = w13a(fix, s, e)
        rows.append(
            {
                "period": name,
                "start": str(s.date()),
                "end": str(e.date()),
                "compounded": r,
                "from_index": ri,
                "diff_bp": (r - ri) * 1e4,
                "w13a": w,
                "w13a_diff_bp": None if w is None else (w - r) * 1e4,
            }
        )
    worst = max(abs(x["diff_bp"]) for x in rows)
    return {
        "label": "verification (not a research result)",
        "sofr_sha256": f.attrs["sha256"],
        "sofr_index_url": INDEX_URL,
        "sofr_index_sha256": idx_sha,
        "tolerance_bp": 0.01,
        "max_abs_diff_bp": float(worst),
        "pass": bool(worst < 0.01),
        "w13a_available": rows[0]["w13a"] is not None,
        "periods": rows,
    }


if __name__ == "__main__":
    out = main()
    name = "sofr_2024_check_w13a.json" if out["w13a_available"] else "sofr_2024_check.json"
    (HERE / name).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for p in out["periods"]:
        print(
            f"{p['period']:32s} {p['compounded']:.8f} {p['from_index']:.8f} {p['diff_bp']:+.5f}bp"
            f" w13a {p['w13a_diff_bp']}"
        )
    print("pass", out["pass"], "max", out["max_abs_diff_bp"])
    sys.exit(0 if out["pass"] else 1)
