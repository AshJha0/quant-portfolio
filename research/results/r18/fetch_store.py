"""Fill the local ``qp_data`` store from free, keyless sources (Round 18).

Plain idea: every series the Round 18 exploratory runs need is downloaded
once into ``data/store`` (git-ignored, content-addressed), and only the
SHA-256 of each download goes into git (``data_manifest.json``). Anyone can
rerun this script and check that they got the same bytes, or see that the
publisher revised the file.

Sources (all keyless, no account):

* ECB euro reference rates, ``ecb:EXR:<CCY>`` (ECB data portal);
* OECD Main Economic Indicators on FRED: 3-month interbank rates
  ``IR3TIB01<CC>M156N`` and CPI ``<ISO3>CPIALLMINMEI`` (OECD, CC BY 4.0);
* FRED public-domain series: ``DGS*``, ``DTB3``, ``SOFR``, ``CPIAUCSL``;
* NY Fed SOFR (``nyfed:SOFR``), ECB ESTR, BoE SONIA (``IUDSOIA``);
* US Treasury par yield curve 2024 (``ust:par``), CBOE VIX history;
* Kenneth French library ``F-F_Research_Data_Factors``.

Run: ``QP_DATA_ALLOW_NETWORK=1 PYTHONPATH=python/data/src python
research/results/r18/fetch_store.py``. Failures are recorded, not hidden.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pandas as pd
import qp_data

HERE = Path(__file__).resolve().parent

# Universe currency -> (OECD ISO2 used in IR3TIB01, ISO3 used in CPI). EUR uses the euro area.
OECD = {
    "USD": ("US", "USA"),
    "EUR": ("EZ", "EA19"),
    "JPY": ("JP", "JPN"),
    "GBP": ("GB", "GBR"),
    "CHF": ("CH", "CHE"),
    "SEK": ("SE", "SWE"),
    "NOK": ("NO", "NOR"),
    "DKK": ("DK", "DNK"),
    "AUD": ("AU", "AUS"),
    "CAD": ("CA", "CAN"),
    "NZD": ("NZ", "NZL"),
    "KRW": ("KR", "KOR"),
    "ZAR": ("ZA", "ZAF"),
    "CZK": ("CZ", "CZE"),
    "HUF": ("HU", "HUN"),
    "PLN": ("PL", "POL"),
    "SIT": ("SI", "SVN"),
    "SKK": ("SK", "SVK"),
    "EEK": ("EE", "EST"),
    "LVL": ("LV", "LVA"),
    "LTL": ("LT", "LTU"),
    "RUB": ("RU", "RUS"),
    "TRY": ("TR", "TUR"),
    "RON": ("RO", "ROU"),
    "CNY": ("CN", "CHN"),
    "IDR": ("ID", "IDN"),
    "BRL": ("BR", "BRA"),
    "MXN": ("MX", "MEX"),
    "INR": ("IN", "IND"),
    "ILS": ("IL", "ISR"),
    # HKD, SGD, MYR, PHP, THB, CYP, MTL, HRK, BGN: no OECD MEI interbank series; tried anyway.
    "HKD": ("HK", "HKG"),
    "SGD": ("SG", "SGP"),
    "MYR": ("MY", "MYS"),
    "PHP": ("PH", "PHL"),
    "THB": ("TH", "THA"),
    "CYP": ("CY", "CYP"),
    "MTL": ("MT", "MLT"),
    "HRK": ("HR", "HRV"),
    "BGN": ("BG", "BGR"),
}


# qp_core.holidays covers 2000-2099 only, and the store computes a release
# date for every observation, so nothing dated before 2000 can be stored.
START = dt.date(2000, 1, 1)


def keys() -> list[tuple[str, dt.date | None, dt.date | None]]:
    out: list[tuple[str, dt.date | None, dt.date | None]] = []
    for ccy in OECD:
        if ccy != "EUR":
            out.append((f"ecb:EXR:{ccy}", START, None))
        iso2, iso3 = OECD[ccy]
        out.append((f"fred:IR3TIB01{iso2}M156N", START, None))
        out.append((f"fred:{iso3}CPIALLMINMEI", START, None))
    out += [
        ("fred:CPIAUCSL", START, None),
        ("fred:DEUCPIALLMINMEI", START, None),
        ("fred:DTB3", START, None),
        ("fred:DGS2", START, None),
        ("fred:DGS10", START, None),
        ("fred:SOFR", None, None),
        ("nyfed:SOFR", dt.date(2023, 12, 1), dt.date(2025, 1, 31)),
        ("ecb:ESTR", dt.date(2019, 10, 1), None),
        ("boe:IUDSOIA", START, None),
        ("ust:par", None, dt.date(2024, 12, 31)),
        ("cboe:VIX", None, None),
        ("french:F-F_Research_Data_Factors", None, None),
    ]
    return out


def _one(item: tuple[str, dt.date | None, dt.date | None]) -> tuple[str, object]:
    key, start, end = item
    try:
        qp_data.release.rule_for(key)
        frame, payload, url = qp_data.fetch_frame(key, start, end, allow_network=True, timeout=180.0)
        return key, (frame, payload, url)
    except Exception as exc:
        return key, exc


def main() -> int:
    store = qp_data.Store()
    have = {e.series for e in store.manifest()}
    todo = [k for k in keys() if k[0] not in have]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for key, res in pool.map(_one, todo):
            if isinstance(res, Exception):
                print("FAIL", key, type(res).__name__, flush=True)
                continue
            frame, payload, url = cast("tuple[pd.DataFrame, bytes, str]", res)
            try:
                store.put(key, key.split(":", 1)[0], url, payload, frame)
            except Exception as exc:
                print("FAIL", key, type(exc).__name__, str(exc)[:120], flush=True)
                continue
            print("ok  ", key, flush=True)
    rows, failed = [], []
    latest = {e.series: e for e in store.manifest()}
    for key, _, _ in keys():
        e = latest.get(key)
        if e is None:
            failed.append({"series": key, "error": "not fetched (no data at the source or refused)"})
        else:
            rows.append(
                {
                    "series": key,
                    "sha256": e.sha256,
                    "url": e.url,
                    "retrieved_at": e.retrieved_at,
                    "n_rows": e.n_rows,
                    "first": e.first_obs,
                    "last": e.last_obs,
                }
            )
    doc = {
        "what": "SHA-256 of every Round 18 download; bytes live in data/store (git-ignored)",
        "store_root": "data/store",
        "fetched": rows,
        "failed": failed,
    }
    (HERE / "data_manifest.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(len(rows), "fetched,", len(failed), "missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
