# qp-data

One data layer for the quant-portfolio projects (plan workstream W9).
Distribution name `qp-data`, import name `qp_data`. It depends on pandas and
`qp-core` (for the settlement calendars); `pyarrow` is optional and, when
installed, makes the store write parquet instead of CSV.

```
pip install -e python/core -e python/data
python -m pytest -p no:cacheprovider python/data/tests --doctest-modules python/data/src/qp_data
```

## The idea in one paragraph

A backtest that reads a number before the day it was published is looking
ahead, and a result that cannot say which bytes it read cannot be checked.
`qp_data` fixes both. Every download is kept under its SHA-256 and recorded
in a manifest (source, URL, retrieval time, as-of time, release date, schema
version). Every observation gets a release date from the issuer's release
calendar. `get(series, as_of=t)` then returns only what had been released by
`t`, with the values from the latest download made by `t`.

## Formal statement

For a series with release rule `r(.)` and stored vintages `v` retrieved at
`k_v`, the as-of query at time `a` returns `{(t, x_v*(t)) : r(t) <= a}` with
`v* = argmax {k_v : k_v <= a}`. Valid time is the observation date, transaction
time the retrieval time: the bitemporal model of Snodgrass (1999), as used by
ArcticDB and in Lopez de Prado, *Advances in Financial Machine Learning*,
ch. 2. If nothing was retrieved by `a`, `strict=True` raises; the default
returns the first vintage and marks the frame `attrs["revision_risk"] = True`.

## Worked example

```python
import datetime as dt
import qp_data

qp_data.fetch("fred:DGS10", dt.date(2024, 6, 24), dt.date(2024, 7, 8), allow_network=True)
qp_data.get("fred:DGS10", as_of=dt.date(2024, 7, 2)).index.max()
# Timestamp('2024-07-01'): the 2 July yield is published on 3 July (H.15, next business day)

qp_data.release_date("fred:DEXUSEU", dt.date(2024, 7, 1))   # 2024-07-08, the weekly H.10 Monday
qp_data.check_release("fred:FEDFUNDS", [dt.date(2024, 6, 1)], [dt.date(2024, 6, 28)])
# ValueError: the June monthly average is not known until 1 July
```

## Sources

All keyless; nothing needs an account, a key or payment details.

| Key | Source | Release rule |
|---|---|---|
| `fred:<ID>` | FRED `fredgraph.csv`, public-domain allow-list (`DEX*`, `DGS*`, `DTB*`, `DFF`, `SOFR`, `FEDFUNDS`, `CPIAUCSL`) | H.15 next US business day; H.10 weekly Monday; CPI bounded at the 20th of the next month, shutdown delays overridden; FEDFUNDS the business day after month end |
| `ecb:EXR:<CCY>` | ECB Data Portal reference rates | same TARGET2 day (14:15 CET fixing) |
| `ecb:ESTR` | ECB Data Portal, euro short-term rate | next TARGET2 day |
| `frankfurter:<BASE>[:<SYMS>]` | api.frankfurter.dev (ECB rates) | same TARGET2 day |
| `nyfed:SOFR` | NY Fed Markets Data API | next US business day |
| `boe:IUDSOIA` | Bank of England IADB (SONIA) | next London business day |
| `ust:par` | US Treasury daily par yield curve | same US business day |
| `cboe:VIX` | CBOE VIX history | same US business day |
| `coinbase:<PRODUCT>:1d` | Coinbase Exchange public candles | next UTC day |
| `binance:<SYMBOL>:1d` | Binance public klines (`data-api.binance.vision`) | next UTC day |

The sources and the evidence for each rule are in the docstrings of
`qp_data.release` and `qp_data.sources`. A series with no rule cannot be
fetched (`UnknownReleaseRule`): add one with `release.register_rule` from the
issuer's calendar first.

**What is not here.** No ALFRED vintages: the FRED API key is not used
(owner decision D6), so the first-published value of a revised series is
not recoverable from FRED; only revisions seen by this store's own downloads
are kept. Release *times* inside the day are not modelled; a value released
on day `r` counts as known on `r`, so trade it at `r + 1` (the t+1 fill
rule). Not yet wired: Fed GSW curves, the Kenneth French library, OECD, BIS,
SEC EDGAR, Verdelhan's portfolios, and the crypto websockets (W25/W30).

## Network and tests

Network access is off unless the caller passes `allow_network=True` or sets
`QP_DATA_ALLOW_NETWORK=1`. Downloads retry on timeouts, dropped connections,
HTTP 429 and 5xx with waits of 0.5, 1 and 2 seconds; other HTTP errors fail
at once. The tests run every parser on responses recorded on 2026-11-03 in
`tests/fixtures/` and never open a socket. `tests/test_live.py` hits the
real endpoints and runs only with `QP_DATA_LIVE_TESTS=1`.

## The old fetchers

The per-project `data/live.py` fetchers in fx_options, fx_vol, fx_var,
fx_rates, fx_pairs, fx_port and fi_rates now delegate to `qp_data.legacy`,
warn with `DeprecationWarning`, and are removed in v4.0.0. FRED no longer
needs an API key in fx_rates and fi_rates.
