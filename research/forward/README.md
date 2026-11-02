# Forward log

A daily record of the ECB reference rates from 2026-11-03 onward, with the
two registered strategies scored on that record and nothing else.

## Why it exists

The 2022-2025 hold-out of `pairs-fx-g10.md` and `regime-fx-basket.md` can be
scored once. After that, the only data nobody has looked at is data that has
not happened yet. If we wait and download it in a few years, we will know the
headline numbers before we write the evaluation, and the temptation to adjust
is real. Writing each day down as it arrives, with a hash and a commit, takes
that choice away: the log shows what the rules said on the day, in public, in
a git history anyone can check.

The log is also the hold-out for any new pre-registration whose evaluation
window is forward data (see `research/forward/SPEC.md`).

## Where it lives

- Code: `research/forward/forward.py` on `main`.
- Log: `forward/log.jsonl` on the branch `forward-log`. That branch has no
  shared history with `main` and no protection; only the workflow writes to
  it. `main` is never written by the job.
- Job: `.github/workflows/forward-log.yml`, weekdays 16:30 UTC (after the
  ECB fix, published about 16:00 CET). Permission `contents: write` only;
  actions pinned to commit SHAs; one run at a time (concurrency group
  `forward-log`, no cancelling).

## The guard

The script never requests a date before 2026-11-03 and raises
`ForwardGuardError` if a request or a response would include one. The
training years and the single-use 2022-2025 hold-out cannot enter the log,
and the strategies only ever see what the log holds. Tests check the refusal
happens before any download.

## What one line says

Each line is one ECB business day, a JSON object with sorted keys:

| Field | Meaning |
|---|---|
| `schema` | format version, currently 1 |
| `date` | the ECB reference date |
| `rates_per_eur` | the nine ECB rates as published, units of currency per EUR |
| `usd_per_unit` | USD value of one unit of each currency (EUR included), as the studies use them |
| `regime` | the regime rule's state at that close (below) |
| `pairs` | the pairs rule's state at that close (below) |
| `provenance.code_commit` | the `main` commit that wrote the line |
| `provenance.data_sha256` | SHA-256 of the ECB response bytes the line came from |
| `provenance.source` | the request URL |
| `provenance.written_utc` | when the line was written |
| `provenance.prereg_sha256` | registration hashes of the two pre-registrations |

`regime` and `pairs` read `{"status": "warming_up", "scorable_from_year": Y}`
until the registered rule has the history it needs, counted in whole calendar
years of forward data:

- regime: the HMM for year Y is fitted on all forward days before 1 January
  of Y, and the registered training run left four years between the first
  data and the first test year (1999 to 2003). Forward data starts in
  November 2026, so the first scored year is 2031.
- pairs: the formation window is the three calendar years before Y, so the
  first scored year is 2030.

Once scored:

- `regime`: `p_high` (filtered probability of the high-volatility state at
  that close), `high_vol` (`p_high > 0.5`; this classifies the *next* day's
  basket return), `basket_return` (that day's basket log return) and
  `fit_through` (last day the year's fit saw).
- `pairs`: `formation` (first and last formation day), `selected` (pairs
  chosen for the year), `held` (pairs with a position at that close: z,
  beta and the two leg weights) and `net_return` (equal-weight portfolio net
  return earned that day by the positions held at the previous close, after
  1 bp per leg traded).

Rules for reading it:

- One line per date, never rewritten. If the ECB later revises a rate, the
  logged value stays; the first value wins.
- A day's line uses only that day and earlier lines. A test checks that
  writing the log in two sittings gives the same signals as one.
- The scores can be recomputed from the log's rates alone; the scored fields
  are a convenience and a check, not the source of truth.

## Worked example

Read the log into pandas and sum the pairs returns for a year:

```python
import json, pandas as pd
rows = [json.loads(x) for x in open("forward/log.jsonl")]
pairs = pd.Series({r["date"]: r["pairs"].get("net_return") for r in rows}).dropna()
print(pairs.loc["2030"].sum())
```

## Running it locally

```bash
python research/forward/forward.py update --log /tmp/forward/log.jsonl
python research/forward/forward.py update --log /tmp/f.jsonl --csv some_ecb_csvdata.csv
python -m pytest research/tests/test_forward.py
```

The tests use synthetic data dated 2026-11-03 to 2031-03-31 and never touch
the network. They check: the guard on requests and responses, idempotence,
column-order independence, that the pairs positions and returns match
`fx_studies.pair_returns` and `fx_studies.pairs_study` exactly, and that the
regime signal on a day is unchanged when later data is removed.

## Turning it on

The job only runs when the repository variable `FORWARD_LOG_ENABLED` is
`true` (Settings, Secrets and variables, Actions, Variables). Until the
coordinator sets it, a scheduled run is skipped, so merging the workflow does
not start the log. The first real run creates the `forward-log` branch as an
orphan. To pause the log, set the variable to anything else.
