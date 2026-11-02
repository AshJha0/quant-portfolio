# Forward log: what the forward-hold-out registrations need from it

Owner: the pre-registration lane. The log itself, its schema and its job are
documented in `research/forward/README.md` (schema 1) and written by
`research/forward/forward.py`. This file says what the two forward-hold-out
registrations (`prereg/pairs-fx-g10-v2.md`, `prereg/momentum-fx-g10.md`)
rely on. Field names below are schema 1's.

## Why it matters to these registrations

Their hold-out is data that does not exist yet: ECB fixes from 2026-11-03
onward. A forward hold-out is only worth something if nobody can later pick
which days count or what the rates were. The log writes each day once, on
the day, with the hash of the response it came from, as a commit on the
`forward-log` branch.

## What the registrations use from schema 1

| Field | Used for |
|---|---|
| `date` | the scored day; every scored day is >= 2026-11-03 (the log's guard) |
| `usd_per_unit` | the nine USD prices both registrations trade |
| `rates_per_eur` | re-deriving `usd_per_unit` as a check |
| `provenance.data_sha256`, `provenance.source` | tying each rate to the ECB response |
| `provenance.code_commit`, `provenance.written_utc` | showing the row was written on the day |

The log's rules that the registrations depend on, all already in schema 1:
one line per ECB date, never rewritten (first value wins on an ECB
revision); no line dated before 2026-11-03; a day's line uses only that day
and earlier lines.

Not used: the `regime` and `pairs` blocks. They are the signed
registrations' rules, whose 3- and 4-year history requirements are counted
on forward data only, so they show `warming_up` until 2030 and 2031.

## Formation and lookback history

A rule that needs a 3-year formation window cannot score on forward data
alone until about 2030. The two new registrations state their history rule
explicitly:

- One ECB file of fixes 1999-01-04 to 2026-11-02 (the nine currencies plus
  USD, same request format as `data/fetch/qp_fetch.py`), fetched once at
  signing, after the signed registrations' 2022-2025 hold-out has been
  run. Its SHA-256 goes in the signing commit; the file is stored as
  `research/forward/history_1999-01-04_2026-11-02.csv` with its `.sha256`.
- Formation windows (pairs v2) and lookbacks (momentum) read that file
  followed by the log, as one contiguous series of ECB business days.
- No fix dated before 2026-11-03 is ever scored.

The history file is not part of the log and the log job never fetches it;
the log's own guard stays as it is.

## Strategy fields: not needed in the log

Schema 1 carries no `momentum` or `pairs_v2` block, and the registrations do
not need one: the evaluation recomputes every signal and return from
`usd_per_unit` and the history file, under the registration's code
(`research/studies/forward_studies.py: momentum_weights`, `momentum_returns`,
`pair_returns_level`, `select_pairs`). If daily readability is wanted later,
a schema 2 may add them as a convenience; they would never be the source of
truth.

## Evaluation

Both registrations score 2026-11-03 to 2028-11-02 once, on or after
2028-11-13. The evaluation script, `research/forward/evaluate.py`,
gives a verdict only when every check below passes, today is on or after
2028-11-13 and the registration is signed; otherwise it prints interim
numbers labelled INTERIM, decides nothing and writes no result. It will

1. read `forward/log.jsonl` at a stated `forward-log` commit and the
   history file, and check the history file's SHA-256 against the signing
   commit;
2. refuse if any date in 2026-11-03 to 2028-11-02 that the ECB published is
   missing from the log, or any `usd_per_unit` disagrees with
   `rates_per_eur`;
3. recompute the daily returns under each registration and write one result
   JSON with `prereg`, `prereg_sha256`, `period`, `mode: "forward"`, the
   log commit and the history hash.

`prereg.py check` treats any period ending on or after 2022-01-01 as reaching the
signed hold-out window, so a forward result will only pass once its
registration is signed, which is the intended guard.
