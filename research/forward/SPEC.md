# Forward log: what it records and why

Owner of this file: the prereg2 lane (pre-registrations with a forward
hold-out). The forward lane's code (`research/forward/forward.py` and the
daily workflow) writes the log; this file says what the registrations need
from it. If the two disagree, this file is the contract and the code is
fixed.

## Why the log exists

Two registrations (`prereg/pairs-fx-g10-v2.md`, `prereg/momentum-fx-g10.md`)
hold out data that does not exist yet: ECB fixes from 2026-11-03 onward.
A forward hold-out is only worth something if nobody can later quietly pick
which days count. So each day's fix is written down once, on the day, with
the hash of the file it came from, on a branch (`forward-log`) where every
append is a commit with a timestamp. At evaluation the returns are
recomputed from those logged fixes, not from a fresh download.

## Hard rules

1. No row is dated before 2026-11-03. The fetcher requests nothing earlier
   (this also keeps the 2022-2025 hold-out of the signed registrations
   untouched).
2. One row per ECB business day, appended once. Re-running for a date that
   is already in the log changes nothing (idempotent); a fix that differs
   from the logged one is an error, reported, never overwritten.
3. The log is append-only. A correction is a new row type, never an edit.
4. The fixes and their hashes are the record. Signal fields are a
   convenience for reading the log day by day; the evaluation recomputes
   every signal and return from the logged fixes under the registration's
   rules and fails if a logged signal disagrees.

## Record (one JSON object per line, `forward-log` branch)

Fields the registrations rely on, with the names `forward.py` uses:

| Field | Type | Meaning |
|---|---|---|
| `schema` | int | record version; bump on any field change |
| `date` | `YYYY-MM-DD` | ECB fix date, >= 2026-11-03 |
| `rates_per_eur` | object | raw ECB rates, one per currency incl. USD, as parsed |
| `usd_per_unit` | object | USD per unit for AUD, CAD, CHF, EUR, GBP, JPY, NOK, NZD, SEK |
| `provenance.code_commit` | str | commit of the code that wrote the row |
| `provenance.data_sha256` | str | SHA-256 of the exact bytes parsed |
| `provenance.source` | str | URL or file the bytes came from |
| `provenance.written_utc` | str | UTC time of the append |
| `provenance.prereg_sha256` | object | registration path -> hash at write time |
| `regime`, `pairs` | object | the signed registrations' daily signals (forward lane) |
| `momentum` | object or null | this lane's momentum signal, below |
| `pairs_v2` | object or null | this lane's pairs v2 signal, below |

`momentum` and `pairs_v2` may be null until their registrations are signed;
that loses nothing, because rule 4 recomputes them from the fixes.

### `momentum` (prereg/momentum-fx-g10.md)

- `status`: `warming_up` until there are 64 logged fixes dated
  2026-11-03 or later at a month end, then `scored`.
- `rebalanced`: bool, true on the last ECB business day of a month once
  scored.
- `scores`: currency -> log return over the previous 63 fixes (on
  rebalance days).
- `weights`: currency -> target weight after the close (+-1/6 or 0).
- `net_return`: the day's net return of the weights held from the previous
  close, cost included.

Reference implementation: `research/studies/forward_studies.py:
momentum_weights`, `momentum_returns`.

### `pairs_v2` (prereg/pairs-fx-g10-v2.md)

Same shape as `pairs` (`status`, `formation`, `selected`, `held` with
`z`, `beta`, `w_y`, `w_x`, `net_return`), computed with the `diff`
observation-variance rule (`forward_studies.VAR_RULES["diff"]`) and the
forward folds of the registration (2026-11-03 to 2027-11-02, then
2027-11-03 to 2028-11-02).

Its formation windows need fixes from 2023-2026, which overlap the signed
registrations' single-use hold-out. So `pairs_v2.status` stays
`waiting_for_v1_holdout` until both `research/results/pairs_fx_g10_holdout.json`
and `regime_fx_basket_holdout.json` exist on main. The formation fixes are
then fetched once, hashed and stored beside the log as a separate,
dated file; they are never scored.

## Evaluation

At the evaluation dates in the registrations (pairs v2: on or after
2028-11-13; momentum: on or after 2029-03-12) a script reads the log at a
stated `forward-log` commit, checks every `data_sha256` it can re-derive,
recomputes the daily returns over the registered window, and writes one
result JSON with `prereg`, `prereg_sha256`, `period`, `mode: "forward"` and
the log commit. That script is not written yet; it is the next piece of
work after signing.
