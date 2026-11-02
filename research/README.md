# research

The Phase 4 research programme: pre-registered studies on real data.

## How a study runs

1. A pre-registration in `prereg/` fixes the hypothesis, data, universe,
   parameters, costs, primary metric and decision rule. Its registration
   hash is the SHA-256 of everything above its `## Signature` heading (line
   endings normalised to LF).
2. The maintainer signs it by filling `signed_by`, `signed_on` and
   `sha256` under that heading. Signing does not change the hash.
3. The study writes a JSON result to `results/` carrying `prereg`,
   `prereg_sha256` and `period`.
4. `python research/tools/prereg.py check` refuses a result when its hash
   does not match the registration, or when its period reaches the
   2022-2025 hold-out and the registration is not signed with that hash.
   CI runs it on every push.

Every training run calls `models.training_only` on its input, which raises
if the data reaches 2022-01-01. The hold-out run is the only way past it,
and it has its own guards (next section).

## The hold-out run

The 2022-2025 window can be scored once. A second look would turn it into
more training data, so the code makes the first run auditable and a second
run impossible.

```bash
python research/studies/fx_studies.py all --mode holdout
```

Before it reads a byte of data, it refuses (exit code 3, `REFUSED: ...`) if

- either registration is not signed, or its hash is not the one in
  `fx_studies.SIGNED_SHA256` (the hashes signed in commit 35d351a);
- `git status --porcelain` shows anything, untracked files included;
- `research/results/pairs_fx_g10_holdout.json` or
  `regime_fx_basket_holdout.json` already exists. The file is then opened in
  exclusive-create mode, so a race cannot overwrite it either.

It then downloads the ECB file for 1999-01-01 to 2025-12-31 into
`data/raw/` (if not there), hashes the bytes it parses, and scores the
test years 2022, 2023, 2024 and 2025 exactly as each registration says:

- pairs: each year's pairs are chosen on the 3 calendar years before it,
  ending 5 business days before its first day; the Kalman filter runs from
  the start of that window and only sees data up to the day it trades.
  Reported: net Sharpe, Lo (2002) standard error, block-bootstrap 95 %
  interval, the deflated Sharpe with trial count 3, and the decision
  (reject H0 if Sharpe - 1.645 x Lo SE > 0).
- regime: the HMM for year Y is fitted on every day before 1 January of Y;
  the filtered probability at the close of day t classifies the return of
  day t+1. Reported: the annualised high-vol mean, its Newey-West t and the
  decision (reject H0 if t < -1.645).

Every hold-out result carries a `provenance` block: git commit, dirty flag,
SHA-256 of each input file, the registration hashes, a UTC timestamp and the
versions of Python, numpy, pandas, scipy, statsmodels, qp-core, fx-pairs and
fx-regime. `prereg.py check` refuses a `*_holdout*.json` whose provenance is
missing, dirty, a dry run, or disagrees with its `prereg_sha256`.

Tests show that a fold's result does not change when every price after the
fold is altered, which is the plain test of no look-ahead.

### Dry run

```bash
python research/studies/fx_studies.py all --dry-run [--out DIR]
```

runs the same code path, guards included except the clean-tree check, on a
seeded synthetic ECB-format file dated 2012-2025. The dates are fake and
the numbers mean nothing; the point is that the hold-out path has run end
to end before the real one. It writes `*_holdout_dryrun.json` to a temporary
directory, never to `research/results`, and touches no network.

### Deflated Sharpe

The plain idea: with three variants tried, the best of three lucky
strategies would show a positive Sharpe by chance, so the bar is raised to
the Sharpe you would expect from the best of three skill-less tries.

Formally (Bailey and Lopez de Prado 2014, eqs. 2-3), with per-day Sharpe
SR over T returns, skewness g3 and raw kurtosis g4,
SR0 = sqrt(V) ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e))) and
DSR = Phi((SR - SR0) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) SR^2 / 4)).
V, the variance of the trial Sharpes, is taken as 1/(T - 1) because the
three training variants' Sharpes were not logged.

Worked example (the paper's section 4, reproduced in
`tests/test_holdout.py`): annualised SR 2.5 over 1,250 days, g3 = -3,
g4 = 10, N = 100, annualised V = 1/2 gives SR0 = 0.1132 per day and
DSR = 0.9004. Evidence class: published formula, unit-tested against the
published example and against `qp_core.stats.deflated_sharpe_ratio`.

To get a hash for signing:

```bash
python research/tools/prereg.py hash research/prereg/pairs-fx-g10.md
```

## Studies

| Registration | Status | Training result |
|---|---|---|
| `prereg/pairs-fx-g10.md` | signed 2026-11-02; hold-out code ready, not run | `results/pairs_fx_g10_train.json` |
| `prereg/regime-fx-basket.md` | signed 2026-11-02; hold-out code ready, not run | `results/regime_fx_basket_train.json` |
| `prereg/execution-lobster.md` | draft, unsigned, blocked on data access | none |

The two FX registrations were signed by the maintainer under owner decision
9, which delegated pre-registration signing. The signature records the hash
printed by `prereg.py hash`; `prereg.py check` passes on both. The execution
study needs LOBSTER files, which sit behind LOBSTER's terms of use, so it
stays an unsigned draft. Equity pairs on a point-in-time universe need a
Nasdaq Data Link account and are not drafted.

Training results (ECB data, 1999-2021) are numbers from the training
period only. They are not out-of-sample evidence; the hold-out decides.

## Models added here

- `studies/models.py: johansen_trace`, the Johansen (1991) trace test with
  an unrestricted constant. Oracle: statsmodels `coint_johansen`, matched to
  1e-9 relative.
- `studies/models.py: kalman_hedge`, a random-walk hedge ratio (Harvey
  1989, section 3.2). Oracle: with delta = 0 it equals recursive least
  squares.

Tests: `python -m pytest research/tests` (needs qp-core, fx-pairs,
fx-regime; statsmodels for the Johansen oracle).
