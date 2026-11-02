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
python research/studies/fx_studies.py regime --mode holdout
```

The pairs registration was withdrawn before the hold-out (DR-R4-5 below):
`pairs` is refused, `all` means `regime` alone, and `prereg.py check`
refuses any pairs hold-out result.

Before it reads a byte of data, it refuses (exit code 3, `REFUSED: ...`) if

- the study is withdrawn (pairs, DR-R4-5);
- the registration is not signed, or its hash is not the one in
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
  (reject H0 if Sharpe - 1.645 x Lo SE > 0; see DR-R4-1).
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

### Decision records fixed before the hold-out

Four points where the registration text admits more than one reading were
decided on 2026-11-02 by the coordinator, under the owner's delegation,
before any 2022-2025 data had been fetched or scored. The registrations are
not edited (that would void the signatures); these records say how the code
reads them. The hold-out code that implements them is commit 149d3d5; the
records were added in the commit that follows it.

- **DR-R4-1, pairs lower bound.** Prereg: "H0 is rejected if the hold-out
  net Sharpe has a one-sided 95 % lower bound above zero", while the primary
  metric lists both a Lo (2002) Newey-West standard error and a circular
  block-bootstrap 95 % interval. Reading: the decision is
  Sharpe - 1.645 x Lo SE > 0. The lower end of the two-sided bootstrap
  interval is reported beside it under `decision.secondary_bootstrap`,
  labelled, and does not decide. Why: the Lo SE is the only quantity that
  gives a one-sided 95 % bound directly; the two-sided interval's lower end
  is a 97.5 % bound.
- **DR-R4-2, deflated Sharpe variance.** Prereg: "Deflated Sharpe reported
  with that count" (3), with no variance of the trial Sharpes. Reading:
  V = 1/(T - 1), the sampling variance of a zero-Sharpe estimate. Why: the
  three training variants' Sharpes were not logged, and inventing them
  after the fact would be worse than the null default. Recorded in the JSON.
- **DR-R4-3, flat outside the test year.** Prereg: "flat outside the test
  year. Decided at the close of day t, earns the return of t+1", and the
  Kalman filter "run from the start of the formation window". Reading, as
  the training code already did: the filter runs through the 5-day embargo
  without trading; a position held at the close of the last test day earns
  nothing, so the return from the last day of a year to the first day of
  the next is never scored. Why: anything else trades outside the test year.
- **DR-R4-4, regime years.** Prereg: "First test year in training: 2003.
  Hold-out years are scored with yearly refits on all data before each
  year." Reading: the hold-out scores 2022-2025 only; each refit uses every
  day from 1999-01-04 to 31 December of the previous year. Why: that is the
  registered expanding window; the 2003 start applies to the training run.

### DR-R4-5: pairs registration withdrawn (2026-11-02)

Decided by the owner on 2026-11-02, recorded here before any 2022-2025 data
had been fetched or scored. `prereg/pairs-fx-g10.md` (signed, hash
dfca5c7a...) is withdrawn; its file is left untouched so the hash and
signature still verify. Its hold-out will never be run: the harness refuses
`pairs`, and `prereg.py check` refuses a pairs hold-out result.

Why: the registered test cannot detect the effect it is meant to test. The
Kalman hedge's state noise (delta = 1e-6) lets alpha absorb the
mean-reverting spread, so z measures a one-day surprise, not the spread
level. On 200 simulated, truly cointegrated pairs per row (OU spread, sd 2 %,
half-life 20 days, the registered parameters), |z| > 2 on 0.54 % of days
with the second leg near log level 0 and 0.00 % for a JPY-like leg
(log level -4.7); a static OLS residual z crosses 2 on 22 % and 6 % of days
(evidence class: simulation, `review/2026-11-02/area-quant.md` C1). The
training run agrees: max |z| stayed below 2 in 16 of 19 folds, 79 of 4,867
days active. The training Sharpe of 0.24 is about 75 % one fold (CHF/NZD
2015, the SNB de-peg), with skew 44.5 and raw kurtosis 2745, so the Lo
standard error and the deflated Sharpe built on it are not valid. A
hold-out result from this design would be uninformative whichever way it
fell. Its replacement, `pairs-fx-g10-v2` (level-based z, simulated power
check before signing), is scored on forward data only.

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
| `prereg/pairs-fx-g10.md` | signed 2026-11-02; **withdrawn** 2026-11-02 before the hold-out (DR-R4-5); hold-out never run | `results/pairs_fx_g10_train.json` |
| `prereg/regime-fx-basket.md` | signed 2026-11-02; hold-out run once 2026-11-02: H0 not rejected | `results/regime_fx_basket_train.json`, `results/regime_fx_basket_holdout.json` |
| `prereg/execution-lobster.md` | draft, unsigned, blocked on data access | none |

The two FX registrations were signed by the maintainer under owner decision
9, which delegated pre-registration signing. The signature records the hash
printed by `prereg.py hash`; `prereg.py check` passes on both. The execution
study needs LOBSTER files, which sit behind LOBSTER's terms of use, so it
stays an unsigned draft. Equity pairs on a point-in-time universe need a
Nasdaq Data Link account and are not drafted.

The pairs strategy is almost dormant in training: it trades in 3 of 19
test years (2008, 2015, 2020; 7 entries, 79 active days), and the other
years show a net return of exactly 0.0. That is the registered design, not
a code fault. The registration sets the observation variance to the
variance of the OLS residual in levels over the formation window. The
Kalman filter's alpha follows the spread, so its one-step errors are
smaller (typically 0.6-1.5 % against a forecast SD of 1.5-3 %), and |z|
seldom reaches 2.0. Per pair and year the result file now records entry
signal days, entries, pair-days in the market and the largest |z|.
Changing the variance rule would be a new registration.

### Regime hold-out result (run once, 2026-11-02)

The regime hold-out ran once, on commit 68ae51c with a clean tree, on the
ECB file for 1999-01-04 to 2025-12-31 (sha256 eff207fe...). Test days
2022-01-03 to 2025-12-31: 161 of 1,019 days were classified high-vol. The
annualised mean next-day basket return on those days was +12.5 % (low-vol
days -4.0 %), HAC t = +0.94 with 4 lags. The registered rule rejects H0
only if t < -1.645, so **H0 is not rejected**: the hold-out gives no
support for the hypothesis that the basket earns a negative return in the
predicted high-vol state. The sign is the opposite of the hypothesis, but
t = 0.94 is not evidence of a positive effect either. Training gave the
same answer (t = +0.27, 1999-2021). Evidence class: single pre-registered
out-of-sample test, trial count 1. The hold-out is now spent and will not
be rerun.

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
