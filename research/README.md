# research

The research programme: pre-registered studies on real data.

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

## Forward log

From 2026-11-03 a daily job records the ECB rates and the two registered
rules' state on the branch `forward-log`, never reading anything earlier.
See [forward/README.md](forward/README.md) for why and how to read it.

## Studies

| Registration | Status | Training result |
|---|---|---|
| `prereg/pairs-fx-g10.md` | signed 2026-11-02; **withdrawn** 2026-11-02 before the hold-out (DR-R4-5); hold-out never run | `results/pairs_fx_g10_train.json` |
| `prereg/regime-fx-basket.md` | signed 2026-11-02; hold-out run once 2026-11-02: H0 not rejected | `results/regime_fx_basket_train.json`, `results/regime_fx_basket_holdout.json` |
| `prereg/execution-lobster.md` | draft, unsigned, blocked on data access | none |
| `prereg/pairs-fx-g10-v2.md` | signed 2026-11-02; forward-only (fixes from 2026-11-03), evaluation on or after 2028-11-13 | `results/pairs_fx_g10_v2_train.json` |
| `prereg/momentum-fx-g10.md` | signed 2026-11-02; forward-only (fixes from 2026-11-03), evaluation on or after 2028-11-13 | `results/momentum_fx_g10_train.json` |

The two FX registrations were signed by the maintainer, to whom the
repository owner delegated pre-registration signing. The signature records the hash
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

## Forward hold-out registrations

The 2022-2025 window can be used once. The regime registration used it;
the pairs registration was withdrawn rather than spend it. New questions on the same free ECB data therefore need data nobody has
seen: fixes from the signing date onward. Two drafts do that.

- `prereg/pairs-fx-g10-v2.md` keeps the signed pairs selection and replaces
  the Kalman one-step z with a level z: a static OLS hedge fitted on the
  formation window and the spread divided by its formation SD. Why: the
  Kalman intercept follows the spread and absorbs it. On simulated pairs
  that are cointegrated by construction (`studies/pairs_power.py`,
  `power/pairs_v2_power.json`) the Kalman z passes 2 on 0.3 % of test days
  or fewer and never at a JPY-like price level, and the decision rule
  rejects H0 in 0-47 % of runs; the level z rejects in 100 % (10-day
  half-life) and 54 % (30-day), with size under 1 % when there is no
  cointegration. Training (1999-2021) net Sharpe is -0.107 (Lo SE 0.234),
  trading in 17 of 19 years. Trial count 7: every pairs variant scored on
  this training data, including a Kalman observation-variance menu tried
  first and kept in the result file.
- `prereg/momentum-fx-g10.md`: long the three G10 currencies with the best
  63-day spot return, short the worst three, monthly. One variant, taken
  from Menkhoff et al. (2012) before any data was run. Training net
  Sharpe -0.20 (Lo SE 0.20), so the training period weighs against it; the
  forward data decides.

Formation and lookback history comes from one ECB file of 1999-01-04 to
2026-11-02 fixes, fetched at signing after the 2022-2025 hold-out has been
run; nothing dated before 2026-11-03 is scored. Both are evaluated once, on
2026-11-03 to 2028-11-02, by
Sharpe - 1.645 x Lo SE > 0. With about 500 daily returns that can only
detect a true Sharpe above roughly 1.2, which the registrations state. For momentum the
simulated power check (`studies/momentum_power.py`,
`power/momentum-fx-g10.json`, 200 runs per cell) gives about 13 %
rejection at a true Sharpe of about 0.4 and 50 % only near 1.3, so a fail
is the expected outcome and says little. The forward log that supplies the
data is specified in `forward/SPEC.md`; `forward/evaluate.py` scores it and
gives a verdict only on or after 2028-11-13, once the log is complete and
the registration is signed. Before then every report is marked interim.
Training runs:

```bash
python research/studies/forward_studies.py all
python research/studies/pairs_power.py        # synthetic power check
```

## Trials log

The plain idea: a deflated Sharpe needs the number of things that were
tried, and a number recalled from memory is the one most likely to be
wrong. So every training run now writes down that it happened.

Formally: each call of `_write` in `studies/fx_studies.py` and
`studies/forward_studies.py` (training runs, and the stress and calibration
runs below) appends one JSON line to `results/trials.jsonl`, opened in
append mode only. Fields: `timestamp_utc`, `study`, `kind` (`train`,
`stress` or `calibration`), `prereg`, `prereg_sha256`, `result`,
`git_commit`, `git_dirty` (read before the result file is written, with the
log itself left out), `params`, `metric` and `python`. The hold-out path
does not write here; it has its own provenance block. `trials.count(prereg)`
gives the number of logged training runs against a registration.

Worked example: the log was started by re-running the four training studies
on commit 73908e1 with a clean tree. All four result files came out byte for
byte the same as before, and the log holds one `train` line for each.
Earlier trials (the three v1 drafting variants and the variance menu) were
not logged at the time; their count stays as stated in each registration.

## Robustness checks (training period only)

`studies/robustness.py` answers two questions the training results leave
open. Neither is a new registration or a new decision rule; both are
sensitivity tables. Evidence class for every number in this section:
training data (ECB 1999-01-04 to 2021-12-31), not out-of-sample.

```bash
python research/studies/robustness.py stress
python research/studies/robustness.py calibration --perms 200   # about 50 min
```

### Is the pair selection finding anything?

The plain idea: each fold tests 36 pairs at 5 %, so about two pairs a year
pass even if no pair is cointegrated. The question is whether the
selection finds more than that.

Formally, for each training fold the result gives:

- `selected`: pairs passing Engle-Granger at 5 % AND Johansen trace rank
  >= 1, the registered rule;
- `null_mean`, `null_p95`, `perm_pvalue`: the same rule re-run on 200 null
  data sets. A null data set draws whole days (rows of the formation
  window's daily log-return matrix) with replacement and re-cumulates them.
  That keeps each day's cross-section (correlations, fat tails, the dollar
  factor) and destroys the order in time, so no spread can mean-revert.
  Drawing with replacement matters: a permutation keeps every series' end
  point, so a cointegrated spread is rebuilt as a bridge that still looks
  stationary, and the null becomes too hard to beat (the unit tests show
  this). `perm_pvalue` = (1 + #null counts >= selected) / (1 + 200);
- `bh_selected`: pairs whose MacKinnon (1994, 2010) Engle-Granger p-value
  survives Benjamini and Hochberg (1995) at q = 0.05 across the 36
  candidates, and that also pass Johansen.

Result (`results/pairs_selection_calibration.json`, 19 folds, 2003-2021):
53 pairs selected in total against a null mean of 22.2 (null 95th
percentile 34), permutation p = 0.010. So across all years the selection
finds more than chance. Year by year it mostly does not: the null gives
about 1.1 pairs a fold, and only 2013 (7), 2016 (5), 2017 (6) and 2018 (9)
have p <= 0.05. BH at 5 % keeps 14 pairs in total, 8 of them in 2018, and
none in 15 of the 19 years. Reading it: in most years the selected pairs
are what chance alone would produce; a forward result from pairs v2 should
be read with that in mind. The null mean (about 1.1, not 36 x 5 % = 1.8)
is lower than the nominal rate because a pair must pass both tests.

### Does the result survive the fix not being tradeable?

The plain idea: the studies assume a trade at the ECB 14:15 CET fix for
1 bp. On 15 January 2015 nobody could deal CHF anywhere near the fix. So
re-score the rules with higher costs, and with the days on which a fix was
least tradeable either removed or scored against the strategy.

Formally: an event day is one where any currency's log return exceeds 5
times its standard deviation over the 250 fixes before it (needs 60), or
where more than 4 calendar days separate it from the previous fix (Easter
and Christmas closures). Both use only data up to that day. For costs of
1, 3 and 5 bp per unit traded, three versions of the daily net return are
scored: `base`; `exclude` (event days dropped); `adverse` (on event days
the return is min(r, 0): gains forfeited, losses kept, a crude worst case
for a gap).

Result (69 event days in 1999-2021, 2015-01-15 among them; annualised net
Sharpe, Lo standard error in brackets):

| Rule | Cost | base | exclude | adverse |
|---|---|---|---|---|
| pairs v2 | 1 bp | -0.107 (0.234) | -0.095 (0.212) | -0.367 (0.209) |
| pairs v2 | 3 bp | -0.114 (0.234) | -0.103 (0.212) | -0.375 (0.209) |
| pairs v2 | 5 bp | -0.121 (0.235) | -0.112 (0.212) | -0.383 (0.209) |
| momentum | 1 bp | -0.200 (0.196) | -0.290 (0.195) | -0.475 (0.191) |
| momentum | 3 bp | -0.244 (0.197) | -0.337 (0.195) | -0.520 (0.191) |
| momentum | 5 bp | -0.288 (0.197) | -0.385 (0.195) | -0.566 (0.190) |

Reading it: both rules are negative in training at every cost, so the
stress cannot rescue them; it shows how much worse a forward result could
look. Pairs v2 trades little, so cost barely moves it (about -0.007 Sharpe
per 2 bp); its event days lost money (-2.3 % summed over 50 exposed days).
Momentum turns over monthly, so each 2 bp costs about 0.044 Sharpe, and its
event days made money (+6.5 % over 68 days): dropping them makes it worse.
Files: `results/pairs_fx_g10_v2_stress.json`,
`results/momentum_fx_g10_stress.json` (event dates listed in each).

## Models added here

- `studies/models.py: johansen_trace`, the Johansen (1991) trace test with
  an unrestricted constant. Oracle: statsmodels `coint_johansen`, matched to
  1e-9 relative.
- `studies/models.py: kalman_hedge`, a random-walk hedge ratio (Harvey
  1989, section 3.2). Oracle: with delta = 0 it equals recursive least
  squares.

Tests: `python -m pytest research/tests` (needs qp-core, fx-pairs,
fx-regime; statsmodels for the Johansen oracle).
