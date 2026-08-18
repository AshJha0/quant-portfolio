# Validation — FX Regime Switching

All numbers below are produced by `examples/run_pipeline.py` (seeded,
offline, ~50s) on a 2,000-business-day, 12-currency, 3-state synthetic
RORO panel with a planted 2008-style flip, unless a test file is cited.
The suite (`pytest -q`, 125 tests, ~35s) reproduces every claim.

## 1. Cross-checks against reference implementations

The PCA / GMM / HMM engines are written from scratch; scikit-learn and
hmmlearn are used **only** as cross-checks:

| Check | Tolerance | Test |
|---|---|---|
| PCA components / explained variance vs `sklearn.decomposition.PCA` | 1e-8 (up to sign) | `test_pca.py::test_sklearn_cross_check` |
| PCA identities: orthonormality, reconstruction, variance partition, score covariance | 1e-8 – 1e-10 | `test_pca.py` |
| GMM log-density vs `scipy.stats.multivariate_normal` | 1e-10 | `test_gmm.py::test_gaussian_logpdf_matches_scipy` |
| GMM `score_samples` vs sklearn with identical parameters | 1e-8 | `test_gmm.py::test_sklearn_cross_check_fixed_params` |
| GMM fitted optimum vs sklearn `GaussianMixture` | mean loglik within 0.01 | `test_gmm.py::test_sklearn_cross_check_fitted_loglik` |
| HMM likelihood / posteriors / Viterbi vs `hmmlearn` with identical parameters | 1e-6 / 1e-8 / exact | `test_hmm.py::test_hmmlearn_cross_check_fixed_model` |
| HMM fitted transmat & means vs fitted `hmmlearn` | 0.02 / 0.05 after permutation matching | `test_hmm.py::test_hmmlearn_cross_check_fitted` |
| Forward vs backward likelihood identity | 1e-8 | `test_hmm.py` |
| EM monotonicity (GMM and Baum-Welch) | non-decreasing to 1e-6 | both test files |
| Stationary distribution πP = π, Viterbi vs brute-force enumeration, durations 1/(1−Aᵢᵢ) | exact / 1e-10 | `test_hmm.py` |

## 2. Parameter recovery on known truth

**2-state RORO panel (returns-space fit, permutation-matched on
covariances):**

```
est transmat [[0.993 0.007]     true [[0.99 0.01]
              [0.077 0.923]]          [0.05 0.95]]
max abs transition error : 0.027
durations est vs true (d): [137.7  13.0] vs [100.  20.]
Viterbi state accuracy   : 98.4%
```

Well-separated synthetic HMMs recover transitions to 0.03 and means to
0.1 (`test_hmm.py::test_transition_and_state_recovery_permutation_matched`);
Viterbi accuracy > 90% is asserted in the separated case (98%+ typical).

**3-state honesty check.**  With only 82 true squeeze days (4.1% of the
sample) the two crash states largely **merge** in returns space:
risk_on is recovered nearly perfectly (1471/1480 days) but risk_off and
usd_squeeze cannot be reliably told apart at this sample size.  This is
reported, not hidden — and it is why both crash books are defensive
(carry is cut under either label), so label confusion between the two
crash states is economically mild.

**Detection accuracy (expanding refit, filtered, hysteresis):** 79–90%
of days correctly labeled vs truth across seeds on 2-state panels
(≥ 75% asserted in `test_detection.py::test_detection_recovers_regimes`).

## 3. The oracle ▸ filtered ▸ static table (net of costs and carry)

10% vol target, pip costs, carry accrual, one-day execution delay for
**all** strategies (the oracle knows the true state but pays the same
delay and costs):

| strategy | ann. return | ann. vol | Sharpe | max DD | hit rate |
|---|---|---|---|---|---|
| **oracle** | 11.6% | 11.8% | **0.98** | −32.3% | 53.3% |
| **filtered** | 7.8% | 11.3% | **0.69** | −30.9% | 51.7% |
| **static carry** | 5.6% | 11.4% | **0.49** | −30.6% | 51.2% |

* value of the regime filter (filtered − static): **+2.2% p.a.**
* cost of imperfect detection (oracle − filtered): **+3.8% p.a.**

The ordering oracle ≥ filtered ≥ static is additionally asserted
*statistically across seeds* (mean Sharpes over 4 seeds, loose
tolerances) in `test_risk.py::test_oracle_beats_filtered_beats_static_statistically`.

**Per-regime partition (the carry-crash punchline).**  Static carry by
*true* regime: +11.2% p.a. (Sharpe 1.37) in risk-on, **−12.8% p.a.
(Sharpe −0.69) in risk-off**.  The filtered strategy flips the bad
cell: +20.4% p.a. (Sharpe 1.16) in risk-off, +26.2% in squeezes, paying
for it with a lower risk-on return (3.2% vs 11.2%) — the false-alarm
cost quantified below.

**Front-loading.**  93% of static carry's total risk-state losses occur
in the first 10 days of risk-state spells (share_of_risk_pnl at K=10 =
0.926) — the crash happens at the flip, which is exactly why detection
lag is the metric that matters.

## 4. Detection lag vs the oracle

Fresh flips into a risk state (17 in the evaluation window), detection
defined defensively (flagging either crash label counts — both books cut
carry):

| hysteresis | det. rate | mean lag | est. lag cost/flip | false-alarm cost | Sharpe |
|---|---|---|---|---|---|
| fast (0.70/0.30, 2d) | 94% | **1.1 days** | **~10 bp** | 3,905 bp | 0.69 |
| conservative (0.90/0.10, 5d) | 82% | **3.6 days** | **~34 bp** | 4,460 bp | 0.60 |

* The **estimated lag cost per flip** = mean lag × the drift gap
  between the defensive and carry books measured over all true risk
  days (~10 bp/day).  The *realized* per-flip window cost is also
  reported but is noise-dominated: the two books are near-opposite and
  levered, so per-flip daily noise (±2–3%) swamps a 10 bp/day drift at
  17 flips.  Reporting the noisy number next to the stable estimator is
  deliberate.
* **Finding:** with fast hysteresis the filter is ~1 day late and lag
  is *cheap*; the oracle's edge comes almost entirely from **false
  alarms** — days the filter sits defensively through risk-on carry
  (gap decomposition: +3,905 bp on true risk-on days vs −1,395 bp on
  risk days).  Tightening confirmation slows *exits* as much as
  entries, so both costs grow — fast hysteresis dominates on Sharpe.
  The binding constraint on FX regime filters is false alarms, not
  detection speed.

**Planted-flip timeline** (2008-style risk-off planted 2020-10-01):
the filtered probability of a risk state goes 0.08 → 0.37 → 0.77 across
the three days into the flip; the committed book is defensive from the
flip date (via the squeeze label) and carries the correct `risk_off`
label from day +6.  The full probability table prints in section 5 of
the pipeline.

## 5. Null results (no-regime guards)

* **Null GBM panel** (single-state, zero drift): the filtered strategy
  shows **no spurious outperformance** over static carry
  (`test_risk.py::test_null_gbm_no_spurious_outperformance`).
* **BIC on iid returns** prefers k=1 on the null panel and recovers the
  true k=2 on separated mixtures (`test_gmm.py`).
* **All-risk-on sample** (one state never visited): detection still
  runs with valid probabilities; the k=2 model is unidentified and
  will split noise — documented, with the economic guard being the
  null-panel strategy test above (`test_detection.py::test_all_risk_on_sample_handled`).

## 6. Failure modes found during development (kept honest)

1. **Near-unit-root features hijack the HMM.**  With `fwd_ts`
   (expanding z-score of a slowly drifting rate differential) in the
   input, the HMM split the sample into early/late epochs with
   *identical vols* — a regime story with no risk content.  Fix:
   `fwd_ts` excluded from detection inputs by default; kept as a desk
   diagnostic.  This generalises: never feed an HMM a wandering level.
2. **Window length creates phantom states.**  With 10–21-day feature
   windows, transitions smear and a third "transitional" state absorbs
   ~25% of risk-on days (committed as defensive → −2% p.a. of false-alarm
   drag).  Shortening to 8/5/12-day windows raised 2-state detection
   accuracy from ~0.79–0.85 to ~0.85–0.90 and restored the
   oracle ▸ filtered ▸ static ordering on 3-state panels.
3. **HMM-BIC over-selects k on rolling-window features** (picks 4 when
   truth is 3) because the features are autocorrelated and fat-tailed.
   Regime count must be an economic choice; BIC is evidence only.
4. **The oracle is only as good as its playbook.**  With a risk-off
   book that shorted only AUD/NZD, the "oracle" *underperformed* the
   filter's mislabeled squeeze book in risk-off — EM legs crash hardest
   and must be in the defensive short basket.  The state oracle does
   not oracle the book.

## 7. Known failure modes in the real world (documented, some untestable offline)

* **Pegged / managed currencies** distort vol and correlation features:
  a peg contributes zero vol (deflating `avg_vol`) and undefined
  correlations.  Handled numerically (zero-vol legs are skipped in the
  correlation mean; PCA and vol targeting are guarded — tested with a
  pegged column), but a *managed float* silently biases features until
  the band breaks.
* **SNB January 2015**: a peg break is **instantaneous, not
  regime-like**.  A filtered HMM cannot flag a one-tick 20% move in
  advance and the Gaussian emission assigns it likelihood ~0 — the
  posterior jumps *after* the fact.  Regime filters are the wrong tool
  for discontinuous policy risk; that is tail-hedge territory
  (DESK_GUIDE.md).
* **Idiosyncratic EM crises** (Turkey 2018, Argentina): one currency
  crashes while the risk block is calm.  A *systemic* regime model
  correctly does not flip — but a carry basket holding that currency
  still bleeds.  The regime filter is not a substitute for per-country
  risk limits.
* **Slow-bleed carry drawdowns without a vol spike** (e.g. rate-
  differential compression): the features barely move, the filter stays
  risk-on, and carry loses money slowly.  The model detects *crashes*,
  not *mediocrity*.
* **Cross-currency basis / funding breaks in squeezes**: CIP-based
  carry accrual overstates crisis carry (assumption 6).
* **Cost blow-ups at the flip**: constant pip spreads understate
  crisis switching costs (assumption 7).

## 8. Test inventory

125 tests, offline, seeded, ~35s (`pytest -q` from the project root):
synthetic generator recovery (13), features + PIT mutation (14), PCA
identities + sklearn + RORO axis (12), GMM EM/BIC/sklearn (13), HMM
identities/recovery/hmmlearn (17), detection causality/labeling/
hysteresis (15), strategy hand-checks (14), backtest ledger/no-lookahead
(11), risk partitions/oracle ordering/null guards (16), plus edge cases
throughout (k=1, pegged currency, one-state samples, threshold
boundaries, short series raising `ValueError`).
