# Methodology — FX VaR & Expected Shortfall Engine

This document answers contract items **1** (why these models, vs alternatives)
and **2** (assumptions register) of `CONVENTIONS.md`, with the FX-specific
machinery — USD triangulation, CIP-consistent forwards, correlation regimes,
pegged currencies — treated as first-class citizens.

---

## 1. The risk-factor representation (FX-specific by construction)

### 1.1 USD triangulation

Every currency `CCY` is mapped to **one** FX factor:

```
FX:CCY = daily log return of the USD price of 1 unit of CCY  (log CCYUSD)
```

USD is the pivot and has no factor (its USD price is identically 1). A
position in a *cross* pair decomposes into its two USD legs:

```
long N EURJPY  =  (+N EUR leg)  +  (−N·X₀ JPY leg),    X₀ = S_EUR/S_JPY
```

**Why this and not one factor per traded pair?** With a factor per pair
(EURUSD, USDJPY, EURJPY, …) the covariance matrix must satisfy the exact
non-linear constraint `log EURJPY = log EURUSD + log USDJPY` or scenario sets
become arbitrage-inconsistent — a historical scenario could move EURJPY
without moving either leg. With USD triangulation the identity holds *by
construction* in every scenario, historical, parametric or Monte Carlo. The
engine proves the identity exactly (`tests/test_book.py::
test_triangulation_identity_eurjpy`): a EURJPY position's P&L equals the
EURUSD + USDJPY leg decomposition to machine precision, for arbitrary joint
shocks.

Base-currency P&L is `V₁_usd / S₁_base − V₀_usd / S₀_base`, with the base
currency's own USD price shocked consistently — so a base-currency cash
balance carries exactly zero risk (tested), and translation risk of non-base
holdings is captured automatically.

### 1.2 Forwards = spot + two deposit legs (CIP)

An outright forward (long `N` BASE at strike `K`, expiry `T`) is represented
as its covered-interest-parity replication:

```
V_usd = N·e^{−r_f T}·S_base_usd  −  N·K·e^{−r_d T}·S_quote_usd
```

so forward-point risk enters VaR through the **interest-rate factors**
`IR:CCY` (absolute shocks to flat cc zero curves, ACT/365). This is exactly
equivalent to discounting `(F − K)` at the quote rate with
`F = X·e^{(r_d−r_f)T}` — both identities are tested to 1e-10
(`tests/test_forwards.py`). For small FX shocks a forward's P&L equals the
spot P&L up to `O(rT)` carry terms (tested); the rate legs are small next to
the FX delta at daily horizons but they are *present*, which matters for
books of long-dated forwards and for the treasury hedging use case in the
desk guide.

### 1.3 Options: Garman–Kohlhagen, full reval or delta–vega(–gamma)

FX options are revalued with an internal, minimal Garman–Kohlhagen pricer
(GK = Black–Scholes with the foreign rate as the dividend yield; pairs quoted
BASE/QUOTE = QUOTE per 1 BASE; `r_d` = quote rate, `r_f` = base rate; put–call
parity and Greeks-vs-finite-differences tested). Two revaluation modes:

* **full** — reprice GK under shocked spot / rates / vol per scenario
  (default; vectorised, so 100k scenarios cost milliseconds);
* **delta_vega / delta_vega_gamma** — Greek mapping around the reference
  market, the classic RiskMetrics-style approximation.

The mapping's error is quadratic (cubic with gamma) in the shock and is
characterised in tests: for a long option, delta-only P&L understates gains
and overstates losses in *both* directions (positive gamma), so a
mapping-based VaR is conservative for long-gamma books and dangerous for
short-gamma books. Full revaluation is the default for exactly that reason.

### 1.4 Factor summary

| family  | meaning                              | shock units      |
|---------|--------------------------------------|------------------|
| `FX:CCY`  | log return of CCYUSD               | log return       |
| `IR:CCY`  | flat cc zero rate (ACT/365)        | absolute, p.a.   |
| `VOL:PAIR`| annualised ATM implied vol         | absolute         |

---

## 2. Why each VaR method (and against what alternatives)

The engine deliberately implements **three families** and backtests them
against each other — on an FX desk no single method survives all regimes, and
the *disagreement between methods is itself information* (§3 of
VALIDATION.md).

### 2.1 Historical simulation (plain, BRW age-weighted, filtered)

* **Chosen because**: non-parametric — captures FX fat tails, skew and
  cross-pair tail dependence without a distributional assumption; the
  regulator's default; trivially explainable ("that loss is 16 Oct last
  year").
* **Alternatives considered**: parametric var-covar (rejected as headline:
  normal tails demonstrably fail FX 99% backtests — see VALIDATION §2);
  bootstrap HS (adds noise without fixing the real defect, window blindness).
* **Trade-offs**: plain HS is unconditional — it reacts to a vol regime
  change only as fast as the window rolls. The engine therefore also ships:
  * **BRW age weighting** (`w_t ∝ λ^age`): recent scenarios dominate; VaR
    reacts in days, not months;
  * **FHS** (filtered historical simulation): devolatilise each factor by
    its EWMA σ, rescale to today's forecast — keeps the empirical copula,
    makes the tail conditional. FHS is the variant that passes the
    Christoffersen conditional-coverage backtest on vol-clustered data
    (pipeline section 4: parametric-normal 14 exceptions / CC p < 1e-4;
    FHS 7 exceptions / CC p = 0.15).

### 2.2 Parametric (variance–covariance)

* **Chosen because**: instant, differentiable, decomposable — the desk's
  intraday what-if and marginal-VaR tool; exposures × covariance is also the
  natural home of the reverse-stress closed form (§5).
* **Alternatives**: full-reval HS intraday (too slow to iterate on quotes);
  Taylor-only "delta-normal" everywhere (kept, but only as one overlay).
* **Overlays on the same σ**: normal (RiskMetrics classic), **Student-t**
  (variance-matched, df≈4–6 for EM pairs), **Cornish–Fisher** with an
  explicit **monotonicity domain check** (Maillard 2012) — outside the
  domain the CF "quantile" is not a quantile and the engine refuses rather
  than silently reporting 99% VaR below 95% VaR.
* **Trade-offs**: linearises options (documented above); blind to anything
  not in the covariance (pegs — §4).

### 2.3 Monte Carlo (normal / Student-t / jump-mixture)

* **Chosen because**: full revaluation under a *controlled* distribution —
  the only way to (a) stress the distributional assumption itself at fixed
  covariance, and (b) price the peg/devaluation overlay into a quantile.
* **Alternatives**: historical bootstrap (cannot extrapolate beyond the
  window); copula-based MC (deferred — the t-mixture already delivers joint
  tail dependence with two parameters).
* **Design points**: Cholesky with escalating jitter (singular matrices are
  *routine* in FX — two currencies pegged to the same anchor are perfectly
  correlated; tested); multivariate t scaled by `√((ν−2)/ν)` so the
  covariance **matches the normal case exactly** — any 99% VaR difference is
  pure tail shape (+12% for the demo EM book at df=5); jump-mixture adds a
  Bernoulli common devaluation event on selected factors (+140% at 99% for
  the same book); VaR standard error via the order-statistic asymptotic
  `√(α(1−α)/n)/f̂(q)` with KDE density, and MC-vs-closed-form agreement is
  accepted only within 3 SE.

### 2.4 Expected Shortfall

ES is computed alongside VaR everywhere, with the **Acerbi–Tasche
tail-splitting estimator** (fractional weight on the VaR atom). This detail
is not pedantry: with the naive "mean of exceedances" estimator, ES on
discrete peg-jump distributions loses subadditivity — the engine's estimator
provably keeps it, and the test suite contains the classic counterexample
(two independent 0.9%-probability peg breaks: 99% **VaR** of each is zero,
VaR of the pair is the full jump — non-subadditive; **ES** sees both and adds
correctly). Closed forms for normal and variance-matched t ES are tested to
1e-10 / against numerical integration.

### 2.5 Backtesting and Basel traffic light

Kupiec POF (unconditional coverage), Christoffersen independence and
conditional coverage (the FX-relevant one: vol clustering produces *bunched*
exceptions long before the count is wrong), Basel traffic light with the
exact regulatory zones — computed from the cumulative Binomial(250, 1%)
probability with the 95% / 99.99% cuts, reproducing green 0–4 / yellow 5–9
(add-ons 0.40–0.85) / red ≥ 10 — and the Acerbi–Szekely unconditional ES
backtest with a seeded parametric null.

### 2.6 Stress testing and reverse stress

Historical replays (Brexit 2016, CHF depeg 2015, JPY Oct-1998, EM crisis
composite), hypothetical broad-USD moves, a **peg-break scenario generator**
(the mandated companion to the engine's `PegBlindnessWarning`), sensitivity
ladders, and reverse stress: for a linearised book the worst shock at
Mahalanobis radius `k` is `dx* = −kΣw/√(w'Σw)` with loss `k√(w'Σw)` —
closed form, confirmed numerically by constrained optimisation in tests.

---

## 3. Correlation structure: G10 blocks, EM blocks, regimes

The synthetic market (and the documented mental model for the real one) is a
block structure: G10 intra-correlation ≈ 0.55, EM intra ≈ 0.45, cross-block
≈ 0.25 in the calm regime. In the **stress regime** correlations rise
(0.75/0.75/0.60) and **JPY flips sign** against carry currencies (safe-haven
flight) — the empirical "correlations go to one, except the ones you were
hedged with, which go to minus one". The regime-switching simulator (2-state
Markov, stress vol ×2) generates exactly the data that breaks unconditional
VaR in the backtest section, and `default_correlation(regime="stress")` is
available for stressed-VaR-style parametric runs. Tweaked matrices are
eigenvalue-clipped back to PSD.

---

## 4. Pegged and managed currencies

A pegged currency (HKD band, SAR hard peg, CHF floor 2011–15) realises
near-zero daily vol until the regime breaks, then gaps 10–30% with no
intermediate prints. **HS and parametric VaR are structurally blind to
this**: the loss distribution in any window that does not contain the break
is a spike at zero. The engine's policy:

1. any FX factor with daily σ < 0.05% (≈0.8% annualised) triggers a
   `PegBlindnessWarning` naming the factor (tested);
2. the flagged book **must** carry the `peg_break_scenario` stress add-on —
   a configurable revaluation jump (± direction: EM devaluation or CHF-2015
   appreciation), vol spike and contagion co-moves;
3. for a *quantile* including the break, the jump-mixture MC prices the same
   event with a probability attached.

Demo-book numbers: HS 99% VaR $0.69m vs HKD −30% peg-break loss $15.0m —
21.7× the VaR (pipeline section 5).

---

## 5. Assumptions register

Each assumption states *what breaks if violated*.

| # | Assumption | What breaks if violated |
|---|------------|--------------------------|
| A1 | Log FX returns are the right shock space; scenarios apply multiplicatively (`S·e^r`). | Very large shocks in simple-return space could produce negative rates; log space cannot. Violated only if quoted market convention (simple % gaps) is fed in unconverted — scenario calibrations here convert via `log1p`. |
| A2 | USD is a valid pivot: every cross is the ratio of its USD legs (no triangular arbitrage). | If crosses trade persistently off triangulated levels (tiny, fleeting in practice), cross P&L has a basis the factor set cannot see. Basis risk would need a per-cross residual factor. |
| A3 | Flat cc zero curve per currency (one `IR:CCY` factor), ACT/365. | Curve-shape risk (steepeners on the forward book) is invisible. Adequate for FX VaR granularity where FX delta dominates; a desk running large calendar-spread forward books needs tenor buckets. |
| A4 | CIP holds for forward revaluation. | Post-2008 cross-currency basis (10–50bp) makes CIP forwards slightly off market forwards. Error is second-order for P&L *changes*; a basis factor per ccy would fix levels. |
| A5 | One ATM vol per pair (`VOL:PAIR`); no smile risk in VaR. | Risk-reversal/butterfly moves (smile steepening in crashes) are unpriced. Material for large RR positions; the stress library's vol add-ons partially compensate. |
| A6 | VaR horizon P&L ignores theta and carry accrual (revaluation-only). | Overnight carry on huge EM books (TRY at 45%) is ~18bp/day — comparable to a vol move. Documented; add expected carry to the P&L mean if the desk wants it. |
| A7 | IR and VOL factor histories are independent normals in the synthetic generator (FX dominates). | Spot-vol correlation (crash = vol spike) is understated in *synthetic* histories; real histories carry it automatically. Stress scenarios include explicit joint spot+vol shocks. |
| A8 | Multi-day VaR scales as √h from 1-day. | Breaks under autocorrelation/vol clustering (GARCH: 10-day VaR underestimated) and for negatively-skewed carry books. Quantified in VALIDATION §F4. |
| A9 | Exception clustering is first-order Markov (Christoffersen). | Longer-memory clustering partially escapes the LR test; the Basel count still catches level errors. |
| A10 | Sample/EWMA covariance from the chosen window represents tomorrow. | Regime breaks (correlation flips) invalidate it — hence the stress correlation matrix and the FHS/EWMA variants. |
| A11 | Pegs stay pegged within HS/parametric VaR. | The break loss is 20×+ the VaR (demo). Mitigated by warning + mandatory stress add-on + jump-mixture MC (§4). |

---

## 6. Scenario calibration notes (historical replays)

Close-to-close, vs USD, converted to log space:

* **Brexit, 24 Jun 2016** — GBPUSD 1.4877 → 1.3679 (−8.1%); EURUSD −2.4%;
  JPY +3.9%; GBP 1m vols +~12 pts.
* **SNB floor removal, 15 Jan 2015** — CHF +14.9% vs USD close-to-close
  (intraday high near +30%); EURUSD −1.4%; CHF vols +~15 pts.
* **JPY carry unwind, 7–8 Oct 1998** — USDJPY ~131 → ~117 (JPY +11.5% over
  two sessions); AUD −4%.
* **EM composite** — a 1997 THB / 1998 RUB / 2001 ARS-shaped devaluation
  wave: EM −12% to −25%, EM vols +10–20 pts, JPY bid.

These are one-day (or stated two-day) full-revaluation shocks; they are
deliberately *not* scaled to the current vol level — a replay answers "what
if that tape printed again tomorrow".
