# Methodology — Sovereign PD, FX Settlement Risk, Pre-Settlement Exposure

## 1. Why a WOE + logistic scorecard for sovereigns? (model choice, vs alternatives)

**Chosen:** quantile WOE binning (with explicit missing bin and monotone
merge) feeding a from-scratch IRLS logistic regression, scaled to a PDO
scorecard and mapped to AAA…C bands with PD midpoints.

**Alternative 1 — gradient-boosted trees / random forest.** Higher raw AUC on
large corporate/retail books, but for sovereigns: (i) the *entire* modern
default history is a few dozen events — trees overfit catastrophically at
n_events ≈ 60; (ii) country-limit governance requires monotone, explainable
drivers ("less reserve cover ⇒ worse score" must never reverse), which boosted
trees do not guarantee; (iii) regulators and rating committees audit the
binning table, not SHAP values. WOE binning already captures the key
nonlinearity (the reserve-cover threshold) that a linear-in-raw-features
logistic would miss, at zero opacity cost.

**Alternative 2 — structural / market-implied models (CDS-implied PD,
sovereign Merton à la Gray-Merton-Bodie).** Attractive where liquid CDS
exists, but: (i) most EM/frontier sovereigns have no liquid CDS, and quoted
spreads embed large, time-varying risk premia — CDS-implied "PD" can triple
with global risk appetite while fundamentals are unchanged; (ii) sovereign
balance sheets have no observable "asset value" or default barrier, making
the Merton mapping heroic. Market signals belong in an early-warning overlay,
not in the through-the-cycle limit-setting PD.

**Alternative 3 — shadow rating (documented, not chosen as primary).** With
3-6% event rates, an honest low-default alternative is to regress *agency
ratings* (ordered logit on S&P/Moody's sovereign ratings) instead of defaults
— thousands of country-year rating observations instead of ~60 defaults, so
much tighter coefficients. Its limits: it inherits agency behaviour (rating
inertia, pro-cyclical downgrades after the fact, sovereign-ceiling politics)
and can never flag a risk the agencies miss — which is precisely the desk's
job. The right production design is a default-based scorecard (this project)
*benchmarked* against a shadow-rating model; disagreements are the review
committee's agenda.

**The low-default problem, stated honestly.** 63 training events pin down 10
WOE coefficients only loosely (see the standard errors in the pipeline
output: `fiscal_gdp` z = -0.7). We mitigate by (a) WOE compression — each
feature contributes one dimension, monotone by construction; (b) separation
detection with an explicit ridge fallback; (c) wide bootstrap CIs reported,
never hidden (VALIDATION.md). We do *not* pretend a 3-decimal PD for an AAA
sovereign is estimable from data; the AAA/AA band midpoints are anchored to
long-run rating-agency default studies, which is itself Assumption A9.

## 2. The three blocks, mathematically

### 2.1 Sovereign PD

WOE for bin *i*: `WOE_i = ln[(bad_i/Bad) / (good_i/Good)]` (bad-over-good, so
positive = risky); `IV = Σ (bad share − good share)·WOE_i`. IRLS solves the
logistic MLE by Newton steps `β ← β + (XᵀWX)⁻¹ Xᵀ(y − p)` with
`W = diag(p(1−p))`; standard errors from `(XᵀWX)⁻¹`. Scorecard:
`score = 600 + (20/ln2)·ln(odds_good/50)` — 20 points doubles the good odds
(identity unit-tested). Rating bands are fixed PD cut-offs with monotone
midpoints (unit-tested), so band ⇒ PD ⇒ CVA/capital is a single audited map.

`contagion` is deliberately **excluded** from the scorecard: the flag is
contemporaneous with the regional crisis (not observable ex ante), so using
it would be borderline outcome leakage. It is retained in the panel as a
*stress overlay* — which is exactly what makes the planted 2020 global
contagion year a genuine out-of-time calibration stress.

### 2.2 FX settlement (Herstatt) risk

For a trade where we pay currency A and receive currency B, exposure equals
the **full bought principal** from the moment our payment of A becomes
irrevocable (modelled: open of A's RTGS window) until we receive B with
finality (modelled: close of B's window):
`at_risk_hours = max(0, close_B − open_A)`, exposure = USD value of the B
principal, **zero if PvP/CLS-settled** (the two legs settle simultaneously or
not at all). Encoded UTC windows: JPY 00:00-08:00, EUR 06:00-17:00, GBP
08:00-18:00, USD 13:30-23:30. The matrix is asymmetric: paying JPY against
USD carries 23.5 h of principal risk; paying USD against JPY carries none —
the party paying the earlier-time-zone currency always bears Herstatt risk,
exactly the 1974 configuration (banks paid DEM in the European morning;
Herstatt was closed at 15:30 CET before the USD legs paid in New York).

### 2.3 Pre-settlement exposure and CVA

Garman-Kohlhagen GBM under the domestic measure,
`dS = (r_d − r_f)S dt + σS dW` (exact scheme, no discretisation error).
Forward MTM `V_t = ±N(F_t − K)e^{−r_d(T−t)}`, `F_t = S_t e^{(r_d−r_f)(T−t)}`.
Current exposure `max(V_t,0)`; EE/PFE from 100k seeded paths.

**Exposure shape — the correct statement.** An outright forward has *no
interim cashflows*: uncertainty accumulates to maturity and PFE grows
monotonically like `√t` (concave — unit-tested via decreasing quarterly
increments). The famous "peaks mid-life" hump belongs to amortising products
(IR swaps), where remaining-cashflow rundown eventually beats diffusion; it
is **not** the profile of a single FX forward, and tests enforce the correct
shape.

Netting: with an agreement, exposure is `max(ΣV_i, 0)`; without,
`Σ max(V_i,0)`. Pathwise `max(Σv,0) ≤ Σmax(v,0)` gives netted ≤ gross always,
with equality iff MTMs never have mixed signs (both bounds unit-tested).

CVA: `CVA = LGD·Σ EE(t_i)·[PD(t_i)−PD(t_{i−1})]·e^{−r_d t_i}` with
`PD(t) = 1 − e^{−ht}`, `h = −ln(1 − PD_1y)` from the block-1 rating midpoint.

### 2.4 Capital

`EL = PD·LGD·EAD`. Standardized sovereign RW: 0% (AAA/AA), 20% (A), 50%
(BBB), 100% (BB/B), 150% (CCC/C) — with the crucial regulatory footnote that
**0% applies to AAA/AA sovereigns and, under national discretion, to
domestic-currency sovereign debt regardless of rating**. Internal models
differ because Russia 1998 and Greece 2012 defaulted at exactly such
"risk-free" weights; economic capital uses the scorecard PD and Vasicek/ASRF:
`K = LGD·[Φ((Φ⁻¹(PD)+√ρ·Φ⁻¹(0.999))/√(1−ρ)) − PD]` with **ρ_sov = 0.30**,
above the Basel corporate 0.12-0.24, because sovereign defaults cluster on
global factors (USD funding cycle, commodity busts, contagion). The ordering
K(ρ_sov) > K(ρ_corp) is unit-tested across the PD range.

## 3. Assumptions register

| # | Assumption | What breaks if violated |
|---|---|---|
| A1 | Panel rows are conditionally independent given features (logistic likelihood). In truth residual within-country lag-1 autocorrelation is +0.11 (pipeline). | Standard errors and HL p-values are anti-conservative; a random row split would leak country effects into "validation". Mitigated: out-of-time + country-holdout splits, autocorrelation warning utility; SEs read as indicative only. |
| A2 | Annual macro data are point-in-time and correctly lagged. Real World Bank/IMF data arrive with 6-18 month lags and revisions. | Backtests overstate timeliness ("data staleness", VALIDATION §5); production must use vintage data and the `data/live.py` schema keeps year alignment explicit. |
| A3 | No structural regime shifts: WOE bins and coefficients fitted on 1995-2014 hold in 2015-23. | Calibration breaks exactly as demonstrated by the planted 2020 contagion year (predicted 6.4% vs realised 16.7%); PSI monitoring + annual re-bin are the controls. |
| A4 | A peg dummy captures FX-regime risk linearly. In reality **pegs mask risk until they break** (Argentina 2001): observed volatility is suppressed while devaluation risk accumulates. | Peg-country PDs are understated in calm years and jump discontinuously; desk overlay required (DESK_GUIDE scenario 3). |
| A5 | PD term structure is a flat hazard from the 1y PD. | Understates front-loaded risk for stressed names (real CCC curves are inverted) and long-run risk for improving credits; CVA on multi-year trades mis-stated — acceptable for ≤2y FX forwards, not for 10y swaps. |
| A6 | FX spot is GBM with constant vol for exposure simulation. | EM currencies jump and have fat tails/vol smiles; 99% PFE understated for pegged/managed floats (the break IS the tail). Stress PFE with jump scenarios before setting limits on EM pairs. |
| A7 | Exposure independent of counterparty default (no wrong-way risk) in CVA. | For an EM sovereign counterparty on a long-USD forward this is the *worst possible* assumption — default and devaluation coincide; VALIDATION §4 quantifies a 6x CVA understatement. |
| A8 | Settlement timing: pay at sold-currency window open, finality at bought-currency window close, one representative winter day. | DST shifts and intraday payment scheduling change windows by ±1-2h; extended Fedwire/CHIPS hours reduce them. Numbers are conservative bounds, not clock-accurate; treasury should feed actual cut-off agreements. |
| A9 | Rating-band PD midpoints (AAA 1bp … C 40%) anchored to agency long-run studies are correct through the cycle. | All downstream CVA/capital scales linearly in these; a misanchored band shifts every limit. Governance: annual recalibration against realised default studies. |
| A10 | Vasicek single-factor with ρ_sov = 0.30 describes sovereign tail dependence. | Multi-factor reality (regional blocs) means one global ρ can under/overstate joint tails; ρ = 0.30 is a documented, deliberately conservative choice vs Basel corporate. |

## 4. Numerical/engineering notes

- IRLS stops on max |Δβ| < 1e-10; separation detected at |β| > 30 with an
  informative `ValueError` and ridge fallback (unit-tested both ways).
- WOE smoothing 0.5 (Laplace) keeps empty cells finite; tests use
  smoothing = 0 for exact hand arithmetic. Low-cardinality features (regime
  dummy) get one bin per value — quantile edges would collapse them into a
  degenerate constant column (collinear with intercept).
- All Monte Carlo uses `numpy.random.default_rng(seed)`; GBM uses the exact
  lognormal scheme, so the martingale test is a 3-standard-error check, not a
  discretisation compromise.
