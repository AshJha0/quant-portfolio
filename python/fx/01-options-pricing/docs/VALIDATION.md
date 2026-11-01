# Validation: FX Options Pricing and Greeks Engine

All numbers below are produced by `python examples/run_pipeline.py`
(seeded, offline, under 1 s) unless a test is named, and enforced by the
test suite (`python -m pytest tests -q`, 2202 tests, offline and seeded;
most are parametrised grid points).

## 1. Analytic benchmarks

| Check | Evidence | Tolerance |
|---|---|---|
| Textbook value (Haug) | GK call S=1.56, K=1.60, T=0.5, r_d=6%, r_f=8%, σ=12% → **0.029099** vs published 0.0291 | 1e-4 |
| Textbook value (Hull, GBP call) | S=K=1.60, T=4/12, r_d=8%, r_f=11%, σ=14.1% → **0.042958** vs published 0.0430 | 1e-4 |
| Put-call parity (two rates) | `C − P = S·e^{−r_f T} − K·e^{−r_d T}` across 7-point grid incl. negative rates & JPY levels | 1e-10 |
| GK ≡ BS with q=r_f | vs independent dividend-yield BS implementation | 1e-14 |
| Independent quadrature | GK vs direct integration of payoff × lognormal density | 1e-10 |
| CIP / synthetic forward | Option-implied forward `K + (C−P)e^{r_d T}` vs `S·e^{(r_d−r_f)T}`: EURUSD diff 2.2e-16, USDJPY diff 0.0 | 1e-10 |
| Black-76 on F ≡ GK | across grid, both types | 1e-10 |
| Foreign-domestic symmetry | `C_d(S,K) = S·K·P_f(1/S,1/K)` over 54-point grid (3 spots × 3 strikes × 3 tenors × 3 rate pairs) | 1e-10 |
| Digital decomposition | vanilla = foreign-cash digital − K × domestic-cash digital | 1e-14 |
| Greeks vs finite differences | all 8 Greeks, 4 markets × call/put (incl. negative rates, JPY) | price rounding noise / stencil denominator + 1e-6 relative |
| Implied vol vs mpmath | ln(K/S) in [-5, 5], T in [1e-7, 50], sigma in [1e-4, 5]; pipeline grid max error **1.25e-15**; at most 10 evaluations | 4 max(eps max(S,K)/vega, 1e-12 sigma) |
| Golden vectors | 434 records from mpmath at 50 digits: GK price and Greeks, four delta conventions, implied vol, strike-from-delta in all four conventions, DNS ATM, CRR, forwards | per-output tolerance, worst 0.19 of it |
| Property tests (hypothesis) | parity, monotone in vol and strike, strike convexity, delta bounds, strike-from-delta and IV round trips | 300 examples each |
| Cross-language differential | Python vs Rust vs C++ on 790 seeded random requests | max abs deviation 1e-13 (prices), 7e-13 (strikes), 2e-13 (implied vol) |

Delta-convention checks: `Δ_f = Δ_s·e^{r_f T}` (1e-14);
`Δ_pa = Δ_spot − V/S` exactly; PA < unadjusted for all tested calls;
strike-from-delta round-trips for **all four conventions × call/put ×
{10Δ, 25Δ, 45Δ} × {EURUSD, USDJPY}** to 1e-8; ATM-DNS strike zeroes the
straddle delta to < 1e-12 under each convention; PA call solver returns
the larger-strike branch and raises on unattainable deltas.

## 2. Convergence studies

**CRR binomial → GK** (EURUSD 6m ATMF call, GK = 0.02522800):

| steps | tree price | abs error |
|---|---|---|
| 10 | 0.02533598 | 1.08e-4 |
| 100 | 0.02528631 | 5.8e-5 |
| 400 | 0.02522851 | 5.1e-7 |
| 800 | 0.02523450 | 6.5e-6 |
| 2000 | n/a | < 5e-6 (tested) |

(CRR oscillates, error is not monotone step-to-step; the tested claim is
convergence of the envelope, 2000-step error < 5e-6 abs, < 3e-4 rel at
JPY price levels.)

**Monte Carlo → GK** (same option, antithetic + control variate):

| paths | MC price | std error | error / SE |
|---|---|---|---|
| 1,000 | 0.02520693 | 2.7e-4 | 0.08 |
| 10,000 | 0.02509668 | 9.3e-5 | 1.41 |
| 100,000 | 0.02518816 | 3.0e-5 | 1.32 |
| 500,000 | 0.02520461 | 1.3e-5 | 1.74 |

Every statistical test is a 3-SE test. The control-variate coefficient is
fitted on the antithetic pair averages; fitting it on raw draws gave SE
1.39e-4 against 9.9e-5 for the control variate alone (100k paths, EURUSD
1y ATM), and now gives 4.8e-5 (`test_mc_variance_reduction.py`, seven
seeds). Monte Carlo needs two independent samples, three with the control
variate, or it raises;
digitals in both payout currencies match `e^{−r_d T}N(φd2)` /
`S·e^{−r_f T}N(φd1)` within 3 SE, and the test proves the naive
"discount foreign cash at r_f" shortcut misprices by > 1e-3.

**American exercise economics** (tree, 1000 steps): American ≥ European
everywhere; EURUSD call (r_f < r_d) early-exercise premium ≈ 1e-16 (none,
as theory demands); **USDJPY 6m ATMF call (r_f = 5.25% ≫ r_d = 0.50%):
early-exercise premium 0.628 JPY per USD = 14.4% of the European price** -
the foreign-carry-driven case that distinguishes FX from no-dividend
equities.

## 3. Hedging backtest (statistical)

Short EURUSD 6m ATMF call (premium 0.025228 USD/EUR), delta-hedged with
foreign-interest accounting, 2000–4000 GBM paths, seeded:

| rebalances | mean P&L | std P&L |
|---|---|---|
| 4 | +0.00027 | 0.01010 |
| 12 | +0.00019 | 0.00638 |
| 50 | −0.00003 | 0.00312 |
| 100 | +0.00003 | 0.00225 |
| 250 | −0.00000 | 0.00140 |

- Mean ≈ 0 at true vol (within 3 SE, and < 2% of premium), the hedge is
  unbiased. Std scales as **1/√N** (ratio test 25 vs 400 rebalances:
  4.0 expected, 3.94 observed).
- **Wrong vol**: selling at 10.25% vs realised 8.25% earns +0.00616
  (≈ the vega × vol gap, as theory predicts); selling cheap loses
  symmetrically.
- **Transaction costs** (1 pip half-spread, 100 rebalances): average cost
  0.00041 USD/EUR; cost drag grows with frequency while variance falls -
  the classic trade-off, both directions tested.

## 4. Failure modes (documented *and* reproduced where possible)

1. **Pegged currencies / jumps, CHF 15-Jan-2015 case study.** EURCHF
   traded ~1.2010 under the SNB floor with 1m vols in low single digits;
   when the floor was abandoned the pair fell to ~0.98 intraday (−18%, a
   >100-sigma move at quoted vol). GK assigns such a move probability
   ~e^{−5000}; any delta hedge gaps straight through. *Lesson encoded
   here*: the model validates inputs, but no diffusion model prices peg
   break risk, that risk lives in RR/BF quotes (project 9) and in jump
   models, and desks cap notional in managed pairs (see DESK_GUIDE.md).
   Reproduce the hedging failure: `simulate_delta_hedge(1.20, 1.20,
   0.25, -0.0075, -0.005, sigma_true=0.60, option_type="call",
   sigma_hedge=0.03, n_rebalances=50, n_paths=1000, rng=1)`, the P&L
   std is **14.4× the premium collected** (0.0983 vs 0.0068): a book
   priced at peg-regime vol is unhedgeable through the break.
2. **EM pairs, fat tails.** USDTRY/USDZAR realised kurtosis ≫ 3; flat-vol
   GK understates wings. The engine *functions* at EM parameters (tested
   at σ = 35–150%, r_d = 11.25%) but wing prices need the smile.
3. **Long-dated options.** The deterministic-rates assumption becomes
   first-order beyond about 2y. Prices are right within GK's model; against
   a stochastic-rates model the price and both rhos are biased. Not
   implemented.
4. **Implied vol near the bounds.** The solver works on the
   out-of-the-money side, so a tiny out-of-the-money price (2.7e-11, or
   6e-15) still inverts to 1e-12 or better. An in-the-money quote carries
   less information: its time value is the difference of the quote and the
   intrinsic bound, so a time value within `4 eps df max(F, K)` of the
   bound returns the sigma -> 0 limit (0.0). A price within `4 eps upper`
   of the sigma -> infinity bound raises `ValueError`, because no finite
   vol is identifiable there (`test_long_dated_deep_itm_high_vol_flat_plateau_raises`).
   Prices outside the bounds raise with the bound in the message.
5. **CIP basis.** `cip_forward` is the textbook forward; the tradable
   forward embeds the cross-currency basis (5–50bp post-2008,
   quarter-end spikes). Use market forward points in production.
6. **Negative rates: handled, not a failure.** Full EUR/CHF-era test class
   (r_d = −0.75%, r_f = −0.50%): pricing, parity, Greeks, implied vol,
   tree and all four delta conventions pass; note `e^{−r_f T} > 1` makes
   spot delta exceed forward delta.
7. **CRR probability bound.** For extreme `|r_d − r_f|` with tiny σ and
   coarse steps, `p ∉ [0,1]`; the tree raises with the fix (more steps)
   in the message instead of returning garbage.

Additional FX-flavoured edge coverage (`tests/test_edge_cases_extra.py`):

- **Pegged/managed pairs** (EURCHF pre-break parameters, σ = 2%):
  strike-from-delta and implied-vol round-trips at peg vol; the 10Δ–25Δ
  strike ladder compresses to < 25% of its normal-vol width (why peg
  books *look* riskless); clean post-break revaluation at (S −18%, σ×15).
- **PA-delta fold exhibited explicitly**: at USDJPY 35%-vol parameters the
  second (smaller-strike) root sharing the 25Δ PA call delta is located by
  scan and shown to lie left of the solver's returned strike; PA call
  delta vanishes in *both* wings; high-vol PA **puts** stay monotone.
- **σ → ∞ limits**: spot delta saturates at `e^{−r_f T}`, PA call delta
  collapses to 0, put price → `K·e^{−r_d T}`.
- **Digital foreign–domestic symmetry**: domestic-cash digital call =
  S × foreign-cash digital put on the inverted pair (d1′ = −d2), 1e-12;
  digital call+put parity sums to `e^{−r_d T}`.
- **CIP-violation detection**: a 10-pip premium perturbation moves the
  option-implied synthetic forward off CIP by exactly
  `e^{r_d T}·10 pips`, the conversion-arbitrage signal is measurable.

Input hardening on the secondary API (`tests/test_input_hardening.py`):

- **NaN never becomes a price.** A guard spelled `if x <= 0: raise` accepts
  NaN silently, because every comparison against NaN is False. The helpers
  that take scalars *outside* `validate_inputs`, `forward_points`'
  `pip_factor`, `simulate_delta_hedge`'s `sigma_hedge` / `mu` /
  `transaction_cost_pips` / `pip_size`, `premium_adjust_spot_delta`'s
  premium and delta, the two delta-convention converters, and the
  synthetic-quote generator's `base_atm`/`skew`/`smile`/`noise`/`tenors` -
  are each asserted to raise `ValueError` on NaN, +Inf and −Inf rather
  than propagate a NaN result. Before the fix a NaN `pip_factor` returned
  `nan` forward points and a NaN `sigma_hedge` produced an all-NaN P&L
  vector with `mean_pnl = nan` and no error.
- **Arbitrage boundaries.** Calendar monotonicity of the call when
  `r_d = r_f` across T ∈ {0.05 … 5y}; `|Δ_spot| ≤ e^{−r_f T}`;
  domestic-cash digital ∈ `[0, e^{−r_d T}]`; foreign-cash digital
  ∈ `[0, S e^{−r_f T}]`.
- **Managed-pair tree guard reproduced**: HKD-style parameters
  (S = 7.80, r_d − r_f = 4.9%, σ = 0.2%) with `steps = 2` raise on
  `p ∉ [0,1]`; `steps = 4000` prices cleanly.

## 5. What is not validated here

Smile construction (RR/BF to strike vols), stochastic vol and rates,
jumps: projects 9 and later. The golden vectors in
`tests/golden/golden_vectors.json` are consumed by the Python, C++ and Rust
engines.
