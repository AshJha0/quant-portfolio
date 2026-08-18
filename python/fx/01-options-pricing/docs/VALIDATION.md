# Validation — FX Options Pricing & Greeks Engine

All numbers below are produced by `python examples/run_pipeline.py`
(seeded, offline, < 1s) and enforced by the test suite
(`python -m pytest tests -q`, **340 tests**, all offline & deterministic).

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
| Greeks vs finite differences | all 8 Greeks, 4 markets × call/put (incl. negative rates, JPY) | 1e-6 (1st order), 5e-5 (2nd order), scaled |
| Implied vol round-trip | 30-point grid (moneyness 0.85–1.20, T 0.25–2y) + EM 150% vol; pipeline: max error **1.75e-15** | 1e-8 |
| Golden vectors | 30 committed cases recomputed on every test run | 1e-10 |

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
| 2000 | — | < 5e-6 (tested) |

(CRR oscillates — error is not monotone step-to-step; the tested claim is
convergence of the envelope, 2000-step error < 5e-6 abs, < 3e-4 rel at
JPY price levels.)

**Monte Carlo → GK** (same option, antithetic + control variate, seed 42):

| paths | MC price | std error | error / SE |
|---|---|---|---|
| 1,000 | 0.02412280 | 8.2e-4 | 1.35 |
| 10,000 | 0.02508550 | 2.7e-4 | 0.53 |
| 100,000 | 0.02525380 | 8.5e-5 | 0.30 |
| 500,000 | 0.02523376 | 3.8e-5 | 0.15 |

Every statistical test is a 3-SE test. Variance reduction: ITM call SE
shrinks ~70% (antithetic + control variate vs plain, same path count);
digitals in both payout currencies match `e^{−r_d T}N(φd2)` /
`S·e^{−r_f T}N(φd1)` within 3 SE — and the test proves the naive
"discount foreign cash at r_f" shortcut misprices by > 1e-3.

**American exercise economics** (tree, 1000 steps): American ≥ European
everywhere; EURUSD call (r_f < r_d) early-exercise premium ≈ 1e-16 (none,
as theory demands); **USDJPY 6m ATMF call (r_f = 5.25% ≫ r_d = 0.50%):
early-exercise premium 0.628 JPY per USD = 14.4% of the European price** —
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

- Mean ≈ 0 at true vol (within 3 SE, and < 2% of premium) — the hedge is
  unbiased. Std scales as **1/√N** (ratio test 25 vs 400 rebalances:
  4.0 expected, 3.94 observed).
- **Wrong vol**: selling at 10.25% vs realised 8.25% earns +0.00616
  (≈ the vega × vol gap, as theory predicts); selling cheap loses
  symmetrically.
- **Transaction costs** (1 pip half-spread, 100 rebalances): average cost
  0.00041 USD/EUR; cost drag grows with frequency while variance falls —
  the classic trade-off, both directions tested.

## 4. Failure modes (documented *and* reproduced where possible)

1. **Pegged currencies / jumps — CHF 15-Jan-2015 case study.** EURCHF
   traded ~1.2010 under the SNB floor with 1m vols in low single digits;
   when the floor was abandoned the pair fell to ~0.98 intraday (−18%, a
   >100-sigma move at quoted vol). GK assigns such a move probability
   ~e^{−5000}; any delta hedge gaps straight through. *Lesson encoded
   here*: the model validates inputs, but no diffusion model prices peg
   break risk — that risk lives in RR/BF quotes (project 9) and in jump
   models, and desks cap notional in managed pairs (see DESK_GUIDE.md).
   Reproduce the hedging failure: `simulate_delta_hedge(1.20, 1.20,
   0.25, -0.0075, -0.005, sigma_true=0.60, option_type="call",
   sigma_hedge=0.03, n_rebalances=50, n_paths=1000, rng=1)` — the P&L
   std is **14.4× the premium collected** (0.0983 vs 0.0068): a book
   priced at peg-regime vol is unhedgeable through the break.
2. **EM pairs, fat tails.** USDTRY/USDZAR realised kurtosis ≫ 3; flat-vol
   GK understates wings. The engine *functions* at EM parameters (tested
   at σ = 35–150%, r_d = 11.25%) but wing prices need the smile.
3. **Long-dated options.** Deterministic-rates assumption becomes
   first-order beyond ~2y (rate vol ≈ FX vol contribution at 5y+). The 5y
   golden case prices correctly *within GK's world*; against a stochastic-
   rates model the price and both rhos are biased. Flagged, not fixed —
   out of scope by design.
4. **Ultra-short-dated far-OTM implied vol is numerically unrecoverable**:
   at T = 0.05, 15% OTM, the time value (~1e-18) sits below double
   precision resolution of the price (~1e-17 on an ITM premium); the
   solver returns the σ→0 limit rather than noise. Bounds violations raise
   `ValueError` with the offending bound in the message.
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

## 5. What is deliberately *not* validated here

Smile construction (RR/BF → strike vols), stochastic vol/rates, jumps —
projects 9+. The golden vectors in `tests/golden/golden_vectors.json` are
the frozen contract for the forthcoming C++/Rust engines.
