# Methodology — FX Options Pricing & Greeks Engine

## 1. Why Garman–Kohlhagen? (model choice & alternatives)

The core pricer is **Garman–Kohlhagen (1983)** — Black–Scholes where the
continuous dividend yield is replaced by the foreign interest rate
(`q = r_f`). Holding foreign currency pays the foreign deposit rate, exactly
as a stock pays its dividend yield, so under the domestic risk-neutral
measure the spot drifts at `r_d − r_f` and

```
call = S·e^{−r_f T}·N(d1) − K·e^{−r_d T}·N(d2)
d1   = [ln(S/K) + (r_d − r_f + σ²/2)·T] / (σ√T),   d2 = d1 − σ√T
```

Alternatives considered:

| Model | Trade-off | Verdict for this project |
|---|---|---|
| **Garman–Kohlhagen (chosen)** | Closed-form prices and Greeks, exact put-call parity, instant calibration to one vol. Cannot produce a smile. | The market's *quoting* model: FX options are quoted as GK implied vols at deltas. Everything downstream (smile fitting, exotics) is built on top of it, so it must exist first and be bulletproof. |
| **Heston / SABR stochastic vol** | Captures smile dynamics and forward smiles; needed for path-dependent exotics and correct vanna/volga dynamics. Cost: 5+ parameters, unstable calibration on sparse quotes, no closed-form Greeks. | Overkill for vanillas at a single vol; smile construction and stochastic vol are **project 9**. Using Heston to price a quoted vanilla adds calibration risk with zero pricing benefit — the vanilla price *is* its quoted vol. |
| **Local volatility (Dupire)** | Reprices the whole vanilla surface by construction. Cost: needs a full arbitrage-free surface as *input* (which flat-vol GK cannot supply), poor smile dynamics for forward-starting risk. | Requires exactly the surface this project deliberately defers; degenerates to GK at flat vol anyway. |
| **Jump-diffusion (Merton, Bates)** | Captures gap risk (pegs, interventions). Cost: jump parameters unidentifiable from vanillas alone; hedging argument breaks (market incomplete). | Documented as a *failure mode* of GK (see VALIDATION.md §4, CHF depeg) rather than added as a pricer. |

Supporting models in the package and why each exists:

- **Black-76 on the FX forward** (`black76.py`) — with the CIP forward
  `F = S·e^{(r_d−r_f)T}` it is algebraically identical to GK (verified to
  1e-10). Included because market data arrives in forward space and because
  the equivalence is a free cross-implementation check.
- **CRR binomial tree** (`binomial.py`) — independent numerical check on GK
  *and* the only model here that prices **American** exercise (American FX
  options trade OTC; the economically interesting case is early exercise of
  a call on a high-carry currency, `r_f > r_d`).
- **Monte Carlo** (`monte_carlo.py`) — third independent check; the
  workhorse for payoffs without closed forms, demonstrated on
  cash-or-nothing digitals in *either* currency to make the
  measure/numeraire discipline explicit (foreign-cash digital =
  `S·e^{−r_f T}·N(d1)`, not the naive `e^{−r_f T}·N(d2)`).

## 2. FX conventions (used everywhere, tested everywhere)

1. **Quotation**: pairs are BASE/QUOTE — EURUSD = USD per 1 EUR. `S`, `K`,
   premiums are in **domestic** (= quote) currency per unit of **foreign**
   (= base) notional. `r_d` = quote-ccy rate, `r_f` = base-ccy rate,
   continuously compounded, annualised, ACT/365F.
2. **Forward**: covered interest parity `F = S·e^{(r_d−r_f)T}`; forward
   points = `(F − S) × 1e4` (× 1e2 for JPY-quoted pairs). Synthetic forward
   from options via parity: `F = K + (C − P)·e^{r_d T}`.
3. **Delta conventions** — the meaning of "25-delta" is part of the quote:

   | Convention | Call delta | When used |
   |---|---|---|
   | Spot | `e^{−r_f T}·N(d1)` | Premium in quote ccy, short-dated (EURUSD ≤ 1y) |
   | Forward | `N(d1)` | Hedging with forwards; long-dated / EM |
   | Spot premium-adjusted | `e^{−r_f T}·(K/F)·N(d2)` | Premium paid in base ccy — **USDJPY standard** |
   | Forward premium-adjusted | `(K/F)·N(d2)` | PA + forward hedge; long-dated PA pairs |

   Relations: `Δ_f = Δ_s·e^{r_f T}`; `Δ_pa = Δ − premium/S` (spot form).
   The premium adjustment exists because a premium received in base
   currency is *itself* a position in the underlying.

   **Strike-from-delta**: analytic for unadjusted conventions
   (`K = F·exp(−φ·N⁻¹(φΔ_f)·σ√T + σ²T/2)`). For **premium-adjusted calls**
   the map K → Δ is *not monotone* — `(K/F)·N(d2)` rises then falls, so a
   target delta has 0, 1 or 2 solutions. Market convention picks the
   larger-strike (decreasing) branch; the solver locates the fold point of
   `K·N(d2)` (root of `N(d2)σ√T = n(d2)`) and Brent-solves to its right,
   raising `ValueError` for unattainable deltas. PA put deltas are monotone
   and solved directly.
4. **ATM conventions**: ATM-forward `K = F`; ATM delta-neutral straddle
   `K_DNS = F·e^{+σ²T/2}` (unadjusted deltas, d1 = 0) or `F·e^{−σ²T/2}`
   (premium-adjusted, d2 = 0). DNS is the market's default "ATM".
5. **Foreign–domestic symmetry (notional duality)**: a EURUSD call is a
   USDEUR put: `C_d(S,K,T,r_d,r_f,σ) = S·K·P_f(1/S,1/K,T,r_f,r_d,σ)` —
   tested across a 54-point grid to 1e-10.
6. **Vol quotes (terminology only here)**: the market quotes ATM (DNS),
   25Δ/10Δ **risk reversals** `RR = σ_call − σ_put` (skew direction) and
   **butterflies** `BF = (σ_call + σ_put)/2 − σ_ATM` (smile curvature).
   `data/synthetic.py` generates stylised quote sets for illustration;
   **smile construction from these quotes is project 9** — all pricing in
   this project is at flat vol.

## 3. Assumptions register

Each assumption states what breaks when it is violated.

1. **Lognormal spot with flat, deterministic volatility.** FX smiles are
   pronounced — flat-vol GK **misprices the wings**: it undervalues low-Δ
   options in both tails whenever RR/BF ≠ 0 (a 10Δ EURUSD put marked at ATM
   vol can be under-priced by more than the RR alone). Breaks hardest for
   pairs with structural skew (JPY crosses, EM). Remedy: mark each strike
   with its smile vol — construction in **project 9**; the engine here
   accepts any per-strike σ.
2. **Deterministic interest rates in both currencies.** Rho is exact only
   for parallel deterministic shifts; there is no rate-FX correlation and
   no convexity from stochastic discounting. Acceptable below ~2y where
   vol risk dominates; **long-dated FX (5y+, PRDCs, FX-linked notes) needs
   stochastic rates** (e.g. a 3-factor Hull-White × 2 + FX model) because
   rate variance and rate-FX correlation become first-order in both price
   and delta. Breaks: long-dated pricing biased, both rhos understate true
   exposure.
3. **Continuous, frictionless hedging in both currencies.** Discrete
   rebalancing leaves P&L variance ∝ 1/N (verified in `hedging.py`);
   transaction costs (quoted in pips here) create the classic
   cost-vs-variance trade-off. Breaks in fast markets and around fixes/cuts
   when spreads blow out.
4. **No jumps; continuous spot path.** Pegged/managed currencies violate
   this catastrophically (CHF 15-Jan-2015: ~−18% in minutes; see
   VALIDATION.md §4). Delta hedging cannot cross a gap; short-gamma books
   realise the full jump. Breaks: hedged P&L distribution acquires a fat
   left tail invisible to GK.
5. **Covered interest parity holds exactly.** Post-2008 the **cross-currency
   basis** (persistently negative for USD funding pairs, wider at
   quarter-ends) violates textbook CIP by 5–50bp. Breaks: `cip_forward`
   deviates from the tradable outright; production systems must build
   forwards from the basis-adjusted curve, not from deposit rates.
6. **Perfectly liquid, two-way spot market at one price.** EM pairs and
   crisis regimes have spreads of many pips, and one-sided markets in
   stress. Breaks: hedging simulator's cost model understates slippage;
   delta conventions themselves can become ambiguous when the forward
   market dries up.
7. **T+2 settlement and cut-time details abstracted away.** Real desks
   price to the exact expiry cut (10am NY / 3pm Tokyo) and settle spot
   T+2 (T+1 USDCAD); day-count subtleties (NY holidays, weekends) shift T
   by up to ~1% for 1w options. Breaks: short-dated vols misquoted if T is
   naively `days/365`.

## 4. Numerical choices

- **Implied vol**: Newton from a moneyness-aware seed with Brent fallback
  on an expanding bracket; no-arbitrage bounds checked first
  (`[e^{−r_d T}·max(φ(F−K),0), S·e^{−r_f T}]` for calls). Round-trips to
  < 1e-8 across the tested grid (achieved ~1e-15).
- **Limits**: `T = 0` returns intrinsic; `σ = 0` returns discounted forward
  intrinsic — both exact, both unit-tested, no NaNs.
- **Tree**: CRR with `p = (e^{(r_d−r_f)Δt} − d)/(u − d)`; validates
  `p ∈ [0,1]` and raises with an actionable message otherwise.
- **MC**: exact terminal sampling (no discretisation bias), antithetic
  pairs + control variate `e^{−r_d T}S_T` (known mean `S·e^{−r_f T}`),
  seeded `numpy.random.Generator` everywhere; SE/CI returned in a
  dataclass, never a bare point estimate.
- **Golden vectors**: 30 cases (incl. negative rates, EM vol, JPY levels,
  5y and 1w tenors) committed to `tests/golden/golden_vectors.json` at
  1e-10 for the C++/Rust engines to cross-validate against.
