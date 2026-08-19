# Validation — Equity Options Pricing & Greeks Engine

This document answers documentation-contract items 3 and 4: **how the
engine was validated** (analytic benchmarks, convergence studies,
cross-model consistency) and **where it fails** (failure modes with
reproducible numbers). Every number below is produced by the committed
code — `PYTHONPATH=src python examples/run_pipeline.py` regenerates the
tables; `python -m pytest tests -q` (288 tests, offline, seeded) enforces
them permanently.

Reference contract unless stated otherwise:
`S=100, K=100, T=1y, r=5%, q=1%, sigma=20%`.

---

## 1. Analytic benchmarks

| Check | Result | Test |
|---|---|---|
| Put-call parity `C − P = S e^{-qT} − K e^{-rT}` over a 1080-point grid (S, K, T, r∈{−1%,0,5%}, q, σ) | max abs error < 1e-10 | `test_put_call_parity_full_grid_1e10` |
| Hull textbook: S=42, K=40, T=0.5, r=10%, σ=20% | call 4.7594, put 0.8086 (1e-4) | `test_textbook_values` |
| Haug generalized-BSM put (b=5% carry): S=75, K=70, T=0.5, r=10%, σ=35% | 4.0870 (1e-4) | `test_textbook_values` |
| Black-76 with `F = S e^{(r−q)T}` == BSM | abs diff < 1e-10 (8 cases) | `test_black76_equals_bsm_on_model_forward` |
| Analytic Greeks (delta, gamma, vega, theta, rho, vanna, volga) vs central finite differences | rel err < 1e-4, 10 scenarios × call/put | `test_analytic_vs_finite_difference_all_greeks` |
| Implied-vol round trip σ → price → σ, moneyness 0.5–2.0, T = 1w–5y, call+put | abs err < 1e-8 (40 cases) | `test_round_trip_sigma_price_sigma` |
| Golden vectors (32 cases, committed JSON for C++/Rust cross-validation) | reproduce to 1e-10 | `test_golden_vectors_reproduce_to_1e10` |

On the synthetic skewed chain (22 quotes, 2 expiries), the worst
implied-vol round-trip error is **3.1e-13**.

## 2. Convergence: CRR tree → Black-Scholes

European ATM call, reference contract (BS price **9.826298**):

| n_steps | tree price | abs error | error × n |
|---:|---:|---:|---:|
| 10 | 9.632077 | 0.194221 | 1.94 |
| 25 | 9.898850 | 0.072552 | 1.81 |
| 50 | 9.787003 | 0.039295 | 1.96 |
| 100 | 9.806625 | 0.019673 | 1.97 |
| 250 | 9.818423 | 0.007875 | 1.97 |
| 500 | 9.822359 | 0.003939 | 1.97 |
| 1000 | 9.824328 | 0.001970 | 1.97 |
| 2000 | 9.825313 | 0.000985 | 1.97 |

`error × n` is constant (≈1.97) — clean **O(1/n)** convergence. The
odd/even oscillation of CRR is visible at low n (n=25 overshoots from
above). Tests enforce monotone error decay along same-parity doublings and
agreement with BS at 500+ steps.

## 3. Convergence: Monte Carlo → Black-Scholes

Antithetic + control variate, independent seeds per row:

| n_paths | MC price | abs error | std error | SE × √n |
|---:|---:|---:|---:|---:|
| 1,000 | 9.491159 | 0.335138 | 0.241694 | 7.64 |
| 4,000 | 9.782944 | 0.043354 | 0.124725 | 7.89 |
| 16,000 | 9.866177 | 0.039879 | 0.062678 | 7.93 |
| 64,000 | 9.826117 | 0.000181 | 0.031205 | 7.89 |
| 256,000 | 9.833447 | 0.007149 | 0.015635 | 7.91 |

`SE × √n` constant ≈ 7.9 — the **O(n^{-1/2})** law. Every abs error is
within 3 SE of zero (tested for 4 contracts at 200k paths, plus CI
coverage).

Variance reduction at 100k paths (same seed): SE plain **0.0465** →
antithetic **0.0326** → control variate **0.0177** → both **0.0246**;
each technique individually reduces SE (tested). Note antithetic and the
control variate are *not* additive on a convex payoff — the CV alone is
the best single technique here; the harness reports whatever combination
you select, with honest error bars.

MC Greeks: pathwise and likelihood-ratio delta/vega match analytic values
within 3 SE (8 tests); pathwise has strictly lower variance than LR for
vanilla delta (tested), matching theory.

## 4. American exercise (CRR)

- American put ≥ European put on every tested contract; premium ≥ 0.
- American call with q=0 equals European call to 1e-10 (Merton's
  no-early-exercise theorem) — a sharp implementation test, since the
  early-exercise branch must fire *never* despite being evaluated at
  every node.
- Reference 1y ATM put (q=1%): American **6.366622**, early-exercise
  premium **0.423350** at 2000 steps.
- Literature benchmark S=K=100, T=1, r=5%, σ=20%, q=0: American put
  **6.0896 ± 0.005** at 2000 steps (tested).

Premium convergence (same-tree differencing cancels the O(1/n) error):

| n_steps | American put | premium |
|---:|---:|---:|
| 50 | 6.349163 | 0.444201 |
| 200 | 6.362745 | 0.428331 |
| 500 | 6.365355 | 0.425037 |
| 2000 | 6.366622 | 0.423350 |

## 5. Hedging validation

Short 3M ATM call (S=100, r=2%, σ=20%), 4000–8000 paths, seeded:

| n_rebalance | P&L std | std × √N |
|---:|---:|---:|
| 4 | 1.6134 | 3.23 |
| 16 | 0.8436 | 3.37 |
| 64 | 0.4399 | 3.52 |
| 256 | 0.2232 | 3.57 |

std × √N ≈ constant — the **1/√N** hedging-error law.

- Hedged at true vol, N=128: mean P&L **+0.0014** (SE 0.0035) — zero
  within error bars, with and without real-world drift µ=15% (tested).
- Sold at 25 vol, realized 15 vol: mean P&L **+1.996** vs the
  gamma-weighted vol-spread formula on the same paths **+1.987**
  (within 1 SE; tested at 4 SE tolerance).
- 5bp proportional costs at N=128: mean P&L drops from +0.0014 to
  **−0.2341** — the cost drag grows with frequency (tested).

## 6. Failure modes (know where it breaks)

### 6.1 Vol smile contradiction
BS assumes one σ for all strikes; the market does not. Pricing a 3M chain
with the (synthetic, mild) skew vs a single ATM vol:

| K | skew IV | put @ skew IV | put @ ATM IV | mispricing |
|---:|---:|---:|---:|---:|
| 80 | 25.6% | 0.1699 | 0.0344 | −80% of value |
| 90 | 22.4% | 0.9106 | 0.6426 | −29% |
| 110 (ITM) | 18.4% | 10.2500 | 10.4691 | +2% |

A flat-vol user underprices the 80-strike put by **5×**. Consequence:
never feed one vol across strikes; mark a surface (DESK_GUIDE.md). The
engine treats vol as a per-quote input precisely for this reason.

### 6.2 Discrete hedging error
Even with the *correct* vol, hedging 4× over a 3M option leaves P&L noise
of std ≈ **$1.61 on a $4.42 premium** (36%). This is not a bug — it is
the irreducible risk the BS "replication" argument assumes away
(assumption A2). Budget it: std ≈ 3.5/√N per short ATM call.

### 6.3 Dividend modeling error
Continuous yield q smears a discrete dividend across time. Stock pays
D=3 on ex-date t=0.1y; ATM calls, σ=20%, r=3%:

| Expiry | correct (discrete) | smeared q=3% | error |
|---|---:|---:|---:|
| T=0.08 (before ex-date) | 2.3756 | 2.2510 | **−0.125** (drops a dividend the option never sees) |
| T=0.12 (after ex-date) | 1.6015 | 2.7535 | **+1.152** (~72% overpricing: only 36% of the drop is applied by expiry) |

Rule: continuous q is acceptable for index baskets and long-dated
single-name options; for short-dated single names around ex-dates use
escrowed-dividend spot adjustment or a discrete-dividend tree. The
American-call exercise decision is similarly distorted: real early
exercise clusters immediately before ex-dates, which continuous q cannot
represent (it produces a smooth exercise boundary instead).

### 6.4 American pricing via CRR: limitations
- O(1/n) convergence with odd/even oscillation: n=10 misprices the
  reference call by 0.19 (~2%). Never quote off a coarse tree; use
  n ≥ 500 and same-parity Richardson if speed matters.
- Greeks off a tree are noisy (the discrete grid makes bumped prices
  non-smooth); `fd_greeks` on the tree needs bumps ≥ 1e-3 (tested) and
  gamma remains rough. A PDE solver is the right upgrade for smooth
  American Greeks.
- The risk-neutral probability `p = (e^{(r−q)Δt} − d)/(u − d)` leaves
  (0,1) when `|r − q|√Δt > σ` — the implementation *raises* rather than
  silently clamping (tested indirectly via input validation); increase
  n_steps in extreme carry/low-vol corners.
- Discrete dividends, term-structure r(t) and σ(t) are not implemented on
  the tree — constant-parameter CRR only.

### 6.5 Numerical limits (documented + tested)
- **T=0 / σ=0:** defined limits (intrinsic; discounted forward intrinsic)
  rather than NaN — tested for all engines.
- **Implied vol at the arbitrage bound:** price ≤ discounted forward
  intrinsic or ≥ upper bound raises `ValueError` — no silent garbage vol.
  Deep-ITM short-dated quotes lose all vol information in float64 (time
  value underflows); the round-trip domain is |ln(K/F)| ≲ 3σ√T and
  the solver refuses outside it.
- **No NaNs:** price/Greek sweeps across S∈[1e-3,1e5], T∈[1e-6,10],
  σ∈[1e-6,8], r∈[−5%,10%] are finite (tested); `RuntimeWarning`s are
  promoted to errors in the pytest config so silent overflow cannot creep
  in.
- **NaN/Inf inputs:** `validate_inputs` rejects NaN *and* ±Inf in S, K,
  T, σ with a `ValueError` (`"must be finite"`), for every engine —
  garbage market data fails loudly at the boundary instead of propagating
  Inf/NaN into the book (tested in `test_properties.py`).
- **Shape invariants (property tests):** homogeneity of degree 1 in
  (S, K) to 1e-12 relative; call/put monotone in σ; call delta monotone
  in S; butterfly convexity in K for BS, Black-76 and the American tree;
  American ≥ max(European, intrinsic); implied vol monotone in price;
  the T→0 and σ→0 limits are continuous (no jump at the documented
  boundary values).
- **Negative rates:** fully supported and cross-model consistent
  (tested), including the r<0 American-call early-exercise premium that
  surprises people trained on r≥0.
