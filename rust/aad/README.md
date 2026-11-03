# qp-aad

Tape-based reverse-mode algorithmic differentiation (plan workstream W19).
This crate is the Rust port of `qp_core.aad` (Python) and `cpp/aad` (C++).
All three use the same tape layout, the same three pricers and the same
oracles.

## The idea

Most pricers are a long chain of small arithmetic steps. The tape records
each step: its result, the earlier results it used, and the local partial
derivative with respect to each of them. One walk backwards along the tape
then gives the derivative of the price with respect to every input at once.
The walk starts from "d price / d price = 1" and pushes each node's adjoint
into its parents.

Formally, for nodes `v_i = phi_i(v_j)`, the adjoints satisfy
`vbar_j = sum_i vbar_i * d phi_i / d v_j`. They are seeded with `ybar = 1`
and evaluated in reverse order (Griewank and Walther 2008, *Evaluating
Derivatives*, ch. 3; Savine 2018, *Modern Computational Finance*, ch. 9).
The cost is a small constant multiple of one price, however many inputs
there are.

## What is differentiated

| Module | Price | Greeks | Oracle |
|---|---|---|---|
| `bs` | Black-Scholes call or put | delta, vega, rho, rho_q, dK, dT | closed form (to 1e-12) and central bumps (to 1e-6) |
| `mc` | GBM Monte Carlo, European and arithmetic-Asian calls | the same six, pathwise (Broadie and Glasserman 1996; Giles and Glasserman 2006) | central bumps on the same normals (to 1e-6); Black-Scholes within 3 standard errors |
| `swap` | swap PV off the dated OIS curve | `dPV / d ln P` per pillar; `dPV / d quote` per market quote | bumped `ln P` (to 1e-6); quotes bumped and the curve rebuilt with `qp-rates-curves` (to 1e-6) |

**Quote Greeks without re-bootstrapping.** The bootstrapped curve solves
`R(x, q) = 0`. Here `x_j = ln P(t_j)`, and `R_k` is quote `k`'s model rate
minus its quote. By the implicit function theorem, `dx/dq = J^{-1}` with
`J = dR/dx`. So `dPV/dq = g^T J^{-1}`, where `g = dPV/dx`. Each row of `J`
costs one adjoint sweep through one quote's residual, and the solve is an
18 x 18 dense system (Henrard 2014, *Interest Rate Modelling in the
Multi-Curve Framework*, ch. 4).

**Worked example.** The test swap is a payer with a 4 % fixed rate and 1m
notional, on the periods of the SOFR OIS 7Y quote in the QuantLib 1.44 USD
fixture. The adjoint quote Greeks match central bumps of each quote
(`h = 1e-6`, curve rebuilt each time) to 1e-6 relative.

## Cost ratio (AAD time / price time)

`cargo run --release --example cost_ratio` takes the best of 21 runs (MC and
swap). Black-Scholes takes about 20 ns, so its two timings are interleaved
rep by rep over 201 reps of 20,000 calls, with every input passed through
`black_box`; the ratio of the two best times and the median of the per-rep
ratios are both printed. The ratios below come from one Windows laptop
(WSL) that was also running other builds, so treat them as a range, not a
benchmark.

| Pricer | Rust | C++ (g++ 15.2 -O3, WSL) | Python |
|---|---|---|---|
| Black-Scholes (one price), Round 17 | 13-34x | 2.9-3.4x | about 18-20x |
| Black-Scholes (one price), Round 18 | @@BS_RUST@@ | not changed | @@BS_PY@@ |
| MC European, 20k paths x 52 steps | 1.7-2.1x | 1.6-1.8x | about 3.4x (500 x 12) |
| MC Asian, 20k paths x 52 steps | 3.0-6.4x | 2.4-3.5x | n/a (Asian not timed) |
| Swap PV, 18 pillars, 10Y annual | 2.0-6.7x | 2.1-2.4x | about 3.6x |

The Round 17 Rust figure was partly a measuring error: only the spot went
through `black_box`, so the compiler folded most of the plain price and the
denominator was about 9 ns instead of about 20 ns. Measured fairly, the
Round 17 tape gave about 10-12x. Round 18 then made three changes, in
Rust and in Python:

1. **Tape reuse.** `bs::greeks` records on a per-thread tape that is
   cleared, not reallocated, between calls; `bs::greeks_on` takes a
   caller's tape. Inputs are recorded with one borrow (`Tape::vars`) and the
   adjoints read with one (`Tape::adjoints`), and the sweep stops above the
   inputs, which have no parents.
2. **Fused nodes** (statement-level preaccumulation; Griewank and Walther
   2008, s. 10.2; Savine 2018, ch. 15). `exp(-x y)`, `ln(x / y)`,
   `ncdf(c x)` and `x^2 / 2` are each one node holding their local
   partials, and `x * 1` records nothing. The Black-Scholes tape went from
   34 nodes to 25. The `f64` path of each fused operation is the plain
   composition, so the price did not move by a bit (tested).
3. **A leaner sweep**: one bounds check per node instead of three.

The plan's bar is "all Greeks under 5x one PV" (W19).
- The swap and the European MC meet it in every language.
- The Asian MC meets it in C++. In Rust it meets it on a quiet run but not on
  a loaded one.
- Black-Scholes in Rust: @@BS_VERDICT@@
- In Python every operation allocates an object, so its ratios measure
  interpreter overhead as much as the method. Tape reuse and fused nodes
  cut the Black-Scholes ratio, but it stays far above 5x.

The quote Greeks add 18 residual sweeps and one 18 x 18 solve. Bumping
instead would take 36 full bootstraps, each of them a bisection per pillar.

## Build

    cd rust/aad && cargo test && cargo clippy --all-targets -- -D warnings

The crate has its own empty `[workspace]` table until `aad` is added to
`rust/Cargo.toml` members. When it is added, delete that table and the
local `Cargo.lock`.
