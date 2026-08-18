# Methodology

## Why a Rust FX engine at all?

The Python reference project (`python/fx/01-options-pricing`) is the
*model authority*: readable, scipy-backed, exhaustively documented. It is
not, however, what runs inside a latency-sensitive e-trading stack, for the
same three reasons the C++ twin exists:

1. **Quoting latency.** A market-making tile re-prices its full ladder
   (spots × strikes × tenors, both sides) on every spot tick and every curve
   or vol update. This engine prices ~8.4M vanillas/s single-threaded
   (release profile, LTO), putting a full 2,000-instrument tile refresh at
   a fraction of a millisecond without any parallelism.
2. **RFQ response budgets.** The full RFQ path — strike from quoted delta,
   vol from the smile, premium, hedge deltas — is a handful of engine
   calls, i.e. low-single-digit microseconds here, leaving the latency
   budget for credit/limit checks and transport.
3. **Determinism and deployability.** A single statically-linked binary
   with no interpreter, no GC pause, and bit-reproducible Monte Carlo
   (explicit `u64` seed through an in-crate xoshiro256** + inverse-CDF
   normals, never a platform- or version-dependent RNG) that risk can rerun
   and reconcile byte-for-byte.

**Why Rust specifically, alongside the existing C++ engine?** The two
engines are not competing implementations of the same idea; they are two
different points on the safety/toolchain axis for the same production
requirement. C++ buys the widest interoperability with existing quant
libraries and trading infrastructure. Rust buys memory- and data-race
safety enforced by the compiler rather than by discipline and sanitizers —
relevant because a pricing library is exactly the kind of code that ends up
called from many threads (parallel tile refresh, concurrent RFQ handling)
with no room for a use-after-free or a torn read of shared market data to
become a silent, wrong price. Choosing `#![deny(unsafe_code)]`-compatible
safe Rust (this crate contains no `unsafe`) makes that guarantee load-bearing,
not aspirational.

The division of labour is deliberate, and identical to the C++ engine's:
**Python defines the semantics, the fast engines reproduce them.** Any
behavioural question ("what is theta's sign convention?", "which PA-delta
branch?") is answered by the Python source, and the golden-vector suite
proves this engine gives the same answer.

## Why zero dependencies

`Cargo.toml` declares no runtime dependencies. This is a portfolio-quality
choice, not a performance one — the crates this engine would otherwise pull
in (`rand` for the RNG, `statrs`/`libm` for the normal distribution,
`roots`/`argmin` for root finding) are all mature and fine to depend on in a
production system. Choosing to implement them in-crate instead is a
deliberate exercise in owning the numerics end-to-end and keeping the
review surface small and self-contained:

* **RNG**: `src/rng.rs` implements SplitMix64 (seed expansion, per the
  xoshiro authors' recommendation) and xoshiro256** 1.0 (Blackman & Vigna
  2018; passes BigCrush, period `2^256 - 1`) directly from the published
  algorithms, plus an inverse-CDF normal transform. This sidesteps the
  portability trap of `rand`'s `StandardNormal` (Box–Muller/Ziggurat
  variants that consume a variable, RNG-version-dependent number of
  underlying draws): one `u64` in, one normal out, so a recorded seed
  reproduces a Monte Carlo run bit-for-bit forever, independent of crate
  versions.
* **Normal distribution**: `norm_cdf` via W. J. Cody's rational-Chebyshev
  `erfc` approximation (the same algorithm `libm`/glibc use), accurate to
  ~1e-16 relative error including deep tails; `norm_ppf` via Acklam's
  rational approximation polished with two Halley iterations against
  `norm_cdf` to ~1e-15 absolute in `z`.
* **Root finding**: `implied_vol` uses safeguarded Newton–Raphson (fast
  when vega is healthy) falling back to bracketed Brent (guaranteed
  convergence, `scipy.optimize.brentq` semantics) implemented directly in
  `src/lib.rs`; premium-adjusted strike-from-delta locates the fold of
  `K N(d2(K))` analytically and Brent-solves on the market-standard
  decreasing branch.

## Model choice (recap; full discussion in the Python METHODOLOGY)

* **Garman–Kohlhagen** for European vanillas: Black–Scholes with the
  foreign rate as a continuous yield. Chosen over local/stochastic vol
  because vanilla FX quoting happens *in GK implied-vol space* — the model
  is the market's coordinate system, not a claim about dynamics. Chosen
  over numerical-only engines because closed forms give exact Greeks and
  microsecond latency.
* **Black-76 on the CIP forward** is provided separately because forward-
  space market data (outrights + domestic discount factor) is how desks
  actually receive inputs; it is algebraically identical to GK and tested
  as such to 1e-12.
* **CRR binomial** for American exercise (foreign rate enters exactly like
  a dividend yield; early exercise of a call becomes optimal when
  `r_f > r_d`). Chosen over finite differences for transparency and easy
  convergence testing against GK.
* **Monte Carlo** (exact terminal-spot sampling under the domestic measure)
  exists as an independent cross-check of the analytic prices and as the
  template for payoffs with no closed form. Two variance-reduction
  techniques are combined: antithetic pairing `(Z, -Z)` and a control
  variate on the discounted terminal spot `e^{-r_d T} S_T` (known mean
  `S e^{-r_f T}`). The combined estimator fits the control-variate
  coefficient on the *antithetic pair averages*, not on the raw per-draw
  values — see docs/VALIDATION.md for why that distinction matters
  numerically.

## Design choices in the Rust implementation

* **Pure functions, no shared state.** Pricing is `f64 in -> Result<f64,
  FxError> out`; the only stateful object is the MC RNG, constructed fresh
  per call from an explicit seed. Every public function is `Send + Sync`
  by construction (no interior mutability anywhere in the crate), so the
  engine is trivially safe to call from multiple threads over read-only
  market data — the compiler enforces this, it does not just happen to be
  true today.
* **Errors are values.** `FxResult<T> = Result<T, FxError>` mirrors the
  Python `ValueError` contract (same conditions, similar messages) without
  panics or exceptions; `FxError` distinguishes `InvalidInput` (bad market
  data) from `Numerical` (a root finder failed to converge/bracket).
* **Own normal-distribution kernel**, `N(x)` via Cody's `erfc`, `N^{-1}(p)`
  via Acklam + Halley — see "Why zero dependencies" above.
* **Limits handled explicitly**: `T = 0` returns intrinsic;
  `sigma * sqrt(T) <= 1e-12` returns discounted forward intrinsic — the
  same constants and branches as the Python module and the C++ engine, so
  all three agree even at the edges.
* **`#![warn(missing_docs)]`**, rustdoc (with runnable doctests) on every
  public item, `RUSTFLAGS="-D warnings"` in the tested build — the Rust
  analogue of the C++ engine's `-Wall -Wextra -Werror`.

## Assumptions

The full numbered assumptions register (CIP without cross-currency basis,
lognormal spot, constant rates/vol per pricing call, ACT/365F, no default
risk, etc.), with "what breaks if violated" for each, lives with the model
authority:
[`python/fx/01-options-pricing/docs/METHODOLOGY.md`](../../../python/fx/01-options-pricing/docs/METHODOLOGY.md).
This engine inherits every one of those assumptions verbatim; nothing in
the Rust port adds or relaxes any of them. Two implementation-level notes
worth calling out explicitly:

1. **Newton-then-Brent implied vol is only as good as the pricing
   formula's floating-point conditioning.** For deep ITM/OTM, very
   short-dated wings, the GK call/put formula's `S e^{-r_f T} N(d1) -
   K e^{-r_d T} N(d2)` difference-of-near-equal-terms can lose a digit or
   two of precision, and combined with a near-zero vega this bounds the
   recoverable implied vol to roughly `ULP(price) / vega`, independent of
   which root finder is used. `implied_vol` documents and tests this floor
   rather than pretending it away. What breaks if ignored: assuming
   implied vol round-trips to 1e-10 *everywhere* and silently trusting a
   ~1e-8 answer on an extreme short-dated wing as if it were 1e-10-accurate.
2. **Bit-reproducibility is a first-class deliverable of the Monte Carlo
   module**, not an incidental property: the seed → stream → normals path
   uses no OS entropy, no floating-point-order-dependent parallel
   reduction, and no RNG whose internal algorithm the crate doesn't control
   outright. What breaks if violated: an overnight risk number that cannot
   be reproduced to the bit during a P&L dispute.
