# Methodology

## Why a C++ FX engine at all?

The Python reference project (`python/fx/01-options-pricing`) is the
*model authority*: readable, scipy-backed, exhaustively documented. It is
not, however, what runs inside an FX e-trading stack, for three reasons:

1. **Quoting latency.** A market-making tile re-prices its full ladder
   (spots × strikes × tenors, both sides) on every spot tick and every curve
   or vol update. EURUSD ticks arrive in bursts of hundreds per second; a
   2,000-instrument tile refresh must fit in a fraction of a millisecond so
   the quote stream never lags the market it is derived from. This engine
   prices ~13M vanillas/s single-threaded, putting a full tile refresh at
   ~0.15 ms without any parallelism.
2. **RFQ response budgets.** Multi-dealer platforms effectively rank
   dealers by response time; an RFQ answer that takes tens of milliseconds
   loses flow to one that takes hundreds of microseconds. The full RFQ path
   — strike from quoted delta, vol from the smile, premium, hedge deltas —
   is a handful of engine calls, i.e. single-digit microseconds here,
   leaving the latency budget for the credit/limit checks and transport.
3. **Determinism and deployability.** No interpreter, no GIL, no dependency
   on a scipy build; a static library with bit-reproducible Monte Carlo
   (explicit `mt19937_64` seed + inverse-CDF normals, no
   implementation-defined `std::normal_distribution`) that risk can rerun
   and reconcile byte-for-byte.

The division of labour is deliberate: **Python defines the semantics, C++
reproduces them fast.** Any behavioural question ("what is theta's sign
convention?", "which PA-delta branch?") is answered by the Python source,
and the golden-vector suite proves the C++ engine gives the same answer.

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
* **Monte Carlo** (exact terminal-spot sampling under the domestic
  measure) exists as an independent cross-check of the analytic prices and
  as the template for payoffs with no closed form.

## Design choices in the C++ implementation

* **Pure free functions, no state.** Pricing is `double in -> double out`;
  the only stateful object is the MC RNG, which is constructed per call
  from an explicit seed. This makes the engine trivially thread-safe for
  read-only market data.
* **Errors are exceptions.** `std::invalid_argument` mirrors the Python
  `ValueError` contract exactly (same conditions, similar messages), so the
  two implementations reject the same inputs. Validation is *total* at the
  API boundary: every entry point rejects NaN/Inf in any argument —
  including the two rates and the observed premium — because a single
  non-finite quote propagating through a book-level aggregate is far more
  expensive to find than an exception in the service log.
* **Statistics are only reported when they exist.** The Monte Carlo
  standard error is estimated from *independent* samples (pair averages
  under antithetic sampling). With fewer than two of them the SE is
  reported as 0 rather than the `0/0` NaN the naive formula produces, and
  the header documents the minimum path count for a meaningful error bar.
  Likewise the finite-difference Greeks refuse a `T` or `sigma` too small
  for their own central down-bump instead of pricing at a negative input.
* **Own normal-distribution kernel.** `N(x)` via `std::erfc` (full double
  precision in both tails); `N^{-1}(p)` via Acklam's rational approximation
  polished with two Halley steps (~1e-15). No dependence on any math
  library beyond `<cmath>`.
* **Root finding.** Implied vol uses safeguarded Newton (fast when vega is
  healthy) with a bracketed Brent fallback (guaranteed); premium-adjusted
  strike-from-delta locates the fold of `K·N(d2(K))` analytically-in-d2 and
  Brent-solves on the market-standard decreasing branch — bit-matching the
  Python reference's scipy `brentq` strategy.
* **Limits handled explicitly**: `T = 0` returns intrinsic;
  `sigma·sqrt(T) <= 1e-12` returns discounted forward intrinsic — the same
  constants and branches as the Python module, so the two engines agree even
  at the edges.
* **`-Wall -Wextra -Werror -O2`**, C++20, no external dependencies beyond
  GoogleTest for the test binary.

## Assumptions

The full numbered assumptions register (CIP without cross-currency basis,
lognormal spot, constant rates/vol per pricing call, ACT/365F, no default
risk, etc.), with "what breaks if violated" for each, lives with the model
authority:
[`python/fx/01-options-pricing/docs/METHODOLOGY.md`](../../../python/fx/01-options-pricing/docs/METHODOLOGY.md).
This engine inherits every one of those assumptions verbatim; nothing in the
C++ port adds or relaxes any of them.
