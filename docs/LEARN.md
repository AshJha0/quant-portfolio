# Learn: A Structured Guide to This Portfolio

> **Start at [docs/learn/](learn/README.md).** The teaching now runs by level,
> L0 to L4, each with a plain idea, a formal statement, a worked example and
> checks, plus chapters on stochastic calculus, rates, credit and
> microstructure. This file is the appendix those pages point into: the
> orientation in Parts I to III and the question bank in Part IV. Every
> question carries a level tag, and `docs/learn/L0/QUESTIONS.md` to
> `docs/learn/L4/QUESTIONS.md` list them by level.

This is the teaching companion to quant-portfolio. [README.md](../README.md)
tells you what the portfolio contains and how it is tested (test counts
come from the CI counts job, not from hand-typed totals). [ARCHITECTURE.md](ARCHITECTURE.md) and
[DIAGRAMS.md](DIAGRAMS.md) tell you how the 31 sub-projects fit together.
This file teaches the *content*: the quant finance, the numerical methods,
and the engineering practice behind every one of them, from first principles,
for someone who wants to actually learn this material, not just read that it
exists.

It is organized in four parts:

- **Part I: The Finance.** What each of the 13 areas is, in plain language,
  before you open a single line of code.
- **Part II: The Technology.** Why this portfolio is built the way it is:
  Python-first, C++/Rust ports for the hot path, cross-language golden
  vectors, and the testing discipline that ties it together.
- **Part III: Your learning path.** A suggested order to work through the
  portfolio, by background (quant-first, engineer-first, or starting from
  zero), with what to read and run at each step.
- **Part IV: The exercise room.** 532 self-test questions and answers,
  organized into 19 sets by topic, grounded in this portfolio's actual code
  and actual documented assumptions, not generic textbook trivia. Use it to
  check your own understanding after working through a project, or to find
  the gaps before you claim you know an area.

For hands-on practice next to this guide, the four
[learning notebooks](notebooks/README.md) (Black-Scholes and implied vol,
VaR and ES, curve bootstrap, Almgren-Chriss) and the
[career-prep packs](../career-prep/README.md) are described in
[CAREER_PREP.md](CAREER_PREP.md).

Every claim in this file is checkable against the actual repository: file
paths, function names, and test names are real, not illustrative.

---

## Part I: The Finance

### What problem is each area actually solving?

**Options Pricing & Greeks** (areas 01/01) answers "what should this
option be worth today, and how does that value move if the market moves?"
Black-Scholes (equity) and Garman-Kohlhagen (FX, which is Black-Scholes with
the dividend yield replaced by the foreign interest rate) give a closed-form
answer under a specific set of assumptions, constant volatility, frictionless
markets, continuous trading, no dividends beyond a constant yield, European
exercise. The Greeks (delta, gamma, vega, theta, rho) are the sensitivities a
trading desk actually hedges against, not academic curiosities: delta hedging removes
the first-order exposure but leaves gamma and theta, which have opposite signs:
a hedged long-option book is long gamma and pays theta, a hedged short-option
book is short gamma and collects it, and knowing that trade-off
numerically is the entire point of computing Greeks in the first place.
Implied volatility inverts the pricing formula, given a market price, what
constant volatility would have produced it?, which is how the market
actually quotes options (in vol terms, not price terms) on every options desk.

**Volatility Modeling & Forecasting** (areas 02/02) answers a different
question: not "what is the option worth today" but "what will realized
volatility look like tomorrow, next week, next month?" This matters because
Black-Scholes assumes constant volatility, but volatility is famously not
constant, it clusters (a volatile day is followed by more volatile days),
mean-reverts, and spikes around events. GARCH-family models capture the
clustering; the forecasts feed into everything from position sizing to VaR
inputs to vol-surface construction.

**Market Risk VaR / Expected Shortfall** (areas 03/03) answers "how much
could this portfolio lose, and how bad does it get in the tail?" Value-at-Risk
is a quantile of the P&L distribution (e.g., "the loss you won't exceed 99% of
the time"); Expected Shortfall is the average loss *given* you're in that
worst 1%, a materially different, and regulator-preferred, number because VaR
says nothing about how bad the tail actually is beyond the threshold. This
portfolio implements three independent ways to estimate both (historical
simulation, parametric/delta-normal with a Cornish-Fisher tail correction, and
Monte Carlo with full book revaluation), because a real risk function never
trusts a single VaR methodology, see Set 6 and Set 7 for why they
disagree with each other, and by how much, in specific and predictable ways.

**Fixed Income Pricing & Risk** (areas 04/04) covers bond and interest-rate
instrument pricing and the risk measures specific to rates (duration,
convexity, DV01), the analog of options Greeks for the rates world, where the
underlying risk factor is a whole curve, not a single spot price.

**Statistical Pairs Trading** (areas 05/05) is market-neutral relative-value
trading: find two instruments whose spread is stationary (cointegrated), trade
mean-reversion in that spread, and stay hedged against the *level* of the
market while still taking a directional bet on the *relationship* between two
assets. The hard part is not the trading logic, it's proving the spread is
actually stationary, in-sample and out-of-sample, before betting on it.

**Credit Risk / PD Modeling** (areas 06/06) answers "what is the probability
this counterparty defaults, and what do we lose if they do?", probability of
default (PD) modeling, typically via logistic regression or a structural
(Merton-style) model, feeding into expected-loss and capital calculations.

**Portfolio Optimization & Risk Allocation** (areas 07/07) is the
Markowitz mean-variance problem and its descendants (minimum-variance,
maximum-Sharpe, risk parity), given expected returns and a covariance
estimate, what weights should the portfolio hold, and how is risk actually
distributed across positions once you hold them (marginal/component risk
contribution, not just notional weight).

**Algorithmic Trading & Execution** (areas 08/08) is about *how* you get
into or out of a position, not whether you should, scheduling a large parent
order (TWAP, VWAP, participation-rate) to minimize market impact and
implementation shortfall, and measuring afterward how well you actually did
against a benchmark (transaction cost analysis).

**Volatility Surface & Stochastic Vol** (areas 09/09) extends area 01/01:
real markets don't quote one implied vol per underlying, they quote a whole
surface across strike and expiry (the "smile" and "term structure"), which a
constant-volatility model like Black-Scholes cannot produce internally, you
have to either interpolate the observed surface, or move to a model that
generates a smile endogenously, like Heston (stochastic volatility).

**Regime-Switching Quant Strategy** (areas 10/10) is the idea that markets
behave differently in different states (calm vs. stressed, trending vs.
mean-reverting) and a strategy that detects the current regime and adapts,
rather than using one fixed rule set across all conditions, can do better,
if the regime detection itself accounts for the risk of overfitting to
regimes that were obvious only in hindsight.

**The three foundations projects** predate the 10-area buildout and cover
the same ground once, more simply, single-asset, single-language: a
single-instrument risk-metrics toolkit (F1), strict no-look-ahead backtest
discipline for a simple moving-average strategy (F2), and a from-scratch
(zero-scipy) rebuild of Black-Scholes as a model-validation exercise in its
own right (F3). They are the right place to start if area 01-10's scope feels
like too much at once.

### The cross-cutting theme: every model here is validated at least three ways

No project in this portfolio ships with only "the tests pass." Every one
answers, in writing, the same six questions (see `CONVENTIONS.md`): why this
model over at least two alternatives; what assumptions it makes and what
breaks when each is violated; how it was validated (analytic identities,
convergence studies, statistical backtests, cross-model checks); where it
fails, with reproducible examples; how a trading desk would actually use it; and
what the real-life edge cases are (deep ITM/OTM, T→0, zero/negative rates,
illiquid/missing data, crisis regimes), each one both documented *and*
unit-tested, not just described. Part IV's question sets test you against
exactly this bar, area by area.

---

## Part II: The Technology

### Why Python first, and why only two areas get compiled ports

Every model in this portfolio has a Python reference implementation, full
stop, including the two areas (options pricing/Greeks and VaR/ES, each in both
equity and FX, so four engines) that also have C++ and Rust performance ports (the VaR engines port the linear book plus full option repricing per scenario). Python is not a
prototype language here; it is the *source of truth*. The C++/Rust engines are
required to match it, not the other way around, because Python's expressiveness
makes the modeling logic auditable in a way that a hand-optimized C++ kernel
never is, you want to review the math in the language that reads closest to
the math, and optimize a second time, separately, once the semantics are
locked down.

Only options pricing/Greeks and VaR/ES get compiled ports because they are the
two workloads on a trading desk that are actually called often enough, and are
numerically well-enough understood, to justify the engineering cost of a
second (or third) implementation: full-book Greeks revaluation and intraday
VaR both run on the order of millions of evaluations per cycle, on a
closed-form-or-Monte-Carlo core that doesn't change its semantics once
correct. The other eight areas, such as calibration, backtesting and portfolio
construction, are research workflows where Python's iteration speed matters
more than microsecond latency, so a second implementation would be pure
engineering cost with no corresponding desk benefit.

### What the November 2026 release added

The November 2026 release added four things a learner should know
about:

- **An `engine=` switch.** The options and VaR packages can now call the
  C++ or Rust port with `engine="cpp"` or `engine="rust"`. Python stays
  the default and the reference. `tests/test_engine_differential.py`
  compares the engines on 10,000 random inputs per group.
- **A research workflow on real data.** `data/` fetches ECB and FRED
  series; `research/` holds pre-registrations. The order is draft, sign,
  hash check, then run on the 2022-2025 hold-out. The regime study ran on
  the hold-out once and did not reject H0; the pairs study was withdrawn
  before it. Two later registrations (pairs v2, momentum), signed on
  2026-11-02, are scored once on forward data from 2026-11-03 to
  2028-11-02, on or after 2028-11-13. Every training, stress and
  calibration run is logged in `research/results/trials.jsonl`. Equity pairs and the order-book execution study are
  not done because their data needs an account.
- **More models, each with an oracle.** American pricers (Leisen-Reimer,
  Crank-Nicolson, ALO), Bachelier, SABR, higher Greeks, FX smiles and
  cuts, SSVI, Dupire, Bates, HAR-RV, OIS multi-curve, caps, swaptions,
  Hull-White, float-float cross-currency swaps, credit calibration, CDS,
  CVA and SA-CCR, HRP and robust portfolio methods, Euler risk attribution,
  FRTB IMA pieces, EVT and t-copula VaR. The README table pairs each with
  its oracle.
- **Performance kernels.** Philox, O(n) quantiles and blocked GEMM in
  `cpp/core` and `rust/core`. The batch `erfc` and `exp` kernels turned
  out slower than libm on an SSE2 build, and the README reports that with
  the numbers.
- **One option kernel per language.** Black-Scholes in cost-of-carry form
  lives once in `qp_core.black_scholes` (Python), `qp/black_scholes.hpp`
  (C++) and `qp_core::black_scholes` (Rust); the equity and FX packages
  are wrappers that keep their names and validation. Every output stayed
  bit-identical. NumPy entry points (`bs_price_array`, `gk_price_array`
  and the Greeks versions) and vectorised C++ `_fast` batches sit on it.
- **Builds you can check.** CI installs every Python package from
  `uv.lock` with hashes checked, the C++ engines install as the CMake
  package `QuantPortfolio`, CI builds wheels for the bindings, and the
  bench job compares base and head on the same runner. Set 19 in Part IV
  covers the ideas behind these.

### The determinism discipline

Every stochastic component in all 31 projects takes an explicit seed. This
sounds like a small thing until you've debugged a Monte Carlo test that fails
one run in twenty, a flaky test doesn't get fixed, it gets ignored, and an
ignored red CI signal is worse than no test at all. Determinism is what makes
"3 standard errors" a meaningful tolerance rather than a number nobody
actually checks because the test is already known to be unreliable.

### The golden-vector bridge

The single most distinctive engineering pattern in this portfolio is how the
C++ and Rust engines are validated against the Python reference: not by
re-deriving the math independently in each language and hoping they agree
(they might agree because both are wrong the same way), but by freezing a
fixed set of input/output pairs as "golden vectors" in a committed JSON file
and asserting every engine reproduces them. For the options engines the
expected outputs come from an independent mpmath oracle at 50 digits, with a
tolerance per output; for the VaR engines they come from the Python
reference on deterministic inputs. Generator scripts
(`tools/gen_golden_*.py`) turn the JSON into C++ headers and Rust modules,
and differential tests compare the engines on random inputs as well. A market-risk function uses the
same approach when a new pricing library has to be proven against
the incumbent before touching P&L, see [ARCHITECTURE.md](ARCHITECTURE.md)
"Cross-language validation" and [DIAGRAMS.md](DIAGRAMS.md) diagram 2 for the
full pipeline, and Set 15 below for the engineering theory in depth.

### The testing philosophy in one sentence

A test that only checks "does this match a number I computed by hand once"
catches typos, not wrong models, this portfolio instead leans on tests that
check properties the *correct* model must have regardless of implementation
detail: put-call parity, monotonicity in volatility, convexity in strike,
tree-to-closed-form convergence as steps increase, Greeks agreeing with
finite differences, and (for VaR/ES) Monte Carlo agreeing with the closed
form within a stated number of standard errors. Two portfolio-wide review
passes found real bugs this way, see Set 16 (the NaN-guard defect class)
and Set 18 (the numerical-robustness bugs and how each was actually found)
for the concrete stories, not just the abstract principle.

---

## Part III: Your learning path

Three suggested paths through the portfolio, depending on where you're
starting from. All three converge on the same question sets in Part IV.

### Path A: starting from zero (new to quant finance)

1. Read `python/foundations/03-black-scholes-replication/README.md` and its
   `docs/METHODOLOGY.md`, then read the actual `black_scholes.py` source,
   it's short, zero-scipy, and every line maps to a formula in the docstring.
2. Do the same for `python/foundations/01-risk-metrics` (VaR/ES/Sharpe on a
   single asset) and `python/foundations/02-trading-signal-backtest`
   (backtest discipline, read this one specifically for the no-look-ahead
   argument, it's the single most common real-world backtesting bug).
3. Work Part IV Set 1 (quant & math foundations) and Set 14 (the
   foundations trilogy) before moving on.
4. Move to `python/equity/01-options-pricing`, same model as F3, now
   at full scale: full input validation, implied-vol solving, binomial
   trees for American exercise, Monte Carlo cross-checks.
5. Work Set 2 and Set 3.
6. Pick two or three more areas that interest you (VaR/ES and portfolio
   optimization are a natural next pair, one measures risk, the other
   allocates around it) and read their `docs/METHODOLOGY.md` +
   `docs/DESK_GUIDE.md` before the code.

### Path B: quant-first (comfortable with the math, new to this codebase)

1. Skim `README.md` and [ARCHITECTURE.md](ARCHITECTURE.md) for the shape of
   the portfolio.
2. Go straight to whichever area's math you already know best, read its
   `docs/METHODOLOGY.md` "why this model vs. alternatives" section first,
   this portfolio's answer may differ from your intuition (e.g. why
   Cornish-Fisher rather than a Student-t or EVT tail for parametric
   VaR's tail correction), and that disagreement is exactly where you'll
   learn something.
3. Read `docs/VALIDATION.md` for the same project and try to find a gap in
   its stated failure modes before reading the edge-case tests, then check
   yourself against the actual test file.
4. Work the corresponding Part IV round, then the next area.
5. Set 17 (business/desk context) and Set 18 (postmortems) last, they
   assume you already know the models cold.

### Path C: engineer-first (comfortable with code, want the cross-language story)

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) "Cross-language validation" and
   [DIAGRAMS.md](DIAGRAMS.md) diagrams 1-2, then Set 15 in Part IV.
2. Pick one of the four compiled-port areas (equity options pricing is the
   simplest starting point). Read the Python `tests/golden/generate_golden.py`
   script, then `tools/gen_golden_header.py` and `tools/gen_golden_rs.py`,
   understand the generation pipeline before reading either engine.
3. Read the C++ engine's headers (`include/eqopt/*.hpp` or equivalent) side
   by side with the Rust engine's modules (`src/*.rs`), same algorithm,
   two idiomatic implementations, same golden vectors.
4. Work Set 15 and Set 16 (the NaN defect class, this is where the
   "just validate your inputs" advice gets concrete and portfolio-specific).
5. Read Set 18 (postmortems) for the numerical-robustness bugs, these are
   the kind of bug that only shows up when you already trust your tests, and
   the write-ups are about what *kind* of test would have
   caught each one sooner, which is the transferable lesson.

---

## Part IV: The exercise room

**532 questions across 19 sets**, organized to follow Part I's area-by-area
structure, then widen into the cross-cutting engineering and business
material. Each answer is grounded in this portfolio's actual code,
methodology docs, and test suite, where a number, function name, or file
path appears, it is real and checkable. Use this as a self-test: read the
question, try to answer it from what you've learned, then open the answer
(each one is folded under "Answer") and check yourself.

Every question carries tags: its topic (one per set, for example
`options` or `execution`), `numeric` when the answer quotes figures that you
can check against a VALIDATION file or test, and `code-reading` when it
points at specific functions or files. Search for a tag, for example
`` `execution` ``, to study one topic across the sets. Each question also
carries a level, `L0` (orientation) to `L4` (desk and validation judgement);
the levels are explained in [learn/README.md](learn/README.md). A `numeric`
answer always names the file, test or module its figures come from, and
`docs/learn/tools/lint_learn.py` reports any that does not.

| Set | Topic | Questions |
|---|---|---|
| 1 | Quant & Math Foundations | 28 |
| 2 | Options Pricing & Greeks | 34 |
| 3 | Implied Volatility & Numerical Solvers | 26 |
| 4 | Volatility Modeling & Forecasting | 26 |
| 5 | Volatility Surface & Stochastic Vol | 29 |
| 6 | Market Risk: VaR Methods | 32 |
| 7 | Expected Shortfall, Backtesting & Basel | 30 |
| 8 | Fixed Income Pricing & Risk | 28 |
| 9 | Statistical Pairs Trading | 29 |
| 10 | Credit Risk / PD Modeling | 29 |
| 11 | Portfolio Optimization & Risk Allocation | 29 |
| 12 | Algorithmic Trading & Execution | 26 |
| 13 | Regime-Switching Quant Strategy | 27 |
| 14 | The Foundations Trilogy | 22 |
| 15 | Cross-Language Engineering | 31 |
| 16 | Testing, Validation & the NaN Defect Class | 29 |
| 17 | Business, Desk Usage & Model Risk Governance | 26 |
| 18 | Real-World Postmortems & Judgment | 30 |
| 19 | Shared Kernels, Honest Statistics & Safe Builds | 21 |
| **Total** | | **532** |


## Set 1: Quant & Math Foundations

**Q1. Every VaR method in this portfolio ultimately reduces to "find a quantile of a return distribution." Walk through what that means concretely for `var_historical`.**

<sub>Tags: `L1`, `foundations-maths`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`var_historical` in
`python/foundations/01-risk-metrics/src/eq_risk_metrics/var_es.py` sorts
losses `L = -r` from worst to best, sets `k = n * (1 - confidence)` and `m = ceil(k)`
(clamped to `[1, n]`), and returns `L(m)`, the `m`-th largest loss. A quantile
is defined by sorting the sample into order statistics and picking the one at
the target rank: at 95% confidence you want the loss exceeded by 5% of
observations. This is the "historical" or empirical VaR: no distributional
shape is assumed, the quantile is read directly off the realised order
statistics with no interpolation, and losses are reported as positive numbers
by convention.

</details>

**Q2. Why is historical VaR a noisy estimator when n is small?**

<sub>Tags: `L1`, `foundations-maths`</sub>

<details><summary>Answer</summary>

At the 99th percentile on a 250-day sample, the empirical quantile is
determined by roughly the 2nd- or 3rd-worst observation out of 250, swapping
one bad day in or out of the window can move the estimate noticeably.
`docs/METHODOLOGY.md` for `01-risk-metrics` (§3.1) flags this directly: unlike
a parametric estimator that uses every observation to fit `mu` and `sigma`, an
order statistic in the tail is supported by only a handful of data points, so
its sampling variance is large exactly where you need precision most.

</details>

**Q3. What does it mean for the normal distribution to have "thin tails," and why does that make Gaussian VaR systematically wrong for daily equity returns?**

<sub>Tags: `L1`, `foundations-maths`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

The normal density decays like `exp(-x^2/2)`, so the probability it assigns to
an extreme move falls off extremely fast, a 5-standard-deviation daily move is
astronomically unlikely under Normal(mu, sigma). Real daily equity returns
have excess kurtosis (fatter tails than normal: more mass in the extremes than
a Gaussian would predict). `var_parametric` in `var_es.py` assumes `returns ~ Normal(mu, sigma)`
and computes `VaR = -(mu + sigma*z)`; because the true tail is fatter than the
normal model believes, this quantile sits closer to the center than the real
one, so at deep quantiles (99% and beyond) Gaussian VaR *understates* tail
risk, precisely the gap the project ships historical VaR alongside it to
expose (`docs/METHODOLOGY.md` §3.2).

</details>

**Q4. Where does the Student-t distribution show up in this portfolio, and why is it the standard fix for fat tails?**

<sub>Tags: `L1`, `foundations-maths`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

The Student-t density has heavier tails than the normal, controlled by its
degrees-of-freedom parameter `df`, with lower `df` meaning fatter tails and
`df -> infinity` recovering the normal exactly. It appears in two places doing
two different jobs: as a *return generator*,
`python/foundations/01-risk-metrics/src/eq_risk_metrics/data/synthetic.py`
builds its synthetic daily-return series from i.i.d. Student-t(6) shocks
standardised to unit variance (`_T_DF = 6.0`), so the bundled test data has
genuine fat tails rather than being secretly Gaussian; and as a *risk-factor
model*, `simulate_factor_returns` in
`python/equity/03-var-es-engine/src/eq_var/monte_carlo_var.py` offers
`dist="t"`, building a multivariate Student-t via `Z / sqrt(W/df)` (a normal
divided by a scaled chi-squared) with the scale matrix set to `cov * (df-2)/df`
so the *covariance* still matches the target exactly while the marginal tails
fatten.

</details>

**Q5. In `simulate_factor_returns`, why is the covariance scaled by `(df-2)/df` before it's fed to Cholesky for the Student-t case?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

A multivariate Student-t built as `Z / sqrt(W/df)` (with `Z` multivariate
normal from the Cholesky-factored scale matrix and `W ~ chi2(df)`) has
covariance equal to `df/(df-2)` times the *scale* matrix, not the scale matrix
itself, the division by `sqrt(W/df)` inflates variance by that factor. To make
the simulated covariance actually equal the caller's target `cov`, the code
pre-multiplies by the inverse factor `(df-2)/df` before factorising, so the
fattened tails come "for free" without silently also changing the second
moment the caller asked for.

</details>

**Q6. `01-risk-metrics/docs/METHODOLOGY.md` calls the Cornish-Fisher expansion "only a good approximation for mild skew/kurtosis" and warns it can become "non-monotonic." What does non-monotonic mean here and why is it disqualifying?**

<sub>Tags: `L1`, `foundations-maths`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Cornish-Fisher adjusts the Gaussian quantile `z` via `z_cf = z + (z^2-1)s/6 + (z^3-3z)k/24 - (2z^3-5z)s^2/36`,
a cubic-ish polynomial in `z` parameterised by sample skew `s` and excess
kurtosis `k`. A valid quantile function must be monotonically increasing in
the confidence level: the 99% quantile must always sit further into the tail
than the 95% quantile. With large enough `s`/`k`, the derivative of that
expansion can go negative on part of the domain, so the "99% VaR" it produces
can actually be *smaller* than the "95% VaR", a number that is mathematically
not a quantile at all. `python/equity/03-var-es-engine` handles this with
`cornish_fisher_valid(skew, excess_kurt, alpha)` in `parametric_var.py`: the
derivative of the CF map is a quadratic in `z`, so its minimum on `[-z_a, z_a]`
(with `z_a = Phi^{-1}(1 - alpha)`) has a closed form, and `cornish_fisher_var`
raises rather than returning a broken number when that minimum is not
positive. The check depends on `alpha`, because a dip in the derivative beyond
`z_a` does not affect the quantile being used.

</details>

**Q7. What is an order statistic, and how does the historical-VaR quantile relate to it?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

Sort a sample of `n` observations from smallest to largest; the `k`-th
smallest value is the `k`-th order statistic, written `X_(k)`. A sample
quantile at probability `p` is (up to interpolation convention) approximately
the `X_(round(p*n))` order statistic. `python/equity/03-var-es-engine`'s
`var_confidence_interval` (`monte_carlo_var.py`) makes this explicit and
quantitative: it treats the *rank* of the true alpha-quantile among `n` i.i.d.
draws as Binomial(n, alpha), inverts that via the Beta distribution of order
statistics (`beta_dist.cdf(alpha, ranks, n-ranks+1)`), and returns a
distribution-free confidence bracket `(-X_(hi), -X_(lo))` around the VaR
estimate, a direct, textbook application of order-statistic theory to quantify
how uncertain a single empirical quantile actually is.

</details>

**Q8. `03-var-es-engine` computes historical VaR as the order statistic `L(ceil(n*alpha))` with no interpolation. Why not use NumPy's default (type-7) quantile, and how does ES fit with it?**

<sub>Tags: `L1`, `foundations-maths`, `numeric`</sub>

<details><summary>Answer</summary>

NumPy's default (`method="linear"`, "type 7") linearly interpolates between
two bracketing order statistics, which produces a number that was never an
observed loss. `eq_var._tail` instead defines one estimator for every VaR
path: losses sorted descending, `k = n * alpha`, `m = ceil(k)` clamped to `[1, n]`,
`VaR = L(m)`. ES is the exact Acerbi-Tasche integral of the empirical quantile
function over the worst `alpha` of mass, `ES = [sum_{i<m} L(i) + (k - (m-1)) L(m)] / k`,
so when `k` is not an integer the order statistic `L(m)` enters with a
fractional weight. The age-weighted (BRW) version uses the same rule with
cumulative weights in place of ranks (`weighted_var_es`). One definition
everywhere means VaR and ES are consistent with each other and across methods.
`alpha` is the tail probability, in `(0, 0.5)`, in both the equity and FX
engines.

</details>

**Q9. What is a covariance matrix doing in `simulate_factor_returns`, and why can't you just simulate each risk factor independently?**

<sub>Tags: `L0`, `foundations-maths`</sub>

<details><summary>Answer</summary>

A covariance matrix `Sigma` encodes not just each factor's variance (the
diagonal) but how every pair of factors co-moves (the off-diagonal entries).
Simulating each factor with its own independent normal draw would produce a
joint distribution where every factor is uncorrelated with every other, which
is wrong whenever, say, two equity indices actually move together.
`simulate_factor_returns` needs the *simulated* scenarios to reproduce the
target `cov` exactly (in expectation), which requires generating correlated
draws, the entire reason Cholesky factorisation appears in the function at
all.

</details>

**Q10. Explain how Cholesky factorisation is actually used to turn independent normal draws into correlated ones, as in `simulate_factor_returns`.**

<sub>Tags: `L1`, `foundations-maths`</sub>

<details><summary>Answer</summary>

Cholesky factorisation writes a symmetric positive-definite covariance matrix
as `Sigma = L L^T` for a unique lower-triangular `L`. If `Z` is a vector of
independent standard normal draws, then `Y = mu + L Z` has covariance
`E[(LZ)(LZ)^T] = L E[ZZ^T] L^T = L I L^T = L L^T = Sigma` exactly, because
`Z`'s covariance is the identity. `simulate_factor_returns` does exactly this:
`chol = safe_cholesky(sig); z = rng.standard_normal((n_paths, n)); return mu + z @ chol.T`,
each row of independent normals is linearly mixed by `L` into a row with the
desired covariance structure, at essentially the cost of one matrix multiply
per path.

</details>

**Q11. What does "jitter escalation" mean in `safe_cholesky`, and why is it needed at all?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

Cholesky factorisation is only defined for a matrix that is (numerically)
positive *definite*; a matrix that is only positive semi-definite (has a zero
eigenvalue) or is indefinite due to floating-point noise makes
`np.linalg.cholesky` raise `LinAlgError`. `cholesky` in
`python/equity/03-var-es-engine/src/eq_var/monte_carlo_var.py` handles this by
adding a small multiple of the identity to the diagonal, starting at `1e-12 * mean(diag(cov))`
and rising x10 per try, capped at `1e-6 * mean(diag)`; any jitter used is
reported with a `RuntimeWarning`. This is "escalation": start with a
barely-perceptible nudge and grow it only as far as needed, so a borderline
matrix gets a negligible perturbation. Past the cap the function raises unless
the caller asked for an explicit repair (`repair="clip"` for eigenvalue
clipping or `repair="higham"` for the nearest correlation matrix). The older
`safe_cholesky` is a deprecated wrapper over the same capped ladder.

</details>

**Q12. Why would a covariance matrix built from pegged-FX-currency data be singular or near-singular in the first place?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

A currency peg (e.g. a currency pinned to the same anchor as another) makes
the two currencies' returns move together almost perfectly, correlation at or
extremely close to 1, or, if the peg is perfectly rigid, one factor's realised
variance is essentially zero. Either condition makes rows/columns of the
covariance matrix nearly (or exactly) linearly dependent, which is precisely
what "singular" means: the matrix has a zero (or numerically negative, due to
rounding) eigenvalue. The C++ FX-VaR engine states this explicitly: "A
covariance containing pegged currencies is routinely singular or numerically
indefinite (near-zero-vol factors, perfectly correlated pegs to the same
anchor)" (`cpp/fx-var-engine/include/fxvar/matrix.hpp`), which is exactly why
its capped-jitter `cholesky` (the C++ mirror of the Python `robust_cholesky`)
exists as a first-class, documented feature rather than an edge case to
special-case away: a singular FX covariance is a *legitimate* input, not a
data error.

</details>

**Q13. If jitter escalation lets a materially indefinite covariance matrix fail rather than being silently repaired, why is that the right design choice rather than always jittering until it works?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

The engines are explicit about the boundary: jitter starts at `1e-12 * mean(diag)`,
rises x10 per try and is capped at `1e-6 * mean(diag)`, small enough to fix
only numerical near-singularity, not to paper over a matrix that is *structurally*
wrong (e.g. built from bad or contradictory correlation inputs, with a large
negative eigenvalue). Silently jittering an arbitrarily bad matrix into
something factorisable would let a data-quality bug produce a
plausible-looking but meaningless simulation. Past the cap, with
`Repair::None` (the default) the C++ engine throws `std::runtime_error` naming
the most negative eigenvalue; a caller who wants a repair has to ask for
`Clip` or `Higham` explicitly. Any jitter used is recorded in
`CholeskyResult::jitter_added` and `warning`, so even a successful jitter is
surfaced as a diagnostic (`cpp/equity-var-engine/include/eqvar/matrix.hpp`).

</details>

**Q14. What is geometric Brownian motion (GBM), and where does it appear explicitly in this portfolio's code?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

GBM is the stochastic process `dS = mu*S*dt + sigma*S*dW`, where `dW` is a
Brownian motion increment: the *proportional* (not absolute) change in `S` has
constant drift `mu` and constant volatility `sigma`.
`python/foundations/03-black-scholes-replication/src/eq_bs_replication/black_scholes.py`
states this as assumption 1 in its module docstring, and `monte_carlo.py` uses
the SDE's known closed-form solution directly to simulate terminal prices:
`S_T = S0 * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z)` for `Z ~ N(0,1)`, the
risk-neutral version of GBM's exact solution, not a discretised approximation.

</details>

**Q15. Why does the GBM solution have a `- sigma^2/2` term in the drift (`S_T = S0*exp((r - sigma^2/2)T + sigma*sqrt(T)*Z)`) instead of just `r`?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

This is Ito's lemma at work: applying Ito's lemma to `ln(S)` under `dS = r*S*dt + sigma*S*dW`
(risk-neutral drift `r`) does not simply give `d(ln S) = r*dt + sigma*dW`,
because Ito's lemma for a function of a stochastic process includes a
second-order term (`-1/2 * sigma^2*S^2 * (d^2/dS^2) ln S * dt = -1/2*sigma^2*dt`)
absent from ordinary calculus. So `ln(S_T/S_0) ~ N((r - sigma^2/2)T, sigma^2 T)`,
and exponentiating gives the `- sigma^2/2` "Ito correction" in the exponent.
This is exactly why `S` itself has expected growth rate `r` (its mean grows at
the risk-free rate, as risk-neutral pricing requires) while `ln(S)` grows at
the *lower* rate `r - sigma^2/2`, the correction is what reconciles the two,
and it's the same correction that appears inside the Black-Scholes `d1`
formula (`(r + sigma^2/2)*T` in the numerator). The sign flips because `d1`
belongs to the stock-numeraire measure, under which the drift of `ln S` is `r + sigma^2/2`
rather than `r - sigma^2/2`.

</details>

**Q16. At a qualitative level, why does Ito's lemma applied to a delta-hedged options portfolio lead to the Black-Scholes PDE, and hence to a formula with no dependence on the stock's real-world drift `mu`?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

Building a portfolio long an option and short `delta = dV/dS` shares of stock,
then applying Ito's lemma to the option value `V(S,t)` as a function of the
GBM process `S`, produces a portfolio whose stochastic (`dW`) term cancels
exactly, the hedge removes all first-order sensitivity to the random shock,
leaving only deterministic terms in `dt`. A position with literally zero risk
must, under no-arbitrage, earn exactly the risk-free rate; setting the
portfolio's deterministic drift equal to `r` times its value is what produces
the Black-Scholes PDE. Because the risky drift `mu` cancelled out entirely in
the hedging argument (it enters through the `dt` term, `V_S mu S dt` from the
option and `-delta mu S dt` from the hedge, and those two cancel because
`delta = V_S`), the resulting price and PDE contain no `mu` at all, which is
why `call_price` in `black_scholes.py` takes `S, K, r, sigma, T` and never a
real-world expected-return parameter; risk preferences and drift beliefs are
hedged away, not needed. (`METHODOLOGY.md` calls this "the entire chain from
the SDE to the price... a single well-understood special function" once you
accept the replication argument, the full derivation is left to a later round;
this is the qualitative shape of why the formula has no `mu`.)

</details>

**Q17. `03-black-scholes-replication/docs/METHODOLOGY.md` interprets `N(d2)` and `N(d1)` economically. What do they mean, and how does that follow from the GBM/Ito machinery above?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

`N(d2)` is the risk-neutral probability that the option finishes in the money,
it falls directly out of the lognormal terminal distribution of `S_T` that
Ito's lemma produces: `d2` is (up to the `sigma*sqrt(T)` scaling) how many
standard deviations `ln(K)` sits below the mean of `ln(S_T)`, so `N(d2) = P(S_T > K)`
under the risk-neutral measure. `S0*N(d1)` is the discounted expectation
`e^{-rT} E[S_T 1{S_T > K}]`, the value of receiving the stock in the states
where the option is exercised (an unconditional expectation, not one
conditional on exercise). `N(d1)` itself is `P(S_T > K)` under the
stock-numeraire measure, the analogue of `N(d2)` under a different measure.
Both interpretations only make sense because `ln(S_T)` is exactly normal
(GBM's defining property under Ito's lemma), the whole closed form is an
integral against that specific lognormal density (`black_scholes.py` module
docstring).

</details>

**Q18. Why does Monte Carlo estimation error shrink as `1/sqrt(n)`, and where is that fact used explicitly in this portfolio?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

For `n` i.i.d. draws with finite variance `sigma^2`, the standard error of the
sample mean is `sigma/sqrt(n)` by the Central Limit Theorem / basic
variance-of-a-sum arithmetic (`Var(mean) = Var(X)/n`). `mc_call_price` in
`python/foundations/03-black-scholes-replication/src/eq_bs_replication/monte_carlo.py`
computes exactly this: `stderr = disc * std(units, ddof=1) / sqrt(len(units))`.
The module docstring is explicit that "the estimate must converge to the
Black-Scholes price at rate O(1/sqrt(n))," and `docs/VALIDATION.md` documents
a convergence table across four sample sizes each agreeing with the closed
form within 3 standard errors, with the standard error itself shrinking at the
predicted rate.

</details>

**Q19. What does "1/sqrt(n) convergence" actually imply about how many extra paths you need to halve your Monte Carlo error?**

<sub>Tags: `L0`, `foundations-maths`, `numeric`</sub>

<details><summary>Answer</summary>

Because error scales as `1/sqrt(n)`, halving the error requires *quadrupling*
`n`, not doubling it, going from 100,000 paths to 10,000,000 paths (100x more
compute) only buys you a 10x reduction in standard error. This is why Monte
Carlo is described in the codebase as reliable but expensive to push further:
`python/equity/03-var-es-engine/docs/METHODOLOGY.md` notes MC's "estimation
noise... controllable (paths up) but model risk remains" and quantifies a
concrete operating point, "100k paths -> SE ~ 0.8% of the 99% VaR on the
synthetic demo book" (assumption A10), rather than assuming more paths
trivially buys arbitrary precision.

</details>

**Q20. Why does `mc_call_price` need to compute its standard error from "pair averages" rather than treating every simulated payoff as an independent draw, when antithetic sampling is used?**

<sub>Tags: `L1`, `foundations-maths`, `numeric`</sub>

<details><summary>Answer</summary>

Antithetic variance reduction pairs each draw `Z` with `-Z` and averages the
two payoffs; because the payoff is a monotone function of `Z` for a call, the
two mirrored payoffs are *negatively correlated* by construction, that's
precisely the mechanism that reduces variance. Treating all `2m` mirrored
payoffs as `2m` independent observations for the standard-error formula
ignores that negative correlation and overstates the reported error, the
module docstring quantifies this at "about a third" too large for a 100k-path
ATM call. The correct estimator instead treats the `m` *pair averages* as the
independent unit: `units = 0.5*(payoff[:m] + payoff[m:])`, then `stderr = disc * std(units, ddof=1) / sqrt(m)`.
The broader lesson stated in `METHODOLOGY.md`: "a tolerance expressed in units
of your own error estimate is only as trustworthy as that estimate", a
validation suite that certifies "within 3 sigma" using a wrongly-computed
sigma is certifying its own arithmetic, not the model.

</details>

**Q21. What is the bootstrap, and what does `var_standard_error_bootstrap` in the VaR engine actually estimate with it?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

The bootstrap estimates the sampling distribution of a statistic by repeatedly
resampling the *observed* data with replacement (each resample the same size
as the original) and recomputing the statistic on each resample; the spread of
those recomputed values approximates the statistic's true sampling
variability, without requiring any assumption about the population's
underlying distribution. `var_standard_error_bootstrap` in
`python/equity/03-var-es-engine/src/eq_var/monte_carlo_var.py` resamples the
simulated (or historical) P&L array with replacement `n_boot` times
(`rng.integers(0, arr.size, size=(n_boot, arr.size))`), recomputes the
empirical VaR quantile on each resample, and returns the standard deviation of
those `n_boot` VaR estimates, a standard error for the VaR quantile itself,
attached to a number that otherwise ships as a single point estimate with no
error bar.

</details>

**Q22. Why is the bootstrap described as "distribution-free," and why does that matter for VaR specifically?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

The bootstrap makes no parametric assumption about the shape of the P&L
distribution, it works directly off the empirical sample by resampling it, so
whatever skew, kurtosis, or fat-tailedness the real data has is automatically
reflected in the resampled statistic's spread, with no need to specify (and
potentially misspecify) a normal, Student-t, or any other family. This matters
especially for VaR because VaR is a tail quantile, exactly the region where a
wrong distributional assumption (as `var_parametric`'s Gaussian assumption
demonstrably is, per Q3) does the most damage; the bootstrap's standard error
inherits the true, empirical tail shape rather than a possibly-wrong assumed
one, which is the codebase's stated reason it is "the desk-standard way to
attach error bars to an MC or historical VaR" (`monte_carlo_var.py`
docstring).

</details>

**Q23. Why does essentially every stochastic function in this portfolio, `mc_call_price`, `simulate_factor_returns`, `var_standard_error_bootstrap`, the synthetic data generators, take an explicit `seed` argument?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

A pseudo-random number generator seeded with the same integer produces the
exact same sequence of "random" draws every time.
`python/foundations/03-black-scholes-replication/src/eq_bs_replication/monte_carlo.py`
states the rationale directly in `mc_call_price`'s docstring: "Every
stochastic routine in this project takes an explicit seed so results are
reproducible." Without a fixed seed, a test asserting "MC price is within 3
standard errors of the closed form" would be nondeterministic, passing on some
runs and failing on others purely from RNG luck, which is exactly the kind of
flaky test the portfolio's testing contract (`CONVENTIONS.md`: "tests...
deterministic seeds") is designed to rule out. It also makes bug reports
reproducible: "seed 2 on the 01-risk-metrics synthetic generator produced this
exact number" is a debuggable claim; "some random run produced a weird number"
is not.

</details>

**Q24. `eq_risk_metrics/data/synthetic.py` documents that seed 2 is "the seed all documented results use." Why does a fixed default seed matter beyond just making individual tests deterministic?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

Every number quoted in `docs/METHODOLOGY.md` and `docs/VALIDATION.md` for
`01-risk-metrics`, the measured gap between Gaussian and historical VaR, the
specific Kupiec exception counts, the Jarque-Bera rejection, was computed
against one specific realisation of the synthetic Student-t return series. If
the default seed ever silently changed, every one of those hard-coded numbers
in the documentation would become stale and unverifiable without regenerating
them; pinning the default seed (and stating it explicitly in the docstring) is
what lets a reader independently re-run `examples/run_pipeline.py` and get the
identical numbers the documentation claims, rather than "numbers in this
general ballpark."

</details>

**Q25. What is the difference between correlation and causation, and why is that distinction relevant to a portfolio built around statistical VaR and (later) pairs trading?**

<sub>Tags: `L0`, `foundations-maths`</sub>

<details><summary>Answer</summary>

Correlation measures how two series move together statistically; it says
nothing about whether one *causes* the other, or whether both are driven by a
shared third factor. Two currencies pegged to the same anchor (Q12) are a
clean illustration purely of shared mechanism, not causation between the pair.
This matters directly for risk management:
`03-var-es-engine/docs/METHODOLOGY.md` assumption A9 warns that "correlations
spike toward 1 in crashes, diversification benefit evaporates exactly when
needed," i.e. a covariance matrix estimated in calm markets can badly
understate how correlated positions become under stress, precisely because a
historically-measured correlation is a statistical regularity of the sample
period, not a causal law that guarantees to hold going forward.

</details>

**Q26. Correlation and cointegration sound similar but answer different questions: what's the distinction, at the level needed before opening the pairs-trading projects?**

<sub>Tags: `L1`, `foundations-maths`, `code-reading`</sub>

<details><summary>Answer</summary>

Correlation measures how the *changes* (returns) of two series move together;
two series can be highly correlated in their day-to-day moves while drifting
arbitrarily far apart in level over time (e.g. two assets both trending up,
correlated in their daily wiggles, with no relationship between their price
levels at all). Cointegration is a statement about the *levels*: two (or more)
non-stationary series are cointegrated if some linear combination of them is
stationary, i.e. their spread doesn't wander off to infinity but keeps
reverting toward some equilibrium. `docs/LEARN.md` describes this exactly as
the foundation of the portfolio's pairs-trading projects: "find two
instruments whose spread is stationary (cointegrated), trade mean-reversion in
that spread... The hard part is not the trading logic, it's proving the spread
is actually stationary." A pairs trade is a bet on cointegration (a persistent
equilibrium relationship), not merely on correlation (which says nothing about
whether a divergence ever reverts), the full statistical machinery for testing
this (e.g. in `python/equity/05-pairs-trading/src/eq_pairs/cointegration.py`)
is the subject of a later round.

</details>

**Q27. Why does `01-risk-metrics/docs/METHODOLOGY.md` insist on using the Jarque-Bera test rather than just eyeballing a histogram before deciding whether Gaussian VaR is trustworthy?**

<sub>Tags: `L1`, `foundations-maths`, `numeric`</sub>

<details><summary>Answer</summary>

Jarque-Bera formalises the normality question into a test statistic, `JB = n/6 * (S^2 + K^2/4)`
(sample skewness `S`, excess kurtosis `K`), which is asymptotically
chi-squared(2) under the null of normality, it directly targets the same two
moments (skew, kurtosis) that Gaussian VaR ignores and that Cornish-Fisher
corrects for, giving a quantitative, checkable answer ("reject the normal
assumption at p < 0.05") rather than a subjective visual judgment. The
documentation is careful to also state the test's own limitation as an
assumption with "what breaks if violated": the chi-squared(2) reference is
only asymptotically valid, so on a small sample (tens to low hundreds of
observations) the test "has poor size... and low power," meaning a rejection
or non-rejection on a short series is weaker evidence than the raw p-value
suggests.

</details>

**Q28. Tie it together: why does this portfolio consistently prefer "ship multiple methods and measure where they disagree" over "pick the theoretically best method and use only that"?**

<sub>Tags: `L1`, `foundations-maths`</sub>

<details><summary>Answer</summary>

Every methodology document read for this round makes the same structural move:
`01-risk-metrics` ships historical, Gaussian, and Cornish-Fisher VaR side by
side specifically "to demonstrate its failure" (the Gaussian method, against
historical); `03-var-es-engine` runs historical, parametric, and Monte Carlo
VaR together "so the disagreement itself becomes a diagnostic";
`03-black-scholes-replication` ships a Monte Carlo pricer that "shares no
code" with the closed form specifically because internal self-consistency
checks (put-call parity, Greeks-vs-finite-differences) "share the same blind
spots as the code they check." The underlying quant-math reason connects
everything in this round: every method here rests on an assumption (normality,
i.i.d. returns, a representative sample window, a valid local expansion) that
is *sometimes* wrong, and the only way to find out which assumption is failing
on a given day's data is to compute an independent estimate that doesn't share
that assumption and measure the gap, which is exactly what quantile theory
(order statistics), fat-tailed distributions (Student-t), simulation error
bounds (`1/sqrt(n)`, bootstrap), and correlation structure (Cholesky,
covariance) collectively exist in this portfolio to make precise rather than
left as a vague expectation of "the model might be wrong somewhere."

</details>


## Set 2: Options Pricing & Greeks

**Q1. What does the Black-Scholes-Merton formula compute, and what does each term mean?**

<sub>Tags: `L1`, `options`</sub>

<details><summary>Answer</summary>

For a European call with continuous dividend yield,
`python/equity/01-options-pricing/src/eq_options/black_scholes.py::bs_price`
implements `C = S e^{-qT} N(d1) - K e^{-rT} N(d2)`. Read left to right: `S e^{-qT}`
is the present value of receiving the stock at expiry (spot minus the
dividends you forgo by holding the option instead of the stock), weighted by
`N(d1)`; `K e^{-rT}` is the present value of paying the strike, weighted by
`N(d2)`. `N(d1)` and `N(d2)` are the standard normal CDF evaluated at the two
moneyness/vol-adjusted terms defined by `d1_d2`. The formula is the
discounted, risk-neutral expected payoff of the option under geometric
Brownian motion (GBM), not a real-world probability statement.

</details>

**Q2. What are d1 and d2, and how is N(d1) interpreted?**

<sub>Tags: `L1`, `options`</sub>

<details><summary>Answer</summary>

`d1_d2` computes `d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))`
and `d2 = d1 - sigma sqrt(T)`. `d1` measures how many
volatility-standard-deviations the (risk-neutral, drift-adjusted)
log-moneyness is from zero. `N(d1)` has two standard readings: it is the
option's delta under a continuous dividend yield (`bs_greeks` returns `delta_ = df_q * norm.cdf(d1)`
for a call, i.e. `e^{-qT} N(d1)`, not `N(d1)` alone), and it is `P(S_T > K)`
under the measure that uses the stock as numeraire (Hull, ch. 15 and 28). With
`q = 0` the first reading is plain `N(d1)`, the hedge ratio a desk actually
trades against.

</details>

**Q3. How is N(d2) interpreted, and why isn't it just "the real-world probability of finishing in the money"?**

<sub>Tags: `L2`, `options`, `code-reading`</sub>

<details><summary>Answer</summary>

`N(d2)` is the risk-neutral probability that the call finishes in the money,
i.e. that `S_T > K` under the risk-neutral measure used to price the option
(the measure under which the discounted stock is a martingale). It is *not*
the real-world (physical-measure) probability, that would require the stock's
actual expected return, not `r - q`. This distinction is why `bs_price` prices
under `r` and `q` only, never a drift/expected-return input, and why the
`mc_price` engine in `monte_carlo.py` also simulates under `r - q - sigma^2/2`
drift rather than any real-world mean.

</details>

**Q4. What is the GBM / constant-volatility assumption, and what breaks when it's violated?**

<sub>Tags: `L1`, `options`, `numeric`</sub>

<details><summary>Answer</summary>

`METHODOLOGY.md` §2 registers this as A1: the underlying follows lognormal GBM
with continuous paths and a single constant `sigma`. Real returns have fat
tails, jumps and stochastic vol, so a single BS vol cannot price every strike
consistently, the market shows a skew/smile. `VALIDATION.md` §6.1 demonstrates
this directly: pricing a synthetic skewed chain with one flat ATM vol
underprices the 80-strike put by about 5x (0.0345 at the ATM vol against
0.1703 at its 25.6% skew vol, an 80% shortfall). Short-dated gamma/jump risk
from the same assumption shows up again in the `DESK_GUIDE.md` earnings-jump
scenario.

</details>

**Q5. What does the "frictionless, continuous hedging" assumption cover, and what's the real cost of violating it?**

<sub>Tags: `L1`, `options`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

A2 in `METHODOLOGY.md` assumes hedging can be done continuously with no
transaction costs. The engine's discrete-hedging simulator quantifies the
resulting irreducible P&L noise: the Derman-Kamal estimate of its std is
`sqrt(pi/4) * sigma * vega / sqrt(N)`, tested to shrink as `1/sqrt(N)`. Only
at the money is `sigma * vega` close to the premium, so the shorthand "premium
over root N" is an ATM rule. For the 3-month ATM case (S=100, r=2%, sigma=20%,
vega about 19.85, premium about $4.42) the estimate at N=4 is $1.76, while the
simulation gives $1.60 (36% of the premium, `VALIDATION.md` §5, §6.2). The gap
is expected: the estimate is a large-N approximation that treats gamma as
fixed over each interval, which is poor with only four rebalances. By N=256
the two agree (0.2195 simulated against 0.2199). With 5bp proportional
transaction costs, mean P&L drops from +$0.0014 to −$0.2341 at N=128
rebalances, cost drag that grows with hedging frequency even as the noise it
removes shrinks.

</details>

**Q6. What does the constant risk-free rate assumption cover, and where does it break?**

<sub>Tags: `L1`, `options`</sub>

<details><summary>Answer</summary>

A3 assumes a constant, deterministic `r` with no borrow/lend spread. Negative
rates are fully supported and tested (`bs_price`, `crr_price` and `mc_price`
all accept `r < 0`), but the assumption still breaks in two ways: rho hedging
is mis-stated when the real funding curve isn't flat, and discounting error
compounds with `T` in a stochastic-rates world where rates correlate with the
equity, a factorisation the deterministic-discounting BS formula can't
represent for long-dated options.

</details>

**Q7. Why model dividends as a continuous yield q, and what mispricing does that cause in practice?**

<sub>Tags: `L2`, `options`, `numeric`</sub>

<details><summary>Answer</summary>

A4 trades a single stock's discrete, lumpy dividend payments for a continuous
yield `q` for tractability. `VALIDATION.md` §6.3 quantifies the resulting
error on a stock paying D=3 on an ex-date at t=0.1y: an ATM call expiring at
T=0.08 (before the ex-date) is underpriced by 0.125 (the smeared yield spreads
a dividend the option never actually sees), while one expiring at T=0.12 (just
after) is overpriced by 1.152, about 72% too high, because only 36% of the
drop has been "applied" by the smeared-yield model by that date. The rule
documented there: continuous q is fine for index baskets and long-dated single
names; short-dated single-name options around ex-dates need an
escrowed-dividend spot adjustment or a discrete-dividend tree, and it also
distorts American-call early-exercise timing (see Q25).

</details>

**Q8. Why assume European exercise, and what does that assumption cost?**

<sub>Tags: `L2`, `options`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

A5 prices `bs_price`, `black76_price` and `mc_price` under European exercise
only. Since single-name listed equity options are actually American, ignoring
the early-exercise premium under-marks puts: `VALIDATION.md` §4 gives 0.423350
of early-exercise premium on the 1y ATM reference put (about 7% of its value)
at 2000 tree steps. The CRR tree in `binomial.py` is the one engine in the
project that handles this correctly by comparing continuation value against
intrinsic value at every node.

</details>

**Q9. What does assumption A6 (no counterparty/liquidity effects) protect against, and how is it enforced in code?**

<sub>Tags: `L1`, `options`, `code-reading`</sub>

<details><summary>Answer</summary>

A6 assumes frictionless, mid-market pricing with no bid/ask or crossed-quote
noise. In practice, quotes can be stale or crossed, and feeding a garbage
price into an implied-vol solver would otherwise silently return a garbage
vol. `implied_vol` in `black_scholes.py` checks the observed price against the
no-arbitrage bounds (`lower` = discounted forward intrinsic at sigma→0,
`upper` = `S e^{-qT}`/`K e^{-rT}` at sigma→∞) and raises `ValueError` if the
price sits outside them, rather than returning a number, this is documented as
the "first-line filter against crossed/stale quotes polluting the surface" in
`DESK_GUIDE.md` §1.1.

</details>

**Q10. What is A7 ("flat vol term structure"), and why does a desk care?**

<sub>Tags: `L1`, `options`</sub>

<details><summary>Answer</summary>

A7 assumes a single `sigma` covers the option's entire life, with no term
structure across expiries. `METHODOLOGY.md` notes this makes calendar-spread
risk invisible to a single-vol model, and theta computed at one fixed vol
misattributes P&L as the term structure rolls. `DESK_GUIDE.md` §1.4 states the
operational fix: desks mark a full vol surface (per strike *and* expiry) and
treat the engine's flat-vol interface as the contract that surface service
must fill, not as a claim that vol really is flat.

</details>

**Q11. What is A8, and how does the test suite quantify "the discretisation is large enough"?**

<sub>Tags: `L1`, `options`, `numeric`</sub>

<details><summary>Answer</summary>

A8 assumes tree/MC discretisation parameters (n_steps, n_paths) are large
enough that numerical error is negligible relative to a real bid/ask. It's not
free: `VALIDATION.md` §2 shows CRR error decaying as O(1/n) with visible
odd/even oscillation, at n=10 steps the reference ATM call misprices by 0.194
(~2% of the $9.826 premium), a real mispricing if you quoted off it. Both
convergence tables (CRR→BS and MC→BS) are enforced by tests, and
`VALIDATION.md` §6.4 explicitly warns never to quote off a coarse tree; use
n≥500.

</details>

**Q12. What is put-call parity, and how does this portfolio use it as both a pricing identity and a test?**

<sub>Tags: `L1`, `options`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Put-call parity states `C - P = S e^{-qT} - K e^{-rT}` (equity) or `C - P = S e^{-r_f T} - K e^{-r_d T}`
(FX, `garman_kohlhagen.py`). It follows purely from no-arbitrage replication,
independent of any distributional assumption, so it's a much stronger check
than "does the model match itself." `VALIDATION.md` (equity) enforces it to `< 1e-10`
over a 1080-point grid (`S` in {50, 80, 100, 120, 200}, `K` in {60, 100, 140},
`T` from 0.02 to 3 years, `r` in {-1%, 0, 5%}, `q` in {0, 3%}, `sigma` in {5%,
20%, 60%}; `test_put_call_parity_full_grid_1e10`); the FX project checks the
two-rate version to 1e-10 over a 7-point grid including negative rates and JPY
levels. As a *test*, it catches sign/discounting bugs that a single-price
benchmark against a textbook value would miss.

</details>

**Q13. What is delta, and what does a trading desk actually do with it?**

<sub>Tags: `L1`, `options`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Delta (`dV/dS`) is `bs_greeks(...).delta`, `df_q * norm.cdf(d1)` for a call.
`DESK_GUIDE.md` describes delta as auto-hedged with futures within a band on
the intraday risk desk, the hedging simulator's `1/sqrt(N)` law sets how tight
that band can be: rebalancing a 3-month ATM book 64x per quarter leaves about
$0.44 of P&L noise per option (`VALIDATION.md` §5), while tighter bands buy
less noise but pay more of the 5bp transaction-cost drag.

</details>

**Q14. What is gamma, and how does a desk manage gamma risk?**

<sub>Tags: `L1`, `options`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Gamma (`d^2V/dS^2`) is `bs_greeks(...).gamma`. It spikes near expiry when spot
sits on the strike: `DESK_GUIDE.md` §3.3 gives gamma at T=1/365, ATM, 20-vol
as 0.38, 20x the 1-year value, while delta flips 0↔1 through the strike
("pinning"). Since discrete-hedging error scales with `gamma * S^2`, a 1/√N
rebalancing budget that was fine at 3 months is inadequate on expiry day;
desks respond with expiry-day gamma limits and shrunk hedge bands. The
earnings-jump scenario (`DESK_GUIDE.md` §3.1) shows the same effect from a gap
rather than time decay: a short 1-week ATM straddle collecting $3.87 premium
carries gamma 0.164, so an 8% overnight jump costs ½·Γ·8² ≈ $5.26, more than
the premium collected, which is why the desk calls earnings risk "gamma risk,
not vega risk."

</details>

**Q15. What's the gamma/theta trade-off?**

<sub>Tags: `L1`, `options`</sub>

<details><summary>Answer</summary>

For a delta-hedged option under Black-Scholes, `theta + 0.5 sigma^2 S^2 gamma = r(V - delta S)`,
so with small rates `theta ≈ -0.5 sigma^2 S^2 gamma` (Hull, ch. 19): theta
scales with gamma and has the opposite sign. A long-gamma position pays theta;
a short option collects positive theta as compensation for its negative-gamma
exposure to large moves. Near expiry both grow together, so the extra risk on
expiry day is not that convexity outruns decay in the model, it is gap risk
and discrete-hedging risk: a move between rebalances on a position whose delta
swings from 0 to 1 over a small range of spot (the pinning scenario, Q14).

</details>

**Q16. What is vega, and how does a desk manage vega exposure?**

<sub>Tags: `L1`, `options`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Vega (`dV/dsigma`) is `bs_greeks(...).vega`, `S e^{-qT} phi(d1) sqrt(T)`,
currency per unit of annualised vol (divide by 100 for "per vol point," per
the `greeks.py` docstring). `DESK_GUIDE.md` §1.1 gives a market-making
example: a 1y ATM call (S = K = 100, sigma = 20%, r = 4%, q = 0) is worth
$9.93 with vega ≈ 38.1 per unit vol (0.381 per vol point), so a market quoted
0.4 vols wide spans about $0.15, about $0.076 of edge on each side of mid.
§1.2 aggregates vega to book level by expiry bucket, with vanna/volga limits
layered on top for the skew book, vega alone doesn't capture how the position
moves when the *smile* moves, only a parallel shift.

</details>

**Q17. What is rho, and why is it usually a lower-priority Greek?**

<sub>Tags: `L1`, `options`, `code-reading`</sub>

<details><summary>Answer</summary>

Rho (`dV/dr`) is `bs_greeks(...).rho`, `K T e^{-rT} N(d2)` for a call. It's
typically small relative to delta/gamma/vega risk for short- and medium-dated
equity options because rate moves are small relative to spot and vol moves
over the option's life, but assumption A3 (Q6) flags that rho hedging becomes
mis-stated whenever the real funding curve isn't flat, the model reports one
number under one deterministic rate, not a curve-shift-consistent exposure.

</details>

**Q18. Why is Garman-Kohlhagen "Black-Scholes with q = r_f"?**

<sub>Tags: `L2`, `options`, `code-reading`</sub>

<details><summary>Answer</summary>

`fx_options/garman_kohlhagen.py`'s docstring states it directly: holding
foreign currency pays the foreign risk-free rate, exactly as a dividend-paying
stock pays its yield, so under the domestic risk-neutral measure the spot
drifts at `r_d - r_f`. Substituting `q -> r_f` and `r -> r_d` into the BS
formula gives `gk_price`: `call = S e^{-r_f T} N(d1) - K e^{-r_d T} N(d2)`
with `d1 = [ln(S/K) + (r_d - r_f + sigma^2/2) T] / (sigma sqrt(T))`. The
economic content of the substitution: a dividend yield is compensation you
give up by holding the derivative instead of the asset; the foreign deposit
rate is exactly that same forgone-carry cost when the "asset" is a unit of
foreign currency. The FX project's `VALIDATION.md` confirms this isn't just an
analogy: `test_gk_equals_bs_with_dividend_yield` checks GK against an
independent dividend-yield BS implementation to 1e-14.

</details>

**Q19. What does the BASE/QUOTE convention mean, and how do r_d and r_f map onto it?**

<sub>Tags: `L1`, `options`, `code-reading`</sub>

<details><summary>Answer</summary>

Per `CONVENTIONS.md` and `garman_kohlhagen.py`'s docstring, FX pairs are
quoted BASE/QUOTE, e.g. EURUSD = USD per 1 EUR, so EUR is the base currency
and USD is the quote (domestic) currency. `S`, `K` and premiums are all in
domestic currency per unit of foreign notional. `r_d` is the quote-currency
rate, `r_f` is the base-currency rate; `gk_price`'s call formula discounts the
base-currency (foreign) leg at `r_f` and the quote-currency (domestic) leg at
`r_d`. This is the same domestic/foreign asymmetry as `q` vs `r` in equity,
the currency you're "long" through the option (base) plays the role of the
dividend-paying stock.

</details>

**Q20. What are the FX project's four delta conventions?**

<sub>Tags: `L1`, `options`</sub>

<details><summary>Answer</summary>

`fx/01-options-pricing/docs/METHODOLOGY.md` §2.3 documents all four: spot
delta `e^{-r_f T} N(d1)` (premium in quote currency, short-dated pairs like
EURUSD ≤ 1y); forward delta `N(d1)` (hedging with forwards, long-dated/EM);
spot premium-adjusted delta `e^{-r_f T} (K/F) N(d2)` (premium paid in base
currency, the USDJPY market standard); and forward premium-adjusted delta
`(K/F) N(d2)`. The relations `Delta_f = Delta_s * e^{r_f T}` and `Delta_pa = Delta - premium/S`
connect them, and are tested to 1e-14 in the FX project's `VALIDATION.md`.

</details>

**Q21. Why does premium adjustment exist, and what numerical wrinkle does it create?**

<sub>Tags: `L2`, `options`, `numeric`</sub>

<details><summary>Answer</summary>

Premium adjustment exists because a premium paid in the base currency is
itself a position in the underlying, e.g. a USDJPY option premium paid in USD
(the base currency) changes your dollar exposure, so the "true" hedge ratio
must net that out. The wrinkle: for premium-adjusted call deltas, the map from
strike to delta, `(K/F) N(d2)`, is not monotone, it rises then falls, so a
target delta can have 0, 1 or 2 solutions. `METHODOLOGY.md` documents the fix:
market convention picks the larger-strike (decreasing) branch, and the
strike-from-delta solver locates the fold point of `K N(d2)` (where `N(d2) sigma sqrt(T) = n(d2)`)
and Brent-solves to its right, raising `ValueError` for unattainable deltas.
`VALIDATION.md` §4 in the FX project reproduces the fold explicitly at USDJPY
35%-vol parameters, showing the second root lies left of the solver's returned
strike.

</details>

**Q22. Why doesn't equity need this delta-convention distinction?**

<sub>Tags: `L2`, `options`</sub>

<details><summary>Answer</summary>

Equity options have one currency in the trade, the premium, the strike, and
the underlying are all quoted in the same currency, so there's no analogue of
"premium paid in the other leg's currency" to adjust for, and no separate
base/foreign rate pair to create a spot-vs-forward delta split.
`CONVENTIONS.md`'s asset-class section states this plainly: equity conventions
are just `q`, ACT/365F day count and log-return vol; the
delta/forward/premium-adjusted quoting matrix is listed only under the FX
conventions, because it's an FX-specific artifact of trading a price that is
itself a ratio of two rate-bearing currencies.

</details>

**Q23. How does the CRR binomial tree price options, and what governs u, d and p?**

<sub>Tags: `L2`, `options`</sub>

<details><summary>Answer</summary>

`binomial.py::crr_price` builds a recombining tree with up-move `u = exp(sigma sqrt(dt))`,
down-move `d = 1/u`, and risk-neutral probability `p = (exp((r - q) dt) - d) / (u - d)`.
Terminal payoffs are computed in log-space for numerical stability, then
backward-induced through the tree by discounted expectation `pu * values[1:] + pd * values[:-1]`;
if the American flag is set, every node also takes the max against
immediate-exercise intrinsic. If `p` falls outside `(0, 1)`, a sign that `dt`
is too coarse relative to `|r - q|` and `sigma`, `crr_price` raises
`ValueError` rather than silently returning a mispriced tree.

</details>

**Q24. When does American early exercise actually matter, and how does the tree detect it?**

<sub>Tags: `L2`, `options`, `numeric`</sub>

<details><summary>Answer</summary>

For calls with `q = 0`, Merton's no-early-exercise theorem says American and
European calls are worth the same, `VALIDATION.md` §4 treats this as a sharp
implementation test, since the tree's early-exercise branch must fire *never*
despite being evaluated at every node, and confirms agreement to 1e-10. Early
exercise matters whenever there's a carry cost to holding the option instead
of the underlying: dividends (`q > 0`) make American calls valuable to
exercise early, and any positive interest rate (`r > 0`) gives the American
put an early-exercise premium over the European, because deep in the money the
interest earned on the strike received early outweighs the remaining
optionality. `binomial.py::early_exercise_premium` isolates this by
differencing American-minus-European on the *same* tree, which cancels the
shared O(1/n) discretisation error, the reference 1y ATM put (q=1%) shows a
0.423350 early-exercise premium at 2000 steps.

</details>

**Q25. Where do American and European prices diverge most sharply, and why?**

<sub>Tags: `L2`, `options`, `numeric`</sub>

<details><summary>Answer</summary>

Two concrete cases from the tests. Equity dividends: `VALIDATION.md` §6.3
notes continuous-q smears a discrete dividend across time, so the
American-call exercise decision is distorted too, real early exercise clusters
right before the ex-date (exercise when the dividend exceeds remaining time
value), which a continuous yield can't represent as a sharp boundary. FX
foreign carry: the FX project's `VALIDATION.md` §2 shows a USDJPY 6-month ATMF
call with `r_f = 5.25% >> r_d = 0.50%` carries an early-exercise premium of
0.628 JPY per USD, 14.4% of the European price, versus essentially zero
(1e-16) for an EURUSD call where `r_f < r_d`. High foreign-currency carry is
the FX-specific analogue of a high dividend yield.

</details>

**Q26. How is tree-to-closed-form convergence used as a validation technique, not just a sanity check?**

<sub>Tags: `L2`, `options`, `numeric`</sub>

<details><summary>Answer</summary>

`VALIDATION.md` §2 (equity) tabulates CRR error against the reference BS price
(9.826298) across `n_steps` from 10 to 2000, and shows `error x n` holds
constant at ≈1.97, clean O(1/n) convergence, with visible odd/even oscillation
at low n (n=25 overshoots from above). The constant `error x n` is itself the
evidence: if it weren't roughly constant, that would indicate a bug in the
tree rather than expected discretisation error. This is why the project ships
convergence tables as generated test output, not hand-picked numbers, `python -m pytest tests -q`
enforces monotone error decay along same-parity doublings, and
`test_black76_equals_bsm_on_model_forward` provides an analogous
closed-form-to-closed-form check.

</details>

**Q27. Why include Monte Carlo pricing when it's admittedly the "worst" tool for a vanilla?**

<sub>Tags: `L2`, `options`, `numeric`</sub>

<details><summary>Answer</summary>

`METHODOLOGY.md` §1.4 is explicit: for a vanilla under GBM, MC's O(n^{-1/2})
convergence is strictly worse than the closed form, and the comparison harness
quantifies that gap rather than hiding it. It earns its place for two reasons:
it's the only engine here that generalises to path-dependence, baskets, and
any dynamics you can simulate, and an MC-vs-closed-form agreement test (within
3 standard errors) is a powerful bug detector for *both* engines
simultaneously. `monte_carlo.py::mc_price` samples the terminal stock exactly
(`S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)`, no time-discretisation
bias) and combines antithetic variates with a discounted-terminal-stock
control variate, cutting standard error by about 2.6x at equal path count
(`VALIDATION.md` §3: SE goes from 0.0465 plain to 0.0177 with the control
variate alone at 100k paths).

</details>

**Q28. What edge cases does this portfolio's test suite actually cover for T→0, vol→0, vol→∞ and rates?**

<sub>Tags: `L2`, `options`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`black_scholes.py` and `binomial.py` both document explicit, tested limits
rather than letting these regimes fall through to NaN: `T == 0` returns
intrinsic value in every engine; `sigma == 0` returns discounted intrinsic on
the forward. `VALIDATION.md` §6.5 (equity) adds the harder cases: price/Greek
sweeps across `S in [1e-3, 1e5]`, `T in [1e-6, 10]`, `sigma in [1e-6, 8]`, `r in [-5%, 10%]`
are all finite (RuntimeWarnings are promoted to errors in pytest config, so
silent overflow can't creep in), and negative rates are fully supported and
cross-model consistent, including the counterintuitive result that American
calls can carry an early-exercise premium when `r < 0`. `validate_inputs`
rejects NaN and ±Inf in every numeric argument with a `ValueError` across
every engine.

</details>

**Q29. What is the "vol smile contradiction" failure mode, and why is it not treated as a bug?**

<sub>Tags: `L1`, `options`, `numeric`</sub>

<details><summary>Answer</summary>

`VALIDATION.md` §6.1 (equity) prices a 3-month synthetic skewed chain both at
each strike's own skew-implied vol and at a single flat ATM vol. The 80-strike
put is worth 0.1699 at its true 25.6% skew vol but only 0.0344 at the
18.4%-ish ATM vol used flat, an 80% underpricing (a 5x error), while the
110-strike ITM put is off by only about 2%. The document frames this as a
demonstration of assumption A1 (Q4), not a defect: the engine treats vol as a
per-quote *input*, taking one vol per strike/expiry from a marked surface,
precisely so this contradiction is visible and actionable rather than hidden
inside a model that pretends to fit the whole smile.

</details>

**Q30. What is the discrete-hedging failure mode, and how does it connect back to assumption A2?**

<sub>Tags: `L1`, `options`, `numeric`</sub>

<details><summary>Answer</summary>

BS's replication argument assumes continuous, frictionless rebalancing (A2,
Q5); the hedging simulator in the equity project shows what survives when
that's relaxed. Even hedged at the *correct* volatility, 4x rebalancing over a
3-month option leaves P&L noise of std ≈ $1.61 on a $4.42 premium (36% of the
premium), `VALIDATION.md` §6.2 explicitly calls this "not a bug" but the
irreducible risk the replication argument assumes away. The FX project's
version of the same experiment (`fx/01-options-pricing/docs/VALIDATION.md`
§4.1) sharpens the point: hedging a CHF option priced at peg-regime vol (3%)
through the 15-Jan-2015 SNB depeg leaves a P&L std 14.4x the premium
collected, proof that no amount of rebalancing frequency compensates for a
jump the diffusion model assumed away entirely.

</details>

**Q31. Why does `eq_options.lr_tree` (Leisen-Reimer) converge so much faster than the CRR tree, and how is that tested?**

<sub>Tags: `L2`, `options`</sub>

<details><summary>Answer</summary>

CRR's error oscillates with the number of steps because the strike falls at a
different place relative to the nodes each time. Leisen and Reimer (1996)
choose the up and down probabilities with the Peizer-Pratt inversion
(`peizer_pratt`) so the tree's terminal distribution is centred on the strike,
which removes the oscillation and gives smooth second-order convergence on an
odd number of steps. The oracle test prices Europeans against Black-Scholes
and checks that the error falls by a factor between 3 and 5 each time the step
count doubles, which is what second order predicts.

</details>

**Q32. `eq_options.alo` prices American options with the Andersen-Lake-Offengeim method. What is it checked against, and what does it refuse to do?**

<sub>Tags: `L2`, `options`, `numeric`</sub>

<details><summary>Answer</summary>

It is checked against a Richardson-extrapolated Leisen-Reimer tree (2001 and
4001 steps) in four cases, to 2e-5. That is a different numerical method with
the same model, so agreement is real evidence. `alo_price` iterates a fixed
point for the exercise boundary from a crude start, up to 60 times. Negative
rates are covered too. Whether early exercise pays depends on the carry of the
exercised position, `r K - q S`: with `r > 0` there is one boundary as usual;
with `r <= 0` and `q >= r` the carry never pays and the price is European;
with `q < r < 0` exercise pays only inside a band of spots that opens at
expiry and closes at a finite maturity. For that double-boundary case
`alo_price` solves the two value-matching equations for the band by Chebyshev
collocation: the band's mid-line and half-width live on nine nodes, and when
the band closes before expiry the closing time is one more unknown, fixed by
the condition that the option value touches the exercise value tangentially at
the tip. That takes about 15 ms and agrees with the extrapolated tree to 1e-5.
The earlier solver, which marches the two equations forward in time, is kept
as the reference (about 0.2 s) and as a fallback.

Source: `python/equity/01-options-pricing/src/eq_options/alo.py` and its tests in `python/equity/01-options-pricing/tests/`.

</details>

**Q33. What is Bachelier's vol in `eq_options.bachelier`, and why is it tested with mpmath rather than against another formula?**

<sub>Tags: `L1`, `options`</sub>

<details><summary>Answer</summary>

Bachelier models the forward as arithmetic Brownian motion, so its vol
`sigma_n` is in price units per square root of a year, not a percentage
(CONVENTIONS states this). The closed form is short, so a second closed form
would share any slip. The test instead integrates the payoff against the
normal density with mpmath at 30 digits and differentiates numerically for the
Greeks, which is an independent route to the same numbers.

</details>

**Q34. What two things does `eq_options.sabr` add beyond Hagan's 2002 formula, and what does it not implement?**

<sub>Tags: `L2`, `options`, `code-reading`</sub>

<details><summary>Answer</summary>

It offers Obloj's (2008) leading term and West's (2005) stable evaluation, and
`sabr_density_check` flags strikes where the implied density goes negative,
which Hagan's expansion does at low strikes. A precision bug was found and
fixed in testing: the naive `log` form of `x(z)` lost accuracy just above the
series cut-over and showed as a jump in the smile slope near ATM; it now uses
`log1p`. `sabr_pde` fixes it: the arbitrage-free SABR PDE of Hagan et al.
(2014) on a Le Floc'h-Kennedy grid. The density it produces is non-negative by
construction, total probability and the forward are conserved to rounding, and
in the test case where the expansion's density goes negative the PDE's call
prices are convex in strike. At short expiry it agrees with the expansion to
2e-5 in vol; at ten years it differs by about half a vol point at the money,
and that difference is second order in `T`, which is the part of the expansion
that was wrong.

</details>


## Set 3: Implied Volatility & Numerical Solvers

**Q1. Why is there no closed-form way to go from an observed option price back to the volatility that produced it?**

<sub>Tags: `L2`, `implied-vol`, `code-reading`</sub>

<details><summary>Answer</summary>

Black-Scholes (and Garman-Kohlhagen) give price as a function of sigma,
`price(sigma)`, but that function mixes sigma into the normal CDF terms
`N(d1)`, `N(d2)` in a way that cannot be algebraically inverted to `sigma = f(price)`.
The map is monotone and smooth, which is exactly what makes it solvable
numerically, but there is no elementary closed-form inverse. So implied vol is
always found by root-finding: pick sigma, price it, compare to the market
price, adjust, repeat.

</details>

**Q2. What equation is `implied_vol` actually solving?**

<sub>Tags: `L2`, `implied-vol`, `code-reading`</sub>

<details><summary>Answer</summary>

Both
`python/equity/01-options-pricing/src/eq_options/black_scholes.py::implied_vol`
and
`python/fx/01-options-pricing/src/fx_options/garman_kohlhagen.py::implied_vol`
map the quote to normalised Black form and call the shared core
`qp_core.iv_core.total_vol` (`python/core/src/qp_core/iv_core.py`). With
forward `F`, discount factor `df` and `x = -|ln(F/K)| <= 0`, the
out-of-the-money time value is normalised as `beta = (price - lower) / (df * sqrt(F K))`,
and the solver finds the total volatility `s = sigma * sqrt(T)` with `b(x, s) = beta`,
where `b(x, s) = e^{x/2} N(x/s + s/2) - e^{-x/2} N(x/s - s/2)` is the
normalised out-of-the-money call. The answer is `sigma = s / sqrt(T)`. Because
`b` is strictly increasing in `s`, the root is unique. The same algorithm,
constants and branch thresholds run in the C++ (`eqopt/iv_core.hpp`) and Rust
engines, so all six engines solve the same equation the same way.

</details>

**Q3. Why does the solver always work on the out-of-the-money side?**

<sub>Tags: `L2`, `implied-vol`, `numeric`</sub>

<details><summary>Answer</summary>

An in-the-money price is mostly intrinsic value, and the vol only lives in the
small time value on top of it. The solver converts an in-the-money quote to
its out-of-the-money counterpart by subtracting the discounted forward
intrinsic (put-call parity), so it always solves on time value. This does not
create information that is not there: the `implied_vol` docstring gives the
example S=100, K=90, T=0.02, sigma=10%, where one ulp of the call price
(1.8e-15) moves the vol by about 5e-4. The returned vol reprices the quote to
a few ulps but cannot be closer than that
(`test_k90_itm_call_is_backward_stable`), while the out-of-the-money put at
the same strike recovers the vol to 1e-12
(`test_k90_otm_put_recovers_to_1e12`). The practical rule is to quote the
out-of-the-money side when the vol matters.

</details>

**Q4. Write out the Newton and Halley update rules, and say which one this solver uses.**

<sub>Tags: `L2`, `implied-vol`, `numeric`</sub>

<details><summary>Answer</summary>

Newton: `s_{n+1} = s_n - f/f'`. Halley adds a curvature correction: `s_{n+1} = s_n - (f/f') / (1 - (f/f') f'' / (2 f'))`.
In `qp_core.iv_core.total_vol` this is literally `newton = f / fp; den = 1 - 0.5 * newton * fpp / fp; s_new = s - newton / den`.
The solver takes the Halley step when `den > 0.5` and the step stays inside
the current bracket; otherwise it falls back to a Newton step, and only then
to a bracket step (Q9). Newton converges quadratically near the root (correct
digits roughly double each step); Halley converges cubically (they roughly
triple), which is why the solver needs at most 10 evaluations and typically
2-4.

Source: `python/core/src/qp_core/iv_core.py`.

</details>

**Q5. Where do the derivatives come from: analytic or finite differences?**

<sub>Tags: `L2`, `implied-vol`</sub>

<details><summary>Answer</summary>

Analytic. The derivative of the normalised price in total vol is `db/ds = b_max * phi(d1)`
with `b_max = e^{x/2}` and `d1 = x/s + s/2` (this is vega in normalised
units), and the ratio `b''/b' = x^2/s^3 - s/4` is also closed form. The
objective on each branch is a function of `b` (Q6), so `f'` and `f''` follow
by the chain rule from these two expressions. No extra price evaluation and no
finite-difference noise enter the step.

</details>

**Q6. Why does the solver use three different objective functions instead of `b(s) - beta` directly?**

<sub>Tags: `L2`, `implied-vol`, `code-reading`</sub>

<details><summary>Answer</summary>

`b(s)` has an inflection point at `s_c = sqrt(2|x|)`. Below it, `b` is tiny
and grows roughly like `exp(-x^2/(2 s^2))`, so a raw residual is flat and a
Newton step in price space is badly scaled. The solver therefore picks the
objective by where `beta` sits: `f = 1/ln(beta) - 1/ln b(s)` on the lower
branch (`beta < b(s_c)`), `f = ln b(s) - ln(beta)` in the middle, and `f = ln(u) - ln(b_max - b(s))`
on the upper branch, where `u = b_max - beta` and the complement `b_max - b`
is computed as a sum of two positive terms with no cancellation. Each
transform makes the objective close to linear in `s` on its branch, which is
what lets a Halley step from a good initial guess land within a few ulps in
very few iterations.

</details>

**Q7. How does the solver get its initial guess?**

<sub>Tags: `L2`, `implied-vol`, `numeric`</sub>

<details><summary>Answer</summary>

On the lower branch it uses a Mills-ratio asymptotic for `ln b` in `y = s^2`,
iterated three times as a fixed point. On the middle and upper branches it
uses the Acklam inverse-normal approximation applied to `u / (b_max + 1/b_max)`
(always below 0.5), floored at `s_c`. The guess does not need to be exact,
only good enough that the first Halley step is already in the
cubic-convergence region; the tests confirm at most 10 evaluations over the
whole tested domain (`test_iv_round_trip_against_mpmath` counts them).

</details>

**Q8. What can make vega near-zero, and why does that matter for a Newton-type solver?**

<sub>Tags: `L2`, `implied-vol`, `code-reading`</sub>

<details><summary>Answer</summary>

Vega `~ S e^{-qT} phi(d1) sqrt(T)` shrinks toward zero deep in or out of the
money (large `|d1|`), for very short-dated options, and for very long-dated,
very high-vol options where `|d1|` is large again. In raw price space a Newton
step is `diff / vega`, so a tiny vega blows the step up into a jump far
outside any sensible vol range, or into `inf`/`nan`. The log and
reciprocal-log objectives divide this scale out: the solver works with ratios
such as `b'/b`, which stay well scaled even when `b` and `b'` are both tiny.
The hard regimes are pinned by `test_round_trip_deep_itm_low_vega`,
`test_round_trip_deep_otm_short_dated` and
`test_round_trip_long_dated_high_vol_flat_vega` in
`tests/test_implied_vol.py`.

</details>

**Q9. What keeps the iteration safe if a Halley step goes wrong?**

<sub>Tags: `L2`, `implied-vol`</sub>

<details><summary>Answer</summary>

A bracket. It starts at the a-priori bounds for the branch (`[0, s_c]` below
the inflection point, `[s_c, inf)` above it), and every evaluation shrinks it
using the sign of `f`. If the Halley step leaves the bracket, the solver tries
the plain Newton step; if that also leaves it, it probes the bracket end once
(the root often sits next to `s_c`), and only after that bisects (or doubles
`s` while the upper end is still infinite). Bisection is a last-resort step
inside the loop, not a separate solver stage, and there is no Brent or
bisection refinement pass at the end.
`test_root_at_the_inflection_point_converges_fast` shows why the probe
matters: bisecting toward `s_c` took 12 evaluations, probing it takes 2.

</details>

**Q10. What are the no-arbitrage bounds on an option's price, and why must implied vol be checked against them before solving?**

<sub>Tags: `L1`, `implied-vol`, `numeric`</sub>

<details><summary>Answer</summary>

As sigma runs from 0 to infinity, the option price runs monotonically between
two limits: the `sigma -> 0` bound (the discounted intrinsic value on the
forward) and the `sigma -> infinity` bound (`S*exp(-qT)` for a call,
`K*exp(-rT)` for a put in equity terms; `S*df_f` / `K*df_d` in FX terms). A
quoted price outside `(lower, upper)` cannot correspond to *any* volatility;
it's a violation of arbitrage, not a hard-to-find root, so both `implied_vol`
implementations check these bounds first and raise `ValueError`. The FX engine
makes the checks scale-aware: a price within `tau_lo = 4 eps df max(F, K)` of
the lower bound returns 0.0 (the time value is below the bound's own
rounding), and a price within `tau = 4 eps * upper` of the upper bound raises,
because no finite vol is identifiable there.

Source: `python/core/src/qp_core/iv_core.py` (the `lower` and `upper` bounds in the solver).

</details>

**Q11. What is the solver's stopping rule?**

<sub>Tags: `L1`, `implied-vol`</sub>

<details><summary>Answer</summary>

It stops on the size of the step in `s`, relative to `s`: `|s_new - s| <= max(tol, 4 eps) * s`,
with `tol = 1e-7` after a Halley step (the next error is about the cube of the
step) and `1e-10` after a Newton step (about the square), or when the bracket
has shrunk to `4 eps * s`. After a bracket step `tol` is zero, so the loop
continues until a real Halley or Newton step confirms convergence. Failing to
converge in 100 iterations raises `ValueError`; it never returns a best guess.

</details>

**Q12. Why does the stopping rule test the step in sigma-space rather than the price residual?**

<sub>Tags: `L2`, `implied-vol`, `numeric`</sub>

<details><summary>Answer</summary>

Because the quantity that has to be accurate is sigma, and in flat-vega
regions a tiny price residual can map through a near-zero vega to a large
sigma error. A loop that exits as soon as `|price(sigma) - target| < tol` can
declare convergence while sigma is still whole vol points away from the root.
The canonical corner is `test_round_trip_long_dated_high_vol_flat_vega`:
S=K=100, T=25y, sigma=300% put, where `|d1| ~ 7.7` and the price sits within
double-precision noise of the `sigma -> inf` bound. A rule based on the step
in `s`, combined with an objective that is nearly linear in `s`, measures the
error in the variable that matters.

</details>

**Q13. What accuracy does the solver promise, and why is it stated relative to vega?**

<sub>Tags: `L2`, `implied-vol`, `code-reading`</sub>

<details><summary>Answer</summary>

The docstrings state: within `max(eps * price / vega, 1e-12 * sigma)` of the
true root, over T in [1e-7, 50], sigma in [1e-4, 5] and `|ln(F/K)|` up to 5,
in at most 10 evaluations. The `eps * price / vega` term is the information
limit of the quote itself: a price is only known to about one rounding, `eps * price`,
and dividing by vega converts that price uncertainty into vol uncertainty. No
solver can beat it. `test_iv_round_trip_against_mpmath` checks the claim on a
grid against prices and vegas computed in mpmath, and allows the engine to
refuse (raise) only when the time value is within a few ulps of a bound.

</details>

**Q14. Why is a silently imprecise answer a worse failure than a raised error?**

<sub>Tags: `L2`, `implied-vol`</sub>

<details><summary>Answer</summary>

An error is loud: the caller's code stops, a test fails, someone investigates.
A silently imprecise return value looks like success (a plausible-looking
float, no exception, no warning) and gets used downstream in a Greeks
calculation, a P&L attribution, or a vol surface fit, corrupting everything
built on top of it with no signal that anything went wrong. That is why the
solver raises at the upper-bound plateau (Q15), raises on non-convergence, and
has no "return the best so far" path.

</details>

**Q15. What happens to the FX solver on the deep-in-the-money, long-dated, high-vol plateau?**

<sub>Tags: `L2`, `implied-vol`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Deep ITM plus long-dated plus high vol drives `N(d1)` and `N(d2)` to saturate
to 0/1 in double precision, so `gk_price(sigma)` equals the `sigma -> inf`
bound for every sigma across a whole plateau, not just at the true root. Any
point on that plateau would reprice the quote equally well, so returning one
would be an arbitrary answer (a 4.0 vol when the true value is 3.0, say) with
no signal to the caller. The FX `implied_vol` instead raises when the price is
within `tau = 4 eps * upper` of the bound, and
`test_long_dated_deep_itm_high_vol_flat_plateau_raises` in
`tests/test_garman_kohlhagen.py` pins that behaviour.

</details>

**Q16. The old `tol`, `max_iter`, `sigma_lo` and `sigma_hi` arguments still exist. What do they do?**

<sub>Tags: `L2`, `implied-vol`</sub>

<details><summary>Answer</summary>

Nothing: they are deprecated and ignored, and passing any of them raises a `DeprecationWarning` (`test_old_tuning_keywords_warn`). The solver has no tuning parameters because its accuracy is set by the information in the quote (Q13), not by a caller-chosen tolerance. Keeping the arguments for one release lets existing callers run unchanged while they are told to stop passing them.

</details>

**Q17. Why is the same implied-vol algorithm shared by all six options engines rather than each engine choosing its own?**

<sub>Tags: `L2`, `implied-vol`, `code-reading`</sub>

<details><summary>Answer</summary>

Because the C++ and Rust engines mirror the Python reference
(`CONVENTIONS.md`: "Engines mirror the Python reference implementations and
are cross-validated against them"), and one algorithm is easier to validate
than six. The equity and FX Python engines share `qp_core/iv_core.py`, and the
C++ and Rust engines implement the same constants and branch thresholds. The
risk of sharing is that a design weakness would be reproduced everywhere,
which is why the goldens are not generated from the shared code: the
implied-vol golden cases invert an mpmath-computed quote whose true vol is
known by construction (Set 15, Q3), and the differential tests compare Python,
Rust and C++ on seeded random inputs through JSON-lines shims
(`tests/test_differential.py`).

</details>

**Q18. Confirm: do the C++ and Rust equity engines document the same algorithm and accuracy as the Python reference?**

<sub>Tags: `L2`, `implied-vol`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Yes. `cpp/equity-options-engine/include/eqopt/implied_vol.hpp` describes the
same out-of-the-money mapping by parity, the shared core in
`eqopt/iv_core.hpp` running safeguarded Halley iteration on transformed
objectives with a relative stopping rule, the same accuracy statement
(`max(eps * price / vega, 1e-12 * sigma)` over the same domain, at most 10
evaluations, typically 2-4), and the same deep in-the-money example (S=100,
K=90, T=0.02, sigma=10%). `rust/equity-options-engine/src/implied_vol.rs`
mirrors it.

</details>

**Q19. How is `b(x, s)` computed near the money without losing digits?**

<sub>Tags: `L2`, `implied-vol`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Near the money `N(d1)` and `N(d2)` are almost equal, so their difference
cancels. `normalised_otm_call` rewrites `b` as `cosh(x/2) A + sinh(x/2) (N(d1) + N(d2))`
with `A = N(d1) - N(d2)`, and computes `A` three ways depending on the region:
a Hermite-polynomial series when `|x| < 0.1` and `s < 0.2`, a difference of
`erf` values when `|d1|` and `|d2|` are below 1, and a plain difference of
`erfc`-based CDFs otherwise. The upper-branch complement `b_max - b` is
computed directly as `e^{x/2} N(-d1) + e^{-x/2} N(d2)`, a sum with no
cancellation.

Source: `python/core/src/qp_core/iv_core.py` (the series branch is `if -x < 0.1 and s < 0.2`).

</details>

**Q20. Why is the log-moneyness computed as `log1p((F - K) / K)` rather than `log(F / K)`?**

<sub>Tags: `L2`, `implied-vol`, `code-reading`</sub>

<details><summary>Answer</summary>

Near the money `F` and `K` are close, so `F - K` is computed exactly
(Sterbenz's lemma), and `log1p` of a small argument keeps full relative
accuracy. `log(F / K)` would first round the ratio to about one ulp of 1,
losing most of the digits of a small `x` before the logarithm is even taken.
Since every later step depends on `x`, this one line protects the
near-the-money accuracy of the whole solver.

</details>

**Q21. Why does the normal CDF in these engines use `erfc` rather than `erf`?**

<sub>Tags: `L2`, `implied-vol`, `numeric`</sub>

<details><summary>Answer</summary>

`N(x) = 0.5 * erfc(-x / sqrt(2))` keeps full relative accuracy in the far left tail, where `N(x)` is tiny. The algebraically equal `0.5 * (1 + erf(x / sqrt(2)))` cancels to zero there, because `erf` is close to -1 and `1 + erf` loses all its digits. Out-of-the-money options and the lower branch of the implied-vol solver live exactly in that tail, so the choice changes answers, not just style.

Source: `python/core/src/qp_core/normal.py` (`norm_cdf`).

</details>

**Q22. How are the implied-vol golden vectors built so that they test the solver rather than echo it?**

<sub>Tags: `L2`, `implied-vol`, `code-reading`</sub>

<details><summary>Answer</summary>

`tests/golden/generate_golden.py` prices each case in mpmath at 50 significant
digits, at a vol chosen in advance, and stores that quote with the known vol
as the expected output. Every engine then inverts the quote and is checked
against the known vol, with a tolerance per record from the file's
`tolerance_rules` (for implied vol, derived from the conditioning `eps * price / vega`).
A solver bug that also existed in the generator could not hide, because the
generator never runs the solver.

</details>

**Q23. What kind of test would catch a solver that "converges" to the wrong precision in a hard corner?**

<sub>Tags: `L2`, `implied-vol`</sub>

<details><summary>Answer</summary>

A test against an independent high-precision oracle over a grid that covers
the corners, with a tolerance derived from the problem's conditioning, rather
than a few round trips at comfortable inputs with one fixed tolerance.
`test_iv_round_trip_against_mpmath` does this across log-moneyness, expiry,
vol and call/put. A fixed-tolerance spot check only tells you the solver got *close
enough this time*; it cannot distinguish a solver that converges in sigma from
one that stops early on a proxy residual, and a loose tolerance in a flat-vega
corner would pass either.

</details>

**Q24. Does this portfolio have an existing test pattern that checks a rate of convergence rather than one snapshot?**

<sub>Tags: `L2`, `implied-vol`, `code-reading`</sub>

<details><summary>Answer</summary>

Yes, a fitted-exponent convergence-rate test for the CRR binomial tree's step
count: `test_convergence_rate_fitted_exponent` in
`python/fx/01-options-pricing/tests/test_binomial.py` and
`convergence_rate_fitted_exponent_matches_theoretical_order_one` in
`rust/equity-options-engine/tests/binomial.rs` fit an empirical convergence
order to the error-vs-step-count curve and assert it matches the theoretical
O(1/n) rate, rather than just checking the error is small at one fixed step
count. The implied-vol tests take the related form of an evaluation budget
(Q25).

</details>

**Q25. How do the tests check the solver's speed, not just its answer?**

<sub>Tags: `L2`, `implied-vol`</sub>

<details><summary>Answer</summary>

`black_implied_vol` takes an optional `counter` argument, a one-element list
incremented once per pass of the iteration loop.
`test_iv_round_trip_against_mpmath` asserts `counter[0] <= 10` on every grid
case, and `test_root_at_the_inflection_point_converges_fast` asserts at most 4
on the case where the root sits at `s_c`. A change that kept the answers
correct but quietly fell back to bisection would fail these tests, so a
performance regression in the hard corners shows up as a red test rather than
a slow batch.

</details>

**Q26. Summarize, in one line, the general lesson about validating numerical solvers.**

<sub>Tags: `L2`, `implied-vol`</sub>

<details><summary>Answer</summary>

Measure convergence in the variable that has to be accurate (here sigma, not the price residual), state the accuracy the input can actually support, and test the solver against an independent oracle across the corners of its domain, so that a stopping rule that looks converged on a proxy cannot pass.

</details>


## Set 4: Volatility Modeling & Forecasting

**Q1. Black–Scholes assumes constant volatility. Why does this portfolio bother modeling volatility as time-varying at all?**

<sub>Tags: `L2`, `vol-forecasting`, `code-reading`</sub>

<details><summary>Answer</summary>

Because the constant-vol assumption is empirically false in a specific,
exploitable way: daily returns exhibit volatility clustering, a large move
today makes a large move tomorrow more likely, and calm periods cluster too.
`eq_vol/evaluation.py` builds this into its diagnostics directly:
`arch_lm_test` (Engle's ARCH-LM) applied to raw returns is described as "the
standard pre-test for whether a GARCH-type model is needed at all," and
`ljung_box_squared` checks whether squared standardised residuals are still
serially correlated after fitting. If returns were iid with constant variance,
both tests would come back clean on the raw series and there would be nothing
for a GARCH model to explain.

</details>

**Q2. What does "volatility clustering" mean mechanically, in terms of what a GARCH-type recursion does with it?**

<sub>Tags: `L1`, `vol-forecasting`, `code-reading`</sub>

<details><summary>Answer</summary>

It means today's shock feeds directly into tomorrow's conditional variance. In
GARCH(1,1), `sigma2_t = omega + alpha*r_{t-1}^2 + beta*sigma2_{t-1}`
(`eq_vol/garch.py`): a large `r_{t-1}^2` pushes `sigma2_t` up immediately via
the `alpha` term, and `beta*sigma2_{t-1}` carries the elevated level forward.
EGARCH and GJR-GARCH add that the *sign* of the shock matters too (leverage
effect), but all three models encode the same clustering mechanism, variance
is autocorrelated even though returns themselves are roughly uncorrelated.

</details>

**Q3. Exactly which volatility models does this project implement, not in theory, but in the actual `src/` modules?**

<sub>Tags: `L2`, `vol-forecasting`, `code-reading`</sub>

<details><summary>Answer</summary>

`python/equity/02-volatility-modeling/src/eq_vol/` has six: range/realized
historical estimators (`historical.py`, close-to-close, Parkinson,
Garman–Klass, Rogers–Satchell), RiskMetrics EWMA (`ewma.py`), GARCH(1,1)
(`garch.py`), EGARCH(1,1) (`egarch.py`), GJR-GARCH(1,1) (`gjr.py`), and the
HAR-RV regression (`har_rv.py`), tied together by `forecasting.py` and scored
by `evaluation.py`. The FX sibling (`fx_vol/`) has the same six plus a GARCH-X
variant with an exogenous event-dummy term (`omega + gamma_x'x_t + alpha r^2 + beta sigma2`)
and `vol_premium.py` for comparing model/realized vol against implied.

</details>

**Q4. Why GARCH-family models rather than a full stochastic-volatility model like Heston?**

<sub>Tags: `L2`, `vol-forecasting`</sub>

<details><summary>Answer</summary>

METHODOLOGY.md's "Alternative A" is explicit: SV models are the theoretically
"right" description (variance as its own latent process) but the likelihood
requires filtering, particle filters, MCMC, or Kalman-type quasi-ML, an order
of magnitude more machinery with harder diagnostics and governance. For
one-step-ahead vol forecasting from daily returns, it cites Hansen–Lunde
(2005), "Does anything beat a GARCH(1,1)?", which compares 330 ARCH-type
models and finds little forecast gain over GARCH(1,1) for exchange rates. That
study does not include stochastic-volatility models, so it supports the choice
of GARCH(1,1) within its family, not a verdict against SV. GARCH instead gives
an exact, closed-form likelihood, auditable to machine precision, which the
project exploits directly in its `arch`-package cross-validation.

</details>

**Q5. Why not realized-volatility / HAR models (Corsi 2009), which the docs admit often beat daily GARCH?**

<sub>Tags: `L2`, `vol-forecasting`, `code-reading`</sub>

<details><summary>Answer</summary>

METHODOLOGY.md's "Alternative B" concedes that HAR *would* be the preferred
production choice on liquid index futures if clean intraday data were
available, because realized variance typically beats daily GARCH out of
sample. The blocker is data budget: HAR needs a reliable tick/5-minute
pipeline with microstructure-noise corrections, doesn't produce a full
conditional distribution without more assumptions, and degrades to nothing on
names without intraday history. This project's scope is daily OHLC-and-close
data, so the range estimators in `historical.py` are described as "the poor
man's realized vol appropriate to that data budget", and `forecasting.py`'s
harness is built to accept an RV proxy unchanged if one becomes available.

</details>

**Q6. Why not just use option-implied volatility as the forecast?**

<sub>Tags: `L2`, `vol-forecasting`, `code-reading`</sub>

<details><summary>Answer</summary>

"Alternative C" in METHODOLOGY.md: implied vol is forward-looking and hard to
beat for direction where liquid options exist, but it embeds a time-varying
variance risk premium, it is a biased forecast by construction, is unavailable
for most single names and custom baskets, and can't be used to mark the very
options it comes from without circularity. The project's stance is that desks
use implied and model vol *together*; this package supplies the model leg, and
the FX side goes further with `fx_vol/vol_premium.py`, which explicitly
computes `VRP = IV - RV` as a tradeable signal rather than treating implied
vol as ground truth.

</details>

**Q7. What is the actual difference between realized volatility and implied volatility, and why do they diverge?**

<sub>Tags: `L1`, `vol-forecasting`, `code-reading`</sub>

<details><summary>Answer</summary>

Realized volatility is computed backward-looking from historical returns (or
the historical range), `eq_vol.historical.realized_vol` and friends. Implied
volatility is backed out of observed option prices via inverting
Black–Scholes/Garman–Kohlhagen (Set 2's territory) and is inherently
forward-looking. They diverge because implied vol prices in a variance risk
premium, compensation option sellers demand for jump/crash risk, so implied
systematically runs above subsequently realized vol on average;
`fx_vol/vol_premium.py`'s `vol_risk_premium` function quantifies exactly this
gap, and DESK_GUIDE.md notes a *negative* premium (realized exceeding implied)
as a crisis signal to stand down a short-vol program.

</details>

**Q8. What does EWMA actually buy you, and what is its key structural limitation?**

<sub>Tags: `L1`, `vol-forecasting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`eq_vol/ewma.py`: `sigma2_t = lambda*sigma2_{t-1} + (1-lambda)*r_{t-1}^2` with
`lambda = 0.94` (11.2-day half-life). It's IGARCH(1,1) with `omega = 0`, so
persistence `alpha+beta` is exactly 1, no parameters to estimate and it's
never badly mis-calibrated, but there is no long-run level to revert to, so
the k-step forecast is flat at the 1-step value (`ewma_forecast`'s term
structure is structurally flat). DESK_GUIDE.md is blunt about the consequence:
this flat structure "cannot price calendar effects at all," which is exactly
why a GARCH-family model is layered on top of it rather than used alone.

</details>

**Q9. What does GARCH(1,1) add over EWMA, and what does its long-run/unconditional variance parameter mean?**

<sub>Tags: `L1`, `vol-forecasting`, `code-reading`</sub>

<details><summary>Answer</summary>

GARCH(1,1) (`eq_vol/garch.py`) adds an intercept `omega > 0` and requires
`alpha + beta < 1`, which gives a genuine unconditional variance `omega / (1 - alpha - beta)`,
the level volatility reverts to over time, instead of EWMA's flat-forever
forecast. The k-step forecast decays geometrically toward that level at rate
`alpha+beta` (`forecast_garch` in `forecasting.py`), which is why GJR after a
sell-off, with current variance above the long-run level, produces a
downward-sloping (inverted) term structure that decays from an elevated short
end toward the long-run vol, the shape listed markets show after a shock
(DESK_GUIDE.md), something EWMA structurally cannot do.

</details>

**Q10. Mean reversion in volatility: what does it mean physically, and how is "persistence" and "half-life" defined in this codebase?**

<sub>Tags: `L2`, `vol-forecasting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Persistence is `alpha + beta` for GARCH, `alpha + gamma/2 + beta` for GJR
(`gjr_persistence` in `gjr.py`), it's the rate at which a variance shock
decays back toward the unconditional level. `eq_vol/garch.py`'s docstring
gives the half-life of a shock as `ln(0.5) / ln(alpha+beta)`; DESK_GUIDE.md
reports fitted half-lives of roughly 14–19 days on realistic parameters and
uses this (`extra["halflife_days"]`) to tell a risk manager how long an
elevated-vol episode, and an associated de-risking flag, is expected to
persist.

</details>

**Q11. What does EGARCH add that plain GARCH can't do, and what's the cost?**

<sub>Tags: `L1`, `vol-forecasting`, `code-reading`</sub>

<details><summary>Answer</summary>

EGARCH(1,1) (`eq_vol/egarch.py`) models the *log* of variance: `ln sigma2_t = omega + beta*ln sigma2_{t-1} + alpha(|z_{t-1}| - E|z|) + gamma*z_{t-1}`.
Because the recursion lives in log-space, `sigma2` is positive for *any* real
parameter values (only `|beta| < 1` is needed for stationarity), and `gamma < 0`
captures the leverage effect, negative shocks raising vol more than positive
ones of the same size. The cost is that multi-step forecasts have no clean
closed form (they need `E[exp(alpha|z| + gamma*z)]` across steps), so
`forecast_egarch` in `forecasting.py` falls back to seeded Monte Carlo
simulation rather than an analytic recursion.

</details>

**Q12. GJR-GARCH also captures leverage: how does its mechanism differ from EGARCH's, and why does the sign convention matter?**

<sub>Tags: `L2`, `vol-forecasting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

GJR-GARCH (`eq_vol/gjr.py`): `sigma2_t = omega + (alpha + gamma*1[r_{t-1}<0]) r_{t-1}^2 + beta*sigma2_{t-1}`,
it switches on an *extra* loading `gamma` only when the prior return was
negative, so `gamma > 0` is the leverage signal (opposite sign convention from
EGARCH's `gamma < 0`). METHODOLOGY.md flags that both conventions are
explicitly stated and unit-tested so nobody confuses them; VALIDATION.md's
simulated recovery shows the sign check works both ways, GJR's `gamma` t-stat
exceeds +3 on leveraged data and EGARCH's is below -3, while both sit within
±0.02 of zero on symmetric data.

</details>

**Q13. How does the historical/realized estimator suite differ from just squaring daily returns, and why does it matter?**

<sub>Tags: `L2`, `vol-forecasting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`eq_vol/historical.py` implements four range-based one-day variance estimators
beyond plain close-to-close `r_t^2`: Parkinson, Garman–Klass, Rogers–Satchell
(drift-robust) and Yang–Zhang (`yang_zhang_var`, which also handles the
overnight gap). "Efficiency" is the variance of `r_t^2` divided by the
variance of the estimator, for driftless Brownian motion monitored
continuously, so 5x means the same precision from a fifth of the days. The
figures and where each comes from:

| estimator | efficiency | source |
|---|---:|---|
| Parkinson (1980) | 4.91 | Exact: with range `R`, `E[R^2] = 4 ln2 σ²` and `E[R^4] = 9 ζ(3) σ⁴`, so `Var(R²/(4 ln 2)) = (9ζ(3)/(16 ln²2) − 1) σ⁴ = 0.407 σ⁴`, against `Var(r²) = 2σ⁴`; 2/0.407 = 4.91. |
| Garman–Klass (1980) | 7.4 | Garman and Klass (1980), *Journal of Business* 53(1), their best estimator. |
| Rogers–Satchell (1991) | about 6 | Simulation: 20,000 driftless days of 2,000 steps each give 6.0 (and 5.0 and 7.6 for the other two, discrete monitoring pulling the range in a little). |
| Yang–Zhang (2000) | depends on the open-to-close share | Yang and Zhang (2000), *Journal of Business* 73(3); it is the minimum-variance mix of overnight, open-to-close and Rogers–Satchell terms. |

The same figures appear in the `historical.py` docstrings and in the equity 02
METHODOLOGY table. METHODOLOGY.md also notes a 21-day Parkinson estimate is
roughly as precise as a ~100-day close-to-close one. The price is stronger
assumptions (continuous monitoring, no overnight gaps, no microstructure
noise); discrete trading biases the observed range down and bid–ask bounce
biases it up, both demonstrated in `tests/test_historical.py`.

</details>

**Q14. How is a GARCH-type forecast actually validated in this codebase: what's the real backtest?**

<sub>Tags: `L2`, `vol-forecasting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`examples/run_pipeline.py` runs an out-of-sample forecast race: true model GJR
with leverage (persistence 0.97), 3,000 training days, 500 out-of-sample
1-step forecasts, parameters refit every 25 days, scored with QLIKE against
squared returns via `forecast_race_table` in `evaluation.py`. It isn't graded
against a raw error metric alone, every model is also compared to a naive
benchmark (EWMA(0.94) and a rolling 21-day historical estimate) using the
Diebold–Mariano test with Harvey small-sample correction, computed by
`diebold_mariano`.

</details>

**Q15. What did that out-of-sample race actually find (VALIDATION.md §4), and what does it mean that GJR beat GARCH?**

<sub>Tags: `L2`, `vol-forecasting`, `numeric`</sub>

<details><summary>Answer</summary>

Ranking by QLIKE: GJR (-7.6027) > EGARCH (-7.5785) > GARCH (-7.5638) >
EWMA(0.94) (-7.4996) > rolling 21-day historical (-7.4963), against an oracle
QLIKE of -7.6067 (the true conditional variance of the data-generating
process). GJR's DM stat vs GARCH is -2.91 (p=0.004), a statistically
significant win, and it captures about 91% of the achievable QLIKE gap between
GARCH and the oracle ((7.6027 − 7.5638) / (7.6067 − 7.5638)), which is
expected because the simulated truth had genuine leverage (gamma=0.12) that
only GJR and EGARCH can represent.

</details>

**Q16. Why is QLIKE the headline loss function here rather than plain MSE, and is that an arbitrary choice?**

<sub>Tags: `L2`, `vol-forecasting`, `code-reading`</sub>

<details><summary>Answer</summary>

No, it follows Patton (2011), cited directly in `evaluation.py`'s module
docstring: since true conditional variance is unobservable and forecasts must
be scored against a noisy proxy (squared returns), only a specific class of
losses is "robust," meaning the proxy-based ranking equals the ranking under
the true variance. QLIKE (`ln f + p/f`) and MSE are the two robust members,
but QLIKE is preferred as the headline metric because MSE's expectation is
dominated by proxy noise in high-variance episodes, making rankings noisier,
and QLIKE also penalizes under-prediction of variance more than
over-prediction, which VALIDATION.md and DESK_GUIDE.md both flag as the
economically correct asymmetry (under-forecasting vol is the expensive error).

</details>

**Q17. The project independently re-implements GARCH/EGARCH/GJR from scratch. How does it know the implementation is correct?**

<sub>Tags: `L2`, `vol-forecasting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`tests/test_arch_crosscheck.py` benchmarks against Kevin Sheppard's `arch`
package purely as an oracle. On 5,000 simulated GARCH observations,
alpha/beta/omega agree to `arch` to within 1e-6–4e-6, and, the strongest
statement in VALIDATION.md, evaluating this project's own log-likelihood at
`arch`'s fitted parameters agrees with `arch`'s own log-likelihood to 1.8e-12,
meaning the recursion and density are "line-for-line equivalent" and the
residual parameter differences are pure optimizer tolerance, not
implementation bugs.

</details>

**Q18. What exactly breaks in a volatility forecast during a genuine regime change, according to this project's own crisis test?**

<sub>Tags: `L2`, `vol-forecasting`, `numeric`</sub>

<details><summary>Answer</summary>

VALIDATION.md's failure mode F1, tested in
`test_edge_cases.py::TestCrisisRegimeJump`, simulates a COVID-March-2020-scale
jump (true vol 15% to 75% annualized for 60 days, then settling at 30%).
EWMA(0.94) adapts but is lagged by construction, it read 13.3% pre-break and
only reached 65.4% thirty days *into* the crisis, meaning 1-day 99% VaR
computed from it would have been breached repeatedly in the first week. GARCH
fitted across the break estimates persistence 0.983 (vs ~0.95 on break-free
data), the break masquerades as near-IGARCH persistence, dragging the
estimated long-run variance far above either true regime, so its forecast term
structure mean-reverts to a level that means nothing.

</details>

**Q19. Why does a structural break make GARCH's estimated persistence spike, and what's the practical tell?**

<sub>Tags: `L2`, `vol-forecasting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Because GARCH has no mechanism to distinguish "variance is near a unit root"
from "there was one level shift in the middle of my sample", both produce
highly autocorrelated squared returns, and the MLE fits persistence toward 1
either way. The practical tell in this codebase is `unconditional_variance()`,
the library refuses to compute it (raises `ValueError`) rather than return a
near-infinite or negative number once `alpha+beta` hits the 0.9999 transform
ceiling, and `METHODOLOGY.md` and `VALIDATION.md` both flag persistence pinned
at that ceiling as "a red flag surfaced by the parameter table, not hidden"
and "often symptom of F1" (the structural-break failure mode).

</details>

**Q20. What's the desk's actual mitigation once a regime break is confirmed, rather than just a documented limitation?**

<sub>Tags: `L1`, `vol-forecasting`, `numeric`</sub>

<details><summary>Answer</summary>

DESK_GUIDE.md's workflow step 4 is explicit: after a suspected regime break,
shorten the estimation window (`scheme="rolling"` in
`rolling_one_step_forecasts`) or refit from the break date, and watch the
EWMA-vs-GARCH gap, EWMA repricing much faster than the fitted GARCH is called
"the classic break signature." VALIDATION.md's crisis case study quantifies
the window trade-off directly: 30 days into the simulated crisis, a 10-day
rolling window reads 75.1% (correct) while a 250-day window reads 35.9%, less
than half the truth, so shortening the window trades noise for speed of
adaptation.

</details>

**Q21. Beyond regime breaks, what other failure modes does VALIDATION.md document, and are they just described or also enforced in code?**

<sub>Tags: `L2`, `vol-forecasting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Both. F2 (IGARCH boundary): as persistence approaches 1, unconditional
variance explodes, so `unconditional_variance()` raises instead of returning
garbage. F3 (fat tails): Gaussian QMLE stays consistent for alpha/beta even
under Student-t innovations, but Gaussian VaR quantiles understate risk by
about 8% when the truth is t(8) (the unit-variance t(8) 1% quantile is 2.508
against the normal 2.326; VALIDATION.md rounds this to 1.10x), mitigated by
`dist="t"`, with `nu` recovered within 0.2 of truth on 20k simulated
observations. F5 (short samples): fitters refuse fewer than 100 observations
with an informative error rather than returning a badly-identified fit. F6
(optimizer failure): the library never fails silently, a non-converging fit
raises `ConvergenceError` or is flagged `converged=False`, tested by forcing a
failure.

</details>

**Q22. How do these volatility forecasts actually get used downstream, per DESK_GUIDE.md, not hypothetically, concretely?**

<sub>Tags: `L2`, `vol-forecasting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Four consumers are named explicitly. Option pricing/marking:
`term_structure()`'s model vol is compared against implied vol, with `implied - forecast`
used as the carry signal for variance-risk-premium strategies, and the model
mark serves as a fallback for illiquid expiries. VaR/ES: 1-day 99% VaR is
computed as `sigma_{t+1|t} * q_{0.01}` from `rolling_one_step_forecasts`
combined with the fitted innovation quantile, using `dist="t"` because
Gaussian tails understate 1% VaR by about 8% under t(8) returns. Vol
targeting: a book scales exposure by `target_vol / forecast_vol`, which is
exactly why QLIKE (which penalizes under-forecasting more) is the right
evaluation metric, under-forecasting means being over-levered into a storm.
Risk limits: persistence and half-life (`extra["halflife_days"]`) tell a risk
manager how long a de-risking flag should stay on after a shock.

</details>

**Q23. Why does DESK_GUIDE.md insist on refitting parameters only every 5–25 days rather than every day, if markets move daily?**

<sub>Tags: `L2`, `vol-forecasting`, `numeric`</sub>

<details><summary>Answer</summary>

Because the variance *recursion* is updated every day using the current
parameters (`rolling_one_step_forecasts` does exactly this, "recursion daily,
refit sparse"), while the parameters themselves (`omega`, `alpha`, `beta`,
`gamma`) move slowly; DESK_GUIDE.md states daily refitting "adds noise and
mark instability without forecast gain," and the headline pipeline race itself
uses a 25-day refit cadence. The guide adds explicit alerting triggers on top
of that cadence: a parameter jump greater than 3 standard errors versus the
previous fit, persistence exceeding 0.99, `nu` below 5, or a
`ConvergenceError`.

</details>

**Q24. How does this project frame model governance: what stops a bad fit from silently entering marks?**

<sub>Tags: `L2`, `vol-forecasting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

DESK_GUIDE.md lays out a benchmark chain, never a single model in isolation:
rolling historical (sanity check) → EWMA(0.94) (regulator-recognised,
parameter-free) → GARCH (mean reversion) → GJR/EGARCH (asymmetry), with each
escalation required to be justified by a DM test on QLIKE out of sample,
exactly the committed pipeline in `run_pipeline.py`. Independently, the
from-scratch implementation is cross-validated against the `arch` package to
machine precision (the "re-implement and reconcile" procedure model-validation
teams use on front-office pricers), every fit result carries parameters,
standard errors, log-likelihood, AIC/BIC and a convergence flag
(`VolatilityFitResult`), and any code change that shifts a fitted parameter
beyond optimizer noise on the synthetic-truth regression tests fails CI (CI
runs every project's test suite on each pull request).

</details>

**Q25. What's actually different about the FX volatility project versus the equity one, not just "same models, different asset class"?**

<sub>Tags: `L1`, `vol-forecasting`, `numeric`</sub>

<details><summary>Answer</summary>

Two things are FX-specific and directly tested. First, quote direction is a
modelling choice: inverting BASE/QUOTE negates log returns, leaving volatility
invariant but *flipping the sign of asymmetry*, METHODOLOGY.md's pipeline
shows GJR finding gamma=0 on USDMXN but gamma=+0.123±0.040 on the inverted
MXNUSD, so the desk rule is to fit the quote direction actually traded and
sanity-check gamma's sign against the underlying economics. Second, leverage
itself is weak for G10 and strong for EM, because FX has no debt-equity
channel, the fitted EGARCH gamma is about 0.007 on a G10-style pair versus
about 0.046 on an EM-style pair, and `fx_vol` adds a GARCH-X variant that puts
known scheduled events (FOMC/ECB/BoJ dummies known at t-1) directly into the
variance equation, recovering the event-dummy coefficient within 3% of its
true simulated value.

</details>

**Q26. What does HAR-RV (`eq_vol.har_rv`) forecast, and what are its oracles?**

<sub>Tags: `L1`, `vol-forecasting`</sub>

<details><summary>Answer</summary>

HAR-RV (Corsi 2009) regresses next-day realised variance on daily, weekly and
monthly averages of past realised variance, a cheap way to get long memory.
`fit_har` is checked by recovering the coefficients exactly from a noiseless
recursion, by comparing the design matrix with a naive loop, by fitting
simulated HAR data within 4 HAC standard errors, and by checking that the
multi-step forecast converges to the unconditional mean.

</details>


## Set 5: Volatility Surface & Stochastic Vol

**Q1. Set 2 covers pricing and Greeks under a single implied volatility per option. Why isn't one number enough to describe a real options market?**

<sub>Tags: `L3`, `vol-surface`</sub>

<details><summary>Answer</summary>

Because the market does not quote one vol per underlying, it quotes a
different implied vol for every strike and every expiry. Plot implied vol
against strike at fixed expiry and you get the "smile" or "skew" (equities
skew negatively: downside puts trade at higher implied vol than upside calls,
reflecting crash risk and leverage effects); plot it against expiry at fixed
moneyness and you get the term structure. A single Black-Scholes vol can
reprice exactly one option, but a desk quotes and risk-manages hundreds of
strikes and expiries simultaneously off the same underlying, so it needs the
whole two-dimensional object, not one point on it.

</details>

**Q2. What, precisely, is a volatility surface?**

<sub>Tags: `L3`, `vol-surface`, `code-reading`</sub>

<details><summary>Answer</summary>

It's implied volatility expressed as a function of two coordinates spanning
strike/moneyness and expiry, `sigma_imp(K, T)` in the equity convention used
by `eq_surface.surface.VolSurface`, where `K` is absolute strike and `T` is
time to expiry in years. FX quotes the same object differently:
`fx_surface.surface`'s `vol(Δ, T)` is implied vol as a function of *delta* (a
probability-like moneyness coordinate) and tenor, because FX brokers quote in
delta space rather than strike space (see Q19). Both are the same mathematical
object, a 2-D vol function, expressed in the coordinate system each market
actually trades.

</details>

**Q3. How does this portfolio actually build the equity surface? Don't assume a textbook method, cite the real construction.**

<sub>Tags: `L3`, `vol-surface`, `code-reading`</sub>

<details><summary>Answer</summary>

`eq_surface.surface.VolSurface` fits one raw-SVI slice per expiry pillar
(`smile.fit_svi`), samples each slice's *total variance* `w(k) = svi_total_variance(k, p)`
on a common log-moneyness grid, and then interpolates **total variance
linearly in T** between pillars (`VolSurface.total_variance`), not vol.
Outside the pillar range it applies a documented extrapolation policy: flat
vol below the first pillar (`w` scales linearly to zero as `T → 0`) and a
slope-floored linear continuation of `w` beyond the last pillar. Querying
`vol(K, T)` converts the strike to forward log-moneyness `k = ln(K / F(T))`
first, then calls `vol_k`.

</details>

**Q4. Why total variance, and specifically why not just linearly interpolate implied vol itself across expiries?**

<sub>Tags: `L3`, `vol-surface`, `code-reading`</sub>

<details><summary>Answer</summary>

Absence of calendar-spread arbitrage is equivalent (at fixed forward
log-moneyness) to total variance `w(k, T) = sigma_imp(k,T)^2 * T` being
non-decreasing in `T`. Linear interpolation of a monotone quantity in `T`
stays monotone, so pillars that are individually calendar-free interpolate to
a calendar-free surface. The `surface.py` module docstring makes the
counterexample explicit: linear interpolation *of vol* has no such property, a
high-vol short pillar sitting next to a low-vol long pillar can interpolate to *decreasing*
total variance between the two, which is negative forward variance, i.e.
manufactured arbitrage that wasn't present in either individual slice.

</details>

**Q5. What is raw SVI, and what does each of its five parameters control?**

<sub>Tags: `L2`, `vol-surface`, `code-reading`</sub>

<details><summary>Answer</summary>

`smile.SVIParams` parameterises total implied variance in forward
log-moneyness `k = ln(K/F)` as `w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))`.
`a` sets the overall variance level, `b >= 0` sets the wing slope, `rho` in
`(-1, 1)` sets skew/asymmetry, `m` translates the smile's minimum
horizontally, and `sigma > 0` controls how rounded the vertex is. `fit_svi`
fits these five parameters per expiry and, by default (`arbitrage_free=True`),
imposes the raw-SVI domain, Lee's wing bound `b(1 + |rho|) <= 2` and
Durrleman's `g(k) >= 0` on a grid covering the quotes, so the fitted slice is
free of butterfly arbitrage by construction. Starting points come from the
Zeliade quasi-explicit split (the linear `(a, d, c)` problem solved in closed
form on an `(m, sigma)` grid), and the best `n_restarts` (default 8) seed an
SLSQP solve of the full constrained problem. At least 5 valid quotes per
expiry are required, since a 5-parameter model needs at least that many points
to be identified.

</details>

**Q6. The METHODOLOGY.md table compares SVI against a polynomial-in-delta baseline. What does that comparison actually show, and where does the baseline live in the code?**

<sub>Tags: `L3`, `vol-surface`, `numeric`</sub>

<details><summary>Answer</summary>

The naive baseline is `smile.fit_quadratic_delta`, implied vol modeled as a
quadratic in BS forward call delta, with no no-arbitrage machinery.
METHODOLOGY.md reports it fits equity smiles 10-100x worse than SVI (0.50-0.75
vol points versus 0.0006-0.04 for SVI on the same slices) and its odd-power
polynomial wings either explode or collapse outside the fitted range, whereas
SVI's wings are linear in total variance by construction, consistent with
Lee's moment bounds. The baseline exists specifically to make the "why SVI"
argument concrete rather than asserted.

</details>

**Q7. Why can Black-Scholes never produce a smile on its own, no matter how you calibrate it?**

<sub>Tags: `L3`, `vol-surface`</sub>

<details><summary>Answer</summary>

Black-Scholes assumes volatility is a single constant parameter, the same
number for every strike and every expiry on a given underlying. Inverting a
market price for "the" implied vol under that model necessarily returns one
number per (K, T) pair, there is no internal mechanism by which the model
could generate different vols at different strikes, because the model has
exactly one vol parameter to begin with. Any smile you observe when you invert
real market prices under Black-Scholes is therefore evidence the *model* is
wrong for those prices, not a feature Black-Scholes can reproduce; you either
interpolate the observed smile empirically (SVI) or move to a model whose
dynamics generate a smile on their own (Heston).

</details>

**Q8. What does Heston add to Black-Scholes that lets it generate a smile endogenously?**

<sub>Tags: `L2`, `vol-surface`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Heston makes variance itself a second stochastic process rather than a
constant: `dv = kappa(theta - v) dt + xi*sqrt(v) dW_v`, correlated with the
spot's Brownian motion `dW_S` via `rho`. `heston.py`'s module docstring gives
the full risk-neutral SDE pair. That correlation is what buys the smile "for
free": when `rho < 0` (the typical equity fit, the calibration recovery table
in VALIDATION.md finds `rho = -0.65`), falling spot and rising variance move
together, which fattens the downside and produces negative skew without any
per-strike parameter, the skew is a structural consequence of `rho != 0`, not
something fitted strike-by-strike the way SVI's `rho` parameter is fitted per
expiry. The model also fixes how the smile moves with spot: Heston is
homogeneous of degree one in `(S, K)`, so at fixed `v0` its smile is
sticky-moneyness and its spot delta is the sticky-moneyness delta, above the
sticky-strike (plain Black-Scholes) delta for a call under negative skew
(`eq_surface/greeks.py`, `smile_adjusted_delta` reports `delta_sticky_strike`
and `delta_sticky_moneyness` side by side).

</details>

**Q9. Heston reprices "today's vanillas" only approximately, per METHODOLOGY.md's own comparison table. What's the actual trade-off against local vol (Dupire), and why does this portfolio still prefer Heston?**

<sub>Tags: `L3`, `vol-surface`</sub>

<details><summary>Answer</summary>

Local vol reprices today's vanillas *exactly* by construction, but Hagan et
al. (2002) show its smile dynamics are wrong-way: when spot moves, the
local-vol smile moves in the opposite direction to the one markets show, which
mis-hedges anything sensitive to how the smile evolves. A related, separate
weakness is that local-vol forward smiles flatten over time (Gatheral, 2006,
ch. 4), which mis-prices forward-starts, cliquets and autocalls. Heston gives
up exact repricing (METHODOLOGY.md documents a 2-3 vol-point short-dated
residual) in exchange for forward smile dynamics that are qualitatively
realistic, the smile moves with spot, vol clusters and mean-reverts, which is
what actually matters for path- and dynamics-dependent exotics. The FX
project's METHODOLOGY.md makes the identical trade-off explicit for Dupire vs.
Heston under Garman-Kohlhagen, adding that FX pillar data (5 strikes x 6
expiries) is in any case too sparse to extract a stable local-vol surface.

</details>

**Q10. What no-arbitrage conditions must a real vol surface satisfy, and does this portfolio's documentation show it actually checks for them?**

<sub>Tags: `L3`, `vol-surface`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Two classic conditions: **butterfly (strike) arbitrage**, the risk-neutral
density must be non-negative at every strike, and **calendar-spread
arbitrage**, total variance must be non-decreasing in expiry at fixed
moneyness/delta. Both are checked, not just described. Butterfly arbitrage is
diagnosed by the Durrleman condition `smile.durrleman_g` /
`smile.check_butterfly`, and calendar arbitrage by `surface.check_calendar`,
which flags any grid point where total variance decreases across pillars.
VALIDATION.md §6 (equity) turns both into executable property tests
(`tests/test_arbitrage_and_wings.py`), e.g.
`test_butterfly_checker_fires_on_arbitrage_violating_quotes` plants a V-shaped
smile and asserts `min_g < 0` is actually detected, and
`test_calendar_checker_detects_decreasing_total_variance` does the equivalent
for calendar. The FX project's F7 failure mode documents the same two
planted-violation checks (SVI `b=0.35, rho=-0.9` for butterfly; a 1m ATM
bumped to 20% for calendar).

</details>

**Q11. If a slice fails the Durrleman check, does the surface silently fix it, or does something else happen?**

<sub>Tags: `L3`, `vol-surface`, `code-reading`</sub>

<details><summary>Answer</summary>

It depends on the fit mode. With the default `fit_svi(..., arbitrage_free=True)`
the Durrleman condition is a constraint of the fit, so a returned slice cannot
fail it; if no feasible solution exists, `fit_svi` raises `RuntimeError` and
points at the input quotes. With `arbitrage_free=False` (the old unconstrained
fit) the slice is diagnosed, not repaired: `fit_svi` emits a warning when
`check_butterfly` finds `min_g < 0`. Calendar violations across slices are
handled the same way: `VolSurface.__init__` warns on any violation from
`check_calendar`, with the monotone running-max fix applied only when the
caller opts in via `enforce_calendar=True`. DESK_GUIDE.md's daily workflow
makes the human step explicit: "a flagged slice is a quote problem or an
arbitrage, either way a human looks before the mark goes out." The checkers
are themselves tested against planted violations so a silent regression in a
checker would fail the suite rather than let a bad surface publish unnoticed.

</details>

**Q12. Why does the calendar-arbitrage running-max "fix" come with a warning that it's dangerous to blindly apply, per the equity DESK_GUIDE.md earnings scenario?**

<sub>Tags: `L3`, `vol-surface`</sub>

<details><summary>Answer</summary>

Real event variance (e.g. a single-name earnings print, or a central-bank date
in FX) concentrates a jump in total variance at one specific expiry pillar.
That jump is not itself a calendar-arbitrage violation, total variance still
rises with expiry, and the post-event annualised vol sitting far below the
event pillar's is consistent with that. What it breaks is the smooth term
structure the surface assumes in ACT/365F time, so the calendar check can fire
around it (for example when one pillar's quote predates the event and its
neighbour's does not). Running `enforce_calendar=True` would smear that event
variance across the term structure via the monotone running-max adjustment,
destroying real information. The documented handling instead strips the event
component (`w = w_diffusive + w_event * 1{T >= t_event}`), marks the diffusive
surface, and re-adds the event, leaving `enforce_calendar` off and treating
the calendar flag as the correct alarm, not a bug to suppress.

</details>

**Q13. Heston's characteristic function has a well-known numerical trap. What is it, and how does this code avoid it?**

<sub>Tags: `L3`, `vol-surface`, `code-reading`</sub>

<details><summary>Answer</summary>

`heston.py`'s module docstring explains: the original 1993 Heston CF
formulation contains `ln((1 - g1*e^{dT})/(1 - g1))` with a *growing*
exponential `e^{dT}`, so as `u` or `T` grows the log's argument spirals across
the negative real axis and the principal-branch complex logarithm jumps by
`2*pi*i`, silently wrong, discontinuous prices unless the branch is tracked by
hand. `heston_cf` instead uses the algebraically equivalent "little Heston
trap" form (Albrecher et al. 2007) built from the conjugate root `g2 = (b-d)/(b+d)`,
whose exponential `e^{-dT}` decays to zero instead of growing, so the log
argument never winds around the origin. `test_little_trap_cf_continuous_in_u`
checks this by scanning a dense `u`-grid at `T=10, xi=1` and asserting no
branch jumps.

</details>

**Q14. What is the Feller condition, and why does this codebase warn on violation instead of rejecting the parameters?**

<sub>Tags: `L2`, `vol-surface`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Feller is `2*kappa*theta >= xi^2` (equivalently `feller_condition` returning a
ratio `>= 1`); it's the condition under which the CIR variance process stays
strictly positive. Both projects' `heston.py` warn rather than raise
(`FellerWarning`) because market calibrations routinely violate it,
short-dated equity and FX skew both demand high `xi` relative to
`kappa*theta`, and the model stays mathematically well-defined even so:
variance touches zero and reflects, and the characteristic function remains
valid. The equity fitted surface sits at Feller ratio 0.80; both FX presets
sit near 0.46. What Feller violation *does* change is which Monte Carlo scheme
is safe to use (Q15).

</details>

**Q15. What breaks in Monte Carlo simulation when Feller is violated, and how does this portfolio handle it?**

<sub>Tags: `L3`, `vol-surface`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Full-truncation Euler carries an O(dt) truncation bias from clipping negative
variance draws to zero, and that bias explodes exactly when Feller is
violated: equity VALIDATION.md's "extreme set" (`xi=1`, Feller ratio 0.08)
shows Euler still running **+20 standard errors** off the Fourier reference at
64 steps/year, and **+135 SE** at 8 steps/year, nearly 50% of the ATM price.
Andersen's QE scheme (moment-matched CIR transition, implemented as the
`scheme="qe"` branch in both `heston_mc.py` files) stays within Monte Carlo
noise at 8 steps/year on the same parameter set. Both desk guides make this a
hard scheme-control rule: never use Euler for pricing under a Feller-violating
fit, and both `heston_mc.py` implementations default `scheme="qe"`.

</details>

**Q16. What are antithetic variates, and why bother with them at all in a Monte Carlo pricer?**

<sub>Tags: `L2`, `vol-surface`</sub>

<details><summary>Answer</summary>

Antithetic variates are a variance-reduction technique: instead of drawing `n`
independent standard normals, you draw `n/2` and pair each one `z` with its
negation `-z`, then average the payoff over each pair. For a payoff that is
monotone in `z`, the values at `z` and `-z` are negatively correlated, so
their errors partially cancel and the pairwise-averaged estimator has lower
variance than `n` fully independent draws for the same simulation cost
(Glasserman, 2003, §4.2). The guarantee is for monotone payoffs only: for a
symmetric payoff such as a straddle the pair is positively correlated and
antithetics can do worse than plain sampling.
`fx_surface.heston_mc.simulate_terminal` implements this via `antithetic: bool = True`:
for both Euler and QE it draws `n_base` normals and mirrors them
(`np.concatenate([z1, -z1])`), and `mc_price` averages payoffs pairwise before
computing the sample standard error, so the reported SE is not artificially
inflated by the negative correlation the pairing itself introduces.

</details>

**Q17. Why does `simulate_terminal` refuse an odd `n_paths` when `antithetic=True`? What would go wrong without the guard?**

<sub>Tags: `L3`, `vol-surface`</sub>

<details><summary>Answer</summary>

An odd `n_paths` would break the antithetic pairing invariant the whole
standard-error calculation depends on. With `n_paths` odd, `n_base = (n_paths + 1) // 2`
antithetic normals would be generated, mirrored to `[z, -z]`, and truncated
back down to `n_paths`, which drops the last element of the mirrored half, so
the last "base" draw loses its antithetic partner while every other draw still
has one. `mc_price`'s pairwise averaging (`payoff[:half] + payoff[half:]`)
assumes a clean 50/50 split of matched antithetic pairs; with an unpaired
leftover path in the mix, that assumption is violated and the averaging
misaligns the pairs, so the reported standard error no longer describes the
estimator.

</details>

**Q18. Why is that failure mode specifically dangerous, arguably worse than a bug that produces a visibly wrong price?**

<sub>Tags: `L3`, `vol-surface`</sub>

<details><summary>Answer</summary>

Because the *price* estimate itself stays approximately correct (antithetic
pairing barely nudges the mean); what goes wrong is the reported *standard
error*: with the halves misaligned, the averaged "pairs" are no longer
`z`/`-z` partners, so the SE no longer describes the estimator, and its error
can go either way. A visibly wrong price gets caught immediately, because
someone compares it to the Fourier reference and it's obviously off. A
confidently-wrong error bar does not get caught the same way: a risk or
model-validation process that trusts "3 SE agreement" as its acceptance bar
(this portfolio's own documented tolerance, per CONVENTIONS.md) would be
judging the result against an error bar that does not describe it: a
wrong-but-tight SE makes a marginal or biased result look like it passed
comfortably, and a wrong-but-loose one lets a real bias through. An error bar
that lies about its own size is more dangerous than a wrong point estimate
precisely because it undermines the mechanism you'd normally use to catch the
point estimate being wrong.

</details>

**Q19. What is the guard, and where does it live in the code?**

<sub>Tags: `L2`, `vol-surface`, `code-reading`</sub>

<details><summary>Answer</summary>

`simulate_terminal` rejects odd `n_paths` outright under `antithetic=True`,
raising `ValueError(f"n_paths must be even when antithetic=True (odd n_paths breaks pairing and silently understates the standard error), got {n_paths}")`
before any simulation happens (`fx_surface/heston_mc.py`), and
`tests/test_heston_mc.py::test_odd_n_paths_with_antithetic_raises` pins it
(odd counts without antithetics still run). The guard is a validation check,
not an attempt to silently handle the odd case some other way, consistent with
this portfolio's stated preference (see equity VALIDATION.md's short-end-floor
writeup and FX VALIDATION.md's F8 finding) for raising loudly over degrading
silently. `mc_price`'s docstring also states the invariant it depends on
directly: "`simulate_terminal` enforces `n_paths` even under
`antithetic=True`, so this split is always a clean pairing of `[z, -z]`
halves."

</details>

**Q20. How exactly does `mc_price` use the antithetic pairing to compute its standard error, once `n_paths` is guaranteed even?**

<sub>Tags: `L3`, `vol-surface`, `numeric`</sub>

<details><summary>Answer</summary>

After `simulate_terminal` returns `n_paths` terminal spots, `mc_price`
computes the discounted payoff for all of them, then, only when
`antithetic=True`, splits the payoff vector exactly in half and averages each
`z`/`-z` pair: `payoff = 0.5 * (payoff[:half] + payoff[half:])`. The standard
error is then `np.std(payoff, ddof=1) / sqrt(len(payoff))` computed on that *halved,
pre-averaged* array, not on the raw `n_paths` payoffs. This is the step that
actually removes the antithetic correlation from the SE estimate: treating the
`n_paths` raw draws as independent would overstate the SE because paired draws
are negatively correlated by construction (for a monotone payoff), so
averaging within each pair first and computing dispersion across pairs is what
makes "sample std / sqrt(n)" a valid SE formula again.

Source: `python/equity/01-options-pricing/src/eq_options/monte_carlo.py`.

</details>

**Q21. Does the equity project's Monte Carlo engine have the same antithetic-pairing exposure?**

<sub>Tags: `L3`, `vol-surface`, `code-reading`</sub>

<details><summary>Answer</summary>

It has the same option and the same guard. `eq_surface/heston_mc.py`'s
`simulate_heston_terminal` takes `antithetic: bool = False`; when it is on,
every Gaussian (and every variance draw) is mirrored so that path `i + n_paths/2`
is the antithetic partner of path `i`, and `mc_price` averages payoffs within
each pair before computing the standard error. An odd `n_paths` with
`antithetic=True` raises `ValueError`, tested by
`tests/test_heston_mc.py::test_odd_n_paths_with_antithetic_raises`. The
difference from FX is the default: the equity engine draws independent normals
unless the caller asks for antithetics, while the FX engine defaults to
`antithetic=True`.

</details>

**Q22. FX vol surfaces are quoted differently from equity ones. What does "ATM/RR/BF by tenor" actually mean, and why does the FX market quote this way instead of strike/expiry directly?**

<sub>Tags: `L3`, `vol-surface`, `numeric`</sub>

<details><summary>Answer</summary>

Per `fx_surface`'s METHODOLOGY.md, brokers quote five numbers per expiry, ATM
(the delta-neutral straddle strike), 25-delta risk reversal `RR25 = sigma(25∆call) - sigma(25∆put)`
(the skew, tradable as a package), and 25-delta butterfly `BF25 = 0.5*(sigma(25∆call) + sigma(25∆put)) - sigma_ATM`
(the convexity), each repeated at 10-delta. This is quoted in delta space
rather than strike space because G10 FX spot moves about 0.5-0.7% a day,
making any strike-based quote sheet stale within hours and incomparable across
pairs trading at wildly different absolute levels (EURUSD ≈ 1.10 vs USDJPY ≈
150); delta is a moneyness measured in probability units, so it's
self-normalising across spot level, pair, and vol regime, exactly why the
market settled on it.

</details>

**Q23. Given quotes in ATM/RR/BF form, how does the FX pipeline turn that into strikes it can actually build a surface from?**

<sub>Tags: `L3`, `vol-surface`, `code-reading`</sub>

<details><summary>Answer</summary>

`smile_from_quotes.py` first maps the five ATM/RR/BF numbers to the two wing
vols by the exact linear relations `sigma_C = ATM + BF + RR/2`, `sigma_P = ATM + BF - RR/2`
(round-trip tested to 1e-14). Getting from a target delta to the strike behind
it is not always closed-form: under unadjusted delta conventions the strike
inverts directly, but under premium-adjusted conventions (used for USDJPY and
most USD-base pairs) the call delta is *non-monotone* in strike, it vanishes
at both `K→0` and `K→∞` with an interior maximum, so a given delta has two
candidate strikes, and `strike_from_delta_pa_candidates` exposes both while
the market convention (the higher, OTM, falling-branch strike) is selected and
tested explicitly.

</details>

**Q24. Why does the FX surface interpolate at fixed delta rather than fixed log-moneyness, unlike the equity surface?**

<sub>Tags: `L3`, `vol-surface`, `code-reading`</sub>

<details><summary>Answer</summary>

Because FX is a sticky-delta market: METHODOLOGY.md §4 explains that the
quoted objects (ATM/RR/BF) float with spot and vol, so a fixed log-moneyness
represents a very different number of deltas out-of-the-money at a 1-week
tenor than at a 1-year tenor, and interpolating in moneyness would badly
distort the short-dated wings. `fx_surface.surface.FXVolSurface` interpolates
total variance linearly in `T` at fixed delta. By default
(`delta_convention="native"`) each slice is read in the delta convention its
own quotes use (spot, forward, or premium-adjusted), so the surface returns
the broker quotes at the pillars and "the 25-delta call" between two pillars
means each pillar's 25-delta call in its own convention. Getting `vol(K, T)`
for an absolute strike runs a sticky-delta fixed point (strike, delta, vol,
delta). Note the split in the calendar checks: the no-arbitrage condition is
monotone total variance at fixed forward moneyness, checked by
`calendar_arbitrage_report`; the fixed-delta view, `delta_calendar_report`, is
a monitoring report of the quoted pillars and neither implies nor is implied
by absence of calendar arbitrage.

</details>

**Q25. FX has a second smile model, vanna-volga, alongside SVI: what does it do differently and why keep both?**

<sub>Tags: `L3`, `vol-surface`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Vanna-volga prices any strike as the flat-ATM Black-Scholes price plus the
cost of a hedge: the portfolio of the three traded pillars (25-delta put, ATM,
25-delta call) that exactly replicates the target option's vega, vanna and
volga at the reference vol, solved as an exact 3x3 linear system per strike
(`smile.py`, residual < 1e-12 tested) rather than the common first-order
shortcut. It's exact at the pillars, cheap, and is what desks actually use to
smile-adjust barriers and touches, but METHODOLOGY.md is explicit that it has
no dynamics and its quadratic wing extrapolation can violate no-arbitrage
beyond about 5-delta. SVI is kept alongside it because SVI carries the
analytic Durrleman diagnostic VV lacks; the pipeline compares both at 15-delta
(agreeing to ~5bp) and in the wings (diverging 10+bp at 5-delta), and that
divergence is treated as the measure of interpolation-model risk rather than
something to reconcile away.

</details>

**Q26. Both Heston MC engines in this portfolio are cross-validated against Fourier pricing within "3 standard errors." Why does that specific tolerance depend on the odd-`n_paths` guard above?**

<sub>Tags: `L3`, `vol-surface`, `numeric`</sub>

<details><summary>Answer</summary>

The "agree within 3 SE" contract (stated in CONVENTIONS.md as the
portfolio-wide Monte Carlo tolerance, and used throughout both `VALIDATION.md`
files, e.g. equity's `|z| <= 1.6`/`2.4` cross-checks and FX's Euler/QE-vs-COS
table) is only a meaningful acceptance bar if the reported SE is a correct
estimate of the pricer's actual sampling uncertainty. If an odd `n_paths` were
silently understating the SE, a biased or high-variance MC price could still
land inside a falsely narrow "3 SE" band and pass a test or a model-validation
sign-off that should have failed it, the tolerance's usefulness is entirely
downstream of the SE calculation being correct, which is exactly the invariant
`simulate_terminal`'s odd-`n_paths` guard now protects.

</details>

**Q27. Why does `eq_surface.ssvi` matter if the repository already fits raw SVI per expiry?**

<sub>Tags: `L3`, `vol-surface`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Per-expiry SVI fits can cross each other, which is calendar arbitrage. SSVI
(Gatheral and Jacquier 2014) parametrises the whole surface with one `rho` and
a function `phi(theta)`, and their Theorem 4.2 gives parameter conditions
under which no slice has butterfly arbitrage. The tests check that each SSVI
slice equals a raw SVI slice to 1e-15, that parameters meeting the conditions
give a non-negative Durrleman density function, and that
`calendar_arbitrage_free` flags crossing slices. For eSSVI the calendar check
is on a grid, not proved.

Source: `python/equity/09-vol-surface/src/eq_surface/ssvi.py` and `python/equity/09-vol-surface/tests/`.

</details>

**Q28. How is `eq_surface.dupire` checked, and what happens with a noisy surface?**

<sub>Tags: `L3`, `vol-surface`, `numeric`</sub>

<details><summary>Answer</summary>

Local vol is computed two ways, from total variance (Gatheral eq. 1.10) and
from call prices (Dupire 1994), and the two must agree. A flat surface must
give flat local vol, a term-only surface must give `dw/dT`, and a local-vol
Monte Carlo must reprice the surface it came from. Dupire uses finite
differences, so noise in the input surface becomes noise in local vol;
negative local variance raises `LocalVolArbitrageError` rather than being
clipped.

Source: `python/equity/09-vol-surface/src/eq_surface/dupire.py` and `python/equity/09-vol-surface/tests/`.

</details>

**Q29. Bates adds jumps to Heston. Which two limits make the oracles for `eq_surface.bates`?**

<sub>Tags: `L3`, `vol-surface`</sub>

<details><summary>Answer</summary>

With jump intensity zero, `bates_call` must equal the Heston price (to 1e-7). With vol of vol zero and `v0 = theta`, variance is constant and the price must equal the Merton (1976) jump-diffusion series, `merton_call` (to 1e-8). Put-call parity and an Euler Monte Carlo within 4 standard errors complete the checks.

</details>


## Set 6: Market Risk: VaR Methods

**Q1. What does a 1-day 99% VaR of $54,681 actually tell you?**

<sub>Tags: `L1`, `var`, `numeric`</sub>

<details><summary>Answer</summary>

It is a quantile of the P&L distribution, not a prediction of the worst loss:
it says there is a 1% (unconditional) probability that tomorrow's P&L falls
below −$54,681 (the historical 99% VaR of the synthetic demo book in the
equity `docs/VALIDATION.md` §3). Formally the engines define `VaR_alpha = -Q_alpha(P&L)`,
so VaR is a threshold, not a loss estimate. It says nothing about how bad the
loss is *if* that threshold is breached, that is exactly what Set 7's Expected
Shortfall is for.

</details>

**Q2. Why is "VaR ignores tail severity" considered VaR's most-cited structural weakness?**

<sub>Tags: `L2`, `var`, `numeric`</sub>

<details><summary>Answer</summary>

Because a single number can pass every backtest while hiding an arbitrarily
fat tail beyond it: `docs/VALIDATION.md` §5.1 shows the synthetic equity demo
book's 99% VaR is $55k, but the 2020 COVID-replay stress loss is $254k and the
historical 99% ES is $68k, a 5× gap between "the VaR number" and what actually
happens once you're in the tail. VaR is a pass/fail line; it carries no
information about the average or worst loss conditional on crossing it. This
is the motivating defect fixed by Expected Shortfall (`ES_alpha = -E[P&L | P&L <= Q_alpha]`),
which FRTB adopted precisely because it forces the tail's magnitude into the
number.

</details>

**Q3. What does historical simulation assume, and what is its single biggest limitation?**

<sub>Tags: `L1`, `var`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Plain historical VaR (`historical_var` in `eq_var/historical_var.py`) makes no
distributional assumption at all, it just takes the order statistic
`L(ceil(n*alpha))` of realised losses (no interpolation), so it captures
whatever fat tails, skew, and cross-asset dependence actually happened. Its
limitation is that it is bounded by how much history you have and by the
assumption that the future resembles that specific window: `MIN_OBS = 50` is
enforced because a 1–5% tail is unresolvable on shorter samples, and even at
250+ days a 1% quantile rests on only 2–3 tail observations per year
(`docs/METHODOLOGY.md` §2 table, "Estimation noise at 99%: high").

</details>

**Q4. What is "window myopia" (the "great moderation" problem) in historical VaR, and how is it demonstrated?**

<sub>Tags: `L1`, `var`, `numeric`</sub>

<details><summary>Answer</summary>

It is the failure mode where plain HS is only as good as its window: after a
calm stretch followed by a short wild one, plain HS barely moves because the
wild days are still a minority of the sample
(`test_fhs_scales_up_after_vol_regime_switch`,
`test_factor_fhs_reacts_to_a_vol_spike_and_hs_does_not`, which triples the
last 10 days of factor moves and finds per-factor FHS roughly triples while
plain HS rises by under half). Symmetrically, after markets calm, plain HS
overstates current risk for as long as the crisis days stay in the window,
then drops abruptly on the day they roll out, with no change in the market
(the ghost effect). The 500-day walk-forward backtest on synthetic
GARCH(1,1)-t(5) data in `docs/VALIDATION.md` §4 turns this into p-values:
plain HS draws 10 exceptions against an expectation of 5 (Kupiec p = 0.048,
yellow zone), while per-factor FHS draws 4 (Kupiec p = 0.64, green). This is
one seed of a synthetic process; the tests pin the properties, not the counts.

</details>

**Q5. What does age-weighting (BRW) add over plain historical VaR, and why choose a particular decay `lam`?**

<sub>Tags: `L1`, `var`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`brw_weights` in `eq_var/historical_var.py` assigns observation `i` weight
proportional to `lam**(n-1-i)`, so more recent P&L dominates the weighted
empirical CDF that `age_weighted_var` inverts; `lam -> 1` recovers plain HS
(`test_age_weighting_converges_to_plain_hs_as_lambda_goes_to_one`). The reason
to weight recent history more heavily is that risk is regime-dependent, a
calm-then-turbulent transition should show up in VaR within days, not once the
window rolls over. The choice of decay is a bias-variance trade-off: a `lam`
close to 1 barely differs from plain HS (slow to react, but stable, low
estimation noise), while a fast decay concentrates the effective sample on
very few recent points, reacting quickly but at the cost of a noisier quantile
estimate, the same trade-off the engine flags for EWMA's `lam=0.94` in
assumption A6 of `docs/METHODOLOGY.md`: "too-slow decay lags regime shifts;
too-fast is noisy."

</details>

**Q6. Why does the portfolio treat Filtered Historical Simulation (FHS), not plain HS or BRW, as the main historical estimator?**

<sub>Tags: `L2`, `var`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

FHS (`filtered_historical_var`) devolatilises each historical P&L by its
one-step-ahead EWMA vol forecast (`z_t = pnl_t / sigma_t`), then rescales
every standardised innovation to *tomorrow's* forecast before taking the
empirical quantile, so it keeps the empirical (fat-tailed, skewed) shape of
returns while being conditional on the current vol regime, unlike plain HS
(unconditional) or BRW (reweights the sample but does not rescale its scale).
`docs/METHODOLOGY.md` §2 calls this combination "the industry workhorse," and
the synthetic GARCH backtest shows why: per-factor FHS lands in the green zone
(4 exceptions, Kupiec p = 0.64, CC p = 0.87) while parametric-normal (14
exceptions, Kupiec p = 0.0009) and plain HS (10, p = 0.048) land in yellow.
The EWMA recursion is seeded with the mean of `x^2` over the first `min(30, n)`
observations (`SEED_WINDOW = 30`).

</details>

**Q7. What does the parametric (delta-normal) method assume, and how is portfolio sigma computed?**

<sub>Tags: `L1`, `var`, `code-reading`</sub>

<details><summary>Answer</summary>

It assumes P&L is well described by a distribution, the normal baseline, or
Student-t / Cornish-Fisher tail corrections, and computes VaR analytically
from that distribution's mean and variance rather than resampling history.
`portfolio_sigma` in `eq_var/parametric_var.py` computes `sigma_p = sqrt(w' Sigma w)`
from dollar exposures `w` (the factor mapping) and a sample or EWMA covariance
`Sigma`; `parametric_var` then returns `-(mu + z_alpha * sigma)` with `z_alpha = norm.ppf(alpha)`
for the normal case. It is cheap (uses the whole sample for sigma rather than
just tail points) and analytically decomposable, which is why the FX desk
guide calls it "the desk's intraday what-if and marginal-VaR tool."

</details>

**Q8. Why does pure normality understate real, fat-tailed market risk?**

<sub>Tags: `L2`, `var`, `numeric`</sub>

<details><summary>Answer</summary>

A normal distribution has no excess kurtosis, so its tail decays faster than
real daily P&L, which typically shows excess kurtosis (the equity GARCH sample
has 4.76 at 1-day per `docs/METHODOLOGY.md` §6). The cross-model table in
`docs/VALIDATION.md` §3 makes this concrete: on the synthetic demo book,
parametric-normal 99% VaR is $46,864 while parametric-t(6) is $51,691 and
historical/MC-t agree near that higher number, "the normal family sits ~10%
lower at 99%, the missing kurtosis." The 500-day GARCH backtest converts this
into an actual failure: parametric-normal draws 14 exceptions against an
expectation of 5, Kupiec p = 0.0009.

</details>

**Q9. What is the Cornish-Fisher expansion doing, mathematically, and what does it need beyond a plain z-score?**

<sub>Tags: `L1`, `var`, `code-reading`</sub>

<details><summary>Answer</summary>

CF adjusts the standard normal quantile using the P&L's sample skewness `S`
and excess kurtosis `K`: `z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36`
(`cornish_fisher_z` in both `eq_var/parametric_var.py` and
`fx_var/parametric_var.py`), reducing exactly to `z` when `S = K = 0`
(unit-tested to 1e-12). It is attractive because it is cheap and uses
distributional information, skew and fat tails, that the plain normal quantile
ignores entirely, without requiring a full Monte Carlo or historical resample.

</details>

**Q10. Why does the Cornish-Fisher quantile need a domain/monotonicity check at all: what goes wrong without one?**

<sub>Tags: `L2`, `var`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`z_cf` is a cubic polynomial in `z`, and a cubic is only a valid quantile
function where it is monotone increasing in `z`; for large enough
skew/kurtosis combinations the polynomial turns over inside the range of `z`
the requested quantile uses, so a *smaller* `z` (less extreme alpha) can map
to a more extreme `z_cf` than a larger one. `docs/METHODOLOGY.md` §5 states
the consequence directly: "the 99% 'quantile' can cross the 95% one; a number
produced there is not a VaR." That is why `cornish_fisher_var` in both engines
raises `ValueError` by default (`check_domain=True`) instead of silently
returning a number when `cornish_fisher_valid(skew, excess_kurt, alpha)` is
false, e.g. S=3 or K=10 is rejected, S=−0.3, K=2 passes
(`test_var_raises_outside_domain`).

</details>

**Q11. Derive the domain-check condition: why is checking `dz_cf/dz > 0` equivalent to checking a quadratic's minimum?**

<sub>Tags: `L3`, `var`, `code-reading`</sub>

<details><summary>Answer</summary>

Differentiating the CF cubic with respect to `z` gives `dz_cf/dz = 1 + zS/3 + (3z^2-3)K/24 - (6z^2-5)S^2/36`,
which, collecting terms in `z`, is itself a quadratic `g(z) = A z^2 + B z + C`
with `A = K/8 - S^2/6`, `B = S/3`, `C = 1 - K/8 + 5S^2/36` (see
`cornish_fisher_valid` in `eq_var/parametric_var.py`). The expansion is
monotone on an interval exactly when this quadratic derivative stays strictly
positive there, so the domain check reduces to finding `g`'s minimum on
`[-z_a, z_a]`, with `z_a = Phi^{-1}(1 - alpha)`, and testing its sign, a
textbook one-variable optimisation, not a property that needs numerical
search.

</details>

**Q12. Why does the Cornish-Fisher domain check use a closed-form minimum instead of scanning a grid of `z` values?**

<sub>Tags: `L2`, `var`, `numeric`</sub>

<details><summary>Answer</summary>

A grid scan flags non-monotonicity only if a sampled point shows a negative
derivative, so it can miss a thin non-monotone region that falls entirely
between two adjacent nodes: the region is real, but no sampled point lands
inside it. The equity `docs/VALIDATION.md` §5.7 gives the example: `(skew, excess_kurt) = (-0.0105, 8.0001)`
has its minimum derivative, about `-2.5e-7`, at `z = 0.00175`, and a
2001-point grid over `|z| <= 3.5` misses it
(`test_domain_check_is_exact_not_grid_resolution_dependent`). Since the
derivative is a quadratic, its minimum on a closed interval has a closed form,
so no grid is needed. All six VaR engines (Python, C++ and Rust, equity and
FX) share the same closed-form `cornish_fisher_valid` rule.

</details>

**Q13. How does `cornish_fisher_valid` compute the minimum, and why is the result exact rather than just "less likely to fail"?**

<sub>Tags: `L2`, `var`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`cornish_fisher_valid(skew, excess_kurt, alpha)` takes the exact minimum of
the quadratic derivative `g(z) = A z^2 + B z + C` on `[-z_a, z_a]`: it
evaluates both endpoints, plus the vertex `z* = -B/(2A)` when `A > 0` and the
vertex lies inside the interval; for concave or linear `g` (`A <= 0`) the
minimum on a closed interval is always at an endpoint. This is not a
resolution improvement, it is a different kind of guarantee: a grid, however
fine, leaves gaps between nodes where a violation can hide, whereas the closed
form evaluates the *actual* minimum of a one-dimensional quadratic, which
cannot be missed however thin the non-monotone region is. The interval depends
on `alpha` because only the part of the curve up to the requested quantile
matters. The old `cornish_fisher_domain_ok(skew, excess_kurt, z_range=3.5)` is
kept for one release as a deprecated predicate on the fixed interval
`[-z_range, z_range]`.

Source: `python/equity/03-var-es-engine/src/eq_var/parametric_var.py` (`cornish_fisher_valid`).

</details>

**Q14. Why is `(skew, excess_kurt) = (0.122, -0.427)` valid at `alpha = 0.01` but rejected by the old fixed-interval check, while `(-0.0105, 8.0001)` is invalid at every alpha?**

<sub>Tags: `L2`, `var`, `numeric`</sub>

<details><summary>Answer</summary>

`(0.122, -0.427)` gives a concave derivative (`A < 0`), so its minimum is at
an endpoint. On the FX engine's old fixed interval `|z| <= 4` the minimum is
at `z = -4`, derivative about `-9.15e-4`, so the deprecated FX predicate
rejects it (the equity engine's deprecated check uses `|z| <= 3.5`, where the
derivative is still positive and the pair passes). At `alpha = 0.01` the
interval is `[-2.326, 2.326]`, the derivative there is positive, and
`cornish_fisher_valid` accepts it; it is still valid at `alpha = 0.001` (`z_a = 3.090`).
The dip at `z = -4` is beyond any quantile the engine is asked for, so
rejecting the pair was too strict. `(-0.0105, 8.0001)` is the opposite case:
its minimum, about `-2.5e-7`, sits at `z = 0.00175`, near the centre, so it
lies inside every `[-z_a, z_a]` and the pair is invalid at every alpha (FX
`docs/VALIDATION.md` table;
`test_cf_counterexample_minimum_is_at_the_endpoint_and_alpha_aware`,
`test_cf_valid_tightens_with_smaller_alpha`). The lesson: the validity region
depends on which quantile you need, and an exact check can say so precisely.

</details>

**Q15. Why does Monte Carlo VaR fully revalue the book per scenario instead of using a linear or delta-gamma approximation?**

<sub>Tags: `L2`, `var`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Full revaluation reprices every instrument (e.g. Black-Scholes /
Garman-Kohlhagen at the shocked spot, rates, and vol) at each simulated
scenario, so it captures the true, non-linear P&L including option convexity
and the exact payoff shape, it makes no local-approximation error at all,
unlike a delta or delta-gamma mapping. `monte_carlo_pnl` in
`eq_var/monte_carlo_var.py` defaults to `method="full"` for exactly this
reason, and `docs/METHODOLOGY.md` §2 calls MC "the only family that combines a
chosen tail model with exact treatment of option convexity." The cost of the
approximation it avoids is measured directly elsewhere in the portfolio
(`Portfolio.approximation_error`): delta-gamma understates the COVID stress
loss by $93k (37% of P&L) once a long put goes deep ITM, because the quadratic
Taylor term keeps growing while the true payoff is bounded.

</details>

**Q16. What three factor-return distributions does the FX Monte Carlo engine support, and when is each the right choice?**

<sub>Tags: `L2`, `var`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`fx_var/monte_carlo_var.py`'s `simulate_factor_returns` supports `"normal"`
(MVN(0, Sigma·h) via `robust_cholesky`), `"t"` (multivariate Student-t built
as `Z·sqrt((df-2)/df)/sqrt(W/df)`, scaled so the *covariance still matches
Sigma exactly*, any VaR difference from normal is pure tail shape, not a sigma
mismatch), and `"jump"` (normal diffusion plus a Bernoulli(prob) common jump
with per-factor mean/std via `JumpSpec`). Normal is the RiskMetrics-style
baseline when tail shape isn't the concern; Student-t is the right choice for
EM-currency-like fat tails at matched covariance (df≈4-6, +12% at 99% VaR for
the demo EM book per `docs/METHODOLOGY.md` §2.3); the jump-mixture is
specifically for devaluation/peg-break risk that the covariance matrix
structurally cannot see, it "adds variance on top of Sigma by design" and
lifts the synthetic demo book's 99% VaR by +140%.

</details>

**Q17. Concretely, what does `JumpSpec` let you model, and how does it enter the simulated scenarios?**

<sub>Tags: `L2`, `var`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`JumpSpec` (a frozen dataclass in `fx_var/monte_carlo_var.py`) carries a
per-scenario jump probability `prob`, a per-factor jump mean (e.g. `{"FX:TRY": -0.15}`
for a 15% log devaluation of the lira), and an optional per-factor jump std (0 =
deterministic jump size). Inside `simulate_factor_returns`, a Bernoulli draw
`hit = rng.random(n_scenarios) < jumps.prob` selects which scenarios get the
jump, and for every factor listed in `jumps.mean` a random jump size `mean + std * standard_normal`
is added to that factor's diffusion path only where `hit` is true. This is the
mechanism behind the peg-break/devaluation overlay described in
`docs/METHODOLOGY.md` §4, it prices a discrete, low-probability,
large-magnitude event into a quantile, which a continuous covariance-driven
diffusion cannot represent.

</details>

**Q18. Why is a "legitimate" case for a singular or near-singular covariance matrix specifically an FX/pegged-currency scenario, and how does `robust_cholesky` handle it?**

<sub>Tags: `L2`, `var`, `code-reading`</sub>

<details><summary>Answer</summary>

Two currencies pegged to the same anchor (e.g. an HKD-band or SAR hard peg
both effectively tracking USD) realise near-zero daily vol and near-perfect
correlation to each other and to the anchor, which makes the factor covariance
matrix exactly or numerically singular; this is a real, structural feature of
FX markets, not a data artefact to be cleaned up. `robust_cholesky` in
`fx_var/monte_carlo_var.py` first tries plain Cholesky; on failure it adds
jitter starting at `1e-12 * mean(diag)` and rising x10 per try, capped at
`1e-6 * mean(diag)`. Any jitter raises a `NumericalWarning`, so the escalation
is visible rather than silent, and because the jitter is tiny relative to the
real variances, simulated moments are unchanged to within MC noise. Past the
cap it raises, naming the most negative eigenvalue, unless the caller passes
`repair="clip"` or `repair="higham"` for an explicit nearest-PSD repair. The
equity engine's `cholesky` follows the same capped rule.

</details>

**Q19. Why can't parametric or historical VaR see peg-break risk, and what is the engine's policy response?**

<sub>Tags: `L2`, `var`, `numeric`</sub>

<details><summary>Answer</summary>

A pegged pair's return history inside any window that doesn't contain the
actual break is a spike at zero, near-zero daily vol, so the empirical
quantile (HS) and the covariance-implied sigma (parametric) are both near-zero
right up until the peg snaps. `docs/METHODOLOGY.md` §4 states the policy
directly: any FX factor with daily sigma below 0.05% (~0.8% annualised)
triggers a `PegBlindnessWarning` naming the factor; the flagged book must then
carry a `peg_break_scenario` stress add-on (a configurable jump, vol spike,
and contagion co-move); and for a genuine quantile that includes the break,
the jump-mixture Monte Carlo is the tool, since it prices the event with an
attached probability rather than assuming it away. The demo-book numbers make
the gap concrete: HS 99% VaR of $0.69m vs. a −30% HKD peg-break stress loss of
$15.0m, 21.7× the VaR.

</details>

**Q20. What is the real architectural difference between how equity VaR and FX VaR handle Monte Carlo revaluation?**

<sub>Tags: `L1`, `var`, `code-reading`</sub>

<details><summary>Answer</summary>

Equity's Monte Carlo (`eq_var/monte_carlo_var.py`) revalues a `Portfolio` of
linear equity/index exposures plus options, where non-option positions are
pure delta exposures (`P&L = dollar-exposure * return`) and only options get a
full Black-Scholes reprice. FX's Monte Carlo (`fx_var/monte_carlo_var.py`, via
`fx_var/book.py`) does full book revaluation for *every* position type through
a `Book`/`Market` layer: `Book._position_value_usd` reprices cash, spot,
forward (as two CIP deposit legs), and option (via Garman-Kohlhagen) positions
scenario by scenario from shocked spot/rate/vol curves, then differences
base-currency value before and after the shock (`Book.pnl`). There is no
"linear exposure times return" shortcut anywhere in the FX P&L path, even a
plain spot FX position is repriced as two USD legs (`pos.notional * spot[b] - pos.notional * x0 * spot[q]`)
rather than approximated by a single delta, because USD triangulation requires
every cross to be reconstructed consistently from its legs in every scenario.

</details>

**Q21. Why does FX VaR need the USD-triangulation factor scheme instead of one factor per traded pair?**

<sub>Tags: `L2`, `var`</sub>

<details><summary>Answer</summary>

If every traded pair (EURUSD, USDJPY, EURJPY, …) got its own risk factor, the
covariance/scenario set would have to satisfy the constraint `log EURJPY = log EURUSD + log USDJPY`
(linear in logs) exactly, or a historical/parametric/MC scenario could move
EURJPY without moving either USD leg, an arbitrage-inconsistent scenario.
`python/fx/03-var-es-engine/docs/METHODOLOGY.md` §1.1 explains that mapping
every currency to a single `FX:CCY` factor (the USD price of one unit of that
currency) makes the identity hold *by construction* in every scenario, and a
cross position such as EURJPY is simply decomposed into its two USD legs. This
is tested to machine precision (`test_triangulation_identity_eurjpy`).

</details>

**Q22. How does `Book.pnl` compute base-currency P&L, and why does a pure base-currency cash balance carry zero risk?**

<sub>Tags: `L2`, `var`</sub>

<details><summary>Answer</summary>

`Book.pnl` computes `PnL = V1_usd / S1_base - V0_usd / S0_base`, where
`V0_usd`/`V1_usd` are the book's total USD value before/after the shock
(summed via `_position_value_usd` over all positions) and `S0_base`/`S1_base`
are the base currency's own USD price before/after, critically, the base
currency's own FX factor is applied consistently to *both* the book value and
the denominator. A pure base-currency `Cash` position has `V_usd = amount * spot[base]`,
so dividing by `spot[base]` again cancels exactly regardless of how the
base-currency factor is shocked, leaving zero P&L, this is a direct
consequence of measuring P&L in the position's own currency, not an extra rule
bolted on.

</details>

**Q23. What does `linear_exposures` compute in the FX engine, and where in the pipeline is a linear approximation still used despite full-reval MC being available?**

<sub>Tags: `L1`, `var`, `code-reading`</sub>

<details><summary>Answer</summary>

`Book.linear_exposures` in `fx_var/book.py` computes central finite-difference
deltas `dPnL/dfactor` for every factor the book touches, by bumping each
factor by `+/-bump` (default 1e-6) through the *same* full-revaluation `pnl`
method and dividing by `2*bump`, so even the "linear" exposures used by
parametric VaR are derived from the true GK/CIP repricing, not a separate
analytic Greek formula. These exposures are what `parametric_var`/`var_covar`
use to compute `sigma_p = sqrt(w' Sigma w)`, since parametric VaR is
inherently a linear (delta) method by construction, the linearisation happens
in how parametric VaR *consumes* the book, not in how the book itself is
priced.

</details>

**Q24. Why does the FX options pricer default to full Garman-Kohlhagen revaluation rather than delta-vega(-gamma) mapping, and what does the mapping systematically get wrong?**

<sub>Tags: `L2`, `var`</sub>

<details><summary>Answer</summary>

`python/fx/03-var-es-engine/docs/METHODOLOGY.md` §1.3 states the mapping's
error is quadratic (cubic once gamma is added) in the shock size, and
characterises it precisely: for a long option (positive gamma), delta-only P&L *understates*
gains and *overstates* losses in both directions, which makes a mapping-based
VaR conservative for long-gamma books but dangerous, understating risk, for
short-gamma books. Because the desk cannot assume every book is long gamma,
full revaluation (`option_method="full"` in `Book._position_value_usd`, which
calls `gk_price` at the shocked spot/rate/vol per scenario) is the default;
the vectorised implementation makes this cheap (100k scenarios in
milliseconds), so there is no real performance reason to prefer the
approximation for VaR.

</details>

**Q25. How does the equity engine quantify the size of the delta-gamma approximation error, rather than just asserting full reval is "more correct"?**

<sub>Tags: `L2`, `var`, `numeric`</sub>

<details><summary>Answer</summary>

`Portfolio.approximation_error` computes both the full-revaluation P&L and the
delta-gamma-vega P&L (`dV ~ Delta*dS + 1/2*Gamma*dS^2 + nu*dsigma`) for the
same scenario and reports the difference directly, so the error is *measured*,
not assumed. On the synthetic demo book it is small for a −15% scenario
(−$1.2k, 0.9%) but balloons to +$93k (37% of P&L) in the −34% COVID replay,
because the quadratic Taylor term keeps growing with the shock while the true
option payoff is bounded, this is unit-tested as monotone in shock size
(`test_approximation_error_grows_with_shock_size`) and sign-correct (long
gamma cushions losses, short gamma amplifies them,
`test_long_gamma_reduces_loss_vs_pure_delta`). `docs/DESK_GUIDE.md` calls this
divergence the stress committee's evidence that "Greeks-based intraday risk is
unsafe for gap moves and the overnight batch must fully revalue."

</details>

**Q26. Why does the FX Monte Carlo engine's variance-matched Student-t scale by `sqrt((df-2)/df)` rather than sampling a raw multivariate-t?**

<sub>Tags: `L2`, `var`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

A raw multivariate Student-t with `df` degrees of freedom has covariance
`df/(df-2) * Sigma_scale`, so sampling directly from a t with scale matrix
`Sigma` would give factor returns with a *larger* covariance than `Sigma`,
confounding "the tails are fatter" with "the variance is also bigger."
`fx_var/monte_carlo_var.py`'s `simulate_factor_returns` instead draws `x = z @ chol.T`
from the normal Cholesky factor of `Sigma` and then multiplies by
`sqrt((df-2)/df) / sqrt(W/df)` (`W` a chi-square(df) draw), which rescales the
t draw so its covariance matches `Sigma` exactly. This means any VaR
difference between the `"normal"` and `"t"` runs at the same `Sigma` is
attributable purely to tail shape, the module docstring states this
explicitly: "any 99% VaR difference is pure tail shape."

</details>

**Q27. What is the practical trap in "switching to Student-t VaR to be more conservative," according to this portfolio's tests?**

<sub>Tags: `L1`, `var`, `numeric`</sub>

<details><summary>Answer</summary>

The variance-matched Student-t is fatter than the normal *only* in the deep
tail, near the shoulder of the distribution (alpha around 5-10%) it is
actually thinner, because matching the variance makes the t density more
peaked at the centre and fatter in the far tail, which leaves less mass in the
shoulders. `docs/VALIDATION.md` §6.1 documents the crossover explicitly: t(5)
charges *less* than normal at 95% VaR, with the crossover between t and normal
occurring near alpha ≈ 2.8%
(`test_t_vs_normal_var_crossover_is_between_2_5_and_5_percent`), while ES,
which averages the whole tail rather than reading off one point, stays higher
than normal at every alpha. The practical trap: someone switching a 95% VaR
limit from normal to Student-t "to be conservative" would actually see the
reported VaR *fall*.

</details>

**Q28. Why does the engine implement three VaR families instead of picking the "best" one?**

<sub>Tags: `L2`, `var`</sub>

<details><summary>Answer</summary>

`docs/METHODOLOGY.md` §2 states the reasoning directly: no single VaR
estimator dominates in every regime, so every real risk department runs at
least two and reconciles them, the engine implements historical, parametric,
and Monte Carlo so the *disagreement between methods itself becomes a
diagnostic*. The cross-model consistency table in `docs/VALIDATION.md` §3
shows how to read that disagreement: MC-normal tracking parametric-normal
closely cross-validates both implementations (same model, only MC noise
separates them); parametric-t, MC-t, and historical clustering together says
the fat-tail-aware methods agree because the underlying history really is
t-distributed; and FHS sitting noticeably below the unconditional methods is
not a bug but information, "current vol is quiet relative to the window,"
exactly what a conditional estimator is supposed to show when the regime has
calmed.

</details>

**Q29. Why is `sigma_p = sqrt(w' Sigma w)` guarded with a "not positive semi-definite" check, and what would trigger it?**

<sub>Tags: `L2`, `var`, `code-reading`</sub>

<details><summary>Answer</summary>

`portfolio_sigma` in `eq_var/parametric_var.py` computes `var = w @ Sigma @ w`
and raises `ValueError` if `var` is negative beyond a small numerical
tolerance (`-1e-10 * max(1, max|w|^2)`), rather than silently taking `sqrt` of
a negative number (which would produce NaN or a complex value depending on the
code path) or clamping without comment. A valid covariance matrix is positive
semi-definite by construction, so `w' Sigma w >= 0` for any `w`; a violation
signals that `Sigma` itself is broken, e.g. built from mismatched/inconsistent
inputs, or an EWMA/sample estimate corrupted upstream, and the engine's stated
policy across both `eq_var` and `fx_var` is "refuse, never impute": a NaN or
structurally invalid VaR must fail loudly rather than colour a traffic light
incorrectly, exactly the concern documented for FX assumption A12.

</details>

**Q30. Given all three VaR families and the Cornish-Fisher fix, what is the one-sentence takeaway for how a desk should treat any single VaR number?**

<sub>Tags: `L2`, `var`</sub>

<details><summary>Answer</summary>

Every method here encodes a different modelling choice, how much history to
trust and how to weight it (historical/BRW/FHS), what distribution to assume
and how to correct it (parametric normal/t/Cornish-Fisher, now with an exact,
alpha-dependent domain check), or what factor model to simulate and whether to
fully revalue (Monte Carlo normal/t/jump), so a single VaR print is a
modelling artifact, not a physical constant of the portfolio, and the
portfolio's own design choice (running all three and reading their
disagreement, per `docs/METHODOLOGY.md` §2 and the desk guide's "morning
pack") treats that plurality as the actual risk signal rather than a nuisance
to be averaged away.

</details>

**Q31. What does `eq_var.risk_attribution` give a risk manager, and what property is tested first?**

<sub>Tags: `L1`, `var`</sub>

<details><summary>Answer</summary>

It splits VaR and ES into Euler marginal, component and incremental
contributions, in closed form (`parametric_attribution`) and from scenarios
(`simulation_attribution`). VaR and ES are homogeneous of degree one in the
positions, so by Euler's theorem the components must sum to the total. That
sum, a hand-worked example and a finite-difference gradient are the first
oracles. The scenario version uses a box kernel around the VaR scenario, which
is noisy on small samples.

</details>

**Q32. Why does `eq_var.revaluation` exist when the VaR engines already handle linear books?**

<sub>Tags: `L2`, `var`</sub>

<details><summary>Answer</summary>

An option book is not linear: delta-only P&L misses gamma and vega, and
delta-gamma misses more in large moves. `full_revaluation_pnl` reprices each
option under every scenario; `taylor_pnl` gives the delta, delta-gamma and
full Taylor approximations; `pnl_explain` splits P&L into delta, gamma, vega,
vanna, volga, theta and unexplained. The C++ and Rust VaR engines gained
`option_reval`, matched to Python to 1e-11, and their historical and Monte
Carlo VaR use it to reprice every option leg in every scenario (vol held at
today's level).

</details>


## Set 7: Expected Shortfall, Backtesting & Basel

**Q1. VaR already tells you the loss threshold at a confidence level. Why does this portfolio also compute Expected Shortfall?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

VaR only answers "how bad, at worst, one time in `alpha`", it says nothing
about how bad things get *beyond* that threshold. ES answers exactly that:
it's the average loss conditional on being in the tail past VaR
(`expected_shortfall.py`'s module docstring: `ES_alpha = -(1/alpha) * integral_0^alpha Q_u(pnl) du`,
"the average loss in the worst `alpha` tail"). Two books can share the same
99% VaR while one has a much fatter tail beyond it, VaR can't distinguish
them, ES can. `normal_es`/`student_t_es`/`empirical_es` all satisfy `ES >= VaR`
by construction, which is the formal expression of "ES sees what's past the
VaR cutoff."

</details>

**Q2. What does "ES is coherent and VaR isn't" mean in practice, without going through the subadditivity proof?**

<sub>Tags: `L2`, `es-backtesting`, `code-reading`</sub>

<details><summary>Answer</summary>

A coherent risk measure is guaranteed to reward diversification: the risk of a
combined portfolio is never *more* than the sum of the risks of its parts. ES
has this property; VaR in general does not, you can construct portfolios (the
FX docstring points at "a VaR subadditivity counterexample built from peg-jump
assets" in `tests/test_expected_shortfall.py`) where the combined VaR exceeds
the sum of the standalone VaRs, so netting two books together looks *riskier*
by VaR than holding them apart. That's a red flag for any measure used to
aggregate risk across desks or set diversified limits, and it's the headline
reason FRTB moved off VaR.

</details>

**Q3. Where does the FX engine's ES estimator differ from "just average the P&L below the VaR line," and why does that distinction matter?**

<sub>Tags: `L3`, `es-backtesting`, `code-reading`</sub>

<details><summary>Answer</summary>

`fx_var/expected_shortfall.py` uses the Acerbi–Tasche tail-splitting estimator
(`_es_from_tail`): it averages the worst losses over *exactly* `alpha` of
probability mass (`alpha` is the tail probability), taking only a fractional
share of the observation sitting right at the VaR boundary rather than
including or excluding it wholesale. The docstring is explicit that the naive
"mean of observations beyond VaR" over-weights probability atoms and "can
spuriously break subadditivity exactly in the pegged-currency jump case this
project cares about", i.e. the naive estimator can fail the very coherence
property that's ES's main selling point over VaR, so getting the
tail-splitting right isn't cosmetic.

</details>

**Q4. What does Kupiec's proportion-of-failures (POF) test actually check, mechanically?**

<sub>Tags: `L2`, `es-backtesting`, `code-reading`</sub>

<details><summary>Answer</summary>

It counts exceptions, days where realised P&L breaches the VaR forecast
(`exceptions_from_pnl`: `pnl_t < -VaR_t`, strict inequality), over a window of
`n_obs` days, and asks whether that count is statistically consistent with the
stated confidence level. `kupiec_pof(n_obs=..., n_exceptions=..., alpha=...)`
(keyword-only; the positional form is deprecated) builds the likelihood ratio
`LR_uc = -2 ln[(1-p)^{T-x} p^x / ((1-x/T)^{T-x}(x/T)^x)]` comparing the
model's claimed exception probability `p = alpha` against the empirical rate
`x/T`, and refers `LR_uc` to a chi-squared(1) distribution to get a p-value.
It's a test of unconditional coverage only, it says nothing about whether
exceptions cluster in time (that's Christoffersen's job).

</details>

**Q5. `docs/VALIDATION.md` flags Kupiec's chi-squared(1) reference distribution as "oversized exactly at the regulatory window." What does that mean quantitatively?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

At `n_obs=250, alpha=0.01`, the Basel backtesting window, expected exception
count 2.5, the chi-squared(1) asymptotic is a poor approximation to the true
discrete distribution of the LR statistic under a rare binomial. Computing the *exact*
rejection probability by summing `Binomial(250, 0.01)` mass over the LR
statistic's rejection region gives roughly **9.5%**, not the nominal 5% the
chi-squared reference implies. In plain terms: a correctly-calibrated model
gets flagged "reject at the 5% level" about twice as often as the p-value
would lead you to believe, purely from using a windows-and-sample-size
combination where the asymptotic approximation is thin.

</details>

**Q6. Why is that oversizing described as "a property of the classical test, not a bug in this implementation" rather than something to fix?**

<sub>Tags: `L3`, `es-backtesting`</sub>

<details><summary>Answer</summary>

Because the chi-squared(1) reference is what Kupiec's test *is*, it's the
standard, universally-recognised convention every regulator and every other
implementation uses, and the asymptotic approximation does converge to the
true size as the expected exception count grows (the equity engine's
`docs/VALIDATION.md` shows it improving toward nominal as `n_obs` grows past
the Basel window). Swapping in an exact small-sample reference distribution
would make the numbers non-standard and non-comparable to every other bank's
Kupiec p-value, trading one well-understood convention for a nonstandard one
that nobody else uses. The right fix here is documentation plus an exact test
that pins the effect
(`test_kupiec_asymptotic_chi2_reference_is_oversized_at_the_regulatory_window`
in both engines), not a different statistic.

</details>

**Q7. Does the chi-squared(1) oversizing get better or worse as the backtesting window grows, and what drives it?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

It's driven by the *expected exception count* (`alpha * n_obs`), not `n_obs`
in isolation, the equity `VALIDATION.md` verifies this at several `(n_obs, alpha)`
pairs. At the Basel window (`n_obs=250`, expected count 2.5) the exact
rejection rate is ≈9.5%; the equity engine's docs record it falling to ≈7.1%
at `n_obs=500` (expected count 5), and the FX engine's docs record ≈5.5% at
`n_obs=1000` (expected count 10), close to nominal. So a longer backtest
window with the same `alpha` tightens the asymptotic approximation, it's a
real remedy, just not one Basel's 250-day standard offers.

</details>

**Q8. Christoffersen's independence and conditional-coverage tests also use a chi-squared reference (1 df and 2 df respectively). Do they inherit the same small-sample caveat?**

<sub>Tags: `L3`, `es-backtesting`</sub>

<details><summary>Answer</summary>

Yes, both `docs/VALIDATION.md` files note that Christoffersen's independence
and CC tests "share the same asymptotic chi2 machinery and are subject to the
same class of caveat at low exception counts." The exact size wasn't computed
for Christoffersen the way it was for Kupiec, though: Kupiec's rejection
region depends on a single sufficient statistic (the exception count), which
makes an exact binomial calculation tractable, while Christoffersen's LR
depends on the full transition-count triple (`n00, n01, n10, n11`) and would
need Monte Carlo to pin down exactly.

</details>

**Q9. Walk through what the Basel traffic-light zones mean and what a bank actually pays for landing in each one.**

<sub>Tags: `L3`, `es-backtesting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`basel_traffic_light(n_obs=250, n_exceptions=x, alpha=0.01)` maps the 250-day,
99% VaR exception count to green (0–4 exceptions), yellow (5–9), or red (10+).
Green carries no penalty: the capital multiplier stays at the base 3.0. Yellow
adds a graduated plus factor (`_YELLOW_ADDON`: 0.40 at 5 exceptions, then
0.50, 0.65, 0.75, up to 0.85 at 9), so the multiplier climbs from 3.40 to
3.85: the model isn't presumed broken yet, but capital charges rise as a
precaution. Red sets the plus factor to 1.0 (multiplier 4.0) and carries a
presumption of a flawed model; at that point the regulator expects the bank to
explain or fix the model, not just pay more. The function works for any window
and alpha: with `F(x) = P(X <= x)`, `X ~ Binomial(n_obs, alpha)`, the zone is
green if `F < 0.95`, yellow if `F < 0.9999`, red otherwise, and the yellow
plus factor is read at the 250-day 99% count with the same cumulative
probability. On 250 days at 99% this reproduces the regulatory table.

Source: `python/equity/03-var-es-engine/src/eq_var/backtesting.py` (`basel_traffic_light`, `_YELLOW_ADDON`).

</details>

**Q10. Where do the 4-exception and 9-exception zone boundaries actually come from: are they arbitrary round numbers?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

No, they're calibrated off the exact `Binomial(250, 0.01)` distribution of
exception counts a correctly-specified model would produce: green is the
region where the cumulative probability is below 95%, red where it reaches
99.99%. Green (≤4 exceptions) covers about 89% of outcomes for a correct
model, and red (10+) has only about 0.03% probability.
`basel_zone_probabilities` exposes the exact PMF/CDF per exception count so
this can be checked directly rather than taken on faith; both test suites pin
the green cumulative probability `binom.cdf(4) = 0.8922`. Because the zones
are defined by these cumulative probabilities, the same rule gives the right
cutoffs for a 500-day window or a 97.5% VaR.

Source: `python/equity/03-var-es-engine/src/eq_var/backtesting.py` (`basel_zone_probabilities`).

</details>

**Q11. A point-estimate VaR or ES number with no error bar: what's actually wrong with reporting just that one number?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

It invites false precision. A Monte Carlo VaR of "$1.212m" looks exact, but
it's a statistic computed from a finite number of simulated (or historical)
scenarios and carries real sampling uncertainty, two runs with different
seeds, or two historical windows, will disagree by an amount that depends on
`alpha`, the sample size, and the tail shape. Reporting the point estimate
alone hides whether a day-over-day VaR move of a few percent is a real risk
change or just noise in the estimator. That's exactly why both engines compute
a standard error alongside the point estimate (`se_var` field on
`MonteCarloVaRResult`, computed by `var_standard_error_bootstrap`) and why the
convergence tests in `docs/VALIDATION.md` are stated "within 3 SE" rather than
as bare tolerances.

</details>

**Q12. Describe the KDE-based standard-error method in `fx_var/monte_carlo_var.py`. What does it compute, and what role does it play?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

`var_standard_error` (FX) and `var_standard_error_kde` (equity) use the
classical asymptotic order-statistic formula `SE = sqrt(alpha*(1-alpha)/n) / f_hat(q)`,
where the denominator is an estimate of the P&L density *at the reported VaR
quantile itself*: a low density there means the quantile is on a flat part of
the distribution, and small sampling noise in the tail count translates into
large noise in the loss level. Both estimate `f_hat` with a Gaussian kernel
density using Silverman's bandwidth `h = 0.9 min(sd, IQR/1.34) n^(-1/5)`,
evaluated at the same order-statistic quantile the engine reports. In every
engine (Python, C++ and Rust) this is a diagnostic only: the reported `se_var`
is the bootstrap SE. The method needs *some* estimate of the local density,
which is the source of its known weakness (Q13).

</details>

**Q13. What exactly is that weakness, and how big is it?**

<sub>Tags: `L3`, `es-backtesting`, `code-reading`</sub>

<details><summary>Answer</summary>

A fixed-bandwidth kernel is tuned to fit the *bulk* of the P&L distribution,
not the sparse tail region where VaR/ES live, so the density estimate at the
quantile is biased at deep tails and at small scenario counts. That is a
directional bias, not extra noise, and it hits where a desk is most likely to
be operating: deep confidence levels, or MC runs sped up with fewer paths.
This is why the engines report the bootstrap SE and keep the KDE number only
as a cross-check (the `fx_var/monte_carlo_var.py` module docstring: "the
bootstrap is the primary figure").

</details>

**Q14. How does the bootstrap standard-error estimator sidestep that problem, and what does it cost instead?**

<sub>Tags: `L3`, `es-backtesting`</sub>

<details><summary>Answer</summary>

`var_standard_error_bootstrap` resamples the P&L vector with replacement
`n_boot` times and applies the *exact same order-statistic VaR rule* to each
resample, then takes the standard deviation of the resulting VaR estimates: no
density estimate, no bandwidth choice, distribution-free. The cost is
trial-to-trial variance in the SE estimate itself: because it's a Monte Carlo
estimate built on top of an already-Monte-Carlo VaR, its own precision depends
on `n_boot` (default 500). Resamples are drawn in chunks, so memory stays O(n)
and no `n_boot x n` index matrix is built. The bootstrap of a sample quantile
is also lumpy (it can only land on observed values), which is fine for a
standard error but is a reason not to read too much into the third significant
figure.

</details>

**Q15. `var_standard_error_bootstrap`'s docstring makes a point of computing the tail rank `idx` once, outside the resampling loop, rather than recomputing per resample. Why does that matter for correctness, not just speed?**

<sub>Tags: `L3`, `es-backtesting`, `code-reading`</sub>

<details><summary>Answer</summary>

Because the resample rule must apply the *exact same* quantile/rank convention
as the original point estimate. Here the uniform-weight tail rank depends only
on `(n, alpha)`, not on the data values, so it's identical for the original
sample and every resample of the same size, computed once via the shared
`_tail()` rank rule and reused. If the bootstrap instead used a different
quantile convention (a different interpolation rule, a different rank formula)
than `empirical_var` used for the point estimate, the resulting standard
deviation would be describing the sampling variability of a *different*
statistic than the one being reported, the SE and the number it's supposed to
bracket would no longer be talking about the same quantity, silently
invalidating any "point estimate ± k*SE" statement built from them.

</details>

**Q16. If a desk is running a 99% VaR on 50,000 daily-P&L scenarios, which standard error should it quote?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

The bootstrap SE, which is what `monte_carlo_var` reports as `se_var` in every
engine. At 99% on 50,000 scenarios the KDE diagnostic and the bootstrap
usually agree closely, and a large gap between them is itself a signal worth
looking at (a thin or lumpy tail). The KDE number gets worse at deeper
confidence levels (99.5%, 99.9%, stress alphas) or when scenario counts are
cut for speed, which is why it is never the figure reported.

Source: `python/equity/03-var-es-engine/src/eq_var/monte_carlo_var.py` (`se_var`).

</details>

**Q17. Same question, but the desk wants ES97.5 off a 250-day historical window rather than an MC VaR: does the SE story change?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

The KDE bias doesn't apply to a purely historical ES (there's no
density-at-a-quantile step there), but the small-sample problem gets *worse*,
not better: ES averages over the whole `alpha` tail rather than reading off a
single order statistic, so under fat tails its estimation error exceeds VaR's
at the same alpha, because the average is dominated by a few extreme points
(`es_standard_error_bootstrap`'s docstring makes this point explicitly). A
250-day window at `alpha=0.025` has only about 6 points in the tail being
averaged, `docs/VALIDATION.md` calls this out directly: "a 250-day ES₉₇.₅
averages ~6 points, quote it with error bars." The equity engine's bootstrap
SE test (`test_es_se_larger_for_smaller_alpha`) formalizes that ES's bootstrap
SE is larger than VaR's bootstrap SE on the same sample.

</details>

**Q18. Why did FRTB replace 99% VaR with 97.5% ES as the headline market-risk capital measure, given ES's SE is worse at a fixed alpha?**

<sub>Tags: `L3`, `es-backtesting`</sub>

<details><summary>Answer</summary>

Two separate properties are in play and FRTB is trading one for the other. ES
is coherent (subadditive) where VaR in general is not (Q2), and ES captures
tail *severity*, how bad losses get beyond the threshold, where VaR by
construction is blind to everything past its own cutoff (Q1). Moving the
confidence level down to 97.5% (from 99%) is precisely what keeps the *tail
sample size* reasonable when averaging rather than reading a single quantile,
more like ~6 points on a 250-day window as noted above, rather than the ~2.5
expected exceptions VaR99 would have. So FRTB accepts a somewhat noisier
per-day statistic (ES vs VaR at a fixed alpha) in exchange for a measure that
behaves better under aggregation and actually reflects tail severity, and
picks the alpha level, in part, to keep that noisiness manageable.

</details>

**Q19. What does the Acerbi–Szekely Z2 test add on top of Kupiec/Christoffersen, and why was it chosen for backtesting ES specifically?**

<sub>Tags: `L2`, `es-backtesting`</sub>

<details><summary>Answer</summary>

Kupiec and Christoffersen only ever look at the exception *indicator*
(breached or not); they're VaR-only tests and say nothing about whether the ES
number itself was a good estimate of tail severity. `acerbi_szekely_z2`
instead sums, over exception days, the ratio of realised P&L to the model's
ex-ante ES, `Z2 = sum_t pnl_t I_t / (T alpha ES_t) + 1` with `I_t = 1{pnl_t < -VaR_t}`,
in the paper's sign: `E[Z2] = 0` under a correct model, and a negative Z2
means realised tail losses are running worse than the model's ES claimed (ES
understated). It uses every exception's magnitude relative to the ex-ante ES,
not just a count, and needs no normality assumption for the statistic itself.

</details>

**Q20. How does `acerbi_szekely_z2` turn Z2 into a p-value, and how does that compare with Kupiec's p-value?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

Z2 has no closed-form null distribution, so the function simulates one. It
draws `n_sims` (default 10,000) P&L paths under the null with the same `VaR_t`
and `ES_t` forecasts, recomputes Z2 on each, and returns the one-sided p-value
`P(Z2_sim <= Z2_obs)` alongside `z2` and `n_exceptions`. The default null is
`pnl_t ~ Normal(0, sigma_t)` with `sigma_t` chosen so the normal ES equals
`ES_t`; passing `null_sampler` draws from the model's own distribution
instead, which is the better choice when the model is not normal. Unlike
Kupiec's LR-to-chi2 p-value, this is a Monte Carlo p-value: exact in the limit
of many simulations, and only as good as the null model it simulates. The
fixed -0.70 rule of thumb from the paper is no longer used.

Source: `python/equity/03-var-es-engine/src/eq_var/backtesting.py` (`acerbi_szekely_z2`).

</details>

**Q21. What does Christoffersen's independence test catch that Kupiec's POF test misses entirely?**

<sub>Tags: `L2`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

Kupiec only checks the *total* exception count against the expected rate, ten
exceptions scattered randomly across 250 days and ten exceptions clustered in
one bad fortnight score identically under Kupiec.
`christoffersen_independence` instead builds a first-order Markov transition
count (`n00, n01, n10, n11`, exception yesterday vs today) and compares the
Markov likelihood against the i.i.d. one; clustered exceptions inflate `n11`
(exception-follows-exception) and the LR rejects. This matters operationally
because clustering is the signature of a model that's slow to react to a
genuine regime change (e.g. vol clustering under GARCH) rather than one that's
simply miscalibrated on average, `exception_cluster_table`'s docstring notes
gaps between exceptions should be roughly geometric with mean ~100 days under
a correct i.i.d. 99% model, and a run of small gaps is the visual signature;
that gap reading is a heuristic, the LR test itself uses only the day-to-day
transition counts.

Source: `python/equity/03-var-es-engine/src/eq_var/backtesting.py` (`christoffersen_independence`).

</details>

**Q22. Walk through what `rolling_var_backtest` actually simulates, and why it uses a walk-forward design rather than one static fit.**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

It re-estimates VaR out of sample: for each day `t` past the initial `window`,
it calls the supplied `var_fn` on only the trailing `window` days of P&L
history (`p[t-window:t]`), producing one VaR forecast per day, then compares
each forecast against the *next* day's realised P&L via `exceptions_from_pnl`.
This mimics how a desk operates: the model never gets to see the day it's
forecasting, which is the only way a backtest can measure predictive coverage
rather than in-sample fit. `BacktestResult.summary()` then runs Kupiec, both
Christoffersen tests, and (at `alpha=0.01`) rescales the observed exception
count to a 250-day-equivalent to report the Basel zone even when the actual
window isn't exactly 250 days.

Source: `python/equity/03-var-es-engine/src/eq_var/backtesting.py` (`rolling_var_backtest`).

</details>

**Q23. The documented equity engine backtest (500-day rolling, 99%, synthetic GARCH(1,1)-t(5) data) shows parametric-normal far outside the green zone while filtered historical simulation (FHS) lands green. What's driving that gap?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

**[Synthetic data, one seed.]** Over 500 forecast days, parametric-normal
produced 14 exceptions against an expected 5 (Kupiec p = 0.0009,
conditional-coverage p = 0.00), badly miscalibrated, landing yellow over the
full 500-day window under the any-window traffic light; on the first 250
forecast days alone it drew 13 exceptions, red (multiplier 4.0). Per-factor
FHS produced 4 exceptions, all tests p > 0.05, green (3.0). The difference is
that FHS conditions its VaR on the *current* volatility regime, while the
unconditional parametric-normal model doesn't adapt to the data's volatility
clustering, so its exceptions pile up in the high-vol periods because its VaR
lags the regime. This is the case Christoffersen's test (Q21) is built to
catch, and the traffic-light system (Q9) is built to price.

Source: `python/equity/03-var-es-engine/docs/VALIDATION.md` §4 (the walk-forward table).

</details>

**Q24. `es_standard_error_bootstrap` and `var_standard_error_bootstrap` both resample with replacement and recompute the estimator on each resample: are they otherwise the same procedure?**

<sub>Tags: `L3`, `es-backtesting`</sub>

<details><summary>Answer</summary>

Structurally yes, both draw `n_boot` resamples of the same size as the
original P&L array (with replacement), recompute the point-estimate function
(`expected_shortfall` or the order-statistic VaR rule) on each, and report the
standard deviation across resamples (`ddof=1`). They differ in what's being
resampled *around*: VaR's bootstrap SE only has to track one order statistic
per resample, while ES's has to re-average the whole tail-sum (`sum_{i<m} L(i) + (k - (m-1)) L(m)`)
each time, which is part of why ES's bootstrap SE comes out larger than VaR's
bootstrap SE at the same alpha and sample (Q17), more of the resample's
randomness feeds into the final number.

</details>

**Q25. Both engines validate their Kupiec implementation against hand-computed numbers rather than only checking asymptotic behaviour. Why does that matter here specifically?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

Because the finding in Q5–Q7 is about the *reference distribution*, not the LR
statistic's arithmetic, so the test suite needs to separately pin down that
the LR formula itself is exactly right (equity: `Kupiec LR vs independent hand computation, 5 (T,x) pairs; LR(250,0) = -2*250*ln(0.99)`
to 1e-10; FX: `Kupiec LR(n=250, x=5, p=1%) = 1.9568` to 1e-12 vs hand formula)
before asking any question about how well chi-squared(1) approximates its null
distribution. Getting the LR formula numerically exact and then separately
characterizing the chi-squared reference's known small-sample bias are two
different validation claims, and conflating them would make it unclear whether
a documented discrepancy is a code bug or the well-known
asymptotic-approximation property it actually is.

Source: `cpp/equity-var-engine/tests/golden_vectors.hpp` and `cpp/fx-var-engine/tests/golden_vectors.hpp` (the 1.9568 vector).

</details>

**Q26. What happens at the Kupiec test's degenerate boundaries, zero exceptions, or every day an exception: and why does that need special handling rather than just erroring?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Zero exceptions is a completely legitimate outcome for a well-calibrated model
on a short window (its probability is exactly `(1-alpha)^n_obs`, e.g.
`binom.pmf(0, 250, 0.01) ≈ 8.1%`), so the LR formula has to evaluate cleanly
there rather than raising on a `log(0)` term. `kupiec_pof` handles this via
the `0 * ln 0 = 0` convention (`_ll` only adds the `x*log(p)` term when `x > 0`,
and the `t-x` term only when `t-x > 0`), and it's exercised directly by
`test_zero_exceptions_known_value` and the "x=0 or x=T" edge-case row in both
`docs/VALIDATION.md` tables, the same convention appears in
`christoffersen_independence`'s `_xlogy` helper for its own degenerate
transition counts.

</details>

**Q27. Suppose a Kupiec test on a 250-day, 99% VaR backtest comes back with p=0.04, reject at 5%. Given everything above, how should a desk actually read that number?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

With caution rather than a flat "the model failed." Because the true rejection
rate of a nominally-5%-size Kupiec test at this exact window is ≈9.5% (Q5),
roughly double the nominal rate, a p-value just under 0.05 is landing in a
region where a correctly-calibrated model gets flagged almost twice as
frequently as the asymptotic reference suggests it should. That doesn't make
the test useless (it's still the industry-standard, regulator-recognised
statistic, per Q6), but it means a marginal reject at this specific
window/alpha combination is weaker evidence of a broken model than the same
p-value would be at a longer window (Q7), and it's exactly the kind of caveat
that belongs alongside the number, not baked into a different, nonstandard
test.

Source: `python/equity/03-var-es-engine/tests/test_backtesting.py` and `python/equity/03-var-es-engine/docs/DESK_GUIDE.md` §3 (the 9.5 % rejection rate).

</details>

**Q28. What parts of FRTB IMA does `eq_var.frtb_ima` implement, and what is still missing?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

The liquidity-horizon ES cascade (`liquidity_adjusted_es`), stressed ES
scaling, IMCC with rho = 0.5, capital with the backtesting plus factor, the
desk backtest (12 exceptions at 99%, 30 at 97.5%) and the P&L attribution test
with Spearman and Kolmogorov-Smirnov zones. Missing: NMRF/SES (taken as an
input), the default risk charge and risk-factor eligibility. The MAR paragraph
numbers in the docstring are from the team's reading and have not been
re-checked against the bis.org text.

</details>

**Q29. Why fit a generalised Pareto tail (`eq_var.evt`) instead of reading the far quantile from the data?**

<sub>Tags: `L3`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

At 99.9% a year of data has no observations in the tail, so the empirical
quantile is the worst day or an interpolation off it. Peaks over threshold
fits a GPD to losses above a high threshold (`fit_pot`) and extrapolates with
`pot_var` and `pot_es` (McNeil and Frey 2000). The tests check exact GPD
quantiles and integrated ES, MLE convergence on simulated GPD data, and that a
Student-t with 4 degrees of freedom gives a tail index near 1/4. The answer
still depends on the threshold choice; `mean_excess` helps pick it.

</details>

**Q30. What does `eq_var.stressed_var.find_stress_window` do, and how is it tested?**

<sub>Tags: `L2`, `es-backtesting`, `numeric`</sub>

<details><summary>Answer</summary>

Basel 2.5 stressed VaR uses the 250-day window with the worst VaR in the history. The function slides a window over the P&L history and returns the worst; `basel25_capital` combines VaR and stressed VaR with their multipliers. The test plants a crisis in synthetic data and checks the window lands on it, then checks the order statistic by hand.

</details>


## Set 8: Fixed Income Pricing & Risk

**Q1. Why does a bond's price move inversely with yield, and where does that fall out of the code?**

<sub>Tags: `L2`, `fixed-income`, `code-reading`</sub>

<details><summary>Answer</summary>

A bond's price is the sum of its future cashflows discounted at the yield: in
`fi_rates/bond.py::price_from_ytm`, dirty price is `sum(cf_i * (1 + y/m)^(-exp_i))`.
Raising `y` raises every discount factor's exponent base, shrinking every
`(1+y/m)^(-exp_i)` term, so price falls monotonically as yield rises, this is
unit-tested as a structural invariant (price monotonicity in yield) rather
than just checked pointwise. Because price is strictly monotone in yield,
`price_from_ytm` can be inverted by `ytm_from_price` with a single-bracket
Brent search: the price-yield curve crosses any target once, so a bracket with
a sign change holds exactly one root. (The curve is also convex, but Brent
does not need that.)

</details>

**Q2. What does Macaulay duration actually measure, as implemented here?**

<sub>Tags: `L1`, `fixed-income`, `code-reading`</sub>

<details><summary>Answer</summary>

`macaulay_duration` in `fi_rates/risk/__init__.py` computes `D_mac = (1/P) * sum(t_i * cf_i * (1+y/m)^(-exp_i))`,
the present-value-weighted average time (in years) until each cashflow is
received. It is a time measure, not a price-sensitivity measure by itself: the
docstring notes it equals the weighted average cashflow time, and the test
suite pins `macaulay_duration(zero_coupon_bond) == maturity` exactly, since a
ZCB has only one cashflow and its weighted-average time is trivially its own
maturity.

</details>

**Q3. How does modified duration differ from Macaulay duration, and what does it measure?**

<sub>Tags: `L2`, `fixed-income`, `code-reading`</sub>

<details><summary>Answer</summary>

`modified_duration` is `macaulay_duration / (1 + y/m)`, the code implements
this as a one-line division in `risk.py`. Macaulay duration is a time (years);
modified duration is a price sensitivity: the docstring states `D_mod = -(1/P) dP/dy`,
the percentage price change per unit change in yield. VALIDATION.md pins the
identity `D_mod == D_mac/(1+y/m)` to ≤1e-14 and separately verifies `D_mod`
against central finite differences of the actual pricer
(`numerical_modified_duration`) to ≤1e-8 relative, so the analytic formula is
cross-checked against brute-force bumping, not just algebra.

</details>

**Q4. Why isn't duration alone enough to estimate a price move for a large yield shock?**

<sub>Tags: `L2`, `fixed-income`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Duration is a first-order (linear) Taylor term; the price-yield relationship
is convex, not linear, so a straight-line duration estimate systematically
deviates from the true curve as the shock size grows.
`pnl_approximation_table` in `risk.py` builds exactly this comparison,
`duration_only = -D_mod * P * dy` versus `full_repricing = price_from_ytm(ytm+dy) - price_from_ytm(ytm)`,
across a shock grid from -200bp to +200bp, and VALIDATION.md §4.4 shows the
duration-only error growing from -0.401 at -100bp to -1.664 at -200bp on a 10y
government bond (-0.374 and -1.448 on the upside); adding convexity cuts the -100bp
error to -0.014.

</details>

**Q5. What is convexity, and how is it computed here?**

<sub>Tags: `L1`, `fixed-income`, `code-reading`</sub>

<details><summary>Answer</summary>

`convexity` in `risk.py` computes the analytic second derivative `C = (1/P) d2P/dy2 = (1/P) sum(cf_i * n_i*(n_i+1)/m^2 * (1+y/m)^(-n_i-2))`,
where `n_i` are the period exponents from `_period_exponents`. It is the
curvature of the price-yield relationship, the second-order correction that a
linear duration estimate misses. Like modified duration, it's cross-checked
against `numerical_convexity` (a central second difference of the pricer) to
≤1e-6 relative per VALIDATION.md.

</details>

**Q6. Why does duration alone underestimate price gains and overestimate price losses for large yield moves?**

<sub>Tags: `L2`, `fixed-income`, `numeric`</sub>

<details><summary>Answer</summary>

For a bond with positive convexity, the true price-yield curve lies above its
tangent line at every point away from the tangent point, a straight line from
the tangent will always sit below the actual (convex) curve.
`pnl_approximation_table` demonstrates this concretely on a 10y bond: at -200bp
the duration-only estimate misses +17.747 of full repricing by -1.664
(understating the gain), and at +200bp it misses -14.635 by -1.448
(overstating the loss, i.e. predicting a bigger drop than actually happens).
Adding the 0.5·C·P·dy² convexity term in `duration_convexity` cuts the 100bp
error by roughly 30x, and this is enforced as an inequality (not a
magic-number check) across the whole shock grid so the sign relationship,
dur+conv always closer than dur-only for a positive-convexity bond, is what's
actually tested.

Source: `python/equity/04-fixed-income/docs/VALIDATION.md` (the ±200bp rows) and `python/equity/04-fixed-income/src/fi_rates/risk/__init__.py` (`pnl_approximation_table`).

</details>

**Q7. What is DV01, and how does the project compute it two different ways?**

<sub>Tags: `L1`, `fixed-income`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

DV01 ("dollar value of a basis point") is the price change for a 1bp yield
move, in currency units. The analytic version, `dv01()` in `risk.py`, is
`D_mod * P * 1e-4`, a closed-form scaling of modified duration. The
curve-based ("effective") version, `dv01_curve()`, instead bumps every pillar
zero rate in `DiscountCurve` by ±1bp via `curve.bumped_parallel(bump)` and
fully reprices: `DV01_eff = (P(-h) - P(+h)) / 2`. METHODOLOGY.md §4 notes they
are not made to agree exactly, they differ by the Jacobian `dy/dz ≈ (1 + y/m)`
between periodic-yield space and continuous-zero space, and the test suite
asserts precisely that relationship rather than treating the two numbers as
interchangeable.

</details>

**Q8. Why is DV01 the number a rates desk actually uses day to day, rather than duration or convexity alone?**

<sub>Tags: `L2`, `fixed-income`</sub>

<details><summary>Answer</summary>

DESK_GUIDE.md's 07:15 EOD risk step runs `portfolio_risk`, which reports DV01
per position and in aggregate; DV01 is what traders check against limits and
what risk managers aggregate, because, unlike modified duration or YTM, it is
a plain currency amount that sums additively across positions of different
price levels, coupons and maturities. VALIDATION.md §4.7 makes the point
sharply: for a duration-neutral long/short book with net market value ≈ 0,
market-value-weighted duration and convexity become undefined `0/0`
(`portfolio_risk` emits a `ZeroNetValueWarning` and reports `NaN` for those
columns), but `dv01`, because it's a plain sum, not a weighted average,
remains exact throughout and is, per the docs, "the correct aggregate for such
a book."

</details>

**Q9. What day-count convention does the curve engine use for mapping dates to time, and does it match the portfolio's stated equity convention?**

<sub>Tags: `L2`, `fixed-income`, `code-reading`</sub>

<details><summary>Answer</summary>

`fi_rates/bond.py::curve_time` maps a calendar date to curve time as
`year_fraction(settlement, date, "ACT/365F")`, actual days divided by 365,
fixed, matching the ACT/365F convention that `CONVENTIONS.md` states as the
portfolio-wide default for equity-style day counting. `DiscountCurve` pillars
are likewise stated (in `curve.py`'s module docstring) to live "in years
(ACT/365F from the valuation date when built from dates)". This is a separate
choice from the bond's own accrual day count (see Q10), ACT/365F is
specifically the date→time mapping onto the curve, not the coupon accrual
convention.

</details>

**Q10. Why do day-count conventions matter for getting accrued interest right, and what conventions does this project support?**

<sub>Tags: `L2`, `fixed-income`, `code-reading`</sub>

<details><summary>Answer</summary>

`fi_rates/daycount.py` implements four conventions, `ACT/365F`, `ACT/360`
(money-market, used for deposits/FRAs), `30/360US` (the default bond accrual
convention, without the end-of-February special rule), and `ACT/ACT-ISDA`
(calendar-year-split, used for Treasuries in this codebase).
`accrued_interest()` computes `bond.face * bond.coupon * year_fraction(period_start, settlement, bond.daycount)`,
so a mismatched convention directly misprices accrued interest and therefore
the dirty price. The module docstring documents this as a real simplification:
`30/360US` here omits the Feb-EOM rule, and the code uses ACT/ACT **ISDA**
rather than the period-based ACT/ACT **ICMA** that actual US Treasuries use,
assumption 5 in METHODOLOGY.md states explicitly that Treasury stub-period
accruals will differ by a day or two from production as a result.

</details>

**Q11. How is the yield curve actually built in this project, bootstrapped from market instruments, or fit globally?**

<sub>Tags: `L2`, `fixed-income`, `numeric`</sub>

<details><summary>Answer</summary>

`fi_rates/bootstrap.py::bootstrap_curve` performs a **sequential bootstrap**:
instruments (deposits, FRAs, par swaps) are sorted by pillar maturity, and
each new pillar's discount factor is solved with Brent's method (`brentq`,
bracket `(1e-8, 20)` on the DF) so that instrument reprices exactly off the
partially built curve, deposits close-form as `P(T) = 1/(1+r·T)`, FRAs as
`P(T2) = P(T1)/(1+r·(T2-T1))`, and par swaps via the pillar condition `r·Σα_i P(t_i) + P(T) = 1`.
`bootstrap_bond_curve` does the analogous sequential scheme off coupon-bond
dirty prices. METHODOLOGY.md §1 explicitly rejects a global parametric fit
(Nelson–Siegel/Svensson) for this purpose, because a trading desk needs to
reprice its own hedge instruments exactly, NS/Svensson leaves 1-10bp
residuals, which "creates phantom P&L and phantom DV01 on every mark."

</details>

**Q12. What interpolation choices does `DiscountCurve` offer, and why does the choice matter beyond just smoothness?**

<sub>Tags: `L2`, `fixed-income`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Three schemes share one code path (`curve.py`): `loglinear_df` (default,
linear in ln P(t), equivalent to piecewise-constant instantaneous forwards),
`linear_zero` (linear in the zero rate), and `pchip_zero` (monotone cubic on
zeros). The choice is exposed as a constructor argument rather than hard-coded
because it's a risk decision, not plumbing: VALIDATION.md §4.1 shows actual
computed forward rates around the 10y/15y/20y pillars where `linear_zero`
produces a sawtooth (a 22.4bp jump down at each pillar as the forward resets),
`loglinear_df` jumps only *at* pillars (up to 29bp, but flat and unbiased
between them), and `pchip_zero` is continuous (max ~2bp jump from discrete
differencing) at the cost of being non-local.

</details>

**Q13. Why does locality of the interpolation scheme matter for hedging, specifically?**

<sub>Tags: `L2`, `fixed-income`, `numeric`</sub>

<details><summary>Answer</summary>

With `loglinear_df` or `linear_zero`, both **local** schemes, bumping one
market quote (say the 10y swap) moves the 10y pillar and, because the
bootstrap is sequential, the later pillars solved from it; earlier pillars do
not move, so a bucketed DV01 report stays clean on the short end and a hedge
trade in the 10y point doesn't leak risk into the 2y bucket. `pchip_zero` is **not**
local: VALIDATION.md §1.1 measures that appending a new pillar under PCHIP
reshapes the monotone cubic over already-solved earlier segments, so
previously-repriced quotes drift by up to 3.6e-06 in rate terms, small, but
enough that a PCHIP-built curve "cannot pass a 1e-10 repricing gate" that
production build controls typically require (DESK_GUIDE.md §5).

</details>

**Q14. What happens when the bootstrap is given a quote that can't be repriced, or when instruments are supplied out of order?**

<sub>Tags: `L2`, `fixed-income`</sub>

<details><summary>Answer</summary>

`bootstrap_curve` raises `ValueError` naming the instrument, the pillar, and
the quote when no discount factor in the admissible bracket `(1e-8, 20)` can
reprice it, e.g. a deposit rate below -1/T, or (for `bootstrap_bond_curve`) a
bond price above the sum of its cashflows' maximum PV. Order independence is a
separate, explicitly tested property: VALIDATION.md §1.1 states bootstrapping
is verified order-independent, shuffled instrument input reproduces identical
pillar discount factors to 1e-15, because `bootstrap_curve` internally sorts
by `pillar` before solving.

</details>

**Q15. What would key-rate duration (or partial DV01) add beyond a single parallel-shift duration number, and does this project implement it?**

<sub>Tags: `L2`, `fixed-income`, `code-reading`</sub>

<details><summary>Answer</summary>

Yes, `fi_rates/keyrates.py` implements key-rate DV01s and durations at
configurable tenors (`DEFAULT_KEY_TENORS = (2, 5, 10, 30)` years) using **triangular
bumps**: `triangle_weights` builds a weight function that peaks at 1 at its
own key tenor and decays linearly to 0 at the neighboring key tenors,
flat-extending at the ends, forming a partition of unity across every curve
pillar. `key_rate_dv01s` then central-differences full revaluation under each
triangular pillar-bump. A single parallel duration/DV01 number only measures
sensitivity to a uniform shift of the whole curve; the KRD ladder decomposes
that into sensitivity at each point on the curve, which matters whenever the
curve doesn't move in parallel (see Q17).

</details>

**Q16. How does the KRD ladder relate mathematically to the single parallel DV01 number, and how exactly do they reconcile?**

<sub>Tags: `L2`, `fixed-income`, `numeric`</sub>

<details><summary>Answer</summary>

Because the triangular weights form a partition of unity (unit-tested to
1e-14), summing the key-rate bumps reproduces exactly a 1bp parallel bump, but
the *sum of the resulting KRDV01s* only matches the parallel DV01 "up to
tolerance," not exactly, for two documented reasons (VALIDATION.md §3): (1)
finite-difference cross-gamma, since PV isn't linear in the pillar zeros, and
(2) non-local interpolation reshaping neighboring buckets under PCHIP. On the
sample portfolio (DV01 ≈ 980.3/bp), the measured relative residual is 5.3e-08
under `loglinear_df` versus 5.5e-07 under `pchip_zero`, three orders of
magnitude apart, which the docs cite as a reason production risk systems
conventionally use a local interpolator.

</details>

**Q17. Why does the desk guide say hedging the total DV01 with one instrument "leaves the book flat to parallel moves but fully exposed to twists"?**

<sub>Tags: `L2`, `fixed-income`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

VALIDATION.md §4.3 quantifies this directly with a
full-revaluation-vs-duration-and-convexity comparison: a `steepener_50bp`
scenario produces a true P&L of -4,345 against a duration+convexity-proxy
estimate of -12,296, a 183% error, "duration proxy useless" per the doc's own
annotation, because a single duration number compresses the whole curve into
one factor and cannot see that the curve twisted rather than shifted in
parallel. `krd_report` in `keyrates.py` is the fix: DESK_GUIDE.md's sample
book shows a 2y/5y/10y/30y KRD ladder (154.3/278.0/368.4/179.5 DV01) that sums
to the parallel 980.2 DV01 to 5e-8 relative, and hedging *that* ladder, not
the aggregate number, is what actually flattens exposure to a steepener or
flattener.

</details>

**Q18. Why does this project use a single self-discounting curve rather than the post-2008 multi-curve/OIS framework, and what breaks as a result?**

<sub>Tags: `L2`, `fixed-income`</sub>

<details><summary>Answer</summary>

METHODOLOGY.md's assumptions register lists this as "the big one": one curve
both projects forwards and discounts, so the par-swap condition assumes the
floating leg is worth par on that same curve. This has not been production
standard since 2008: desks discount on OIS (SOFR/ESTR) while projecting
forwards off separate tenor curves. What breaks: with a wide OIS-IBOR-style
basis, single-curve swap PVs are wrong by roughly basis × annuity (tens of bp
of upfront in 2008-2012), the FRN price-equals-par identity breaks by the
discounting/projection basis, and DV01 that should split across curves
doesn't. The doc notes the architecture (curve objects + root-solved
bootstrap) extends to multi-curve by adding a second curve and re-deriving the
par condition, but it's left out to keep the project reviewable.

</details>

**Q19. How does the curve behave beyond its last pillar, and what's the documented risk?**

<sub>Tags: `L2`, `fixed-income`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Both `curve.py` and VALIDATION.md §4.2 are explicit: beyond the last pillar,
the zero rate is held flat, and **every query in that region emits an
`ExtrapolationWarning`**, not suppressible by configuration, so callers must
handle it consciously (DESK_GUIDE.md lists this as a control: warnings are
treated as errors in the CI test profile except where explicitly expected).
The risk is concrete, not academic: discounting a 40y cashflow off a
30y-last-pillar curve embeds an arbitrary flatness assumption directly into
PV, and an error of 10bp in the 40y zero rate moves that cashflow's PV by
about 4% (40 years × 0.10%); VALIDATION.md §4.2's figure of about 0.4%
corresponds to a 1bp error. Either is material for pension/insurance
liabilities.

</details>

**Q20. How does the project handle zero or negative rates: is that an edge case that's broken, or supported?**

<sub>Tags: `L2`, `fixed-income`</sub>

<details><summary>Answer</summary>

It's an explicitly supported, unit-tested path, not an error. `DiscountCurve`
accepts discount factors greater than 1 (the docstring notes DFs "may exceed 1
for negative-rate curves"), and `price_from_ytm` only requires `1 + ytm/frequency > 0`,
so negative periodic yields work as long as that inequality holds,
METHODOLOGY.md §3 states this is unit-tested, including that "negative yields
work for `1 + y/m > 0`." VALIDATION.md §4.6 further lists "negative-rate
curves (DFs > 1)" among the "degenerate but valid inputs" tested as *working*
paths, alongside the four curve variants (upward, inverted, flat, negative)
that the bootstrap repricing tests in VALIDATION §1.1 are run against.

</details>

**Q21. What happens at the extremes of yield, very high or deeply negative, and does the pricer hold up?**

<sub>Tags: `L2`, `fixed-income`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`test_edge_cases.py::TestExtremeYields`, cited in VALIDATION.md §1.2,
round-trips YTM at 500% and at -1% back through `price_from_ytm` →
`ytm_from_price` → price to 1e-10 relative tolerance. Mechanically this works
because `ytm_from_price`'s Brent bracket in `bond.py` actively expands the
upper bound (doubling `hi` until the price function brackets the target,
capped at 1e4) and fixes the lower bound at `-0.999 * frequency` to stay
inside the `1 + y/m > 0` domain, raising an informative `ValueError` if a
price is unreachable at either bound (e.g. a price at or below the theoretical
maximum-yield floor).

</details>

**Q22. Does the project document behavior on an inverted yield curve, and is it tested the same way as a normal upward curve?**

<sub>Tags: `L2`, `fixed-income`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Yes, VALIDATION.md §1.1 states the bootstrap repricing tests (max error
4.0e-16 against a 1e-10 tolerance) are run "for all four curve variants
(upward, inverted, flat, negative)," and structural invariants in
`test_properties.py` (per VALIDATION §5) pin curve identities and bond
duration/convexity ordering "across yields from -1% to +12%" without
special-casing the curve's shape. DESK_GUIDE.md's 2022 hiking-cycle scenario
(a bear-flattening/inverting move: 2y +370bp vs 10y +235bp) is the concrete
real-world case: the desk-guide lesson is explicitly that short-end key-rate
durations, "usually ignored," dominated the P&L in that episode, and the
duration+convexity Taylor proxy still missed the true loss by 13% (VALIDATION
§4.3) because full revaluation, not a single duration number, was needed to
capture the twist.

</details>

**Q23. What does "negative convexity" mean, and why is it explicitly out of scope here?**

<sub>Tags: `L1`, `fixed-income`, `code-reading`</sub>

<details><summary>Answer</summary>

Assumption 7 in METHODOLOGY.md states the project assumes deterministic rates
with no optionality; VALIDATION.md §4.5 explains that callable bonds and MBS
exhibit *negative* convexity as rates approach the call/refinancing region,
price appreciation caps out and duration actually *shortens* into a rally, the
opposite of the positive-convexity behavior demonstrated in Q6. Nothing in
`risk.py` models this, so applying `duration`/`convexity`/`dv01` to a callable
would overstate upside and misdirect a hedge, the docs note the 30x
Taylor-error improvement from adding convexity (Q6) "would reverse sign" for a
negatively convex instrument. Pricing that correctly requires an
option-adjusted-spread model with an interest-rate lattice or simulation,
which the docs flag as explicitly out of scope, with the recommendation to
"route to an OAS model" (DESK_GUIDE.md §4).

</details>

**Q24. Does the FX fixed-income project add anything to the rates picture, and if so what?**

<sub>Tags: `L2`, `fixed-income`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`python/fx/04-fixed-income` (`fx_rates`) bootstraps one discount curve per
currency (deposits + annual par swaps, same log-linear-DF machinery) and then
prices FX forwards via covered interest parity, `F(T) = S * DF_f(t_s, T) / DF_d(t_s, T)`,
where domestic = quote currency and foreign = base currency per
`CONVENTIONS.md`'s BASE/QUOTE rule. The carry runs from the spot date `t_s`
(T+2, or T+1 for USDCAD), not from today, because the quoted spot rate is
itself for delivery on the spot date; reading spot as a t = 0 price overstates
a 1y EURUSD forward at a 2% rate gap by 2.43 pips
(`tests/test_fx_spot_forwards.py`). Since 2008, market FX forwards
systematically deviate from single-curve CIP, so `fx_rates` adds a
maturity-dependent **cross-currency basis spread** `s(T)` applied to the
foreign curve (`DF_f_adj(T) = DF_f(T) * exp(-s(T)*T)`), backed out from market
forwards or basis-swap quotes via `bootstrap.implied_basis_from_forwards` /
`bootstrap.curve_from_fx_forwards`. METHODOLOGY.md's own worked example shows
the size of the effect: ignoring basis misprices the 5y EURUSD forward by 147
pips, a fictitious -$1.20m PV on a EUR 100m forward at the true market rate.

</details>

**Q25. How does the FX project's risk framework extend DV01 and KRD to a cross-currency setting?**

<sub>Tags: `L2`, `fixed-income`, `numeric`</sub>

<details><summary>Answer</summary>

Per `fx_rates` METHODOLOGY.md §1.5: DV01 is computed per currency as a
central-difference parallel ±1bp bump of *that* currency's zero pillars, and
key-rate duration bumps one pillar at a time, locality of the `loglinear_df`
interpolation is exactly what makes that bump local, the same locality
argument as the equity/rates project (Q13). On top of ordinary rates DV01, the
FX project adds a **basis DV01**, a ±1bp bump of the whole cross-currency
basis spread curve with full curve rebuild, kept as a separate risk factor
from rates DV01 precisely so that interest-rate risk and basis risk are
reported separately rather than conflated, per METHODOLOGY.md §1.3's note that
"all pricing uses the adjusted foreign curve; the pure foreign curve is
retained so that interest rate risk and basis risk are reported as separate
factors."

</details>

**Q26. The November 2026 release added `fi_rates.multicurve`. How does it change the answer to the single-curve question above?**

<sub>Tags: `L2`, `fixed-income`</sub>

<details><summary>Answer</summary>

The single-curve bootstrap is still there and still the default.
`bootstrap_ois` builds a discount curve from OIS quotes and
`bootstrap_projection` builds a tenor curve that reprices swaps while
discounting on OIS. The tests require every OIS and swap quote to reprice to
1e-13 and require the two curves to collapse to one when there is no basis.
`ois_ibor_basis` reports the forward spread between them.

</details>

**Q27. How are caps and swaptions in `fi_rates.rate_options` checked, and why offer Bachelier as well as Black?**

<sub>Tags: `L2`, `fixed-income`</sub>

<details><summary>Answer</summary>

Black-76 needs positive forward and strike; near or below zero rates it fails, so desks quote normal vol. Both models are checked with parity (cap minus floor equals the swap; payer minus receiver equals the annuity times `F - K`), ATM closed forms and an implied-vol round trip.

</details>

**Q28. What are the oracles for the Hull-White model in `fi_rates.hull_white`?**

<sub>Tags: `L1`, `fixed-income`</sub>

<details><summary>Answer</summary>

The model is fitted to today's curve, so zero bonds must reprice it.
Bond-option parity and Jamshidian payer-receiver parity are checked to 1e-13.
An exact Ornstein-Uhlenbeck Monte Carlo with 200,000 paths prices a bond, a
caplet and a swaption within 4 standard errors. Limits close it off: as sigma
goes to 0 a caplet is worth its intrinsic value, and small mean reversion
gives a normal vol close to sigma. `calibrate_hull_white` fits it to a
swaption grid by least squares on prices, with mean reversion either fixed
(the usual desk choice) or fitted.

</details>


## Set 9: Statistical Pairs Trading

**Q1. Two stocks have a 0.92 return correlation. Why isn't that enough to put on a pairs trade?**

<sub>Tags: `L2`, `pairs`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Return correlation only says the two names move together *day to day*; it says
nothing about whether the *level* of one relative to the other is bounded. Two
random walks driven by correlated shocks, with no cointegration, can have high
return correlation while their price levels drift apart without limit, there
is no force pulling the spread back to a mean, so a "pairs trade" on them is
really just a directional bet with extra steps. The equity project's own
selection funnel demonstrates this directly. In the seed-7 panel the three
trap pairs have return correlations of 0.919, 0.921 and 0.923, *higher* than
all four true cointegrated pairs (0.767 to 0.893), yet their Engle-Granger
statistics are −1.45, −1.88 and −2.13 against a 5% critical value of −3.34, so
all three are rejected, while the true pairs score −7.40 to −9.76. The counts
are in pairs `docs/VALIDATION.md` §3 (3 traps through the screen, 3 rejected);
the per-pair figures are printed by
`python/equity/05-pairs-trading/examples/run_pipeline.py`, step 1 (seed 7).

</details>

**Q2. What does cointegration add that correlation doesn't?**

<sub>Tags: `L1`, `pairs`</sub>

<details><summary>Answer</summary>

Cointegration asks a different question: is there a linear combination of two *non-stationary*
(I(1)) price series that is itself *stationary* (mean-reverting)? Correlation
measures co-movement of changes; cointegration certifies that a specific
combination of the *levels*, `y_t − β x_t`, the spread, has a fixed mean and
finite variance it keeps returning to. That stationary combination is exactly
what a pairs trade needs: a tradeable object with a defined equilibrium to bet
on reverting to.

</details>

**Q3. Which cointegration test does this codebase actually implement, read the code, don't assume?**

<sub>Tags: `L2`, `pairs`, `code-reading`</sub>

<details><summary>Answer</summary>

`src/eq_pairs/cointegration.py` implements the **Engle-Granger two-step test
from scratch**: step 1 is an OLS cointegrating regression `y_t = α + β x_t + u_t`
(function `hedge_ratio`/`ols`); step 2 runs an **augmented Dickey-Fuller
test** on the fitted residuals û (function `adf_test`, called from
`engle_granger` with `regression="n", n_series=2`). The FX project
(`fx_pairs/cointegration.py`) mirrors the identical EG-from-scratch design on
log rates. Both are cross-validated against
`statsmodels.tsa.stattools.adfuller` to 1e-8 (`test_fixed_lag_stat_matches`,
`test_autolag_aic_matches`).

</details>

**Q4. What is the null hypothesis of the ADF test run on the EG residuals, and what does rejecting it mean?**

<sub>Tags: `L1`, `pairs`</sub>

<details><summary>Answer</summary>

The ADF regression is `Δû_t = ρ û_{t-1} + Σᵢ φᵢ Δû_{t-i} + ε_t` (no constant,
since the step-1 intercept already makes û mean-zero); the statistic is the
t-value of ρ̂ (`ADFResult.stat`). H0 is **ρ = 0**, i.e. û has a unit root, the
residual spread is itself non-stationary, so there is *no* cointegration.
Rejecting H0 (`EGResult.cointegrated()` returns `adf.reject(level)`, i.e.
`stat < crit[level]`) means the spread is stationary, i.e. the pair is
cointegrated.

</details>

**Q5. Why can't the step-2 ADF statistic be compared against ordinary ADF critical values?**

<sub>Tags: `L2`, `pairs`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Because û isn't an observed series, OLS in step 1 chose (α, β) specifically to
make the residuals look as stationary as possible, which biases the ADF
statistic toward rejection. The correct comparison uses MacKinnon's response
surfaces indexed by N, the number of I(1) series in the cointegrating
regression: N=1 is plain ADF (5% ≈ −2.86), N=2 is the Engle-Granger surface
for a two-variable regression (5% ≈ −3.34, stricter). `cointegration.py`'s own
docstring calls this "the classic mistake," and
`test_spurious_rejection_rate_close_to_size` proves it isn't hypothetical: on
200 replications of independent random walks, the correct N=2 values reject at
the nominal ~5%, while using N=1 values on the same statistics rejects more
than 1.5× as often, false "discoveries" of cointegration that aren't there
(`docs/VALIDATION.md` §3).

</details>

**Q6. How is the hedge ratio estimated, and why an OLS regression rather than just using the two prices directly?**

<sub>Tags: `L2`, `pairs`, `code-reading`</sub>

<details><summary>Answer</summary>

`hedge_ratio(y, x, intercept=True)` runs OLS `y_t = α + β x_t + u_t` on price *levels*
(in dollars for equities, log rates for FX), β is literally the EG step-1
regression coefficient, and the residual `u_hat` is the tradeable spread. β̂
is "super-consistent" under cointegration (converges at rate T, faster than
the usual √T). The intercept is included by default because share price levels
are arbitrary (a stock split changes the level, not the economics), forcing
the line through the origin would misspecify the relation (`METHODOLOGY.md`
§1, "Hedge ratio: intercept, and static vs adaptive").

</details>

**Q7. Why does trading in ratio β:1 (not 1:1 dollars) make the combined position market-neutral, when 1:1 doesn't?**

<sub>Tags: `L2`, `pairs`, `code-reading`</sub>

<details><summary>Answer</summary>

The neutrality here is to the pair's shared stochastic trend, the common shock
both legs carry, not to the market's beta in the CAPM sense: the position's
P&L should not move with that shared trend. If the two legs don't move
one-for-one per dollar (β ≠ 1), which is the generic case, a naive
equal-dollar 1:1 hedge still carries residual exposure to whatever the "1
share of y moves β× as much as 1 dollar of x" relationship implies. Sizing in
the cointegrating ratio (`size_positions(..., mode="beta")`: `q_x = −β q_y`)
makes the position track the *spread itself*, so a parallel move in both legs'
common factor cancels exactly by construction, it's the ratio the regression
estimated, not an assumed 1:1, that neutralizes the shared factor. The
dollar-neutral mode (`mode="dollar"`, equal gross on each leg) is simpler and
exactly dollar-neutral at entry but *deviates* from the cointegrating ratio,
leaving small residual exposure, a trade-off the code documents explicitly in
`signals.py`'s `size_positions` docstring.

</details>

**Q8. What's the z-score entry/exit framework, and what are the actual default parameters this project's backtest uses?**

<sub>Tags: `L1`, `pairs`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`signals.py`'s `generate_signals` runs a state machine over a z-score series
with `SignalRules` defaults: **enter** when `|z| ≥ entry_z = 2.0`; **exit**
when z reverts through `exit_z = 0.0` (full mean touch); **hard stop** when
`|z| ≥ stop_z = 4.0`; optional **time stop** at `max_holding` bars, typically
set via `time_stop_bars(half_life, k=3.0)` = ⌈3 × half-life⌉ bars capped at 252.
z>0 (spread rich) triggers a short-the-spread entry; z<0 triggers long.
Positions are quoted in {−1,0,+1} spread units.

</details>

**Q9. Two z-score construction methods appear in the code: what are they, and why does one avoid a warm-up period?**

<sub>Tags: `L2`, `pairs`, `code-reading`</sub>

<details><summary>Answer</summary>

`zscore_rolling(spread, window)` is the standard rolling-window z-score `(s − rolling_mean)/rolling_std`,
non-parametric but needs `window` observations to warm up (NaN before that),
and it conflates the spread's *level* deviation with its *speed* of reversion.
`zscore_ou(spread, ou)` instead uses the OU model's stationary distribution,
`z = (s − μ)/(σ/√(2κ))`, since μ and the stationary variance are model
parameters (not rolling statistics), there's no warm-up period, but it treats
them as frozen and, in walk-forward use, they must come only from the
formation window (`signals.py` docstring).

</details>

**Q10. Why "re-entry arming": what problem does it solve, and how large is its measured effect?**

<sub>Tags: `L2`, `pairs`, `numeric`</sub>

<details><summary>Answer</summary>

Without it, a hard stop at `|z| ≥ 4` would be immediately followed by re-entry
at `|z| ≥ 2` on the very next bar, the state machine would re-fight the exact
trade the stop just cut, since the spread is still extreme. The fix
(`generate_signals`, the `armed` flag): after *any* exit, the pair cannot
re-enter until `|z|` first drops back inside the entry band. This is what caps
the loss in the project's regime-break case study, with stops and arming, the
post-break loss on seed 42 is about **−$25k** (one entry, then the stop fires
and arming blocks re-entry while z stays extreme); without any
stop-loss/time-stop, the same book loses about **−$724k**, averaging into a
spread that never reverts (`docs/VALIDATION.md` §8).

</details>

**Q11. Why is in-sample cointegration necessary but not sufficient for a live pairs trade: what does VALIDATION.md's regime-break case study show?**

<sub>Tags: `L2`, `pairs`, `numeric`</sub>

<details><summary>Answer</summary>

A1 in the assumptions register states it plainly: cointegration found in the
formation window must *persist* into the trading window, and nothing in-sample
warns you if it won't. The case study plants an OU spread (true half-life 11.6
days) for 750 days, then switches it to a drifting random walk. Fitted on the
pre-break data alone, the fit looks *good* and is indistinguishable from a
stable pair. On the synthetic seed-42 path, +$80k of pre-break P&L is followed
by −$25k after the break with stops and re-entry arming, or **−$724k** without
them; across 30 synthetic seeds the means are +$82k, −$20k and −$472k
(`docs/VALIDATION.md` §8). The desk-level causes named for this kind of break
are regime change, mergers (a cash-target leg pins to the offer price so κ
collapses), and fundamental shifts (`docs/DESK_GUIDE.md` §5, "Merger breaks a
pair").

</details>

**Q12. Beyond the regime-break study, what other failure mode does VALIDATION.md flag about trusting an OU fit alone (assumption A10)?**

<sub>Tags: `L2`, `pairs`</sub>

<details><summary>Answer</summary>

On a *pure random walk*, OLS on the discretised AR(1) returns `b` slightly
below 1 due to finite-sample bias, so `OUFit.mean_reverting` comes back **True**
with a spurious half-life of 300–950 days across seeds
(`test_an_ou_fit_alone_cannot_reject_a_random_walk`). The `mean_reverting`
flag only trips exactly at `b ≥ 1`, so an OU fit by itself gives *no*
protection against trading a random walk. This is the concrete reason the
funnel gates on the Engle-Granger test (with correct N=2 critical values) *before*
any OU fit is trusted, plus a hard cap on the accepted half-life, an estimate
in the hundreds of days is the tell.

</details>

**Q13. There's also a documented bias in the half-life estimate itself (A9). What direction, and what's the desk consequence?**

<sub>Tags: `L2`, `pairs`, `numeric`</sub>

<details><summary>Answer</summary>

OLS on the discretised AR(1) inherits the Kendall (1954) small-sample bias:
with a fitted mean, the AR(1) coefficient `phi` is biased *down* by about `(1 + 3 phi)/n`,
so `kappa = -ln(phi)` is biased **up** and estimated half-lives come out **too
short** (`docs/VALIDATION.md` §11).
`test_ols_kappa_is_biased_upward_over_many_seeds` checks this over 200 seeds
at `(n, kappa) = (12,000, 0.01)` and `(2,000, 0.03)`: the mean relative bias
is positive by more than 3 Monte Carlo SE and close to the Kendall value.
Since a time stop is set at *k × estimated half-life*, the bias makes the time
stop slightly **too tight**, cutting trades before the spread has had its
expected time to revert. The effect is largest for slow-reverting pairs and
short samples; the mitigant is a bias correction or a slightly larger `k`.

</details>

**Q14. How does the project actually charge transaction costs, and why does a mean-reversion strategy feel this more than a buy-and-hold one?**

<sub>Tags: `L2`, `pairs`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`backtest.py`'s `CostModel` charges commission (`cost_bps`, default 5bp/leg)
and slippage (`slippage_bps`, default 2bp/leg) as explicit cash on every leg's *traded
notional* at each rebalance, plus annualised borrow (`borrow_bps`, default
50bp) accrued daily (ACT/252) on the short leg's market value; the accounting
identity `net = gross − commission − slippage − borrow` is asserted to the
penny. A mean-reversion strategy trades every entry/exit/stop cycle, the
walk-forward book turns over ~6.7x annually, so costs compound with turnover
in a way a buy-and-hold position never sees.

</details>

**Q15. What does the equity project's cost-sensitivity table (VALIDATION.md §8) actually show, quantitatively?**

<sub>Tags: `L1`, `pairs`, `numeric`</sub>

<details><summary>Answer</summary>

Costs scale with turnover, and the project's synthetic seed-7 study
(`docs/VALIDATION.md` §7) shows how fast they bite once market impact is
added. With square-root impact coefficient `Y = 0.7` and an assumed $20mm ADV
per name, the walk-forward Sharpe is 1.09 at $1mm gross per pair (2.5% of ADV
per leg) but 0.10 at $10mm per pair (25% of ADV), and it turns negative at `Y = 1.0`
and $10mm. The fill rule matters too: closing-auction execution at 10bp per
leg cuts the seed-7 walk-forward Sharpe from 1.09 to 0.92. All of this is
synthetic, but the shape is the point: a mean-reversion book that looks fine
at small size can have almost no edge left at the size a desk would want to
run.

</details>

**Q16. Why does the pipeline screen candidate pairs on *return* correlation rather than price correlation?**

<sub>Tags: `L2`, `pairs`, `numeric`</sub>

<details><summary>Answer</summary>

Because price levels of two *independent* random walks are spuriously
correlated, a classic Granger-Newbold result, while their returns are not.
`test_price_corr_spurious_return_corr_not` measures independent walks
averaging |price corr| > 0.45 but |return corr| < 0.10. A price-level screen
would select pairs that merely share a common drift; a return-correlation
screen selects pairs that share common shocks, which is the more meaningful
pre-filter. METHODOLOGY.md is explicit that this screen is still just a *cheap
filter*, not evidence of cointegration, the trap pairs in the panel have
return correlation ≈0.92 and are still rejected at the ADF stage.

</details>

**Q17. Why does this project choose Engle-Granger over Johansen, and over the classic distance method?**

<sub>Tags: `L2`, `pairs`</sub>

<details><summary>Answer</summary>

Against Johansen (ML rank/vector estimation for N-series VARs): for a **two-leg
pair** the cointegration rank can only be 0 or 1, which is exactly what EG's
single test answers; at N=2 Johansen's VAR lag-order selection and eigenvalue
machinery buy little beyond one real advantage, its result is symmetric in the
two legs where EG's depends on which leg is the dependent variable, and it's
documented as the scaling path for baskets of 3+ names instead. Against the
distance method (Gatev-Goetzmann-Rouwenhorst SSD, ranking pairs by
sum-of-squared normalized price gaps): it's cheap and assumption-light, kept
as a pre-screen (`ssd_screen`), but has no null hypothesis, so it can't
distinguish "these paths track because of a real common factor" from "they
tracked by luck," and it gives no spread model, hence no principled half-life
or time-stop. EG gives a testable null and its residual *is* the tradeable
hedge ratio directly (`docs/METHODOLOGY.md` §1).

</details>

**Q18. What's the adaptive alternative to the static OLS hedge ratio, and what does it trade off?**

<sub>Tags: `L1`, `pairs`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`spread.py`'s `rls_hedge_ratio` implements exponentially-weighted **recursive
least squares** with a forgetting factor λ. It is labelled RLS, not Kalman: it
is algebraically the Kalman filter for a random-walk-coefficient model where
the state-noise-to-observation-noise ratio is tied to λ instead of estimated
separately. It has one knob (memory ≈ 1/(1−λ)) and runs O(1) per step, versus
a full Kalman filter's two separately-calibrated noise covariances and
explicit smoothing. The full filter is `eq_pairs.kalman.kalman_hedge_ratio`
(added in milestone v2.1), which estimates the observation and state noise by
maximum likelihood. Tests show the RLS tracker follows a hedge ratio drifting
1.0→2.0 with mean error < 0.02. The documented identification caveat: with
short memory, an intercept becomes nearly collinear with the price level, so
RLS tracking is run with `intercept=False`, letting the OU mean absorb the
level instead.

</details>

**Q19. What is the project's no-lookahead guarantee, and how is it tested?**

<sub>Tags: `L1`, `pairs`, `numeric`</sub>

<details><summary>Answer</summary>

The backtester decides the signal at the close of day t and fills at a
modelled price `p_t + f (p_{t+1} - p_t)` with `f = 0.5` by default
(`ExecutionConfig`); `f = 1` is the next close, and a close-auction mode
charges an explicit auction cost instead.
`test_engine_reports_the_lagged_losing_result` constructs a spread that
alternates ±3 daily, engineered so same-day execution is profitable (the
reversion happens overnight) while the lagged execution loses money, and
asserts the engine produces the *losing* number. `TestExecutionConvention`
adds a hand ledger for the fill price and a perfect-foresight signal that
earns a mean gross P&L within 4 SE of zero at `f = 1` but more than 10 SE when
filled at the close of t; a mutant engine that fills at the close of t fails
five of these tests (`docs/VALIDATION.md` §9).

</details>

**Q20. What's the difference between the in-sample and walk-forward backtest results, and why does that gap matter?**

<sub>Tags: `L1`, `pairs`, `numeric`</sub>

<details><summary>Answer</summary>

**[Synthetic, seed 7.]** In-sample: net P&L $2.11mm, annual return 5.9% on
$6mm, Sharpe 2.41, hit rate 90.8%. Walk-forward (formation window fitted,
frozen, traded out-of-sample): net P&L $0.99mm, 3.3%, Sharpe 1.09, hit rate
58.5% (`docs/VALIDATION.md` §5). Across 30 synthetic seeds the in-sample
Sharpe is 2.68 [2.54, 2.83] and the walk-forward Sharpe 1.33 [1.14, 1.53]
(§4), so the in-sample figure is about twice the walk-forward one. Selection
criteria and OU parameters fitted on the same data they're tested on flatter
every number, so the walk-forward figures, not the in-sample ones, are the
expectation to size a book on.

</details>

**Q21. Sharpe ratio standard errors are computed two ways in this project: why, and what does the second method show?**

<sub>Tags: `L2`, `pairs`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`metrics.py` computes the Sharpe SE both under the iid assumption
(`√((1+SR²/2)/T)`, Lo 2002) and **Lo-adjusted** for serial correlation, using
a Bartlett/Newey-West long-run variance ratio on the mean return.
Mean-reversion strategies produce autocorrelated P&L (a position held over
several days generates correlated daily P&L within the trade), so with
positive autocorrelation the iid SE understates uncertainty, tested to widen
the SE by >30% at AR(1) φ=0.6. On the synthetic seed-7 book the Lo SE is the
smaller one in both quoted cases (in-sample 0.41 iid vs 0.29 Lo, walk-forward
0.45 vs 0.37, `docs/VALIDATION.md` §5), so there the iid SE overstated the
uncertainty. That is the point: the direction of the adjustment depends on the
sign of the autocorrelation, and the iid number is not a safe default.

</details>

**Q22. What is the "trap" the equity selection funnel is specifically built to catch, and how many pairs actually survive it?**

<sub>Tags: `L1`, `pairs`, `numeric`</sub>

<details><summary>Answer</summary>

Starting from 40 same-sector candidate pairs in the synthetic seed-7 20-name
panel: 8 pass the return-correlation screen (ρ ≥ 0.6), and only 4 of those
pass Engle-Granger at 5%, all four planted pairs. EG rejects the other four,
including all three correlated-random-walk traps that passed the correlation
screen (`docs/VALIDATION.md` §3). Over 30 synthetic seeds, planted pairs pass
EG 3.93 of 4 times on average and traps 0.20 times. This is the funnel doing
the job correlation alone cannot: separating shared-shock co-movement from a
stationary spread.

</details>

**Q23. How does the FX project frame its version of this strategy differently from the equity one: what's actually being traded?**

<sub>Tags: `L2`, `pairs`</sub>

<details><summary>Answer</summary>

The FX project trades **pairs of currency pairs**, e.g. AUDUSD vs NZDUSD,
EURUSD vs GBPUSD, rather than pairs of single stocks, and the spread is built
on **log rates**, not dollar price levels: `s_t = log P1_t − α − β log P2_t`
(`fx/05-pairs-trading/docs/METHODOLOGY.md` §1). Log rates are used because FX
P&L is naturally multiplicative, the hedge ratio gets a clean elasticity
interpretation, and cross rates are exact ratios of USD legs, so triangular
consistency becomes linear and testable in logs. The FX methodology otherwise
reuses the identical from-scratch Engle-Granger + OU machinery as the equity
project (same MacKinnon N=2 surfaces, same OLS/MLE OU cross-check).

</details>

**Q24. Why does the FX methodology say cointegration is *rarer* in FX than in equities, and what "common-factor" trap does it specifically warn about?**

<sub>Tags: `L2`, `pairs`</sub>

<details><summary>Answer</summary>

An equity pair can cointegrate because both prices load on the same firm-level
cash-flow trend; an FX rate is already a *ratio* of two money stocks with no
"company" anchoring its level, so any anchor (PPP, real-rate differentials)
acts only over years-to-decades, weak glue at trading horizons. The specific
trap: any two USD-quoted pairs share the USD leg by construction, so high
return correlation between, say, AUDUSD and NZDUSD is partly mechanical (a
common USD factor), and a correlation screen alone floods the funnel with
USD-factor pairs that are *not* cointegrated. The project's synthetic
two-block panel reproduces this directly: 0 survivors among 7
factor-correlated candidates, versus 1 survivor out of 1 planted true
cointegration (`fx/05-pairs-trading/docs/METHODOLOGY.md` §2). Where FX
cointegration does hold, the doc says it reflects a real policy or economic
linkage, AUD/NZD (paired commodity economies), NOK/SEK (Scandinavian bloc),
EUR/CHF pre-2015 (SNB floor).

</details>

**Q25. What extra P&L component does the FX version model that the equity version doesn't, and why can it flip the strategy's sign?**

<sub>Tags: `L2`, `pairs`</sub>

<details><summary>Answer</summary>

Carry. A held FX position is financed, long the base currency earns its
deposit rate, funded in the quote currency at its rate, so total backtest P&L
is `spot P&L + carry accrual − transaction costs`, with the daily roll
computed from covered interest parity forward points. The methodology states a
mean-reversion signal computed on spot alone can be systematically **wrong-carry**,
and the pipeline constructs an explicit example where including carry flips
the sign of the strategy's overall P&L
(`fx/05-pairs-trading/docs/METHODOLOGY.md` §1, §2). The FX signal machine also
adds a **carry-aware entry filter**: an entry is vetoed when expected carry
drag over the expected holding period (`carry/day × k·half-life`) exceeds the
expected reversion gain (`(|z|−exit)·σ`), carry-favourable entries are never
vetoed.

</details>

**Q26. What is the FX project's "triangular null case," and what does it guard against?**

<sub>Tags: `L1`, `pairs`</sub>

<details><summary>Answer</summary>

`log EURUSD + log USDJPY − log EURJPY = 0` holds identically because cross
rates are exact ratios of USD legs, feeding this combination through the
cointegration/spread machinery produces a "spread" with variance ~1e-31, which
must be flagged **degenerate**, never reported as "perfectly cointegrated." A
degeneracy detector (spread std below 1e-7 in log units) exists specifically
to catch this, and a funnel test asserts it fires
(`fx/05-pairs-trading/docs/METHODOLOGY.md` §6). Economically the doc notes
there's no exploitable triangular arbitrage at daily frequency once three
legs' bid-ask spreads are paid, any measured deviation in real data is inside
costs and is arbitraged away at the tick level by HFT, so a "cointegrated"
identity like this is a data-hygiene bug to catch, not a trading opportunity.

</details>

**Q27. What is a pre-registration in `research/`, and why does the hash cover only the text above `## Signature`?**

<sub>Tags: `L1`, `pairs`</sub>

<details><summary>Answer</summary>

It is a file that fixes the hypothesis, data, parameters, costs, primary
metric and decision rule before the hold-out (2022 onward) is touched. The
registration hash is the SHA-256 of the text above `## Signature`, line
endings normalised, so filling in `signed_by`, `signed_on` and `sha256` does
not change the hash being signed. `python research/tools/prereg.py check`
refuses a result whose hash does not match, or a hold-out result whose
registration is not signed with that hash.

</details>

**Q28. The FX pairs study had a training Sharpe of 0.24. Why is that not evidence of an edge?**

<sub>Tags: `L2`, `pairs`, `numeric`</sub>

<details><summary>Answer</summary>

It is a training-period number (ECB data, 1999 to 2021) and the parameters
were changed twice on that same data before signing, so all three variants
count as trials. The block-bootstrap 95% interval [-0.17, 0.46] includes zero,
and the strategy held a position on only 79 of 4,867 days. It also turned out
that the signal as registered could almost never fire: on simulated pairs that
are cointegrated by construction the Kalman one-step z passed 2 on about 0.5%
of days, and never at JPY-like price levels. The registration was therefore
withdrawn before any hold-out data was read (DR-R4-5 in `research/README.md`),
and its hold-out will never be run. A redesigned draft, `pairs-fx-g10-v2`,
uses a level-based z and is scored only on forward data.

</details>

**Q29. What do `johansen_trace` and `kalman_hedge` in `research/studies/models.py` add to the pairs toolkit, and what are their oracles?**

<sub>Tags: `L2`, `pairs`</sub>

<details><summary>Answer</summary>

Johansen (1991) tests for cointegration among several series at once instead of one Engle-Granger regression; it is matched to statsmodels `coint_johansen` to 1e-9 relative. `kalman_hedge` lets the hedge ratio drift as a random walk (Harvey 1989); with the state noise `delta` set to 0 it must equal recursive least squares, which is the test.

</details>


## Set 10: Credit Risk / PD Modeling

**Q1. What does "probability of default" (PD) mean, and over what horizon is it typically quoted in this portfolio?**

<sub>Tags: `L1`, `credit`</sub>

<details><summary>Answer</summary>

PD is the probability that an obligor defaults within a fixed reference
period, most commonly one year. Both credit-risk projects in this portfolio
(`python/equity/06-credit-risk` and `python/fx/06-credit-risk`) quote PD as a
1-year figure: the equity project's Bernoulli target is "did this loan default
within one year of origination" (METHODOLOGY.md §5, assumption 2), and the FX
project derives a hazard rate `h = −ln(1 − PD_1y)` explicitly from a "1y PD"
rating-band midpoint (METHODOLOGY.md §2.3) to extend it to the multi-year
horizon of its expected loss on exposure for FX forwards.

</details>

**Q2. What model does the equity project actually implement for estimating PD, logistic regression, a structural/Merton model, or something else?**

<sub>Tags: `L2`, `credit`, `code-reading`</sub>

<details><summary>Answer</summary>

It is a WOE (Weight-of-Evidence) scorecard: each raw feature is binned, each
bin replaced by its WOE value, and a logistic regression is fitted on the
WOE-transformed features via a from-scratch Newton-Raphson / IRLS solver
(`fit_logistic` in `src/eq_credit/model.py`). There is no Merton/structural
component in the equity project, METHODOLOGY.md §2 explicitly considers and
rejects a structural/distance-to-default model as "Alternative 3," noting the
book is mid-market lending with no market cap or asset volatility to observe.

</details>

**Q3. Why is a linear-in-log-odds model (logistic regression) a natural fit for a binary default/no-default outcome?**

<sub>Tags: `L2`, `credit`, `code-reading`</sub>

<details><summary>Answer</summary>

The target is Bernoulli (default = 1, survive = 0), so a model needs to map
linear predictors to values in (0,1) while keeping estimation tractable; the
logit link `ln(p/(1-p)) = Xβ` does this and turns maximum-likelihood
estimation into a smooth, concave optimization solvable by Newton's method.
`fit_logistic` implements exactly this: it iterates `β ← β + (XᵀWX)⁻¹Xᵀ(y − p)`
with `W = diag(p(1−p))`, which is IRLS on the Bernoulli log-likelihood.
WOE-binning first puts every feature on the same log-odds scale, so the fitted
coefficients (all ≈ −1 on the pipeline book) are directly comparable across
features, a property a raw linear-probability model wouldn't give you.

</details>

**Q4. Why doesn't this project use a Merton-style structural PD model instead?**

<sub>Tags: `L2`, `credit`</sub>

<details><summary>Answer</summary>

METHODOLOGY.md's "Alternative 3" explains: Merton/KMV-style
distance-to-default models need an observable market asset value and asset
volatility (typically backed out from traded equity), which mid-market
borrowers in this book don't have, no market cap, no listed equity. The FX
project's sovereign methodology makes the same point about sovereigns (no
observable "asset value" or default barrier), calling the Merton mapping
"heroic" there. Structural models also tend to produce point-in-time, cyclical
PDs, which the equity project flags as being in tension with a stable
origination cutoff (TTC vs PIT, METHODOLOGY.md §6).

</details>

**Q5. What is "expected loss" and what are its three standard components?**

<sub>Tags: `L1`, `credit`, `code-reading`</sub>

<details><summary>Answer</summary>

Expected loss (EL) is the average credit loss a lender expects to realize on
an exposure, decomposed as EL = PD × LGD × EAD: probability of default, loss
given default (the fraction of exposure not recovered), and exposure at
default (the amount outstanding when default happens). `expected_loss` in
`src/eq_credit/portfolio_risk.py` implements this multiplication directly,
after validating that PD, LGD, and EAD are all finite and in-range
(`_validate_pd_lgd_ead`).

</details>

**Q6. Does this project implement all three components of EL (PD, LGD, EAD), or only PD?**

<sub>Tags: `L2`, `credit`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

The scorecard itself only estimates PD from data, LGD and EAD are treated as
inputs, not modeled outputs. `expected_loss(pd, lgd, ead)` in
`portfolio_risk.py` takes LGD and EAD as arguments rather than deriving them,
and METHODOLOGY.md assumption 4 states this plainly: "LGD and EAD independent
of PD... EL = PD·LGD·EAD multiplies point estimates" using externally supplied
LGD/EAD. The synthetic demo book applies a 25% downturn-LGD haircut,
multiplicative with LGD capped at 1 (EL rises from 605.3m to 754.2m, a little
under 1.25x because of the cap), as a regulator-style stress adjustment rather
than a fitted LGD model.

</details>

**Q7. What is "wrong-way risk" and how does this project handle (or fail to handle) it?**

<sub>Tags: `L1`, `credit`</sub>

<details><summary>Answer</summary>

Wrong-way risk is exposure that rises as the counterparty's PD rises, for
example a long-USD forward with an EM sovereign whose currency collapses as it
defaults. The related effect where LGD rises with PD, collateral values
falling exactly when defaults spike, as in 2008 commercial real estate, is
recovery (PD-LGD) correlation. Because EL = PD·LGD·EAD multiplies independent
point estimates, this project cannot capture that correlation directly
(METHODOLOGY.md assumption 4); its only concession is the crude, flat
downturn-LGD haircut applied portfolio-wide rather than a joint PD-LGD model.
The FX project names the same gap explicitly for its expected loss on exposure
(assumption A7): assuming exposure independent of counterparty default is "the
worst possible assumption" for an EM sovereign on a long-USD forward, and
VALIDATION.md §4.4 works a hand example (not code) in which ignoring it
understates the loss about 6x.

</details>

**Q8. How is the equity PD model validated for discrimination: how well does it rank-order risk?**

<sub>Tags: `L2`, `credit`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

VALIDATION.md reports AUC, Gini, and KS computed from scratch (`roc_auc`,
`gini`, `ks_statistic` in `src/eq_credit/validation.py`) and cross-checked
against `sklearn.roc_auc_score` to 1e-12. On the synthetic training book, AUC =
0.7850 (95% bootstrap CI 0.770–0.800) against a true-model AUC ceiling of
0.7813; the generator and the scorecard share a functional form, so this
closeness is by construction rather than evidence of a strong model. Decile
analysis (`decile_table`) shows default rates rising monotonically from 0.10%
in the safest decile to 11.2% in the riskiest, a 112x spread, which is also
unit-tested (`test_decile_default_rates_monotone_for_good_model`).

</details>

**Q9. What does "calibration" mean for a PD model, and how is it distinct from discrimination?**

<sub>Tags: `L1`, `credit`, `numeric`</sub>

<details><summary>Answer</summary>

Discrimination measures whether the model ranks obligors correctly by relative
risk (AUC/Gini/KS answer "does a higher score mean higher risk?"). Calibration
measures whether the model's absolute predicted probabilities match realized
default rates (does a group of obligors scored at 3% PD actually default about
3% of the time?), a model can discriminate well while being badly
miscalibrated, e.g. systematically too low or too high. VALIDATION.md's own
OOT results make this concrete: on the synthetic book, OOT AUC only drops
modestly (0.7850 → 0.7699) but calibration is flatly rejected by
Hosmer-Lemeshow (out-of-sample χ² = 43.2, df 10, p = 4.7e-06), and the
realized OOT default rate (4.78%) far exceeds mean predicted PD (3.81%): the
model still ranks obligors reasonably well but its absolute PDs are wrong.

</details>

**Q10. What is the Hosmer-Lemeshow test, and what does this project's implementation and testing of it show?**

<sub>Tags: `L1`, `credit`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Hosmer-Lemeshow is a goodness-of-fit test for calibration: it groups
observations into `g` PD groups (deciles by default), compares predicted vs.
observed default counts per group, and forms a chi-square statistic under the
null that the model is well calibrated. `hosmer_lemeshow(..., out_of_sample=...)`
in `src/eq_credit/validation.py` uses the right reference for each case:
`chi2(g - 2)` for PDs fitted on the same data, `chi2(g)` for PDs scored on
data not used in fitting. The out-of-sample size is checked by simulation with
known PDs (`test_hl_out_of_sample_size_is_nominal`: the 5% test rejects within
[3%, 7%] of 1,000 replications). On the synthetic pipeline, HL fails to reject
calibration in-sample at 1% (χ² = 17.9, df 8, p = 0.022) but sharply rejects
it out-of-time (χ² = 43.2, df 10, p = 4.7e-06), which is the intended
diagnostic behavior. Calling it without `out_of_sample` is deprecated, because
the degrees of freedom depend on it.

</details>

**Q11. What is PSI (Population Stability Index), and what does it measure that Hosmer-Lemeshow does not?**

<sub>Tags: `L1`, `credit`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

PSI measures how much the distribution of a score or feature has shifted
between two samples (e.g. development vs. current portfolio), it is a
covariate/input-drift monitor, not a correctness check. `psi`,
`psi_from_proportions`, and `psi_status` in `validation.py` implement it,
hand-validated to 1e-14 against known proportions, with bands (< 0.10 stable,
0.10–0.25 monitor, > 0.25 shifted) also validated on planted shifts.
VALIDATION.md's failure-mode #1 makes the crucial distinction: on the OOT
sample, score PSI stayed at 0.074 (stable band) even as HL calibration
collapsed, PSI would not have caught the regime break because the input mix
barely moved even though the feature→default relationship shifted; both
monitors are required together (DESK_GUIDE.md §7).

</details>

**Q12. What is the Brier score, and where does it appear in this project's validation?**

<sub>Tags: `L1`, `credit`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

The Brier score is the mean squared error between predicted probabilities and
realized binary outcomes, a combined discrimination-and-calibration score
(lower is better). `brier_score(y, pd_hat)` in `validation.py` computes it
and, per the hardening notes in VALIDATION.md §6, explicitly rejects
non-finite inputs and gates predicted PDs to [0, 1]. On the pipeline it rises
from 0.0272 in-sample to 0.0431 out-of-time, consistent with the AUC/HL story
of degraded but not collapsed performance under regime drift.

</details>

**Q13. Walk through a concrete edge case this project's tests cover around near-zero or near-one predicted probabilities.**

<sub>Tags: `L2`, `credit`, `code-reading`</sub>

<details><summary>Answer</summary>

`test_edge_cases.py` and `test_input_validation.py` cover PD = 0/1 explicitly:
scores are clamped to finite values while preserving ordering rather than
producing ±infinity from `ln(p/(1-p))`. More broadly, every entry point that
consumes a probability, `brier_score`, `hosmer_lemeshow`, `expected_loss`,
`basel_k`, `asset_correlation`, validates inputs are finite and gates
predicted PDs to [0, 1] (VALIDATION.md §6, "NaN/Inf rejection everywhere"),
because `NaN < 0` silently evaluates to `False` and would otherwise pass naive
range checks.

</details>

**Q14. How does the model handle a zero-default (or all-default) training sample?**

<sub>Tags: `L2`, `credit`</sub>

<details><summary>Answer</summary>

With zero defaults, the WOE calculation, the logistic MLE, and AUC are all
mathematically undefined (division by zero bad-count, no positive class to
rank against). Rather than returning garbage, every relevant entry point
raises an informative `ValueError`, tested explicitly in
`test_zero_defaults_raises_everywhere` (VALIDATION.md §5.3). The same
failure-mode section notes that at the opposite extreme of very few
observations (~30 rows / 3 defaults) the fit stays numerically finite but
standard errors balloon, and this is reported openly via the Wald table rather
than hidden.

</details>

**Q15. What happens when a feature has missing data for a given obligor, and what edge case does this create?**

<sub>Tags: `L2`, `credit`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Missing values get their own WOE bin rather than being imputed or dropped,
`fit_numeric_binning` in `src/eq_credit/woe.py` treats missingness as
informative, and on the pipeline book the missing bin for `behavioral_score`
carries a WOE of −0.80 (thin-file borrowers score riskier). VALIDATION.md
documents this as a tested edge case ("all-missing feature raises," "unseen
categories at transform time are mapped to missing-bin WOE"), but
METHODOLOGY.md assumption 6 flags the real fragility: the fitted missing-bin
WOE assumes the *reason* for missingness is stable over time, if a
data-pipeline change later makes the same field missing-at-random, that
penalty mis-prices otherwise-clean borrowers. DESK_GUIDE.md's COVID scenario
(§7) is exactly this failure realized: payment holidays made
`behavioral_score` go missing for moratorium accounts for a different reason
than the historical missingness pattern.

</details>

**Q16. What is separation, and how does this project detect and handle it?**

<sub>Tags: `L1`, `credit`</sub>

<details><summary>Answer</summary>

Separation occurs when a feature (or combination) perfectly or near-perfectly
predicts the outcome, causing the unpenalized logistic MLE to diverge
(coefficients run to infinity as the fit tries to push predicted probabilities
to exactly 0 or 1). `fit_logistic` detects runaway iterates and raises a
`SeparationWarning`, recommending ridge regularization; supplying a positive
ridge penalty causes the fit to converge instead, both directions are
unit-tested (`test_separation_detected_and_warned`,
`test_separation_ridge_regularizes`, VALIDATION.md §5.4).

</details>

**Q17. What is target/outcome leakage, and how does this project guard against it, concretely?**

<sub>Tags: `L1`, `credit`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Leakage is when a feature encodes information that would not actually be
available (or would already imply the outcome) at prediction time, a planted
post-outcome field like `writeoff_flag` is the textbook example, since it is
essentially the label in disguise. Two independent controls catch it: the IV
screen automatically flags any feature with IV > 0.5 as suspicious
(`SuspiciousIVWarning` in `src/eq_credit/woe.py`; `writeoff_flag` triggers it
with IV ≈ 6.4), and a separate cleaning-layer deny-list
(`FORBIDDEN_POST_OUTCOME_FIELDS`) refuses such fields outright with a
`LeakageError` before they ever reach binning. VALIDATION.md calls this
"defence in depth" because both controls fire independently of each other.

</details>

**Q18. What out-of-sample / regime edge case does this project's validation construct, and what does it reveal?**

<sub>Tags: `L2`, `credit`, `numeric`</sub>

<details><summary>Answer</summary>

The OOT (out-of-time) sample is built with a planted 0.5 drift plus a 0.5
log-odds calibration shift, lifting the realized default rate from 2.95%
(training) to 4.78% (OOT), modeling a genuine regime change rather than
resampling noise. This reveals that a PIT (point-in-time) scorecard degrades
gracefully in discrimination (AUC only −1.5 points) but breaks sharply in
calibration (HL p-value collapses from 0.022 to 4.7e-06, mean predicted PD
3.81% undershoots realized 4.78%), and crucially, PSI alone would have missed
it (stayed at 0.074, "stable"), which is exactly why VALIDATION.md's
failure-mode #1 insists monitoring must track realized-vs-predicted rates
directly, not PSI alone.

</details>

**Q19. What is a Through-the-Cycle (TTC) vs Point-in-Time (PIT) PD, and which is this model?**

<sub>Tags: `L1`, `credit`, `numeric`</sub>

<details><summary>Answer</summary>

PIT PDs track the current state of the credit cycle using current, often
cyclical inputs (like a behavioral score) and are recalibrated frequently,
appropriate for IFRS 9/CECL provisioning, which wants cycle-sensitive expected
losses. TTC PDs average over a full cycle for stability, which is what Basel
IRB capital is designed around. METHODOLOGY.md §6 states this model is
"essentially PIT" because it's calibrated to the realized default rate of its
training window and includes an intrinsically cyclical behavioral score, with
the consequence that feeding these PDs directly into the IRB capital formula
makes capital procyclical (VALIDATION.md failure mode #5 shows repricing the
synthetic demo book at OOT-implied calibration raises every PD by ×1.26 on
average).

</details>

**Q20. How does the equity project's Basel IRB capital calculation get validated for correctness?**

<sub>Tags: `L2`, `credit`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`basel_k` in `portfolio_risk.py` implements the regulatory capital function
K(PD, LGD, M), the Vasicek 99.9% conditional PD minus expected loss, times a
maturity adjustment. It is hand-checked at PD = 1%, LGD = 45%, M = 2.5 against
the published Basel II corporate risk weight of 92.32% from the BCBS
Explanatory Note (July 2005), independently re-derived in the test with the
exact regulatory constants R(PD), b(PD), and Φ⁻¹(0.999)
(`test_basel_k_reproduces_independent_hand_calculation`). The Vasicek
machinery underneath it is separately validated: quantile/CDF round-trip to
1e-10, and the Vasicek 99.9% quantile matches the Basel conditional-PD term at
ρ = R(PD) to 1e-12.

</details>

**Q21. What does the Vasicek economic-capital layer add on top of the PD model, and where is the seeded Monte Carlo used?**

<sub>Tags: `L1`, `credit`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

The PD model gives obligor-level default probabilities; the Vasicek
single-factor Gaussian copula (`vasicek_cdf`, `vasicek_quantile`,
`simulate_portfolio_losses`, `economic_capital` in `portfolio_risk.py`) turns
those into a portfolio-level loss distribution and its 99.9% tail, i.e.
economic capital, a quantity a single-obligor PD/EL calculation can't produce
on its own. Monte Carlo (seeded via `numpy.random.default_rng`) is used
because the analytic Vasicek formula assumes an infinitely granular
(homogeneous, infinitely diversified) portfolio; the MC engine handles the
actual finite, heterogeneous book, and is checked against an exact quadrature
of the finite-N homogeneous portfolio (`finite_portfolio_quantile`). The
finite-size correction is also available analytically: the granularity
adjustment (Gordy 2003; Gordy and Lutkebohmert 2013, `granularity_adjustment`)
agrees with the exact quantile to within one lattice step `1/N` for N from 200
to 20,000 (`test_ga_matches_exact_quadrature_within_one_lattice_step`), and
the Monte Carlo order-statistic CIs (`mc_quantile_ci`) cover the exact value.

</details>

**Q22. What does the FX project's methodology cover that's specific to credit risk in an FX context?**

<sub>Tags: `L1`, `credit`</sub>

<details><summary>Answer</summary>

`python/fx/06-credit-risk` covers three blocks beyond the equity project's
corporate scorecard: (1) a sovereign PD scorecard using the same WOE + IRLS
logistic approach but for country-year panel data with only ~63 training
default events; (2) FX settlement (Herstatt) risk, the exposure window between
when a bank's paid currency leg becomes irrevocable and when the received leg
settles with finality, which is asymmetric by RTGS time zone and zero under
PvP/CLS settlement; and (3) pre-settlement exposure on FX forwards via
Garman-Kohlhagen GBM simulation and netting rules, with
`expected_exposure_loss_for_forward` combining the exposure profile with the
sovereign scorecard's PD term structure. That number is an expected loss on
exposure under real-world PD, not a CVA (which would need risk-neutral default
probabilities and a funding/discounting treatment); the old name
`cva_for_forward` is deprecated for that reason.

</details>

**Q23. Why does the FX project use ρ_sov = 0.30 for sovereign Vasicek capital instead of the equity project's corporate range?**

<sub>Tags: `L2`, `credit`, `numeric`</sub>

<details><summary>Answer</summary>

METHODOLOGY.md §2.4 sets the sovereign asset-correlation parameter at ρ_sov =
0.30, above the Basel corporate range of ρ ∈ [0.12, 0.24], because sovereign
defaults cluster on global systematic factors, USD funding cycles, commodity
busts, contagion, more than idiosyncratic firm-level risk does. The ordering
K(ρ_sov) > K(ρ_corp) across the PD range is unit-tested, and assumption A10
documents this as a conservative single-factor simplification of what is in
reality a multi-region, multi-factor tail-dependence structure.

</details>

**Q24. What is the low-default-portfolio problem, and how does the FX sovereign model address it openly rather than hide it?**

<sub>Tags: `L1`, `credit`, `numeric`</sub>

<details><summary>Answer</summary>

With only ~63 historical sovereign default events feeding 10 WOE coefficients,
standard errors are wide and some coefficients are barely significant (e.g.
`fiscal_gdp` z = -0.7, per METHODOLOGY.md §1), a classic low-default-portfolio
(LDP) problem where there simply isn't enough data to pin down a precise
model. The project addresses this by (a) using WOE compression so each feature
contributes one monotone dimension rather than many free parameters, (b)
separation detection with a ridge fallback, and (c) reporting wide bootstrap
confidence intervals rather than hiding them, and explicitly refuses to claim
precision it doesn't have, anchoring AAA/AA band PD midpoints to long-run
rating-agency default studies (assumption A9) instead of fitting them from
sparse sovereign default data.

</details>

**Q25. How does the equity project's validated output feed into desk-level credit limits and origination decisions?**

<sub>Tags: `L2`, `credit`, `numeric`</sub>

<details><summary>Answer</summary>

DESK_GUIDE.md §2 describes cutoffs set directly from the KS/decile table:
since the default rate is 11.2% in the worst score decile versus 0.10% in the
best (112x spread), the desk sets a score cutoff (e.g. auto-approve ≥ 620,
manual review 560–620, decline < 560) that excludes the riskiest deciles,
removing roughly 55% of expected defaults at the cost of about 20% of loan
volume. Reason codes for declines read directly off the points table (e.g.
"leverage > 0.94: 58 pts vs 117 for < 0.18"), which is only possible because
the scorecard is interpretable by construction, a key reason METHODOLOGY.md
gives for choosing it over a GBM.

</details>

**Q26. How does the PD model feed into loss provisioning and regulatory capital, and why does the desk need to treat those two uses differently?**

<sub>Tags: `L2`, `credit`, `numeric`</sub>

<details><summary>Answer</summary>

For provisioning, DESK_GUIDE.md §4 states IFRS 9 Stage-1 expected credit loss
is literally this model's output, 12-month ECL = PD₁y·LGD·EAD (605m, 1.47% of
EAD on the synthetic demo book; 754m with the downturn-LGD haircut), and
because the model is PIT, ECL rising with the cycle is *intended* under IFRS 9.
US CECL has no stages: it books lifetime ECL from day one, so it would need
the PD term structure, not this one-year PD alone. For capital, however,
DESK_GUIDE.md §5 and VALIDATION.md failure mode #5 warn that feeding the same
PIT PDs into the Basel IRB formula makes capital procyclical (repricing at
OOT-implied calibration raises every PD by ×1.26, demanding more capital
exactly when it's scarcest); Basel expects long-run-average (TTC) PDs per
grade, so IRB use requires a TTC overlay that this project documents as needed
but does not itself build (METHODOLOGY.md §6). Treasury separately consumes
the Vasicek economic-capital number (8.20% of EAD at 99.9% vs Basel K's
10.03%) for ICAAP and stress testing (DESK_GUIDE.md §5).

</details>

**Q27. What is a margin of conservatism in `eq_credit.calibration`, and what floor is applied?**

<sub>Tags: `L1`, `credit`</sub>

<details><summary>Answer</summary>

A PD model is calibrated to a long-run central tendency of default rates; the margin of conservatism adds for estimation error and data deficiencies (categories A, B and C in the EBA guidelines). `calibrate_with_moc` applies the shift and the margins and then the CRR floor of 3 basis points.

</details>

**Q28. How does `eq_credit.cds.bootstrap_hazards` build a hazard curve, and what simplification does it make?**

<sub>Tags: `L2`, `credit`</sub>

<details><summary>Answer</summary>

It solves maturity by maturity for the piecewise-constant hazard that sets each CDS's protection leg equal to its premium leg, with accrual on default. The bootstrapped curve must reprice the input spreads. It uses a flat discount rate and does not apply the ISDA standard-model date conventions, so it is a study model, not a match to market tools.

</details>

**Q29. What does SA-CCR (`fx_credit.saccr`) compute, and how is it different from the CVA in `fx_credit.xva`?**

<sub>Tags: `L1`, `credit`, `numeric`</sub>

<details><summary>Answer</summary>

SA-CCR is the Basel standardised exposure (CRE52): replacement cost plus a
multiplier times the add-on, scaled by 1.4, giving `saccr_ead`. It is a
regulatory capital input with fixed supervisory factors. `bilateral_cva` is a
pricing adjustment: expected exposure profiles weighted by default
probabilities of both parties, with collateral terms (threshold, MTA, margin
period of risk) and wrong-way risk.

</details>


## Set 11: Portfolio Optimization & Risk Allocation

**Q1. State the Markowitz mean-variance optimization problem. What is the efficient frontier?**

<sub>Tags: `L2`, `portfolio`, `code-reading`</sub>

<details><summary>Answer</summary>

Markowitz (1952) mean-variance optimization (MVO) either minimizes portfolio
variance `w'Σw` subject to a target expected return `w'μ = m` and a budget
`sum(w) = 1` (`target_return_portfolio` in `eq_port/mvo.py`), or equivalently
maximizes expected return for a variance cap (`target_risk_portfolio`, which
maximizes `w'μ` subject to `sqrt(w'Σw) ≤ target_vol`). Sweeping the target
across its feasible range and recording the minimum-variance weight at each
point traces the minimum-variance frontier; its upper branch, the targets
above the global minimum-variance portfolio's return, is the efficient
frontier, the set of portfolios where no other combination gives higher return
at the same risk. `efficient_frontier()` implements this sweep, either
analytically (unconstrained) or numerically via repeated SLSQP solves
(bounded).

</details>

**Q2. Give the closed-form unconstrained tangency (maximum-Sharpe) portfolio, and say what code implements it.**

<sub>Tags: `L2`, `portfolio`, `code-reading`</sub>

<details><summary>Answer</summary>

`w_tan = Σ⁻¹(μ − rf·1) / (1'Σ⁻¹(μ − rf·1))`, implemented in
`tangency_weights()` in `mvo.py`. It is validated in
`test_tangency_matches_formula` to 1e-12. The function raises a `ValueError`
when `1'Σ⁻¹(μ − rf) ≤ 0`, since a non-positive denominator means no
fully-invested tangency portfolio with positive excess return exists for the
given inputs, usually a sign that the mean estimates are unreliable rather
than a real economic conclusion.

</details>

**Q3. Why is mean-variance optimization described in this project's docs as an "estimation-error amplifier" or "error maximizer"?**

<sub>Tags: `L2`, `portfolio`, `numeric`</sub>

<details><summary>Answer</summary>

`METHODOLOGY.md` §1 frames it via Merton (1980) and Michaud (1989): the
standard error of an estimated mean return over `Y` years is `σ/√Y`,
independent of sampling frequency, with `σ ≈ 20%` and one year of data the SE
of the annual mean is ≈20%, an order of magnitude larger than realistic
cross-sectional spreads in true expected returns. Unconstrained tangency
weights are `w ∝ Σ⁻¹(μ − rf)`: the optimizer takes differences of noisy means
and multiplies by an inverse covariance that amplifies its worst-estimated
(smallest) eigenvalues. Assets that got lucky in-sample (high estimated mean,
low estimated vol/correlation) receive extreme weights, the optimizer "loads
up on exactly the estimation errors," which is precisely why Michaud calls MVO
an error maximizer rather than a return maximizer.

</details>

**Q4. What sampling frequency fact makes the mean-estimation problem structurally different from the variance-estimation problem?**

<sub>Tags: `L2`, `portfolio`</sub>

<details><summary>Answer</summary>

Variance estimates improve with the *number of observations*, so sampling
daily instead of monthly sharpens them. Mean estimates improve only with *calendar
span* (`σ/√Y`), so daily sampling gives no benefit at all for the mean, you
cannot out-sample your way past Merton's problem, only wait years for it to
shrink. This asymmetry is exactly why the covariance side of this project
(Ledoit-Wolf, EWMA) can be estimated with real confidence from a 252-day
window while the mean side needs an entirely different strategy (shrinkage,
equilibrium priors, or avoiding it altogether).

</details>

**Q5. What is James-Stein shrinkage, and quantitatively how much does the project's estimation-error study shrink toward it?**

<sub>Tags: `L1`, `portfolio`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

James-Stein shrinks each asset's sample mean toward the cross-sectional grand
mean, with data-driven intensity `phi = clip((N−3)·σ̄²/T / Σᵢ(μᵢ−m̄)², 0, 1)`
(`returns_est.py`, per `METHODOLOGY.md` §2). It is biased toward "all assets
are equal" but dominates the sample mean in total squared error for N ≥ 4
(Stein, 1956, proved the sample mean inadmissible; the estimator is James and
Stein, 1961). In the project's synthetic estimation-error study (20 windows on
each of 10 panels), average James-Stein intensity is φ = 0.94, so the data
essentially say to ignore the sample means, and shrunk-mean tangency's mean
true Sharpe (0.328, long-only) beats raw-mean long-only tangency (0.279), a
paired difference whose 95% CI excludes zero (`VALIDATION.md` §3).

</details>

**Q6. What are Black-Litterman and reverse optimization, and how do they sidestep the raw-sample-mean problem differently from James-Stein?**

<sub>Tags: `L1`, `portfolio`</sub>

<details><summary>Answer</summary>

Reverse optimization discards sample means entirely and backs out the returns
that would make a reference portfolio (e.g. the market) optimal: `π = δΣw_mkt`,
no time-series noise, you inherit the market's implied (CAPM equilibrium)
view. Black-Litterman is a Bayesian blend of that equilibrium prior with
explicit investor views: `μ_BL = π + τΣPᵀ(PτΣPᵀ+Ω)⁻¹(Q−Pπ)`. Unlike
James-Stein, which shrinks blindly toward the cross-sectional mean, BL only
moves weights where you actually hold a view (`Ω`→0 honors a view exactly, no
view leaves the prior untouched, both tested to machine precision in
`test_bl_no_views_posterior_equals_prior` and
`test_bl_infinitely_confident_view_holds_exactly`).

</details>

**Q7. What is the minimum-variance portfolio, and why does the project favor it in practice specifically because it needs no return forecast?**

<sub>Tags: `L1`, `portfolio`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`min_variance_weights(cov) = Σ⁻¹1/(1'Σ⁻¹1)` requires only a covariance
estimate, no `μ` enters the formula at all. Because mean estimation is the
dominant source of error (Q3), a portfolio that never touches `μ` structurally
sidesteps that entire failure mode. The synthetic estimation-error study bears
this out: min-variance with Ledoit-Wolf achieves mean true Sharpe 0.333 (95%
CI 0.289–0.377) versus 0.115 (0.068–0.161) for raw-mean unconstrained
tangency, against an achievable (true tangency) Sharpe of 0.427
(`VALIDATION.md` §3).

</details>

**Q8. In the out-of-sample walk-forward race (VALIDATION.md §4), how does minimum-variance actually perform relative to raw-mean tangency and equal weight?**

<sub>Tags: `L2`, `portfolio`, `numeric`</sub>

<details><summary>Answer</summary>

**[Synthetic.]** On the seed-1 8-asset panel (2400 days, 120-day crisis,
monthly rebalance, 10bp costs), MinVar (LW) posts AnnRet −0.031, AnnVol 0.181,
Sharpe −0.083, MaxDD 0.384, better on every dimension than raw-mean tangency
(AnnRet −0.101, AnnVol 0.241, Sharpe −0.320, MaxDD 0.682) and better than
equal weight on vol and drawdown too. Across 20 independent synthetic panels
the picture is more modest: raw tangency has the lowest mean net Sharpe
(−0.200 vs −0.083 for MinVar), but no difference from min-variance is
significant at 5% (`VALIDATION.md` §5). Absolute Sharpes are negative because
every panel contains a crisis with a large negative market drift.

</details>

**Q9. Define risk parity / equal risk contribution (ERC). What optimization problem does it actually solve, and where is it implemented?**

<sub>Tags: `L1`, `portfolio`, `code-reading`</sub>

<details><summary>Answer</summary>

ERC allocates so every position contributes equally to total portfolio *risk*,
not equal notional, and not "optimal" in the return sense. It is implemented
in `erc_weights()` (`risk_parity.py`) by minimizing the strictly convex `F(y) = ½y'Σy − Σᵢ bᵢ ln(yᵢ)`
over `y > 0` via cyclical coordinate descent (Griveau-Billion, Richard &
Roncalli 2013); its first-order condition `yᵢ(Σy)ᵢ = bᵢ` is exactly the
equal-risk-contribution condition, and normalizing `y/sum(y)` gives the ERC
weights. Existence/uniqueness for PD `Σ` is guaranteed by Spinu (2013).
Critically, ERC needs no mean vector at all and no matrix inversion, the
coordinate update is the positive root of a scalar quadratic per asset.

</details>

**Q10. Why does risk parity behave so differently from mean-variance in practice? Ground the answer in the project's failure-mode data.**

<sub>Tags: `L2`, `portfolio`, `numeric`</sub>

<details><summary>Answer</summary>

Because ERC ignores return forecasts entirely, it cannot be seduced by a lucky
in-sample mean the way tangency can; it will never concentrate into one or two
"great Sharpe" assets. In the synthetic estimation-error study, ERC's mean
true Sharpe is 0.322 (95% CI 0.304–0.340, the narrowest interval of any
strategy) versus 0.115 for raw unconstrained tangency. In the seed-1 crisis
subperiod (`VALIDATION.md` §6), ERC stays broadly diversified (average
effective N 7.8), while the concentrated raw-mean tangency book (average
effective N 1.92) posts the worst crisis vol (0.542), worst drawdown (0.448)
and worst crisis return (−0.347) of every strategy tested.

</details>

**Q11. Why does ERC "collapse" to naive inverse-vol weighting in a special case, and where is that tested?**

<sub>Tags: `L2`, `portfolio`, `code-reading`</sub>

<details><summary>Answer</summary>

When all pairwise correlations are equal, the equal-risk-contribution
condition reduces algebraically to weights proportional to `1/σᵢ`,
`inverse_vol_weights()`, because correlation contributes identically to every
position's marginal risk. This is tested exactly
(`test_erc_constant_correlation_equals_inverse_vol`, tolerance 1e-9). Outside
that special case, true ERC diverges from naive inverse-vol because it also
accounts for how each asset's covariance with the rest of the book affects its
marginal risk contribution, whereas inverse-vol ignores correlation structure
entirely.

</details>

**Q12. Because ERC is mean-free, does it care about the sign of correlations? Explain using the 2022 stock-bond scenario in DESK_GUIDE.md.**

<sub>Tags: `L2`, `portfolio`</sub>

<details><summary>Answer</summary>

ERC budgets *volatility*, not directional exposure, so its weights are largely
insensitive to a correlation sign flip, DESK_GUIDE notes that re-running ERC
with the stock-bond off-diagonal block sign-flipped leaves the ERC *weights*
barely changed. What breaks instead is the *portfolio's realized risk level*:
in 2022 the historically negative stock-bond correlation flipped positive
(inflation shock), and both the static 60/40 book and risk parity, which both
size risk assuming diversification from that negative correlation, drew down
together. The lesson: it's the risk target that breaks, not the allocation
logic, which is why RP desks monitor rolling stock-bond correlation as a named
risk factor rather than trusting the ERC weights to self-correct.

</details>

**Q13. Define marginal risk contribution and component (Euler) risk contribution. Give the formulas as implemented.**

<sub>Tags: `L1`, `portfolio`, `code-reading`</sub>

<details><summary>Answer</summary>

For the portfolio variance `w'Σw` as the risk measure, the marginal
contribution of asset `i` is `∂(portfolio vol)/∂wᵢ`, but this project works
directly in variance terms: `risk_contributions()` in `risk_parity.py`
computes `RCᵢ = wᵢ·(Σw)ᵢ`, i.e. weight times `(Σw)ᵢ`, the marginal variance
contribution, rather than a vol-normalized marginal. `(Σw)ᵢ` is the derivative
of `w'Σw` with respect to `wᵢ` (up to the factor of 2 that cancels against the
`wᵢ` multiplication), and `RCᵢ` is that derivative weighted by the position's
own size, the Euler/component decomposition.

</details>

**Q14. Why does summing component risk contributions recover total portfolio risk exactly? What's the mathematical reason, and how is it tested?**

<sub>Tags: `L2`, `portfolio`</sub>

<details><summary>Answer</summary>

`w'Σw` is a homogeneous function of degree 2 in `w` (scaling all weights by
`k` scales variance by `k²`). Euler's theorem for homogeneous functions says
`Σᵢ wᵢ·∂f/∂wᵢ = deg(f)·f(w)`; here `∂(w'Σw)/∂wᵢ = 2(Σw)ᵢ`, so `Σᵢ wᵢ·2(Σw)ᵢ = 2·w'Σw`,
i.e. `Σᵢ wᵢ(Σw)ᵢ = w'Σw` exactly (the factor of 2 cancels because the module
defines `RCᵢ` without it). This is an algebraic identity, not an
approximation, and `test_rc_euler_identity_exact` verifies `Σᵢ RCᵢ = w'Σw` to
1e-18, effectively machine precision. It is what makes "risk contribution" a
well-posed accounting concept: contributions partition the total exactly, with
nothing left over and nothing double-counted.

</details>

**Q15. How does the project prove ERC actually equalizes risk contributions, and to what precision?**

<sub>Tags: `L3`, `portfolio`, `code-reading`</sub>

<details><summary>Answer</summary>

`test_erc_contributions_all_equal` checks that all `RCᵢ` from `erc_weights()`
output agree to a relative spread under 1e-8 (observed ~1e-15 in practice),
essentially exact equality, consistent with the coordinate-descent solving the
first-order condition `yᵢ(Σy)ᵢ = bᵢ` to convergence tolerance `tol=1e-14` on
weight changes per sweep. There's also a "hardening" property test in
`test_infeasible_and_properties.py` that checks equal risk contributions plus
the exact Euler identity on random SPD matrices, not just a fixed example,
i.e. the equal-contribution property is verified structurally, not just on one
hand-picked covariance.

</details>

**Q16. What box/budget constraints does the SLSQP-based optimizer actually support? Name the functions and cite the code.**

<sub>Tags: `L2`, `portfolio`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`min_variance_constrained(cov, bounds=(0,1), budget=1.0)` supports per-asset
box bounds (`bounds`, a scalar tuple applied to every asset, e.g. `(0,1)` for
long-only, or a per-asset list) and a budget equality `sum(w) = budget`.
`target_return_portfolio` adds a return equality `w'μ = target` on top of the
same box+budget machinery, with a necessary-condition pre-check that raises an
informative `ValueError` when `target` is outside the range achievable within
the box bounds. `target_risk_portfolio` instead maximizes return subject to
`sqrt(w'Σw) ≤ target_vol` as a smooth inequality (imposed on variance) plus
box+budget. `max_sharpe_constrained` maximizes `(w'μ − rf)/sqrt(w'Σw)` subject
to `sum(w)=1` and bounds.

Source: `python/equity/07-portfolio-optimization/src/eq_port/mvo.py`.

</details>

**Q17. Does this project's equity optimizer support sector limits or turnover constraints directly? What about leverage?**

<sub>Tags: `L2`, `portfolio`, `code-reading`</sub>

<details><summary>Answer</summary>

Reading `mvo.py`, the only constraint primitives implemented are box bounds
(per-asset lower/upper), a budget (sum) equality, a target-return equality,
and a variance/vol-cap inequality, there is no dedicated sector-limit or
turnover-constraint parameter in the optimizer itself. Turnover is instead
handled downstream in `backtest.py`, where the walk-forward engine charges
two-sided turnover `Σ|w_new − w_drift|` at `cost_bps` and DESK_GUIDE
recommends a turnover *budget* (e.g. ≤2x/yr) as an operational governance
check rather than a solver constraint. Leverage in the risk-parity context is
handled separately by `vol_target_overlay(max_leverage=...)`, which caps the
scaling factor `L = σ_target/σ(w)` applied after ERC weights are computed, a
post-hoc cap, not a constraint inside the optimization itself. Sector limits
would have to be expressed as generic linear inequality constraints added to
the SLSQP `cons` list by a caller; the module doesn't expose a named "sector
cap" argument.

</details>

**Q18. How does the FX project's optimizer extend the constraint set beyond the equity project, and why?**

<sub>Tags: `L2`, `portfolio`</sub>

<details><summary>Answer</summary>

`python/fx/07-portfolio-optimization/docs/METHODOLOGY.md` §5 states the FX
SLSQP solvers add an "FX-native constraint set": a net budget that can be 0
(for a dollar-neutral long-short book) rather than fixed at 1, a
gross-leverage cap `Σ|w| ≤ G` solved smoothly via a long/short split `w = p − q`,
and per-currency boxes. This differs from the equity project because FX
portfolios are naturally long-short (carry longs high-yielders and shorts
funders), a fixed budget-of-1 long-only frame doesn't fit that structure, so
the FX module generalizes the budget constraint and adds the gross-leverage
cap needed to bound a book with no natural net-notional limit.

</details>

**Q19. What does long-only constraint actually do to the estimation-error problem, according to the assumptions register?**

<sub>Tags: `L1`, `portfolio`, `numeric`</sub>

<details><summary>Answer</summary>

Assumption 6 in `METHODOLOGY.md` states that long-only box constraints act as *implicit
shrinkage* (citing Jagannathan & Ma 2003): they are helpful against noise (the
raw tangency's true Sharpe improves from 0.115 unconstrained to 0.279 when
long-only is imposed, synthetic study) but they also cap the achievable
frontier (the long-only frontier has vol ≥ the unconstrained frontier at every
return level) and concentrate portfolios at the top of the frontier (max
weight reaching 0.74/0.75). So the constraint is a real estimation-error
mitigant, not a free lunch, it trades frontier optimality for robustness, and
can itself produce concentrated (low effective-N) portfolios at extreme target
returns.

</details>

**Q20. What real edge case involving near-singular/correlated covariance matrices does VALIDATION.md document, and how does the code respond?**

<sub>Tags: `L2`, `portfolio`, `code-reading`</sub>

<details><summary>Answer</summary>

VALIDATION.md §5 ("Numerical limits") states that perfectly correlated or `T ≤ N`
sample covariance matrices are singular, and the closed-form solvers
(`min_variance_weights`, `tangency_weights` via `_solve_spd`) raise an
informative `ValueError` directing the caller to `psd_repair` or
`ledoit_wolf_cc` rather than failing silently or returning garbage from a
failed Cholesky. `covariance.py`'s `_solve_spd` wraps `np.linalg.cholesky` in
a try/except specifically for this. §3 also documents a concrete stress test:
a 5×8 panel (T < N, sample covariance singular, condition number ∞) where
Ledoit-Wolf still returns a finite-condition PSD matrix and long-only
min-variance solves successfully on it
(`test_short_window_singular_sample_but_lw_invertible`).

</details>

**Q21. Walk through what `psd_repair` does mechanically and why eigenvalue clipping (rather than, say, adding a ridge term uniformly) is the chosen repair.**

<sub>Tags: `L2`, `portfolio`, `code-reading`</sub>

<details><summary>Answer</summary>

`psd_repair(cov, eps=1e-10)` symmetrizes the input, eigendecomposes it, floors
every eigenvalue at `eps * max_eigenvalue` (or an absolute floor `eps` if the
matrix is all-zero/negative), and rebuilds `V·diag(clipped)·Vᵀ`,
re-symmetrized. This is a *relative* floor tied to the matrix's own scale
rather than a fixed ridge added to the diagonal, so it doesn't distort
well-estimated large eigenvalues and only intervenes on the smallest, most
estimation-error-prone directions, exactly the eigenvalues that a
near-singular correlated basket would otherwise blow up when inverted. The
module's docstring is explicit that this is "a projection, not an estimator",
used purely for numerical hygiene after the actual covariance estimate
(sample, EWMA, LW) has been computed, not as a substitute for shrinkage.

</details>

**Q22. What specifically goes wrong in a highly correlated basket in a crisis, quantitatively, per the project's crisis-regime results?**

<sub>Tags: `L2`, `portfolio`, `numeric`</sub>

<details><summary>Answer</summary>

`VALIDATION.md` §6 reports realized average pairwise correlation jumping from
0.470 (calm) to 0.867 (crisis) in the synthetic seed-1 panel. With
correlations pushed toward 1, diversification degrades for every strategy (all
books' crisis vol runs ~2.5–3x their calm-period vol), but the degradation is
uneven: broad, diversified books (ERC effective N 7.8, equal-weight 8.0)
degrade "gracefully," while the concentrated raw-mean tangency book (effective
N 1.92), which was already carrying idiosyncratic risk it thought it was
diversifying away, posts the worst crisis vol (0.542), worst max drawdown
(0.448), and worst crisis total return (−0.347) of any strategy tested.
Min-variance, which holds the low-beta corner by construction, degrades least
(vol 0.380, MaxDD 0.308).

</details>

**Q23. Why is single-factor covariance always guaranteed PSD, and what's the trade-off documented against Ledoit-Wolf?**

<sub>Tags: `L2`, `portfolio`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`single_factor_cov()` constructs `Σ = var_m·bb' + D` where `D` is a diagonal
of non-negative residual variances (`np.maximum(d, 0.0)`); a rank-one outer
product plus a non-negative diagonal is PSD by construction regardless of the
input data, so this estimator can never produce a singular or
negative-eigenvalue result even from very short or degenerate samples.
`METHODOLOGY.md` §3 documents the trade-off: it is guaranteed PSD and
low-variance, but biased, it forces all comovement through one factor and
misses sector-block structure (the project's synthetic truth has three
sectors), whereas Ledoit-Wolf's constant-correlation target captures the
average cross-sectional correlation without collapsing everything to a single
factor.

</details>

**Q24. How does the FX project handle the fact that mean-variance is blind to skew, specifically for carry's crash risk: and why is variance alone considered insufficient there?**

<sub>Tags: `L2`, `portfolio`</sub>

<details><summary>Answer</summary>

FX `METHODOLOGY.md` §5 notes that variance is symmetric while carry's risk is
not: carry has a modest historical vol but a fat, negative-skew left tail
(currency crashes), so mean-variance "sees carry's premium and modest vol and
loads up, precisely because its skew lives outside the variance." The
project's answer is CVaR-constrained sizing via the Rockafellar-Uryasev (2000)
linear-programming formulation over historical scenarios, solved with
`scipy.optimize.linprog` (HiGHS), giving both a "min CVaR" and a "max μ'w s.t.
CVaR_α ≤ c" mode. This mirrors the equity project's Assumption 3
(quadratic-utility/no-higher-moments breaks when a short-vol, negative-skew
asset looks ideal to MVO) but the FX module goes a step further by adding an
explicit non-variance risk measure rather than only documenting the
limitation.

</details>

**Q25. Contrast the two Ledoit-Wolf shrinkage targets used in the equity vs FX projects, and explain why each is the natural choice for its asset class.**

<sub>Tags: `L2`, `portfolio`, `code-reading`</sub>

<details><summary>Answer</summary>

The equity project shrinks the sample covariance toward a *constant-correlation*
target (`ledoit_wolf_cc`, `F` keeps sample variances, replaces every
off-diagonal correlation with the average correlation `rbar`), natural for a
single-market equity universe where all pairs share a broadly similar
comovement level. The FX project instead shrinks toward a *scaled identity*
with intensity `δ = min(b²,d²)/d²`, described as "minimax-safe", appropriate
because FX pairs don't share a single natural "market" correlation the way
same-market equities do (idiosyncratic pairs like EUR-CHF break a
constant-correlation assumption), so shrinking toward "no correlation, equal
variance" (every diagonal entry set to the average sample variance,
`trace(S)/n`) is the safer default before layering in the explicit risk-on/off
one-factor structure for stress logic.

</details>

**Q26. What diagnostic thresholds does DESK_GUIDE.md specify for gating a rebalance, and what do they protect against?**

<sub>Tags: `L2`, `portfolio`, `numeric`</sub>

<details><summary>Answer</summary>

`DESK_GUIDE.md` §2 and §7 list several gates checked before a trade is
released: Ledoit-Wolf shrinkage intensity δ (alert if δ > 0.5, spiking toward
1 means the sample covariance is uninformative and should be investigated
before being trusted), condition number (alert if > 1e3, flagging
near-singular inputs of the kind in Q20), effective N ≥ 4 (a concentration
floor, guards against the effective-N-1.92 collapse seen in the raw-tangency
walk-forward results, `VALIDATION.md` §5), ex-ante vol versus target, and
realized risk contributions staying within ±25% of budget (catching drift away
from the risk-parity mandate between rebalances). Together these turn the
estimation-error and singularity failure modes documented in `METHODOLOGY.md`
and `VALIDATION.md` into concrete, checkable pre-trade controls rather than
only a backtest observation.

</details>

**Q27. What does hierarchical risk parity (`eq_port.hrp.hrp_weights`) avoid that mean-variance cannot?**

<sub>Tags: `L1`, `portfolio`</sub>

<details><summary>Answer</summary>

Mean-variance inverts the covariance matrix, and small errors in a near-singular matrix become large weight errors (the breakdown recipe in COOKBOOK shows it). HRP (Lopez de Prado 2016) clusters assets by correlation distance, orders them so similar assets sit together, and splits risk top-down by inverse variance. It never inverts the full matrix.

</details>

**Q28. Why test a Sharpe-ratio difference with `sharpe_diff_hac` instead of comparing two numbers?**

<sub>Tags: `L2`, `portfolio`, `code-reading`</sub>

<details><summary>Answer</summary>

Two Sharpe ratios estimated on the same dates are correlated and their returns
are autocorrelated, so a plain t-test is wrong. Ledoit and Wolf (2008) use the
delta method with a HAC covariance; `sharpe_diff_hac(kernel="qs_pw")` uses
their prewhitened quadratic-spectral estimator (Andrews and Monahan 1992, with
Andrews' automatic bandwidth), and the Bartlett kernel at fixed lags stays as
the default for old callers. `sharpe_diff_bootstrap` is their studentised
circular block bootstrap: each draw is divided by its own block-based standard
error, which makes the p-value more accurate in short samples than the plain
centred bootstrap (still available with `studentize=False`).

</details>

**Q29. How does `eq_port.cost_qp.rebalance` handle transaction costs, and what is its oracle?**

<sub>Tags: `L2`, `portfolio`</sub>

<details><summary>Answer</summary>

It maximises expected return minus risk minus linear (L1) and quadratic costs, with a budget, box bounds and an optional turnover cap. With linear costs, the optimum leaves positions alone inside a no-trade band; `no_trade_band_solution` gives that band in closed form for the diagonal case and the solver must match it. The solver is SLSQP, so it agrees to about 1e-6, not to the bit.

</details>


## Set 12: Algorithmic Trading & Execution

**Q1. Why can't a portfolio manager just send a large parent order to the market as one block trade?**

<sub>Tags: `L2`, `execution`, `code-reading`</sub>

<details><summary>Answer</summary>

Two costs work against you. First, market impact: a large order consumes the
available liquidity at each price level, so the act of trading pushes the
price against you (you buy higher / sell lower than the price that existed
before you started). Second, information leakage: a visible large order
signals your intent, inviting other participants to trade ahead of you and
worsening your fill. `python/equity/08-algo-execution/src/eq_algo/intraday.py`
models exactly this by splitting a parent order into a per-bucket child-order
schedule and charging each child a temporary and permanent impact cost rather
than filling the whole order at one price.

</details>

**Q2. What is TWAP and how is it implemented in this project?**

<sub>Tags: `L1`, `execution`, `code-reading`</sub>

<details><summary>Answer</summary>

TWAP (time-weighted average price) spreads a parent order into equal-sized
slices over a fixed number of buckets, independent of how volume is actually
distributed through the day. `twap_schedule(parent_qty, n_buckets)` in
`eq_algo/benchmarks.py` literally returns `np.full(n_buckets, parent_qty / n_buckets)`.
As a benchmark rather than a schedule, `twap(prices)` is just the simple mean
of the bucket mid prices over the horizon.

</details>

**Q3. What is the main weakness of TWAP scheduling?**

<sub>Tags: `L1`, `execution`, `code-reading`</sub>

<details><summary>Answer</summary>

It ignores where volume actually concentrates during the day. Equity trading
volume is empirically U-shaped, heavy at the open and close, thin at midday,
which `intraday.py`'s `u_shaped_profile` models explicitly (`p_j ∝ 1 + c(2u_j−1)^2`).
A TWAP schedule trades the same size in a thin midday bucket as in a heavy
open/close bucket, so each equal-sized clip represents a much larger fraction
of available liquidity (and thus more impact) in the thin buckets than a
volume-aware schedule would take on.

</details>

**Q4. What is VWAP, both as a benchmark and as a schedule, and how does the project distinguish the two?**

<sub>Tags: `L1`, `execution`, `code-reading`</sub>

<details><summary>Answer</summary>

As a benchmark, `vwap(prices, volumes)` in `benchmarks.py` computes
`sum(p*v)/sum(v)` over the market tape for the order's horizon, the price a
"fair" execution should track. As a schedule, `vwap_schedule(parent_qty, profile)`
sizes each child order proportional to the expected volume profile (`q_j = X * p_j / sum(p)`),
so the order's own fills track the market's volume distribution by
construction. The module docstring is explicit that these are separate
concepts sharing a name.

</details>

**Q5. Why does VWAP scheduling usually outperform TWAP for a liquid name, and what does it depend on?**

<sub>Tags: `L2`, `execution`, `numeric`</sub>

<details><summary>Answer</summary>

Because it concentrates trading where the market is naturally absorbing more
volume (open/close), each child order represents a smaller fraction of that
bucket's liquidity than an equal-sized TWAP clip would, so
participation-driven impact is lower on average. On synthetic seed 42 with 500
replications, `evaluate_schedules` reports a model-implied IS of 49.6 bp for
VWAP against 53.0 bp for TWAP, with Monte Carlo std 109.5 and 113.7 bp
(`docs/VALIDATION.md` §6): a small edge, well inside the noise of any single
order, but this depends entirely on the input `profile` being an accurate
forecast of where volume will actually land; `vwap_schedule` takes that
profile as a parameter, and a bad forecast (e.g. an index-rebalance day where
the close's share doubles) makes the schedule lag the tape, which
`docs/DESK_GUIDE.md` calls out explicitly under "Index rebalance days."

</details>

**Q6. What is participation-rate (POV) scheduling, and what trade-off does it manage?**

<sub>Tags: `L1`, `execution`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`pov_schedule(parent_qty, forecast_volumes, participation)` in `benchmarks.py`
trades a fixed fraction (`participation * F_j`) of each bucket's *forecast*
volume until the parent is filled, capped at that percentage in every bucket.
The schedule is set before the day, so it must come from a forecast curve,
never from the realised tape of the day it trades: that would be foresight. On
the day, `realised_participation(result)` shows how far the achieved rate
drifted from the target. (v2.0.0 called the argument `market_volumes`, which
invited passing the realised tape; the old keyword still works for one release
with a `FutureWarning`.) The trade-off is urgency versus impact: a higher
participation rate finishes the order faster but takes a larger share of each
bucket's liquidity (more impact per clip); a lower rate spreads out longer,
reducing per-clip impact but exposing the unfilled remainder to more price
drift over a longer horizon (timing risk). The participation rate is a share
of each bucket's volume, not of ADV. Order size against ADV is a separate
limit: `docs/VALIDATION.md` failure mode 2 notes that above about 10% of ADV
the fitted square-root law understates cost, which is the desk's cue to split
the order across days rather than push participation higher.

</details>

**Q7. What happens in `pov_schedule` if the parent order is larger than the day's capacity at the chosen participation rate?**

<sub>Tags: `L2`, `execution`, `numeric`</sub>

<details><summary>Answer</summary>

It raises an informative `ValueError` naming the shortfall: `f"parent order of {parent_qty:.0f} shares cannot complete at {participation:.1%} participation: day capacity is {max_qty:.0f} shares..."`,
advising the caller to split across days or raise the cap, this is tested
explicitly (`test_parent_order_larger_than_day_volume_informative_error` in
`docs/VALIDATION.md` §6). This is a deliberate design choice: silently
under-filling would hide a scheduling error, so the function fails loud with
an actionable message rather than returning a schedule that can't be executed.

</details>

**Q8. What is implementation shortfall (IS), and why is it the benchmark an execution desk is actually measured against?**

<sub>Tags: `L1`, `execution`, `code-reading`</sub>

<details><summary>Answer</summary>

Implementation shortfall (Perold 1988) compares the return of a hypothetical
"paper" portfolio traded instantly and costlessly at the decision price
against what was actually realized. Unlike VWAP or TWAP slippage, which only
measure how you traded relative to the market during the order's life, IS
captures the full cost of the entire decision-to-fill process, including the
price drift before the order even started and the cost of any unfilled tail.
`eq_algo/tca.py`'s `is_decomposition` implements this as three additive
components: delay (decision price → arrival price drift), trading (spread +
impact + intraday drift while executing), and opportunity (cost of the
unfilled remainder, marked at the end-of-horizon price).

</details>

**Q9. What are the three components of the Perold IS decomposition, in the actual formulas used here?**

<sub>Tags: `L1`, `execution`, `code-reading`</sub>

<details><summary>Answer</summary>

From `tca.py`, with side `s`, parent size `X`, fills `(q_j, p_j)`, arrival
price `p_a`, decision price `p_d`, and end-of-horizon price `p_T`: delay `= s*(p_a − p_d)*X`
(how much the price moved between the investment decision and when the order
was released to the desk); trading `= s*Σ q_j*(p_j − p_a)` (spread, impact,
and intraday drift incurred while filling); opportunity `= s*(X − Q)*(p_T − p_a)`
(the cost of not having filled the full order, marked against the unfilled
quantity). The module docstring states, and a dedicated test confirms to
1e-10, that the three sum exactly to the total IS.

</details>

**Q10. How is the Perold identity validated, and why does that matter?**

<sub>Tags: `L2`, `execution`, `numeric`</sub>

<details><summary>Answer</summary>

`docs/VALIDATION.md` §3 lists `test_is_components_sum_to_total_exactly` (25
random orders including partial fills, tolerance 1e-10) and a hand-checked toy
order, 60 shares filled at 10.2, 20 at 10.3, 20 unfilled, that decomposes to
delay 10 + trading 10 + opportunity 8 = 28 = 280 bps exactly
(`test_is_decomposition_hand_checked_toy_order`, tolerance 1e-12). This
matters because a decomposition that doesn't sum exactly to the total would be
useless for attributing "why" a trade cost what it cost, a TCA committee needs
to trust that delay, trading, and opportunity are a clean partition of the
total, not overlapping or leaky categories.

</details>

**Q11. What market-impact model does this project actually implement: is it linear, square-root, or something else?**

<sub>Tags: `L2`, `execution`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

It's a hybrid, per the Almgren-Chriss / Huberman-Stanzl split documented in
`docs/METHODOLOGY.md` §2 and implemented in `intraday.py`: permanent impact is
linear in participation, `perm_move = side * perm_coef * sigma_daily * (q_j / day_volume) * mid`,
and temporary impact (which only affects the fill price and fully reverts by
the next bucket) is a power of the bucket's participation rate, `temp_frac = temp_coef * sigma_daily * (q_j / V_j) ** impact_exponent`,
with `V_j` the market volume in the same bucket and exponent 0.5 by default
(the square-root law). Units: `q_j / V_j` is dimensionless, `sigma_daily` is a
fraction of price per day, `temp_coef` is dimensionless.

Worked example: 3,846 shares in a bucket that prints 38,462 shares is 10%
participation; with `temp_coef` 0.3 and 2% daily vol the temporary cost is 0.3
× 0.02 × √0.1 = 0.190%, 19.0 bp, however many buckets the day is cut into
(`test_twap_cost_does_not_fall_with_finer_slicing`). Up to v2.0.0 the code
divided by `n_buckets * V_j`, counting the bucket width twice, and the same
order cost 5.26, 3.72, 2.63 and 1.86 bp on 13, 26, 52 and 104 buckets. The two
shapes are not interchangeable choices, Huberman-Stanzl (2004) shows permanent
impact *must* be linear or the model admits price-manipulation round-trips,
while the square-root law is the empirically-robust shape for temporary cost
(Almgren et al. 2005; Toth et al. 2011).

</details>

**Q12. The closed-form Almgren-Chriss optimiser assumes linear temporary impact, but the simulator charges square-root impact. How does the project deal with the mismatch?**

<sub>Tags: `L2`, `execution`, `numeric`</sub>

<details><summary>Answer</summary>

The plain idea: a schedule is only optimal for the cost law it was solved under, so solve under the law you evaluate with, and show both.

Formally, with cost `c_k n_k^(1+δ)` per slice, `c_k = S·Y·σ / V_k^δ`, the
problem `min Σ c_k |n_k|^(1+δ) + λ Σ s² x_k²` is convex for any `δ > 0`. At `δ = 1`
it is quadratic and gives the closed form `x_j = X sinh(κ(T−t_j))/sinh(κT)`
(Almgren and Chriss 2000). For other `δ` there is no closed form, and
`power_law_trajectory` solves it by Newton's method; each slice couples only
`x_{k-1}` and `x_k`, so each Newton step is a tridiagonal backward-forward
sweep, which is dynamic programming on the local quadratic model. At `δ = 1`
it reproduces the closed form to 1e-8 of `X`
(`test_power_law_solver_matches_closed_form_at_exponent_one`).

Worked example (synthetic seed 42, 50,000 shares, `Y = 1`, λ = 5e-6; equity 08
`docs/VALIDATION.md` §6): under the square-root law the power-law schedule has
a model-implied IS of 55.5 bp against 59.3 bp for the linear-law schedule, and
under the linear law the order flips on the objective (12,698.6 for the
linear-law schedule against 13,017.2). In the Monte Carlo race the power-law
schedule also has the lower std, 71.6 against 74.5 bp. `compare_impact_laws`
prints this table. Every one of these figures is model-implied: it is a
schedule-level cost model on a bucket simulator, not a forecast of what a real
market would charge.

</details>

**Q13. What is the closed-form Almgren-Chriss trajectory, and what do the two limiting cases of the risk-aversion parameter λ give?**

<sub>Tags: `L1`, `execution`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`ac_trajectory` (via `ac_kappa`) implements `x_j = X * sinh(κ(T−t_j)) / sinh(κT)`,
where `κ` solves `cosh(κτ) = 1 + κ̃²τ²/2` and `κ̃² = λσ²/η̃`. As λ → 0 (no
risk aversion), `κ → 0` and the trajectory collapses to equal slices, TWAP
exactly (`test_zero_risk_aversion_ac_is_twap`). As λ → ∞ (maximal urgency),
the schedule front-loads more than 99.9% of the order into the very first
slice (`test_higher_risk_aversion_front_loads`,
`test_ac_infinite_urgency_limit_dumps_first_slice`). Both limits are
unit-tested, confirming AC is a strict generalization that subsumes TWAP as a
special case.

</details>

**Q14. What trade-off does the Almgren-Chriss risk-aversion parameter λ actually let a trader dial, and how is that shown concretely?**

<sub>Tags: `L2`, `execution`, `numeric`</sub>

<details><summary>Answer</summary>

λ trades expected cost against cost variance: minimizing `E[cost] + λ·V[cost]`
means a higher λ accepts a higher expected cost in exchange for a tighter
distribution (less exposure to adverse price moves while the order is still
working). `docs/VALIDATION.md` §4 shows the efficient frontier is monotonic
across six λ values, expected cost strictly rising, variance strictly falling,
with concrete numbers (E, std) in bps: (15.5, 99.0) at λ=1e-6, (21.1, 73.8) at
λ=5e-6, (53.9, 36.5) at λ=5e-5. `docs/DESK_GUIDE.md` §3 uses this frontier as
the menu for urgent orders: the PM picks λ by deciding how much expected cost
to pay for a tighter cost distribution.

</details>

**Q15. How does AC compare to plain TWAP empirically in this project's simulator, and is the difference statistically real?**

<sub>Tags: `L2`, `execution`, `numeric`</sub>

<details><summary>Answer</summary>

On the synthetic seed-42 schedule race (`docs/VALIDATION.md` §6), AC has a
higher model-implied IS than TWAP (57.1 vs 53.0 bp) and a much smaller Monte
Carlo std (74.5 vs 113.7 bp).
`test_ac_beats_twap_on_cost_variance_on_simulator` turns that into a test over
300 paired replications: it requires both `std_AC < 0.85·std_TWAP` and a
Levene test p < 0.01, so the variance reduction is a validated property, not
an anecdote. A separate analytic-consistency test
(`test_ac_variance_reduction_matches_theory_direction`) confirms the
model-implied frontier agrees in direction with the simulated result.

</details>

**Q16. Why does the project solve Almgren-Chriss numerically (as a QP) for FX rather than using the closed-form equity solution?**

<sub>Tags: `L2`, `execution`, `code-reading`</sub>

<details><summary>Answer</summary>

Classic closed-form AC assumes constant temporary impact η and volatility σ
over the whole horizon. `python/fx/08-algo-execution/docs/METHODOLOGY.md` §5
notes that over a 24-hour FX day both vary roughly 10× across sessions
(Asia/London/Overlap/NY/Late), so the FX project instead solves a
bucket-specific discrete mean-variance QP, `min_n Σ η_j n_j²/τ + λ Σ σ_j² τ x_j²`,
via its KKT system with an active-set loop for one-sided execution,
implemented in `fx_algo/execution/optimal.py`. It's verified against the
closed-form: with constant η, σ the numerical solution matches the analytic AC
trajectory to 1e-9, and λ→0 reduces exactly to the liquidity-weighted
(TWAP-analog) schedule.

</details>

**Q17. What execution benchmarks does the FX project use, and why not VWAP?**

<sub>Tags: `L2`, `execution`, `code-reading`</sub>

<details><summary>Answer</summary>

`fx_algo`'s `execution/tca.py` implements arrival price (implementation
shortfall), interval TWAP, and the WM/R 4pm London fix, explicitly *not* VWAP.
The FX METHODOLOGY.md §1 explains why: spot FX is an OTC dealer market with no
consolidated tape and no official volume print (liquidity fragments across
dealer streams, ECNs, and internalisation pools), so a volume-weighted
benchmark is structurally ill-defined the way it is on a lit equity exchange.
The project treats "pretend a tape exists and implement VWAP" as an
alternative it explicitly rejected as "structurally wrong for FX."

</details>

**Q18. How does FX execution handle the analog of a volume profile, and how does its "VWAP-reversion" signal differ from the equity version?**

<sub>Tags: `L2`, `execution`</sub>

<details><summary>Answer</summary>

Since realised market volume is unobservable in real time, `fx_algo`'s
`pov_schedule` is described as a POV-*analog*: it participates at a capped
rate of a *modeled* session depth profile (Asia/London/Overlap/NY/Late, each
with its own spread/depth/vol in `sessions.SESSION_BOUNDS`) rather than actual
printed volume. Similarly, the mean-reversion feature reverts to the running
time-weighted average mid of the day (`features.reversion_to_session_mean`)
instead of a volume-weighted average, described in METHODOLOGY.md as the FX
analog of VWAP-reversion.

</details>

**Q19. What is transaction cost analysis (TCA) in this project, and what does it actually measure after the fact?**

<sub>Tags: `L1`, `execution`, `code-reading`</sub>

<details><summary>Answer</summary>

TCA is the post-trade evaluation of realized execution cost against a chosen
benchmark, it answers whether the schedule/algorithm choice was actually good,
not just what it cost in isolation. `eq_algo/tca.py`'s `tca_report` runs the
Perold IS decomposition on a completed `ExecutionResult`, and
`benchmarks.py`'s `benchmark_slippage` separately reports slippage vs VWAP,
TWAP, arrival, and decision price in bps. `aggregate_tca` rolls many orders'
IS components (delay/trading/opportunity/total) up into
mean/std/min/max/notional-weighted statistics, matching the desk-level format
`docs/DESK_GUIDE.md` describes: weekly aggregation cut by strategy, side, size
bucket (%ADV), and schedule type, with the realised
trading-cost-vs-sigma·sqrt(Q/ADV) slope used to re-estimate the impact model's
coefficient.

</details>

**Q20. How does the TCA feedback loop close back into the cost model, per `docs/DESK_GUIDE.md`?**

<sub>Tags: `L2`, `execution`, `numeric`</sub>

<details><summary>Answer</summary>

Weekly, `aggregate_tca` output is fit against `sigma * sqrt(Q/ADV)`: the
regression slope re-estimates the square-root-law coefficient `k` that feeds
both the daily backtest cost model and the capacity curve, and the intercept
re-estimates the effective spread. A drift in `k` of more than 25%
quarter-on-quarter triggers a capacity review. Separately, realised AC-vs-VWAP
slippage is compared against the simulator's predicted distribution via
`evaluate_schedules`; DESK_GUIDE is explicit that if a schedule is
consistently outside its predicted band, "the impact model, not the trader, is
wrong", i.e., TCA is used to diagnose model misspecification, not just to
grade traders.

</details>

**Q21. What is `slippage_attribution`, and what does it decompose per-bucket fill slippage into?**

<sub>Tags: `L1`, `execution`, `code-reading`</sub>

<details><summary>Answer</summary>

`eq_algo/tca.py`'s `slippage_attribution(result)` breaks each filled bucket's
slippage versus arrival price into three additive currency-per-share terms:
`drift` (the bucket's pre-trade mid minus arrival, market noise plus the
accumulated permanent impact of the parent's own earlier child orders),
`spread` (the half-spread paid), and `temporary` (the square-root impact
cost). These three sum exactly to the total per-bucket slippage, and a
quantity-weighted `TOTAL` row aggregates across the whole order, validated to
1e-10 by `test_slippage_attribution_components_sum_exactly`.

</details>

**Q22. Where does this project's impact model documentedly understate true cost, the illiquid/thin-market edge case?**

<sub>Tags: `L2`, `execution`, `numeric`</sub>

<details><summary>Answer</summary>

`docs/VALIDATION.md` §8, failure mode 2 ("Impact model misspecification for
illiquid names"), states this directly: the square-root law is calibrated on
liquid equities, and for names traded above roughly 10% ADV participation,
real costs are convex-worse than sqrt because of book depletion and
information leakage, the capacity curve and POV caps understate the pain in
that regime. The simulator's only guard against this is a blunt one:
`IntradayMarket.execute` raises a `ValueError` if a child order would exceed
100% of a bucket's market volume, which prevents an impossible fill but does
nothing to correct the cost *shape* below that hard limit.

</details>

**Q23. What happens mechanically in the simulator, per `docs/VALIDATION.md`, when volume in a bucket is zero but the schedule still routes an order there?**

<sub>Tags: `L2`, `execution`</sub>

<details><summary>Answer</summary>

`IntradayMarket.execute` raises an informative `ValueError`, "bucket {j} has
zero market volume but the schedule routes {q[j]:.0f} shares there; reschedule
around the halt/auction", rather than silently producing a divide-by-zero or
NaN fill (`test_zero_volume_bucket_raises_informative_error`). By contrast, a
POV schedule handles the same zero-volume bucket gracefully by skipping it and
continuing (`test_zero_volume_bucket_pov_skips_it`), since `pov_schedule`
naturally trades zero when `participation * V_j = 0`. This distinction, hard
failure for a schedule that assumed liquidity that wasn't there, versus
graceful degradation for a schedule that adapts to realised volume, is exactly
the operational argument for participation-based scheduling around thin or
halted markets.

</details>

**Q24. Why does FX execution explicitly exclude "internalisation" from its cost model, and what does that mean for interpreting its cost numbers?**

<sub>Tags: `L2`, `execution`, `numeric`</sub>

<details><summary>Answer</summary>

`docs/METHODOLOGY.md` (FX) §4 states that major FX dealers internalise 60–90%
of EURUSD flow against opposing client flow, so a real bank algo's realised
impact can be far below what a pure external-liquidity model predicts. The
project models only external liquidity consumption (assumption A7) and flags
its own cost numbers as an *upper bound* for an internalising desk, "a
documented simplification, not an accident." This is analogous to the equity
project's illiquid-name caveat: both projects are explicit that their
calibrated impact numbers are valid within the modeled regime but not a
universal cost oracle outside it.

</details>

**Q25. What is the "last-look" mechanism modeled in the FX project, and why does it matter for algorithmic execution specifically?**

<sub>Tags: `L1`, `execution`, `numeric`</sub>

<details><summary>Answer</summary>

On dealer (non-firm) FX streams, `LastLookVenue` in `fx_algo` models a dealer
holding a quoted order briefly (`hold_seconds = 2`) before accepting, with
rejection probability rising logistically in how far the price has moved
against the dealer during the hold (`p = expit((m − 0.6)/0.2)`), asymmetric by
design, moves in the dealer's favor are filled. A rejected child resubmits at
the post-move price plus an aggression penalty. METHODOLOGY.md §6 quantifies
the resulting "trap": despite quoting a 40% tighter spread, the last-look
stream is cheaper for uninformed flow (−0.018 pips) but strictly more
expensive for informed flow (+0.275 pips), because adverse selection converts
the quoted spread saving into rejection cost. An execution algorithm that
routes purely on quoted spread, ignoring rejection dynamics, will
systematically mis-rank venues for informed (alpha-bearing) flow.

</details>

**Q26. How does the WM/R 4pm fix benchmark illustrate the interaction between execution scheduling and benchmark design?**

<sub>Tags: `L2`, `execution`, `numeric`</sub>

<details><summary>Answer</summary>

`docs/METHODOLOGY.md` (FX) §2 explains that the pre-2015 fix was computed over
just a 60-second window, which several dealers exploited by trading on client
fix-order information inside that window to move the print in their favor
("The Cartel," ~$10bn in fines). The 2015 reform widened the window to 5
minutes specifically to make it harder to move. The project models the
post-2015 mechanics: `sessions.fix_window_mask` and `tca.fix_benchmark`
compute the fix as the TWAP of mids over a 5-minute window centered on 16:00
London, and a dedicated `fix_schedule` executes flat across exactly that
window. The print (`tca.fix_print`) is built from market observations drawn
independently of the execution, so tracking error has a real variance: over
200 synthetic replications the fix schedule tracks the print to +1.02 pips
mean with 0.54 pips sd, against about 9.9 pips sd for a 3-hour TWAP, and its
own permanent impact moves the print by 0.035 pips (`docs/VALIDATION.md` §5;
v2.0.0 reported +0.50 and 0.078 because the 1-minute grid scaled impact by the
bucket vol). The lesson generalizes beyond FX: any execution schedule that
trades *inside* the window used to compute its own benchmark can move that
benchmark, which is why benchmark design (window length, what counts as
"inside") is itself a control, not just a measurement choice.

</details>


## Set 13: Regime-Switching Quant Strategy

**Q1. What does "regime-switching" mean as a trading-strategy premise, and why should it beat a strategy with one fixed rule set?**

<sub>Tags: `L2`, `regimes`</sub>

<details><summary>Answer</summary>

The premise is that markets are not stationary: they spend long stretches in a
calm, trending, low-correlation state and shorter stretches in a stressed,
high-vol, high-correlation state, and the statistics that matter for a
portfolio, mean return, volatility, cross-asset correlation, differ materially
between the two. A single fixed rule (e.g. always fully invested, or a static
vol target) is calibrated to the blend of both states and is therefore wrong
in both. A regime-aware strategy infers which state is currently more likely
and conditions its behavior, position size, which book it trades, on that
inference, so it can behave differently in a 2022-style grinding bear than in
a calm 2021 bull instead of applying one policy everywhere.

</details>

**Q2. Concretely, which regime-detection method does this project implement: is it a Markov-switching model, a volatility-threshold classifier, or something else?**

<sub>Tags: `L3`, `regimes`, `code-reading`</sub>

<details><summary>Answer</summary>

It is a Gaussian Hidden Markov Model (HMM) fit by Baum–Welch EM, implemented
from scratch in `src/eq_regime/hmm.py`, sitting downstream of a features → PCA
→ HMM pipeline (`docs/METHODOLOGY.md` §1). A hidden state `s_t` follows a
first-order Markov chain with transition matrix `A`, and observations are
conditionally Gaussian, `x_t | s_t = k ~ N(μ_k, Σ_k)`. A volatility-threshold
rule (the 200-day moving average) and a GMM (a "no-dynamics" special case of
the HMM) are both present in the codebase, but only as an explicit benchmark
and a k-selection/cross-check tool respectively, not as the trading model.

</details>

**Q3. Why an HMM instead of the two obvious alternatives, a simple threshold rule and a static Gaussian mixture model?**

<sub>Tags: `L3`, `regimes`</sub>

<details><summary>Answer</summary>

`docs/METHODOLOGY.md` §2 lays out the comparison directly: a threshold rule
(VIX > 25, price vs 200d MA) has no notion of persistence or uncertainty, it
is a binary flag with an arbitrary cutoff. A GMM classifies every day
independently (i.i.d.), so its posterior "flip-flops" day to day with no
transition structure. The HMM adds exactly one thing on top of the GMM's
emission model: a transition matrix, which gives regime persistence (expected
duration `1/(1-p_ii)`), a filtered probability that blends today's evidence
with yesterday's belief, and a principled way to avoid noise-driven flips. The
threshold rule is kept in the codebase specifically as the benchmark to beat:
if the HMM can't beat it net of costs, its extra complexity isn't earning its
keep.

</details>

**Q4. What role does the GMM in `gmm.py` play if it isn't the trading model?**

<sub>Tags: `L3`, `regimes`, `numeric`</sub>

<details><summary>Answer</summary>

Two roles, both stated explicitly in `docs/METHODOLOGY.md` §2. First,
BIC-based selection of the number of hidden states `K`, since the GMM is
algebraically the special case of the HMM where every row of the transition
matrix is equal (no dynamics), which makes it a cheap way to test how many
clusters the data supports before paying for a full Baum–Welch fit. Second, an
independent cross-check of the HMM's emission parameters, the project
validates the from-scratch GMM against `sklearn.mixture.GaussianMixture`
(per-observation log-likelihood delta measured at 3.4e-14,
`docs/VALIDATION.md` §1) and separately validates the HMM against `hmmlearn`,
so the two implementations corroborate each other.

</details>

**Q5. What is a "filtered" probability, what is a "smoothed" probability, and why does this project insist that only filtered probabilities ever reach the trading strategy?**

<sub>Tags: `L2`, `regimes`, `code-reading`</sub>

<details><summary>Answer</summary>

Filtered: `α̂_t(k) = P(s_t = k | x_{1..t})`, computed causally from the
forward recursion using only data up to and including `t`. Smoothed: `γ_t(k) = P(s_t = k | x_{1..T})`,
computed from the backward pass over the *entire* sample, meaning it
conditions on future observations. `docs/METHODOLOGY.md` treats this as the
central point of the project: smoothed probabilities look far cleaner around
regime turns because they've effectively seen the future, so backtesting on
them is lookahead bias that overstates performance. Every trading signal in
`detection.py` and `backtest.py` is built exclusively from filtered
probabilities, and this is not just a stated convention, it's test-enforced.

</details>

**Q6. How is the filtered-vs-smoothed causality claim actually tested, rather than just asserted in the docs?**

<sub>Tags: `L3`, `regimes`</sub>

<details><summary>Answer</summary>

Via mutation tests. `docs/VALIDATION.md` §3 and §6.1 describe perturbing
(mutating) observations *after* time `t` and checking what happens to the
outputs at `t`: the filtered probability at `t` must be unchanged (drift ≤
machine epsilon) because it mathematically cannot depend on future data, while
the smoothed probability at `t` is expected to shift by "4+ orders of
magnitude more," which is the quantitative proof it does use the future. This
is implemented in `tests/test_detection.py::TestCriticalCausality` at the
detection layer and `tests/test_backtest.py::TestNoLookahead` at the
full-pipeline/ledger layer, so the "no lookahead" claim is checked end to end,
not just at one module boundary.

</details>

**Q7. Show me the concrete numeric evidence that smoothed probabilities are "clairvoyant" around a regime turn.**

<sub>Tags: `L3`, `regimes`, `numeric`</sub>

<details><summary>Answer</summary>

`docs/VALIDATION.md` §3 gives a specific table around the first true bull→bear
transition (day index 320, 2016-03-25): at t−1 (still in the "transition" true
state) the filtered `P_bear` is 0.103, but the smoothed `P_bear` is already
0.743, three-quarters bear a full day *before* the true regime changes,
because the smoother has read what happens next. The filtered probability only
crosses into clearly-bear territory at t+1 (0.973). That gap between 0.103 and
0.743 at the same date is the entire lookahead-bias problem in one row.

</details>

**Q8. What is the core risk of any regime-based strategy, the failure mode that makes these strategies look great in a backtest and fail live?**

<sub>Tags: `L2`, `regimes`</sub>

<details><summary>Answer</summary>

Overfitting to regime boundaries that were only obvious in hindsight. If you
choose where the "bear" episodes start and end by eyeballing the whole price
history, the classifier will look perfect in-sample, it's fit to data it
already knows the outcome of, and can degrade sharply out-of-sample because a
live detector has to infer the state from noisy, incomplete evidence in real
time, without the benefit of already knowing how the story ends. This
project's own methodology names the analogous statistical version of that risk
explicitly: smoothed probabilities are exactly "peeking at the whole history"
in miniature, applied per-day instead of per-regime-boundary, and the project
treats using them for trading as the single worst mistake a regime strategy
can make.

</details>

**Q9. How does this project's backtest guard against that overfitting risk structurally, not just at the level of "don't use smoothed probabilities"?**

<sub>Tags: `L3`, `regimes`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Two structural defenses, per `docs/METHODOLOGY.md` and `backtest.py`. First,
walk-forward, expanding-window refitting: `walk_forward_backtest()` refits the
HMM every `refit_every=63` trading days (a quarter) on an expanding window,
and the docstring in `backtest.py` states the timing convention explicitly,
"the HMM used at `t` was last refitted on features strictly BEFORE its refit
date and never sees data after `t`." Second, a null-data false-positive guard
(`docs/VALIDATION.md` §4): on synthetic no-regime GBM panels the strategy must
not beat random timing at equal exposure, judged on Sharpe across 12 seeds
(measured +0.08, SE 0.10, inside the 3-SE bound), and with equal drifts across
regimes it must not beat plain vol targeting (−0.01, SE 0.06), rather than
manufacture spurious alpha, and BIC on that null data is required to prefer
K=1. Both are asserted in `tests/test_null_guard.py`, so "the model doesn't
hallucinate regimes when there aren't any" is a testable property, not a hope.

</details>

**Q10. Isn't picking K (the number of regimes) itself a place where hindsight overfitting can creep back in?**

<sub>Tags: `L3`, `regimes`</sub>

<details><summary>Answer</summary>

Yes, and the docs treat it as a governance problem rather than a purely
statistical one. `docs/VALIDATION.md` §5 lists "overfitting k" as a named
failure mode: BIC on a finite sample can select a K that fits noise, and more
states mechanically produce shorter durations and more churn. The mitigation
in `docs/DESK_GUIDE.md` §3 is procedural: K is fixed at 3 between annual
reviews, and changing it requires BIC evidence *and* economically distinct
state volatilities *and* out-of-sample confirmation, not just "BIC went up."
`docs/METHODOLOGY.md` assumption 3 adds the failure-mode detail: if the true
world has more states than the model allows, EM doesn't fail loudly, it
silently absorbs the extra structure into existing states (e.g. a 2-state fit
on a 3-state world merges "calm" and "transition," understating risk right
before turns, quantified in `docs/VALIDATION.md` §5.4).

</details>

**Q11. Walk-forward validation sounds expensive: what specifically does `walk_forward_backtest()` do differently from a full-sample fit?**

<sub>Tags: `L3`, `regimes`, `code-reading`</sub>

<details><summary>Answer</summary>

Looking at `src/eq_regime/backtest.py::walk_forward_backtest`, it builds
point-in-time features (`build_features`), then calls
`expanding_fit_detect(..., min_train=252, refit_every=63)`, which fits the HMM
only on an expanding training window and refits it periodically rather than
once on the whole series. Contrast that with the "parameter recovery" exercise
in `docs/VALIDATION.md` §2, which explicitly fits a full-sample HMM to check
the model can recover known synthetic ground-truth parameters, that
full-sample fit is a calibration sanity check, never the version used to
generate the P&L numbers in §7. The walk-forward version is the only one whose
returns are reported as "the backtest."

</details>

**Q12. Why are log returns used for return aggregation in this backtest instead of simple returns?**

<sub>Tags: `L3`, `regimes`</sub>

<details><summary>Answer</summary>

Because log returns are additive across time and simple returns are not
(across assets it is the other way round: a portfolio's simple return is the
weighted sum of its assets' simple returns, and log returns do not add that
way): the log return over two periods is exactly `r_1 + r_2`, while the
two-period simple return is `(1+r_1)(1+r_2) - 1 ≈ r_1 + r_2` only
approximately (the cross term `r_1 r_2` is dropped). That makes log returns
the natural unit for anything that sums or averages returns over time, feature
construction (`build_features`), the HMM's Gaussian emission model, and vol
targeting in `strategy.py::vol_target_scale` all consume log returns, and
`docs/METHODOLOGY.md` §6 states this as a project-wide convention: "Daily
log-returns; vols annualised ACT/252." Simple returns are still needed
elsewhere, because P&L compounding is a multiplicative, not additive, process,
which is exactly what `np.expm1` bridges (see Q13).

</details>

**Q13. Where in the code does `np.expm1` show up, and what numerical problem does it solve?**

<sub>Tags: `L3`, `regimes`, `code-reading`</sub>

<details><summary>Answer</summary>

In `src/eq_regime/backtest.py::walk_forward_backtest`: `asset_log_ret = np.log(prices / prices.shift(1)).iloc[1:]`
computes log returns once, and then both `log_ret = asset_log_ret.mean(axis=1)`
(fed to the HMM/vol-targeting side) and `simple_ret = np.expm1(asset_log_ret).mean(axis=1)`
(fed to `run_ledger` for compounding) are derived from that single
computation. `expm1(x)` computes `exp(x) - 1` directly rather than as two
separate floating-point operations, which matters because for small `x` (a
typical daily log return is on the order of 1e-3 to 1e-2), `exp(x)` is very
close to 1, and subtracting 1 from it afterward cancels almost all of the
significant digits, catastrophic cancellation. `expm1` uses a numerically
stable formula/series near zero so the small daily return is recovered to full
precision instead of losing digits to rounding.

</details>

**Q14. The module docstring says returns are converted "internally" for compounding, walk through why the ledger needs simple returns specifically, not log returns.**

<sub>Tags: `L3`, `regimes`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`run_ledger()` compounds P&L as `equity = initial_equity * np.cumprod(1.0 + net)`,
i.e. it treats each period's return as multiplicative growth: a portfolio
holding weight `w` earns `w * r_simple` of the asset's simple return, and
equity multiplies by `(1 + net_ret)` each day. That identity, `equity_t = equity_{t-1} * (1 + r_t)`,
is only true for simple returns; log returns would need to be summed and then
exponentiated instead, or the ledger's cost/weight arithmetic (which is linear
in return, e.g. `cost = cost_rate * |Δw|`, `gross = w * next_ret`) would need
to change form. So the pipeline uses log returns everywhere the additive
property is wanted (features, vol estimation) and converts to simple returns,
via the numerically careful `expm1`, exactly at the one place, ledger
compounding, where the multiplicative identity is what's actually needed.

Source: `python/equity/10-regime-switching/src/eq_regime/backtest.py` (`run_ledger`).

</details>

**Q15. How does the current code derive log and simple returns from one computation, and why does it matter?**

<sub>Tags: `L4`, `regimes`</sub>

<details><summary>Answer</summary>

The current `walk_forward_backtest` computes `asset_log_ret` exactly once and
derives both series it needs from that single array, `log_ret` by taking the
mean directly, `simple_ret` by applying `np.expm1` to the same array before
averaging, rather than computing log returns and simple returns via two
independent expressions (e.g. a second, separately-written `prices / prices.shift(1) - 1`
for the simple-return side). That's a small but meaningful numerical-hygiene
fix: two independently written return computations can silently drift out of
sync (different `shift`/`iloc` alignment, different NaN handling at the
warm-up row) and are strictly more surface area to audit for correctness,
whereas deriving both series from one already-log-returned array guarantees
they cover the same dates with the same indexing by construction, with `expm1`
doing the precision-preserving conversion rather than a second lossy
log/subtract round-trip.

</details>

**Q16. What tests specifically defend the `run_ledger` arithmetic, beyond just checking it runs without error?**

<sub>Tags: `L3`, `regimes`, `code-reading`</sub>

<details><summary>Answer</summary>

`docs/VALIDATION.md` §6 lists "ledger arithmetic: exact hand-computed
scenario, entry cost included" as a specific edge-case test in
`tests/test_edge_cases.py`, meaning the test constructs a small scenario with
known weights and returns and checks the ledger's `net_ret`/`equity` columns
against a value computed by hand, not just against another run of the same
code. `run_ledger`'s own docstring is precise about the two conventions that
would otherwise be easy to get subtly wrong: `dw.iloc[0] = w.iloc[0]`
(entering from flat charges a cost on day one) and the shift-by-one alignment
where weight decided at `t` earns the return realized from `t` to `t+1`, both
of which are exactly the kind of off-by-one that "hand-computed" tests are
designed to catch.

</details>

**Q17. What is the hysteresis band in `strategy.py`, and what specific problem does it solve?**

<sub>Tags: `L2`, `regimes`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`hysteresis_regime()` implements a two-threshold rule on the filtered bear
probability: enter the bear state only when `p_bear` rises strictly above
0.70, and exit only when it falls strictly below 0.30; inside the band `[0.30, 0.70]`
the previous state simply persists. This replaces a naive rule ("bear if
`p_bear > 0.5`"), which flips every time noisy probability crosses one line,
near a regime boundary the filtered probability oscillates around that line
for days, so a single-threshold rule pays transaction costs on each crossing
while capturing no real signal. `docs/METHODOLOGY.md` §3 quantifies the effect
directly: on a noisy probability path, turnover falls from 263 trades to 48
(−82%) with hysteresis, while on a genuine two-switch path turnover falls from
24 to 8 and both real switches are still captured.

</details>

**Q18. Are the hysteresis boundary values themselves, exactly 0.70 or exactly 0.30, tested, or only the general behavior?**

<sub>Tags: `L3`, `regimes`, `code-reading`</sub>

<details><summary>Answer</summary>

Tested precisely, and the convention is strict-inequality on both ends.
`strategy.py::hysteresis_regime`'s docstring states: "at exactly `p == enter`
the rule does NOT enter (strict `>`), at exactly `p == exit` it does NOT exit
(strict `<`)"; the band is closed. `docs/VALIDATION.md` §6 lists this as its
own edge case, "probabilities exactly at hysteresis thresholds: no flip
(strict inequalities, documented convention)", with test coverage in
`tests/test_edge_cases.py::TestBoundaries` and `tests/test_strategy.py`. That
level of care about a single inequality direction is the kind of detail that
matters in practice: get it backwards and a probability sitting exactly on the
boundary trades every day.

</details>

**Q19. How does the strategy translate a hysteresis-confirmed regime into a portfolio weight, including the "unconfirmed bear" case?**

<sub>Tags: `L3`, `regimes`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`regime_target_weight()` in `strategy.py` maps the HMM's raw `argmax` label
(bull/transition/bear) to a base weight (1.0/0.5/0.0), but the hysteresis flag *overrides*
that label: whenever `bear_flag[t]` is True the weight is forced to
`bear_weight` regardless of the raw label, and whenever it's False, an
unconfirmed "bear" label (probability still inside the band, not yet past
0.70) is floored at `transition_weight` rather than allowed to fully de-risk.
That means full de-risking only ever happens through the confirmed hysteresis
band, never off a single noisy argmax flip, the docstring states this
explicitly: "de-risking obeys the banded rule rather than raw argmax flips."

</details>

**Q20. What does vol targeting add on top of the regime weight, and why is it computed only from trailing (past) data?**

<sub>Tags: `L2`, `regimes`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`vol_target_scale()` scales the regime-conditional weight by `target_vol / trailing_21d_realized_vol`,
clipped at a maximum leverage of 1.5x. It's trailing-only, using returns "up
to and including `t`", so that the scale decided at the close of `t` is a
genuine ex-ante forecast applied to the position held over `t → t+1`,
consistent with the rest of the pipeline's no-lookahead discipline; using a
centered or forward-looking vol estimate here would reintroduce exactly the
kind of lookahead bias the filtered-vs-smoothed distinction is designed to
eliminate elsewhere. `docs/VALIDATION.md` §6 separately validates that this
actually hits its target: "vol targeting realizes the ex-ante target within
15% on constant-vol data."

</details>

**Q21. How does a regime-switching strategy differ from a plain trend-following or mean-reversion strategy applied on its own?**

<sub>Tags: `L3`, `regimes`</sub>

<details><summary>Answer</summary>

A trend-following or mean-reversion rule applies one fixed logic to price
action everywhere, e.g. always buy strength, or always fade extremes,
regardless of the surrounding market context. A regime-switching strategy
instead conditions its *entire policy* on an inferred state:
`docs/DESK_GUIDE.md` §1 describes how the filtered probabilities become
factor-rotation weights ("bull: momentum/size; bear: quality/low-vol;
transition: blend"), meaning the strategy can effectively behave like a
trend-follower in one state and like something far more defensive in another.
This project even keeps the 200-day-MA rule, a classic trend rule, in the
codebase specifically as a *benchmark* that the regime strategy must beat net
of costs (`walk_forward_backtest` runs `ma_timing_weights` on the same dates
with the same cost model), which makes the distinction concrete: one is a
single always-on rule, the other is a policy that switches between behaviors
based on a probabilistic state estimate.

</details>

**Q22. What is the FX-specific twist on "regime" in `python/fx/10-regime-switching`, and why does the FX project treat it as a distinct economic story rather than a generic reuse of the equity approach?**

<sub>Tags: `L2`, `regimes`</sub>

<details><summary>Answer</summary>

FX regimes there are framed specifically as risk-on/risk-off (RORO): risk-on
funds cheap carry trades (borrow JPY/CHF, lend AUD/NZD/EM) that grind higher
with low daily vol; risk-off is that crowded position unwinding violently,
with carry/EM currencies gapping down, havens bid, vol multiplying, and
pairwise correlations spiking as "every USD pair trades the same one trade."
`docs/METHODOLOGY.md` (FX) also carries a third, economically distinct state,
a "USD squeeze", where a dollar-funding scramble makes *everything* fall
against USD, including normal safe havens, which is exactly what fails during
2008/March-2020-style episodes and is why the FX project sometimes prefers k=3
(RORO + squeeze) over a plain two-state risk-on/risk-off split.

</details>

**Q23. Given that framing, what specific book does the FX strategy trade in each detected regime?**

<sub>Tags: `L3`, `regimes`, `numeric`</sub>

<details><summary>Answer</summary>

Per `docs/METHODOLOGY.md` (FX) §7: in `risk_on` it runs a rank-carry basket,
long the top-3 yielding currencies and short the bottom-3, dollar-neutral, to
harvest the rate differential while vol is low; in `risk_off` it flips to long
JPY+CHF against the risk block (AUD, NZD, EM), i.e. it explicitly cuts carry
and owns the unwind rather than staying exposed to it; in `usd_squeeze` it
goes long USD against everything, because that's the one state where even the
usual haven currencies fail and only the dollar itself works. All books are
vol-targeted at 10% annualized using a trailing 63-day covariance, capped at
4x leverage, and pay carry accrual plus pip-denominated transaction costs,
this is a much richer regime-conditional action set than the equity project's
single bull/transition/bear equity-weight dial.

</details>

**Q24. Does the FX project use the same filtered-probability, hysteresis-band discipline as the equity project, or something different?**

<sub>Tags: `L3`, `regimes`, `numeric`</sub>

<details><summary>Answer</summary>

The core discipline is the same, filtered-only probabilities, enforced by an
analogous mutation test
(`tests/test_detection.py::test_filtered_only_past_mutation`,
`tests/test_backtest.py::test_no_lookahead_mutation`), but the confirmation
rule is stricter in form: `docs/METHODOLOGY.md` (FX) §7 requires the
challenger regime's filtered probability to be ≥0.70 (or the incumbent's <0.30)
for **2 consecutive days**, not just a single-day threshold crossing. The doc
is explicit that this buys lower turnover and flicker resistance at the price
of additional detection lag, and frames it as "a governed, measured trade-off"
documented quantitatively in `docs/VALIDATION.md`, i.e. the same explicit
treatment of costs as the equity project's hysteresis band, just with an added
time-confirmation dimension appropriate to how fast FX risk episodes move.

</details>

**Q25. What is the real edge case unique to regime-switching strategies, the period right when the market actually flips state, and how is its cost documented here?**

<sub>Tags: `L2`, `regimes`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

The core failure mode is detection lag: a filter needs evidence to *accumulate*
before it moves, so by construction it identifies a regime change some days
after it has actually happened, and that gap is exactly when the strategy is
still positioned for the old regime. `docs/VALIDATION.md` §5 quantifies this
on the synthetic 3-state panel: mean bear-entry detection lag of 1.5 days
(median 1) and mean bear-exit lag of 1.0 days, and then goes further with a
"flip-aftermath" report (implemented as `flip_aftermath()` in
`src/eq_regime/risk.py`) measuring the mean 10-day P&L immediately after a
flip to bear: −0.15% on average and −3.2% in the worst case over 17 flips
(synthetic, one path). `docs/DESK_GUIDE.md` §4 translates this to a real
scenario: on a COVID-March-2020-style fast crash, the first de-risking trade
would realistically trigger only after roughly −8 to −12% of drawdown had
already happened, the doc's own framing is "the detection lag is the price of
admission... a regime model is a loss limiter, not a crash predictor."

</details>

**Q26. Are there edge cases around the regime transition itself beyond simple lag, e.g. false alarms, or the model being fooled by a single noisy day?**

<sub>Tags: `L3`, `regimes`</sub>

<details><summary>Answer</summary>

Yes, two distinct sub-cases are both documented and tested. First, "false
alarms in corrections" (`docs/DESK_GUIDE.md` §4, citing 2015-Q3/2018-Q4
analogues): a sharp vol spike can trigger a bear entry right near the local
low, and the recovery re-entry then lags, each false alarm costs "cost × 2
turns + missed rebound days," which is exactly what the flip-aftermath table
is measuring. Second, single-day Gaussian-emission mis-assignment
(`docs/METHODOLOGY.md` assumption 1, `docs/VALIDATION.md` §5.5): because daily
returns are fatter-tailed than the Gaussian emissions the HMM assumes, a
single extreme day can look overconfidently like a regime shift under the
model, and the hysteresis band plus a minimum-duration filter (removing
one/two-day flickers) are the two mechanisms that absorb these single-day
flips before they reach the order pipe, with `docs/DESK_GUIDE.md` §3 stating
explicitly that the governance response to a false alarm is *not* to widen the
band ad hoc (which would invalidate the backtest) but to log the episode and
review bands only on an annual cycle.

</details>

**Q27. What did the training run of the regime study (`regime-fx-basket`) find, and what happens next?**

<sub>Tags: `L3`, `regimes`, `numeric`</sub>

<details><summary>Answer</summary>

The hypothesis was that next-day returns are negative in the high-volatility
state. In training (ECB data, 1999 to 2021) the high state had a mean of +1.2%
a year against +0.3% in the low state, HAC t = +0.27, so the hypothesis is not
supported. The hold-out (2022-2025) was then run once: on 161 of 1,019
high-vol days the basket earned +12.5% a year against -4.0% on the rest, HAC t = +0.94
with 4 lags. The rule rejects H0 only if t < -1.645, so H0 is not rejected.
The effect has the opposite sign to the hypothesis, but t = +0.94 is not
evidence of a positive effect either. The hold-out is spent, so the study is
closed.

Source: `README.md` (the regime hold-out paragraph) and `python/fx/10-regime-switching/README.md`.

</details>


## Set 14: The Foundations Trilogy

**Q1. `01-risk-metrics`'s README says "this is a single-asset project, portfolio-level risk (correlation, diversification) is out of scope." Concretely, what does staying single-asset let the project skip that `python/equity/03-var-es-engine` cannot?**

<sub>Tags: `L1`, `foundations-trilogy`, `code-reading`</sub>

<details><summary>Answer</summary>

A single return series has one variance and no cross-terms, so every metric in
F1, historical/Gaussian/Cornish-Fisher VaR, Expected Shortfall,
Sharpe/Sortino, the Kupiec backtest, is a function of one vector of numbers.
`python/equity/03-var-es-engine` prices a book of correlated positions
(`demo_portfolio()`, `demo_covariance()`), so its parametric and Monte Carlo
VaR need a covariance (or scale) matrix and a Cholesky factorisation
(`ev.parametric_var(pf.delta_exposures(), cov, ...)`, `ev.monte_carlo_var(pf, cov, ...)`)
just to generate a jointly consistent set of scenarios before any VaR
arithmetic starts. Assumption A9 in F1's `docs/METHODOLOGY.md` names this
directly: "Real books hold multiple correlated positions; portfolio VaR/ES
require a covariance (or copula) model and are *not* simply the sum of
single-asset VaRs." F1 is a smaller problem by construction, and that
smallness is the point, it isolates "which distributional assumption is right"
from "how do positions co-move," so a reader can see the first question
answered cleanly before the second one is layered on.

</details>

**Q2. If you wanted to turn F1 into a two-asset risk engine, what would actually have to change, and why is that not a small patch?**

<sub>Tags: `L1`, `foundations-trilogy`, `code-reading`</sub>

<details><summary>Answer</summary>

It is not just "run the same functions twice and add them", VaR and ES are not
additive across positions (that's exactly why ES's sub-additivity is a *theorem*
worth stating in F1's `docs/METHODOLOGY.md` §4, not a triviality). You would
need a covariance matrix estimate (sample or EWMA), a way to draw or simulate
correlated returns (Cholesky factorisation, as `03-var-es-engine`'s
`monte_carlo_var.py` does), and a joint P&L definition before any of F1's
historical/Gaussian/Cornish-Fisher logic could be reused on the *portfolio*
return series rather than each asset's own. The single-asset version has no
correlation to estimate, no factor mapping, and no joint-tail behavior to
worry about, which is precisely the machinery `03-var-es-engine`'s
`portfolio.py` and `monte_carlo_var.py` exist to carry. F1 answers "which VaR
method is right on this one series"; the multi-asset engine answers that
question *and* "how do these series interact," and the second question is a
materially bigger modeling problem, not a bigger loop.

</details>

**Q3. What is the "no-look-ahead" bug class in backtesting, in general terms: what specifically goes wrong?**

<sub>Tags: `L0`, `foundations-trilogy`</sub>

<details><summary>Answer</summary>

It is using information on day t that would not actually have been available
to a trader at the moment the trade was placed. The most common concrete form
is computing a signal from day t's closing price and then trading as if you
entered the position *at that same close*, but you cannot know a bar's close
until it happens, so a strategy that "trades on the close it used to generate
the signal" is silently assuming perfect foresight of the rest of that day.
The bug is dangerous precisely because it is invisible in the code's
arithmetic: `signal * return` on the same day looks like an ordinary
vectorised backtest line, and it will produce a return series, a Sharpe ratio,
an equity curve, all internally consistent, all wrong, because part of every
day's "prediction" is really that day's own outcome leaking into its own
evaluation.

</details>

**Q4. How exactly does F2's `run_backtest` avoid this, and where is the line?**

<sub>Tags: `L1`, `foundations-trilogy`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`src/eq_signal_backtest/engine.py` decides the signal at the close of day t
and fills it during day t+1 at a modelled price `p_fill = p_t + f (p_{t+1} - p_t)`,
with `f = 0.5` by default (`ExecutionConfig`; `f = 1` is a next-close fill,
and a close-auction mode fills at `p_t` and pays an explicit auction cost). In
code, `position = signal.shift(1).fillna(0.0)` is the holding after the day's
fill and `old = signal.shift(2).fillna(0.0)` the holding before it; the old
position earns `p_fill / p_t - 1` of the day and the new one earns `p_{t+1} / p_fill - 1`.
Because neither leg can depend on the close of the day the signal was formed
before that close is known, there is no path by which the strategy's P&L can
use information it would not have had. The lag is not a validation check
bolted on afterward: it is in the lines that define what "trading the
strategy" means.

</details>

**Q5. `tests/test_engine.py` has a "detector test", `test_cheat_profits_from_a_jump_engine_does_not`. What does it actually construct, and why is that a stronger proof than reading the code?**

<sub>Tags: `L1`, `foundations-trilogy`, `numeric`</sub>

<details><summary>Answer</summary>

It builds a price series that is flat and then jumps 30% overnight, with a
signal engineered to fire exactly on the jump day. A same-day ("cheat")
execution captures that jump (`cheat_equity.iloc[-1] > 1.25`); the engine's
position on the jump day was set by the *previous* day's signal, decided
before the jump happened, so its equity is unchanged (`== 1.0` to `1e-9`).
This is stronger than inspecting `position = signal.shift(1)` and agreeing it
looks right, because it constructs a case where the bug, if present, would be *profitable
and visible*, and then shows the code under test does not take the free money
available to a cheating implementation. A structural assertion
(`position.iloc[t] == signal.iloc[t-1]` for every `t`, per
`test_position_equals_prior_day_signal_structurally`) proves the mechanism;
the jump test proves the mechanism actually matters.

</details>

**Q6. Why does it matter that `position.iloc[0]` is forced to `0.0` rather than left as `NaN`?**

<sub>Tags: `L1`, `foundations-trilogy`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

There is no day `t=-1` to supply a signal for day 0, so the correct answer to
"what position did the strategy hold before it had ever observed a signal" is
"none", `0.0`, not an undefined value that would need special-casing
downstream. `test_first_day_position_is_zero_not_nan` in
`tests/test_engine.py` pins this regardless of what `signal.iloc[0]` happens
to be. It is a small detail, but it follows the same rule as the shift itself:
an engine that let day 0 default to `NaN` would either crash on the first `NaN * return`
multiplication or, worse, silently propagate `NaN` through the whole equity
curve, exactly the kind of "runs fine, produces a wrong number" failure the
no-look-ahead lag is designed to prevent elsewhere.

</details>

**Q7. F2's headline out-of-sample result is a losing number. Why is that the right outcome to publish rather than a failure of the project?**

<sub>Tags: `L1`, `foundations-trilogy`, `numeric`</sub>

<details><summary>Answer</summary>

A backtest's job is to tell you the truth about a strategy, not to produce an
impressive number; the entire evaluation machinery (train/test split,
walk-forward, cost accounting, no-look-ahead fills) exists to answer "would
this have actually worked". If the answer is "no," reporting that *is* the
deliverable; reporting only the flattering number the machinery was built to
catch would defeat the purpose of building it. On F2's synthetic generator the
100-seed result is that the crossover does not beat buy and hold:
out-of-sample strategy minus buy-and-hold Sharpe is −0.11 [−0.21, −0.00]
(`docs/VALIDATION.md` §6). A strategy that only looks good in-sample is a
warning sign about the evaluation, not a result worth publishing on its own.

</details>

**Q8. What happens to the Sharpe ratio between in-sample and out-of-sample on the bundled synthetic data, and what does that teach about parameter selection?**

<sub>Tags: `L1`, `foundations-trilogy`, `numeric`</sub>

<details><summary>Answer</summary>

On the seed-32 worked example (synthetic), the in-sample Sharpe is 0.86
(parameters `fast=40, slow=100` chosen by `select_best_params` on the training
window only) and the out-of-sample Sharpe is −0.19. Over 100 synthetic seeds
the selected in-sample Sharpe exceeds the out-of-sample Sharpe by 0.35 on
average (`docs/VALIDATION.md` §6). The mechanism is Assumption A5 in
`docs/METHODOLOGY.md`: picking the `argmax` Sharpe of a grid of 49 noisy
Sharpe estimates is, in expectation, biased upward relative to any cell's true
Sharpe, a "winner's curse" of multiple testing. The pipeline logs every grid
cell as a trial and reports the deflated Sharpe ratio and PBO against that
count. The in-sample number measures how good the best of many noisy estimates
looks by chance, and the out-of-sample test is what catches that inflation
instead of reporting it as skill.

</details>

**Q9. F2 also runs a walk-forward variant that re-selects parameters every year (−0.02 Sharpe on seed 32). Why does that matter more than either number alone?**

<sub>Tags: `L1`, `foundations-trilogy`, `numeric`</sub>

<details><summary>Answer</summary>

A single 70/30 split is one draw from history: a lucky or unlucky split
boundary could flatter or damn the strategy on its own (Assumption A6).
Walk-forward re-runs the selection across seven rolling formation/trading
windows, each time selecting parameters using only that window's formation
slice and scoring only its out-of-sample trading slice, then stitches the
trading segments into one continuous curve. On seed 32 (synthetic) the single
split gives −0.19 and the walk-forward −0.02, with per-window Sharpes swinging
from −1.59 to +1.65. One seed is still one path, so the project repeats both
protocols on 100 synthetic seeds: walk-forward Sharpe 0.39 [0.29, 0.49]
against buy and hold 0.44 [0.34, 0.54]. Two protocols, repeated across many
paths, pointing the same way is much stronger evidence than any single number.

Source: `python/foundations/02-trading-signal-backtest/docs/VALIDATION.md` (the 100-seed table and the seed-32 window Sharpes).

</details>

**Q10. The in-sample parameter grid has a broad "plateau", 39% of cells within 25% of the best Sharpe. Isn't a plateau usually read as evidence *against* overfitting?**

<sub>Tags: `L1`, `foundations-trilogy`, `numeric`</sub>

<details><summary>Answer</summary>

Conventionally yes: a single lucky cell in a sea of bad ones is the classic
signature of curve-fitting, while a broad plateau is supposed to mean the
result doesn't depend on one fragile parameter choice. F2's own synthetic data
shows why that reading is incomplete: on seed 32 the plateau is real (39% of
cells within 25% of the best) and the out-of-sample Sharpe is still negative.
A plateau says the in-sample result does not hang on one cell; it says nothing
about out-of-sample performance (`docs/VALIDATION.md` §6a). A wide plateau
distinguishes "one lucky cell" from "a lucky neighbourhood", but it cannot
distinguish either from "the whole in-sample surface is fitted to the same
noise".

</details>

**Q11. How does F2 quantify the effect of transaction costs on a fast-trading moving-average crossover, and what's the general lesson?**

<sub>Tags: `L1`, `foundations-trilogy`, `numeric`</sub>

<details><summary>Answer</summary>

`docs/VALIDATION.md` §7 varies `(fast, slow)` and `cost_bps` together on the
synthetic seed-32 path: the fast pair (10, 50), which trades 55 times over 10
years, loses 1.21 percentage points of CAGR going from 0 to 20 bps; the slow
pair (60, 250), 12 trades, loses only 0.24pp, roughly a 5x difference in cost
sensitivity. The selected pair (40, 100), 24 trades, sits in between. The
general lesson, stated in `docs/METHODOLOGY.md` A2: cost drag scales with
turnover, and the bias a cost-free backtest introduces is *systematic*, always
in the same direction: it flatters over-trading parameter pairs.

</details>

**Q12. Why does F2 insist `parameter_grid`/`select_best_params` always run with `cost_bps` set (default 5.0), and never cost-free?**

<sub>Tags: `L1`, `foundations-trilogy`</sub>

<details><summary>Answer</summary>

Because the absence of costs is a known bias of unknown size on any new
dataset, not a neutral default. `docs/VALIDATION.md` §7 is explicit that
whether the bias is large enough to *reverse* a ranking depends on the sample,
on the synthetic seed-32 path it does not, but that is a fact about this one
seed, not a property you can assume holds elsewhere. Running the search
cost-free would answer a different, easier question ("which parameters look
best if trading were free") that has no connection to what a desk could
actually capture; keeping costs in the search by default means the number the
search optimizes for is the same number that would matter on a desk.

</details>

**Q13. F2's Assumption A9 says a seed is part of a single-path result, and the headline numbers are a 100-seed distribution with seed 32 shown only as a worked example. Why call that out explicitly?**

<sub>Tags: `L1`, `foundations-trilogy`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Because seed selection is itself a place backtest-overfitting can hide, one
level removed from parameter selection. `docs/METHODOLOGY.md` A9 notes that
ten-year synthetic paths range from about −14% to +28% CAGR by seed. If a
project quietly tried several seeds and reported the one where the strategy
happened to work, it would be repeating the selection bias the train/test and
walk-forward apparatus exists to guard against, applied to the data instead of
the parameters. Reporting a distribution over 100 seeds removes the choice
altogether; seed 32 stays as the default of `generate()` only because its
buy-and-hold path is near the generator's central case, and it is labelled a
worked example, not evidence.

</details>

**Q14. Why does F3 implement Black-Scholes pricing and Greeks using only `math.erfc`, instead of calling `scipy.stats.norm.cdf`?**

<sub>Tags: `L1`, `foundations-trilogy`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`docs/METHODOLOGY.md` §1.4 states the reasoning directly: the standard normal
CDF is not a peripheral detail of Black-Scholes, it *is* the model, `N(d2)` is
the risk-neutral exercise probability, `N(d1)` weights the expected stock
receipt in the replicating portfolio. Calling `scipy.stats.norm.cdf` would
compute the right number in one line but leave that piece as a black box; the
point of a from-scratch replication is to understand every link from the SDE
to the price. `math.erfc` is chosen as the stopping point rather than building
the CDF from even more primitive parts because it is a single, well-tested,
general-purpose special function in the standard library. The CDF is written
`N(x) = 0.5*erfc(-x/sqrt(2))` rather than `0.5*(1 + erf(x/sqrt(2)))` because
the latter cancels to zero in the far left tail, while the `erfc` form keeps
full relative accuracy there; general-purpose math, not
option-pricing-specific machinery, so using it doesn't hide any model logic
the way a library CDF or a hypothetical `black_scholes_call()` would.

</details>

**Q15. F3's closed form already checks put-call parity and its own Greeks against finite differences. What does agreement with an *independent* Monte Carlo pricer prove that those internal checks cannot?**

<sub>Tags: `L2`, `foundations-trilogy`, `code-reading`</sub>

<details><summary>Answer</summary>

`docs/METHODOLOGY.md` §1.2 names the gap precisely: `call_price`/`put_price`
share one `_d1_d2` helper, so a bug in `_d1_d2`, a wrong sign, a misremembered
`+sigma^2/2` vs `-sigma^2/2`, a subtly wrong `N(x)`, would corrupt every
internal check identically and still pass all of them. Put-call parity,
monotonicity, and Greeks-vs-finite-differences are all checks *internal* to
the same closed-form code, so they share its blind spots by construction. The
Monte Carlo pricer in `monte_carlo.py` shares no code with the closed form: it
draws standard normals and builds `S_T = S0 exp((r - sigma^2/2)T + sigma sqrt(T) Z)`
directly from the SDE's solution, then averages discounted payoffs. If the
closed form had a sign or constant error, the two would disagree by far more
than Monte Carlo noise, they would have to be wrong in *exactly the same way*
to agree by coincidence across a convergence table spanning four sample sizes,
each within 3 standard errors, with the error bar shrinking at the correct
O(1/√n) rate. An independent implementation using a different numerical method
is the only kind of check that rules out a whole class of "consistently wrong
in the same way" bugs; a self-consistency check, no matter how many identities
it verifies, cannot.

</details>

**Q16. F3's methodology says the antithetic-variate Monte Carlo standard error must be computed from "pair averages," not from the raw `2m` draws treated as independent. Why does that distinction matter for the cross-check in Q15 to mean anything?**

<sub>Tags: `L1`, `foundations-trilogy`</sub>

<details><summary>Answer</summary>

Antithetic sampling makes the mirrored draws negatively correlated by design,
so treating the `2m` values as `2m` independent observations overstates the
standard error, by about a third on an ATM contract, per `docs/METHODOLOGY.md`
§1.2. That inflation is invisible to a test suite: every "agrees within 3
standard errors" check keeps passing, just against a bar that is too loose to
mean anything. The project's own framing generalizes the point: "a tolerance
expressed in units of your own error estimate is only as trustworthy as that
estimate... A validation suite that certifies 'within 3 sigma' while computing
sigma incorrectly is certifying its own arithmetic, not the model." The fix,
computing the standard error from the `m` independent pair averages, and
testing that reported error against the estimator's actual dispersion across
60 seeds (`docs/VALIDATION.md` §2.1), is what makes the Monte Carlo
cross-check in Q15 an independent witness rather than a check that would pass
no matter what the closed form did wrong.

</details>

**Q17. F3 produces a volatility smile from a model that assumes flat volatility. What is that experiment, and why publish a demonstrated failure of your own model?**

<sub>Tags: `L1`, `foundations-trilogy`, `numeric`</sub>

<details><summary>Answer</summary>

Assumption A1 in `docs/METHODOLOGY.md` states Black-Scholes assumes geometric
Brownian motion with constant volatility, so log-returns are exactly normal.
The experiment prices a strike ladder by Monte Carlo under a Student-t(df=4),
fat-tailed return distribution instead, then reads those prices back through
the constant-vol closed form to recover an "implied" volatility per strike. On
the reference numbers, that implied vol comes out higher at the wings than at
the money (27.6% at K=70, 22.9% at K=105, 28.6% at K=140, a 5.2-vol-point
smile) instead of the flat line constant-vol GBM predicts. A symmetric
fat-tailed distribution gives a symmetric smile; real equity-index options
since the 1987 crash show a downward skew instead, low strikes much richer
than high ones, which needs negative skewness or spot-vol correlation that
this experiment does not include. The project publishes this because it is the
necessary completion of "prove the model right where theory says it should be
right, and prove it wrong where reality says it should be wrong, with the same
codebase", a from-scratch rebuild that only ever showed the model succeeding
would be demonstrating agreement with itself, not understanding of where the
model's assumptions actually bind.

</details>

**Q18. F1's `docs/VALIDATION.md` says the Kupiec pass for historical VaR is "close to tautological" even though it's presented as a validation result. What does that mean, and what does the test actually falsify?**

<sub>Tags: `L1`, `foundations-trilogy`, `numeric`</sub>

<details><summary>Answer</summary>

Both VaR estimators in F1 are scored on the same window they were fitted to,
but historical VaR's 99% quantile is defined as (approximately) the empirical
1% worst observation of that exact sample, so of course roughly 1% of that
sample exceeds it; passing Kupiec is close to true by construction, not
evidence the estimator is good at predicting *new* data. `docs/VALIDATION.md`
§2 states this directly: "What the in-sample test *can* falsify is a model
whose distributional assumption is wrong, which is exactly what happens to the
Gaussian one", the Gaussian estimator makes a real, falsifiable claim (returns
are Normal(mu, sigma)) that the data can and does contradict (60 exceptions
against 25.2 expected, LR = 35.02, p ≈ 0, rejected), while the historical
estimator makes almost no claim to be wrong about in-sample. A genuine test of
predictive power needs an out-of-sample VaR re-estimated on a trailing window
and scored on the next day's return, which is exactly the discipline
`02-trading-signal-backtest` builds and `01-risk-metrics` explicitly defers to
`docs/DESK_GUIDE.md` §3.

</details>

**Q19. All three foundations projects ship at least one method they know is weaker, on purpose, next to a better one. What's the common teaching pattern?**

<sub>Tags: `L1`, `foundations-trilogy`</sub>

<details><summary>Answer</summary>

F1 computes Gaussian VaR specifically "to demonstrate its failure" against
historical VaR (`docs/METHODOLOGY.md` §3.2); F2 always runs its parameter
search *with* transaction costs because a cost-free version would silently
flatter over-trading parameters, and reports the in-sample Sharpe next to the
out-of-sample one specifically so the gap is visible rather than hidden; F3
ships a Monte Carlo pricer whose only job is to disagree with the closed form
if the closed form is wrong, and engineers a case (the smile experiment) where
the model's own assumption breaks. In each case the "worse" number or method
is not a mistake left in the codebase, it's instrumentation, there so the gap
between "what a naive approach would tell you" and "what's actually true" is a
measured quantity instead of an assertion a reader has to take on faith. That
is the shared lesson across the trilogy: doing quant work properly means
building the thing that could prove you wrong into the same pipeline as the
thing that could make you look good, and reporting whichever one the data
actually supports.

</details>

**Q20. Why do these three specific projects, not the equity or FX ones, form a natural starting trilogy for the whole portfolio?**

<sub>Tags: `L1`, `foundations-trilogy`</sub>

<details><summary>Answer</summary>

Each is the smallest version of a problem the rest of the portfolio solves at
full scale: F1 is single-asset risk (no covariance matrix) versus
`equity/03-var-es-engine`'s multi-asset book; F2 is a two-parameter, long/flat
crossover with a hand-checkable sensitivity grid versus a full walk-forward/ML
strategy stack; F3 is dependency-minimal, single-model, European-only
Black-Scholes versus `equity/01-options-pricing`'s vectorised multi-model
engine with C++/Rust ports. Every one of the three explicitly says so: F1
points to `equity/03-var-es-engine` for the multi-asset extension; F2's
assumptions register names the selection-bias controls it applies (trial log,
deflated Sharpe, PBO, a 100-seed distribution); F3's README calls itself "a
warm-up for" `equity/01-options-pricing` and states outright it is "not a
pricing library." They are the same ideas the rest of the portfolio scales up,
stripped down to the point where the core mechanism, a quantile disagreement,
an execution lag, an independent numerical witness, is the *whole* project
rather than one module inside a much larger one.

</details>

**Q21. What does the full-scale successor to each foundations project add, and why would meeting that machinery first make the core idea harder to see?**

<sub>Tags: `L0`, `foundations-trilogy`</sub>

<details><summary>Answer</summary>

`equity/03-var-es-engine` adds a covariance/factor model, Cholesky-simulated
correlated scenarios, Christoffersen's clustering test, Basel traffic-light
zones, and Acerbi-Székely ES backtesting on top of F1's single-series
VaR/ES/Kupiec. F2 itself already logs its grid as trials and reports a
deflated Sharpe ratio and PBO; the larger strategy projects
(`equity/05-pairs-trading`, `equity/10-regime-switching`) add multi-asset
books, volatility-scaled sizing, market impact and multi-seed confidence
intervals on every headline. `equity/01-options-pricing` adds multiple pricing
models, American exercise via a CRR tree, dividends, and vectorised
performance. Every one of those additions is real and matters on a trading
desk, but each is also a second layer of correctness-checking wrapped around a
mechanism (a quantile estimate, a lagged position, a closed-form price) that a
reader needs to understand cleanly first. Meeting the covariance matrix before
understanding why Gaussian VaR understates tail risk on one series, or meeting
the deflated Sharpe correction before seeing an unadjusted in-sample Sharpe
evaporate out-of-sample with your own eyes, means learning the fix before
feeling the problem it fixes, the validation machinery ends up looking like
ceremony instead of an answer to a question you've already asked.

</details>

**Q22. Practically, how should a student use these three projects relative to the portfolio's 10-area buildout?**

<sub>Tags: `L1`, `foundations-trilogy`</sub>

<details><summary>Answer</summary>

Read them first, and read them as complete stories, not reference material to
skim past on the way to the "real" projects, each one is small enough to hold
in your head in full: one return series, one signal with one execution rule,
one closed-form price with one independent witness. The goal at this stage is
to be able to explain, without looking anything up, why historical VaR beats
Gaussian VaR on fat-tailed data, why `position = signal.shift(1)` is the one
line standing between a valid backtest and a fabricated one, and what an
independent Monte Carlo pricer proves that a self-consistency check cannot.
Only after that intuition is solid does it make sense to move into
`equity/03-var-es-engine`, the bigger backtest engine, and
`equity/01-options-pricing`, at that point the covariance matrices, deflated
Sharpe corrections, and CRR trees read as answers to problems you already
understand, rather than machinery to memorize. Treat F1–F3 as the place the
ideas get built, and the 10-area equity/FX buildout as the place they get
scaled and hardened.

</details>


## Set 15: Cross-Language Engineering

**Q1. Why does this portfolio validate its C++/Rust engines against golden vectors generated FROM the Python reference's output, rather than having each language's engineer independently re-derive the pricing/VaR formulas from the math?**

<sub>Tags: `L4`, `cross-language`</sub>

<details><summary>Answer</summary>

Independent re-derivation sounds more rigorous, but it isn't: if the C++ and
Rust engineers both made the same conceptual mistake, say, both used the wrong
sign convention for dividend yield, or both mis-derived the Cornish-Fisher
correction term, their two implementations would still agree with each other
perfectly while both being wrong. Cross-language *agreement* would then be
indistinguishable from cross-language *correctness*, which is exactly the
failure mode a validation scheme is supposed to catch. Freezing golden vectors
instead proves a narrower, checkable claim. For the options engines the
expected values come from an independent mpmath oracle at 50 digits, not from
Python output; for the VaR engines they come from the already-validated Python
reference on deterministic inputs. Either way the claim is "the C++/Rust
engine reproduces this specific, independently anchored set of values", which
only holds if the ported code actually has the reference's semantics.

</details>

**Q2. `DIAGRAMS.md` diagram 2 says golden-vector agreement proves the C++/Rust engines "compute the same numbers as the Python reference," but explicitly does NOT prove "that the Python reference itself is correct." Where does that second burden of proof live instead?**

<sub>Tags: `L4`, `cross-language`</sub>

<details><summary>Answer</summary>

It lives in each Python project's own analytic-identity and convergence tests,
put-call parity, Greeks-vs-finite-difference, tree-to-Black-Scholes
convergence, and (for VaR/ES) Kupiec/Basel backtests, per
`docs/ARCHITECTURE.md` "Cross-language validation." Those tests check the
reference against independent mathematical truths (an identity that must hold
regardless of implementation, a limit that must be approached), not against
another implementation. Golden vectors and analytic identities are different
kinds of evidence by design: one says "matches a trusted source," the other
says "the trusted source itself is not internally inconsistent."

</details>

**Q3. Walk through the actual golden-vector generation pipeline end to end, for the options engines.**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

Per `docs/DIAGRAMS.md` diagram 2 and the files it names: (1)
`python/equity/01-options-pricing/tests/golden/generate_golden.py` computes
reference values from an independent oracle, not from the code under test:
closed forms and Greeks are evaluated in `mpmath` at 50 significant digits,
implied-vol cases invert an mpmath-computed quote whose true vol is known by
construction, and CRR trees run the same recursion in mpmath arithmetic. It
writes `tests/golden/golden_vectors.json`, where each record carries its
inputs and, for every output, a value and its own absolute tolerance computed
by the `tolerance_rules` stored in the file. (2)
`cpp/equity-options-engine/tools/gen_golden_header.py` reads that JSON and
emits `tests/golden_vectors.hpp`. (3)
`rust/equity-options-engine/tools/gen_golden_rs.py` reads the same JSON and
emits `src/golden.rs`. (4) Each engine's own test suite (pytest / GoogleTest /
`cargo test`) asserts its computed values against the committed constants,
each to its own tolerance. Every generator accepts `--check`, which fails if a
committed file is stale.

</details>

**Q4. Why are the golden vectors COMMITTED to the repo (as `golden_vectors.json`, `golden_vectors.hpp`, `golden.rs`) rather than regenerated at test time by invoking Python from the C++/Rust test run?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

A test that regenerates its own oracle at run time isn't testing against an
independent reference at all, it's testing against whatever the reference
computes *right now*, so a regression in the Python reference and a regression
in the C++/Rust port could land at the same time and the test would still
pass, having compared two simultaneously-broken outputs to each other.
Committing the vectors freezes the oracle: `tests/golden_vectors.hpp` and
`src/golden.rs` are fixed files under version control, so a C++/Rust test
failure means "this engine's output moved," full stop, independent of whatever
the Python reference happens to compute on a later run. A change to a golden
value shows up as a diff in a committed file, reviewed like any other code
change, which is only possible because the vectors are static, diffable,
committed artifacts.

</details>

**Q5. How does a golden vector cross the Python-to-C++ and Python-to-Rust boundary without losing precision?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

`gen_golden_header.py` and `gen_golden_rs.py` both format every double with
Python's `repr()` via a small `fmt()` helper, the shortest decimal string that
round-trips back to the exact same IEEE-754 bit pattern. That string is
emitted as a C++/Rust floating-point literal (`fmt()` appends `.0` if the repr
has no `.`/`e`/`E`, so it parses as a double/`f64` rather than an integer),
and both languages' floating-point literal parsers reproduce the identical
64-bit value on read-back. So the committed header and Rust module carry the
JSON values exactly, with no rounding introduced by the transport.

</details>

**Q6. Do the VaR/ES engines get their golden values the same way as the options engines?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

Yes. Each VaR/ES Python project has its own `tests/golden/generate_golden.py`
that writes `tests/golden/golden_vectors.json`, and the C++ and Rust engines
consume it through the same `tools/gen_golden_header.py` (to
`tests/golden_vectors.hpp`) and `tools/gen_golden_rs.py` (to
`tests/golden/mod.rs`) generators, each with a `--check` mode. The generated
header records its provenance and the tolerance consumers should use (equity
VaR: 1e-9 relative plus 1e-12 absolute). The inputs are closed-form sin/cos
series with no RNG, so the cases are fully deterministic. The difference from
the options engines is the oracle: the VaR goldens come from the Python
reference (numpy/scipy, printed with shortest round-trip `repr()`), not from
an mpmath computation, so they prove the ports match the reference, and the
reference carries its own proof through analytic identities and backtests.

</details>

**Q7. Why does each language use a completely different RNG, NumPy's PCG64 in Python, `std::mt19937_64` in C++, and a custom xoshiro256++ in Rust, instead of trying to synchronize them for identical draws?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

No attempt is made at identical cross-language seeding because it isn't
achievable or valuable: PCG64, Mersenne Twister, and xoshiro256++ are
structurally different generators that consume seeds and produce output
streams in incompatible ways, so "same seed, same draws" across three
algorithms is not a coherent goal. Each language instead picks the RNG that
best fits its own standard-library situation,
`cpp/equity-options-engine/include/eqopt/monte_carlo.hpp` uses
`std::mt19937_64` because it's a `<random>` standard fixture;
`rust/equity-options-engine/src/rng.rs` implements xoshiro256++ from scratch
because "Rust's standard library ships no RNG at all." The cross-language
contract is about the *statistical* answer, not the *path*.

</details>

**Q8. Given three different RNGs, what does "cross-language Monte Carlo agreement" actually mean in this portfolio?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

Statistical, not bitwise. The three generators produce different bit streams,
so a C++ and a Rust engine pricing the same option with the same nominal seed
get different individual paths and slightly different point estimates
(`cpp/equity-var-engine/include/eqvar/monte_carlo.hpp`: "RNG streams differ
from NumPy, so agreement with the Python engine is statistical (within Monte
Carlo error), not exact"). What's checked is that each estimate falls within
its own reported confidence interval of the analytic (or reference) value;
`eqopt::MCResult::contains()` and the Rust `McResult::contains()` exist to
test that CI-membership property.

</details>

**Q9. What does "deterministic given a seed" mean WITHIN one engine, and why is that a different, and in some sense stronger, guarantee than cross-language agreement?**

<sub>Tags: `L2`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

Within a single engine on one platform, a fixed seed reproduces the exact same
sequence of draws, hence the exact same output, on every run:
`rust/equity-var-engine/tests/test_monte_carlo.rs` asserts bitwise equality
across runs, and `cpp/equity-var-engine/include/eqvar/monte_carlo.hpp` states
that results "repeat exactly for a given seed on one platform." Across
platforms the last bits can differ, because `std::log`, `std::exp` and
`std::erfc` are not correctly rounded and differ between glibc, MSVC and Apple
libm, and compilers may contract to FMA. So the guarantee is an equality, not
a tolerance band, but only for a fixed compiler and libm. It is a *different*
claim from cross-language agreement, not a bigger version of the same one:
cross-language agreement is about two different generators converging on the
same distribution; within-engine determinism is about one fixed generator
being reproducible.

</details>

**Q10. Why do the C++ and Rust engines write their own inverse-CDF / Box-Muller normal transforms instead of using `std::normal_distribution` or a library normal sampler?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

Because the same-platform determinism guarantee would break otherwise.
`cpp/equity-options-engine/include/eqopt/monte_carlo.hpp` says it directly:
normal deviates come from "an in-house inverse-CDF transform (Acklam initial
guess + one Halley refinement against the erfc-based CDF) rather than
std::normal_distribution, whose algorithm is implementation-defined," so that
the same `(seed, threads)` gives identical results on a given compiler and
libm. With `std::normal_distribution`, two standard libraries on the same
machine could produce different streams from the same seed.
`cpp/equity-var-engine/include/eqvar/monte_carlo.hpp`'s
`RandomStream::gaussian()` does the same via `Phi^{-1}(uniform())`. The Rust
RNGs (`rust/equity-options-engine/src/rng.rs`,
`rust/equity-var-engine/src/rng.rs`) use Box-Muller instead, chosen in the
module doc over inverse-CDF and over Ziggurat ("large tables, harder to
audit"), but the reasoning is the same: own the transform so the stream layout
is fixed rather than delegated to an unspecified standard-library algorithm.

</details>

**Q11. `eqvar::RandomStream` in `cpp/equity-var-engine/include/eqvar/monte_carlo.hpp` exposes four draw methods, `uniform()`, `gaussian()`, `gamma(shape)`, `chi_squared(df)`. Why does a VaR engine's RNG need gamma and chi-squared draws, not just normals?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

The multivariate Student-t factor model needs them: the file's header comment
describes simulating Student-t returns via "the common-mixing-variable
construction `Z / sqrt(W/df)`" where `Z` is multivariate normal and `W` is
chi-squared, fatter tails than a pure Gaussian factor model without changing
the target covariance (the scale matrix is pre-multiplied by `(df-2)/df` so
the *realized* covariance still matches `cov` exactly). `gamma()` is
implemented via Marsaglia-Tsang (with the `U^{1/a}` boost for shape < 1), and
`chi_squared(df)` is literally `2 * gamma(df/2)`, both built on the same
`uniform()`/`gaussian()` primitives so the whole chain stays inside one
deterministic, from-scratch generator.

</details>

**Q12. `eqopt::mc_price` in `cpp/equity-options-engine/include/eqopt/monte_carlo.hpp` supports multiple `threads`. How does multithreading interact with the determinism guarantee?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

It's preserved per thread count, not across thread counts. The header
documents: "Multithreading partitions the paths into per-thread chunks with
independently seeded (splitmix64-derived) RNG streams, so the result is
deterministic for a given thread count," and the parameter doc is explicit
that "different thread counts use different streams (statistically equivalent,
not bit-equal)." So `mc_price(..., seed=42, threads=4)` reproduces exactly on
every run with `threads=4` on the same platform, but will not match
`mc_price(..., seed=42, threads=1)` bit-for-bit, same statistical answer,
different stream partition, same pattern as the cross-language case one level
down.

</details>

**Q13. Why does the Rust `standard_normal()` in `rust/equity-options-engine/src/rng.rs` cache a value between calls, and why does that matter for reproducibility rather than just being a performance optimization?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

Box-Muller naturally produces two independent normal variates per pair of
uniforms (`R cos(theta)`, `R sin(theta)`); `standard_normal()` returns the
cosine term immediately and stashes the sine term in `cached_normal`,
returning it on the *next* call without consuming any new uniforms. This is a
performance win (half the `ln`/`sqrt`/`sin`/`cos` calls), but the doc comment
frames it as a reproducibility property too: "consuming exactly two `u64`
draws per pair makes the stream layout easy to reason about for
bit-reproducibility tests", the number of underlying RNG draws consumed by N
calls to `standard_normal()` is fixed and predictable (`2 * ceil(N/2)`), which
is exactly what a determinism test needs to reason about.

</details>

**Q14. What do the Rust engines' Cargo.toml files actually confirm about the "no third-party dependencies" claim?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

Each engine's `[dependencies]` table has one entry, `qp-core = { path = "../core" }`,
the workspace's own shared crate. `rust/core/Cargo.toml` has an empty
`[dependencies]` table. The comment in `rust/equity-options-engine/Cargo.toml`
says the RNG (SplitMix64 -> xoshiro256++) and the normal CDF (Cody's erfc)
live in the std-only `qp-core` crate and that `proptest` is test-only.
`rust/equity-var-engine/Cargo.toml` says the same; its special functions
now come from `qp_core::stats` in the same std-only crate. So the claim is checkable by reading two manifests:
no crate from outside the repository is compiled into an engine library.
`proptest` appears only under `[dev-dependencies]`.

</details>

**Q15. Why does this portfolio treat "zero external crates" as a Rust convention worth calling out, rather than just using a well-tested `rand` or `statrs` crate?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

`docs/ARCHITECTURE.md`'s design invariants section states the rule: the
compiled engines use only the standard library, and the RNG, normal CDF/PPF
and Cholesky factorisation are written in this repository, in `cpp/core` and
`rust/core`. The reason is that a reviewer should not have to trust an outside
crate's numerics to trust this portfolio's numbers. A dependency on `rand` or
`statrs` would mean the RNG algorithm, seeding procedure, and normal-CDF/PPF
implementation could change silently on a crate version bump, and a reviewer
auditing correctness would have to read someone else's code (or just trust it)
rather than the code actually shipped in `rust/core/src/rng.rs`, `special.rs`
and `linalg.rs`. Owning the RNG, the normal CDF/PPF inverse transform, and the
Cholesky factorization in-repo means every numerically load-bearing line in
the engine is visible in the same codebase the golden-vector tests are
validating.

</details>

**Q16. What does "warnings-as-errors", `-Wall -Wextra -Werror` in C++, `RUSTFLAGS=-D warnings` / `#![deny(warnings)]` in Rust, actually buy beyond "the build succeeds"?**

<sub>Tags: `L2`, `cross-language`</sub>

<details><summary>Answer</summary>

`cpp/equity-options-engine/CMakeLists.txt` attaches `-Wall -Wextra -Wpedantic`
to its targets (`/W4` on MSVC), plus `-Werror` (`/WX`) when `EQOPT_WERROR` is
on; `CONVENTIONS.md` states the Rust equivalent as `#![deny(warnings)]` in the
CI profile, and `docs/DIAGRAMS.md` diagram 12 shows CI actually invoking
`RUSTFLAGS='-D warnings'` before `cargo test --release`. The value isn't
compilation success, a warning-free build with warnings merely printed would
compile identically. It's that the class of bug a compiler warning flags (a
signed/unsigned comparison that silently wraps, an unused-but-assigned
variable that hints at a forgotten branch, an unreachable pattern arm;
shadowing is not in the set, since `-Wshadow` is not part of `-Wall` or
`-Wextra`) never gets the chance to sit in the tree as "just a warning nobody
looked at", it fails CI the same day it's introduced, at the point where the
author still has full context, rather than surfacing later as a numerical bug
that has to be root-caused from scratch.

</details>

**Q17. What numerical problem does the Cholesky-with-jitter-escalation technique solve, and where does it appear across the three languages?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

Plain Cholesky factorization requires the input covariance matrix to be
strictly positive definite; it fails (or the code has to guard against a
`sqrt` of a negative pivot) on a matrix that is only positive *semi*-definite,
exactly singular or numerically indefinite due to floating-point noise.
`cpp/equity-var-engine/include/eqvar/matrix.hpp`'s `cholesky(cov, repair)`
(returning a `CholeskyResult` with `lower`, `jitter_added`, `repaired` and
`warning`), and the Rust `Matrix` types in
`rust/equity-var-engine/src/matrix.rs` and `rust/fx-var-engine/src/matrix.rs`
all implement the same fix: add a small multiple of the identity to the
diagonal and retry.

</details>

**Q18. How does the jitter escalation actually work, mechanically?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

Per `cpp/equity-var-engine/include/eqvar/matrix.hpp`: plain Cholesky first; on
failure `eps * I` is added with `eps = 1e-12 * mean(diag)`, rising x10 per try
and capped at `1e-6 * mean(diag)`. Any jitter used is reported in
`CholeskyResult::jitter_added` and `warning`. Past the cap, with
`Repair::None` (the default) it throws `std::runtime_error` naming the most
negative eigenvalue, because a matrix that needs more than that is materially
indefinite, and factoring it would simulate a *different* covariance from the
one supplied. A caller can instead ask for `Repair::Clip` (eigenvalue
clipping) or `Repair::Higham` (nearest correlation matrix), and the repaired
matrix goes through the same ladder. The Python engines expose the same choice
as `repair=None | "clip" | "higham"`. The Rust equivalents mirror this: both
Rust VaR crates name the cap `MAX_RELATIVE_JITTER`, and
`rust/equity-var-engine/src/matrix.rs` has `cholesky_repair(PsdRepair)`; the
old jitter-start/try-count entry points are deprecated.

</details>

**Q19. Why is a covariance matrix ever legitimately singular in this portfolio, rather than the jitter technique being purely a numerical-robustness hack for buggy input?**

<sub>Tags: `L3`, `cross-language`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Perfectly (or near-perfectly) correlated risk factors are a real market
condition, not just bad data, the canonical example is a pegged FX pair, where
two currencies' returns move in lockstep by policy, giving a correlation of
essentially 1.0 and a covariance matrix with a near-zero eigenvalue by
construction. `rust/fx-var-engine/src/monte_carlo.rs` has a test named exactly
`pegged_currencies_trigger_cholesky_jitter`, with a comment noting "the second
Cholesky pivot is exactly zero and the jitter path is" exercised, i.e. the
jitter escalation is explicitly designed around, and tested against, a
legitimate real-world input, not treated as a defensive catch-all for
malformed data.

</details>

**Q20. Why does the jitter escalation stop and throw instead of continuing to add larger and larger perturbations until *something* factors?**

<sub>Tags: `L3`, `cross-language`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Because an unbounded escalation would eventually "succeed" on a matrix that
isn't merely singular but materially indefinite (a real negative eigenvalue),
e.g. from a correlation-matrix typo (a correlation entered as 1.2, or a
covariance built inconsistently), and factoring it after enough jitter
silently simulates a materially different covariance from the one the caller
supplied, understating or overstating risk with no diagnostic that anything
was wrong. `cpp/equity-var-engine/include/eqvar/matrix.hpp` makes repair an
explicit caller choice (`Repair::Clip` or `Repair::Higham`) rather than
something the Cholesky call does on its own; the cap turns a silent
risk-number distortion into a loud
`std::runtime_error`/`EqVarError::Numerical`.

</details>

**Q21. How tight are the golden-vector tolerances, and where are they enforced?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

For the options engines there is no single tolerance: each output in
`golden_vectors.json` carries its own absolute tolerance, computed by
`tolerance_rules` from the conditioning of that output (for example, an
implied vol is only determined to about `eps * price / vega`, which is large
deep in the wings). `cpp/equity-options-engine/tests/test_golden.cpp` checks
every value against its own record's tolerance and reports the worst deviation
as a fraction of its tolerance (`worst_dev_over_tol`), so drift toward a limit
is visible before it fails. For the VaR/ES engines the generated header states
the consumer tolerance: equity uses 1e-9 relative plus 1e-12 absolute
(libm-vs-numpy ulp noise in sin/cos); FX uses 1e-6 absolute for P&L-scale case
A, 1e-8 relative for case B, and tighter bounds for Kupiec statistics,
p-values and Basel cumulative probabilities.

</details>

**Q22. Why are some VaR/ES tolerances looser than the options tolerances, and why is that a correct choice rather than a way to paper over imprecision?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

`cpp/fx-var-engine/tests/test_golden_python.cpp`'s header explains the
mechanism directly: "the only cross-language differences are libm sin/cos/exp
rounding (~1 ulp per call) and the special-function implementations (scipy
Cephes vs this library)." An option price is one closed-form formula evaluated
once; a VaR/ES pipeline chains a synthetic return series through book
revaluation (itself calling sin/cos), covariance estimation, sorting into
order statistics, quantile interpolation, and a tail integral, each stage's
ULP-level rounding difference between numpy/scipy's implementation and this
codebase's own libm calls compounds through the next stage. The options
goldens are also compared against a 50-digit mpmath oracle, so their tolerance
can be derived from each output's conditioning rather than from a chain of
library differences. A single-formula options price simply has fewer
floating-point operations in series to accumulate noise across, so 1e-9 is
achievable there in a way it structurally isn't for a multi-stage VaR
pipeline; picking 1e-6/1e-8 for P&L-scale VaR/ES is admitting that arithmetic
reality rather than hiding a real discrepancy behind a loose bound.

</details>

**Q23. How do we know the looser VaR/ES tolerance isn't quietly absorbing an actual numerical bug rather than legitimate rounding noise?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

Three facts rule that out. First, the discrepancy is attributed to a *named,
bounded* source, "libm sin/cos/exp rounding (~1 ulp per call)" and "scipy
Cephes vs this library" for special functions, not a vague "close enough";
ULP-level libm divergence is a known, quantifiable phenomenon. Second, the
tolerances are per case and per output: probability-scale figures (Kupiec LR,
p-values, Basel cumulative probabilities) are held to 1e-10 to 1e-12, so a
real bug in the statistics would not hide inside a P&L-scale bound. Third, the
options engines are additionally run through differential tests
(`tests/test_differential.py`): seeded random requests go through JSON-lines
shims for Python, Rust and C++ (`eq_jsonl`, `eqopt_jsonl`) and every answer is
compared with the Python reference to a tolerance derived from the formula's
error propagation, so agreement is checked well beyond the fixed golden cases.

</details>

**Q24. What does a golden-vector test failure actually tell an engineer: and what doesn't it tell them?**

<sub>Tags: `L2`, `cross-language`</sub>

<details><summary>Answer</summary>

It tells them the C++/Rust engine's output for that specific input diverged
from the Python reference's committed value by more than the stated tolerance,
i.e., the port's semantics drifted from the reference it's supposed to mirror.
It does *not* by itself tell them which side is "right": per Diagram 2's own
caveat, cross-language agreement never proved the Python reference was correct
in the first place, so a failure could equally mean the C++/Rust port has a
bug, or (much less likely, since the reference is separately validated) that
the Python reference itself changed. In practice the diagnostic path is: check
whether `golden_vectors.json` was regenerated (a reviewed reference update,
which `--check` would flag if the derived files were not regenerated with it)
versus the engine code changed (the port drifted), the committed-artifact
design from Q4 is what makes that distinction checkable at all.

</details>

**Q25. Summarize the "validated three ways" structure `docs/ARCHITECTURE.md` describes for each of the four compiled engine pairs.**

<sub>Tags: `L3`, `cross-language`</sub>

<details><summary>Answer</summary>

(1) The Python reference is validated against analytic identities and
statistical backtests, put-call parity, Greeks-vs-finite-difference,
tree-to-BS convergence, Kupiec/Basel, none of which involve the C++/Rust
engines at all. (2) The C++ engine mirrors the Python semantics and is checked
via GoogleTest against golden vectors: mpmath-oracle values with per-output
tolerances for the options engines, and Python-reference values with the
tolerances stated in the generated header for VaR/ES. (3) The Rust engine is
checked against the same golden values via `cargo test`, to the same
tolerances. The three legs carry different burdens of proof: leg 1 proves the *reference*
is mathematically sound; legs 2 and 3 each prove their engine matches that
reference. For the options engines the goldens come from an mpmath oracle, so
legs 2 and 3 also check every engine against an independent high-precision
computation, and the differential tests compare the three engines on random
inputs. A market-risk function uses the same pattern when a new pricing
library has to be proven against the incumbent before it is allowed to touch
P&L.

</details>

**Q26. Is the golden-vector methodology the same generation mechanism for every one of the four engine pairs (equity/FX × options/VaR), or does it vary?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

The pipeline is the same for all four: `generate_golden.py` →
`golden_vectors.json` → `gen_golden_header.py`/`gen_golden_rs.py` → committed
`golden_vectors.hpp` and `golden.rs` (options) or `tests/golden/mod.rs` (VaR),
each generator with a `--check` mode. What differs is the oracle. The options
goldens come from mpmath at 50 digits, with a tolerance per output. The VaR/ES
goldens come from the Python reference itself on deterministic sin/cos inputs,
so they prove port agreement, and the Python reference is validated separately
by analytic identities and backtests.

</details>

**Q27. What role does `RandomStream`/`Xoshiro256PlusPlus` play in `mc_bootstrap_se` / the bootstrap standard-error cross-check, and why does its determinism matter there specifically?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

`cpp/equity-var-engine/include/eqvar/monte_carlo.hpp`'s `mc_bootstrap_se`
resamples the P&L sample with replacement `n_boot` times using the file's own
seeded `RandomStream`, recomputes the same order-statistic VaR `L(m)` on each
resample (with `std::nth_element`), and returns the `ddof = 1` standard
deviation. This bootstrap SE is the primary standard error in every engine;
the KDE-based asymptotic SE is reported alongside it only as a diagnostic. Its
determinism matters here for the same reason as everywhere else in the
portfolio: a bootstrap SE that changed between identical runs would be
indistinguishable from an actual change in estimated sampling error.

</details>

**Q28. Tie it together: why is "matches the reference" a different, and in some ways stronger, claim than "two independent teams re-derived the same formula," even though it sounds like a weaker bar?**

<sub>Tags: `L3`, `cross-language`, `code-reading`</sub>

<details><summary>Answer</summary>

Independent re-derivation only proves *convergence*: two engineers ending up
in the same place tells you nothing about whether that place is correct,
because a shared conceptual error survives the comparison undetected. "Matches
the reference to a stated tolerance," by contrast, is a claim with a named,
separately-validated anchor: the C++/Rust engine is being measured against a
specific artifact (`golden_vectors.json` / a documented `repr()` dump) whose
values carry their *own* independent burden of proof: for the options engines
an mpmath computation at 50 digits, for the VaR engines a Python reference
validated by Kupiec backtests and analytic identities, none of which depend on
the C++/Rust ports existing at all. The chain of trust is linear and auditable
(reference validated by math → engines validated against reference) rather
than circular (two implementations validated by mutual agreement), which is
why a market-risk function uses this pattern when proving a new pricing
library against an incumbent before it's allowed to touch P&L.

</details>

**Q29. How does `engine="cpp"` reach the C++ code, and what happens if the extension is not installed?**

<sub>Tags: `L3`, `cross-language`</sub>

<details><summary>Answer</summary>

Each of `eq_options.engine`, `fx_options.engine`, `eq_var.engine` and
`fx_var.engine` calls `qp_core.engines.load_engine`, which imports `qp_cpp` or
`qp_rust`. An unknown name raises `ValueError`. A missing or broken extension
raises `ImportError` whose message is the `pip install ./bindings/...`
command. `"python"` is the default and calls the reference function unchanged,
so nothing changes for existing callers.

</details>

**Q30. The batch `erfc` and `exp` kernels added with the performance kernels are slower than libm. Why ship them?**

<sub>Tags: `L3`, `cross-language`, `numeric`</sub>

<details><summary>Answer</summary>

On the measured target (default x86-64, SSE2 only) `vm::erfc_batch` ran at
0.66x and `vm::exp_batch` at 0.38x the speed of the scalar libm calls, because
the loops do not vectorise without AVX2 and the erfc pays for a 25-step
Clenshaw recurrence and two exps. They are opt-in, have stated error bounds,
and no engine calls them. They may pay off on x86-64-v3 or with `std::simd`,
which was not measured. The honest record is that, on this target, they are a
loss.

Source: `README.md` (the batch kernel table) and `cpp/core/include/qp/vmath.hpp`.

</details>

**Q31. Why did the blocked GEMM and the O(n) quantile change engine code but no goldens?**

<sub>Tags: `L3`, `cross-language`</sub>

<details><summary>Answer</summary>

`syrk_upper_blocked` keeps the k order and row order of the naive loop, so
every sum is formed in the same order and the doubles are identical; the tests
compare bit for bit, including sizes that are not multiples of the 64 tile.
Selection returns the same order statistic as a full sort. A fused
multiply-add (`-march=native` or `-ffp-contract=fast`) would break that
identity; the default target has none.

</details>


## Set 16: Testing, Validation & the NaN Defect Class

**Q1. What is the single IEEE 754 fact that this entire defect class hinges on?**

<sub>Tags: `L2`, `testing`</sub>

<details><summary>Answer</summary>

Under IEEE 754, NaN compares as unordered with everything, including itself:
`NaN < x`, `NaN <= x`, `NaN > x`, `NaN >= x`, and even `NaN == NaN` all
evaluate to `False`, for any `x`. There is no ordering comparison that returns
`True` when one operand is NaN. That single fact is why a guard built purely
out of ordering operators can never catch a NaN, not because NaN is "positive"
or "in range", but because every comparison silently fails to fire.

</details>

**Q2. Concretely, why does `if x <= 0: raise ValueError` fail to catch a NaN input?**

<sub>Tags: `L3`, `testing`</sub>

<details><summary>Answer</summary>

The guard is written to reject non-positive numbers by checking `x <= 0`. When `x` is NaN, `x <= 0` evaluates to `False` (per Q1), so the `raise` branch is never taken, the same as if a valid positive number had been passed. The guard doesn't misclassify NaN as positive; it simply never gets a `True` answer from any comparison involving NaN, so it falls through the way any valid input would.

</details>

**Q3. Where in this portfolio was this defect class actually found, and how widespread was it?**

<sub>Tags: `L3`, `testing`</sub>

<details><summary>Answer</summary>

Per the README's "On input validation" section, a portfolio-wide audit found the *same* defect class, `if x <= 0: raise` silently accepting NaN, in project after project, across all three languages (Python, C++, Rust). It wasn't a one-off bug in one function; it was a systemic pattern repeated everywhere validation had been written with ordering comparisons instead of explicit finiteness checks.

</details>

**Q4. What is the first documented consequence, a NaN spot reaching a pricer?**

<sub>Tags: `L2`, `testing`</sub>

<details><summary>Answer</summary>

A NaN spot price `S` reaching a Black-Scholes-style pricer that didn't reject it produces a NaN "price" as output. This is the mildest of the three consequences: a NaN output is visibly broken. Anyone consuming that number, a downstream aggregator, a P&L report, a human, sees `nan` and knows immediately something is wrong.

</details>

**Q5. What is the second documented consequence, NaN covariance into a VaR aggregator?**

<sub>Tags: `L2`, `testing`, `numeric`</sub>

<details><summary>Answer</summary>

A NaN entry in a factor covariance matrix reaching the VaR aggregation step in
the Rust VaR engine produced a portfolio VaR of *exactly zero*, because the
aggregation step computes `var.max(0.0).sqrt()`, and Rust's `f64::max`, like
C's `fmax`, follows the IEEE 754-2008 `maxNum` rule and returns the non-NaN
operand. The C++ engine's `std::max(var, 0.0)` behaves differently: it is the
comparison `var < 0.0 ? 0.0 : var`, which returns NaN in that argument order
(and 0.0 with the arguments swapped). Both engines now reject a non-finite
variance before this step. The NaN doesn't survive to the output, it gets
silently discarded and replaced by the other operand.

Source: `rust/equity-var-engine/src/parametric.rs` (`var.max(0.0).sqrt()`).

</details>

**Q6. Why is a VaR of exactly zero specifically called "more dangerous than a NaN" in the README?**

<sub>Tags: `L3`, `testing`</sub>

<details><summary>Answer</summary>

A NaN output is self-evidently broken and gets caught by any sanity check or
human glance. A VaR of exactly zero on a hedged book is a *plausible* number,
a well-hedged portfolio really can have very low VaR, so it looks like good
news rather than a red flag. It is far less likely to be double-checked, which
means the corrupted input can silently pass all the way through to a capital
or hedging decision.

</details>

**Q7. What is the third documented consequence, NaN in a P&L feed reaching backtesting?**

<sub>Tags: `L2`, `testing`</sub>

<details><summary>Answer</summary>

A NaN in a historical P&L feed reaching VaR-exception counting caused the
backtest to record zero breaches for that observation, because the breach test
(P&L worse than VaR) is itself an ordering comparison against NaN, which
evaluates `False`. The practical effect: a broken model with bad P&L data
passed its Kupiec proportion-of-failures test and Basel traffic-light backtest
green, because the corrupted days were silently excluded from the exception
count rather than flagged.

</details>

**Q8. What do these three consequences have in common as a mechanism?**

<sub>Tags: `L3`, `testing`</sub>

<details><summary>Answer</summary>

In all three cases, the failure mode is not a crash, it's a *silent
substitution*: a comparison (`<=`, `max`, an exception-count check) that is
supposed to gate or aggregate a value instead quietly drops the NaN and
behaves as if the input were absent or benign. The output in each case is a
number that looks like a legitimate answer, not an error signal, which is
exactly what makes the class dangerous rather than merely annoying.

</details>

**Q9. What is the actual fix pattern applied portfolio-wide?**

<sub>Tags: `L2`, `testing`</sub>

<details><summary>Answer</summary>

Replace ordering-comparison guards with explicit finiteness checks: `math.isfinite()` / `np.isfinite()` in Python, `std::isfinite()` in C++, `.is_finite()` in Rust. A finiteness check directly asks "is this value NaN or ±Infinity?" rather than inferring badness indirectly through a failed ordering comparison, so it catches NaN by construction instead of by accident.

</details>

**Q10. Show a real example of this fix in the equity options pricer.**

<sub>Tags: `L3`, `testing`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`validate_inputs()` in
`python/equity/01-options-pricing/src/eq_options/black_scholes.py` iterates
over `S`, `K`, `T`, `sigma` and, for each, does `if not math.isfinite(value): raise ValueError(...)` *before*
the separate `if value < 0.0: raise ValueError(...)` non-negativity check. The
finiteness check and the sign check are two distinct guards, the finiteness
check exists specifically because the sign check alone cannot catch NaN.

</details>

**Q11. Show a real example of this fix in the VaR engine.**

<sub>Tags: `L3`, `testing`, `code-reading`</sub>

<details><summary>Answer</summary>

`_validate_pnl()` in `python/equity/03-var-es-engine/src/eq_var/historical_var.py` calls `np.all(np.isfinite(arr))` on the P&L array and raises `ValueError("pnl contains NaN or infinite values")` if it fails, before any quantile computation touches the data. `historical_var.py`, `expected_shortfall.py`, and `parametric_var.py` in the same package all carry equivalent `np.isfinite(...)` guards.

</details>

**Q12. Is the fix confined to Python, or does it show up in the compiled engines too?**

<sub>Tags: `L3`, `testing`, `code-reading`</sub>

<details><summary>Answer</summary>

It's cross-language. `cpp/equity-var-engine/src/returns.cpp` guards P&L with
`if (!std::isfinite(v)) throw std::invalid_argument(...)`, and
`parametric.cpp` checks weights, VaR, `z_range`, `skew`, and `excess_kurt` the
same way. `rust/equity-var-engine/src/historical.rs`, `expected_shortfall.rs`,
and `backtest.rs` use `.is_finite()` identically. The three languages
independently reimplement the same finiteness-first validation contract.

</details>

**Q13. Does fixing this defect class change any golden-vector reference values?**

<sub>Tags: `L3`, `testing`</sub>

<details><summary>Answer</summary>

No, this is strictly an input-validation fix, not a change to any pricing or
risk formula. It rejects a class of input (NaN/Inf) that a correct guard
should never have let through in the first place; every previously-valid input
still produces the same output. The same principle applies to any robustness
change: for well-conditioned inputs the pinned golden values stay inside their
tolerances, so a change that moves them is a formula change and gets reviewed
as one.

</details>

**Q14. Why is this specific defect class more dangerous in a quant/risk codebase than in, say, a typical web application?**

<sub>Tags: `L3`, `testing`</sub>

<details><summary>Answer</summary>

In a web app, a validation gap that lets bad data through more often surfaces
as a crash, a 500 error, or a visibly malformed page, something a user or
monitoring system notices quickly. In a risk codebase, the same gap produces a *number*,
a price, a VaR, a breach count, that looks exactly like every other legitimate
number. Nothing crashes. The system keeps running and keeps reporting, and the
wrong number quietly feeds a hedging or capital-allocation decision made by a
human or another system that has no way to tell it apart from a correct one.

</details>

**Q15. Why is `max(NaN, 0.0)` propagating the non-NaN operand not itself a bug in IEEE 754, and why does that matter here?**

<sub>Tags: `L4`, `testing`</sub>

<details><summary>Answer</summary>

It's specified behavior: IEEE 754-2008 `maxNum`, which C's `fmax` and Rust's
`f64::max` implement, returns the non-NaN argument when exactly one operand is
NaN (IEEE 754-2019 added `maximum`, which propagates NaN instead), precisely
so that a single missing/undefined data point doesn't poison an
otherwise-valid aggregate elsewhere in a computation. That's a reasonable
default for many uses of `max`, but it means `max` is not a safe substitute
for an explicit NaN check when the *presence* of the NaN is itself the signal
you need to detect, which is exactly the situation in VaR aggregation.

</details>

**Q16. What does a good test for this defect class actually look like, as opposed to a merely "adequate" one?**

<sub>Tags: `L2`, `testing`, `code-reading`</sub>

<details><summary>Answer</summary>

A test that only exercises "normal" invalid inputs, negative spot, negative
time-to-expiry, an out-of-range alpha, will pass against a guard that has this
bug, because those inputs are exactly what ordering comparisons *do* catch. A
test that actually catches this class explicitly constructs `float("nan")` (or
`np.nan` / `f64::NAN` / `std::numeric_limits<double>::quiet_NaN()`) and passes
it into every public entry point, asserting that it raises rather than
silently succeeding or returning NaN.

</details>

**Q17. Point to a real test in this portfolio that does this.**

<sub>Tags: `L3`, `testing`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`test_negative_or_nan_inputs_raise_everywhere` in
`python/equity/01-options-pricing/tests/test_edge_cases.py` is parametrized
over both ordinary invalid inputs (negative `S`, `K`, `T`, `sigma`) *and*
`(float("nan"), 100, 1, 0.2)`, and asserts `pytest.raises(ValueError)` against
both `bs_price` and `crr_price` for every case in the same table, NaN is
tested as a first-class member of the invalid-input set, not as an
afterthought.

</details>

**Q18. Point to a real test on the VaR side that does this.**

<sub>Tags: `L3`, `testing`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`test_nan_raises` in
`python/equity/03-var-es-engine/tests/test_historical_var.py` builds a
100-element zero P&L array, sets `pnl[3] = np.nan`, and asserts
`pytest.raises(ValueError, match="NaN")` from `historical_var(pnl, 0.01)`. It
doesn't just check that *some* invalid array is rejected, it checks that a
single NaN buried inside an otherwise-valid array specifically is caught,
which is the realistic failure mode (one bad tick in a feed, not a wholly
corrupted series).

</details>

**Q19. Is there a complementary style of test, checking that NaN never *appears* in valid output, not just that it's rejected on input?**

<sub>Tags: `L3`, `testing`, `code-reading`</sub>

<details><summary>Answer</summary>

Yes. `test_no_nans_price_sweep` and `test_no_nans_greeks_sweep` in the same
`test_edge_cases.py` sweep a grid of legitimate `S`, `K`, `T`, `r`, `sigma`
combinations (including extreme-but-valid values like `1e-3`/`1e5`) and assert
every output is finite. This is the mirror image of Q17–Q18: rather than
asserting bad input is rejected, it asserts that no combination of *good*
input can accidentally manufacture a NaN downstream (e.g. via a `0/0` or
`log(0)` inside the formula), the other place NaN contamination can originate
besides a bad upstream feed.

</details>

**Q20. What is this portfolio's broader testing philosophy, per `CONVENTIONS.md`'s "Testing contract"?**

<sub>Tags: `L2`, `testing`</sub>

<details><summary>Answer</summary>

Beyond happy paths and edge cases, the contract requires "analytic identities
(put-call parity, Greeks vs finite differences, martingale checks),
convergence (tree → BS, MC → BS), and failure behaviour (invalid inputs raise
`ValueError` with informative messages)", plus "property-based checks where
natural (monotonicity in vol, convexity in strike)". The emphasis throughout
is on properties the model is *mathematically required* to have, not just a
table of input/output pairs.

</details>

**Q21. What does it mean to test a PROPERTY rather than a fixed expected output, concretely?**

<sub>Tags: `L2`, `testing`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

A fixed-output test asserts `bs_price(100, 100, 1, 0.05, 0.2, 0.0, "call") == 10.4506`
(or close to it), it only tells you that one specific input still produces one
specific output. A property test asserts something that must hold for a whole *family*
of inputs regardless of the specific numbers, e.g.
`test_put_monotone_increasing_in_vol` and `test_call_delta_monotone_in_spot`
in `python/equity/01-options-pricing/tests/test_properties.py` check that
price/delta move in the required direction across a swept range of vol or spot
values, and `test_parity_and_forward_consistency_stressed` checks that call
price minus put price equals the forward-discounted difference across stressed
parameter combinations.

</details>

**Q22. Why does property-based testing catch defect classes rather than just individual bugs?**

<sub>Tags: `L3`, `testing`</sub>

<details><summary>Answer</summary>

A fixed expected-output test only fails if the specific case it happened to
encode is broken. A property test, "for every input in this swept range, X
must hold", fails on *any* input in the range that violates the property,
including ones nobody specifically thought to hand-pick. Put-call parity,
monotonicity in vol, and tree→BS convergence are structural truths of the
model; if a NaN-guard bug, a solver-plateau bug, or a grid-resolution bug
produces even one input where the property breaks, a property test sweeping
that region will catch it, whereas a hand-picked expected-output table might
simply never include that input.

</details>

**Q23. Is `test_no_nans_price_sweep` (Q19) itself a property test, and if so, what property does it encode?**

<sub>Tags: `L3`, `testing`</sub>

<details><summary>Answer</summary>

Yes, its property is "for every combination of valid inputs, the pricer's
output is finite." That's exactly the kind of blanket, model-wide invariant
that a single `if x <= 0: raise` guard bug would violate somewhere in the
swept grid even if it happened to pass every individually-chosen edge case
test. It's the direct test-side counterpart of the fix in Q9–Q13: the fix
makes the guard reject non-finite input, and this test makes sure valid input
can never accidentally *produce* non-finite output.

</details>

**Q24. How does `test_infinite_or_nan_price_rejected_by_implied_vol` extend this beyond simple parameter validation?**

<sub>Tags: `L3`, `testing`, `code-reading`</sub>

<details><summary>Answer</summary>

That test (in `python/equity/01-options-pricing/tests/test_properties.py`)
checks that the implied-vol solver itself rejects a NaN or Infinite *target
price*, not a spot, strike, or vol, but the market-observed price the solver
is trying to invert. It's a reminder that "every public entry point" in the
README's fix description really does mean every one: an inversion routine
consuming a market quote is just as much an attack surface for a
silently-accepted NaN as a forward pricing call is.

</details>

**Q25. Does the finiteness-check fix generalize past NaN, or is it NaN-specific?**

<sub>Tags: `L3`, `testing`</sub>

<details><summary>Answer</summary>

It generalizes: `math.isfinite`, `np.isfinite`, `std::isfinite`, and
`.is_finite()` all reject both NaN *and* ±Infinity in one check. An `if x <= 0: raise`
guard has the same blind spot for `+Infinity` reaching code that assumes
bounded input (it *would* be caught by a `<= 0` check only if it were
non-positive infinity, positive infinity sails through a positivity guard
exactly like NaN does, just for a different reason: it is `> 0`). The
finiteness check closes both holes with one guard instead of needing separate
NaN and Inf logic.

</details>

**Q26. How was this defect class found, and why does the README state the covered scope explicitly?**

<sub>Tags: `L3`, `testing`, `code-reading`</sub>

<details><summary>Answer</summary>

By a dedicated audit looking specifically for input-validation gaps, not by
the original per-project test suites. The first sweep did not reach every
module: a later review found that the equity Python VaR backtester
(`eq_var.backtesting.exceptions_from_pnl`) still counted a NaN P&L day as "no
exception" while its C++/Rust ports refused it, and the fixed-income YTM stack
and the foundations Black-Scholes `_d1_d2` still had the `if x <= 0` hole.
Those are now closed and pinned in per-project `test_input_hardening.py`
suites. The lesson is about scope: a defect class is only closed for the entry
points a test actually covers, so the README states the covered scope
explicitly rather than claiming "every public entry point".

</details>

**Q27. Why didn't the original test suites: which did test invalid inputs like negative numbers, catch this?**

<sub>Tags: `L3`, `testing`</sub>

<details><summary>Answer</summary>

Because a test suite that asserts negative and out-of-range inputs raise
`ValueError` will pass against a buggy `if x <= 0: raise` guard just as
readily as against a correct one, negative numbers *are* correctly caught by
that comparison. The gap only shows up if a test specifically constructs NaN
and hands it to the same code path, which is a qualitatively different test
input, not a more extreme version of "a bad number." This is precisely why
Q16's distinction, NaN as a first-class test case, not an extrapolation of
"very negative", matters: no amount of testing more negative numbers would
ever have found this bug.

</details>

**Q28. What is the general lesson this round is meant to teach about writing input guards?**

<sub>Tags: `L2`, `testing`</sub>

<details><summary>Answer</summary>

Never infer "this value is invalid" from a guard reaching `False`, a `False`
comparison result can mean "the value is valid" or "the value is NaN, so no
comparison could evaluate true," and ordering operators cannot distinguish the
two. Any function whose validity range is expressed with `<`, `<=`, `>`, `>=`,
or `==` needs an explicit, independent finiteness check ahead of those
comparisons whenever the input could plausibly originate from a live feed, a
prior calculation, or any other source that can produce NaN, which, in a
quant/risk pipeline, is effectively everywhere.

</details>

**Q29. What does `tests/test_engine_differential.py` check that the binding tests did not?**

<sub>Tags: `L2`, `testing`</sub>

<details><summary>Answer</summary>

Scale. Each compiled engine is compared with the Python reference on 10,000
random inputs per group: options prices, Greeks, implied vol and
batch-versus-scalar bit equality; empirical VaR and ES; parametric VaR; FX
closed forms. The first wide run found three tolerance issues, not engine
bugs, among them a parametric VaR cancellation that needed an absolute bound
scaled to the terms. In CI the job fails if any compiled case is skipped, so a
missing build cannot pass quietly.

</details>


## Set 17: Business, Desk Usage & Model Risk Governance

**Q1. What is "model risk" as a formal risk category, and how is it different from market, credit, or operational risk?**

<sub>Tags: `L3`, `desk-governance`</sub>

<details><summary>Answer</summary>

Model risk is the risk that a model itself is wrong, poorly specified, or
misused, even when it is implemented flawlessly against its own spec. Market
risk is the risk that prices move against you; credit risk is the risk a
counterparty doesn't pay; operational risk is the risk a process or system
fails. Model risk sits one level up: it's the risk that the *tool you use to
measure* those other risks is itself miscalibrated, a VaR engine that assumes
normal returns (`python/equity/03-var-es-engine`), a scorecard whose
missing-data mechanism silently changed (`python/equity/06-credit-risk`), an
options pricer whose no-jump assumption gets exploited by an earnings gap
(`python/equity/01-options-pricing`). The code can pass every unit test and
still be the wrong model for the market regime it's now facing.

</details>

**Q2. What is the standard model-governance lifecycle a real bank applies to a model like these, end to end?**

<sub>Tags: `L3`, `desk-governance`</sub>

<details><summary>Answer</summary>

Development (build it, document assumptions), independent validation (a
separate team stress-tests it against its own spec and against challengers),
approval/sign-off (a model risk committee formally accepts it for a defined
use), ongoing monitoring (automated triggers watch live performance), periodic
re-validation (a scheduled full re-review, not just monitoring), and
retirement (the model is formally decommissioned when it fails, or is
superseded). The credit-risk `DESK_GUIDE.md` §6 spells this out almost
verbatim under "Model governance, annual validation cycle," citing SR 11-7 /
EBA expectations explicitly, with independent validation, hard monitoring
triggers, change control, and a use test as the four numbered steps.

</details>

**Q3. Why does "independent" validation specifically matter: why can't the model's own developer validate it?**

<sub>Tags: `L4`, `desk-governance`</sub>

<details><summary>Answer</summary>

A developer who built a model already believes its assumptions are reasonable;
that belief is precisely what makes them the worst-positioned person to find
where those assumptions break. They will test the paths they thought of, not
the ones they didn't, someone who never held the belief has to actually
interrogate it. The credit-risk project's governance section makes this
concrete: independent validation includes "a challenger model (GBM benchmark)
and the sklearn cross-check", a second, differently-built model whose
disagreement with the first is itself the validation signal, something the
original developer has no incentive to go looking for.

</details>

**Q4. This portfolio's own documentation contract requires six things per project. How does that map onto a real model-validation submission package?**

<sub>Tags: `L4`, `desk-governance`</sub>

<details><summary>Answer</summary>

CONVENTIONS.md's six items, why you chose the model, the assumptions register,
how you validated it, where it fails, how a desk would use it, and edge cases,
are almost exactly the sections a bank's model risk management team expects in
a validation submission: model selection rationale and alternatives
considered, a documented assumptions log with "what breaks if violated,"
independent validation evidence, a known-limitations register, a
use-and-controls section, and a documented edge-case/boundary-behavior
appendix. The equity options `DESK_GUIDE.md` §2 literally uses the phrase
"known-limitations register" pointing back at the assumptions (A1–A8) and
validation sections, the same artifact a real submission package would attach.

</details>

**Q5. Why is "how would a desk use this?" one of the six mandatory questions, rather than an afterthought bolted on at the end?**

<sub>Tags: `L4`, `desk-governance`</sub>

<details><summary>Answer</summary>

Because a model whose real-world usage was never thought through tends to have
a specification gap that only shows up in production, never in a clean unit
test suite, which only exercises the inputs someone already anticipated. The
FX VaR engine's `DESK_GUIDE.md` makes this explicit as a design principle: the
engine "refuses NaN" at every boundary, market snapshots, factor histories,
scenario shocks, specifically because "every risk number this desk publishes
is consumed by a limit check, a traffic light or a capital multiplier, and
none of those fire on NaN." That behavior isn't something a math-correctness
test would ever catch; it only surfaces once you ask who consumes the number
and what happens when the pipeline hits a stale fixing at 06:00.

</details>

**Q6. What is P&L attribution, and why is it described as a live check on the risk model rather than just an accounting exercise?**

<sub>Tags: `L3`, `desk-governance`</sub>

<details><summary>Answer</summary>

P&L attribution decomposes today's actual P&L move into the pieces the risk
model says should explain it, for options, `dV ≈ delta·dS + ½gamma·dS² + vega·dsigma + theta·dt + rho·dr + residual`.
If the attribution reconciles, the model's sensitivities are being borne out
in reality; if there's a persistent unexplained residual, that's evidence the
model or its calibration is wrong, not just noise. The options `DESK_GUIDE.md`
§1.3 states this directly: "A persistent residual is the model-risk alarm
(wrong q around ex-dates, smile move not captured by parallel vega...)." The
algo-execution project extends the same idea to execution: the PM's daily
sheet splits "signal P&L (gross), cost drag (ledger), execution quality (IS vs
VWAP), each with its own owner", attribution as an organizational control, not
just a number.

</details>

**Q7. Give a concrete example of a persistent P&L residual acting as a model-risk red flag.**

<sub>Tags: `L4`, `desk-governance`, `numeric`</sub>

<details><summary>Answer</summary>

The options desk guide ties the residual explicitly to specific
mis-specifications: a wrong dividend yield `q` around ex-dividend dates, or a
smile move that parallel vega (a single number) can't capture and needs
splitting into vanna/volga buckets instead. Section 3.2's
dividend-announcement scenario quantifies it, up to +1.15 mispricing on a
2.75-value call straddling the ex-date when the continuous-q assumption gets
the direction right but the timing wrong. That's not a bug in the pricer; it's
the assumption register (A-series in METHODOLOGY.md) manifesting as an actual,
measurable residual on the P&L line, which is exactly why it's monitored
rather than assumed away.

</details>

**Q8. What's an analogous "unexplained residual" check in the VaR/backtest world, distinct from options P&L attribution?**

<sub>Tags: `L3`, `desk-governance`, `numeric`</sub>

<details><summary>Answer</summary>

Backtest exceptions and their clustering. The equity VaR desk guide §3
describes checking, on any exception (realized loss worse than VaR), whether
losses cluster within a short window: `exception_cluster_table` flags gaps of
5 days or fewer (a simple heuristic), and the Christoffersen independence
test, a likelihood-ratio test on the day-to-day Markov transition counts,
formalises the clustering check, because a cluster with an acceptable count
can still fail conditional coverage. The FX desk guide states the lesson more
sharply: clustered exceptions (independence p < 5 %) trigger a model review
even in the green zone, because clustering means the model is slow, not
unlucky. An occasional, isolated exception is expected statistical noise at a
99% confidence level; a cluster is the residual-attribution equivalent of "the
model stopped explaining reality."

Source: `python/equity/03-var-es-engine/docs/DESK_GUIDE.md` §3 and `python/fx/03-var-es-engine/docs/DESK_GUIDE.md` §4.

</details>

**Q9. What are position limits and risk limits, and why do they function as controls that don't require anyone to fully trust the model's exact number?**

<sub>Tags: `L3`, `desk-governance`, `numeric`</sub>

<details><summary>Answer</summary>

A limit is a hard cap on exposure (delta, VaR, ES, gross/net notional,
participation rate) enforced regardless of how confident anyone is in the
underlying model's precision, a limit only needs the number to be
directionally useful, not exactly right. The FX VaR desk guide §3 lays out a
full limit sheet (firm 99%/1d ES at $2.5m, per-pair delta caps, vega-per-pair,
a separate peg-exposure notional limit) and notes that peg exposure gets its
own notional limit because it does not consume VaR limit: the desk knows VaR
is *wrong* for peg risk and layers an independent control on top rather than
trying to fix the VaR number. That is model risk management in action: don't
wait for the model to be perfect, bound the damage it can do if it's wrong.

Source: `python/fx/03-var-es-engine/docs/DESK_GUIDE.md` §3 (the limit sheet).

</details>

**Q10. Give a second, different example of a limit compensating for a known model weakness rather than assuming the model is complete.**

<sub>Tags: `L4`, `desk-governance`, `numeric`</sub>

<details><summary>Answer</summary>

The equity portfolio-optimization desk guide lists a concentration floor
(`effective_n` ≥ 4) and a turnover budget (≤2x/yr) as controls that exist
independent of the optimizer's own math, in the synthetic race raw-mean
tangency averaged 6.8x a year turnover (James-Stein tangency 2.7x), so both
would breach a 2x budget, and raw tangency averaged an effective N of 1.9 on
the seed-1 panel, about two names. The limit doesn't ask whether the
optimizer's covariance estimate is exactly right, it just refuses to let a bad
estimate turn into a two-name book, full stop.

Source: `python/equity/07-portfolio-optimization/docs/DESK_GUIDE.md` §3, §4 and its controls table.

</details>

**Q11. Walk through the equity options desk's actual daily workflow, who uses the pricer, in what order, and for what?**

<sub>Tags: `L4`, `desk-governance`, `code-reading`</sub>

<details><summary>Answer</summary>

Per `python/equity/01-options-pricing/docs/DESK_GUIDE.md` §1: market-makers
use the kernel intraday to generate theo prices and bid/ask quotes off the vol
surface (one marked vol per strike/expiry, never a flat vol); the same kernel
feeds an intraday risk pipeline that re-prices the whole book off current
marks and aggregates to book-level delta/gamma/vega/theta/rho, with delta
auto-hedged with futures inside a band; and overnight, an EOD batch re-prices
off closing marks, produces the Greeks report, runs P&L attribution, and runs
a nightly `comparison.py` harness checking BS vs tree vs MC agreement as an
automated regression gate. Three different consumers (market-making, intraday
risk, EOD governance) touch the same pure pricing functions for three
different purposes.

</details>

**Q12. What triggers an escalation on the equity VaR desk, and who does it go to?**

<sub>Tags: `L4`, `desk-governance`, `numeric`</sub>

<details><summary>Answer</summary>

Per `python/equity/03-var-es-engine/docs/DESK_GUIDE.md` §2, a limit breach on
99% 1d VaR or 97.5% ES triggers same-day escalation to the desk head and the
market risk manager, and the desk must either cut risk or obtain a documented
temporary limit increase, no quiet override. A VaR SE-adjusted breach (inside
one standard error of the limit) is still treated as a breach out of
conservatism, just flagged as statistically marginal rather than definitive.
Separately, §3 describes a quarterly model review where a backtest method
failing Kupiec/CC tests for two consecutive quarters is formally remediated or
replaced, with the backtest evidence attached going through model-risk
governance, an escalation path distinct from the daily breach path.

</details>

**Q13. What triggers an escalation on the credit-scoring pipeline, and how is it different in character from the VaR desk's triggers?**

<sub>Tags: `L4`, `desk-governance`, `numeric`</sub>

<details><summary>Answer</summary>

Per `python/equity/06-credit-risk/docs/DESK_GUIDE.md` §6, three hard triggers
govern ongoing monitoring: PSI > 0.25 on the score or any input forces
recalibration/rebinning (0.10–0.25 is an investigate-band);
realized-vs-predicted default rate falling outside the 95% binomial band, or a
Hosmer-Lemeshow p-value < 0.01 on the latest vintage, forces at minimum an
intercept recalibration; and an AUC drop of more than 5 points from
development triggers a full redevelopment. Unlike the VaR desk's same-day
breach escalation (a market event demands an immediate response), these are
statistical monitoring triggers evaluated on a slower monthly/quarterly
cadence against a stable population, the credit desk guide's §7 scenario on
2020 payment holidays even shows PSI staying "stable" (0.074) while
calibration silently failed, underscoring that the two monitors catch
different failure modes and neither substitutes for the other.

</details>

**Q14. What triggers an escalation on the algo-execution desk, and how many distinct kill-switches does it run?**

<sub>Tags: `L4`, `desk-governance`</sub>

<details><summary>Answer</summary>

Per `python/equity/08-algo-execution/docs/DESK_GUIDE.md` §4, there are five
automated, pre-trade-and-intraday kill-switches: a participation kill (a child
order exceeding its cap on bucket volume is rejected outright), book limits
(gross/net/per-name caps block new opening trades), a cost kill (realized
slippage exceeding roughly 3x the modeled cost halts the algo and routes
residuals manually), a data kill (stale prices or feature NaN spikes freeze
the signal, with a stale-signal counter escalating after two days), and a
drawdown kill (strategy MDD beyond backtest MDD plus a buffer forces a
de-gross to half and a PM review). This is a materially higher density of
automated, real-time triggers than either the VaR desk (same-day human
escalation) or the credit desk (monthly/quarterly statistical triggers),
appropriate given execution risk compounds in minutes, not days.

</details>

**Q15. Who actually consumes a VaR number, and does everyone see the same thing?**

<sub>Tags: `L3`, `desk-governance`, `numeric`</sub>

<details><summary>Answer</summary>

No, the equity VaR desk guide §7 lays out a consumer table with different
numbers at different cadences: the desk head sees VaR and ES against limits,
ladders and exceptions daily; the CRO and risk committee see firm VaR and ES,
the traffic light and the stress table daily or weekly; the regulator sees 99%
VaR, the 250-day exception count and the multiplier quarterly; model
validation sees backtest p-values, method disagreement and Cornish-Fisher
validity flags quarterly; and finance sees the capital charge monthly. The
same underlying risk engine serves five audiences with five different slices
and five different refresh rates, a single "VaR number" is really a family of
derived views.

Source: `python/equity/03-var-es-engine/docs/DESK_GUIDE.md` §7.

</details>

**Q16. Who consumes the credit scorecard's PD, and what does the "use test" require of them?**

<sub>Tags: `L3`, `desk-governance`</sub>

<details><summary>Answer</summary>

Per `python/equity/06-credit-risk/docs/DESK_GUIDE.md` §1 and §6, origination
uses the score plus cutoff and reason codes per application; pricing turns the
PD into a risk-based spread per application; finance turns point-in-time PDs
into IFRS 9/CECL expected credit losses monthly or quarterly; treasury
computes Basel capital (K, RWA) monthly; risk appetite runs Vasicek economic
capital and stress quarterly/annually; and model risk management owns the
annual validation pack. The "use test" (§6.4) is the governance rule tying
them together: "pricing, origination and provisioning must consume the same PD
(one model, many uses) or divergences must be documented", i.e., a bank isn't
allowed to quietly run a friendlier PD for pricing than the one it reports for
provisioning without that gap being an explicit, documented decision.

</details>

**Q17. How does model governance differ between a "champion/challenger" setup and a single blessed model, and where does this portfolio show a champion/challenger structure explicitly?**

<sub>Tags: `L4`, `desk-governance`, `numeric`</sub>

<details><summary>Answer</summary>

A champion/challenger setup runs one model as the production number while one
or more alternative models run in parallel purely to sanity-check it,
disagreement between them is itself a monitored signal, not noise to be
ignored. The FX VaR desk guide §8 states this as policy: FHS is the champion
for headline VaR and ES; parametric normal and t (speed, decomposition) and MC
t and jump (EM and peg overlays) are challengers; the daily report prints all
of them, and a champion-challenger gap above 20 % is a standing agenda item.
The equity VaR desk guide's morning pack (§1) does the same thing under a
different name, the method disagreement between FHS and parametric VaR, framed
as information, not a bug.

Source: `python/fx/03-var-es-engine/docs/DESK_GUIDE.md` §8 and `python/equity/03-var-es-engine/docs/DESK_GUIDE.md` §1.

</details>

**Q18. Why does the portfolio-optimization project treat "beat equal weight net of costs" as a formal retirement test rather than just informal commentary?**

<sub>Tags: `L4`, `desk-governance`</sub>

<details><summary>Answer</summary>

Because equal weight is cheap, has no estimation error, and is hard to beat
after realistic costs, so a strategy that can't clear that bar isn't adding
value, however sophisticated its math.
`python/equity/07-portfolio-optimization/docs/DESK_GUIDE.md` §2 states it as
governance: "the walk-forward race (EW benchmark included) is rerun quarterly;
a strategy that cannot beat equal weight net of costs over rolling windows is
a candidate for retirement. Equal weight is the null hypothesis of this
business." That's the retirement stage of the model lifecycle (Q2) made
concrete and mechanical, a scheduled, quantitative test rather than a
subjective judgment call, run on the same cadence (quarterly) as the VaR
desk's backtest method review.

</details>

**Q19. What does "change control" mean in a model-governance context, and how do at least two of these projects implement it in code rather than in a policy document?**

<sub>Tags: `L3`, `desk-governance`</sub>

<details><summary>Answer</summary>

Change control means any modification to a model's calibrated parameters,
binning tables, or constants is versioned, tracked, and re-approved rather
than silently edited, the model in production today should be traceable to a
specific signed-off version. The FX VaR desk guide §8 makes this literal:
cross-language golden vectors pin specific numeric outputs so "a refactor here
fails here rather than surfacing as a mystery failure in another language's
CI... regenerating those constants is a deliberate, signed-off act, not a side
effect." The credit-risk desk guide §6.3 states the same principle for its
domain: "binning tables and coefficients are the model; any change is
versioned and re-approved. The leakage deny-list
(`FORBIDDEN_POST_OUTCOME_FIELDS`) is part of the model definition", i.e., even
the list of fields the model is *forbidden* to use is itself a governed
artifact.

</details>

**Q20. The algo-execution desk guide describes a four-stage pipeline before a signal reaches real money: what are the stages, and what stops a signal from skipping ahead?**

<sub>Tags: `L4`, `desk-governance`, `numeric`</sub>

<details><summary>Answer</summary>

Per `python/equity/08-algo-execution/docs/DESK_GUIDE.md` §1: research (a new
feature needs a point-in-time mutation test and a hand-computed unit test
before anyone even looks at its information coefficient), simulation (the
candidate must clear an IC t-stat > 3 *and* a deflated Sharpe ratio > 0.9 at
the full trial count logged in the research diary, punishing exactly the
p-hacking that comes from trying many variants), paper trading (4-8 weeks in
shadow, with live IC compared against the backtest's confidence band, a
divergence beyond 2 standard errors for 20 days "triggers a research review,
not a quiet re-fit"), and production (where the signal's code hash, parameter
set, trial count, and approval date are all recorded, and any parameter change
restarts paper trading from scratch). Nothing skips a stage, the paper-trading
rule specifically closes off the shortcut of a developer quietly re-tuning a
live signal without new independent scrutiny.

</details>

**Q21. What is the difference between a model's "development team" believing an assumption and a model's actual assumptions register, and why does this portfolio insist on the latter being explicit and numbered?**

<sub>Tags: `L3`, `desk-governance`</sub>

<details><summary>Answer</summary>

A developer's belief lives in their head and changes silently as they iterate;
an assumptions register is a written, numbered, externally auditable list,
each entry paired with what breaks if it's violated, which lets someone who
never built the model check it against reality without re-deriving the model's
logic. The credit-risk desk guide's scenarios section directly cites specific
numbered assumptions failing in named historical episodes: "assumption 6" (the
missingness mechanism) breaking during 2020 payment holidays, and single-name
factor blindness described in the FX desk guide as "assumption A1's failure
mode." Numbering assumptions turns "the model has limits" from a vague
disclaimer into a checklist that ongoing monitoring and validation can be run
against item by item.

</details>

**Q22. How does the FX VaR desk's three-hub, 24-hour operating cycle illustrate a governance problem that a single-timezone desk wouldn't face?**

<sub>Tags: `L4`, `desk-governance`</sub>

<details><summary>Answer</summary>

Per `python/fx/03-var-es-engine/docs/DESK_GUIDE.md` §1, FX never closes, so
"the official close is a convention, not a fact of nature", the desk snaps
positions at 17:00 NY by policy, and the EOD batch's NaN policy becomes a live
governance decision every single night: "Tokyo and São Paulo holidays do not
line up; the desk decides fill-vs-drop explicitly, the engine never imputes."
A single-timezone equity desk has one unambiguous close and doesn't face this;
the FX desk's governance has to specify, as policy, what "today's data" even
means before any model can run on it, a data-completeness decision that in
most desks is invisible because the market conveniently closes once.

</details>

**Q23. What does "netting across desks" reveal about model risk that a single desk's risk report can hide?**

<sub>Tags: `L3`, `desk-governance`, `numeric`</sub>

<details><summary>Answer</summary>

The FX VaR desk guide §2 shows that because every position, spot, forward leg,
option delta, maps to the same USD-pivot factor convention, firm-level risk is
literally the sum of desk-level books, and diversification is reported as
ΣVaR_desk − VaR_firm. But it flags the model risk this creates directly: the
engine's peg counterexample is the one-slide explanation of why the desk-level
99% VaRs of two peg books can both be zero while the firm carries a full
devaluation risk. A model that looks fine at the desk level can be
systematically blind at the firm level if the risk factor decomposition itself
has a gap, which is why peg exposure gets tracked as a separate notional limit
(Q9) rather than being trusted to show up in VaR at any level of aggregation.

Source: `python/fx/03-var-es-engine/docs/DESK_GUIDE.md` §2.

</details>

**Q24. What role does a stress-testing committee play that a daily VaR report and backtest cannot substitute for?**

<sub>Tags: `L4`, `desk-governance`, `numeric`</sub>

<details><summary>Answer</summary>

VaR and its backtest tell you whether the model's *routine* quantile forecast
has been reliable historically; a stress committee asks what happens in
scenarios the historical window may never have contained. The equity VaR desk
guide §5 makes the point explicit with numbers: on the synthetic demo book the
historical replays (1987, 2008 Lehman, 2020 COVID) produce full-revaluation
losses of $134k, $198k and $254k, 2.5 to 4.6 times the 99% VaR: VaR is not a
worst case. The same section's reverse-stress exercise ("what joint 3-sigma
move hurts most?") goes further than either VaR or forward stress by naming
the book's actual concentration (AAPL −4.7%, JPM −3.6%, SPX −1.9%) rather than
assuming a scenario and checking the loss, it's a control built specifically
to compensate for the fact that no stress scenario library can be assumed
complete.

Source: `python/equity/03-var-es-engine/docs/DESK_GUIDE.md` §5.

</details>

**Q25. Give an example where a model's own governance section admits a scenario where its headline risk measure becomes actively useless, and what compensates for it.**

<sub>Tags: `L4`, `desk-governance`, `numeric`</sub>

<details><summary>Answer</summary>

The equity VaR desk guide §6 describes a single-name squeeze, run with
`shocks_by_name` as a two-sided shock on one name, and states plainly that VaR
calibrated on calm vol does not see it and that the ladder is the control.
This is a rare and useful kind of candour in a desk guide, rather than
claiming the risk model covers this case, it identifies the case where the
primary risk number breaks down and names the specific alternative control
(the sensitivity ladder, which shows raw exposure rather than a probabilistic
quantile) that catches what VaR structurally cannot.

Source: `python/equity/03-var-es-engine/docs/DESK_GUIDE.md` §6.

</details>

**Q26. Tie it together: why is "model risk" ultimately a governance discipline about people and process, not just a mathematical property of the model?**

<sub>Tags: `L4`, `desk-governance`</sub>

<details><summary>Answer</summary>

Because every mechanism surveyed above, independent validation,
champion/challenger disagreement flags, escalation triggers, limits that don't
require trusting the model exactly, retirement tests, versioned change
control, exists to catch the case where the model is wrong in a way its own
math cannot self-diagnose. A model cannot notice its own blind spot; only an
external process (a different team, a different model, a limit, a scheduled
re-test, a human escalation) can. That is why CONVENTIONS.md makes "how would
it be used on a desk" one of six mandatory documentation questions alongside
the math: a model whose usage, consumers, and controls were never worked out
on paper is a model whose governance gap will be discovered by a regulator or
a loss, not by a code review.

</details>


## Set 18: Real-World Postmortems & Judgment

**Q1. In plain language, what was the NaN-guard bug, and where in the code does it live?**

<sub>Tags: `L3`, `postmortems`</sub>

<details><summary>Answer</summary>

A guard written as `if x <= 0: raise ValueError(...)` looks like it rejects
bad input, but under IEEE 754 every comparison against NaN, including `NaN <= 0`,
evaluates to `False`. So instead of raising, the guard silently falls through
and lets a NaN spot, rate, vol, or covariance entry continue into the pricer
or risk calculation. This was found across nearly every project in all three
languages (Python, C++, Rust) during a portfolio-wide audit.

</details>

**Q2. Why is this specific failure mode more dangerous than the NaN itself?**

<sub>Tags: `L4`, `postmortems`, `numeric`</sub>

<details><summary>Answer</summary>

Because of a second IEEE 754 quirk one step downstream: `NaN.max(0.0)` (Rust's
`f64::max`) returns the *non-NaN* operand, so a NaN covariance that reaches
the portfolio VaR aggregator does not surface as a NaN, it comes out as a
portfolio VaR of exactly zero. A NaN price is at least visibly broken; a VaR
of exactly 0.0 is a plausible number for a well-hedged book, so nothing
downstream (a risk report, a limit check) has any reason to question it.
README.md's "On input validation" section calls this out explicitly as the
most dangerous manifestation.

</details>

**Q3. The README also mentions a third symptom involving backtesting. What was it?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

A NaN in a P&L feed reaching the exception-counting logic caused VaR-exception
counters to record zero breaches for that day, because the NaN comparison
against the VaR threshold (another `<=`/`>` style check) silently failed to
count as a breach. The practical effect: a broken model, one that was actually
producing garbage risk numbers, passed its Kupiec proportion-of-failures and
Basel traffic-light backtests green, the exact opposite of what those tests
exist to catch.

</details>

**Q4. Why did the pre-fix test suites not catch this, given the CONVENTIONS.md testing contract calls for "failure behaviour... raise `ValueError`" tests?**

<sub>Tags: `L4`, `postmortems`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

Those tests existed and passed, they fed in `-1.0`, `0.0`, or other ordinary
invalid values and confirmed the raise fired, which it did. The blind spot was
that nobody had a test case that specifically fed `float('nan')` through the
same entry points and asserted that it *also* raises. A comparison-based guard
and a finiteness-based guard produce identical behavior on every ordinary
invalid input; they only diverge on NaN, which no example-based "does invalid
input raise" test happened to include.

</details>

**Q5. What kind of test catches the NaN-guard class, and does the finiteness check change any computed number?**

<sub>Tags: `L4`, `postmortems`, `code-reading`</sub>

<details><summary>Answer</summary>

A test that explicitly parametrizes `float('nan')` (and `float('inf')` where
relevant) as one of the "invalid input" cases at every public entry point,
rather than relying on ordinary negative/zero examples to stand in for the
whole invalid-input space. The fix itself (explicit `isfinite`/`is_finite`
checks replacing bare comparisons) only changes what gets rejected before
computation begins; it does not touch any formula, so for every finite input
the computed numbers are unchanged.

</details>

**Q6. Spot-checking `parametric_var.py`, how does `portfolio_sigma` guard against exactly this class of bug rather than relying on a bare `<=` check?**

<sub>Tags: `L4`, `postmortems`, `numeric`, `code-reading`</sub>

<details><summary>Answer</summary>

`portfolio_sigma` validates both inputs as finite first
(`finite_array("exposures", ...)`, `finite_array("covariance", ...)` and
`_validate_finite_array`, whose docstring notes that "every comparison with
NaN is False, so sign/shape guards alone let a NaN covariance or exposure
through"). Only then does it compute `var = w @ Sigma @ w`, raise if `var < -1e-10 * max(1, |w|_max^2)`,
and return `sqrt(max(var, 0.0))`. The order matters: by the time the `max(x, 0.0)`
shape that is dangerous elsewhere runs, a NaN can no longer be present, so it
cannot be turned into a VaR of zero.

</details>

---

**Q7. In plain language, what failure mode does the implied-vol stopping rule have to avoid?**

<sub>Tags: `L3`, `postmortems`</sub>

<details><summary>Answer</summary>

Declaring convergence on a proxy. A Newton-type solver that stops as soon as
the *price* residual is small, or stops improving, can exit well before sigma
is accurate wherever vega is small (deep ITM/OTM, long-dated, high vol),
because a tiny price gap maps through a near-zero vega to a large vol gap. The
shared solver in all six options engines avoids this by stopping on the
relative step in total vol, `|s_new - s| <= max(tol, 4 eps) * s`, after a
Halley or Newton step on an objective transformed to be nearly linear in `s`
(Set 3, Q11-Q12).

</details>

**Q8. Why is the deep-ITM, long-dated, high-vol FX corner the most dangerous case for an implied-vol solver?**

<sub>Tags: `L4`, `postmortems`, `code-reading`</sub>

<details><summary>Answer</summary>

Because there the price saturates: `N(d1)` and `N(d2)` round to 0/1 in double
precision, so `gk_price(sigma)` equals the `sigma -> inf` bound across a whole
plateau of sigma values. Any point on that plateau reprices the quote equally
well, so a solver that returns one is returning an arbitrary number that looks
converged. The FX `implied_vol` raises instead when the price is within `tau = 4 eps * upper`
of the bound (`test_long_dated_deep_itm_high_vol_flat_plateau_raises`).

</details>

**Q9. Why can't a fixed-tolerance round-trip test at typical inputs tell a correct solver from one that stops early?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

Because a solver that stops early but happens to land close enough to the true answer passes that test exactly the same way a solver that converged precisely does. The two only diverge in the regions (deep ITM/OTM, long-dated, high-vol) where "close enough" and "correct" separate, and a suite built around at-the-money, moderate-expiry cases rarely visits them.

</details>

**Q10. What tests does the portfolio use instead to show the solver is accurate everywhere it claims to be?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

`test_iv_round_trip_against_mpmath` sweeps a grid of log-moneyness, expiry,
vol and call/put, prices each case in mpmath, and requires the recovered vol
to be within `4 * max(eps * max(price, S, K) / vega, 1e-12 * sigma)`, the
accuracy the quote can support; it also caps the iteration count at 10.
Targeted cases cover the corners
(`test_round_trip_long_dated_high_vol_flat_vega`,
`test_k90_otm_put_recovers_to_1e12`, `test_k90_itm_call_is_backward_stable`),
and the golden implied-vol cases invert mpmath quotes whose true vol is known
by construction.

</details>

**Q11. How does the current solver guarantee it never returns a stalled, unconverged answer?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

It has only two exits: a converged step (the relative step test, or a bracket
narrower than `4 eps * s`) or a `ValueError` after 100 iterations. Every
Halley or Newton step is checked against a bracket that shrinks with each
evaluation, and a step that would leave the bracket is replaced by a probe of
the bracket end or a bisection step, so the iteration always makes progress.
There is no "return the best so far" path.

</details>

**Q12. Spot-checking the equity Python engine, where does `implied_vol` in `black_scholes.py` delegate the actual solve, and what does it do itself?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

`implied_vol` validates the inputs, checks the quote against the no-arbitrage bounds, maps it to normalised Black form on the out-of-the-money side, and calls `qp_core.iv_core.black_implied_vol`, which runs `total_vol` (Set 3). Its old `tol`, `max_iter`, `sigma_lo` and `sigma_hi` keywords are deprecated and ignored with a `DeprecationWarning`, because the solver has no tuning parameters.

</details>

---

**Q13. In plain language, why can a grid-based monotonicity check for Cornish-Fisher accept an invalid quantile?**

<sub>Tags: `L3`, `postmortems`</sub>

<details><summary>Answer</summary>

The Cornish-Fisher expansion is only a valid quantile function where its derivative in `z` stays positive. A check that samples the derivative on a finite grid of `z` values can miss a thin non-monotone dip that falls entirely between two nodes, and then silently accepts a "quantile" that is not one. That is why all six VaR engines use the closed-form `cornish_fisher_valid` instead of a grid.

</details>

**Q14. How narrow can such a dip be? Is there a concrete documented example?**

<sub>Tags: `L4`, `postmortems`, `numeric`</sub>

<details><summary>Answer</summary>

Yes: `(skew, excess_kurt) = (-0.0105, 8.0001)` has its minimum derivative,
about `-2.5e-7`, at `z = 0.00175`. A 2001-point grid over `|z| <= 3.5` misses
it, while the closed form finds it and marks the pair invalid at every alpha
(`test_domain_check_is_exact_not_grid_resolution_dependent`). Contrast
`(0.122, -0.427)`, whose minimum is at the endpoint `z = -4`: it is valid at
`alpha = 0.01` because that endpoint is outside `[-z_a, z_a]`.

</details>

**Q15. Why would a test suite built from a few hand-picked (skew, kurtosis) pairs fail to expose a grid check's blind spot?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

Because a grid-based check and a test built from a few example points share the same blind spot: if none of the test's pairs lands in the narrow band where the grid steps over the dip, the implementation and the test agree, wrongly, that the region is monotone. "Obviously fine" and "obviously bad" examples never force a test into that band.

</details>

**Q16. What kind of test does catch it?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

A comparison against a resolution-independent reference across many inputs:
for random or swept (skew, kurtosis, alpha) triples, check that the verdict
agrees with the exact minimum of the quadratic derivative.
`test_cf_valid_tightens_with_smaller_alpha` and
`test_cf_counterexample_minimum_is_at_the_endpoint_and_alpha_aware` pin the
alpha dependence, and the golden vectors include CF validity cases
(`conv_cf_valid_*` in the equity VaR generator) so every engine is checked on
the same pairs.

</details>

**Q17. How does `cornish_fisher_valid` work, and does it depend on any resolution parameter?**

<sub>Tags: `L4`, `postmortems`, `code-reading`</sub>

<details><summary>Answer</summary>

The derivative of the CF expansion is an exact quadratic in `z`: `g(z) = A z^2 + B z + C`
with `A = K/8 - S^2/6`, `B = S/3`, `C = 1 - K/8 + 5S^2/36`.
`cornish_fisher_valid(skew, excess_kurt, alpha)` takes its exact minimum on
`[-z_a, z_a]`, `z_a = Phi^{-1}(1 - alpha)`: both endpoints, plus the vertex
`-B/(2A)` when `A > 0` and the vertex lies inside. There is no grid and no
resolution parameter, so there is no spacing for a non-monotone region to hide
in.

</details>

**Q18. What happened to the old `cornish_fisher_domain_ok` function?**

<sub>Tags: `L4`, `postmortems`, `numeric`</sub>

<details><summary>Answer</summary>

It is deprecated and kept for one release with its old meaning: monotonicity
on the fixed interval `[-z_range, z_range]` regardless of alpha, now computed
with the same exact minimum. Calling it raises a `DeprecationWarning` pointing
to `cornish_fisher_valid`, and its `n_grid` argument is validated but unused.
The fixed interval was the second problem with the old check: it ignored
alpha, so it could reject pairs like `(0.122, -0.427)` that are valid at the
quantile actually requested.

Source: `python/equity/03-var-es-engine/src/eq_var/parametric_var.py` (`cornish_fisher_domain_ok`).

</details>

---

**Q19. In plain language, why is a kernel-density standard error for Monte Carlo VaR unreliable in the tail?**

<sub>Tags: `L3`, `postmortems`</sub>

<details><summary>Answer</summary>

The asymptotic formula `SE = sqrt(alpha (1 - alpha) / n) / f(q)` needs the P&L
density at the VaR quantile. A Gaussian kernel density with a bandwidth chosen
for the bulk of the distribution (Silverman's rule) under-resolves the sparse
tail, so the density estimate there is biased and the SE inherits a
directional error, worst at deep alphas and small scenario counts. That is why
every engine reports the bootstrap SE and keeps the KDE number only as a
diagnostic.

</details>

**Q20. Why is a systematically understated standard error worse than a noisy one?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

Because a standard error is a claim about how much a VaR estimate might swing
from one sample to the next; a desk uses it to size confidence intervals and
to judge whether an observed VaR change is signal or noise. An SE that is too
small every time makes the estimate look more precise than it is,
consistently, and does so exactly where the estimate matters most (deep
tails). Random noise in the SE averages out across days; a directional bias
does not.

</details>

**Q21. Why can't a "within 3 SE of the reference" convergence test detect a biased SE?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

Because that test uses the *reported* SE to build its band. A systematically
too-narrow SE just makes the band a bit tighter, and a modest understatement
rarely turns a pass into a fail on any single run. Nothing in that design
compares the reported SE itself against an independent estimate of the true
sampling variability: the SE is trusted as an input to the test rather than
treated as a quantity under test.

</details>

**Q22. What does test an SE estimator properly, and what do the engines report as a result?**

<sub>Tags: `L3`, `postmortems`</sub>

<details><summary>Answer</summary>

A benchmark of the reported SE against ground truth: many independent full
Monte Carlo VaR replications, whose standard deviation across replications is
the true sampling SE, compared with what the estimator reports on a single
replication, averaged over many trials to separate bias from noise. On that
basis every engine reports the distribution-free bootstrap SE as `se_var`
(`var_standard_error_bootstrap` in Python, `mc_bootstrap_se` in C++, the
bootstrap SE in Rust) and keeps the KDE SE as a diagnostic
(`var_standard_error_kde` in equity Python, `var_standard_error` in FX
Python).

</details>

**Q23. Spot-checking `monte_carlo_var.py`, how does the bootstrap SE avoid needing a bandwidth at all?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

It resamples the realized scenario P&L with replacement `n_boot` times,
applies the exact same order-statistic VaR rule used for the point estimate to
each resample (the tail rank depends only on `n` and `alpha`, so it is
identical across resamples of the same size), and reports the standard
deviation of the resulting VaR values (`ddof = 1`). There is no density
estimate anywhere in that computation, so it cannot inherit a bandwidth bias;
the tradeoff is trial-to-trial variance in the SE estimate itself unless
`n_boot` is generous.

</details>

---

**Q24. In plain language, why does an odd `n_paths` break antithetic Monte Carlo in the Heston engines?**

<sub>Tags: `L3`, `postmortems`</sub>

<details><summary>Answer</summary>

`simulate_terminal` (FX) and `simulate_heston_terminal` (equity, when
`antithetic=True`) mirror base normals to `[z, -z]`, and `mc_price` averages
each `(z, -z)` pair before computing the sample variance, which is what
removes the antithetic correlation from the SE. With an odd count, the last
base draw would lose its partner and the halves would no longer line up, so
the pairwise averaging would treat a broken pairing as clean and report a
standard error that does not describe the estimator. Both engines therefore
raise `ValueError` on odd `n_paths` with antithetics, tested by
`test_odd_n_paths_with_antithetic_raises` in each project.

</details>

**Q25. Why is that the same class of problem as the other four in this round?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

Because the common thread is a plausible-looking wrong number rather than a
crash: an odd `n_paths` would not error out on its own, it would quietly
degrade the pairing invariant the SE depends on, giving an SE that looks
normal but is wrong. That is the same "confidently wrong, not visibly broken"
shape as a NaN turned into a zero VaR, a solver that stops on a proxy, a grid
that misses a dip, and a bandwidth-biased SE.

</details>

**Q26. How is the odd-`n_paths` guard different in kind from the solver and domain-check designs in this round?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

It rejects an input the code cannot handle correctly, instead of computing more robustly. The message states the mechanism ("odd n_paths breaks pairing and silently understates the standard error"). That puts it closer to the NaN guards (validate before computing) than to the implied-vol stopping rule or the exact Cornish-Fisher check (compute the right thing in every case).

</details>

---

**Q27. Across all five cases, what is the one property that makes them dangerous, and how do you tell a formula change from a robustness change?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

Each produces a plausible-looking, silently wrong number rather than a crash:
a zero VaR that looks hedged, an implied vol that looks converged, a
Cornish-Fisher quantile that looks like a quantile, an SE that looks
appropriately small, an antithetic SE that looks normal. That is more
dangerous than a loud failure, because a loud failure gets investigated and a
quiet one gets trusted. To tell a formula change from a robustness change,
compare against the golden vectors and the independent oracle: a robustness
change leaves every well-conditioned golden case inside its tolerance and
changes behaviour only at the edges; a formula change moves the middle of the
domain too.

</details>

**Q28. Why does it matter that a robustness change leaves the well-conditioned golden cases untouched?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

Because it narrows the review to the edge-case behaviour. If a robustness
change also moved values in the well-behaved middle of the input space, you
could no longer tell from the diff whether you fixed the edge case or
introduced a regression in the core computation. With the middle pinned by
golden vectors (and, for options, by an mpmath oracle), all that is left to
review is the narrow behaviour the change targeted: does it now handle NaN, a
flat-vega corner, a thin non-monotone dip, a deep-tail SE or an odd path count
correctly.

</details>

**Q29. Stated as a general principle, what is the transferable lesson across all five cases?**

<sub>Tags: `L4`, `postmortems`</sub>

<details><summary>Answer</summary>

A passing test suite is evidence of correctness only within the space of
inputs and conditions it actually exercises; it says nothing about the inputs
it never tried. Each of these failure modes lives outside the representative
examples a suite is naturally built from (typical valid/invalid inputs,
typical moneyness and expiry, typical skew/kurtosis, typical scenario counts,
even path counts), so each needs a test aimed at the corner where it lives.

</details>

**Q30. What is the specific question this round wants you to walk away asking about any piece of numerical code you review, and how does it differ from asking "does this test pass"?**

<sub>Tags: `L3`, `postmortems`</sub>

<details><summary>Answer</summary>

Ask "what property should always hold here, regardless of implementation, and
am I actually testing that property, or just a handful of examples?": a NaN
input should always be rejected, not just `-1` and `0`; a solver should be
accurate to what the input supports across its whole domain, checked against
an independent oracle; a domain check should agree with the exact analytic
answer for *any* (skew, kurtosis, alpha), not just a hand-picked set; a
reported standard error should track the true sampling variability across
scenario counts and tail depths; an invariant like antithetic pairing should
hold for *any* valid input shape. "Does this test pass" only tells you the
code agrees with the examples you already thought to write down; asking what
property must always hold is what surfaces the examples you didn't think to
write down.

</details>


## Set 19: Shared Kernels, Honest Statistics and Safe Builds

This set covers seven ideas behind the most recent changes: one option
kernel in cost-of-carry form, the quantile convention for historical VaR,
optional stopping, statistical power, false discovery rate, hash-checked
installs and honest latency measurement. Each starts with a short primer
(the plain idea, the formal statement, a worked example), then questions.

### Primer A: Black-Scholes in cost-of-carry form

Plain idea. A stock paying a dividend yield, a currency earning a foreign
interest rate and a futures contract are priced by one formula. Only the
rate at which the underlying drifts under the pricing measure changes, and
that rate is the cost of carry `b`.

Formal statement (Generalised Black-Scholes; Garman and Kohlhagen 1983;
Haug 2007, ch. 1). With `phi = +1` for a call, `-1` for a put, and the
carry yield `q = r - b`:

    d1 = [ln(S/K) + (b + sigma^2/2) T] / (sigma sqrt T),   d2 = d1 - sigma sqrt T
    V  = phi [S e^{-qT} N(phi d1) - K e^{-rT} N(phi d2)]

Equity: `b = r - q`. FX (Garman-Kohlhagen): `r = r_d`, `b = r_d - r_f`.
Futures (Black 1976): `S = F`, `b = 0`.

Worked example (`python/core/src/qp_core/black_scholes.py` docstring).
S = K = 100, T = 1, r = 5%, q = 2%, sigma = 20%: `b = 3%`, `d1 = 0.25`,
`d2 = 0.05`, call = 9.2270. A EURUSD-style call with r_d = 5%, r_f = 2%
gives the same number.

### Primer B: which sample quantile is "the" VaR

Plain idea. "The 1% worst day" of 1,000 days is not one number until you
say how to read between sorted observations. Software packages offer nine
ways (Hyndman and Fan 1996).

Formal statement. Type 1 is the inverse of the empirical distribution
function: VaR is the loss L(m) with m = ceil(n alpha), no interpolation.
Type 7 (numpy's default) interpolates linearly at position alpha (n - 1).
Every historical VaR path here uses type 1 (`docs/MARKET_RISK.md`), the
lower-quantile VaR on which Acerbi and Tasche (2002) build expected
shortfall.

Worked example. 1,000 N(0,1) draws (numpy seed 0), 99%: plain historical
VaR is the 10th worst loss, 2.506. The deprecated P&L-level filtered
historical VaR read its tail with type 7 and gave 2.778; with type 1 it
gives 2.815. Golden `a_fhs_var_5pct_lam094` moved from 109.3777910164513
to 112.05212695060598, and no other golden moved.

### Primer C: optional stopping

Plain idea. If you check a running test every month and stop the first time
it looks significant, you will "find" an effect far more often than the 5%
your test promises, even when there is none.

Formal statement. A one-sided test at level 5% has a 5% false-positive rate
for one look at a fixed sample. With k looks and "stop at the first
rejection", the false-positive rate is P(at least one of k correlated test
statistics crosses 1.645), which grows with k towards 1 (Armitage, McPherson
and Rowe 1969). Sequential designs fix this by spending alpha across looks;
the forward registrations here fix it more simply, with one look.

Worked example. Both forward registrations (`research/prereg/pairs-fx-g10-v2.md`,
`research/prereg/momentum-fx-g10.md`) score fixes from 2026-11-03 to
2028-11-02 once. `research/forward/evaluate.py` gives a verdict only on or
after 2028-11-13, once the log is complete and checked; before that it
prints INTERIM numbers and writes nothing.

### Primer D: statistical power

Plain idea. Power is the chance that the test says "effect" when the effect
is really there. A low-power test usually fails even when the idea is
right, so a fail tells you little.

Formal statement. Power = P(reject H0 | H1 true at a stated size). The
momentum registration's rule is "reject if Sharpe - 1.645 x Lo SE > 0" on
504 forward days. `research/studies/momentum_power.py` plants momentum of
known strength in synthetic G10 returns and counts rejections over 200
runs per cell (`research/power/momentum-fx-g10.json`).

Worked example. Planted drift with a mean Sharpe of 0.36 (the "about 0.4"
cell): 13% rejection. Mean Sharpe 1.29: 53%. No planted drift: 1.5%. So
if momentum in G10 FX really earns a Sharpe near 0.4, the forward test
still fails about 87 times in 100. `prereg.py sign` refuses a registration
that names no power file.

### Primer E: false discovery rate in pair selection

Plain idea. Test 36 pairs at 5% and about two pass by luck each year. The
question is how many of the passes are more than luck.

Formal statement. Benjamini and Hochberg (1995): sort m p-values,
p(1) <= ... <= p(m), and keep the largest k with p(k) <= k q / m. Under
independence or positive regression dependence (PRDS) the expected share
of false discoveries among the kept is at most q; under arbitrary
dependence the Benjamini-Yekutieli version (q divided by the sum of 1/i)
is needed. A
complementary check compares the number selected with a null built by
resampling whole days, which keeps the cross-section but destroys
mean reversion (`research/studies/robustness.py`).

Worked example (`research/results/pairs_selection_calibration.json`). Over
19 training folds, 2003-2021, the registered rule selected 53 pairs; the
null averaged 22.2 (95th percentile 34), permutation p = 0.010. Only 4 of
the 19 years beat the null at 5%, and BH at q = 0.05 keeps 14 pairs, 8 of
them in 2018, none in 15 years.

### Primer F: hash-checked installs

Plain idea. A version number names a package; a hash names the exact file.
Pinning versions alone trusts the package index to serve the same file
forever. Pinning hashes does not.

Formal statement. `uv.lock` records the sha256 of every wheel.
`.github/scripts/hashed_install.sh` exports that closure and installs it
with `pip --require-hashes --only-binary :all: --no-deps`: a file whose
hash differs, a missing hash, an sdist (which would run arbitrary build
code) or a package not in the export all fail. Local projects then go in
with `--no-deps --no-build-isolation`, so the build backend comes from the
same hashed set, and `pip check` confirms the set is consistent.

Worked example. In the local dry run, 79 of 79 installs (core, the 23
projects, research and bindings sets on Python 3.11, 3.12 and 3.13, plus
lint) succeeded, and an export with one hash edited was refused with
"THESE PACKAGES DO NOT MATCH THE HASHES".

### Primer G: honest latency measurement

Plain idea. One timing is noise. Report the middle of many timings and how
wide they spread, and compare two versions on the same machine, in turns.

Formal statement (`docs/BENCHMARKS.md`). Each sample is a batch of calls
lasting at least 2 us, divided by its size; at least 31 samples; p10, p50
and p90 by nearest rank. The CI gate runs base and head alternately three
times on one runner pinned to one CPU, and fails a kernel only if the head
median is more than 25% above the base median and head p10 is above base
p90.

Worked example. When the option maths moved into the shared kernel, an
unchanged kernel (`black76_price`) timed in the same runs moved by a ratio
of 0.83 to 1.10. Every changed kernel's before/after ratio fell inside that
band (largest: Clang `gk_greeks_batch_fast`, 1.12 with overlapping
bands), so none was called a regression.

**Q1. Why can one function price an equity option, an FX option and an option on a futures price?**

<sub>Tags: `L3`, `kernels-stats-builds`</sub>

<details><summary>Answer</summary>

All three are European payoffs on an underlying that grows at a known rate under the pricing measure, so only the carry `b` differs: `r - q` for a stock with yield `q`, `r_d - r_f` for a currency, 0 for a futures price. `qp_core.black_scholes.carry_price(S, K, T, r, q, sigma, phi)` takes the discount rate and the carry yield `q = r - b`; FX passes `r_d` and `r_f`, futures pass `q = r`.

</details>

**Q2. Why does the kernel take `r` and `q` rather than `r` and `b`?**

<sub>Tags: `L3`, `kernels-stats-builds`</sub>

<details><summary>Answer</summary>

So the two discount factors `e^{-qT}` and `e^{-rT}` are computed from the caller's own numbers. Recovering `q` as `r - b` would round in the last bit, and the packages' golden values, which were generated before the kernel existed, would move. Taking `q` directly is how every in-scope output stayed bit-identical on the 5,944-quote check grid.

</details>

**Q3. Why do the C++ and Rust kernels take the normal CDF as a template or type parameter?**

<sub>Tags: `L3`, `kernels-stats-builds`</sub>

<details><summary>Answer</summary>

The equity and FX engines already used different normal CDF implementations (in C++, `norm_cdf_rcp` against `norm_cdf`; in Rust, plain erfc against a C1-truncated erfc). Forcing one would change bits in one engine's goldens. A type parameter shares the formula and keeps each engine's numbers exactly as they were.

</details>

**Q4. The Python FX `analytic_greeks` now uses the shared normal CDF. How large is the change, and why make it?**

<sub>Tags: `L3`, `kernels-stats-builds`, `numeric`</sub>

<details><summary>Answer</summary>

It used to call `scipy.stats.norm.cdf` for its N(phi d) terms while everything
else used `0.5 erfc(-x / sqrt 2)`. On the 5,944-quote grid the switch changed
price by at most 2.13e-14 relative to max(1, |value|) and the Greeks by at
most 2.66e-15 (gamma, vega, vanna and volga not at all; table in
`python/fx/01-options-pricing/docs/VALIDATION.md` section 7). That is far
inside every test tolerance, and it leaves one normal CDF across the Python
option code with full relative precision in the lower tail.

</details>

**Q5. Why does `carry_price` with `q = r` not match `black76_price` bit for bit?**

<sub>Tags: `L4`, `kernels-stats-builds`, `numeric`</sub>

<details><summary>Answer</summary>

They are mathematically equal (Black 1976 is the `b = 0` case), but `black76_price` is separate code that forms the terms in a different order. Floating-point addition is not associative, so the results differ in the last bits: 7.577082146427273 against 7.57708214642728, about 1e-15 relative.

Source: `docs/COOKBOOK.md` (the carry-form recipe) and `python/core/src/qp_core/black_scholes.py`.

</details>

**Q6. Historical VaR at 99% on 1,000 days: what are the type 1 and type 7 answers, and why pick type 1?**

<sub>Tags: `L3`, `kernels-stats-builds`, `numeric`</sub>

<details><summary>Answer</summary>

Type 1 is the 10th worst loss, L(ceil(1000 x 0.01)). Type 7 interpolates at
position alpha (n - 1) = 9.99 between two order statistics. Type 1 returns an
observed P&L, so Python, C++ and Rust agree exactly, and VaR and expected
shortfall come from the same sorted tail (Acerbi and Tasche 2002). On the
seed-0 N(0,1) sample the deprecated filtered VaR moved from 2.778 (type 7) to
2.815 (type 1); plain historical VaR there is 2.506.

Source: `docs/MARKET_RISK.md` (the quantile convention table).

</details>

**Q7. After the convention change, the gap between plain historical VaR (2.506) and filtered VaR (2.815) is larger than before. Is that worse?**

<sub>Tags: `L3`, `kernels-stats-builds`, `numeric`</sub>

<details><summary>Answer</summary>

No. Before, part of the 2.506 vs 2.778 gap was the quantile convention, so the two numbers were not comparable. Now both use type 1, and the whole gap is the volatility filter, which is the thing you want to measure.

Source: `docs/MARKET_RISK.md` (the quantile convention table).

</details>

**Q8. What is optional stopping, and how large can its effect be?**

<sub>Tags: `L2`, `kernels-stats-builds`</sub>

<details><summary>Answer</summary>

Repeatedly testing accumulating data and stopping at the first significant result. Each look is another chance to cross the threshold by luck, so the false-positive rate climbs with the number of looks and tends to 1 if you look forever. A test that promises 5% for one look can give several times that with monthly looks over two years.

</details>

**Q9. Why is the forward test scored only on or after 2028-11-13, when the window ends on 2028-11-02?**

<sub>Tags: `L3`, `kernels-stats-builds`, `code-reading`</sub>

<details><summary>Answer</summary>

The window is 2026-11-03 to 2028-11-02. The extra days let the last ECB fixes arrive and the log be checked for gaps before anyone looks. `evaluate.py` refuses a verdict before 2028-11-13, if any ECB business day is missing, if the history file's SHA-256 does not match, or if the registration is unsigned.

</details>

**Q10. What should you do with an INTERIM figure from `evaluate.py`?**

<sub>Tags: `L3`, `kernels-stats-builds`</sub>

<details><summary>Answer</summary>

Use it to check that the log is being written and parsed. Do not change parameters, stop the study or announce a result on it: the decision rule was set for one look, and acting on interim looks is optional stopping. The script writes no result file for that reason.

</details>

**Q11. The momentum test has about 13% power at a true Sharpe of about 0.4. If it fails in 2028, what have we learned?**

<sub>Tags: `L3`, `kernels-stats-builds`, `numeric`</sub>

<details><summary>Answer</summary>

Very little about whether momentum exists at that size: a real Sharpe-0.4 effect would fail about 87 times in 100. A fail rules out large effects (power is 53% near Sharpe 1.3), not small ones. That is why the registration says a fail is the expected outcome, and why `prereg.py sign` demands a power file: it makes the weakness visible before the data arrives, not after.

</details>

**Q12. Why is a power check run on synthetic data with a planted effect, rather than on history?**

<sub>Tags: `L3`, `kernels-stats-builds`, `numeric`</sub>

<details><summary>Answer</summary>

Power is defined relative to a known truth. On history you do not know whether momentum is present or how strong it is, so you cannot count how often the test detects it. Planting an effect of known size in synthetic returns with realistic volatility (daily sd 0.006 here) gives a true positive to measure against.

Source: `research/power/momentum-fx-g10.json` (`daily_sd`).

</details>

**Q13. Pair selection picked 53 pairs over 19 years against 22.2 expected by chance. Is the selection working?**

<sub>Tags: `L3`, `kernels-stats-builds`, `numeric`</sub>

<details><summary>Answer</summary>

Partly. Across all years it finds more than chance (permutation p = 0.010). Year by year it mostly does not: only 4 of 19 years beat the null at 5%, and Benjamini-Hochberg at q = 0.05 keeps 14 pairs, none in 15 of the years. So in most years the selected pairs are what chance alone would produce, and a forward pairs result should be read with that in mind.

Source: `research/README.md` and `research/results/pairs_selection_calibration.json`.

</details>

**Q14. Why does the pair-selection null resample whole days with replacement rather than permute them?**

<sub>Tags: `L3`, `kernels-stats-builds`, `code-reading`</sub>

<details><summary>Answer</summary>

Resampling rows keeps each day's cross-section (correlations, fat tails, the
dollar factor) and destroys order in time, so no spread can mean-revert.
Permuting each series' returns also destroys order in time, but it keeps each
series' start and end points, so the rebuilt spread is a random-walk bridge
pinned at both ends. A bridge is not stationary, but because it is pulled back
to its fixed end point it passes stationarity tests more often than a free
random walk, and the null becomes too hard to beat. The unit tests in
`research/tests/test_robustness.py` show this.

</details>

**Q15. What does Benjamini-Hochberg control, and how does that differ from Bonferroni?**

<sub>Tags: `L2`, `kernels-stats-builds`</sub>

<details><summary>Answer</summary>

BH controls the false discovery rate: the expected share of false positives among the pairs kept, at most q. Bonferroni controls the family-wise error rate, the chance of any false positive at all, at alpha by testing each at alpha / m. With 36 candidates a year, Bonferroni would keep almost nothing; BH keeps more while bounding the share of mistakes.

</details>

**Q16. Version pins already fix which package is installed. What do hashes add?**

<sub>Tags: `L3`, `kernels-stats-builds`</sub>

<details><summary>Answer</summary>

A version number names a release; a hash names an exact file. If an index served a different file under the same version (a compromised account, a re-upload, a mirror), version pins would install it and hashes would not. `--only-binary :all:` also refuses sdists, whose build step runs arbitrary code on the CI machine.

</details>

**Q17. Why does `hashed_install.sh` install the local projects with `--no-build-isolation`?**

<sub>Tags: `L3`, `kernels-stats-builds`</sub>

<details><summary>Answer</summary>

With build isolation, pip fetches the build backend (setuptools, scikit-build-core, maturin) into a temporary environment from the index, unpinned and unhashed. Without it, the backend comes from the hashed set already installed (the `backend` and `build` dependency groups in `uv.lock`), so nothing enters the build that the lock did not name.

</details>

**Q18. Why does every bench sample time a batch of calls rather than one call?**

<sub>Tags: `L3`, `kernels-stats-builds`</sub>

<details><summary>Answer</summary>

A 100 ns call is close to what the clock can resolve, and reading the clock costs time itself. Timing a batch of at least 2 us and dividing by the batch size puts the clock's cost and resolution well below the measurement. The price is that p99 and max of a fast kernel are batch averages, not single-call tails, which `docs/BENCHMARKS.md` states.

</details>

**Q19. Why does the CI gate compare base and head on the same runner, in alternating rounds, rather than against stored numbers?**

<sub>Tags: `L3`, `kernels-stats-builds`</sub>

<details><summary>Answer</summary>

Cloud runners differ in CPU and load from run to run, so a stored number from another machine mixes the machine's speed with the code's. Building both commits on one runner and running base, head, base, head, base, head means both see the same hardware, and slow drift in the machine's state hits both sides.

</details>

**Q20. Why does the gate need both a 25% median rise and non-overlapping bands?**

<sub>Tags: `L3`, `kernels-stats-builds`, `numeric`</sub>

<details><summary>Answer</summary>

Either test alone gives false alarms. A 25% median rise with overlapping p10-p90 bands can be noise on a shared runner; separate bands with a 5% rise are real but too small for three rounds to measure reliably. Requiring both means a failure is large and clearly outside the spread. Smaller regressions are reported in the summary, not failed.

Source: `.github/scripts/bench_ab.py` (the FAIL rule).

</details>

**Q21. How do you tell whether a 1.08 before/after latency ratio on a laptop is a regression?**

<sub>Tags: `L3`, `kernels-stats-builds`, `numeric`</sub>

<details><summary>Answer</summary>

Time an unchanged kernel in the same runs as a control. When the option kernels moved into the shared core, the control `black76_price` moved by 0.83 to 1.10 with no code change, so a 1.08 on a changed kernel is inside the noise. The arithmetic was also shown bit-identical, so any real difference could only come from instruction scheduling; the right next step is the CI gate, not a code change.

Source: `docs/COOKBOOK.md` (the latency recipe).

</details>
