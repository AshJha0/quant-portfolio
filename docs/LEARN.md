# Learn: A Structured Guide to This Portfolio

This is the teaching companion to quant-portfolio. [README.md](../README.md)
tells you what the portfolio contains and proves it works (5,963 passing
tests, reproduced from a clean build). [ARCHITECTURE.md](ARCHITECTURE.md) and
[DIAGRAMS.md](DIAGRAMS.md) tell you how the 31 sub-projects fit together.
This file teaches the *content* — the quant finance, the numerical methods,
and the engineering practice behind every one of them — from first principles,
for someone who wants to actually learn this material, not just read that it
exists.

It is organized in four parts:

- **Part I — The Finance.** What each of the 13 areas is, in plain language,
  before you open a single line of code.
- **Part II — The Technology.** Why this portfolio is built the way it is:
  Python-first, C++/Rust twins for the hot path, cross-language golden
  vectors, and the testing discipline that ties it together.
- **Part III — Your learning path.** A suggested order to work through the
  portfolio, by background (quant-first, engineer-first, or starting from
  zero), with what to read and run at each step.
- **Part IV — The exercise room.** ~450 self-test questions and answers,
  organized into 18 rounds by topic, grounded in this portfolio's actual code
  and actual documented assumptions — not generic textbook trivia. Use it to
  check your own understanding after working through a project, or to find
  the gaps before you claim you know an area.

Every claim in this file is checkable against the actual repository: file
paths, function names, and test counts are real, not illustrative.

---

## Part I — The Finance

### What problem is each area actually solving?

**Options Pricing & Greeks** (areas 01/01) answers "what should this
option be worth today, and how does that value move if the market moves?"
Black-Scholes (equity) and Garman-Kohlhagen (FX, which is Black-Scholes with
the dividend yield replaced by the foreign interest rate) give a closed-form
answer under a specific set of assumptions — constant volatility, frictionless
markets, continuous trading, no dividends beyond a constant yield, European
exercise. The Greeks (delta, gamma, vega, theta, rho) are the sensitivities a
trading desk actually hedges against, not academic curiosities: a delta-hedged
book is short gamma and long theta by construction, and knowing that trade-off
numerically is the entire point of computing Greeks in the first place.
Implied volatility inverts the pricing formula — given a market price, what
constant volatility would have produced it? — which is how the market
actually quotes options (in vol terms, not price terms) on every real desk.

**Volatility Modeling & Forecasting** (areas 02/02) answers a different
question: not "what is the option worth today" but "what will realized
volatility look like tomorrow, next week, next month?" This matters because
Black-Scholes assumes constant volatility, but volatility is famously not
constant — it clusters (a volatile day is followed by more volatile days),
mean-reverts, and spikes around events. GARCH-family models capture the
clustering; the forecasts feed into everything from position sizing to VaR
inputs to vol-surface construction.

**Market Risk VaR / Expected Shortfall** (areas 03/03) answers "how much
could this portfolio lose, and how bad does it get in the tail?" Value-at-Risk
is a quantile of the P&L distribution (e.g., "the loss you won't exceed 99% of
the time"); Expected Shortfall is the average loss *given* you're in that
worst 1% — a materially different, and regulator-preferred, number because VaR
says nothing about how bad the tail actually is beyond the threshold. This
portfolio implements three independent ways to estimate both (historical
simulation, parametric/delta-normal with a Cornish-Fisher tail correction, and
Monte Carlo with full book revaluation), because a real risk function never
trusts a single VaR methodology — see Round 6 and Round 7 for why they
disagree with each other, and by how much, in specific and predictable ways.

**Fixed Income Pricing & Risk** (areas 04/04) covers bond and interest-rate
instrument pricing and the risk measures specific to rates (duration,
convexity, DV01) — the analog of options Greeks for the rates world, where the
underlying risk factor is a whole curve, not a single spot price.

**Statistical Pairs Trading** (areas 05/05) is market-neutral relative-value
trading: find two instruments whose spread is stationary (cointegrated), trade
mean-reversion in that spread, and stay hedged against the *level* of the
market while still taking a directional bet on the *relationship* between two
assets. The hard part is not the trading logic — it's proving the spread is
actually stationary, in-sample and out-of-sample, before betting on it.

**Credit Risk / PD Modeling** (areas 06/06) answers "what is the probability
this counterparty defaults, and what do we lose if they do?" — probability of
default (PD) modeling, typically via logistic regression or a structural
(Merton-style) model, feeding into expected-loss and capital calculations.

**Portfolio Optimization & Risk Allocation** (areas 07/07) is the
Markowitz mean-variance problem and its descendants (minimum-variance,
maximum-Sharpe, risk parity) — given expected returns and a covariance
estimate, what weights should the portfolio hold, and how is risk actually
distributed across positions once you hold them (marginal/component risk
contribution, not just notional weight).

**Algorithmic Trading & Execution** (areas 08/08) is about *how* you get
into or out of a position, not whether you should — scheduling a large parent
order (TWAP, VWAP, participation-rate) to minimize market impact and
implementation shortfall, and measuring afterward how well you actually did
against a benchmark (transaction cost analysis).

**Volatility Surface & Stochastic Vol** (areas 09/09) extends area 01/01:
real markets don't quote one implied vol per underlying, they quote a whole
surface across strike and expiry (the "smile" and "term structure"), which a
constant-volatility model like Black-Scholes cannot produce internally — you
have to either interpolate the observed surface, or move to a model that
generates a smile endogenously, like Heston (stochastic volatility).

**Regime-Switching Quant Strategy** (areas 10/10) is the idea that markets
behave differently in different states (calm vs. stressed, trending vs.
mean-reverting) and a strategy that detects the current regime and adapts —
rather than using one fixed rule set across all conditions — can do better,
if the regime detection itself is honest about the risk of overfitting to
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
fails, with reproducible examples; how a real desk would actually use it; and
what the real-life edge cases are (deep ITM/OTM, T→0, zero/negative rates,
illiquid/missing data, crisis regimes) — each one both documented *and*
unit-tested, not just described. Part IV's exercise rounds test you against
exactly this bar, area by area.

---

## Part II — The Technology

### Why Python first, and why only four areas get compiled twins

Every model in this portfolio has a Python reference implementation, full
stop — including the four areas (options pricing/Greeks, VaR/ES, both equity
and FX) that also have C++ and Rust performance twins. Python is not a
prototype language here; it is the *source of truth*. The C++/Rust engines are
required to match it, not the other way around, because Python's expressiveness
makes the modeling logic auditable in a way that a hand-optimized C++ kernel
never is — you want to review the math in the language that reads closest to
the math, and optimize a second time, separately, once the semantics are
locked down.

Only options pricing/Greeks and VaR/ES get compiled twins because they are the
two workloads on a real desk that are actually called often enough, and are
numerically well-enough understood, to justify the engineering cost of a
second (or third) implementation: full-book Greeks revaluation and intraday
VaR both run on the order of millions of evaluations per cycle, on a
closed-form-or-Monte-Carlo core that doesn't change its semantics once
correct. The other six areas — calibration, backtesting, portfolio
construction — are research workflows where Python's iteration speed matters
more than microsecond latency, so a second implementation would be pure
engineering cost with no corresponding desk benefit.

### The determinism discipline

Every stochastic component in all 31 projects takes an explicit seed. This
sounds like a small thing until you've debugged a Monte Carlo test that fails
one run in twenty — a flaky test doesn't get fixed, it gets ignored, and an
ignored red CI signal is worse than no test at all. Determinism is what makes
"3 standard errors" a meaningful tolerance rather than a number nobody
actually checks because the test is already known to be unreliable.

### The golden-vector bridge

The single most distinctive engineering pattern in this portfolio is how the
C++ and Rust engines are validated against the Python reference: not by
re-deriving the math independently in each language and hoping they agree
(they might agree because both are wrong the same way), but by literally
generating the Python engine's own output on a fixed set of inputs, freezing
those input/output pairs as "golden vectors," and asserting the C++ and Rust
engines reproduce them to tight tolerance. This is the same discipline a real
market-risk function uses when a new pricing library has to be proven against
the incumbent before touching P&L — see [ARCHITECTURE.md](ARCHITECTURE.md)
"Cross-language validation" and [DIAGRAMS.md](DIAGRAMS.md) diagram 2 for the
full pipeline, and Round 15 below for the engineering theory in depth.

### The testing philosophy in one sentence

A test that only checks "does this match a number I computed by hand once"
catches typos, not wrong models — this portfolio instead leans on tests that
check properties the *correct* model must have regardless of implementation
detail: put-call parity, monotonicity in volatility, convexity in strike,
tree-to-closed-form convergence as steps increase, Greeks agreeing with
finite differences, and (for VaR/ES) Monte Carlo agreeing with the closed
form within a stated number of standard errors. Two portfolio-wide review
passes found real bugs this way — see Round 16 (the NaN-guard defect class)
and Round 18 (the numerical-robustness bugs and how each was actually found)
for the concrete stories, not just the abstract principle.

---

## Part III — Your learning path

Three suggested paths through the portfolio, depending on where you're
starting from. All three converge on the same exercise rounds in Part IV.

### Path A — starting from zero (new to quant finance)

1. Read `python/foundations/03-black-scholes-replication/README.md` and its
   `docs/METHODOLOGY.md`, then read the actual `black_scholes.py` source —
   it's short, zero-scipy, and every line maps to a formula in the docstring.
2. Do the same for `python/foundations/01-risk-metrics` (VaR/ES/Sharpe on a
   single asset) and `python/foundations/02-trading-signal-backtest`
   (backtest discipline — read this one specifically for the no-look-ahead
   argument, it's the single most common real-world backtesting bug).
3. Work Part IV Round 1 (quant & math foundations) and Round 14 (the
   foundations trilogy) before moving on.
4. Move to `python/equity/01-options-pricing` — same model as F3, now
   production-grade: full input validation, implied-vol solving, binomial
   trees for American exercise, Monte Carlo cross-checks.
5. Work Round 2 and Round 3.
6. Pick two or three more areas that interest you (VaR/ES and portfolio
   optimization are a natural next pair — one measures risk, the other
   allocates around it) and read their `docs/METHODOLOGY.md` +
   `docs/DESK_GUIDE.md` before the code.

### Path B — quant-first (comfortable with the math, new to this codebase)

1. Skim `README.md` and [ARCHITECTURE.md](ARCHITECTURE.md) for the shape of
   the portfolio.
2. Go straight to whichever area's math you already know best, read its
   `docs/METHODOLOGY.md` "why this model vs. alternatives" section first —
   this portfolio's answer may differ from your intuition (e.g. why
   Cornish-Fisher rather than a pure historical simulation for parametric
   VaR's tail correction), and that disagreement is exactly where you'll
   learn something.
3. Read `docs/VALIDATION.md` for the same project and try to find a gap in
   its stated failure modes before reading the edge-case tests — then check
   yourself against the actual test file.
4. Work the corresponding Part IV round, then the next area.
5. Round 17 (business/desk context) and Round 18 (postmortems) last — they
   assume you already know the models cold.

### Path C — engineer-first (comfortable with code, want the cross-language story)

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) "Cross-language validation" and
   [DIAGRAMS.md](DIAGRAMS.md) diagrams 1-2, then Round 15 in Part IV.
2. Pick one of the four compiled-twin areas (equity options pricing is the
   simplest starting point). Read the Python `tests/golden/generate_golden.py`
   script, then `tools/gen_golden_header.py` and `tools/gen_golden_rs.py` —
   understand the generation pipeline before reading either engine.
3. Read the C++ engine's headers (`include/eqopt/*.hpp` or equivalent) side
   by side with the Rust engine's modules (`src/*.rs`) — same algorithm,
   two idiomatic implementations, same golden vectors.
4. Work Round 15 and Round 16 (the NaN defect class — this is where the
   "just validate your inputs" advice gets concrete and portfolio-specific).
5. Read Round 18 (postmortems) for the numerical-robustness bugs — these are
   the kind of bug that only shows up when you already trust your tests, and
   the write-ups are deliberately about what *kind* of test would have
   caught each one sooner, which is the transferable lesson.

---

## Part IV — The exercise room

**481 questions across 18 rounds**, organized to follow Part I's area-by-area
structure, then widen into the cross-cutting engineering and business
material. Each answer is grounded in this portfolio's actual code,
methodology docs, and test suite — where a number, function name, or file
path appears, it is real and checkable. Use this as a self-test: read the
question, try to answer it from what you've learned, then check yourself
against the answer.

| Round | Topic | Questions |
|---|---|---|
| 1 | Quant & Math Foundations | 28 |
| 2 | Options Pricing & Greeks | 30 |
| 3 | Implied Volatility & Numerical Solvers | 26 |
| 4 | Volatility Modeling & Forecasting | 25 |
| 5 | Volatility Surface & Stochastic Vol | 26 |
| 6 | Market Risk: VaR Methods | 30 |
| 7 | Expected Shortfall, Backtesting & Basel | 27 |
| 8 | Fixed Income Pricing & Risk | 25 |
| 9 | Statistical Pairs Trading | 26 |
| 10 | Credit Risk / PD Modeling | 26 |
| 11 | Portfolio Optimization & Risk Allocation | 26 |
| 12 | Algorithmic Trading & Execution | 26 |
| 13 | Regime-Switching Quant Strategy | 26 |
| 14 | The Foundations Trilogy | 22 |
| 15 | Cross-Language Engineering | 28 |
| 16 | Testing, Validation & the NaN Defect Class | 28 |
| 17 | Business, Desk Usage & Model Risk Governance | 26 |
| 18 | Real-World Postmortems & Judgment | 30 |
| **Total** | | **481** |


## Round 1 — Quant & Math Foundations

**Q1. Every VaR method in this portfolio ultimately reduces to "find a quantile of a return distribution." Walk through what that means concretely for `var_historical`.**

`var_historical` in `python/foundations/01-risk-metrics/src/eq_risk_metrics/var_es.py` computes `-np.percentile(returns, 100 * (1 - confidence))`. A quantile is defined by sorting the sample into order statistics and picking (or interpolating between) the one at the target rank — at 95% confidence you want the value below which 5% of observations fall, i.e. `np.percentile(returns, 5)`. This is the "historical" or empirical VaR: no distributional shape is assumed, the quantile is read directly off the realised order statistics, and the function just negates the result because losses are reported as positive numbers by convention.

**Q2. Why is historical VaR described in the codebase as "a single order statistic... a genuinely noisy estimator when n is small"?**

At the 99th percentile on a 250-day sample, the empirical quantile is determined by roughly the 2nd- or 3rd-worst observation out of 250 — swapping one bad day in or out of the window can move the estimate noticeably. `docs/METHODOLOGY.md` for `01-risk-metrics` (§3.1) flags this directly: unlike a parametric estimator that uses every observation to fit `mu` and `sigma`, an order statistic in the tail is supported by only a handful of data points, so its sampling variance is large exactly where you need precision most.

**Q3. What does it mean for the normal distribution to have "thin tails," and why does that make Gaussian VaR systematically wrong for daily equity returns?**

The normal density decays like `exp(-x^2/2)`, so the probability it assigns to an extreme move falls off extremely fast — a 5-standard-deviation daily move is astronomically unlikely under Normal(mu, sigma). Real daily equity returns have excess kurtosis (fatter tails than normal: more mass in the extremes than a Gaussian would predict). `var_parametric` in `var_es.py` assumes `returns ~ Normal(mu, sigma)` and computes `VaR = -(mu + sigma*z)`; because the true tail is fatter than the normal model believes, this quantile sits closer to the center than the real one, so Gaussian VaR *understates* tail risk — precisely the gap the project ships historical VaR alongside it to expose (`docs/METHODOLOGY.md` §3.2).

**Q4. Where does the Student-t distribution show up in this portfolio, and why is it the standard fix for fat tails?**

The Student-t density has heavier tails than the normal — controlled by its degrees-of-freedom parameter `df`, with lower `df` meaning fatter tails and `df -> infinity` recovering the normal exactly. It appears in two places doing two different jobs: as a *return generator*, `python/foundations/01-risk-metrics/src/eq_risk_metrics/data/synthetic.py` builds its synthetic daily-return series from i.i.d. Student-t(6) shocks standardised to unit variance (`_T_DF = 6.0`), so the bundled test data has genuine fat tails rather than being secretly Gaussian; and as a *risk-factor model*, `simulate_factor_returns` in `python/equity/03-var-es-engine/src/eq_var/monte_carlo_var.py` offers `dist="t"`, building a multivariate Student-t via `Z / sqrt(W/df)` (a normal divided by a scaled chi-squared) with the scale matrix set to `cov * (df-2)/df` so the *covariance* still matches the target exactly while the marginal tails fatten.

**Q5. In `simulate_factor_returns`, why is the covariance scaled by `(df-2)/df` before it's fed to Cholesky for the Student-t case?**

A multivariate Student-t built as `Z / sqrt(W/df)` (with `Z` multivariate normal from the Cholesky-factored scale matrix and `W ~ chi2(df)`) has covariance equal to `df/(df-2)` times the *scale* matrix, not the scale matrix itself — the division by `sqrt(W/df)` inflates variance by that factor. To make the simulated covariance actually equal the caller's target `cov`, the code pre-multiplies by the inverse factor `(df-2)/df` before factorising, so the fattened tails come "for free" without silently also changing the second moment the caller asked for.

**Q6. `01-risk-metrics/docs/METHODOLOGY.md` calls the Cornish-Fisher expansion "only a good approximation for mild skew/kurtosis" and warns it can become "non-monotonic." What does non-monotonic mean here and why is it disqualifying?**

Cornish-Fisher adjusts the Gaussian quantile `z` via `z_cf = z + (z^2-1)s/6 + (z^3-3z)k/24 - (2z^3-5z)s^2/36`, a cubic-ish polynomial in `z` parameterised by sample skew `s` and excess kurtosis `k`. A valid quantile function must be monotonically increasing in the confidence level — the 99% quantile must always sit further into the tail than the 95% quantile. With large enough `s`/`k`, the derivative of that expansion can go negative on part of the domain, so the "99% VaR" it produces can actually be *smaller* than the "95% VaR" — a number that is mathematically not a quantile at all. `python/equity/03-var-es-engine` handles this by explicitly checking the analytic derivative on a grid over `|z| <= 3.5` and raising rather than silently returning a broken number (`docs/METHODOLOGY.md` §5).

**Q7. What is an order statistic, and how does the historical-VaR quantile relate to it?**

Sort a sample of `n` observations from smallest to largest; the `k`-th smallest value is the `k`-th order statistic, written `X_(k)`. A sample quantile at probability `p` is (up to interpolation convention) approximately the `X_(round(p*n))` order statistic. `python/equity/03-var-es-engine`'s `var_confidence_interval` (`monte_carlo_var.py`) makes this explicit and quantitative: it treats the *rank* of the true alpha-quantile among `n` i.i.d. draws as Binomial(n, alpha), inverts that via the Beta distribution of order statistics (`beta_dist.cdf(alpha, ranks, n-ranks+1)`), and returns a distribution-free confidence bracket `(-X_(hi), -X_(lo))` around the VaR estimate — a direct, textbook application of order-statistic theory to quantify how uncertain a single empirical quantile actually is.

**Q8. `03-var-es-engine/docs/METHODOLOGY.md` notes that plain historical VaR uses "type-7 (linear) interpolation between order statistics." What does that mean, and why does it matter?**

When the target rank `p*(n-1)` falls between two integers, NumPy's default (`method="linear"`, "type 7") linearly interpolates between the two bracketing order statistics rather than picking one or the other outright — it's the convention used by R, Excel, and most statistical packages by default. The METHODOLOGY doc notes this differs from the age-weighted VaR's necessarily different interpolation (a weighted step-CDF, since interpolating *weighted* order statistics has no single standard definition), and quantifies the practical gap: "the two differ by at most one order statistic (unit-tested: 95.0 vs 95.05 on a 100-point grid)" — small, but a genuine, documented implementation choice rather than an accident.

**Q9. What is a covariance matrix doing in `simulate_factor_returns`, and why can't you just simulate each risk factor independently?**

A covariance matrix `Sigma` encodes not just each factor's variance (the diagonal) but how every pair of factors co-moves (the off-diagonal entries). Simulating each factor with its own independent normal draw would produce a joint distribution where every factor is uncorrelated with every other — which is wrong whenever, say, two equity indices actually move together. `simulate_factor_returns` needs the *simulated* scenarios to reproduce the target `cov` exactly (in expectation), which requires generating correlated draws — the entire reason Cholesky factorisation appears in the function at all.

**Q10. Explain how Cholesky factorisation is actually used to turn independent normal draws into correlated ones, as in `simulate_factor_returns`.**

Cholesky factorisation writes a symmetric positive-definite covariance matrix as `Sigma = L L^T` for a unique lower-triangular `L`. If `Z` is a vector of independent standard normal draws, then `Y = mu + L Z` has covariance `E[(LZ)(LZ)^T] = L E[ZZ^T] L^T = L I L^T = L L^T = Sigma` exactly, because `Z`'s covariance is the identity. `simulate_factor_returns` does exactly this: `chol = safe_cholesky(sig); z = rng.standard_normal((n_paths, n)); return mu + z @ chol.T` — each row of independent normals is linearly mixed by `L` into a row with the desired covariance structure, at essentially the cost of one matrix multiply per path.

**Q11. What does "jitter escalation" mean in `safe_cholesky`, and why is it needed at all?**

Cholesky factorisation is only defined for a matrix that is (numerically) positive *definite*; a matrix that is only positive semi-definite (has a zero eigenvalue) or is indefinite due to floating-point noise makes `np.linalg.cholesky` raise `LinAlgError`. `safe_cholesky` in `python/equity/03-var-es-engine/src/eq_var/monte_carlo_var.py` handles this by adding a small multiple of the identity to the diagonal — `jitter * mean(diag(cov))` — and retrying; if that still fails it multiplies the jitter by 10 and tries again, up to `max_tries` (default 12) times. This is "escalation": start with a barely-perceptible nudge (`1e-10` of the mean variance) and only grow it as much as actually needed, so a matrix that's merely borderline gets a negligible perturbation while a genuinely singular one gets pushed just far enough to factorise — with the perturbation small enough that "simulated moments are unchanged to within MC noise" (module docstring).

**Q12. Why would a covariance matrix built from pegged-FX-currency data be singular or near-singular in the first place?**

A currency peg (e.g. a currency pinned to the same anchor as another) makes the two currencies' returns move together almost perfectly — correlation at or extremely close to 1 — or, if the peg is perfectly rigid, one factor's realised variance is essentially zero. Either condition makes rows/columns of the covariance matrix nearly (or exactly) linearly dependent, which is precisely what "singular" means: the matrix has a zero (or numerically negative, due to rounding) eigenvalue. The C++ FX-VaR engine states this explicitly: "A covariance containing pegged currencies is routinely singular or numerically indefinite (near-zero-vol factors, perfectly correlated pegs to the same anchor)" (`cpp/fx-var-engine/include/fxvar/matrix.hpp`), which is exactly why its `robust_cholesky` (the C++ mirror of the Python `safe_cholesky`) exists as a first-class, documented feature rather than an edge case to special-case away — a genuinely singular FX covariance is a *legitimate* input, not a data error.

**Q13. If jitter escalation lets a "genuinely indefinite" covariance matrix fail rather than being silently repaired, why is that the right design choice rather than always jittering until it works?**

The C++ engine's docstring is explicit about the boundary: with `max_tries = 8` and jitter escalating x10 from `1e-12 * mean(diag)`, the largest perturbation reachable is `1e-5 * mean(diag)` — small enough to only fix numerical near-singularity, not to paper over a matrix that is *structurally* wrong (e.g. built from bad or contradictory correlation inputs, with a genuinely large negative eigenvalue). Silently jittering an arbitrarily bad matrix into something factorisable would let a data-quality bug produce a plausible-looking but meaningless simulation; capping the escalation means a badly indefinite matrix instead raises an error the caller has to investigate — "the result records the jitter actually used" so even a successful jitter is surfaced as a diagnostic, not hidden (`matrix.hpp`, `CholeskyResult`).

**Q14. What is geometric Brownian motion (GBM), and where does it appear explicitly in this portfolio's code?**

GBM is the stochastic process `dS = mu*S*dt + sigma*S*dW`, where `dW` is a Brownian motion increment: the *proportional* (not absolute) change in `S` has constant drift `mu` and constant volatility `sigma`. `python/foundations/03-black-scholes-replication/src/eq_bs_replication/black_scholes.py` states this as assumption 1 in its module docstring, and `monte_carlo.py` uses the SDE's known closed-form solution directly to simulate terminal prices: `S_T = S0 * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z)` for `Z ~ N(0,1)` — the risk-neutral version of GBM's exact solution, not a discretised approximation.

**Q15. Why does the GBM solution have a `- sigma^2/2` term in the drift (`S_T = S0*exp((r - sigma^2/2)T + sigma*sqrt(T)*Z)`) instead of just `r`?**

This is Ito's lemma at work: applying Ito's lemma to `ln(S)` under `dS = r*S*dt + sigma*S*dW` (risk-neutral drift `r`) does not simply give `d(ln S) = r*dt + sigma*dW`, because Ito's lemma for a function of a stochastic process includes a second-order term (`-1/2 * sigma^2*S^2 * (d^2/dS^2) ln S * dt = -1/2*sigma^2*dt`) absent from ordinary calculus. So `ln(S_T/S_0) ~ N((r - sigma^2/2)T, sigma^2 T)`, and exponentiating gives the `- sigma^2/2` "Ito correction" in the exponent. This is exactly why `S` itself has expected growth rate `r` (its mean grows at the risk-free rate, as risk-neutral pricing requires) while `ln(S)` grows at the *lower* rate `r - sigma^2/2` — the correction is what reconciles the two, and it's the same correction that appears inside the Black-Scholes `d1` formula (`(r + sigma^2/2)*T` in the numerator, with the sign flipped because `d1` is derived from a different — but related — conditional expectation).

**Q16. At a qualitative level, why does Ito's lemma applied to a delta-hedged options portfolio lead to the Black-Scholes PDE, and hence to a formula with no dependence on the stock's real-world drift `mu`?**

Building a portfolio long an option and short `delta = dV/dS` shares of stock, then applying Ito's lemma to the option value `V(S,t)` as a function of the GBM process `S`, produces a portfolio whose stochastic (`dW`) term cancels exactly — the hedge removes all first-order sensitivity to the random shock, leaving only deterministic terms in `dt`. A position with literally zero risk must, under no-arbitrage, earn exactly the risk-free rate; setting the portfolio's deterministic drift equal to `r` times its value is what produces the Black-Scholes PDE. Because the risky drift `mu` cancelled out entirely in the hedging argument (it only ever entered through the `dW` term that the hedge removed), the resulting price and PDE contain no `mu` at all — which is why `call_price` in `black_scholes.py` takes `S, K, r, sigma, T` and never a real-world expected-return parameter; risk preferences and drift beliefs are hedged away, not needed. (`METHODOLOGY.md` calls this "the entire chain from the SDE to the price... a single well-understood special function" once you accept the replication argument — the full derivation is left to a later round; this is the qualitative shape of why the formula has no `mu`.)

**Q17. `03-black-scholes-replication/docs/METHODOLOGY.md` interprets `N(d2)` and `N(d1)` economically. What do they mean, and how does that follow from the GBM/Ito machinery above?**

`N(d2)` is the risk-neutral probability that the option finishes in the money — it falls directly out of the lognormal terminal distribution of `S_T` that Ito's lemma produces: `d2` is (up to the `sigma*sqrt(T)` scaling) how many standard deviations `ln(K)` sits below the mean of `ln(S_T)`, so `N(d2) = P(S_T > K)` under the risk-neutral measure. `S0*N(d1)` is the discounted expected value of *receiving the stock*, conditional on exercise — `N(d1)` is not itself a probability under the same measure as `N(d2)`, but the analogous quantity under the "stock numeraire" measure. Both interpretations only make sense because `ln(S_T)` is exactly normal (GBM's defining property under Ito's lemma) — the whole closed form is an integral against that specific lognormal density (`black_scholes.py` module docstring).

**Q18. Why does Monte Carlo estimation error shrink as `1/sqrt(n)`, and where is that fact used explicitly in this portfolio?**

For `n` i.i.d. draws with finite variance `sigma^2`, the standard error of the sample mean is `sigma/sqrt(n)` by the Central Limit Theorem / basic variance-of-a-sum arithmetic (`Var(mean) = Var(X)/n`). `mc_call_price` in `python/foundations/03-black-scholes-replication/src/eq_bs_replication/monte_carlo.py` computes exactly this: `stderr = disc * std(units, ddof=1) / sqrt(len(units))`. The module docstring is explicit that "the estimate must converge to the Black-Scholes price at rate O(1/sqrt(n))," and `docs/VALIDATION.md` documents a convergence table across four sample sizes each agreeing with the closed form within 3 standard errors — with the standard error itself shrinking at the predicted rate.

**Q19. What does "1/sqrt(n) convergence" actually imply about how many extra paths you need to halve your Monte Carlo error?**

Because error scales as `1/sqrt(n)`, halving the error requires *quadrupling* `n`, not doubling it — going from 100,000 paths to 10,000,000 paths (100x more compute) only buys you a 10x reduction in standard error. This is why Monte Carlo is described in the codebase as reliable but expensive to push further: `python/equity/03-var-es-engine/docs/METHODOLOGY.md` notes MC's "estimation noise... controllable (paths up) but model risk remains" and quantifies a concrete operating point — "100k paths -> SE ~ 0.8% of the 99% VaR on the demo book" (assumption A10) — rather than assuming more paths trivially buys arbitrary precision.

**Q20. Why does `mc_call_price` need to compute its standard error from "pair averages" rather than treating every simulated payoff as an independent draw, when antithetic sampling is used?**

Antithetic variance reduction pairs each draw `Z` with `-Z` and averages the two payoffs; because the payoff is a monotone function of `Z` for a call, the two mirrored payoffs are *negatively correlated* by construction — that's precisely the mechanism that reduces variance. Treating all `2m` mirrored payoffs as `2m` independent observations for the standard-error formula ignores that negative correlation and overstates the reported error — the module docstring quantifies this at "about a third" too large for a 100k-path ATM call. The correct estimator instead treats the `m` *pair averages* as the independent unit: `units = 0.5*(payoff[:m] + payoff[m:])`, then `stderr = disc * std(units, ddof=1) / sqrt(m)`. The broader lesson stated in `METHODOLOGY.md`: "a tolerance expressed in units of your own error estimate is only as trustworthy as that estimate" — a validation suite that certifies "within 3 sigma" using a wrongly-computed sigma is certifying its own arithmetic, not the model.

**Q21. What is the bootstrap, and what does `var_standard_error_bootstrap` in the VaR engine actually estimate with it?**

The bootstrap estimates the sampling distribution of a statistic by repeatedly resampling the *observed* data with replacement (each resample the same size as the original) and recomputing the statistic on each resample; the spread of those recomputed values approximates the statistic's true sampling variability, without requiring any assumption about the population's underlying distribution. `var_standard_error_bootstrap` in `python/equity/03-var-es-engine/src/eq_var/monte_carlo_var.py` resamples the simulated (or historical) P&L array with replacement `n_boot` times (`rng.integers(0, arr.size, size=(n_boot, arr.size))`), recomputes the empirical VaR quantile on each resample, and returns the standard deviation of those `n_boot` VaR estimates — a standard error for the VaR quantile itself, attached to a number that otherwise ships as a single point estimate with no error bar.

**Q22. Why is the bootstrap described as "distribution-free," and why does that matter for VaR specifically?**

The bootstrap makes no parametric assumption about the shape of the P&L distribution — it works directly off the empirical sample by resampling it, so whatever skew, kurtosis, or fat-tailedness the real data has is automatically reflected in the resampled statistic's spread, with no need to specify (and potentially misspecify) a normal, Student-t, or any other family. This matters especially for VaR because VaR is a tail quantile — exactly the region where a wrong distributional assumption (as `var_parametric`'s Gaussian assumption demonstrably is, per Q3) does the most damage; the bootstrap's standard error inherits the true, empirical tail shape rather than a possibly-wrong assumed one, which is the codebase's stated reason it is "the desk-standard way to attach error bars to an MC or historical VaR" (`monte_carlo_var.py` docstring).

**Q23. Why does essentially every stochastic function in this portfolio — `mc_call_price`, `simulate_factor_returns`, `var_standard_error_bootstrap`, the synthetic data generators — take an explicit `seed` argument?**

A pseudo-random number generator seeded with the same integer produces the exact same sequence of "random" draws every time. `python/foundations/03-black-scholes-replication/src/eq_bs_replication/monte_carlo.py` states the rationale directly in `mc_call_price`'s docstring: "Every stochastic routine in this project takes an explicit seed so results are reproducible." Without a fixed seed, a test asserting "MC price is within 3 standard errors of the closed form" would be nondeterministic — passing on some runs and failing on others purely from RNG luck, which is exactly the kind of flaky test the portfolio's testing contract (`CONVENTIONS.md`: "tests... deterministic seeds") is designed to rule out. It also makes bug reports reproducible: "seed 2 on the 01-risk-metrics synthetic generator produced this exact number" is a debuggable claim; "some random run produced a weird number" is not.

**Q24. `eq_risk_metrics/data/synthetic.py` documents that seed 2 is "the seed all documented results use." Why does a fixed default seed matter beyond just making individual tests deterministic?**

Every number quoted in `docs/METHODOLOGY.md` and `docs/VALIDATION.md` for `01-risk-metrics` — the measured gap between Gaussian and historical VaR, the specific Kupiec exception counts, the Jarque-Bera rejection — was computed against one specific realisation of the synthetic Student-t return series. If the default seed ever silently changed, every one of those hard-coded numbers in the documentation would become stale and unverifiable without regenerating them; pinning the default seed (and stating it explicitly in the docstring) is what lets a reader independently re-run `examples/run_pipeline.py` and get the identical numbers the documentation claims, rather than "numbers in this general ballpark."

**Q25. What is the difference between correlation and causation, and why is that distinction relevant to a portfolio built around statistical VaR and (later) pairs trading?**

Correlation measures how two series move together statistically; it says nothing about whether one *causes* the other, or whether both are driven by a shared third factor. Two currencies pegged to the same anchor (Q12) are a clean illustration purely of shared mechanism, not causation between the pair. This matters directly for risk management: `03-var-es-engine/docs/METHODOLOGY.md` assumption A9 warns that "correlations spike toward 1 in crashes — diversification benefit evaporates exactly when needed," i.e. a covariance matrix estimated in calm markets can badly understate how correlated positions become under stress, precisely because a historically-measured correlation is a statistical regularity of the sample period, not a causal law that guarantees to hold going forward.

**Q26. Correlation and cointegration sound similar but answer different questions — what's the distinction, at the level needed before opening the pairs-trading projects?**

Correlation measures how the *changes* (returns) of two series move together; two series can be highly correlated in their day-to-day moves while drifting arbitrarily far apart in level over time (e.g. two assets both trending up, correlated in their daily wiggles, with no relationship between their price levels at all). Cointegration is a statement about the *levels*: two (or more) non-stationary series are cointegrated if some linear combination of them is stationary — i.e. their spread doesn't wander off to infinity but keeps reverting toward some equilibrium. `docs/LEARN.md` describes this exactly as the foundation of the portfolio's pairs-trading projects: "find two instruments whose spread is stationary (cointegrated), trade mean-reversion in that spread... The hard part is not the trading logic — it's proving the spread is actually stationary." A pairs trade is a bet on cointegration (a persistent equilibrium relationship), not merely on correlation (which says nothing about whether a divergence ever reverts) — the full statistical machinery for testing this (e.g. in `python/equity/05-pairs-trading/src/eq_pairs/cointegration.py`) is the subject of a later round.

**Q27. Why does `01-risk-metrics/docs/METHODOLOGY.md` insist on using the Jarque-Bera test rather than just eyeballing a histogram before deciding whether Gaussian VaR is trustworthy?**

Jarque-Bera formalises the normality question into a test statistic, `JB = n/6 * (S^2 + K^2/4)` (sample skewness `S`, excess kurtosis `K`), which is asymptotically chi-squared(2) under the null of normality — it directly targets the same two moments (skew, kurtosis) that Gaussian VaR ignores and that Cornish-Fisher corrects for, giving a quantitative, checkable answer ("reject the normal assumption at p < 0.05") rather than a subjective visual judgment. The documentation is careful to also state the test's own limitation as an assumption with "what breaks if violated": the chi-squared(2) reference is only asymptotically valid, so on a small sample (tens to low hundreds of observations) the test "has poor size... and low power," meaning a rejection or non-rejection on a short series is weaker evidence than the raw p-value suggests.

**Q28. Tie it together: why does this portfolio consistently prefer "ship multiple methods and measure where they disagree" over "pick the theoretically best method and use only that"?**

Every methodology document read for this round makes the same structural move: `01-risk-metrics` ships historical, Gaussian, and Cornish-Fisher VaR side by side specifically "to demonstrate its failure" (the Gaussian method, against historical); `03-var-es-engine` runs historical, parametric, and Monte Carlo VaR together "so the disagreement itself becomes a diagnostic"; `03-black-scholes-replication` ships a Monte Carlo pricer that "shares no code" with the closed form specifically because internal self-consistency checks (put-call parity, Greeks-vs-finite-differences) "share the same blind spots as the code they check." The underlying quant-math reason connects everything in this round: every method here rests on an assumption (normality, i.i.d. returns, a representative sample window, a valid local expansion) that is *sometimes* wrong, and the only way to find out which assumption is failing on a given day's data is to compute an independent estimate that doesn't share that assumption and measure the gap — which is exactly what quantile theory (order statistics), fat-tailed distributions (Student-t), simulation error bounds (`1/sqrt(n)`, bootstrap), and correlation structure (Cholesky, covariance) collectively exist in this portfolio to make precise rather than left as a vague expectation of "the model might be wrong somewhere."


## Round 2 — Options Pricing & Greeks

**Q1. What does the Black-Scholes-Merton formula compute, and what does each term mean?**

For a European call with continuous dividend yield, `python/equity/01-options-pricing/src/eq_options/black_scholes.py::bs_price` implements `C = S e^{-qT} N(d1) - K e^{-rT} N(d2)`. Read left to right: `S e^{-qT}` is the present value of receiving the stock at expiry (spot minus the dividends you forgo by holding the option instead of the stock), weighted by `N(d1)`; `K e^{-rT}` is the present value of paying the strike, weighted by `N(d2)`. `N(d1)` and `N(d2)` are the standard normal CDF evaluated at the two moneyness/vol-adjusted terms defined by `d1_d2`. The formula is the discounted, risk-neutral expected payoff of the option under geometric Brownian motion (GBM) — not a real-world probability statement.

**Q2. What are d1 and d2, and how is N(d1) interpreted?**

`d1_d2` computes `d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))` and `d2 = d1 - sigma sqrt(T)`. `d1` measures how many volatility-standard-deviations the (risk-neutral, drift-adjusted) log-moneyness is from zero. `N(d1)` has two standard readings: it is the option's delta under a continuous dividend yield (`bs_greeks` returns `delta_ = df_q * norm.cdf(d1)` for a call, i.e. `e^{-qT} N(d1)`, not `N(d1)` alone), and it is the sensitivity of the option price to the discounted stock term — the hedge ratio a desk actually trades against.

**Q3. How is N(d2) interpreted, and why isn't it just "the real-world probability of finishing in the money"?**

`N(d2)` is the risk-neutral probability that the call finishes in the money, i.e. that `S_T > K` under the risk-neutral measure used to price the option (the measure under which the discounted stock is a martingale). It is *not* the real-world (physical-measure) probability — that would require the stock's actual expected return, not `r - q`. This distinction is why `bs_price` prices under `r` and `q` only, never a drift/expected-return input, and why the `mc_price` engine in `monte_carlo.py` also simulates under `r - q - sigma^2/2` drift rather than any real-world mean.

**Q4. What is the GBM / constant-volatility assumption, and what breaks when it's violated?**

`METHODOLOGY.md` §2 registers this as A1: the underlying follows lognormal GBM with continuous paths and a single constant `sigma`. Real returns have fat tails, jumps and stochastic vol, so a single BS vol cannot price every strike consistently — the market shows a skew/smile. `VALIDATION.md` §6.1 demonstrates this directly: pricing a synthetic skewed chain with one flat ATM vol underprices the 80-strike put by 5x (skew IV 25.6% vs a mispricing of −80% of value at ATM IV). Short-dated gamma/jump risk from the same assumption shows up again in the `DESK_GUIDE.md` earnings-jump scenario.

**Q5. What does the "frictionless, continuous hedging" assumption cover, and what's the real cost of violating it?**

A2 in `METHODOLOGY.md` assumes hedging can be done continuously with no transaction costs. The engine's discrete-hedging simulator quantifies the resulting irreducible P&L noise: std scales like `sigma * premium * sqrt(pi/4) / sqrt(N)`, tested to shrink as `1/sqrt(N)`; at N=4 rebalances on a 3-month option the std is about $1.61 on a ~$4.42 premium (36% of the premium, `VALIDATION.md` §5, §6.2). With 5bp proportional transaction costs, mean P&L drops from +$0.0014 to −$0.2341 at N=128 rebalances — cost drag that grows with hedging frequency even as the noise it removes shrinks.

**Q6. What does the constant risk-free rate assumption cover, and where does it break?**

A3 assumes a constant, deterministic `r` with no borrow/lend spread. Negative rates are fully supported and tested (`bs_price`, `crr_price` and `mc_price` all accept `r < 0`), but the assumption still breaks in two ways: rho hedging is mis-stated when the real funding curve isn't flat, and discounting error compounds with `T` in a genuinely stochastic-rates world where rates correlate with the equity — a factorisation the deterministic-discounting BS formula can't represent for long-dated options.

**Q7. Why model dividends as a continuous yield q, and what mispricing does that cause in practice?**

A4 trades a single stock's discrete, lumpy dividend payments for a continuous yield `q` for tractability. `VALIDATION.md` §6.3 quantifies the resulting error on a stock paying D=3 on an ex-date at t=0.1y: an ATM call expiring at T=0.08 (before the ex-date) is underpriced by 0.125 (the smeared yield spreads a dividend the option never actually sees), while one expiring at T=0.12 (just after) is overpriced by 1.152 — about 72% too high, because only 36% of the drop has been "applied" by the smeared-yield model by that date. The rule documented there: continuous q is fine for index baskets and long-dated single names; short-dated single-name options around ex-dates need an escrowed-dividend spot adjustment or a discrete-dividend tree, and it also distorts American-call early-exercise timing (see Q25).

**Q8. Why assume European exercise, and what does that assumption cost?**

A5 prices `bs_price`, `black76_price` and `mc_price` under European exercise only. Since single-name listed equity options are actually American, ignoring the early-exercise premium under-marks puts: `VALIDATION.md` §4 gives 0.423350 of early-exercise premium on the 1y ATM reference put (about 7% of its value) at 2000 tree steps. The CRR tree in `binomial.py` is the one engine in the project that handles this correctly by comparing continuation value against intrinsic value at every node.

**Q9. What does assumption A6 (no counterparty/liquidity effects) protect against, and how is it enforced in code?**

A6 assumes frictionless, mid-market pricing with no bid/ask or crossed-quote noise. In practice, quotes can be stale or crossed, and feeding a garbage price into an implied-vol solver would otherwise silently return a garbage vol. `implied_vol` in `black_scholes.py` checks the observed price against the no-arbitrage bounds (`lower` = discounted forward intrinsic at sigma→0, `upper` = `S e^{-qT}`/`K e^{-rT}` at sigma→∞) and raises `ValueError` if the price sits outside them, rather than returning a number — this is documented as the "first-line filter against crossed/stale quotes polluting the surface" in `DESK_GUIDE.md` §1.1.

**Q10. What is A7 ("flat vol term structure"), and why does a desk care?**

A7 assumes a single `sigma` covers the option's entire life, with no term structure across expiries. `METHODOLOGY.md` notes this makes calendar-spread risk invisible to a single-vol model, and theta computed at one fixed vol misattributes P&L as the term structure rolls. `DESK_GUIDE.md` §1.4 states the operational fix: desks mark a full vol surface (per strike *and* expiry) and treat the engine's flat-vol interface as the contract that surface service must fill, not as a claim that vol really is flat.

**Q11. What is A8, and how does the test suite quantify "the discretisation is large enough"?**

A8 assumes tree/MC discretisation parameters (n_steps, n_paths) are large enough that numerical error is negligible relative to a real bid/ask. It's not free: `VALIDATION.md` §2 shows CRR error decaying as O(1/n) with visible odd/even oscillation — at n=10 steps the reference ATM call misprices by 0.194 (~2% of the $9.826 premium), a real mispricing if you quoted off it. Both convergence tables (CRR→BS and MC→BS) are enforced by tests, and `VALIDATION.md` §6.4 explicitly warns never to quote off a coarse tree; use n≥500.

**Q12. What is put-call parity, and how does this portfolio use it as both a pricing identity and a test?**

Put-call parity states `C - P = S e^{-qT} - K e^{-rT}` (equity) or `C - P = S e^{-r_f T} - K e^{-r_d T}` (FX, `garman_kohlhagen.py`). It follows purely from no-arbitrage replication, independent of any distributional assumption, so it's a much stronger check than "does the model match itself." `VALIDATION.md` (equity) enforces it to `< 1e-10` over a 1080-point grid spanning `S, K, T, r ∈ {-1%, 0, 5%}, q, sigma` (`test_put_call_parity_full_grid_1e10`); the FX project checks the two-rate version to 1e-10 over a 7-point grid including negative rates and JPY levels. As a *test*, it catches sign/discounting bugs that a single-price benchmark against a textbook value would miss.

**Q13. What is delta, and what does a real desk actually do with it?**

Delta (`dV/dS`) is `bs_greeks(...).delta`, `df_q * norm.cdf(d1)` for a call. `DESK_GUIDE.md` describes delta as auto-hedged with futures within a band on the intraday risk desk — the hedging simulator's `1/sqrt(N)` law sets how tight that band can be: rebalancing a 3-month ATM book 64x per quarter leaves about $0.44 of P&L noise per option (`VALIDATION.md` §5), while tighter bands buy less noise but pay more of the 5bp transaction-cost drag.

**Q14. What is gamma, and how does a desk manage gamma risk?**

Gamma (`d^2V/dS^2`) is `bs_greeks(...).gamma`. It spikes near expiry when spot sits on the strike: `DESK_GUIDE.md` §3.3 gives gamma at T=1/365, ATM, 20-vol as 0.38 — 20x the 1-year value — while delta flips 0↔1 through the strike ("pinning"). Since discrete-hedging error scales with `gamma * S^2`, a 1/√N rebalancing budget that was fine at 3 months is inadequate on expiry day; desks respond with expiry-day gamma limits and shrunk hedge bands. The earnings-jump scenario (`DESK_GUIDE.md` §3.1) shows the same effect from a gap rather than time decay: a short 1-week ATM straddle collecting $3.87 premium carries gamma 0.164, so an 8% overnight jump costs ½·Γ·8² ≈ $5.26 — more than the premium collected, which is why the desk calls earnings risk "gamma risk, not vega risk."

**Q15. What's the gamma/theta trade-off?**

Theta (`dV/dt`) is the time-value bleed that funds a long-gamma position: for a short option, positive theta (collected as the option decays) is compensation for the negative-gamma exposure to large moves. The pinning scenario (Q14) makes the trade-off concrete — as gamma explodes near expiry, the same position's convexity risk grows faster than the time decay collected to fund it, which is exactly what makes expiry-day risk management distinct from ordinary theta harvesting.

**Q16. What is vega, and how does a desk manage vega exposure?**

Vega (`dV/dsigma`) is `bs_greeks(...).vega`, `S e^{-qT} phi(d1) sqrt(T)` — currency per unit of annualised vol (divide by 100 for "per vol point," per the `greeks.py` docstring). `DESK_GUIDE.md` §1.1 gives a market-making example: a 1y ATM option has vega ≈ 37.8 per unit vol (0.378 per vol point), so quoting a market 0.4 vols wide is about $0.15 of edge on a $9.83 option. §1.2 aggregates vega to book level by expiry bucket, with vanna/volga limits layered on top for the skew book — vega alone doesn't capture how the position moves when the *smile* moves, only a parallel shift.

**Q17. What is rho, and why is it usually a lower-priority Greek?**

Rho (`dV/dr`) is `bs_greeks(...).rho`, `K T e^{-rT} N(d2)` for a call. It's typically small relative to delta/gamma/vega risk for short- and medium-dated equity options because rate moves are small relative to spot and vol moves over the option's life, but assumption A3 (Q6) flags that rho hedging becomes mis-stated whenever the real funding curve isn't flat — the model reports one number under one deterministic rate, not a curve-shift-consistent exposure.

**Q18. Why is Garman-Kohlhagen "Black-Scholes with q = r_f"?**

`fx_options/garman_kohlhagen.py`'s docstring states it directly: holding foreign currency pays the foreign risk-free rate, exactly as a dividend-paying stock pays its yield, so under the domestic risk-neutral measure the spot drifts at `r_d - r_f`. Substituting `q -> r_f` and `r -> r_d` into the BS formula gives `gk_price`: `call = S e^{-r_f T} N(d1) - K e^{-r_d T} N(d2)` with `d1 = [ln(S/K) + (r_d - r_f + sigma^2/2) T] / (sigma sqrt(T))`. The economic content of the substitution: a dividend yield is compensation you give up by holding the derivative instead of the asset; the foreign deposit rate is exactly that same forgone-carry cost when the "asset" is a unit of foreign currency. The equity project's `VALIDATION.md` confirms this isn't just an analogy — `test_black76_equals_bsm_on_model_forward`-style checks in the FX project verify GK against an independent dividend-yield BS implementation to 1e-14.

**Q19. What does the BASE/QUOTE convention mean, and how do r_d and r_f map onto it?**

Per `CONVENTIONS.md` and `garman_kohlhagen.py`'s docstring, FX pairs are quoted BASE/QUOTE — e.g. EURUSD = USD per 1 EUR, so EUR is the base currency and USD is the quote (domestic) currency. `S`, `K` and premiums are all in domestic currency per unit of foreign notional. `r_d` is the quote-currency rate, `r_f` is the base-currency rate; `gk_price`'s call formula discounts the base-currency (foreign) leg at `r_f` and the quote-currency (domestic) leg at `r_d`. This is the same domestic/foreign asymmetry as `q` vs `r` in equity — the currency you're "long" through the option (base) plays the role of the dividend-paying stock.

**Q20. What are the FX project's four delta conventions?**

`fx/01-options-pricing/docs/METHODOLOGY.md` §2.3 documents all four: spot delta `e^{-r_f T} N(d1)` (premium in quote currency, short-dated pairs like EURUSD ≤ 1y); forward delta `N(d1)` (hedging with forwards, long-dated/EM); spot premium-adjusted delta `e^{-r_f T} (K/F) N(d2)` (premium paid in base currency — the USDJPY market standard); and forward premium-adjusted delta `(K/F) N(d2)`. The relations `Delta_f = Delta_s * e^{r_f T}` and `Delta_pa = Delta - premium/S` connect them, and are tested to 1e-14 in the FX project's `VALIDATION.md`.

**Q21. Why does premium adjustment exist, and what numerical wrinkle does it create?**

Premium adjustment exists because a premium *received* in the base currency is itself a position in the underlying — e.g. a USDJPY option premium paid in JPY changes your yen exposure, so the "true" hedge ratio must net that out. The wrinkle: for premium-adjusted call deltas, the map from strike to delta, `(K/F) N(d2)`, is not monotone — it rises then falls, so a target delta can have 0, 1 or 2 solutions. `METHODOLOGY.md` documents the fix: market convention picks the larger-strike (decreasing) branch, and the strike-from-delta solver locates the fold point of `K N(d2)` (where `N(d2) sigma sqrt(T) = n(d2)`) and Brent-solves to its right, raising `ValueError` for unattainable deltas. `VALIDATION.md` §4 in the FX project reproduces the fold explicitly at USDJPY 35%-vol parameters, showing the second root lies left of the solver's returned strike.

**Q22. Why doesn't equity need this delta-convention distinction?**

Equity options have one currency in the trade — the premium, the strike, and the underlying are all quoted in the same currency, so there's no analogue of "premium paid in the other leg's currency" to adjust for, and no separate base/foreign rate pair to create a spot-vs-forward delta split. `CONVENTIONS.md`'s asset-class section states this plainly: equity conventions are just `q`, ACT/365F day count and log-return vol; the delta/forward/premium-adjusted quoting matrix is listed only under the FX conventions, because it's a genuinely FX-specific artifact of trading a price that is itself a ratio of two rate-bearing currencies.

**Q23. How does the CRR binomial tree price options, and what governs u, d and p?**

`binomial.py::crr_price` builds a recombining tree with up-move `u = exp(sigma sqrt(dt))`, down-move `d = 1/u`, and risk-neutral probability `p = (exp((r - q) dt) - d) / (u - d)`. Terminal payoffs are computed in log-space for numerical stability, then backward-induced through the tree by discounted expectation `pu * values[1:] + pd * values[:-1]`; if the American flag is set, every node also takes the max against immediate-exercise intrinsic. If `p` falls outside `(0, 1)` — a sign that `dt` is too coarse relative to `|r - q|` and `sigma` — `crr_price` raises `ValueError` rather than silently returning a mispriced tree.

**Q24. When does American early exercise actually matter, and how does the tree detect it?**

For calls with `q = 0`, Merton's no-early-exercise theorem says American and European calls are worth the same — `VALIDATION.md` §4 treats this as a sharp implementation test, since the tree's early-exercise branch must fire *never* despite being evaluated at every node, and confirms agreement to 1e-10. Early exercise matters whenever there's a carry cost to holding the option instead of the underlying: dividends (`q > 0`) make American calls valuable to exercise early, and any positive net cost of carry (`r - q` sign) makes American puts carry a premium over European. `binomial.py::early_exercise_premium` isolates this by differencing American-minus-European on the *same* tree, which cancels the shared O(1/n) discretisation error — the reference 1y ATM put (q=1%) shows a 0.423350 early-exercise premium at 2000 steps.

**Q25. Where do American and European prices diverge most sharply, and why?**

Two concrete cases from the tests. Equity dividends: `VALIDATION.md` §6.3 notes continuous-q smears a discrete dividend across time, so the American-call exercise decision is distorted too — real early exercise clusters right before the ex-date (exercise when the dividend exceeds remaining time value), which a continuous yield can't represent as a sharp boundary. FX foreign carry: the FX project's `VALIDATION.md` §2 shows a USDJPY 6-month ATMF call with `r_f = 5.25% >> r_d = 0.50%` carries an early-exercise premium of 0.628 JPY per USD — 14.4% of the European price — versus essentially zero (1e-16) for an EURUSD call where `r_f < r_d`. High foreign-currency carry is the FX-specific analogue of a high dividend yield.

**Q26. How is tree-to-closed-form convergence used as a validation technique, not just a sanity check?**

`VALIDATION.md` §2 (equity) tabulates CRR error against the reference BS price (9.826298) across `n_steps` from 10 to 2000, and shows `error x n` holds constant at ≈1.97 — clean O(1/n) convergence, with visible odd/even oscillation at low n (n=25 overshoots from above). The constant `error x n` is itself the evidence: if it weren't roughly constant, that would indicate a bug in the tree rather than expected discretisation error. This is why the project ships convergence tables as generated test output, not hand-picked numbers — `python -m pytest tests -q` enforces monotone error decay along same-parity doublings, and `test_black76_equals_bsm_on_model_forward` provides an analogous closed-form-to-closed-form check.

**Q27. Why include Monte Carlo pricing when it's admittedly the "worst" tool for a vanilla?**

`METHODOLOGY.md` §1.4 is explicit: for a vanilla under GBM, MC's O(n^{-1/2}) convergence is strictly worse than the closed form, and the comparison harness quantifies that honestly rather than hiding it. It earns its place for two reasons: it's the only engine here that generalises to path-dependence, baskets, and any dynamics you can simulate, and an MC-vs-closed-form agreement test (within 3 standard errors) is a powerful bug detector for *both* engines simultaneously. `monte_carlo.py::mc_price` samples the terminal stock exactly (`S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)`, no time-discretisation bias) and combines antithetic variates with a discounted-terminal-stock control variate, cutting standard error by about 2.6x at equal path count (`VALIDATION.md` §3: SE goes from 0.0465 plain to 0.0177 with the control variate alone at 100k paths).

**Q28. What edge cases does this portfolio's test suite actually cover for T→0, vol→0, vol→∞ and rates?**

`black_scholes.py` and `binomial.py` both document explicit, tested limits rather than letting these regimes fall through to NaN: `T == 0` returns intrinsic value in every engine; `sigma == 0` returns discounted intrinsic on the forward. `VALIDATION.md` §6.5 (equity) adds the harder cases: price/Greek sweeps across `S in [1e-3, 1e5]`, `T in [1e-6, 10]`, `sigma in [1e-6, 8]`, `r in [-5%, 10%]` are all finite (RuntimeWarnings are promoted to errors in pytest config, so silent overflow can't creep in), and negative rates are fully supported and cross-model consistent — including the counterintuitive result that American calls can carry an early-exercise premium when `r < 0`. `validate_inputs` rejects NaN and ±Inf in every numeric argument with a `ValueError` across every engine.

**Q29. What is the "vol smile contradiction" failure mode, and why is it not treated as a bug?**

`VALIDATION.md` §6.1 (equity) prices a 3-month synthetic skewed chain both at each strike's own skew-implied vol and at a single flat ATM vol. The 80-strike put is worth 0.1699 at its true 25.6% skew vol but only 0.0344 at the 18.4%-ish ATM vol used flat — an 80% underpricing (a 5x error), while the 110-strike ITM put is off by only about 2%. The document frames this as a demonstration of assumption A1 (Q4), not a defect: the engine deliberately treats vol as a per-quote *input*, taking one vol per strike/expiry from a marked surface, precisely so this contradiction is visible and actionable rather than hidden inside a model that pretends to fit the whole smile.

**Q30. What is the discrete-hedging failure mode, and how does it connect back to assumption A2?**

BS's replication argument assumes continuous, frictionless rebalancing (A2, Q5); the hedging simulator in the equity project shows what survives when that's relaxed. Even hedged at the *correct* volatility, 4x rebalancing over a 3-month option leaves P&L noise of std ≈ $1.61 on a $4.42 premium (36% of the premium) — `VALIDATION.md` §6.2 explicitly calls this "not a bug" but the irreducible risk the replication argument assumes away. The FX project's version of the same experiment (`fx/01-options-pricing/docs/VALIDATION.md` §4.1) sharpens the point: hedging a CHF option priced at peg-regime vol (3%) through the 15-Jan-2015 SNB depeg leaves a P&L std 14.4x the premium collected — proof that no amount of rebalancing frequency compensates for a jump the diffusion model assumed away entirely.


## Round 3 — Implied Volatility & Numerical Solvers

**Q1. Why is there no closed-form way to go from an observed option price back to the volatility that produced it?**

Black-Scholes (and Garman-Kohlhagen) give price as a function of sigma, `price(sigma)`, but that function mixes sigma into the normal CDF terms `N(d1)`, `N(d2)` in a way that cannot be algebraically inverted to `sigma = f(price)`. The map is monotone and smooth, which is exactly what makes it solvable numerically, but there is no elementary closed-form inverse. So implied vol is always found by root-finding: pick sigma, price it, compare to the market price, adjust, repeat.

**Q2. What equation is `implied_vol` actually solving?**

It defines an objective function `f(sigma) = price(sigma) - target_price` (see `objective` in `python/equity/01-options-pricing/src/eq_options/black_scholes.py::implied_vol` and the identical pattern in `python/fx/01-options-pricing/src/fx_options/garman_kohlhagen.py::implied_vol`) and finds the sigma where `f(sigma) = 0`. Because `price(sigma)` is strictly monotone increasing in sigma for a vanilla option, that root is unique — which is what makes both Newton-Raphson and bisection well-posed here.

**Q3. Write out the Newton-Raphson update rule this solver uses.**

`sigma_{n+1} = sigma_n - (price(sigma_n) - target_price) / vega(sigma_n)`, i.e. `sigma -= diff / vega` where `diff = objective(sigma)`. This is standard Newton's method applied to `f(sigma) = price(sigma) - target`, using the fact that vega, `dV/dsigma`, is exactly `f'(sigma)`. You can see this literally at `black_scholes.py` lines ~338-341: `vega = _bs_vega(...)`, `step = diff / vega`, `candidate = sigma - step`.

**Q4. Where does vega come from in this Newton step — is it computed analytically or by finite differences?**

Analytically. Both engines have a closed-form vega helper (`_bs_vega` in the equity module, `_vega` in the FX module) built from the same `d1` used in the price formula: `S * exp(-qT) * phi(d1) * sqrt(T)` for equity, `S * exp(-r_f T) * phi(d1) * sqrt(T)` for FX. Using the analytic derivative avoids the extra price evaluation (and finite-difference noise) a numerical derivative would cost on every Newton step.

**Q5. Why does Newton-Raphson converge fast when it works?**

Newton's method has quadratic local convergence: once sigma is close enough to the root, the number of correct digits roughly doubles each iteration, because the method uses local curvature information (the derivative) rather than just a sign test. For a well-behaved option (moderate moneyness, normal expiry, healthy vega) this typically converges to machine precision in well under ten iterations — far fewer than bisection would need for the same accuracy.

**Q6. What can make vega near-zero, and why does that matter for Newton's method?**

Vega `~ S*exp(-qT)*phi(d1)*sqrt(T)` shrinks toward zero in three regimes: deep in-the-money or deep out-of-the-money (where `|d1|` is large, so `phi(d1)` is tiny), and very short-dated options (the `sqrt(T)` factor and a d1 dominated by the drift term). All three are explicitly called out in `implied_vol`'s docstring ("Known hard regime... vega ~ S sqrt(T) phi(d1) underflows towards zero") and covered by `test_round_trip_deep_itm_low_vega` and `test_round_trip_deep_otm_short_dated` in `tests/test_implied_vol.py`.

**Q7. Mechanically, what happens to a Newton step when vega is near-zero?**

The update is `step = diff / vega`, so a tiny vega in the denominator blows the step up into a huge, unreliable jump — it can overshoot far outside any sensible vol range, or even divide by a value so small it produces `inf`/`nan`. That's precisely why every implementation in this portfolio guards it explicitly: the equity Python code checks `if vega > 1e-14` before trusting the step and falls back to bisection otherwise (`black_scholes.py` implied_vol loop), and the FX code has an identical `if vega < 1e-12: break # flat objective; Newton unreliable -> Brent` guard in `garman_kohlhagen.py`.

**Q8. What is bisection, and why is it used as the fallback here rather than as the primary method?**

Bisection repeatedly halves a bracket `[lo, hi]` known to contain the root (because the objective has opposite signs at the two ends), always converging linearly — one extra correct bit per iteration, guaranteed, with no derivative and no risk of divergence. It's slower than Newton's quadratic convergence, so using it as the primary method would waste iterations in the easy cases; but unlike Newton it can never take a wild step or fail to make progress, so it's the natural fallback exactly when Newton's derivative-based step becomes untrustworthy (near-zero vega, or a step that would leave the maintained bracket).

**Q9. How does this solver decide when to fall back from Newton to bisection?**

Two conditions trigger it, checked every iteration: vega too small to trust (`vega > 1e-14` in equity Python, `vega < 1e-12` in FX Python), or the Newton candidate landing outside the maintained bracket `[lo, hi]`. In the equity code: `if not (lo < candidate < hi): candidate = 0.5 * (lo + hi) # bisection fallback`. This is a "bracketed Newton" design — Newton is used opportunistically for speed, but the bracket is always shrunk around the true root so bisection is available as an unconditional safety net, never a wild guess.

**Q10. What are the no-arbitrage bounds on an option's price, and why must implied vol be checked against them before solving?**

As sigma runs from 0 to infinity, the option price runs monotonically between two limits: the `sigma -> 0` bound (the discounted intrinsic value on the forward — `bs_price(..., sigma=0, ...)` for equity, `df_d * max(phi*(forward-K), 0)` for FX) and the `sigma -> infinity` bound (`S*exp(-qT)` for a call, `K*exp(-rT)` for a put in equity terms; `S*df_f` / `K*df_d` in FX terms). A quoted price outside `(lower, upper)` cannot correspond to *any* real volatility — it's a violation of arbitrage, not a hard-to-find root — so both `implied_vol` implementations check these bounds first and raise `ValueError` immediately (`"is at or below the sigma->0 arbitrage bound"`, `"is at or above the sigma->inf bound"`) rather than handing an unsolvable problem to Newton/bisection.

**Q11. Concretely, what did "the solver exited as soon as residual improvement stalled" mean in the pre-fix code?**

The Newton/bisection loop's stopping rule looked only at the *price* residual — `if abs(diff) < tol: break` (or, in the older FX code shape, `if abs(diff) < 1e-14: return sigma`) — and returned the current sigma the moment that residual got small enough, without any further check on how far sigma itself might still be from the true root. In a flat-vega region that price residual can look converged while sigma is still off by whole vol points, because the tiny price gap maps through a near-zero vega to a large sigma gap.

**Q12. Why is "the solver returned an imprecise answer" a fundamentally worse failure than "the solver raised an error"?**

An error is loud: the caller's code stops, a test fails, someone investigates. A silently imprecise return value is the opposite — it looks like success (a plausible-looking float, no exception, no warning) and gets used downstream in a Greeks calculation, a P&L attribution, or a vol surface fit, corrupting everything built on top of it with no signal that anything went wrong. `docs/ARCHITECTURE.md`'s "Design invariants" section makes the same point about the related NaN-guard defect class: a check that looks like it protects you but silently passes bad data through is more dangerous than no check, because it manufactures false confidence.

**Q13. Concretely, by how much did the equity solver's early exit understate its own error, per the portfolio's own measurement?**

Per `docs/VALIDATION.md` in the equity project (section 6.5), exiting on the price-residual check alone "previously understated the error by ~300x (1.8e-2 vs 2.4e-5 on the S=K=100, T=25y, sigma=300% put)" — i.e. the solver believed it was accurate to about 2e-5 in vol when the true error was closer to 2e-2, nearly two vol points, in exactly the long-dated/high-vol flat-vega corner from Q6.

**Q14. Why was the FX engines' version of this bug worse than the equity engines' version?**

In the equity engines, the plateau bug caused a *quantitative* error — the returned sigma was in the right neighborhood but insufficiently precise (the ~300x example above). In the FX engines, per the top-level `README.md`, the bug "could saturate the solver at the model's no-arbitrage upper bound and return a vol that did not actually reprice the input" — a *qualitative* failure. `garman_kohlhagen.py`'s docstring for `implied_vol` spells out the mechanism: deep ITM + long-dated + high vol drives `N(d1)`/`N(d2)` to saturate to 0/1 in double precision, so `gk_price(sigma)` is bit-identical to the `sigma -> inf` bound for every sigma across a whole plateau, not just at the true root. The old code could accept an arbitrary point in that plateau (the comment gives the example "4.0 instead of the true 3.0, a one-third relative error") with no signal to the caller that the returned vol doesn't actually round-trip back to the input price at all.

**Q15. How does the current FX `implied_vol` handle that plateau case instead of silently returning a point from it?**

It detects the plateau during Brent-bracket expansion: while doubling the upper end of the bracket looking for a sign change, if `objective(hi)` lands exactly on zero without the objective ever having gone strictly positive, that's the signature of a flat plateau rather than a genuine bracket. The code then raises `ValueError("price ... is within double-precision resolution of the sigma->inf bound ...; implied volatility is unrecoverably large")` instead of accepting an arbitrary `hi` as the answer — see `garman_kohlhagen.py` lines ~237-257, and the regression test `TestImpliedVol.test_long_dated_deep_itm_high_vol_flat_plateau_raises` in `tests/test_garman_kohlhagen.py`.

**Q16. What is the actual fix applied to all six engines?**

Per the top-level `README.md`: "Fixed in all six engines by adding a bracket-bisection refinement stage that always runs to convergence rather than exiting on stalled Newton progress." Concretely, every `implied_vol` now always falls through, after the Newton/bisection loop, to a final refinement pass on the maintained `[lo, hi]` bracket run to full precision — `brentq(objective, lo, hi, xtol=1e-16, rtol=8.9e-16, maxiter=200)` in equity Python, `brentq(objective, lo, hi, xtol=tol, maxiter=200)` in FX Python, and the analogous bisection-to-double-precision-width step in the C++ (`implied_vol.hpp`/`.cpp`) and Rust (`implied_vol.rs`) engines. The key design change is that this final stage is unconditional — it always runs — rather than being skipped whenever the earlier price-residual check happened to look satisfied.

**Q17. Why is "always fall through to bracket refinement" a robust fix rather than just tightening `tol`?**

Because the bug wasn't that `tol` was set to the wrong number — it's that the stopping *rule itself* (price residual only) is unreliable in flat-vega regions regardless of how tight `tol` is, since a tiny price residual can correspond to an arbitrarily large sigma residual there (the whole point of Q13's 300x example). Tightening `tol` would only shrink the price residual further, not fix the underlying mismatch between the residual being checked and the quantity (sigma) that actually needs to be accurate. The fix instead changes what always happens at the end of the loop — a bracket-based refinement that operates directly in sigma-space rather than trusting the price-residual proxy.

**Q18. Does the fitted-exponent Newton step in the loop actually get thrown away by the fix, or is it still doing useful work?**

It's still doing useful work — Newton is retained as the fast path for the well-conditioned cases (most moneyness/expiry combinations), and the maintained bracket it narrows along the way is exactly what the final Brent/bisection refinement operates on. The fix doesn't replace Newton with pure bisection; it removes the early-exit that let a stalled Newton loop's *last* price-residual check stand in for real convergence, and instead always spends the final stage refining the bracket to double-precision width. So well-behaved cases still converge in a handful of Newton steps plus a cheap refinement; only the hard corners actually rely on the refinement doing real work.

**Q19. Why didn't this fix move any golden-vector reference value?**

Per `docs/ARCHITECTURE.md`'s "Design invariants" section: "No golden-pinned value ever changes silently... a numerics fix that also happened to change a pinned reference value would be indistinguishable, from the outside, from a regression, so every fix in this portfolio's history was verified against that constraint before being accepted." This particular fix is a robustness fix, not a formula fix — it doesn't change what `bs_price`/`gk_price` compute, and for the well-conditioned inputs the golden vectors are drawn from, Newton was already converging to full precision before the extra refinement stage even mattered. The refinement only changes the *answer* in the flat-vega tail of the input domain (deep ITM/OTM, long-dated, high vol) that the golden vectors don't probe, so every existing pinned value stayed bit-identical while the failure-mode tests (`test_round_trip_long_dated_high_vol_flat_vega`, `test_long_dated_deep_itm_high_vol_flat_plateau_raises`) newly pass.

**Q20. The top-level README frames this as one of two defect classes found in a "second, deeper review pass." What distinguishes that pass from the first one?**

Per `README.md`, the first review pass was about *input validation* — catching NaN/Inf and out-of-range arguments at the boundary of public entry points. The second pass, described as "benchmarked against the standard a top-tier options/risk desk would hold a new pricing library to before it touches P&L," went past validation "into the solvers' own numerics" — i.e. it assumed inputs were valid and asked instead whether the *algorithm* behaves correctly across its entire input domain, including edge regimes where the math itself becomes ill-conditioned. The implied-vol plateau bug and the Cornish-Fisher domain-check grid-resolution bug (found in the VaR/ES engines) are the two defect classes that pass found.

**Q21. Why do all six options-pricing engines share this exact bug rather than it being isolated to one implementation?**

Because the C++ and Rust engines are deliberately built to mirror the Python reference implementation's semantics exactly (`CONVENTIONS.md`: "Engines mirror the Python reference implementations and are cross-validated against them"), and the equity and FX projects independently implement the same Newton-with-bisection-fallback algorithm design (not shared code, per `ARCHITECTURE.md`'s "Equity and FX share conventions documentation, not code"). A structural weakness in the *algorithm design* — an early-exit stopping rule that only checks the price residual — gets faithfully reproduced across all six independent implementations precisely because each one is a correct translation of the same (flawed) design, which is exactly what cross-language mirroring is supposed to guarantee and exactly why the fix had to be applied six times, once per engine, rather than once.

**Q22. Confirm: does `implied_vol` in the C++ and Rust equity engines document the same flat-vega hard regime as the Python reference?**

Yes. `cpp/equity-options-engine/include/eqopt/implied_vol.hpp` states almost verbatim the same warning as the Python docstring: "very long-dated and very high vol... pushes |d1|, |d2| large enough that vega ~ S sqrt(T) phi(d1) underflows towards zero... recovered vol is only accurate to the 1e-4-1e-3 level in that corner... a property of the inverse problem itself, not a fixable solver bug." `rust/equity-options-engine/src/implied_vol.rs` documents the same bracketed-Newton-with-bisection design and notes in its `Errors` section that once bracketed "the solver always terminates (Newton with a bisection fallback and a final bracket-bisection refinement)" — i.e. the Rust doc comment explicitly names the same "final bracket-bisection refinement" stage that the Python fix added.

**Q23. What kind of test would have caught this bug sooner than a fixed-tolerance pass/fail test did?**

A test that checks *convergence behavior as the tolerance parameter tightens* — running `implied_vol` at a sequence of shrinking `tol` values (say, `1e-2, 1e-4, 1e-6, 1e-8`) in a known hard corner and asserting the recovered-sigma error shrinks roughly in step, rather than a single test that just asserts `implied_vol(...) == pytest.approx(sigma, abs=1e-8)` at one fixed tolerance. A fixed-tolerance test only tells you the solver got *close enough this time*; it can't distinguish "the algorithm genuinely converges as you ask for more precision" from "the algorithm silently plateaus below some accuracy ceiling no matter what tolerance you request" — which is exactly the failure mode this bug had (Q13's 300x-understated-error case would have passed a `1e-2`-tolerance test happily).

**Q24. Does this portfolio have an existing test pattern that matches that "convergence as a parameter tightens" idea, even if not applied to implied vol's tolerance directly?**

Yes — a fitted-exponent convergence-rate test, but for the CRR binomial tree's step count rather than the implied-vol solver's tolerance: `test_convergence_rate_fitted_exponent` in `python/fx/01-options-pricing/tests/test_binomial.py` and `convergence_rate_fitted_exponent_matches_theoretical_order_one` in `rust/equity-options-engine/tests/binomial.rs` fit an empirical convergence order to the error-vs-step-count curve and assert it matches the theoretical O(1/n) rate, rather than just checking the error is small at one fixed step count. That's the same design principle Q23 describes — checking the *rate* of convergence as a control parameter varies, not a single pass/fail snapshot — applied elsewhere in this portfolio; the implied-vol regression coverage that came out of this specific fix instead takes the form of targeted hard-corner round-trip tests (`test_round_trip_long_dated_high_vol_flat_vega`, `test_long_dated_deep_itm_high_vol_flat_plateau_raises`) rather than a tolerance-sweep test of that same shape.

**Q25. If you were adding the tolerance-sweep test from Q23 to this codebase, what would the assertion actually check?**

Something like: for a fixed hard-corner input (e.g. the S=K=100, T=25y, sigma=300% put from `test_round_trip_long_dated_high_vol_flat_vega`), call `implied_vol` with `tol` values spanning several orders of magnitude, record `abs(recovered_sigma - true_sigma)` at each, and assert that error is non-increasing (or decreases by some minimum factor) as `tol` shrinks — rather than asserting a single absolute error bound at one `tol`. A solver with the pre-fix early-exit bug would fail such a test because its recovered-sigma error would plateau (stop improving) below some `tol` value even as smaller `tol` was requested, exposing exactly the "residual improvement stalled" defect at the API level instead of requiring a reviewer to read the loop's stopping condition.

**Q26. Summarize, in one line, the general lesson this bug teaches about validating numerical solvers.**

A stopping rule that checks the wrong quantity — here, the price residual, when the thing that actually needs to be accurate is sigma — can look converged while being badly wrong, so both the solver's exit condition and its test suite need to reason about the variable of interest directly (or its convergence rate under a tightening control parameter), not just a proxy residual that happens to correlate with it everywhere except the exact regime you most need to trust it in.


## Round 4 — Volatility Modeling & Forecasting

**Q1. Black–Scholes assumes constant volatility. Why does this portfolio bother modeling volatility as time-varying at all?**

Because the constant-vol assumption is empirically false in a specific, exploitable way: daily returns exhibit volatility clustering — a large move today makes a large move tomorrow more likely, and calm periods cluster too. `eq_vol/evaluation.py` builds this into its diagnostics directly: `arch_lm_test` (Engle's ARCH-LM) applied to raw returns is described as "the standard pre-test for whether a GARCH-type model is needed at all," and `ljung_box_squared` checks whether squared standardised residuals are still serially correlated after fitting. If returns were iid with constant variance, both tests would come back clean on the raw series and there would be nothing for a GARCH model to explain.

**Q2. What does "volatility clustering" mean mechanically, in terms of what a GARCH-type recursion does with it?**

It means today's shock feeds directly into tomorrow's conditional variance. In GARCH(1,1), `sigma2_t = omega + alpha*r_{t-1}^2 + beta*sigma2_{t-1}` (`eq_vol/garch.py`): a large `r_{t-1}^2` pushes `sigma2_t` up immediately via the `alpha` term, and `beta*sigma2_{t-1}` carries the elevated level forward. EGARCH and GJR-GARCH add that the *sign* of the shock matters too (leverage effect), but all three models encode the same clustering mechanism — variance is autocorrelated even though returns themselves are roughly uncorrelated.

**Q3. Exactly which volatility models does this project implement — not in theory, but in the actual `src/` modules?**

`python/equity/02-volatility-modeling/src/eq_vol/` has five: range/realized historical estimators (`historical.py` — close-to-close, Parkinson, Garman–Klass, Rogers–Satchell), RiskMetrics EWMA (`ewma.py`), GARCH(1,1) (`garch.py`), EGARCH(1,1) (`egarch.py`), and GJR-GARCH(1,1) (`gjr.py`), tied together by `forecasting.py` and scored by `evaluation.py`. The FX sibling (`fx_vol/`) has the same five plus a GARCH-X variant with an exogenous event-dummy term (`omega + gamma_x'x_t + alpha r^2 + beta sigma2`) and `vol_premium.py` for comparing model/realized vol against implied.

**Q4. Why GARCH-family models rather than a full stochastic-volatility model like Heston?**

METHODOLOGY.md's "Alternative A" is explicit: SV models are the theoretically "right" description (variance as its own latent process) but the likelihood requires filtering — particle filters, MCMC, or Kalman-type quasi-ML — an order of magnitude more machinery with harder diagnostics and governance. For one-step-ahead vol forecasting from daily returns, it cites Hansen–Lunde (2005), "Does anything beat a GARCH(1,1)?", which finds essentially no forecast gain from SV over GARCH-family models. GARCH instead gives an exact, closed-form likelihood — auditable to machine precision, which the project exploits directly in its `arch`-package cross-validation.

**Q5. Why not realized-volatility / HAR models (Corsi 2009), which the docs admit often beat daily GARCH?**

METHODOLOGY.md's "Alternative B" concedes that HAR *would* be the preferred production choice on liquid index futures if clean intraday data were available, because realized variance typically beats daily GARCH out of sample. The blocker is data budget: HAR needs a reliable tick/5-minute pipeline with microstructure-noise corrections, doesn't produce a full conditional distribution without more assumptions, and degrades to nothing on names without intraday history. This project's scope is daily OHLC-and-close data, so the range estimators in `historical.py` are described as "the poor man's realized vol appropriate to that data budget" — and `forecasting.py`'s harness is built to accept an RV proxy unchanged if one becomes available.

**Q6. Why not just use option-implied volatility as the forecast?**

"Alternative C" in METHODOLOGY.md: implied vol is forward-looking and hard to beat for direction where liquid options exist, but it embeds a time-varying variance risk premium — it is a biased forecast by construction — is unavailable for most single names and custom baskets, and can't be used to mark the very options it comes from without circularity. The project's stance is that desks use implied and model vol *together*; this package supplies the model leg, and the FX side goes further with `fx_vol/vol_premium.py`, which explicitly computes `VRP = IV - RV` as a tradeable signal rather than treating implied vol as ground truth.

**Q7. What is the actual difference between realized volatility and implied volatility, and why do they diverge?**

Realized volatility is computed backward-looking from historical returns (or the historical range) — `eq_vol.historical.realized_vol` and friends. Implied volatility is backed out of observed option prices via inverting Black–Scholes/Garman–Kohlhagen (Round 2's territory) and is inherently forward-looking. They diverge because implied vol prices in a variance risk premium — compensation option sellers demand for jump/crash risk — so implied systematically runs above subsequently realized vol on average; `fx_vol/vol_premium.py`'s `vol_risk_premium` function quantifies exactly this gap, and DESK_GUIDE.md notes a *negative* premium (realized exceeding implied) as a crisis signal to stand down a short-vol program.

**Q8. What does EWMA actually buy you, and what is its key structural limitation?**

`eq_vol/ewma.py`: `sigma2_t = lambda*sigma2_{t-1} + (1-lambda)*r_{t-1}^2` with `lambda = 0.94` (11.2-day half-life). It's IGARCH(1,1) with `omega = 0`, so persistence `alpha+beta` is exactly 1 — no parameters to estimate and it's never badly mis-calibrated, but there is no long-run level to revert to, so the k-step forecast is flat at the 1-step value (`ewma_forecast`'s term structure is structurally flat). DESK_GUIDE.md is blunt about the consequence: this flat structure "cannot price calendar effects at all," which is exactly why a GARCH-family model is layered on top of it rather than used alone.

**Q9. What does GARCH(1,1) add over EWMA, and what does its long-run/unconditional variance parameter mean?**

GARCH(1,1) (`eq_vol/garch.py`) adds an intercept `omega > 0` and requires `alpha + beta < 1`, which gives a genuine unconditional variance `omega / (1 - alpha - beta)` — the level volatility reverts to over time — instead of EWMA's flat-forever forecast. The k-step forecast decays geometrically toward that level at rate `alpha+beta` (`forecast_garch` in `forecasting.py`), which is why GJR after a sell-off produces an upward-sloping term structure "exactly the shape the listed market shows" (DESK_GUIDE.md), something EWMA structurally cannot do.

**Q10. Mean reversion in volatility — what does it mean physically, and how is "persistence" and "half-life" defined in this codebase?**

Persistence is `alpha + beta` for GARCH, `alpha + gamma/2 + beta` for GJR (`gjr_persistence` in `gjr.py`) — it's the rate at which a variance shock decays back toward the unconditional level. `eq_vol/garch.py`'s docstring gives the half-life of a shock as `ln(0.5) / ln(alpha+beta)`; DESK_GUIDE.md reports fitted half-lives of roughly 14–19 days on realistic parameters and uses this (`extra["halflife_days"]`) to tell a risk manager how long an elevated-vol episode, and an associated de-risking flag, is expected to persist.

**Q11. What does EGARCH add that plain GARCH can't do, and what's the cost?**

EGARCH(1,1) (`eq_vol/egarch.py`) models the *log* of variance: `ln sigma2_t = omega + beta*ln sigma2_{t-1} + alpha(|z_{t-1}| - E|z|) + gamma*z_{t-1}`. Because the recursion lives in log-space, `sigma2` is positive for *any* real parameter values (only `|beta| < 1` is needed for stationarity), and `gamma < 0` captures the leverage effect — negative shocks raising vol more than positive ones of the same size. The cost is that multi-step forecasts have no clean closed form (they need `E[exp(alpha|z| + gamma*z)]` across steps), so `forecast_egarch` in `forecasting.py` falls back to seeded Monte Carlo simulation rather than an analytic recursion.

**Q12. GJR-GARCH also captures leverage — how does its mechanism differ from EGARCH's, and why does the sign convention matter?**

GJR-GARCH (`eq_vol/gjr.py`): `sigma2_t = omega + (alpha + gamma*1[r_{t-1}<0]) r_{t-1}^2 + beta*sigma2_{t-1}` — it switches on an *extra* loading `gamma` only when the prior return was negative, so `gamma > 0` is the leverage signal (opposite sign convention from EGARCH's `gamma < 0`). METHODOLOGY.md flags that both conventions are explicitly stated and unit-tested so nobody confuses them; VALIDATION.md's simulated recovery shows the sign check works both ways — GJR's `gamma` t-stat exceeds +3 on leveraged data and EGARCH's is below -3, while both sit within ±0.02 of zero on symmetric data.

**Q13. How does the historical/realized estimator suite differ from just squaring daily returns, and why does it matter?**

`eq_vol/historical.py` implements four range-based one-day variance estimators beyond plain close-to-close `r_t^2`: Parkinson (~4.9x more efficient), Garman–Klass (~7.4x), and Rogers–Satchell (~6x, and uniquely drift-robust). "Efficiency" means lower sampling variance for the same data — METHODOLOGY.md notes a 21-day Parkinson estimate is roughly as precise as a ~100-day close-to-close one. The price is stronger assumptions (continuous monitoring, no overnight gaps, no microstructure noise); discrete trading biases the observed range down and bid–ask bounce biases it up, both demonstrated in `tests/test_historical.py`.

**Q14. How is a GARCH-type forecast actually validated in this codebase — what's the real backtest?**

`examples/run_pipeline.py` runs an out-of-sample forecast race: true model GJR with leverage (persistence 0.97), 3,000 training days, 500 out-of-sample 1-step forecasts, parameters refit every 25 days, scored with QLIKE against squared returns via `forecast_race_table` in `evaluation.py`. It isn't graded against a raw error metric alone — every model is also compared to a naive benchmark (EWMA(0.94) and a rolling 21-day historical estimate) using the Diebold–Mariano test with Harvey small-sample correction, computed by `diebold_mariano`.

**Q15. What did that out-of-sample race actually find (VALIDATION.md §4), and what does it mean that GJR beat GARCH?**

Ranking by QLIKE: GJR (-7.6027) > EGARCH (-7.5785) > GARCH (-7.5638) > EWMA(0.94) (-7.4996) > rolling 21-day historical (-7.4963), against an oracle QLIKE of -7.6067 (the true conditional variance of the data-generating process). GJR's DM stat vs GARCH is -2.91 (p=0.004) — a statistically significant win — and it captures about 94% of the achievable QLIKE gap between GARCH and the oracle, which is expected because the simulated truth had genuine leverage (gamma=0.12) that only GJR and EGARCH can represent.

**Q16. Why is QLIKE the headline loss function here rather than plain MSE, and is that an arbitrary choice?**

No — it follows Patton (2011), cited directly in `evaluation.py`'s module docstring: since true conditional variance is unobservable and forecasts must be scored against a noisy proxy (squared returns), only a specific class of losses is "robust," meaning the proxy-based ranking equals the ranking under the true variance. QLIKE (`ln f + p/f`) and MSE are the two robust members, but QLIKE is preferred as the headline metric because MSE's expectation is dominated by proxy noise in high-variance episodes, making rankings noisier — and QLIKE also penalizes under-prediction of variance more than over-prediction, which VALIDATION.md and DESK_GUIDE.md both flag as the economically correct asymmetry (under-forecasting vol is the expensive error).

**Q17. The project independently re-implements GARCH/EGARCH/GJR from scratch. How does it know the implementation is correct?**

`tests/test_arch_crosscheck.py` benchmarks against Kevin Sheppard's `arch` package purely as an oracle. On 5,000 simulated GARCH observations, alpha/beta/omega agree to `arch` to within 1e-6–4e-6, and — the strongest statement in VALIDATION.md — evaluating this project's own log-likelihood at `arch`'s fitted parameters agrees with `arch`'s own log-likelihood to 1.8e-12, meaning the recursion and density are "line-for-line equivalent" and the residual parameter differences are pure optimizer tolerance, not implementation bugs.

**Q18. What exactly breaks in a volatility forecast during a genuine regime change, according to this project's own crisis test?**

VALIDATION.md's failure mode F1, tested in `test_edge_cases.py::TestCrisisRegimeJump`, simulates a COVID-March-2020-scale jump (true vol 15% to 75% annualized for 60 days, then settling at 30%). EWMA(0.94) adapts but is lagged by construction — it read 13.3% pre-break and only reached 65.4% thirty days *into* the crisis, meaning 1-day 99% VaR computed from it would have been breached repeatedly in the first week. GARCH fitted across the break estimates persistence 0.983 (vs ~0.95 on break-free data) — the break masquerades as near-IGARCH persistence, dragging the estimated long-run variance far above either true regime, so its forecast term structure mean-reverts to a level that means nothing.

**Q19. Why does a structural break make GARCH's estimated persistence spike, and what's the practical tell?**

Because GARCH has no mechanism to distinguish "variance is genuinely near a unit root" from "there was one level shift in the middle of my sample" — both produce highly autocorrelated squared returns, and the MLE fits persistence toward 1 either way. The practical tell in this codebase is `unconditional_variance()` — the library refuses to compute it (raises `ValueError`) rather than return a near-infinite or negative number once `alpha+beta` hits the 0.9999 transform ceiling, and METHODOLOGY.md/VALIDATION.md both flag persistence pinned at that ceiling as "a red flag surfaced by the parameter table, not hidden" and "often symptom of F1" (the structural-break failure mode).

**Q20. What's the desk's actual mitigation once a regime break is confirmed, rather than just a documented limitation?**

DESK_GUIDE.md's workflow step 4 is explicit: after a suspected regime break, shorten the estimation window (`scheme="rolling"` in `rolling_one_step_forecasts`) or refit from the break date, and watch the EWMA-vs-GARCH gap — EWMA repricing much faster than the fitted GARCH is called "the classic break signature." VALIDATION.md's crisis case study quantifies the window trade-off directly: 30 days into the simulated crisis, a 10-day rolling window reads 75.1% (correct) while a 250-day window reads 35.9% — less than half the truth — so shortening the window trades noise for speed of adaptation.

**Q21. Beyond regime breaks, what other failure modes does VALIDATION.md document, and are they just described or also enforced in code?**

Both. F2 (IGARCH boundary): as persistence approaches 1, unconditional variance explodes, so `unconditional_variance()` raises instead of returning garbage. F3 (fat tails): Gaussian QMLE stays consistent for alpha/beta even under Student-t innovations, but Gaussian VaR quantiles understate risk by roughly 10% when the truth is t(8) — mitigated by `dist="t"`, with `nu` recovered within 0.2 of truth on 20k simulated observations. F5 (short samples): fitters refuse fewer than 100 observations with an informative error rather than returning a badly-identified fit. F6 (optimizer failure): the library never fails silently — a non-converging fit raises `ConvergenceError` or is flagged `converged=False`, tested by deliberately forcing a failure.

**Q22. How do these volatility forecasts actually get used downstream, per DESK_GUIDE.md — not hypothetically, concretely?**

Four consumers are named explicitly. Option pricing/marking: `term_structure()`'s model vol is compared against implied vol, with `implied - forecast` used as the carry signal for variance-risk-premium strategies, and the model mark serves as a fallback for illiquid expiries. VaR/ES: 1-day 99% VaR is computed as `sigma_{t+1|t} * q_{0.01}` from `rolling_one_step_forecasts` combined with the fitted innovation quantile, using `dist="t"` because Gaussian tails understate 1% VaR by about 10% under t(8) returns. Vol targeting: a book scales exposure by `target_vol / forecast_vol`, which is exactly why QLIKE (which penalizes under-forecasting more) is the right evaluation metric — under-forecasting means being over-levered into a storm. Risk limits: persistence and half-life (`extra["halflife_days"]`) tell a risk manager how long a de-risking flag should stay on after a shock.

**Q23. Why does DESK_GUIDE.md insist on refitting parameters only every 5–25 days rather than every day, if markets move daily?**

Because the variance *recursion* is updated every day using the current parameters (`rolling_one_step_forecasts` does exactly this — "recursion daily, refit sparse"), while the parameters themselves (`omega`, `alpha`, `beta`, `gamma`) move slowly; DESK_GUIDE.md states daily refitting "adds noise and mark instability without forecast gain," and the headline pipeline race itself uses a 25-day refit cadence. The guide adds explicit alerting triggers on top of that cadence: a parameter jump greater than 3 standard errors versus the previous fit, persistence exceeding 0.99, `nu` below 5, or a `ConvergenceError`.

**Q24. How does this project frame model governance — what stops a bad fit from silently entering marks?**

DESK_GUIDE.md lays out a benchmark chain, never a single model in isolation: rolling historical (sanity check) → EWMA(0.94) (regulator-recognised, parameter-free) → GARCH (mean reversion) → GJR/EGARCH (asymmetry), with each escalation required to be justified by a DM test on QLIKE out of sample — exactly the committed pipeline in `run_pipeline.py`. Independently, the from-scratch implementation is cross-validated against the `arch` package to machine precision (the "re-implement and reconcile" procedure model-validation teams use on front-office pricers), every fit result carries parameters, standard errors, log-likelihood, AIC/BIC and a convergence flag (`VolatilityFitResult`), and any code change that shifts a fitted parameter beyond optimizer noise on the synthetic-truth regression tests fails CI.

**Q25. What's genuinely different about the FX volatility project versus the equity one — not just "same models, different asset class"?**

Two things are FX-specific and directly tested. First, quote direction is a modelling choice: inverting BASE/QUOTE negates log returns, leaving volatility invariant but *flipping the sign of asymmetry* — METHODOLOGY.md's pipeline shows GJR finding gamma=0 on USDMXN but gamma=+0.123±0.040 on the inverted MXNUSD, so the desk rule is to fit the quote direction actually traded and sanity-check gamma's sign against the underlying economics. Second, leverage itself is weak for G10 and strong for EM, because FX has no debt-equity channel — the fitted EGARCH gamma is about 0.007 on a G10-style pair versus about 0.046 on an EM-style pair — and `fx_vol` adds a GARCH-X variant that puts known scheduled events (FOMC/ECB/BoJ dummies known at t-1) directly into the variance equation, recovering the event-dummy coefficient within 3% of its true simulated value.


## Round 5 — Volatility Surface & Stochastic Vol

**Q1. Round 2 covers pricing and Greeks under a single implied volatility per option. Why isn't one number enough to describe a real options market?**

Because the market does not quote one vol per underlying — it quotes a different implied vol for every strike and every expiry. Plot implied vol against strike at fixed expiry and you get the "smile" or "skew" (equities skew negatively: downside puts trade at higher implied vol than upside calls, reflecting crash risk and leverage effects); plot it against expiry at fixed moneyness and you get the term structure. A single Black-Scholes vol can reprice exactly one option, but a desk quotes and risk-manages hundreds of strikes and expiries simultaneously off the same underlying, so it needs the whole two-dimensional object, not one point on it.

**Q2. What, precisely, is a volatility surface?**

It's implied volatility expressed as a function of two coordinates spanning strike/moneyness and expiry — `sigma_imp(K, T)` in the equity convention used by `eq_surface.surface.VolSurface`, where `K` is absolute strike and `T` is time to expiry in years. FX quotes the same object differently: `fx_surface.surface`'s `vol(Δ, T)` is implied vol as a function of *delta* (a probability-like moneyness coordinate) and tenor, because FX brokers quote in delta space rather than strike space (see Q19). Both are the same mathematical object — a 2-D vol function — expressed in the coordinate system each market actually trades.

**Q3. How does this portfolio actually build the equity surface? Don't assume a textbook method — cite the real construction.**

`eq_surface.surface.VolSurface` fits one raw-SVI slice per expiry pillar (`smile.fit_svi`), samples each slice's *total variance* `w(k) = svi_total_variance(k, p)` on a common log-moneyness grid, and then interpolates **total variance linearly in T** between pillars (`VolSurface.total_variance`), not vol. Outside the pillar range it applies a documented extrapolation policy: flat vol below the first pillar (`w` scales linearly to zero as `T → 0`) and a slope-floored linear continuation of `w` beyond the last pillar. Querying `vol(K, T)` converts the strike to forward log-moneyness `k = ln(K / F(T))` first, then calls `vol_k`.

**Q4. Why total variance, and specifically why not just linearly interpolate implied vol itself across expiries?**

Absence of calendar-spread arbitrage is equivalent (at fixed forward log-moneyness) to total variance `w(k, T) = sigma_imp(k,T)^2 * T` being non-decreasing in `T`. Linear interpolation of a monotone quantity in `T` stays monotone, so pillars that are individually calendar-free interpolate to a calendar-free surface. The `surface.py` module docstring makes the counterexample explicit: linear interpolation *of vol* has no such property — a high-vol short pillar sitting next to a low-vol long pillar can interpolate to *decreasing* total variance between the two, which is negative forward variance, i.e. manufactured arbitrage that wasn't present in either individual slice.

**Q5. What is raw SVI, and what does each of its five parameters control?**

`smile.SVIParams` parameterises total implied variance in forward log-moneyness `k = ln(K/F)` as `w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))`. `a` sets the overall variance level, `b >= 0` sets the wing slope, `rho` in `(-1, 1)` sets skew/asymmetry, `m` translates the smile's minimum horizontally, and `sigma > 0` controls how rounded the vertex is. `fit_svi` fits these five parameters per expiry by constrained least squares with multiple random restarts (`n_restarts`, default 8) against a data-driven initial guess, and requires at least 5 valid quotes per expiry since a 5-parameter model needs at least that many points to be identified.

**Q6. The METHODOLOGY.md table compares SVI against a polynomial-in-delta baseline. What does that comparison actually show, and where does the baseline live in the code?**

The naive baseline is `smile.fit_quadratic_delta` — implied vol modeled as a quadratic in BS forward call delta, with no no-arbitrage machinery. METHODOLOGY.md reports it fits equity smiles 10-100x worse than SVI (0.50-0.75 vol points versus 0.0006-0.04 for SVI on the same slices) and its odd-power polynomial wings either explode or collapse outside the fitted range, whereas SVI's wings are linear in total variance by construction — consistent with Lee's moment bounds. The baseline exists specifically to make the "why SVI" argument concrete rather than asserted.

**Q7. Why can Black-Scholes never produce a smile on its own, no matter how you calibrate it?**

Black-Scholes assumes volatility is a single constant parameter, the same number for every strike and every expiry on a given underlying. Inverting a market price for "the" implied vol under that model necessarily returns one number per (K, T) pair — there is no internal mechanism by which the model could generate different vols at different strikes, because the model has exactly one vol parameter to begin with. Any smile you observe when you invert real market prices under Black-Scholes is therefore evidence the *model* is wrong for those prices, not a feature Black-Scholes can reproduce; you either interpolate the observed smile empirically (SVI) or move to a model whose dynamics generate a smile on their own (Heston).

**Q8. What does Heston add to Black-Scholes that lets it generate a smile endogenously?**

Heston makes variance itself a second stochastic process rather than a constant: `dv = kappa(theta - v) dt + xi*sqrt(v) dW_v`, correlated with the spot's Brownian motion `dW_S` via `rho`. `heston.py`'s module docstring gives the full risk-neutral SDE pair. That correlation is what buys the smile "for free": when `rho < 0` (the typical equity fit — the calibration recovery table in VALIDATION.md finds `rho = -0.65`), falling spot and rising variance move together, which fattens the downside and produces negative skew without any per-strike parameter — the skew is a structural consequence of `rho != 0`, not something fitted strike-by-strike the way SVI's `rho` parameter is fitted per expiry.

**Q9. Heston reprices "today's vanillas" only approximately, per METHODOLOGY.md's own comparison table. What's the actual trade-off against local vol (Dupire), and why does this portfolio still prefer Heston?**

Local vol reprices today's vanillas *exactly* by construction, but the Hagan critique shows its forward smiles are known-wrong — they flatten over time, which mis-prices anything sensitive to how the smile evolves (forward-starts, cliquets, autocalls). Heston gives up exact repricing (METHODOLOGY.md documents a 2-3 vol-point short-dated residual) in exchange for forward smile dynamics that are qualitatively realistic — the smile moves with spot, vol clusters and mean-reverts — which is what actually matters for path- and dynamics-dependent exotics. The FX project's METHODOLOGY.md makes the identical trade-off explicit for Dupire vs. Heston under Garman-Kohlhagen, adding that FX pillar data (5 strikes x 6 expiries) is in any case too sparse to extract a stable local-vol surface.

**Q10. What no-arbitrage conditions must a real vol surface satisfy, and does this portfolio's documentation show it actually checks for them?**

Two classic conditions: **butterfly (strike) arbitrage** — the risk-neutral density must be non-negative at every strike — and **calendar-spread arbitrage** — total variance must be non-decreasing in expiry at fixed moneyness/delta. Both are checked, not just described. Butterfly arbitrage is diagnosed by the Durrleman condition `smile.durrleman_g` / `smile.check_butterfly`, and calendar arbitrage by `surface.check_calendar`, which flags any grid point where total variance decreases across pillars. VALIDATION.md §6 (equity) turns both into executable property tests (`tests/test_arbitrage_and_wings.py`) — e.g. `test_butterfly_checker_fires_on_arbitrage_violating_quotes` plants a V-shaped smile and asserts `min_g < 0` is actually detected, and `test_calendar_checker_detects_decreasing_total_variance` does the equivalent for calendar. The FX project's F7 failure mode documents the same two planted-violation checks (SVI `b=0.35, rho=-0.9` for butterfly; a 1m ATM bumped to 20% for calendar).

**Q11. If a slice fails the Durrleman check, does the surface silently fix it, or does something else happen?**

It's diagnosed, not silently repaired — assumption #11 in the equity METHODOLOGY.md states this explicitly. `fit_svi` emits a `UserWarning` when `check_butterfly` finds `min_g < 0`, and `VolSurface.__init__` similarly warns on any calendar violation from `check_calendar`, with the monotone running-max fix applied only when the caller opts in via `enforce_calendar=True`. DESK_GUIDE.md's daily workflow makes the human step explicit: "a flagged slice is a quote problem or an arbitrage, either way a human looks before the mark goes out." The checkers are themselves tested against planted violations precisely so a silent regression in the checker itself would fail the suite rather than let a bad surface publish unnoticed.

**Q12. Why does the calendar-arbitrage running-max "fix" come with a warning that it's dangerous to blindly apply, per the equity DESK_GUIDE.md earnings scenario?**

Real event variance (e.g. a single-name earnings print, or a central-bank date in FX) concentrates a jump in total variance at one specific expiry pillar in a way that looks identical, in smooth ACT/365F time, to a calendar-arbitrage violation: the post-event annualised vol can sit far below the pre-event pillar. Running `enforce_calendar=True` would smear that genuine event variance across the term structure via the monotone running-max adjustment, destroying real information. The documented handling instead strips the event component (`w = w_diffusive + w_event * 1{T >= t_event}`), marks the diffusive surface, and re-adds the event — leaving `enforce_calendar` off and treating the calendar flag as the correct alarm, not a bug to suppress.

**Q13. Heston's characteristic function has a well-known numerical trap. What is it, and how does this code avoid it?**

`heston.py`'s module docstring explains: the original 1993 Heston CF formulation contains `ln((1 - g1*e^{dT})/(1 - g1))` with a *growing* exponential `e^{dT}`, so as `u` or `T` grows the log's argument spirals across the negative real axis and the principal-branch complex logarithm jumps by `2*pi*i` — silently wrong, discontinuous prices unless the branch is tracked by hand. `heston_cf` instead uses the algebraically equivalent "little Heston trap" form (Albrecher et al. 2007) built from the conjugate root `g2 = (b-d)/(b+d)`, whose exponential `e^{-dT}` decays to zero instead of growing, so the log argument never winds around the origin. `test_little_trap_cf_continuous_in_u` checks this by scanning a dense `u`-grid at `T=10, xi=1` and asserting no branch jumps.

**Q14. What is the Feller condition, and why does this codebase warn on violation instead of rejecting the parameters?**

Feller is `2*kappa*theta >= xi^2` (equivalently `feller_condition` returning a ratio `>= 1`); it's the condition under which the CIR variance process stays strictly positive. Both projects' `heston.py` deliberately warn rather than raise (`FellerWarning`) because market calibrations routinely violate it — short-dated equity and FX skew both demand high `xi` relative to `kappa*theta` — and the model stays mathematically well-defined even so: variance touches zero and reflects, and the characteristic function remains valid. The equity fitted surface sits at Feller ratio 0.80; both FX presets sit near 0.46. What Feller violation *does* change is which Monte Carlo scheme is safe to use (Q15).

**Q15. What breaks in Monte Carlo simulation when Feller is violated, and how does this portfolio handle it?**

Full-truncation Euler carries an O(dt) truncation bias from clipping negative variance draws to zero, and that bias explodes exactly when Feller is violated: equity VALIDATION.md's "extreme set" (`xi=1`, Feller ratio 0.08) shows Euler still running **+20 standard errors** off the Fourier reference at 64 steps/year, and **+135 SE** at 8 steps/year — nearly 50% of the ATM price. Andersen's QE scheme (moment-matched CIR transition, implemented as the `scheme="qe"` branch in both `heston_mc.py` files) stays within Monte Carlo noise at 8 steps/year on the same parameter set. Both desk guides make this a hard scheme-control rule: never use Euler for pricing under a Feller-violating fit, and both `heston_mc.py` implementations default `scheme="qe"`.

**Q16. What are antithetic variates, and why bother with them at all in a Monte Carlo pricer?**

Antithetic variates are a variance-reduction technique: instead of drawing `n` independent standard normals, you draw `n/2` and pair each one `z` with its negation `-z`, then average the payoff over each pair. Because the payoff function applied to `z` and to `-z` tend to have errors that partially cancel (for a monotone or convex payoff, over- and under-shoots trade off), the pairwise-averaged estimator has lower variance than `n` fully independent draws for the same simulation cost. `fx_surface.heston_mc.simulate_terminal` implements this via `antithetic: bool = True`: for both Euler and QE it draws `n_base` normals and mirrors them (`np.concatenate([z1, -z1])`), and `mc_price` averages payoffs pairwise before computing the sample standard error, so the reported SE is not artificially deflated by the correlation the pairing itself introduces.

**Q17. This project's review history found a real bug in `simulate_terminal`/`mc_price` tied to antithetic pairing and odd `n_paths`. What was it, concretely?**

An odd `n_paths` silently broke the antithetic pairing invariant the whole standard-error calculation depends on. With `n_paths` odd, `n_base = (n_paths + 1) // 2` antithetic normals get generated, mirrored to `[z, -z]`, and then truncated back down to `n_paths` — which drops the very last element of the mirrored half, so the last "base" draw loses its antithetic partner while every other draw still has one. `mc_price`'s pairwise averaging (`payoff[:half] + payoff[half:]`) then assumes a clean 50/50 split of matched antithetic pairs; with an unpaired leftover path in the mix, that assumption is violated and the averaging silently starts treating what should be correlated antithetic pairs as if they were independent samples.

**Q18. Why was this bug specifically dangerous — worse, arguably, than a bug that just produces a visibly wrong price?**

Because the *price* estimate itself was still approximately correct (antithetic pairing barely nudges the mean); what was wrong was the reported *standard error* — and it was wrong in the direction of being too small, i.e. it understated the pricer's own uncertainty. A visibly wrong price gets caught immediately, because someone compares it to the Fourier reference and it's obviously off. A confidently-wrong error bar does not get caught the same way: a risk or model-validation process that trusts "3 SE agreement" as its acceptance bar (this portfolio's own documented tolerance, per CONVENTIONS.md) would see an artificially tight SE make a genuinely marginal or biased result look like it passed comfortably, when the true uncertainty band was wider. An error bar that lies about its own size is more dangerous than a wrong point estimate precisely because it undermines the mechanism you'd normally use to catch the point estimate being wrong.

**Q19. What was the actual fix, and where does it live in the code?**

`simulate_terminal` now rejects odd `n_paths` outright under `antithetic=True`, raising `ValueError(f"n_paths must be even when antithetic=True (odd n_paths breaks pairing and silently understates the standard error), got {n_paths}")` before any simulation happens (`heston_mc.py` lines 70-81 in the FX project). The fix is a validation guard, not an attempt to silently handle the odd case some other way — consistent with this portfolio's stated preference (see equity VALIDATION.md's short-end-floor writeup and FX VALIDATION.md's F8 finding) for raising loudly over degrading silently. `mc_price`'s docstring/comment now also states the invariant it depends on directly: "`simulate_terminal` enforces `n_paths` even under `antithetic=True`, so this split is always a clean pairing of `[z, -z]` halves."

**Q20. How exactly does `mc_price` use the antithetic pairing to compute its standard error, once `n_paths` is guaranteed even?**

After `simulate_terminal` returns `n_paths` terminal spots, `mc_price` computes the discounted payoff for all of them, then — only when `antithetic=True` — splits the payoff vector exactly in half and averages each `z`/`-z` pair: `payoff = 0.5 * (payoff[:half] + payoff[half:])`. The standard error is then `np.std(payoff, ddof=1) / sqrt(len(payoff))` computed on that *halved, pre-averaged* array, not on the raw `n_paths` payoffs. This is the step that actually removes the antithetic correlation from the SE estimate: treating the `n_paths` raw draws as independent would understate variance because paired draws are negatively correlated by construction, so averaging within each pair first and computing dispersion across pairs is what makes "sample std / sqrt(n)" a valid SE formula again.

**Q21. Does the equity project's Monte Carlo engine have the same antithetic-pairing exposure?**

No — a search of `eq_surface/heston_mc.py` shows it has no `antithetic` parameter at all; `simulate_heston_terminal` always draws `n_paths` independent normals directly (`rng.standard_normal(n_paths)`), so the pairing invariant that broke in the FX project simply doesn't exist there. That's a useful cross-check when studying the bug: the FX engine's extra sophistication (antithetic variance reduction) is exactly what created the extra invariant that an odd `n_paths` could violate — a simpler, non-antithetic MC engine has no equivalent failure mode, at the cost of needing roughly twice the paths for the same variance.

**Q22. FX vol surfaces are quoted differently from equity ones. What does "ATM/RR/BF by tenor" actually mean, and why does the FX market quote this way instead of strike/expiry directly?**

Per `fx_surface`'s METHODOLOGY.md, brokers quote five numbers per expiry — ATM (the delta-neutral straddle strike), 25-delta risk reversal `RR25 = sigma(25∆call) - sigma(25∆put)` (the skew, tradable as a package), and 25-delta butterfly `BF25 = 0.5*(sigma(25∆call) + sigma(25∆put)) - sigma_ATM` (the convexity), each repeated at 10-delta. This is quoted in delta space rather than strike space because FX spot moves ~1% a day, making any strike-based quote sheet stale within hours and incomparable across pairs trading at wildly different absolute levels (EURUSD ≈ 1.10 vs USDJPY ≈ 150); delta is a moneyness measured in probability units, so it's self-normalising across spot level, pair, and vol regime — exactly why the market settled on it.

**Q23. Given quotes in ATM/RR/BF form, how does the FX pipeline turn that into strikes it can actually build a surface from?**

`smile_from_quotes.py` first maps the five ATM/RR/BF numbers to the two wing vols by the exact linear relations `sigma_C = ATM + BF + RR/2`, `sigma_P = ATM + BF - RR/2` (round-trip tested to 1e-14). Getting from a target delta to the strike behind it is not always closed-form: under unadjusted delta conventions the strike inverts directly, but under premium-adjusted conventions (used for USDJPY and most USD-base pairs) the call delta is *non-monotone* in strike — it vanishes at both `K→0` and `K→∞` with an interior maximum — so a given delta has two candidate strikes, and `strike_from_delta_pa_candidates` exposes both while the market convention (the higher, OTM, falling-branch strike) is selected and tested explicitly.

**Q24. Why does the FX surface interpolate at fixed delta rather than fixed log-moneyness, unlike the equity surface?**

Because FX is a sticky-delta market: METHODOLOGY.md §4 explains that the quoted objects (ATM/RR/BF) float with spot and vol, so a fixed log-moneyness represents a very different number of deltas out-of-the-money at a 1-week tenor than at a 1-year tenor — interpolating in moneyness would badly distort the short-dated wings. `fx_surface.surface` instead interpolates total variance linearly in `T` at fixed delta, and getting `vol(K, T)` for an absolute strike runs a sticky-delta fixed point (strike → delta → vol → delta, converging in a few iterations), with the identity `vol(K,T) = vol(Δ(K),T)` tested to 1e-9.

**Q25. FX has a second smile model, vanna-volga, alongside SVI — what does it do differently and why keep both?**

Vanna-volga prices any strike as the flat-ATM Black-Scholes price plus the cost of a hedge: the portfolio of the three traded pillars (25-delta put, ATM, 25-delta call) that exactly replicates the target option's vega, vanna and volga at the reference vol, solved as an exact 3x3 linear system per strike (`smile.py`, residual < 1e-12 tested) rather than the common first-order shortcut. It's exact at the pillars, cheap, and is what desks actually use to smile-adjust barriers and touches — but METHODOLOGY.md is explicit that it has no dynamics and its quadratic wing extrapolation can violate no-arbitrage beyond about 5-delta. SVI is kept alongside it because SVI carries the analytic Durrleman diagnostic VV lacks; the pipeline compares both at 15-delta (agreeing to ~5bp) and in the wings (diverging 10+bp at 5-delta), and that divergence is treated as the honest measure of interpolation-model risk rather than something to reconcile away.

**Q26. Both Heston MC engines in this portfolio are cross-validated against Fourier pricing within "3 standard errors." Why does that specific tolerance only mean what it's supposed to mean once the antithetic bug above is fixed?**

The "agree within 3 SE" contract (stated in CONVENTIONS.md as the portfolio-wide Monte Carlo tolerance, and used throughout both `VALIDATION.md` files, e.g. equity's `|z| <= 1.6`/`2.4` cross-checks and FX's Euler/QE-vs-COS table) is only a meaningful acceptance bar if the reported SE is an honest estimate of the pricer's actual sampling uncertainty. If an odd `n_paths` were silently understating the SE, a genuinely biased or high-variance MC price could still land inside a falsely narrow "3 SE" band and pass a test or a model-validation sign-off that should have failed it — the tolerance's usefulness is entirely downstream of the SE calculation being correct, which is exactly the invariant `simulate_terminal`'s odd-`n_paths` guard now protects.


## Round 6 — Market Risk: VaR Methods

**Q1. What does a 1-day 99% VaR of $54,256 actually tell you?**

It is a quantile of the P&L distribution, not a prediction of the worst loss: it says there is a 1% (unconditional) probability that tomorrow's P&L falls below −$54,256. Formally the engines define `VaR_alpha = -Q_alpha(P&L)`, so VaR is a threshold, not a loss estimate. It says nothing about how bad the loss is *if* that threshold is breached — that is exactly what Round 7's Expected Shortfall is for.

**Q2. Why is "VaR ignores tail severity" considered VaR's most-cited structural weakness?**

Because a single number can pass every backtest while hiding an arbitrarily fat tail beyond it: `docs/VALIDATION.md` §5.1 shows the equity demo book's 99% VaR is $54k, but the 2020 COVID-replay stress loss is $254k and the ES beyond VaR is $68k — a 5× gap between "the VaR number" and what actually happens once you're in the tail. VaR is a pass/fail line; it carries no information about the average or worst loss conditional on crossing it. This is the motivating defect fixed by Expected Shortfall (`ES_alpha = -E[P&L | P&L <= Q_alpha]`), which FRTB adopted precisely because it forces the tail's magnitude into the number.

**Q3. What does historical simulation assume, and what is its single biggest limitation?**

Plain historical VaR (`historical_var` in `eq_var/historical_var.py`) makes no distributional assumption at all — it just takes the empirical `alpha`-quantile (NumPy type-7/linear interpolation) of realised P&L, so it captures whatever fat tails, skew, and cross-asset dependence actually happened. Its limitation is that it is bounded by how much history you have and by the assumption that the future resembles that specific window: `MIN_OBS = 50` is enforced because a 1–5% tail is unresolvable on shorter samples, and even at 250+ days a 1% quantile rests on only 2–3 tail observations per year (`docs/METHODOLOGY.md` §2 table, "Estimation noise at 99%: high").

**Q4. What is "window myopia" (the "great moderation" problem) in historical VaR, and how is it demonstrated?**

It is the failure mode where plain HS is only as good as its window: after 450 calm days followed by 50 wild days, plain HS reports a VaR ~40% below FHS because the wild days are still a minority of the sample (`test_fhs_scales_up_after_vol_regime_switch`, `docs/VALIDATION.md` §5.4). Symmetrically, once a crisis rolls out of the window, plain HS overstates current risk long after the regime has calmed. The 500-day GARCH walk-forward backtest in `docs/VALIDATION.md` §4 turns this into p-values: plain HS draws 11 exceptions against an expectation of 5 (Kupiec p = 0.02, yellow zone), and on a separate seed it fails the Christoffersen independence test outright (p = 0.02) because the exceptions cluster in the high-vol subwindow.

**Q5. What does age-weighting (BRW) add over plain historical VaR, and why choose a particular decay `lam`?**

`brw_weights` in `eq_var/historical_var.py` assigns observation `i` weight proportional to `lam**(n-1-i)`, so more recent P&L dominates the weighted empirical CDF that `age_weighted_var` inverts; `lam -> 1` recovers plain HS (`test_age_weighting_converges_to_plain_hs_as_lambda_goes_to_one`). The reason to weight recent history more heavily is that risk is regime-dependent — a calm-then-turbulent transition should show up in VaR within days, not once the window rolls over. The choice of decay is a genuine bias-variance trade-off: a `lam` close to 1 barely differs from plain HS (slow to react, but stable, low estimation noise), while a fast decay concentrates the effective sample on very few recent points, reacting quickly but at the cost of a noisier quantile estimate — the same trade-off the engine flags for EWMA's `lam=0.94` in assumption A6 of `docs/METHODOLOGY.md`: "too-slow decay lags regime shifts; too-fast is noisy."

**Q6. Why does the portfolio treat Filtered Historical Simulation (FHS), not plain HS or BRW, as the flagship historical estimator?**

FHS (`filtered_historical_var`) devolatilises each historical P&L by its one-step-ahead EWMA vol forecast (`z_t = pnl_t / sigma_t`), then rescales every standardised innovation to *tomorrow's* forecast before taking the empirical quantile — so it keeps the empirical (fat-tailed, skewed) shape of returns while being conditional on the current vol regime, unlike plain HS (unconditional) or BRW (reweights the sample but does not rescale its scale). `docs/METHODOLOGY.md` §2 calls this combination "the industry workhorse," and the GARCH backtest proves why: FHS lands in the green zone (6 exceptions, Kupiec p = 0.66, CC p = 0.85) while parametric-normal and plain HS land in yellow with Kupiec p ≤ 0.02.

**Q7. What does the parametric (delta-normal) method assume, and how is portfolio sigma computed?**

It assumes P&L is well described by a distribution — the normal baseline, or Student-t / Cornish-Fisher tail corrections — and computes VaR analytically from that distribution's mean and variance rather than resampling history. `portfolio_sigma` in `eq_var/parametric_var.py` computes `sigma_p = sqrt(w' Sigma w)` from dollar exposures `w` (the factor mapping) and a sample or EWMA covariance `Sigma`; `parametric_var` then returns `-(mu + z_alpha * sigma)` with `z_alpha = norm.ppf(alpha)` for the normal case. It is cheap (uses the whole sample for sigma rather than just tail points) and analytically decomposable, which is why the FX desk guide calls it "the desk's intraday what-if and marginal-VaR tool."

**Q8. Why does pure normality understate real, fat-tailed market risk?**

A normal distribution has no excess kurtosis, so its tail decays faster than real daily P&L, which typically shows excess kurtosis (the equity GARCH sample has 4.76 at 1-day per `docs/METHODOLOGY.md` §6). The cross-model table in `docs/VALIDATION.md` §3 makes this concrete: on the demo book, parametric-normal 99% VaR is $46,864 while parametric-t(6) is $51,691 and historical/MC-t agree near that higher number — "the normal family sits ~10% lower at 99% — the missing kurtosis." The 500-day GARCH backtest converts this into an actual failure: parametric-normal draws 14 exceptions against an expectation of 5, Kupiec p = 0.0009.

**Q9. What is the Cornish-Fisher expansion doing, mathematically, and what does it need beyond a plain z-score?**

CF adjusts the standard normal quantile using the P&L's sample skewness `S` and excess kurtosis `K`: `z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36` (`cornish_fisher_z` in both `eq_var/parametric_var.py` and `fx_var/parametric_var.py`), reducing exactly to `z` when `S = K = 0` (unit-tested to 1e-12). It is attractive because it is cheap and uses distributional information — skew and fat tails — that the plain normal quantile ignores entirely, without requiring a full Monte Carlo or historical resample.

**Q10. Why does the Cornish-Fisher quantile need a domain/monotonicity check at all — what goes wrong without one?**

`z_cf` is a cubic polynomial in `z`, and a cubic is only a valid quantile function where it is monotone increasing in `z`; for large enough skew/kurtosis combinations the polynomial turns over inside `|z| <= 3.5`, so a *smaller* `z` (less extreme alpha) can map to a more extreme `z_cf` than a larger one. `docs/METHODOLOGY.md` §5 states the consequence directly: "the 99% 'quantile' can cross the 95% one; a number produced there is not a VaR." That is why `cornish_fisher_var` in both engines raises `ValueError` by default (`check_domain=True`) instead of silently returning a number when `(skew, excess_kurt)` fall outside the monotone region — e.g. S=3 or K=10 is rejected, S=−0.3, K=2 passes (`test_var_raises_outside_domain`).

**Q11. Derive the domain-check condition: why is checking `dz_cf/dz > 0` equivalent to checking a quadratic's minimum?**

Differentiating the CF cubic with respect to `z` gives `dz_cf/dz = 1 + zS/3 + (3z^2-3)K/24 - (6z^2-5)S^2/36`, which — collecting terms in `z` — is itself a quadratic `g(z) = A z^2 + B z + C` with `A = K/8 - S^2/6`, `B = S/3`, `C = 1 - K/8 + 5S^2/36` (see `cornish_fisher_domain_ok` in `eq_var/parametric_var.py`). The expansion is monotone on an interval exactly when this quadratic derivative stays strictly positive there, so the domain check reduces to finding `g`'s minimum on `[-z_range, z_range]` and testing its sign — a textbook one-variable optimisation, not a property that needs numerical search.

**Q12. What was the actual bug in the Cornish-Fisher domain check, and in how many places did it exist?**

The original `cornish_fisher_domain_ok` evaluated the monotonicity derivative on a finite grid (2001 points over `|z| <= 3.5` in equity, an even coarser 801-point value-diff grid over `|z| <= 4.0` in FX) and flagged non-monotonicity only if a grid point showed a negative derivative. A grid scan can miss a thin non-monotone region that falls entirely between two adjacent grid nodes — the region is real, but no sampled point lands inside it. `docs/VALIDATION.md` §5.7 gives the equity counterexample: `(skew, excess_kurt) = (-0.0105, 8.0001)` has a true minimum derivative of about `-1.0e-6` between two adjacent grid nodes, which the 2001-point grid reported as monotone. The same class of bug existed in all six VaR/ES engines in the portfolio — Python, C++, and Rust, for both equity and FX — because they all implemented the same grid-scan pattern; `docs/VALIDATION.md` notes the FX version was "an even easier-to-hit counterexample" since its grid was coarser (801 points over a wider range).

**Q13. What is the fix, and why is it described as "provably correct" rather than just "less likely to fail"?**

The fix replaces the grid scan with the exact closed-form minimum of the quadratic derivative `g(z) = A z^2 + B z + C`: for convex `g` (`A > 0`) the global minimum on `[-z_range, z_range]` is at the vertex `z* = -B/(2A)` if that vertex lies inside the interval, otherwise at whichever endpoint is closer; for concave or linear `g` (`A <= 0`) a concave function's minimum on a closed interval is always at an endpoint, so checking both endpoints suffices (`cornish_fisher_domain_ok`, both `eq_var` and `fx_var`). This isn't a resolution improvement — it's a qualitatively different guarantee: a grid, however fine, always leaves gaps between nodes where a violation can hide, whereas the closed-form check evaluates the *actual* minimum of a one-dimensional quadratic over a closed interval, which by construction cannot be missed regardless of how thin the non-monotone region is. The docstring even retains `n_grid` as a parameter "for API compatibility (validated below) but no longer affects the result" — a visible trace of the old implementation kept only so callers don't break.

**Q14. How does the equity engine's counterexample differ from the FX engine's, and what does that difference tell you about the fix?**

The equity engine's bug needed a fairly extreme, specific pair — `(skew, excess_kurt) = (-0.0105, 8.0001)` — with a tiny minimum derivative (~−1.0e-6) to slip through its finer 2001-point grid. The FX engine's counterexample is described as "an even easier-to-hit counterexample" on a coarser 801-point grid: `skew=0.122, excess_kurtosis=-0.427` has a minimum derivative of about −9e-4 near `z ~ 3.1`, a much less pathological pair of moments, because a coarser grid needs a much less thin dip to fall between its nodes. The lesson is that grid resolution and counterexample "extremeness" trade off directly against each other — which is exactly why a resolution-independent, closed-form check was needed rather than simply making the FX grid finer to match equity's.

**Q15. Why does Monte Carlo VaR fully revalue the book per scenario instead of using a linear or delta-gamma approximation?**

Full revaluation reprices every instrument (e.g. Black-Scholes / Garman-Kohlhagen at the shocked spot, rates, and vol) at each simulated scenario, so it captures the true, non-linear P&L including option convexity and the exact payoff shape — it makes no local-approximation error at all, unlike a delta or delta-gamma mapping. `monte_carlo_pnl` in `eq_var/monte_carlo_var.py` defaults to `method="full"` for exactly this reason, and `docs/METHODOLOGY.md` §2 calls MC "the only family that combines a chosen tail model with exact treatment of option convexity." The cost of the approximation it avoids is measured directly elsewhere in the portfolio (`Portfolio.approximation_error`): delta-gamma understates the COVID stress loss by $93k (37% of P&L) once a long put goes deep ITM, because the quadratic Taylor term keeps growing while the true payoff is bounded.

**Q16. What three factor-return distributions does the FX Monte Carlo engine support, and when is each the right choice?**

`fx_var/monte_carlo_var.py`'s `simulate_factor_returns` supports `"normal"` (MVN(0, Sigma·h) via `robust_cholesky`), `"t"` (multivariate Student-t built as `Z·sqrt((df-2)/df)/sqrt(W/df)`, scaled so the *covariance still matches Sigma exactly* — any VaR difference from normal is pure tail shape, not a sigma mismatch), and `"jump"` (normal diffusion plus a Bernoulli(prob) common jump with per-factor mean/std via `JumpSpec`). Normal is the RiskMetrics-style baseline when tail shape isn't the concern; Student-t is the right choice for EM-currency-like fat tails at matched covariance (df≈4-6, +12% at 99% VaR for the demo EM book per `docs/METHODOLOGY.md` §2.3); the jump-mixture is specifically for devaluation/peg-break risk that the covariance matrix structurally cannot see — it "adds variance on top of Sigma by design" and lifts the demo book's 99% VaR by +140%.

**Q17. Concretely, what does `JumpSpec` let you model, and how does it enter the simulated scenarios?**

`JumpSpec` (a frozen dataclass in `fx_var/monte_carlo_var.py`) carries a per-scenario jump probability `prob`, a per-factor jump mean (e.g. `{"FX:TRY": -0.15}` for a 15% log devaluation of the lira), and an optional per-factor jump std (0 = deterministic jump size). Inside `simulate_factor_returns`, a Bernoulli draw `hit = rng.random(n_scenarios) < jumps.prob` selects which scenarios get the jump, and for every factor listed in `jumps.mean` a random jump size `mean + std * standard_normal` is added to that factor's diffusion path only where `hit` is true. This is the mechanism behind the peg-break/devaluation overlay described in `docs/METHODOLOGY.md` §4 — it prices a discrete, low-probability, large-magnitude event into a quantile, which a continuous covariance-driven diffusion cannot represent.

**Q18. Why is a "legitimate" case for a singular or near-singular covariance matrix specifically an FX/pegged-currency scenario, and how does `robust_cholesky` handle it?**

Two currencies pegged to the same anchor (e.g. an HKD-band or SAR hard peg both effectively tracking USD) realise near-zero daily vol and near-perfect correlation to each other and to the anchor, which makes the factor covariance matrix exactly or numerically singular — this is a real, structural feature of FX markets, not a data artefact to be cleaned up. `robust_cholesky` in `fx_var/monte_carlo_var.py` first tries plain `np.linalg.cholesky`; on `LinAlgError` it adds jitter `1e-12 * mean(diag)` to the diagonal and retries, escalating the jitter ×10 up to `max_tries` (default 8) times, raising `ValueError` only if factorisation still fails at maximum jitter. Each successful jittered factorisation emits a `NumericalWarning` reporting the jitter actually used, so the escalation is visible rather than silent — and because the jitter is tiny relative to the real variances, simulated moments are unchanged to within MC noise (the equity twin, `safe_cholesky`, does the identical thing and is unit-tested to `corr(sim) = 1.000 +/- 0.001` on perfectly correlated assets).

**Q19. Why can't parametric or historical VaR see peg-break risk, and what is the engine's policy response?**

A pegged pair's return history inside any window that doesn't contain the actual break is a spike at zero — near-zero daily vol, so the empirical quantile (HS) and the covariance-implied sigma (parametric) are both near-zero right up until the peg snaps. `docs/METHODOLOGY.md` §4 states the policy directly: any FX factor with daily sigma below 0.05% (~0.8% annualised) triggers a `PegBlindnessWarning` naming the factor; the flagged book must then carry a `peg_break_scenario` stress add-on (a configurable jump, vol spike, and contagion co-move); and for a genuine quantile that includes the break, the jump-mixture Monte Carlo is the tool, since it prices the event with an attached probability rather than assuming it away. The demo-book numbers make the gap concrete: HS 99% VaR of $0.69m vs. a −30% HKD peg-break stress loss of $15.0m — 21.7× the VaR.

**Q20. What is the real architectural difference between how equity VaR and FX VaR handle Monte Carlo revaluation?**

Equity's Monte Carlo (`eq_var/monte_carlo_var.py`) revalues a `Portfolio` of linear equity/index exposures plus options, where non-option positions are pure delta exposures (`P&L = dollar-exposure * return`) and only options get a full Black-Scholes reprice. FX's Monte Carlo (`fx_var/monte_carlo_var.py`, via `fx_var/book.py`) does full book revaluation for *every* position type through a `Book`/`Market` layer: `Book._position_value_usd` reprices cash, spot, forward (as two CIP deposit legs), and option (via Garman-Kohlhagen) positions scenario by scenario from shocked spot/rate/vol curves, then differences base-currency value before and after the shock (`Book.pnl`). There is no "linear exposure times return" shortcut anywhere in the FX P&L path — even a plain spot FX position is repriced as two USD legs (`pos.notional * spot[b] - pos.notional * x0 * spot[q]`) rather than approximated by a single delta, because USD triangulation requires every cross to be reconstructed consistently from its legs in every scenario.

**Q21. Why does FX VaR need the USD-triangulation factor scheme instead of one factor per traded pair?**

If every traded pair (EURUSD, USDJPY, EURJPY, …) got its own risk factor, the covariance/scenario set would have to satisfy the non-linear constraint `log EURJPY = log EURUSD + log USDJPY` exactly, or a historical/parametric/MC scenario could move EURJPY without moving either USD leg — an arbitrage-inconsistent scenario. `fx_var/METHODOLOGY.md` §1.1 explains that mapping every currency to a single `FX:CCY` factor (the USD price of one unit of that currency) makes the identity hold *by construction* in every scenario, and a cross position such as EURJPY is simply decomposed into its two USD legs. This is tested to machine precision (`test_triangulation_identity_eurjpy`).

**Q22. How does `Book.pnl` compute base-currency P&L, and why does a pure base-currency cash balance carry zero risk?**

`Book.pnl` computes `PnL = V1_usd / S1_base - V0_usd / S0_base`, where `V0_usd`/`V1_usd` are the book's total USD value before/after the shock (summed via `_position_value_usd` over all positions) and `S0_base`/`S1_base` are the base currency's own USD price before/after — critically, the base currency's own FX factor is applied consistently to *both* the book value and the denominator. A pure base-currency `Cash` position has `V_usd = amount * spot[base]`, so dividing by `spot[base]` again cancels exactly regardless of how the base-currency factor is shocked, leaving zero P&L — this is a direct consequence of measuring P&L in the position's own currency, not an extra rule bolted on.

**Q23. What does `linear_exposures` compute in the FX engine, and where in the pipeline is a linear approximation still used despite full-reval MC being available?**

`Book.linear_exposures` in `fx_var/book.py` computes central finite-difference deltas `dPnL/dfactor` for every factor the book touches, by bumping each factor by `+/-bump` (default 1e-6) through the *same* full-revaluation `pnl` method and dividing by `2*bump` — so even the "linear" exposures used by parametric VaR are derived from the true GK/CIP repricing, not a separate analytic Greek formula. These exposures are what `parametric_var`/`var_covar` use to compute `sigma_p = sqrt(w' Sigma w)`, since parametric VaR is inherently a linear (delta) method by construction — the linearisation happens in how parametric VaR *consumes* the book, not in how the book itself is priced.

**Q24. Why does the FX options pricer default to full Garman-Kohlhagen revaluation rather than delta-vega(-gamma) mapping, and what does the mapping systematically get wrong?**

`fx_var/METHODOLOGY.md` §1.3 states the mapping's error is quadratic (cubic once gamma is added) in the shock size, and characterises it precisely: for a long option (positive gamma), delta-only P&L *understates* gains and *overstates* losses in both directions, which makes a mapping-based VaR conservative for long-gamma books but dangerous — understating risk — for short-gamma books. Because the desk cannot assume every book is long gamma, full revaluation (`option_method="full"` in `Book._position_value_usd`, which calls `gk_price` at the shocked spot/rate/vol per scenario) is the default; the vectorised implementation makes this cheap (100k scenarios in milliseconds), so there is no real performance reason to prefer the approximation for VaR.

**Q25. How does the equity engine quantify the size of the delta-gamma approximation error, rather than just asserting full reval is "more correct"?**

`Portfolio.approximation_error` computes both the full-revaluation P&L and the delta-gamma-vega P&L (`dV ~ Delta*dS + 1/2*Gamma*dS^2 + nu*dsigma`) for the same scenario and reports the difference directly, so the error is *measured*, not assumed. On the demo book it is small for a −15% scenario (−$1.2k, 0.9%) but balloons to +$93k (37% of P&L) in the −34% COVID replay, because the quadratic Taylor term keeps growing with the shock while the true option payoff is bounded — this is unit-tested as monotone in shock size (`test_approximation_error_grows_with_shock_size`) and sign-correct (long gamma cushions losses, short gamma amplifies them, `test_long_gamma_reduces_loss_vs_pure_delta`). `docs/DESK_GUIDE.md` calls this divergence the stress committee's evidence that "Greeks-based intraday risk is unsafe for gap moves and the overnight batch must fully revalue."

**Q26. Why does the FX Monte Carlo engine's variance-matched Student-t scale by `sqrt((df-2)/df)` rather than sampling a raw multivariate-t?**

A raw multivariate Student-t with `df` degrees of freedom has covariance `df/(df-2) * Sigma_scale`, so sampling directly from a t with scale matrix `Sigma` would give factor returns with a *larger* covariance than `Sigma` — confounding "the tails are fatter" with "the variance is also bigger." `fx_var/monte_carlo_var.py`'s `simulate_factor_returns` instead draws `x = z @ chol.T` from the normal Cholesky factor of `Sigma` and then multiplies by `sqrt((df-2)/df) / sqrt(W/df)` (`W` a chi-square(df) draw), which rescales the t draw so its covariance matches `Sigma` exactly. This means any VaR difference between the `"normal"` and `"t"` runs at the same `Sigma` is attributable purely to tail shape — the module docstring states this explicitly: "any 99% VaR difference is pure tail shape."

**Q27. What is the practical trap in "switching to Student-t VaR to be more conservative," according to this portfolio's tests?**

The variance-matched Student-t is fatter than the normal *only* in the deep tail — near the shoulder of the distribution (alpha around 5-10%) it is actually thinner, because matching the variance forces the t density to be lower near the centre to compensate for its fatter far tail. `docs/VALIDATION.md` §6.1 documents the crossover explicitly: t(5) charges *less* than normal at 95% VaR, with the crossover between t and normal occurring near alpha ≈ 2.8% (`test_t_vs_normal_var_crossover_is_between_2_5_and_5_percent`), while ES — which averages the whole tail rather than reading off one point — stays higher than normal at every alpha. The practical trap: someone switching a 95% VaR limit from normal to Student-t "to be conservative" would actually see the reported VaR *fall*.

**Q28. Why does the engine implement three VaR families instead of picking the "best" one?**

`docs/METHODOLOGY.md` §2 states the reasoning directly: no single VaR estimator dominates in every regime, so every real risk department runs at least two and reconciles them — the engine implements historical, parametric, and Monte Carlo so the *disagreement between methods itself becomes a diagnostic*. The cross-model consistency table in `docs/VALIDATION.md` §3 shows how to read that disagreement: MC-normal tracking parametric-normal closely cross-validates both implementations (same model, only MC noise separates them); parametric-t, MC-t, and historical clustering together says the fat-tail-aware methods agree because the underlying history really is t-distributed; and FHS sitting noticeably below the unconditional methods is not a bug but information — "current vol is quiet relative to the window," exactly what a conditional estimator is supposed to show when the regime has calmed.

**Q29. Why is `sigma_p = sqrt(w' Sigma w)` guarded with a "not positive semi-definite" check, and what would trigger it?**

`portfolio_sigma` in `eq_var/parametric_var.py` computes `var = w @ Sigma @ w` and raises `ValueError` if `var` is negative beyond a small numerical tolerance (`-1e-10 * max(1, max|w|^2)`), rather than silently taking `sqrt` of a negative number (which would produce NaN or a complex value depending on the code path) or clamping without comment. A genuinely valid covariance matrix is positive semi-definite by construction, so `w' Sigma w >= 0` for any `w`; a violation signals that `Sigma` itself is broken — e.g. built from mismatched/inconsistent inputs, or an EWMA/sample estimate corrupted upstream — and the engine's stated policy across both `eq_var` and `fx_var` is "refuse, never impute": a NaN or structurally invalid VaR must fail loudly rather than colour a traffic light incorrectly, exactly the concern documented for FX assumption A12.

**Q30. Given all three VaR families and the Cornish-Fisher fix, what is the one-sentence takeaway for how a desk should treat any single VaR number?**

Every method here encodes a different modelling choice — how much history to trust and how to weight it (historical/BRW/FHS), what distribution to assume and how to correct it (parametric normal/t/Cornish-Fisher, now with a provably-correct domain check instead of a resolution-dependent grid), or what factor model to simulate and whether to fully revalue (Monte Carlo normal/t/jump) — so a single VaR print is a modelling artifact, not a physical constant of the portfolio, and the portfolio's own design choice (running all three and reading their disagreement, per `docs/METHODOLOGY.md` §2 and the desk guide's "morning pack") treats that plurality as the actual risk signal rather than a nuisance to be averaged away.


## Round 7 — Expected Shortfall, Backtesting & Basel

**Q1. VaR already tells you the loss threshold at a confidence level. Why does this portfolio also compute Expected Shortfall?**

VaR only answers "how bad, at worst, one time in `alpha`" — it says nothing about how bad things get *beyond* that threshold. ES answers exactly that: it's the average loss conditional on being in the tail past VaR (`expected_shortfall.py`'s module docstring: `ES_alpha = -(1/alpha) * integral_0^alpha Q_u(pnl) du`, "the average loss in the worst `alpha` tail"). Two books can share the same 99% VaR while one has a much fatter tail beyond it — VaR can't distinguish them, ES can. `normal_es`/`student_t_es`/`empirical_es` all satisfy `ES >= VaR` by construction, which is the formal expression of "ES sees what's past the VaR cutoff."

**Q2. What does "ES is coherent and VaR isn't" mean in practice, without going through the subadditivity proof?**

A coherent risk measure is guaranteed to reward diversification: the risk of a combined portfolio is never *more* than the sum of the risks of its parts. ES has this property; VaR in general does not — you can construct portfolios (the FX docstring points at "a VaR subadditivity counterexample built from peg-jump assets" in `tests/test_expected_shortfall.py`) where the combined VaR exceeds the sum of the standalone VaRs, so netting two books together looks *riskier* by VaR than holding them apart. That's a red flag for any measure used to aggregate risk across desks or set diversified limits, and it's the headline reason FRTB moved off VaR.

**Q3. Where does the FX engine's ES estimator differ from "just average the P&L below the VaR line," and why does that distinction matter?**

`fx_var/expected_shortfall.py` uses the Acerbi–Tasche tail-splitting estimator (`_es_from_tail`): it averages the worst losses over *exactly* `1 - alpha` of probability mass, taking only a fractional share of the observation sitting right at the VaR boundary rather than including or excluding it wholesale. The docstring is explicit that the naive "mean of observations beyond VaR" over-weights probability atoms and "can spuriously break subadditivity exactly in the pegged-currency jump case this project cares about" — i.e. the naive estimator can fail the very coherence property that's ES's main selling point over VaR, so getting the tail-splitting right isn't cosmetic.

**Q4. What does Kupiec's proportion-of-failures (POF) test actually check, mechanically?**

It counts exceptions — days where realised P&L breaches the VaR forecast (`exceptions_from_pnl`: `pnl_t < -VaR_t`, strict inequality) — over a window of `n_obs` days, and asks whether that count is statistically consistent with the stated confidence level. `kupiec_pof(n_obs, n_exceptions, alpha)` builds the likelihood ratio `LR_uc = -2 ln[(1-p)^{T-x} p^x / ((1-x/T)^{T-x}(x/T)^x)]` comparing the model's claimed exception probability `p = alpha` against the empirical rate `x/T`, and refers `LR_uc` to a chi-squared(1) distribution to get a p-value. It's a test of unconditional coverage only — it says nothing about whether exceptions cluster in time (that's Christoffersen's job).

**Q5. `docs/VALIDATION.md` flags Kupiec's chi-squared(1) reference distribution as "oversized exactly at the regulatory window." What does that mean quantitatively?**

At `n_obs=250, alpha=0.01` — the Basel backtesting window, expected exception count 2.5 — the chi-squared(1) asymptotic is a poor approximation to the true discrete distribution of the LR statistic under a rare binomial. Computing the *exact* rejection probability by summing `Binomial(250, 0.01)` mass over the LR statistic's rejection region gives roughly **9.5%**, not the nominal 5% the chi-squared reference implies. In plain terms: a genuinely correctly-calibrated model gets flagged "reject at the 5% level" about twice as often as the p-value would lead you to believe, purely from using a windows-and-sample-size combination where the asymptotic approximation is thin.

**Q6. Why is that oversizing described as "a property of the classical test, not a bug in this implementation" rather than something to fix?**

Because the chi-squared(1) reference is what Kupiec's test *is* — it's the standard, universally-recognised convention every regulator and every other implementation uses, and the asymptotic approximation genuinely does converge to the true size as the expected exception count grows (the equity engine's `docs/VALIDATION.md` shows it improving toward nominal as `n_obs` grows past the Basel window). Swapping in an exact small-sample reference distribution would make the numbers non-standard and non-comparable to every other bank's Kupiec p-value — trading one well-understood convention for a nonstandard one that nobody else uses. The right fix here is documentation plus an exact test that pins the effect (`test_kupiec_asymptotic_chi2_reference_is_oversized_at_the_regulatory_window` in both engines), not a different statistic.

**Q7. Does the chi-squared(1) oversizing get better or worse as the backtesting window grows, and what drives it?**

It's driven by the *expected exception count* (`alpha * n_obs`), not `n_obs` in isolation — the equity `VALIDATION.md` verifies this at several `(n_obs, alpha)` pairs. At the Basel window (`n_obs=250`, expected count 2.5) the exact rejection rate is ≈9.5%; the equity engine's docs record it falling to ≈7.1% at `n_obs=500` (expected count 5), and the FX engine's docs record ≈5.5% at `n_obs=1000` (expected count 10), close to nominal. So a longer backtest window with the same `alpha` genuinely tightens the asymptotic approximation — it's a real remedy, just not one Basel's 250-day standard offers.

**Q8. Christoffersen's independence and conditional-coverage tests also use a chi-squared reference (1 df and 2 df respectively). Do they inherit the same small-sample caveat?**

Yes — both `docs/VALIDATION.md` files note that Christoffersen's independence and CC tests "share the same asymptotic chi2 machinery and are subject to the same class of caveat at low exception counts." The exact size wasn't computed for Christoffersen the way it was for Kupiec, though: Kupiec's rejection region depends on a single sufficient statistic (the exception count), which makes an exact binomial calculation tractable, while Christoffersen's LR depends on the full transition-count triple (`n00, n01, n10, n11`) and would need Monte Carlo to pin down exactly.

**Q9. Walk through what the Basel traffic-light zones mean and what a bank actually pays for landing in each one.**

`basel_traffic_light(n_exceptions, n_obs=250)` maps the 250-day, 99% VaR exception count to green (0–4 exceptions), yellow (5–9), or red (10+). Green carries no penalty — capital multiplier stays at the base 3.0. Yellow adds a graduated add-on (`_YELLOW_ADDON`: 0.40 at 5 exceptions up to 0.85 at 9), so the multiplier climbs from 3.40 to 3.85 — the model isn't presumed broken yet, but capital charges rise as a precaution. Red sets the add-on to 1.0 (multiplier 4.0) and carries "a presumption of a flawed model" per the docstring — at that point the regulator expects the bank to explain or fix the model, not just pay more.

**Q10. Where do the 4-exception and 9-exception zone boundaries actually come from — are they arbitrary round numbers?**

No — they're calibrated off the exact `Binomial(250, 0.01)` distribution of exception counts a correctly-specified model would produce. `basel_traffic_light`'s docstring gives the cumulative probabilities: green (≤4 exceptions) covers about 89% of outcomes for a correct model, and red (10+) has only about 0.03% probability. `basel_zone_probabilities` exposes the exact PMF/CDF per exception count so this can be checked directly rather than taken on faith; the equity test suite pins `binom.cdf(4)=0.8922` and `binom.cdf(5)=0.9588` exactly.

**Q11. A point-estimate VaR or ES number with no error bar — what's actually wrong with reporting just that one number?**

It invites false precision. A Monte Carlo VaR of "$1.212m" looks exact, but it's a statistic computed from a finite number of simulated (or historical) scenarios and carries real sampling uncertainty — two runs with different seeds, or two historical windows, will disagree by an amount that depends on `alpha`, the sample size, and the tail shape. Reporting the point estimate alone hides whether a day-over-day VaR move of a few percent is a genuine risk change or just noise in the estimator. That's exactly why both engines compute a standard error alongside the point estimate (`se_var` field on `MonteCarloVaRResult`, `var_standard_error`/`var_standard_error_bootstrap`) and why the convergence tests in `docs/VALIDATION.md` are stated "within 3 SE" rather than as bare tolerances.

**Q12. Describe the KDE-based standard-error method in `fx_var/monte_carlo_var.py` — what does it actually compute?**

`var_standard_error` uses the classical asymptotic order-statistic formula `SE = sqrt(alpha*(1-alpha)/n) / f_hat(q)`, where the denominator is an estimate of the P&L density *at the VaR quantile itself* — a low density there means the quantile is on a flat part of the distribution and small sampling noise in the tail count translates into large noise in the loss level. The FX/equity-C++/equity-Rust flavor of this estimates `f_hat` with a Gaussian KDE (`scipy.stats.gaussian_kde`, Scott's-rule bandwidth); the equity-C++/equity-Rust engines instead get the same density estimate from an order-statistic finite difference. Either way, the method needs *some* estimate of the local density, which is the source of its known weakness.

**Q13. What exactly is that weakness, and how big is it?**

The KDE's bandwidth (Scott's rule) is tuned to fit the *bulk* of the P&L distribution well, not the sparse tail region where VaR/ES actually live — so the density estimate at the quantile is systematically off. Benchmarked against the true sampling variability of the quantile (many independent resamples, averaged over many trials to separate bias from noise), the FX module docstring reports the KDE SE is within ~2% of the truth at `alpha=0.99` with 50,000 scenarios, but at `alpha=0.999` (still 50,000 scenarios) or at `alpha=0.99` with only 2,000 scenarios it *systematically underestimates* the true SE by 9–17%. That's not extra noise, it's a directional bias — "directionally overconfident, not just noisy" — and it hits precisely where a desk is most likely to be operating: deep confidence levels or MC runs sped up with fewer paths.

**Q14. How does the bootstrap standard-error estimator sidestep that problem, and what does it cost instead?**

`var_standard_error_bootstrap` resamples the P&L vector with replacement `n_boot` times and applies the *exact same order-statistic VaR rule* to each resample, then takes the standard deviation of the resulting VaR estimates — no density estimate, no bandwidth choice, distribution-free. The FX docstring reports it's unbiased to ~1–2% across the same benchmark that showed the KDE method's 9–17% bias. The cost is trial-to-trial variance in the SE estimate itself: because it's a Monte Carlo estimate built on top of an already-Monte-Carlo VaR, its own precision depends on `n_boot`, and that variance is only tamed by making `n_boot` "generous" (the module notes `n_boot=500` on 50k scenarios runs in well under a second, so this is cheap to afford).

**Q15. `var_standard_error_bootstrap`'s docstring makes a point of computing the tail rank `idx` once, outside the resampling loop, rather than recomputing per resample. Why does that matter for correctness, not just speed?**

Because the resample rule must apply the *exact same* quantile/rank convention as the original point estimate. Here the uniform-weight tail rank depends only on `(n, alpha)`, not on the data values, so it's identical for the original sample and every resample of the same size — computed once via the shared `_tail()` rank rule and reused. If the bootstrap instead used a different quantile convention (a different interpolation rule, a different rank formula) than `empirical_var` used for the point estimate, the resulting standard deviation would be describing the sampling variability of a *different* statistic than the one being reported — the SE and the number it's supposed to bracket would no longer be talking about the same quantity, silently invalidating any "point estimate ± k*SE" statement built from them.

**Q16. If a desk is running a 99% VaR on 50,000 daily-P&L scenarios, should they bother with the bootstrap SE, or is the KDE estimate good enough?**

By the module's own guidance: no need. The docstring's rule is to prefer the bootstrap "whenever `alpha >= 0.995` or scenario counts are modest" — at `alpha=0.99` with the module's default 50,000 scenarios, the KDE estimate is benchmarked within ~2% of the truth, which is "safe to use as-is." The bootstrap becomes the one to trust specifically once the desk pushes to deeper confidence levels (99.5%, 99.9% — FRTB's own 97.5% ES tail region, or stress alphas) or cuts scenario counts for speed, which is exactly where the KDE method's bias shows up.

**Q17. Same question, but the desk wants ES97.5 off a 250-day historical window rather than an MC VaR — does the SE story change?**

The bias mechanism named in the MC docstring doesn't directly apply to a purely historical ES (there's no KDE density-at-a-quantile step there), but the small-sample problem gets *worse*, not better: ES averages over the whole `alpha` tail rather than reading off a single order statistic, so its estimation error exceeds VaR's at the same alpha (`es_standard_error_bootstrap`'s docstring makes this point explicitly). A 250-day window at `alpha=0.025` has only about 6 points in the tail being averaged — `docs/VALIDATION.md` calls this out directly: "a 250-day ES₉₇.₅ averages ~6 points — quote it with error bars." The equity engine's bootstrap SE test (`test_es_se_larger_for_smaller_alpha`) formalizes that ES's bootstrap SE is larger than VaR's bootstrap SE on the same sample.

**Q18. Why did FRTB replace 99% VaR with 97.5% ES as the headline market-risk capital measure, given ES's SE is worse at a fixed alpha?**

Two separate properties are in play and FRTB is trading one for the other. ES is coherent (subadditive) where VaR in general is not (Q2), and ES captures tail *severity* — how bad losses get beyond the threshold — where VaR by construction is blind to everything past its own cutoff (Q1). Moving the confidence level down to 97.5% (from 99%) is precisely what keeps the *tail sample size* reasonable when averaging rather than reading a single quantile — more like ~6 points on a 250-day window as noted above, rather than the ~2.5 expected exceptions VaR99 would have. So FRTB accepts a somewhat noisier per-day statistic (ES vs VaR at a fixed alpha) in exchange for a measure that behaves better under aggregation and actually reflects tail severity — and picks the alpha level, in part, to keep that noisiness manageable.

**Q19. What does the Acerbi–Szekely Z2 test add on top of Kupiec/Christoffersen, and why was it chosen for backtesting ES specifically?**

Kupiec and Christoffersen only ever look at the exception *indicator* (breached or not) — they're VaR-only tests and say nothing about whether the ES number itself was a good estimate of tail severity. `acerbi_szekely_z2` instead sums, over exception days, the ratio of realised P&L to the model's ex-ante ES (`Z2 = (1/T) sum_t [pnl_t * I(pnl_t < -VaR_t) / (alpha*ES_t)] + 1`), so a materially negative Z2 means realised tail losses are running worse than the model's ES claimed. The module docstring says it was chosen over an exception-severity z-test "because it uses every exception's magnitude relative to the ex-ante ES and needs no normality assumption" — it uses full exception magnitudes, not just a count.

**Q20. The Z2 test flags `reject = Z2 < -0.70` — where does -0.70 come from, and is it exact like Kupiec's p-value?**

It's an empirical benchmark, not a closed-form critical value: `backtesting.py`'s docstring notes Acerbi–Szekely report the 5% critical value is approximately -0.70 "across realistic P&L distributions," and the code flags rejection at that threshold as a "documented approximation." An exact p-value would require simulating under the specific model being tested, which the implementation deliberately doesn't do here (referenced to `docs/METHODOLOGY.md`) — so unlike Kupiec's LR-to-chi2 p-value, Z2's -0.70 cutoff is a reasonable rule of thumb pulled from the literature rather than an exact reference distribution for this particular P&L series.

**Q21. What does Christoffersen's independence test catch that Kupiec's POF test misses entirely?**

Kupiec only checks the *total* exception count against the expected rate — ten exceptions scattered randomly across 250 days and ten exceptions clustered in one bad week score identically under Kupiec. `christoffersen_independence` instead builds a first-order Markov transition count (`n00, n01, n10, n11` — exception yesterday vs today) and compares the Markov likelihood against the i.i.d. one; clustered exceptions inflate `n11` (exception-follows-exception) and the LR rejects. This matters operationally because clustering is the signature of a model that's slow to react to a genuine regime change (e.g. vol clustering under GARCH) rather than one that's simply miscalibrated on average — `exception_cluster_table`'s docstring notes gaps between exceptions should be roughly geometric with mean ~100 days under a correct i.i.d. 99% model, and a run of small gaps is the visual signature this LR test formalizes.

**Q22. Walk through what `rolling_var_backtest` actually simulates, and why it uses a walk-forward design rather than one static fit.**

It re-estimates VaR out of sample: for each day `t` past the initial `window`, it calls the supplied `var_fn` on only the trailing `window` days of P&L history (`p[t-window:t]`), producing one VaR forecast per day, then compares each forecast against the *next* day's realised P&L via `exceptions_from_pnl`. This mimics how a real desk operates — the model never gets to see the day it's forecasting — which is the only way a backtest can honestly measure predictive coverage rather than in-sample fit. `BacktestResult.summary()` then runs Kupiec, both Christoffersen tests, and (at `alpha=0.01`) rescales the observed exception count to a 250-day-equivalent to report the Basel zone even when the actual window isn't exactly 250 days.

**Q23. The documented equity engine backtest (500-day rolling, 99%, GARCH+regime data) shows parametric-normal landing Basel red while filtered historical simulation (FHS) lands green. What's driving that gap?**

Parametric-normal produced 14 exceptions against an expected 5, Kupiec p=0.0009, independence p=0.0002 — badly miscalibrated *and* clustered, landing red (multiplier 4.00). FHS produced 7 exceptions, all tests p>0.05, green (3.00). The difference is that FHS conditions its VaR on the *current* volatility regime (it's explicitly noted elsewhere as "conditional": same unconditional sample, turmoil at the end vs. the start gives a higher VaR), while the parametric-normal model doesn't adapt fast enough to the regime data's volatility clustering — its exceptions cluster in the high-vol periods precisely because its VaR estimate lags the regime. This is the practical case Christoffersen's test (Q21) is built to catch, and the traffic-light system (Q9) is built to price.

**Q24. `es_standard_error_bootstrap` and `var_standard_error_bootstrap` both resample with replacement and recompute the estimator on each resample — are they otherwise the same procedure?**

Structurally yes — both draw `n_boot` resamples of the same size as the original P&L array (with replacement), recompute the point-estimate function (`expected_shortfall` or the order-statistic VaR rule) on each, and report the standard deviation across resamples (`ddof=1`). They differ in what's being resampled *around*: VaR's bootstrap SE only has to track one order statistic per resample, while ES's has to re-average the whole tail-sum (`sum_{i<=k} x_(i) + frac * x_(k+1)`) each time, which is part of why ES's bootstrap SE comes out larger than VaR's bootstrap SE at the same alpha and sample (Q17) — more of the resample's randomness feeds into the final number.

**Q25. Both engines validate their Kupiec implementation against hand-computed numbers rather than only checking asymptotic behaviour. Why does that matter here specifically?**

Because the finding in Q5–Q7 is about the *reference distribution*, not the LR statistic's arithmetic — so the test suite needs to separately pin down that the LR formula itself is exactly right (equity: `Kupiec LR vs independent hand computation, 5 (T,x) pairs; LR(250,0) = -2*250*ln(0.99)` to 1e-10; FX: `Kupiec LR(n=250, x=5, p=1%) = 1.9568` to 1e-12 vs hand formula) before asking any question about how well chi-squared(1) approximates its null distribution. Getting the LR formula numerically exact and then separately characterizing the chi-squared reference's known small-sample bias are two different validation claims, and conflating them would make it unclear whether a documented discrepancy is a code bug or the well-known asymptotic-approximation property it actually is.

**Q26. What happens at the Kupiec test's degenerate boundaries — zero exceptions, or every day an exception — and why does that need special handling rather than just erroring?**

Zero exceptions is a completely legitimate outcome for a well-calibrated model on a short window (its probability is exactly `(1-alpha)^n_obs`, e.g. `binom.pmf(0, 250, 0.01) ≈ 8.1%`), so the LR formula has to evaluate cleanly there rather than raising on a `log(0)` term. `kupiec_pof` handles this via the `0 * ln 0 = 0` convention (`_ll` only adds the `x*log(p)` term when `x > 0`, and the `t-x` term only when `t-x > 0`), and it's exercised directly by `test_zero_exceptions_known_value` and the "x=0 or x=T" edge-case row in both `docs/VALIDATION.md` tables — the same convention appears in `christoffersen_independence`'s `_xlogy` helper for its own degenerate transition counts.

**Q27. Suppose a Kupiec test on a 250-day, 99% VaR backtest comes back with p=0.04 — reject at 5%. Given everything above, how should a desk actually read that number?**

With real caution rather than a flat "the model failed." Because the true rejection rate of a nominally-5%-size Kupiec test at this exact window is ≈9.5% (Q5) — roughly double the nominal rate — a p-value just under 0.05 is landing in a region where a genuinely correctly-calibrated model gets flagged this often almost twice as frequently as the asymptotic reference suggests it should. That doesn't make the test useless (it's still the industry-standard, regulator-recognised statistic, per Q6), but it means a marginal reject at this specific window/alpha combination is weaker evidence of a broken model than the same p-value would be at a longer window (Q7), and it's exactly the kind of caveat that belongs alongside the number, not baked into a different, nonstandard test.


## Round 8 — Fixed Income Pricing & Risk

**Q1. Why does a bond's price move inversely with yield, and where does that fall out of the code?**

A bond's price is the sum of its future cashflows discounted at the yield: in `fi_rates/bond.py::price_from_ytm`, dirty price is `sum(cf_i * (1 + y/m)^(-exp_i))`. Raising `y` raises every discount factor's exponent base, shrinking every `(1+y/m)^(-exp_i)` term, so price falls monotonically as yield rises — this is unit-tested as a structural invariant (price monotonicity in yield) rather than just checked pointwise. The relationship is convex, not linear, which is exactly why `price_from_ytm` can be inverted by `ytm_from_price` with a single-bracket Brent search: the price-yield curve only crosses any target once.

**Q2. What does Macaulay duration actually measure, as implemented here?**

`macaulay_duration` in `fi_rates/risk.py` computes `D_mac = (1/P) * sum(t_i * cf_i * (1+y/m)^(-exp_i))` — the present-value-weighted average time (in years) until each cashflow is received. It is a time measure, not a price-sensitivity measure by itself: the docstring notes it equals the weighted average cashflow time, and the test suite pins `macaulay_duration(zero_coupon_bond) == maturity` exactly, since a ZCB has only one cashflow and its weighted-average time is trivially its own maturity.

**Q3. How does modified duration differ from Macaulay duration, and what does it measure?**

`modified_duration` is `macaulay_duration / (1 + y/m)` — the code implements this as a one-line division in `risk.py`. Macaulay duration is a time (years); modified duration is a price sensitivity: the docstring states `D_mod = -(1/P) dP/dy`, the percentage price change per unit change in yield. VALIDATION.md pins the identity `D_mod == D_mac/(1+y/m)` to ≤1e-14 and separately verifies `D_mod` against central finite differences of the actual pricer (`numerical_modified_duration`) to ≤1e-8 relative — so the analytic formula is cross-checked against brute-force bumping, not just algebra.

**Q4. Why isn't duration alone enough to estimate a price move for a large yield shock?**

Duration is a first-order (linear) Taylor term; the price-yield relationship is convex, not linear, so a straight-line duration estimate systematically deviates from the true curve as the shock size grows. `pnl_approximation_table` in `risk.py` builds exactly this comparison — `duration_only = -D_mod * P * dy` versus `full_repricing = price_from_ytm(ytm+dy) - price_from_ytm(ytm)` — across a shock grid from -200bp to +200bp, and VALIDATION.md §4.4 shows the duration-only error growing from -0.014 at 100bp to -1.664 at 200bp on a 10y government bond.

**Q5. What is convexity, and how is it computed here?**

`convexity` in `risk.py` computes the analytic second derivative `C = (1/P) d2P/dy2 = (1/P) sum(cf_i * n_i*(n_i+1)/m^2 * (1+y/m)^(-n_i-2))`, where `n_i` are the period exponents from `_period_exponents`. It is the curvature of the price-yield relationship — the second-order correction that a linear duration estimate misses. Like modified duration, it's cross-checked against `numerical_convexity` (a central second difference of the pricer) to ≤1e-6 relative per VALIDATION.md.

**Q6. Why does duration alone underestimate price gains and overestimate price losses for large yield moves?**

For a bond with positive convexity, the true price-yield curve lies above its tangent line at every point away from the tangent point — a straight line from the tangent will always sit below the actual (convex) curve. `pnl_approximation_table` demonstrates this concretely on a 10y bond: at -200bp the duration-only estimate misses +17.746 of full repricing by -1.664 (understating the gain), and at +200bp it misses -14.635 by -1.448 (overstating the loss, i.e. predicting a bigger drop than actually happens). Adding the 0.5·C·P·dy² convexity term in `duration_convexity` cuts the 100bp error by roughly 30x, and this is enforced as an inequality (not a magic-number check) across the whole shock grid so the sign relationship — dur+conv always closer than dur-only for a positive-convexity bond — is what's actually tested.

**Q7. What is DV01, and how does the project compute it two different ways?**

DV01 ("dollar value of a basis point") is the price change for a 1bp yield move, in currency units. The analytic version, `dv01()` in `risk.py`, is `D_mod * P * 1e-4` — a closed-form scaling of modified duration. The curve-based ("effective") version, `dv01_curve()`, instead bumps every pillar zero rate in `DiscountCurve` by ±1bp via `curve.bumped_parallel(bump)` and fully reprices: `DV01_eff = (P(-h) - P(+h)) / 2`. METHODOLOGY.md §4 notes they deliberately are not made to agree exactly — they differ by the Jacobian `dy/dz ≈ (1 + y/m)` between periodic-yield space and continuous-zero space, and the test suite asserts precisely that relationship rather than treating the two numbers as interchangeable.

**Q8. Why is DV01 the number a rates desk actually uses day to day, rather than duration or convexity alone?**

DESK_GUIDE.md's 07:15 EOD risk step runs `portfolio_risk`, which reports DV01 per position and in aggregate; DV01 is what traders check against limits and what risk managers aggregate, because — unlike modified duration or YTM — it is a plain currency amount that sums additively across positions of different price levels, coupons and maturities. VALIDATION.md §4.7 makes the point sharply: for a duration-neutral long/short book with net market value ≈ 0, market-value-weighted duration and convexity become undefined `0/0` (`portfolio_risk` emits a `ZeroNetValueWarning` and reports `NaN` for those columns), but `dv01` — because it's a plain sum, not a weighted average — remains exact throughout and is, per the docs, "the correct aggregate for such a book."

**Q9. What day-count convention does the curve engine use for mapping dates to time, and does it match the portfolio's stated equity convention?**

`fi_rates/bond.py::curve_time` maps a calendar date to curve time as `year_fraction(settlement, date, "ACT/365F")` — actual days divided by 365, fixed — matching the ACT/365F convention that `CONVENTIONS.md` states as the portfolio-wide default for equity-style day counting. `DiscountCurve` pillars are likewise stated (in `curve.py`'s module docstring) to live "in years (ACT/365F from the valuation date when built from dates)". This is a separate choice from the bond's own accrual day count (see Q10) — ACT/365F is specifically the date→time mapping onto the curve, not the coupon accrual convention.

**Q10. Why do day-count conventions matter for getting accrued interest right, and what conventions does this project support?**

`fi_rates/daycount.py` implements four conventions — `ACT/365F`, `ACT/360` (money-market, used for deposits/FRAs), `30/360US` (the default bond accrual convention, without the end-of-February special rule), and `ACT/ACT-ISDA` (calendar-year-split, used for Treasuries in this codebase). `accrued_interest()` computes `bond.face * bond.coupon * year_fraction(period_start, settlement, bond.daycount)` — so a mismatched convention directly misprices accrued interest and therefore the dirty price. The module docstring documents this as a real simplification: `30/360US` here omits the Feb-EOM rule, and the code uses ACT/ACT **ISDA** rather than the period-based ACT/ACT **ICMA** that actual US Treasuries use — assumption 5 in METHODOLOGY.md states explicitly that Treasury stub-period accruals will differ by a day or two from production as a result.

**Q11. How is the yield curve actually built in this project — bootstrapped from market instruments, or fit globally?**

`fi_rates/bootstrap.py::bootstrap_curve` performs a **sequential bootstrap**: instruments (deposits, FRAs, par swaps) are sorted by pillar maturity, and each new pillar's discount factor is solved with Brent's method (`brentq`, bracket `(1e-8, 20)` on the DF) so that instrument reprices exactly off the partially built curve — deposits close-form as `P(T) = 1/(1+r·T)`, FRAs as `P(T2) = P(T1)/(1+r·(T2-T1))`, and par swaps via the pillar condition `r·Σα_i P(t_i) + P(T) = 1`. `bootstrap_bond_curve` does the analogous sequential scheme off coupon-bond dirty prices. METHODOLOGY.md §1 explicitly rejects a global parametric fit (Nelson–Siegel/Svensson) for this purpose, because a trading desk needs to reprice its own hedge instruments exactly — NS/Svensson leaves 1-10bp residuals, which "creates phantom P&L and phantom DV01 on every mark."

**Q12. What interpolation choices does `DiscountCurve` offer, and why does the choice matter beyond just smoothness?**

Three schemes share one code path (`curve.py`): `loglinear_df` (default — linear in ln P(t), equivalent to piecewise-constant instantaneous forwards), `linear_zero` (linear in the zero rate), and `pchip_zero` (monotone cubic on zeros). The choice is exposed as a constructor argument rather than hard-coded because it's a risk decision, not plumbing: VALIDATION.md §4.1 shows actual computed forward rates around the 10y/15y/20y pillars where `linear_zero` produces a sawtooth (a 22.4bp jump down at each pillar as the forward resets), `loglinear_df` jumps only *at* pillars (up to 29bp, but flat and unbiased between them), and `pchip_zero` is continuous (max ~2bp jump from discrete differencing) at the cost of being non-local.

**Q13. Why does locality of the interpolation scheme matter for hedging, specifically?**

With `loglinear_df` or `linear_zero`, both **local** schemes, bumping one market quote (say the 10y swap) only moves discount factors between its neighboring pillars — so a bucketed DV01 report stays clean and a hedge trade in the 10y point doesn't leak risk into the 2y bucket. `pchip_zero` is **not** local: VALIDATION.md §1.1 measures that appending a new pillar under PCHIP reshapes the monotone cubic over already-solved earlier segments, so previously-repriced quotes drift by up to 3.6e-06 in rate terms — small, but enough that a PCHIP-built curve "cannot pass a 1e-10 repricing gate" that production build controls typically require (DESK_GUIDE.md §5).

**Q14. What happens when the bootstrap is given a quote that can't be repriced, or when instruments are supplied out of order?**

`bootstrap_curve` raises `ValueError` naming the instrument, the pillar, and the quote when no discount factor in the admissible bracket `(1e-8, 20)` can reprice it — e.g. a deposit rate below -1/T, or (for `bootstrap_bond_curve`) a bond price above the sum of its cashflows' maximum PV. Order independence is a separate, explicitly tested property: VALIDATION.md §1.1 states bootstrapping is verified order-independent — shuffled instrument input reproduces identical pillar discount factors to 1e-15 — because `bootstrap_curve` internally sorts by `pillar` before solving.

**Q15. What would key-rate duration (or partial DV01) add beyond a single parallel-shift duration number — and does this project implement it?**

Yes — `fi_rates/keyrates.py` implements key-rate DV01s and durations at configurable tenors (`DEFAULT_KEY_TENORS = (2, 5, 10, 30)` years) using **triangular bumps**: `triangle_weights` builds a weight function that peaks at 1 at its own key tenor and decays linearly to 0 at the neighboring key tenors, flat-extending at the ends, forming a partition of unity across every curve pillar. `key_rate_dv01s` then central-differences full revaluation under each triangular pillar-bump. A single parallel duration/DV01 number only measures sensitivity to a uniform shift of the whole curve; the KRD ladder decomposes that into sensitivity at each point on the curve — which matters whenever the curve doesn't move in parallel (see Q17).

**Q16. How does the KRD ladder relate mathematically to the single parallel DV01 number, and how exactly do they reconcile?**

Because the triangular weights form a partition of unity (unit-tested to 1e-14), summing the key-rate bumps reproduces exactly a 1bp parallel bump — but the *sum of the resulting KRDV01s* only matches the parallel DV01 "up to tolerance," not exactly, for two documented reasons (VALIDATION.md §3): (1) finite-difference cross-gamma, since PV isn't linear in the pillar zeros, and (2) non-local interpolation reshaping neighboring buckets under PCHIP. On the sample portfolio (DV01 ≈ 980.3/bp), the measured relative residual is 5.3e-08 under `loglinear_df` versus 5.5e-07 under `pchip_zero` — three orders of magnitude apart, which the docs cite as a reason production risk systems conventionally use a local interpolator.

**Q17. Why does the desk guide say hedging the total DV01 with one instrument "leaves the book flat to parallel moves but fully exposed to twists"?**

VALIDATION.md §4.3 quantifies this directly with a full-revaluation-vs-duration-and-convexity comparison: a `steepener_50bp` scenario produces a true P&L of -4,345 against a duration+convexity-proxy estimate of -12,296 — a 183% error, "duration proxy useless" per the doc's own annotation — because a single duration number compresses the whole curve into one factor and cannot see that the curve twisted rather than shifted in parallel. `krd_report` in `keyrates.py` is the fix: DESK_GUIDE.md's sample book shows a 2y/5y/10y/30y KRD ladder (154.3/278.0/368.4/179.5 DV01) that sums to the parallel 980.2 DV01 to 5e-8 relative, and hedging *that* ladder — not the aggregate number — is what actually flattens exposure to a steepener or flattener.

**Q18. Why does this project use a single self-discounting curve rather than the post-2008 multi-curve/OIS framework, and what breaks as a result?**

METHODOLOGY.md's assumptions register lists this as "the big one": one curve both projects forwards and discounts, so the par-swap condition assumes the floating leg is worth par on that same curve. This has not been production standard since 2008 — real desks discount on OIS (SOFR/ESTR) while projecting forwards off separate tenor curves. What breaks: with a wide OIS-IBOR-style basis, single-curve swap PVs are wrong by roughly basis × annuity (tens of bp of upfront in 2008-2012), the FRN price-equals-par identity breaks by the discounting/projection basis, and DV01 that should split across curves doesn't. The doc notes the architecture (curve objects + root-solved bootstrap) extends to multi-curve by adding a second curve and re-deriving the par condition, but it's deliberately left out to keep the project reviewable.

**Q19. How does the curve behave beyond its last pillar, and what's the documented risk?**

Both `curve.py` and VALIDATION.md §4.2 are explicit: beyond the last pillar, the zero rate is held flat, and **every query in that region emits an `ExtrapolationWarning`** — deliberately not suppressible by configuration, so callers must handle it consciously (DESK_GUIDE.md lists this as a control: warnings are treated as errors in the CI test profile except where explicitly expected). The risk is concrete, not academic: discounting a 40y cashflow off a 30y-last-pillar curve embeds an arbitrary flatness assumption directly into PV, and METHODOLOGY.md assumption 9 estimates a 10bp long-end slope error at 40y is worth about 0.4% of PV — material for pension/insurance liabilities.

**Q20. How does the project handle zero or negative rates — is that an edge case that's broken, or supported?**

It's an explicitly supported, unit-tested path, not an error. `DiscountCurve` accepts discount factors greater than 1 (the docstring notes DFs "may exceed 1 for negative-rate curves"), and `price_from_ytm` only requires `1 + ytm/frequency > 0`, so negative periodic yields work as long as that inequality holds — METHODOLOGY.md §3 states this is unit-tested, including that "negative yields work for `1 + y/m > 0`." VALIDATION.md §4.6 further lists "negative-rate curves (DFs > 1)" among the "degenerate but valid inputs" tested as *working* paths, alongside the four curve variants (upward, inverted, flat, negative) that the bootstrap repricing tests in VALIDATION §1.1 are run against.

**Q21. What happens at the extremes of yield — very high or deeply negative — and does the pricer hold up?**

`test_edge_cases.py::TestExtremeYields`, cited in VALIDATION.md §1.2, round-trips YTM at 500% and at -1% back through `price_from_ytm` → `ytm_from_price` → price to 1e-10 relative tolerance. Mechanically this works because `ytm_from_price`'s Brent bracket in `bond.py` actively expands the upper bound (doubling `hi` until the price function brackets the target, capped at 1e4) and fixes the lower bound at `-0.999 * frequency` to stay inside the `1 + y/m > 0` domain, raising an informative `ValueError` if a price is unreachable at either bound (e.g. a price at or below the theoretical maximum-yield floor).

**Q22. Does the project document behavior on an inverted yield curve, and is it tested the same way as a normal upward curve?**

Yes — VALIDATION.md §1.1 states the bootstrap repricing tests (max error 4.0e-16 against a 1e-10 tolerance) are run "for all four curve variants (upward, inverted, flat, negative)," and structural invariants in `test_properties.py` (per VALIDATION §5) pin curve identities and bond duration/convexity ordering "across yields from -1% to +12%" without special-casing the curve's shape. DESK_GUIDE.md's 2022 hiking-cycle scenario (a bear-flattening/inverting move: 2y +370bp vs 10y +235bp) is the concrete real-world case: the desk-guide lesson is explicitly that short-end key-rate durations, "usually ignored," dominated the P&L in that episode, and the duration+convexity Taylor proxy still missed the true loss by 13% (VALIDATION §4.3) because full revaluation, not a single duration number, was needed to capture the twist.

**Q23. What does "negative convexity" mean, and why is it explicitly out of scope here?**

Assumption 7 in METHODOLOGY.md states the project assumes deterministic rates with no optionality; VALIDATION.md §4.5 explains that callable bonds and MBS exhibit *negative* convexity as rates approach the call/refinancing region — price appreciation caps out and duration actually *shortens* into a rally, the opposite of the positive-convexity behavior demonstrated in Q6. Nothing in `risk.py` models this, so applying `duration`/`convexity`/`dv01` to a callable would overstate upside and misdirect a hedge — the docs note the 30x Taylor-error improvement from adding convexity (Q6) "would reverse sign" for a negatively convex instrument. Pricing that correctly requires an option-adjusted-spread model with an interest-rate lattice or simulation, which the docs flag as explicitly out of scope, with the recommendation to "route to an OAS model" (DESK_GUIDE.md §4).

**Q24. Does the FX fixed-income project add anything to the rates picture, and if so what?**

`python/fx/04-fixed-income` (`fx_rates`) bootstraps one discount curve per currency (deposits + annual par swaps, same log-linear-DF machinery) and then prices FX forwards via covered interest parity, `F(T) = S * DF_f(T) / DF_d(T)`, where domestic = quote currency and foreign = base currency per `CONVENTIONS.md`'s BASE/QUOTE rule. Since 2008, market FX forwards systematically deviate from single-curve CIP, so `fx_rates` adds a maturity-dependent **cross-currency basis spread** `s(T)` applied to the foreign curve (`DF_f_adj(T) = DF_f(T) * exp(-s(T)*T)`), backed out from market forwards or basis-swap quotes via `bootstrap.implied_basis_from_forwards` / `bootstrap.curve_from_fx_forwards`. METHODOLOGY.md's own worked example shows the size of the effect: ignoring basis misprices the 5y EURUSD forward by 147 pips, a fictitious -$1.20m PV on a EUR 100m forward at the true market rate.

**Q25. How does the FX project's risk framework extend DV01 and KRD to a cross-currency setting?**

Per `fx_rates` METHODOLOGY.md §1.5: DV01 is computed per currency as a central-difference parallel ±1bp bump of *that* currency's zero pillars, and key-rate duration bumps one pillar at a time — locality of the `loglinear_df` interpolation is exactly what makes that bump local, the same locality argument as the equity/rates project (Q13). On top of ordinary rates DV01, the FX project adds a **basis DV01** — a ±1bp bump of the whole cross-currency basis spread curve with full curve rebuild — kept as a separate risk factor from rates DV01 precisely so that interest-rate risk and basis risk are reported separately rather than conflated, per METHODOLOGY.md §1.3's note that "all pricing uses the adjusted foreign curve; the pure foreign curve is retained so that interest rate risk and basis risk are reported as separate factors."


## Round 9 — Statistical Pairs Trading

**Q1. Two stocks have a 0.92 return correlation. Why isn't that enough to put on a pairs trade?**

Return correlation only says the two names move together *day to day*; it says nothing about whether the *level* of one relative to the other is bounded. Two independent random walks can have high return correlation while their price levels drift apart without limit — there is no force pulling the spread back to a mean, so a "pairs trade" on them is really just a directional bet with extra steps. The equity project's own selection funnel demonstrates this directly: the trap pairs in the seed-7 panel have return correlations of 0.919–0.923 — as high as the true cointegrated pairs — yet fail the Engle-Granger test (EG stats −1.45 to −2.13 vs a −3.34 critical value) and must be rejected (`docs/VALIDATION.md` §4).

**Q2. What does cointegration add that correlation doesn't?**

Cointegration asks a different question: is there a linear combination of two *non-stationary* (I(1)) price series that is itself *stationary* (mean-reverting)? Correlation measures co-movement of changes; cointegration certifies that a specific combination of the *levels* — `y_t − β x_t`, the spread — has a fixed mean and finite variance it keeps returning to. That stationary combination is exactly what a pairs trade needs: a tradeable object with a defined equilibrium to bet on reverting to.

**Q3. Which cointegration test does this codebase actually implement — read the code, don't assume?**

`src/eq_pairs/cointegration.py` implements the **Engle-Granger two-step test from scratch**: step 1 is an OLS cointegrating regression `y_t = α + β x_t + u_t` (function `hedge_ratio`/`ols`); step 2 runs an **augmented Dickey-Fuller test** on the fitted residuals û (function `adf_test`, called from `engle_granger` with `regression="n", n_series=2`). The FX project (`fx_pairs/cointegration.py`) mirrors the identical EG-from-scratch design on log rates. Both are cross-validated against `statsmodels.tsa.stattools.adfuller` to 1e-8 (`test_fixed_lag_stat_matches`, `test_autolag_aic_matches`).

**Q4. What is the null hypothesis of the ADF test run on the EG residuals, and what does rejecting it mean?**

The ADF regression is `Δû_t = ρ û_{t-1} + Σᵢ φᵢ Δû_{t-i} + ε_t` (no constant, since the step-1 intercept already makes û mean-zero); the statistic is the t-value of ρ̂ (`ADFResult.stat`). H0 is **ρ = 0**, i.e. û has a unit root — the residual spread is itself non-stationary, so there is *no* cointegration. Rejecting H0 (`EGResult.cointegrated()` returns `adf.reject(level)`, i.e. `stat < crit[level]`) means the spread is stationary, i.e. the pair is cointegrated.

**Q5. Why can't the step-2 ADF statistic be compared against ordinary ADF critical values?**

Because û isn't an observed series — OLS in step 1 chose (α, β) specifically to make the residuals look as stationary as possible, which biases the ADF statistic toward rejection. The correct comparison uses MacKinnon's response surfaces indexed by N, the number of I(1) series in the cointegrating regression: N=1 is plain ADF (5% ≈ −2.86), N=2 is the Engle-Granger surface for a two-variable regression (5% ≈ −3.34, stricter). `cointegration.py`'s own docstring calls this "the classic mistake," and `test_spurious_rejection_rate_close_to_size` proves it isn't hypothetical: on 200 replications of independent random walks, the correct N=2 values reject at the nominal ~5%, while using N=1 values on the same statistics rejects more than 1.5× as often — false "discoveries" of cointegration that aren't there (`docs/VALIDATION.md` §3).

**Q6. How is the hedge ratio estimated, and why an OLS regression rather than just using the two prices directly?**

`hedge_ratio(y, x, intercept=True)` runs OLS `y_t = α + β x_t + u_t` on price *levels* (in dollars for equities, log rates for FX) — β is literally the EG step-1 regression coefficient, and the residual `u_hat` is the tradeable spread. β̂ is "super-consistent" under cointegration (converges at rate T, faster than the usual √T). The intercept is included by default because share price levels are arbitrary (a stock split changes the level, not the economics) — forcing the line through the origin would misspecify the relation (`METHODOLOGY.md` §1, "Hedge ratio: intercept, and static vs adaptive").

**Q7. Why does trading in ratio β:1 (not 1:1 dollars) make the combined position market-neutral, when 1:1 doesn't?**

Market-neutrality means the position's P&L doesn't move with a common shock to both legs. If the two legs don't move one-for-one per dollar (β ≠ 1) — which is the generic case — a naive equal-dollar 1:1 hedge still carries residual exposure to whatever the "1 share of y moves β× as much as 1 dollar of x" relationship implies. Sizing in the cointegrating ratio (`size_positions(..., mode="beta")`: `q_x = −β q_y`) makes the position track the *spread itself*, so a parallel move in both legs' common factor cancels exactly by construction — it's the ratio the regression estimated, not an assumed 1:1, that neutralizes the shared factor. The dollar-neutral mode (`mode="dollar"`, equal gross on each leg) is simpler and exactly dollar-neutral at entry but *deviates* from the cointegrating ratio, leaving small residual exposure — a trade-off the code documents explicitly in `signals.py`'s `size_positions` docstring.

**Q8. What's the z-score entry/exit framework, and what are the actual default parameters this project's backtest uses?**

`signals.py`'s `generate_signals` runs a state machine over a z-score series with `SignalRules` defaults: **enter** when `|z| ≥ entry_z = 2.0`; **exit** when z reverts through `exit_z = 0.0` (full mean touch); **hard stop** when `|z| ≥ stop_z = 4.0`; optional **time stop** at `max_holding` bars, typically set via `time_stop_bars(half_life, k=3.0)` = ⌈3 × half-life⌉ bars capped at 252. z>0 (spread rich) triggers a short-the-spread entry; z<0 triggers long. Positions are quoted in {−1,0,+1} spread units.

**Q9. Two z-score construction methods appear in the code — what are they, and why does one avoid a warm-up period?**

`zscore_rolling(spread, window)` is the standard rolling-window z-score `(s − rolling_mean)/rolling_std` — non-parametric but needs `window` observations to warm up (NaN before that), and it conflates the spread's *level* deviation with its *speed* of reversion. `zscore_ou(spread, ou)` instead uses the OU model's stationary distribution, `z = (s − μ)/(σ/√(2κ))` — since μ and the stationary variance are model parameters (not rolling statistics), there's no warm-up period, but it treats them as frozen and, in walk-forward use, they must come only from the formation window (`signals.py` docstring).

**Q10. Why "re-entry arming" — what problem does it solve, and how large is its measured effect?**

Without it, a hard stop at `|z| ≥ 4` would be immediately followed by re-entry at `|z| ≥ 2` on the very next bar — the state machine would re-fight the exact trade the stop just cut, since the spread is still extreme. The fix (`generate_signals`, the `armed` flag): after *any* exit, the pair cannot re-enter until `|z|` first drops back inside the entry band. This is what caps the loss in the project's regime-break case study — with stops and arming, the post-break loss is **−$20,963** (one entry, then the stop fires and arming blocks re-entry while z stays extreme); without any stop-loss/time-stop, the same book loses **−$715,158**, averaging into a spread that never reverts (`docs/VALIDATION.md` §9).

**Q11. Why is in-sample cointegration necessary but not sufficient for a live pairs trade — what does VALIDATION.md's regime-break case study show?**

A1 in the assumptions register states it plainly: cointegration found in the formation window must *persist* into the trading window, and nothing in-sample warns you if it won't. The case study plants an OU spread (true half-life 11.6 days) for 750 days, then switches it to a drifting random walk. Fitted on the pre-break data alone, the fit looks *good* — EG stat −5.01, estimated half-life 10.0 days — indistinguishable from a genuinely stable pair. Post-break: +$81,497 pre-break P&L becomes −$20,963 with stops/arming, or **−$715,158** without them. The desk-level causes named for this kind of break are regime change, mergers (a cash-target leg pins to the offer price so κ collapses), and fundamental shifts (`docs/DESK_GUIDE.md` §5, "Merger breaks a pair").

**Q12. Beyond the regime-break study, what other failure mode does VALIDATION.md flag about trusting an OU fit alone (assumption A10)?**

On a *pure random walk*, OLS on the discretised AR(1) returns `b` slightly below 1 due to finite-sample bias — so `OUFit.mean_reverting` comes back **True** with a spurious half-life of 300–950 days across seeds (`test_an_ou_fit_alone_cannot_reject_a_random_walk`). The `mean_reverting` flag only trips exactly at `b ≥ 1`, so an OU fit by itself gives *no* protection against trading a random walk. This is the concrete reason the funnel gates on the Engle-Granger test (with correct N=2 critical values) *before* any OU fit is trusted, plus a hard cap on the accepted half-life — an estimate in the hundreds of days is the tell.

**Q13. There's also a documented bias in the half-life estimate itself (A9) — what direction, and what's the desk consequence?**

OLS on the discretised AR(1) inherits the Dickey-Fuller/Kendall small-sample bias: κ is systematically **under-estimated**, so estimated half-lives come out **too long** — measured at −20.5% bias in κ at true κ=0.01 (true half-life 69.3 days, estimated 87.2), shrinking to −3.5% at κ=0.20 (`docs/VALIDATION.md` §9, `test_ols_kappa_is_biased_downward_and_worse_for_slow_reversion`). Since a time stop is set at *k × estimated half-life*, this bias makes the time stop systematically **too loose**, letting losing trades run past when the model says they should have converged — worse exactly for the slow-reverting, marginal pairs where discipline matters most. The mitigant is to shrink the estimate or set `k` conservatively.

**Q14. How does the project actually charge transaction costs, and why does a mean-reversion strategy feel this more than a buy-and-hold one?**

`backtest.py`'s `CostModel` charges commission (`cost_bps`, default 5bp/leg) and slippage (`slippage_bps`, default 2bp/leg) as explicit cash on every leg's *traded notional* at each rebalance, plus annualised borrow (`borrow_bps`, default 50bp) accrued daily (ACT/252) on the short leg's market value; the accounting identity `net = gross − commission − slippage − borrow` is asserted to the penny. A mean-reversion strategy trades every entry/exit/stop cycle — the walk-forward book turns over ~6.7x annually — so costs compound with turnover in a way a buy-and-hold position never sees.

**Q15. What does the equity project's cost-sensitivity table (VALIDATION.md §8) actually show, quantitatively?**

Holding slippage (2bp) and borrow (50bp) fixed and varying commission from 0 to 20bp/leg on the walk-forward book: net P&L falls from $1,463,482 (0bp) to $1,209,685 (20bp), Sharpe from 1.66 to 1.37, and hit rate collapses from 68.8% to 56.2% — because marginal trades that were barely profitable at low cost flip to losers as cost eats the reversion gain. The stated rule of thumb: at ~6.7x annual turnover the book loses ≈4.5bp/yr of capital per bp of per-leg commission — survivable at institutional cost levels, but VALIDATION.md explicitly notes it "would not survive retail spreads on less liquid names."

**Q16. Why does the pipeline screen candidate pairs on *return* correlation rather than price correlation?**

Because price levels of two *independent* random walks are spuriously correlated — a classic Granger-Newbold result — while their returns are not. `test_price_corr_spurious_return_corr_not` measures independent walks averaging |price corr| > 0.45 but |return corr| < 0.10. A price-level screen would select pairs that merely share a common drift; a return-correlation screen selects pairs that share common shocks, which is the more meaningful pre-filter. METHODOLOGY.md is explicit that this screen is still just a *cheap filter*, not evidence of cointegration — the trap pairs in the panel have return correlation ≈0.92 and are still rejected at the ADF stage.

**Q17. Why does this project choose Engle-Granger over Johansen, and over the classic distance method?**

Against Johansen (ML rank/vector estimation for N-series VARs): for a **two-leg pair** the cointegration rank can only be 0 or 1, which is exactly what EG's single test answers; Johansen's VAR lag-order selection and eigenvalue machinery buy nothing at N=2 and it's documented as the scaling path for baskets of 3+ names instead. Against the distance method (Gatev-Goetzmann-Rouwenkamp SSD, ranking pairs by sum-of-squared normalized price gaps): it's cheap and assumption-light — kept as a pre-screen (`ssd_screen`) — but has no null hypothesis, so it can't distinguish "these paths track because of a real common factor" from "they tracked by luck," and it gives no spread model, hence no principled half-life or time-stop. EG gives a testable null and its residual *is* the tradeable hedge ratio directly (`docs/METHODOLOGY.md` §1).

**Q18. What's the adaptive alternative to the static OLS hedge ratio, and what does it trade off?**

`spread.py`'s `rls_hedge_ratio` implements exponentially-weighted **recursive least squares** with a forgetting factor λ ("Kalman-lite") — algebraically the Kalman filter for a random-walk-coefficient model where the state-noise-to-observation-noise ratio is tied to λ instead of estimated separately. It has one knob (memory ≈ 1/(1−λ)) and runs O(1) per step, versus a full Kalman filter's two separately-calibrated noise covariances and explicit smoothing. Tests show it tracks a hedge ratio drifting 1.0→2.0 with mean error < 0.02. The documented identification caveat: with short memory, an intercept becomes nearly collinear with the price level, so RLS tracking is run with `intercept=False`, letting the OU mean absorb the level instead.

**Q19. What is the project's no-lookahead guarantee, and how is it tested?**

The backtester enforces a strict one-bar execution lag: a signal decided with information through day t is executed at day t+1's close (`executed_t = target_{t-1}`), enforced structurally — the engine physically reads `target[t-1]`, so it cannot see `target_t` when trading at t. `test_engine_reports_the_honest_losing_result` constructs a spread that alternates ±3 daily, engineered so same-day ("cheating") execution is profitable (mean reversion happens overnight) while the honest lagged execution loses money — the test asserts the engine produces the *losing* number. A companion no-lookahead invariant (property-based) checks that truncating the sample leaves every earlier day's P&L bit-identical (`docs/VALIDATION.md` §5, §10).

**Q20. What's the difference between the in-sample and walk-forward backtest results on the seed-7 panel, and why does that gap matter?**

In-sample: net P&L $2.92mm, annualised return 12.3%, Sharpe 2.42, hit rate 95.6%. Walk-forward (formation window fitted, frozen, traded strictly out-of-sample): net P&L $1.40mm, 7.4%, Sharpe 1.59, hit rate 67.2% (`docs/VALIDATION.md` §7). VALIDATION.md calls the gap itself "a validation result": selection criteria and OU parameters fitted on the same data they're tested on flatter every number, so the walk-forward figures — not the in-sample ones — are the honest expectation for sizing a live book (`docs/DESK_GUIDE.md` explicitly tells the desk to scale on the walk-forward 67%, not the in-sample 96%).

**Q21. Sharpe ratio standard errors are computed two ways in this project — why, and what does the second method show?**

`metrics.py` computes the Sharpe SE both under the iid assumption (`√((1+SR²/2)/T)`, Lo 2002) and **Lo-adjusted** for serial correlation, using a Bartlett/Newey-West long-run variance ratio on the mean return. Mean-reversion strategies produce autocorrelated P&L (a position held over several days generates correlated daily P&L within the trade), so the iid SE understates uncertainty exactly when it matters — tested to widen the SE by >30% at AR(1) φ=0.6. On the walk-forward book the iid SE is 0.46 vs Lo-adjusted 0.50, a real (if modest) difference in how confidently the Sharpe of 1.59 should be read.

**Q22. What is the "trap" the equity selection funnel is specifically built to catch, and how many pairs actually survive it?**

Starting from 40 same-sector candidate pairs in the seed-7 20-name panel: 8 pass the return-correlation screen (ρ ≥ 0.6), and only 4 of those pass Engle-Granger at 5%. The 4 rejected despite passing correlation are three correlated-random-walk traps (return correlation 0.919–0.923, EG stats −1.45 to −2.13, well short of the −3.34 critical value) plus the regime-break pair — every genuinely cointegrated pair is accepted (EG stats −7.4 to −9.8) and no trap slips through (`docs/VALIDATION.md` §4). This is the funnel doing exactly the job correlation alone cannot: separating shared-shock co-movement from a genuinely stationary spread.

**Q23. How does the FX project frame its version of this strategy differently from the equity one — what's actually being traded?**

The FX project trades **pairs of currency pairs** — e.g. AUDUSD vs NZDUSD, EURUSD vs GBPUSD — rather than pairs of single stocks, and the spread is built on **log rates**, not dollar price levels: `s_t = log P1_t − α − β log P2_t` (`fx/05-pairs-trading/docs/METHODOLOGY.md` §1). Log rates are used because FX P&L is naturally multiplicative, the hedge ratio gets a clean elasticity interpretation, and cross rates are exact ratios of USD legs, so triangular consistency becomes linear and testable in logs. The FX methodology otherwise reuses the identical from-scratch Engle-Granger + OU machinery as the equity project (same MacKinnon N=2 surfaces, same OLS/MLE OU cross-check).

**Q24. Why does the FX methodology say cointegration is *rarer* in FX than in equities, and what "common-factor" trap does it specifically warn about?**

An equity pair can cointegrate because both prices load on the same firm-level cash-flow trend; an FX rate is already a *ratio* of two money stocks with no "company" anchoring its level, so any anchor (PPP, real-rate differentials) acts only over years-to-decades — weak glue at trading horizons. The specific trap: any two USD-quoted pairs share the USD leg by construction, so high return correlation between, say, AUDUSD and NZDUSD is partly mechanical (a common USD factor), and a correlation screen alone floods the funnel with USD-factor pairs that are *not* cointegrated. The project's synthetic two-block panel reproduces this directly: 0 survivors among 7 factor-correlated candidates, versus 1 survivor out of 1 planted true cointegration (`fx/05-pairs-trading/docs/METHODOLOGY.md` §2). Where FX cointegration does hold, the doc says it reflects a real policy or economic linkage — AUD/NZD (twin commodity economies), NOK/SEK (Scandinavian bloc), EUR/CHF pre-2015 (SNB floor).

**Q25. What extra P&L component does the FX version model that the equity version doesn't, and why can it flip the strategy's sign?**

Carry. A held FX position is financed — long the base currency earns its deposit rate, funded in the quote currency at its rate — so total backtest P&L is `spot P&L + carry accrual − transaction costs`, with the daily roll computed from covered interest parity forward points. The methodology states a mean-reversion signal computed on spot alone can be systematically **wrong-carry**, and the pipeline constructs an explicit example where including carry flips the sign of the strategy's overall P&L (`fx/05-pairs-trading/docs/METHODOLOGY.md` §1, §2). The FX signal machine also adds a **carry-aware entry filter**: an entry is vetoed when expected carry drag over the expected holding period (`carry/day × k·half-life`) exceeds the expected reversion gain (`(|z|−exit)·σ`) — carry-favourable entries are never vetoed.

**Q26. What is the FX project's "triangular null case," and what does it guard against?**

`log EURUSD + log USDJPY − log EURJPY = 0` holds identically because cross rates are exact ratios of USD legs — feeding this combination through the cointegration/spread machinery produces a "spread" with variance ~1e-31, which must be flagged **degenerate**, never reported as "perfectly cointegrated." A degeneracy detector (spread std below 1e-7 in log units) exists specifically to catch this, and a funnel test asserts it fires (`fx/05-pairs-trading/docs/METHODOLOGY.md` §6). Economically the doc notes there's no exploitable triangular arbitrage at daily frequency once three legs' bid-ask spreads are paid — any measured deviation in real data is inside costs and is arbitraged away at the tick level by HFT, so a "cointegrated" identity like this is a data-hygiene bug to catch, not a trading opportunity.


## Round 10 — Credit Risk / PD Modeling

**Q1. What does "probability of default" (PD) mean, and over what horizon is it typically quoted in this portfolio?**

PD is the probability that an obligor defaults within a fixed reference period, most commonly one year. Both credit-risk projects in this portfolio (`python/equity/06-credit-risk` and `python/fx/06-credit-risk`) quote PD as a 1-year figure: the equity project's Bernoulli target is "did this loan default within one year of origination" (METHODOLOGY.md §5, assumption 2), and the FX project derives a hazard rate `h = −ln(1 − PD_1y)` explicitly from a "1y PD" rating-band midpoint (METHODOLOGY.md §2.3) to extend it to multi-year CVA horizons.

**Q2. What model does the equity project actually implement for estimating PD — logistic regression, a structural/Merton model, or something else?**

It is a WOE (Weight-of-Evidence) scorecard: each raw feature is binned, each bin replaced by its WOE value, and a logistic regression is fitted on the WOE-transformed features via a from-scratch Newton-Raphson / IRLS solver (`fit_logistic` in `src/eq_credit/model.py`). There is no Merton/structural component in the equity project — METHODOLOGY.md §2 explicitly considers and rejects a structural/distance-to-default model as "Alternative 3," noting the book is mid-market lending with no market cap or asset volatility to observe.

**Q3. Why is a linear-in-log-odds model (logistic regression) a natural fit for a binary default/no-default outcome?**

The target is Bernoulli (default = 1, survive = 0), so a model needs to map linear predictors to values in (0,1) while keeping estimation tractable; the logit link `ln(p/(1-p)) = Xβ` does this and turns maximum-likelihood estimation into a smooth, concave optimization solvable by Newton's method. `fit_logistic` implements exactly this: it iterates `β ← β + (XᵀWX)⁻¹Xᵀ(y − p)` with `W = diag(p(1−p))`, which is IRLS on the Bernoulli log-likelihood. WOE-binning first puts every feature on the same log-odds scale, so the fitted coefficients (all ≈ −1 on the pipeline book) are directly comparable across features — a property a raw linear-probability model wouldn't give you.

**Q4. Why doesn't this project use a Merton-style structural PD model instead?**

METHODOLOGY.md's "Alternative 3" explains: Merton/KMV-style distance-to-default models need an observable market asset value and asset volatility (typically backed out from traded equity), which mid-market borrowers in this book don't have — no market cap, no listed equity. The FX project's sovereign methodology makes the same point about sovereigns (no observable "asset value" or default barrier), calling the Merton mapping "heroic" there. Structural models also tend to produce point-in-time, cyclical PDs, which the equity project flags as being in tension with a stable origination cutoff (TTC vs PIT, METHODOLOGY.md §6).

**Q5. What is "expected loss" and what are its three standard components?**

Expected loss (EL) is the average credit loss a lender expects to realize on an exposure, decomposed as EL = PD × LGD × EAD: probability of default, loss given default (the fraction of exposure not recovered), and exposure at default (the amount outstanding when default happens). `expected_loss` in `src/eq_credit/portfolio_risk.py` implements this multiplication directly, after validating that PD, LGD, and EAD are all finite and in-range (`_validate_pd_lgd_ead`).

**Q6. Does this project implement all three components of EL (PD, LGD, EAD), or only PD?**

The scorecard itself only estimates PD from data — LGD and EAD are treated as inputs, not modeled outputs. `expected_loss(pd, lgd, ead)` in `portfolio_risk.py` takes LGD and EAD as arguments rather than deriving them, and METHODOLOGY.md assumption 4 states this plainly: "LGD and EAD independent of PD... EL = PD·LGD·EAD multiplies point estimates" using externally supplied LGD/EAD. The demo book applies a flat 25% downturn-LGD haircut (EL rises from 605.5m to 754.3m) as a regulator-style stress adjustment rather than a fitted LGD model.

**Q7. What is "wrong-way risk" and how does this project handle (or fail to handle) it?**

Wrong-way risk is the phenomenon where PD and LGD (or exposure) rise together — collateral values fall exactly when defaults spike, as in 2008 commercial real estate. Because EL = PD·LGD·EAD multiplies independent point estimates, this project cannot capture that correlation directly (METHODOLOGY.md assumption 4); its only concession is the crude, flat downturn-LGD haircut applied portfolio-wide rather than a joint PD-LGD model. The FX project names the same gap explicitly for CVA (assumption A7): assuming exposure independent of counterparty default is "the worst possible assumption" for an EM sovereign on a long-USD forward, and VALIDATION.md quantifies roughly a 6x CVA understatement from ignoring it.

**Q8. How is the equity PD model validated for discrimination — how well does it rank-order risk?**

VALIDATION.md reports AUC, Gini, and KS computed from scratch (`roc_auc`, `gini`, `ks_statistic` in `src/eq_credit/validation.py`) and cross-checked against `sklearn.roc_auc_score` to 1e-12. On the training book, AUC = 0.7843 (95% bootstrap CI 0.769–0.800) against a true-model AUC ceiling of 0.7813 — meaning the scorecard is essentially at the information frontier the synthetic generator allows. Decile analysis (`decile_table`) shows default rates rising monotonically from 0.10% in the safest decile to 11.2% in the riskiest, a 112x spread, which is also unit-tested (`test_decile_default_rates_monotone_for_good_model`).

**Q9. What does "calibration" mean for a PD model, and how is it distinct from discrimination?**

Discrimination measures whether the model ranks obligors correctly by relative risk (AUC/Gini/KS answer "does a higher score mean higher risk?"). Calibration measures whether the model's absolute predicted probabilities match realized default rates (does a group of obligors scored at 3% PD actually default about 3% of the time?) — a model can discriminate well while being badly miscalibrated, e.g. systematically too low or too high. VALIDATION.md's own OOT results make this concrete: OOT AUC only drops modestly (0.7843 → 0.7671) but calibration is flatly rejected by Hosmer-Lemeshow (p drops from 0.022 to 3.8e-07), and realized OOT default rate (4.78%) far exceeds mean predicted PD (3.80%) — the model still ranks obligors reasonably well but its absolute PDs are wrong.

**Q10. What is the Hosmer-Lemeshow test, and what does this project's implementation and testing of it show?**

Hosmer-Lemeshow is a goodness-of-fit test for calibration: it groups observations into PD deciles, compares predicted vs. observed default counts per group, and forms a chi-square statistic under the null that the model is well calibrated. `hosmer_lemeshow` in `src/eq_credit/validation.py` implements this, and its null-hypothesis behavior is independently verified by simulation — with known true probabilities, the statistic averages ≈10 over 60 replications, consistent with χ²(10) as expected (`test_hosmer_lemeshow_null_distribution`). On the real pipeline, HL correctly fails to reject calibration in-sample (χ² = 17.9, p = 0.022) but sharply rejects it out-of-time (χ² = 44.9, p = 3.8e-07), which is exactly the intended diagnostic behavior.

**Q11. What is PSI (Population Stability Index), and what does it measure that Hosmer-Lemeshow does not?**

PSI measures how much the distribution of a score or feature has shifted between two samples (e.g. development vs. current portfolio) — it is a covariate/input-drift monitor, not a correctness check. `psi`, `psi_from_proportions`, and `psi_status` in `validation.py` implement it, hand-validated to 1e-14 against known proportions, with bands (< 0.10 stable, 0.10–0.25 monitor, > 0.25 shifted) also validated on planted shifts. VALIDATION.md's failure-mode #1 makes the crucial distinction: on the OOT sample, score PSI stayed at 0.074 (stable band) even as HL calibration collapsed — PSI would not have caught the regime break because the input mix barely moved even though the feature→default relationship shifted; both monitors are required together (DESK_GUIDE.md §7).

**Q12. What is the Brier score, and where does it appear in this project's validation?**

The Brier score is the mean squared error between predicted probabilities and realized binary outcomes — a combined discrimination-and-calibration score (lower is better). `brier_score(y, pd_hat)` in `validation.py` computes it and, per the hardening notes in VALIDATION.md §6, explicitly rejects non-finite inputs and gates predicted PDs to [0, 1]. On the pipeline it rises from 0.0272 in-sample to 0.0431 out-of-time, consistent with the AUC/HL story of degraded but not collapsed performance under regime drift.

**Q13. Walk through a concrete edge case this project's tests cover around near-zero or near-one predicted probabilities.**

`test_edge_cases.py` and `test_input_validation.py` cover PD = 0/1 explicitly: scores are clamped to finite values while preserving ordering rather than producing ±infinity from `ln(p/(1-p))`. More broadly, every entry point that consumes a probability — `brier_score`, `hosmer_lemeshow`, `expected_loss`, `basel_k`, `asset_correlation` — validates inputs are finite and gates predicted PDs to [0, 1] (VALIDATION.md §6, "NaN/Inf rejection everywhere"), because `NaN < 0` silently evaluates to `False` and would otherwise pass naive range checks.

**Q14. How does the model handle a zero-default (or all-default) training sample?**

With zero defaults, the WOE calculation, the logistic MLE, and AUC are all mathematically undefined (division by zero bad-count, no positive class to rank against). Rather than returning garbage, every relevant entry point raises an informative `ValueError` — tested explicitly in `test_zero_defaults_raises_everywhere` (VALIDATION.md §5.3). The same failure-mode section notes that at the opposite extreme of very few observations (~30 rows / 3 defaults) the fit stays numerically finite but standard errors balloon, and this is reported honestly via the Wald table rather than hidden.

**Q15. What happens when a feature has missing data for a given obligor, and what edge case does this create?**

Missing values get their own WOE bin rather than being imputed or dropped — `fit_numeric_binning` in `src/eq_credit/woe.py` treats missingness as informative, and on the pipeline book the missing bin for `behavioral_score` carries a WOE of −0.80 (thin-file borrowers score riskier). VALIDATION.md documents this as a tested edge case ("all-missing feature raises," "unseen categories at transform time are mapped to missing-bin WOE"), but METHODOLOGY.md assumption 6 flags the real fragility: the fitted missing-bin WOE assumes the *reason* for missingness is stable over time — if a data-pipeline change later makes the same field missing-at-random, that penalty mis-prices otherwise-clean borrowers. DESK_GUIDE.md's COVID scenario (§7) is exactly this failure realized: payment holidays made `behavioral_score` go missing for moratorium accounts for a genuinely different reason than the historical missingness pattern.

**Q16. What is separation, and how does this project detect and handle it?**

Separation occurs when a feature (or combination) perfectly or near-perfectly predicts the outcome, causing the unpenalized logistic MLE to diverge (coefficients run to infinity as the fit tries to push predicted probabilities to exactly 0 or 1). `fit_logistic` detects runaway iterates and raises a `SeparationWarning`, recommending ridge regularization; supplying a positive ridge penalty causes the fit to converge instead — both directions are unit-tested (`test_separation_detected_and_warned`, `test_separation_ridge_regularizes`, VALIDATION.md §5.4).

**Q17. What is target/outcome leakage, and how does this project guard against it — concretely?**

Leakage is when a feature encodes information that would not actually be available (or would already imply the outcome) at prediction time — a planted post-outcome field like `writeoff_flag` is the textbook example, since it is essentially the label in disguise. Two independent controls catch it: the IV screen automatically flags any feature with IV > 0.5 as suspicious (`SuspiciousIVWarning` in `src/eq_credit/woe.py`; `writeoff_flag` triggers it with IV ≈ 6.4), and a separate cleaning-layer deny-list (`FORBIDDEN_POST_OUTCOME_FIELDS`) refuses such fields outright with a `LeakageError` before they ever reach binning. VALIDATION.md calls this "defence in depth" because both controls fire independently of each other.

**Q18. What out-of-sample / regime edge case does this project's validation deliberately construct, and what does it reveal?**

The OOT (out-of-time) sample is built with a planted 0.5 drift plus a 0.5 log-odds calibration shift, lifting the realized default rate from 2.95% (training) to 4.78% (OOT) — modeling a genuine regime change rather than resampling noise. This reveals that a PIT (point-in-time) scorecard degrades gracefully in discrimination (AUC only −1.7 points) but breaks sharply in calibration (HL p-value collapses from 0.022 to 3.8e-07, mean predicted PD 3.80% undershoots realized 4.78%) — and crucially, PSI alone would have missed it (stayed at 0.074, "stable"), which is exactly why VALIDATION.md's failure-mode #1 insists monitoring must track realized-vs-predicted rates directly, not PSI alone.

**Q19. What is a Through-the-Cycle (TTC) vs Point-in-Time (PIT) PD, and which is this model?**

PIT PDs track the current state of the credit cycle using current, often cyclical inputs (like a behavioral score) and are recalibrated frequently — appropriate for IFRS 9/CECL provisioning, which wants cycle-sensitive expected losses. TTC PDs average over a full cycle for stability, which is what Basel IRB capital is designed around. METHODOLOGY.md §6 states this model is "essentially PIT" because it's calibrated to the realized default rate of its training window and includes an intrinsically cyclical behavioral score — with the consequence that feeding these PDs directly into the IRB capital formula makes capital procyclical (VALIDATION.md failure mode #5 shows repricing the demo book at OOT-implied calibration raises every PD by ×1.26 on average).

**Q20. How does the equity project's Basel IRB capital calculation get validated for correctness?**

`basel_k` in `portfolio_risk.py` implements the regulatory capital function K(PD, LGD, M) — the Vasicek 99.9% conditional PD minus expected loss, times a maturity adjustment. It is hand-checked at PD = 1%, LGD = 45%, M = 2.5 against the published Basel II corporate risk weight of 92.32% from the BCBS Explanatory Note (July 2005), independently re-derived in the test with the exact regulatory constants R(PD), b(PD), and Φ⁻¹(0.999) (`test_basel_k_reproduces_independent_hand_calculation`). The Vasicek machinery underneath it is separately validated: quantile/CDF round-trip to 1e-10, and the Vasicek 99.9% quantile matches the Basel conditional-PD term at ρ = R(PD) to 1e-12.

**Q21. What does the Vasicek economic-capital layer add on top of the PD model, and where is the seeded Monte Carlo used?**

The PD model gives obligor-level default probabilities; the Vasicek single-factor Gaussian copula (`vasicek_cdf`, `vasicek_quantile`, `simulate_portfolio_losses`, `economic_capital` in `portfolio_risk.py`) turns those into a portfolio-level loss distribution and its 99.9% tail, i.e. economic capital — a quantity a single-obligor PD/EL calculation can't produce on its own. Monte Carlo (seeded via `numpy.random.default_rng`) is used because the analytic Vasicek formula assumes an infinitely granular (homogeneous, infinitely diversified) portfolio; the MC engine handles the actual finite, heterogeneous book, and is checked to converge to the analytic limit as portfolio size grows (`test_mc_quantile_approaches_analytic_as_n_grows`) and to stay above the analytic bound for finite N (granularity adjustment, `test_finite_portfolio_tail_at_least_infinitely_granular`).

**Q22. What does the FX project's methodology cover that's specific to credit risk in an FX context?**

`python/fx/06-credit-risk` covers three blocks beyond the equity project's corporate scorecard: (1) a sovereign PD scorecard using the same WOE + IRLS logistic approach but for country-year panel data with only ~63 training default events; (2) FX settlement (Herstatt) risk — the exposure window between when a bank's paid currency leg becomes irrevocable and when the received leg settles with finality, which is asymmetric by RTGS time zone and zero under PvP/CLS settlement; and (3) pre-settlement exposure and CVA on FX forwards via Garman-Kohlhagen GBM simulation, netting rules, and a CVA formula driven off the sovereign scorecard's PD term structure.

**Q23. Why does the FX project use ρ_sov = 0.30 for sovereign Vasicek capital instead of the equity project's corporate range?**

METHODOLOGY.md §2.4 sets the sovereign asset-correlation parameter at ρ_sov = 0.30, deliberately above the Basel corporate range of ρ ∈ [0.12, 0.24], because sovereign defaults cluster on global systematic factors — USD funding cycles, commodity busts, contagion — more than idiosyncratic firm-level risk does. The ordering K(ρ_sov) > K(ρ_corp) across the PD range is unit-tested, and assumption A10 documents this as a deliberately conservative single-factor simplification of what is in reality a multi-region, multi-factor tail-dependence structure.

**Q24. What is the low-default-portfolio problem, and how does the FX sovereign model address it honestly rather than hide it?**

With only ~63 historical sovereign default events feeding 10 WOE coefficients, standard errors are wide and some coefficients are barely significant (e.g. `fiscal_gdp` z = -0.7, per METHODOLOGY.md §1) — a classic low-default-portfolio (LDP) problem where there simply isn't enough data to pin down a precise model. The project addresses this by (a) using WOE compression so each feature contributes one monotone dimension rather than many free parameters, (b) separation detection with a ridge fallback, and (c) reporting wide bootstrap confidence intervals rather than hiding them — and explicitly refuses to claim precision it doesn't have, anchoring AAA/AA band PD midpoints to long-run rating-agency default studies (assumption A9) instead of fitting them from sparse sovereign default data.

**Q25. How does the equity project's validated output feed into desk-level credit limits and origination decisions?**

DESK_GUIDE.md §2 describes cutoffs set directly from the KS/decile table: since the scorecard concentrates 11.2% of defaults in the worst decile versus 0.10% in the best (112x spread), the desk sets a score cutoff (e.g. auto-approve ≥ 620, manual review 560–620, decline < 560) that excludes the riskiest deciles, removing roughly 55% of expected defaults at the cost of about 20% of loan volume. Reason codes for declines read directly off the points table (e.g. "leverage > 0.94: 58 pts vs 117 for < 0.18"), which is only possible because the scorecard is interpretable by construction — a key reason METHODOLOGY.md gives for choosing it over a GBM.

**Q26. How does the PD model feed into loss provisioning and regulatory capital, and why does the desk need to treat those two uses differently?**

For provisioning, DESK_GUIDE.md §4 states IFRS 9/CECL Stage-1 expected credit loss is literally this model's output — 12-month ECL = PD₁y·LGD·EAD (605m, 1.47% of EAD on the demo book; 754m with the downturn-LGD haircut) — and because the model is PIT, ECL rising with the cycle is *intended* under IFRS 9. For capital, however, DESK_GUIDE.md §5 and VALIDATION.md failure mode #5 warn that feeding the same PIT PDs into the Basel IRB formula makes capital procyclical (repricing at OOT-implied calibration raises every PD by ×1.26, demanding more capital exactly when it's scarcest); Basel expects long-run-average (TTC) PDs per grade, so IRB use requires a TTC overlay that this project documents as needed but does not itself build (METHODOLOGY.md §6). Treasury separately consumes the Vasicek economic-capital number (8.20% of EAD at 99.9% vs Basel K's 10.03%) for ICAAP and stress testing (DESK_GUIDE.md §5).


## Round 11 — Portfolio Optimization & Risk Allocation

**Q1. State the Markowitz mean-variance optimization problem. What is the efficient frontier?**

Markowitz (1952) mean-variance optimization (MVO) either minimizes portfolio variance `w'Σw` subject to a target expected return `w'μ = m` and a budget `sum(w) = 1` (`target_return_portfolio` in `eq_port/mvo.py`), or equivalently maximizes expected return for a variance cap (`target_risk_portfolio`, which maximizes `w'μ` subject to `sqrt(w'Σw) ≤ target_vol`). Sweeping the target across its feasible range and recording the minimum-variance weight at each point traces the efficient frontier — the set of portfolios where no other combination gives higher return at the same risk. `efficient_frontier()` implements this sweep, either analytically (unconstrained) or numerically via repeated SLSQP solves (bounded).

**Q2. Give the closed-form unconstrained tangency (maximum-Sharpe) portfolio, and say what code implements it.**

`w_tan = Σ⁻¹(μ − rf·1) / (1'Σ⁻¹(μ − rf·1))`, implemented in `tangency_weights()` in `mvo.py`. It is validated in `test_tangency_matches_formula` to 1e-12. The function deliberately raises a `ValueError` when `1'Σ⁻¹(μ − rf) ≤ 0`, since a non-positive denominator means no fully-invested tangency portfolio with positive excess return exists for the given inputs — usually a sign that the mean estimates are unreliable rather than a real economic conclusion.

**Q3. Why is mean-variance optimization described in this project's docs as an "estimation-error amplifier" or "error maximizer"?**

`METHODOLOGY.md` §1 frames it via Merton (1980) and Michaud (1989): the standard error of an estimated mean return over `Y` years is `σ/√Y`, independent of sampling frequency — with `σ ≈ 20%` and one year of data the SE of the annual mean is ≈20%, an order of magnitude larger than realistic cross-sectional spreads in true expected returns. Unconstrained tangency weights are `w ∝ Σ⁻¹(μ − rf)`: the optimizer takes differences of noisy means and multiplies by an inverse covariance that amplifies its worst-estimated (smallest) eigenvalues. Assets that got lucky in-sample (high estimated mean, low estimated vol/correlation) receive extreme weights — the optimizer "loads up on exactly the estimation errors," which is precisely why Michaud calls MVO an error maximizer rather than a return maximizer.

**Q4. What sampling frequency fact makes the mean-estimation problem structurally different from the variance-estimation problem?**

Variance estimates improve with the *number of observations*, so sampling daily instead of monthly sharpens them. Mean estimates improve only with *calendar span* (`σ/√Y`), so daily sampling gives no benefit at all for the mean — you cannot out-sample your way past Merton's problem, only wait years for it to shrink. This asymmetry is exactly why the covariance side of this project (Ledoit-Wolf, EWMA) can be estimated with real confidence from a 252-day window while the mean side needs an entirely different strategy (shrinkage, equilibrium priors, or avoiding it altogether).

**Q5. What is James-Stein shrinkage, and quantitatively how much does the project's estimation-error study shrink toward it?**

James-Stein shrinks each asset's sample mean toward the cross-sectional grand mean, with data-driven intensity `phi = clip((N−3)·σ̄²/T / Σᵢ(μᵢ−m̄)², 0, 1)` (`returns_est.py`, per `METHODOLOGY.md` §2). It is biased toward "all assets are equal" but dominates the sample mean in total squared error for N ≥ 4 (Stein 1956). In the project's 20-window estimation-error study, average James-Stein intensity is φ = 0.96 — the data essentially beg to ignore the sample means — and shrunk-mean tangency's mean true Sharpe (0.313, long-only) beats raw-mean tangency (0.300, long-only) and is far more stable across windows.

**Q6. What are Black-Litterman and reverse optimization, and how do they sidestep the raw-sample-mean problem differently from James-Stein?**

Reverse optimization discards sample means entirely and backs out the returns that would make a reference portfolio (e.g. the market) optimal: `π = δΣw_mkt` — no time-series noise, you inherit the market's implied (CAPM equilibrium) view. Black-Litterman is a Bayesian blend of that equilibrium prior with explicit investor views: `μ_BL = π + τΣPᵀ(PτΣPᵀ+Ω)⁻¹(Q−Pπ)`. Unlike James-Stein, which shrinks blindly toward the cross-sectional mean, BL only moves weights where you actually hold a view (`Ω`→0 honors a view exactly, no view leaves the prior untouched — both tested to machine precision in `test_bl_no_views_posterior_equals_prior` and `test_bl_infinitely_confident_view_holds_exactly`).

**Q7. What is the minimum-variance portfolio, and why does the project favor it in practice specifically because it needs no return forecast?**

`min_variance_weights(cov) = Σ⁻¹1/(1'Σ⁻¹1)` requires only a covariance estimate — no `μ` enters the formula at all. Because mean estimation is the dominant source of error (Q3), a portfolio that never touches `μ` structurally sidesteps that entire failure mode. The estimation-error study bears this out numerically: min-variance with Ledoit-Wolf achieves mean true Sharpe 0.339 (range 0.317–0.371) versus raw-mean tangency's 0.141 (range −0.125 to 0.325) — not only a higher average but a dramatically tighter, more stable range across the 20 independent windows.

**Q8. In the out-of-sample walk-forward race (VALIDATION.md §4), how does minimum-variance actually perform relative to raw-mean tangency and equal weight?**

On the seed-1 8-asset panel (2400 days, 120-day crisis, monthly rebalance, 10bp costs), MinVar (LW) posts AnnRet −0.031, AnnVol 0.181, Sharpe −0.083, MaxDD 0.384 — better on every dimension than raw-mean tangency (AnnRet −0.114, AnnVol 0.251, Sharpe −0.354, MaxDD 0.658) and better than equal weight on vol and drawdown too. Averaged across six independent seeds, MinVar (LW) is the *only* strategy with positive mean net Sharpe (+0.028), versus −0.049 for equal weight and −0.064 for raw tangency — "raw-mean tangency is the worst strategy in the race... while the mean-free allocators... do fine."

**Q9. Define risk parity / equal risk contribution (ERC). What optimization problem does it actually solve, and where is it implemented?**

ERC allocates so every position contributes equally to total portfolio *risk* — not equal notional, and not "optimal" in the return sense. It is implemented in `erc_weights()` (`risk_parity.py`) by minimizing the strictly convex `F(y) = ½y'Σy − Σᵢ bᵢ ln(yᵢ)` over `y > 0` via cyclical coordinate descent (Griveau-Billion, Richard & Roncalli 2013); its first-order condition `yᵢ(Σy)ᵢ = bᵢ` is exactly the equal-risk-contribution condition, and normalizing `y/sum(y)` gives the ERC weights. Existence/uniqueness for PD `Σ` is guaranteed by Spinu (2013). Critically, ERC needs no mean vector at all and no matrix inversion — the coordinate update is the positive root of a scalar quadratic per asset.

**Q10. Why does risk parity behave so differently from mean-variance in practice? Ground the answer in the project's failure-mode data.**

Because ERC ignores return forecasts entirely, it cannot be seduced by a lucky in-sample mean the way tangency can — it will never concentrate into one or two "great Sharpe" assets. In the estimation-error study, ERC's true Sharpe ranges only 0.328–0.339 across 20 windows (essentially flat) versus raw tangency's −0.125 to 0.325. In the crisis subperiod (`VALIDATION.md` §5), ERC's effective N stays at 7.8 (still broadly diversified) while the concentrated raw-mean tangency book collapses to effective N 1.57 and posts the worst crisis vol (0.568), worst drawdown (0.470), and worst crisis return (−0.368) of every strategy tested.

**Q11. Why does ERC "collapse" to naive inverse-vol weighting in a special case, and where is that tested?**

When all pairwise correlations are equal, the equal-risk-contribution condition reduces algebraically to weights proportional to `1/σᵢ` — `inverse_vol_weights()` — because correlation contributes identically to every position's marginal risk. This is tested exactly (`test_erc_constant_correlation_equals_inverse_vol`, tolerance 1e-9). Outside that special case, true ERC diverges from naive inverse-vol because it also accounts for how each asset's covariance with the rest of the book affects its marginal risk contribution, whereas inverse-vol ignores correlation structure entirely.

**Q12. Because ERC is mean-free, does it care about the sign of correlations? Explain using the 2022 stock-bond scenario in DESK_GUIDE.md.**

ERC budgets *volatility*, not directional exposure, so its weights are largely insensitive to a correlation sign flip — DESK_GUIDE notes that re-running ERC with the stock-bond off-diagonal block sign-flipped leaves the ERC *weights* barely changed. What breaks instead is the *portfolio's realized risk level*: in 2022 the historically negative stock-bond correlation flipped positive (inflation shock), and both the static 60/40 book and risk parity, which both size risk assuming diversification from that negative correlation, drew down together. The lesson: it's the risk target that breaks, not the allocation logic — which is why RP desks monitor rolling stock-bond correlation as a named risk factor rather than trusting the ERC weights to self-correct.

**Q13. Define marginal risk contribution and component (Euler) risk contribution. Give the formulas as implemented.**

For the portfolio variance `w'Σw` as the risk measure, the marginal contribution of asset `i` is `∂(portfolio vol)/∂wᵢ`, but this project works directly in variance terms: `risk_contributions()` in `risk_parity.py` computes `RCᵢ = wᵢ·(Σw)ᵢ`, i.e. weight times `(Σw)ᵢ` — the marginal variance contribution — rather than a vol-normalized marginal. `(Σw)ᵢ` is the derivative of `w'Σw` with respect to `wᵢ` (up to the factor of 2 that cancels against the `wᵢ` multiplication), and `RCᵢ` is that derivative weighted by the position's own size — the Euler/component decomposition.

**Q14. Why does summing component risk contributions recover total portfolio risk exactly? What's the mathematical reason, and how is it tested?**

`w'Σw` is a homogeneous function of degree 2 in `w` (scaling all weights by `k` scales variance by `k²`). Euler's theorem for homogeneous functions says `Σᵢ wᵢ·∂f/∂wᵢ = deg(f)·f(w)`; here `∂(w'Σw)/∂wᵢ = 2(Σw)ᵢ`, so `Σᵢ wᵢ·2(Σw)ᵢ = 2·w'Σw`, i.e. `Σᵢ wᵢ(Σw)ᵢ = w'Σw` exactly (the factor of 2 cancels because the module defines `RCᵢ` without it). This is an algebraic identity, not an approximation, and `test_rc_euler_identity_exact` verifies `Σᵢ RCᵢ = w'Σw` to 1e-18 — effectively machine precision. It is what makes "risk contribution" a well-posed accounting concept: contributions partition the total exactly, with nothing left over and nothing double-counted.

**Q15. How does the project prove ERC actually equalizes risk contributions, and to what precision?**

`test_erc_contributions_all_equal` checks that all `RCᵢ` from `erc_weights()` output agree to a relative spread under 1e-8 (observed ~1e-15 in practice) — essentially exact equality, consistent with the coordinate-descent solving the first-order condition `yᵢ(Σy)ᵢ = bᵢ` to convergence tolerance `tol=1e-14` on weight changes per sweep. There's also a "hardening" property test in `test_infeasible_and_properties.py` that checks equal risk contributions plus the exact Euler identity on random SPD matrices, not just a fixed example — i.e. the equal-contribution property is verified structurally, not just on one hand-picked covariance.

**Q16. What box/budget constraints does the SLSQP-based optimizer actually support? Name the functions and cite the code.**

`min_variance_constrained(cov, bounds=(0,1), budget=1.0)` supports per-asset box bounds (`bounds` — a scalar tuple applied to every asset, e.g. `(0,1)` for long-only, or a per-asset list) and a budget equality `sum(w) = budget`. `target_return_portfolio` adds a return equality `w'μ = target` on top of the same box+budget machinery, with a necessary-condition pre-check that raises an informative `ValueError` when `target` is outside the range achievable within the box bounds. `target_risk_portfolio` instead maximizes return subject to `sqrt(w'Σw) ≤ target_vol` as a smooth inequality (imposed on variance) plus box+budget. `max_sharpe_constrained` maximizes `(w'μ − rf)/sqrt(w'Σw)` subject to `sum(w)=1` and bounds.

**Q17. Does this project's equity optimizer support sector limits or turnover constraints directly? What about leverage?**

Reading `mvo.py`, the only constraint primitives implemented are box bounds (per-asset lower/upper), a budget (sum) equality, a target-return equality, and a variance/vol-cap inequality — there is no dedicated sector-limit or turnover-constraint parameter in the optimizer itself. Turnover is instead handled downstream in `backtest.py`, where the walk-forward engine charges two-sided turnover `Σ|w_new − w_drift|` at `cost_bps` and DESK_GUIDE recommends a turnover *budget* (e.g. ≤2x/yr) as an operational governance check rather than a solver constraint. Leverage in the risk-parity context is handled separately by `vol_target_overlay(max_leverage=...)`, which caps the scaling factor `L = σ_target/σ(w)` applied after ERC weights are computed — a post-hoc cap, not a constraint inside the optimization itself. Sector limits would have to be expressed as generic linear inequality constraints added to the SLSQP `cons` list by a caller; the module doesn't expose a named "sector cap" argument.

**Q18. How does the FX project's optimizer extend the constraint set beyond the equity project, and why?**

`python/fx/07-portfolio-optimization/docs/METHODOLOGY.md` §5 states the FX SLSQP solvers add an "FX-native constraint set": a net budget that can be 0 (for a dollar-neutral long-short book) rather than fixed at 1, a gross-leverage cap `Σ|w| ≤ G` solved smoothly via a long/short split `w = p − q`, and per-currency boxes. This differs from the equity project because FX portfolios are naturally long-short (carry longs high-yielders and shorts funders) — a fixed budget-of-1 long-only frame doesn't fit that structure, so the FX module generalizes the budget constraint and adds the gross-leverage cap needed to bound a book with no natural net-notional limit.

**Q19. What does long-only constraint actually do to the estimation-error problem, according to the assumptions register?**

Assumption 6 in `METHODOLOGY.md` states that long-only box constraints act as *implicit shrinkage* (citing Jagannathan & Ma 2003): they are helpful against noise (the raw tangency's true Sharpe improves from 0.14 unconstrained to 0.30 when long-only is imposed) but they also cap the achievable frontier (the long-only frontier has vol ≥ the unconstrained frontier at every return level) and concentrate portfolios at the top of the frontier (max weight reaching 0.74/0.75). So the constraint is a genuine estimation-error mitigant, not a free lunch — it trades frontier optimality for robustness, and can itself produce concentrated (low effective-N) portfolios at extreme target returns.

**Q20. What real edge case involving near-singular/correlated covariance matrices does VALIDATION.md document, and how does the code respond?**

VALIDATION.md §5 ("Numerical limits") states that perfectly correlated or `T ≤ N` sample covariance matrices are singular, and the closed-form solvers (`min_variance_weights`, `tangency_weights` via `_solve_spd`) raise an informative `ValueError` directing the caller to `psd_repair` or `ledoit_wolf_cc` rather than failing silently or returning garbage from a failed Cholesky. `covariance.py`'s `_solve_spd` wraps `np.linalg.cholesky` in a try/except specifically for this. §3 also documents a concrete stress test: a 5×8 panel (T < N, sample covariance singular, condition number ∞) where Ledoit-Wolf still returns a finite-condition PSD matrix and long-only min-variance solves successfully on it (`test_short_window_singular_sample_but_lw_invertible`).

**Q21. Walk through what `psd_repair` does mechanically and why eigenvalue clipping (rather than, say, adding a ridge term uniformly) is the chosen repair.**

`psd_repair(cov, eps=1e-10)` symmetrizes the input, eigendecomposes it, floors every eigenvalue at `eps * max_eigenvalue` (or an absolute floor `eps` if the matrix is all-zero/negative), and rebuilds `V·diag(clipped)·Vᵀ`, re-symmetrized. This is a *relative* floor tied to the matrix's own scale rather than a fixed ridge added to the diagonal, so it doesn't distort well-estimated large eigenvalues and only intervenes on the smallest, most estimation-error-prone directions — exactly the eigenvalues that a near-singular correlated basket would otherwise blow up when inverted. The module's docstring is explicit that this is "a projection, not an estimator" — used purely for numerical hygiene after the actual covariance estimate (sample, EWMA, LW) has been computed, not as a substitute for shrinkage.

**Q22. What specifically goes wrong in a highly correlated basket in a crisis, quantitatively, per the project's crisis-regime results?**

`VALIDATION.md` §5.1 reports realized average pairwise correlation jumping from 0.470 (calm) to 0.867 (crisis) in the seed-1 panel. With correlations pushed toward 1, diversification degrades for every strategy (all books' crisis vol runs ~2.5–3x their calm-period vol), but the degradation is uneven: broad, diversified books (ERC effective N 7.8, equal-weight 8.0) degrade "gracefully," while the concentrated raw-mean tangency book (effective N 1.57) — which was already carrying idiosyncratic risk it thought it was diversifying away — posts the worst crisis vol (0.568), worst max drawdown (0.470), and worst crisis total return (−0.368) of any strategy tested. Min-variance, which holds the low-beta corner by construction, degrades least (vol 0.380, MaxDD 0.308).

**Q23. Why is single-factor covariance always guaranteed PSD, and what's the trade-off documented against Ledoit-Wolf?**

`single_factor_cov()` constructs `Σ = var_m·bb' + D` where `D` is a diagonal of non-negative residual variances (`np.maximum(d, 0.0)`); a rank-one outer product plus a non-negative diagonal is PSD by construction regardless of the input data, so this estimator can never produce a singular or negative-eigenvalue result even from very short or degenerate samples. `METHODOLOGY.md` §3 documents the trade-off: it is guaranteed PSD and low-variance, but biased — it forces all comovement through one factor and misses sector-block structure (the project's synthetic truth has three sectors), whereas Ledoit-Wolf's constant-correlation target captures the average cross-sectional correlation without collapsing everything to a single factor.

**Q24. How does the FX project handle the fact that mean-variance is blind to skew — specifically for carry's crash risk — and why is variance alone considered insufficient there?**

FX `METHODOLOGY.md` §5 notes that variance is symmetric while carry's risk is not: carry has a modest historical vol but a fat, negative-skew left tail (currency crashes), so mean-variance "sees carry's premium and modest vol and loads up — precisely because its skew lives outside the variance." The project's answer is CVaR-constrained sizing via the Rockafellar-Uryasev (2000) linear-programming formulation over historical scenarios, solved with `scipy.optimize.linprog` (HiGHS), giving both a "min CVaR" and a "max μ'w s.t. CVaR_α ≤ c" mode. This mirrors the equity project's Assumption 3 (quadratic-utility/no-higher-moments breaks when a short-vol, negative-skew asset looks ideal to MVO) but the FX module goes a step further by adding an explicit non-variance risk measure rather than only documenting the limitation.

**Q25. Contrast the two Ledoit-Wolf shrinkage targets used in the equity vs FX projects, and explain why each is the natural choice for its asset class.**

The equity project shrinks the sample covariance toward a *constant-correlation* target (`ledoit_wolf_cc`, `F` keeps sample variances, replaces every off-diagonal correlation with the average correlation `rbar`) — natural for a single-market equity universe where all pairs share a broadly similar comovement level. The FX project instead shrinks toward a *scaled identity* with intensity `δ = min(b²,d²)/d²`, described as "minimax-safe" — appropriate because FX pairs don't share a single natural "market" correlation the way same-market equities do (idiosyncratic pairs like EUR-CHF break a constant-correlation assumption), so shrinking toward "no correlation, own variance" is the safer default before layering in the explicit risk-on/off one-factor structure for stress logic.

**Q26. What diagnostic thresholds does DESK_GUIDE.md specify for gating a rebalance, and what do they protect against?**

`DESK_GUIDE.md` §2 and §7 list several gates checked before a trade is released: Ledoit-Wolf shrinkage intensity δ (alert if δ > 0.5 — spiking toward 1 means the sample covariance is uninformative and should be investigated before being trusted), condition number (alert if > 1e3, flagging near-singular inputs of the kind in Q20/Q26), effective N ≥ 4 (a concentration floor — guards against the effective-N-1.57 collapse seen in the raw-tangency crisis results), ex-ante vol versus target, and realized risk contributions staying within ±25% of budget (catching drift away from the risk-parity mandate between rebalances). Together these turn the estimation-error and singularity failure modes documented in METHODOLOGY.md/VALIDATION.md into concrete, checkable pre-trade controls rather than only a backtest observation.


## Round 12 — Algorithmic Trading & Execution

**Q1. Why can't a portfolio manager just send a large parent order to the market as one block trade?**

Two costs work against you. First, market impact: a large order consumes the available liquidity at each price level, so the act of trading pushes the price against you (you buy higher / sell lower than the price that existed before you started). Second, information leakage: a visible large order signals your intent, inviting other participants to trade ahead of you and worsening your fill. `python/equity/08-algo-execution/src/eq_algo/intraday.py` models exactly this by splitting a parent order into a per-bucket child-order schedule and charging each child a temporary and permanent impact cost rather than filling the whole order at one price.

**Q2. What is TWAP and how is it implemented in this project?**

TWAP (time-weighted average price) spreads a parent order into equal-sized slices over a fixed number of buckets, independent of how volume is actually distributed through the day. `twap_schedule(parent_qty, n_buckets)` in `eq_algo/benchmarks.py` literally returns `np.full(n_buckets, parent_qty / n_buckets)`. As a benchmark rather than a schedule, `twap(prices)` is just the simple mean of the bucket mid prices over the horizon.

**Q3. What is the main weakness of TWAP scheduling?**

It ignores where volume actually concentrates during the day. Equity trading volume is empirically U-shaped — heavy at the open and close, thin at midday — which `intraday.py`'s `u_shaped_profile` models explicitly (`p_j ∝ 1 + c(2u_j−1)^2`). A TWAP schedule trades the same size in a thin midday bucket as in a heavy open/close bucket, so each equal-sized clip represents a much larger fraction of available liquidity (and thus more impact) in the thin buckets than a volume-aware schedule would take on.

**Q4. What is VWAP, both as a benchmark and as a schedule, and how does the project distinguish the two?**

As a benchmark, `vwap(prices, volumes)` in `benchmarks.py` computes `sum(p*v)/sum(v)` over the market tape for the order's horizon — the price a "fair" execution should track. As a schedule, `vwap_schedule(parent_qty, profile)` sizes each child order proportional to the expected volume profile (`q_j = X * p_j / sum(p)`), so the order's own fills track the market's volume distribution by construction. The module docstring is explicit that these are separate concepts sharing a name.

**Q5. Why does VWAP scheduling usually outperform TWAP for a liquid name, and what does it depend on?**

Because it concentrates trading where the market is naturally absorbing more volume (open/close), each child order represents a smaller fraction of that bucket's liquidity than an equal-sized TWAP clip would, so participation-driven impact is lower on average. The 200-replication horse race in `docs/VALIDATION.md` §4 shows VWAP (29.6 ± 109.6 bps) essentially matching TWAP (29.8 ± 114.2 bps) on mean cost in that demo — the difference is within Monte Carlo noise there — but this depends entirely on the input `profile` being an accurate forecast of where volume will actually land; `vwap_schedule` takes that profile as a parameter, and a bad forecast (e.g. an index-rebalance day where the close's share doubles) makes the schedule lag the tape, which `docs/DESK_GUIDE.md` calls out explicitly under "Index rebalance days."

**Q6. What is participation-rate (POV) scheduling, and what trade-off does it manage?**

`pov_schedule(parent_qty, market_volumes, participation)` in `benchmarks.py` trades a fixed fraction (`participation * V_j`) of each bucket's volume until the parent is filled, capped at that percentage in every bucket. The trade-off is urgency versus impact: a higher participation rate finishes the order faster but takes a larger share of each bucket's liquidity (more impact per clip); a lower rate spreads out longer, reducing per-clip impact but exposing the unfilled remainder to more price drift over a longer horizon (timing risk). `docs/DESK_GUIDE.md` sets the default participation cap at 10% ADV and treats it as a "the desk's cue to split across days" limit rather than a knob to push arbitrarily higher.

**Q7. What happens in `pov_schedule` if the parent order is larger than the day's capacity at the chosen participation rate?**

It raises an informative `ValueError` naming the shortfall: `f"parent order of {parent_qty:.0f} shares cannot complete at {participation:.1%} participation: day capacity is {max_qty:.0f} shares..."`, advising the caller to split across days or raise the cap — this is tested explicitly (`test_parent_order_larger_than_day_volume_informative_error` in `docs/VALIDATION.md` §6). This is a deliberate design choice: silently under-filling would hide a scheduling error, so the function fails loud with an actionable message rather than returning a schedule that can't be executed.

**Q8. What is implementation shortfall (IS), and why is it the benchmark a real execution desk is actually measured against?**

Implementation shortfall (Perold 1988) compares the return of a hypothetical "paper" portfolio traded instantly and costlessly at the decision price against what was actually realized. Unlike VWAP or TWAP slippage — which only measure how you traded relative to the market during the order's life — IS captures the full cost of the entire decision-to-fill process, including the price drift before the order even started and the cost of any unfilled tail. `eq_algo/tca.py`'s `is_decomposition` implements this as three additive components: delay (decision price → arrival price drift), trading (spread + impact + intraday drift while executing), and opportunity (cost of the unfilled remainder, marked at the end-of-horizon price).

**Q9. What are the three components of the Perold IS decomposition, in the actual formulas used here?**

From `tca.py`, with side `s`, parent size `X`, fills `(q_j, p_j)`, arrival price `p_a`, decision price `p_d`, and end-of-horizon price `p_T`: delay `= s*(p_a − p_d)*X` (how much the price moved between the investment decision and when the order was released to the desk); trading `= s*Σ q_j*(p_j − p_a)` (spread, impact, and intraday drift incurred while filling); opportunity `= s*(X − Q)*(p_T − p_a)` (the cost of not having filled the full order, marked against the unfilled quantity). The module docstring states, and a dedicated test confirms to 1e-10, that the three sum exactly to the total IS.

**Q10. How is the Perold identity validated, and why does that matter?**

`docs/VALIDATION.md` §3 lists `test_is_components_sum_to_total_exactly` (25 random orders including partial fills, tolerance 1e-10) and a hand-checked toy order — 60 shares filled at 10.2, 20 at 10.3, 20 unfilled — that decomposes to delay 10 + trading 10 + opportunity 8 = 28 = 280 bps exactly (`test_is_decomposition_hand_checked_toy_order`, tolerance 1e-12). This matters because a decomposition that doesn't sum exactly to the total would be useless for attributing "why" a trade cost what it cost — a TCA committee needs to trust that delay, trading, and opportunity are a clean partition of the total, not overlapping or leaky categories.

**Q11. What market-impact model does this project actually implement — is it linear, square-root, or something else?**

It's a hybrid, per the Almgren-Chriss / Huberman-Stanzl split documented in `docs/METHODOLOGY.md` §2 and implemented in `intraday.py`: permanent impact is linear in participation, `perm_move = side * perm_coef * sigma_daily * (q_j / day_volume) * mid`, and temporary impact (which only affects the fill price and fully reverts by the next bucket) follows the empirical square-root law, `temp_frac = temp_coef * sigma_daily * sqrt(q_j / day_volume)`. The two shapes are not interchangeable choices — Huberman-Stanzl (2004) shows permanent impact *must* be linear or the model admits price-manipulation round-trips, while the square-root law is the empirically-robust shape for temporary cost (Almgren et al. 2005; Toth et al. 2011).

**Q12. Why does the project use square-root temporary impact in the simulator but linear temporary impact in the Almgren-Chriss optimizer — isn't that inconsistent?**

Yes, and it's deliberate, not a bug. Classic Almgren-Chriss (`almgren_chriss.py`) is solved analytically under *linear* temporary impact (`eta`), which is what yields the closed-form `x_j = X*sinh(κ(T−t_j))/sinh(κT)` trajectory. The `IntradayMarket` simulator, used to *evaluate* any schedule (AC's or otherwise), charges the empirically-truer square-root temporary impact. `docs/METHODOLOGY.md` calls this "optimise in a tractable model, evaluate in a richer one" and states it "mirrors desk reality." `docs/VALIDATION.md` §8 quantifies the consequence: concentrating flow (an aggressive 2-bucket schedule) pays 37.6 bps mean vs TWAP's 29.8 in the 200-replication demo, more than a linear-impact model would have predicted — so the *ranking* of schedules by variance is robust, but the absolute expected-cost numbers are model-dependent.

**Q13. What is the closed-form Almgren-Chriss trajectory, and what do the two limiting cases of the risk-aversion parameter λ give?**

`ac_trajectory` (via `ac_kappa`) implements `x_j = X * sinh(κ(T−t_j)) / sinh(κT)`, where `κ` solves `cosh(κτ) = 1 + κ̃²τ²/2` and `κ̃² = λσ²/η̃`. As λ → 0 (no risk aversion), `κ → 0` and the trajectory collapses to equal slices — TWAP exactly (`test_zero_risk_aversion_ac_is_twap`). As λ → ∞ (maximal urgency), the schedule front-loads more than 99.9% of the order into the very first slice (`test_higher_risk_aversion_front_loads`, `test_ac_infinite_urgency_limit_dumps_first_slice`). Both limits are unit-tested, confirming AC is a strict generalization that subsumes TWAP as a special case.

**Q14. What trade-off does the Almgren-Chriss risk-aversion parameter λ actually let a trader dial, and how is that shown concretely?**

λ trades expected cost against cost variance: minimizing `E[cost] + λ·V[cost]` means a higher λ accepts a higher expected cost in exchange for a tighter distribution (less exposure to adverse price moves while the order is still working). `docs/VALIDATION.md` §4 shows the efficient frontier is monotonic across six λ values — expected cost strictly rising, variance strictly falling — with concrete numbers (E, std) in bps: (15.5, 99.0) at λ=1e-6, (21.1, 73.8) at λ=5e-6, (53.9, 36.5) at λ=5e-5. `docs/DESK_GUIDE.md` calls this table "literally the menu shown to the PM": e.g. "+3.7 bps expected buys a 35% cut in cost volatility" for AC(λ=5e-6) vs TWAP in the 300-replication paired test.

**Q15. How does AC compare to plain TWAP empirically in this project's simulator, and is the difference statistically real?**

`docs/VALIDATION.md` §4 reports 300 paired replications (5% ADV buy, 2%/day sigma): TWAP mean IS 8.3 bps (std 113.0) vs AC(λ=5e-6) mean IS 12.0 bps (std 74.0) — a 35% variance reduction for +3.7 bps of expected cost. The variance difference is tested with a Levene test (W=42.9, p=1.2e-10), and `test_ac_beats_twap_on_cost_variance_on_simulator` requires both `std_AC < 0.85·std_TWAP` and `p < 0.01` to pass — so the variance reduction is a validated, not just anecdotal, result. A separate analytic-consistency test confirms the model-implied frontier (V_AC < V_TWAP, E_AC > E_TWAP) agrees in direction with the simulated result.

**Q16. Why does the project solve Almgren-Chriss numerically (as a QP) for FX rather than using the closed-form equity solution?**

Classic closed-form AC assumes constant temporary impact η and volatility σ over the whole horizon. `python/fx/08-algo-execution/docs/METHODOLOGY.md` §5 notes that over a 24-hour FX day both vary roughly 10× across sessions (Asia/London/Overlap/NY/Late), so the FX project instead solves a bucket-specific discrete mean-variance QP — `min_n Σ η_j n_j²/τ + λ Σ σ_j² τ x_j²` — via its KKT system with an active-set loop for one-sided execution, implemented in `fx_algo/execution/optimal.py`. It's verified against the closed-form: with constant η, σ the numerical solution matches the analytic AC trajectory to 1e-9, and λ→0 reduces exactly to the liquidity-weighted (TWAP-analog) schedule.

**Q17. What execution benchmarks does the FX project use, and why not VWAP?**

`fx_algo`'s `execution/tca.py` implements arrival price (implementation shortfall), interval TWAP, and the WM/R 4pm London fix — explicitly *not* VWAP. The FX METHODOLOGY.md §1 explains why: spot FX is an OTC dealer market with no consolidated tape and no official volume print (liquidity fragments across dealer streams, ECNs, and internalisation pools), so a volume-weighted benchmark is structurally ill-defined the way it is on a lit equity exchange. The project treats "pretend a tape exists and implement VWAP" as an alternative it explicitly rejected as "structurally wrong for FX."

**Q18. How does FX execution handle the analog of a volume profile, and how does its "VWAP-reversion" signal differ from the equity version?**

Since realised market volume is unobservable in real time, `fx_algo`'s `pov_schedule` is described as a POV-*analog*: it participates at a capped rate of a *modeled* session depth profile (Asia/London/Overlap/NY/Late, each with its own spread/depth/vol in `sessions.SESSION_BOUNDS`) rather than actual printed volume. Similarly, the mean-reversion feature reverts to the running time-weighted average mid of the day (`features.reversion_to_session_mean`) instead of a volume-weighted average — described in METHODOLOGY.md as "the honest FX analog" of VWAP-reversion.

**Q19. What is transaction cost analysis (TCA) in this project, and what does it actually measure after the fact?**

TCA is the post-trade evaluation of realized execution cost against a chosen benchmark — it answers whether the schedule/algorithm choice was actually good, not just what it cost in isolation. `eq_algo/tca.py`'s `tca_report` runs the Perold IS decomposition on a completed `ExecutionResult`, and `benchmarks.py`'s `benchmark_slippage` separately reports slippage vs VWAP, TWAP, arrival, and decision price in bps. `aggregate_tca` rolls many orders' IS components (delay/trading/opportunity/total) up into mean/std/min/max/notional-weighted statistics, matching the desk-level format `docs/DESK_GUIDE.md` describes: weekly aggregation cut by strategy, side, size bucket (%ADV), and schedule type, with the realised trading-cost-vs-sigma·sqrt(Q/ADV) slope used to re-estimate the impact model's coefficient.

**Q20. How does the TCA feedback loop close back into the cost model, per `docs/DESK_GUIDE.md`?**

Weekly, `aggregate_tca` output is fit against `sigma * sqrt(Q/ADV)`: the regression slope re-estimates the square-root-law coefficient `k` that feeds both the daily backtest cost model and the capacity curve, and the intercept re-estimates the effective spread. A drift in `k` of more than 25% quarter-on-quarter triggers a capacity review. Separately, realised AC-vs-VWAP slippage is compared against the simulator's predicted distribution via `evaluate_schedules`; DESK_GUIDE is explicit that if a schedule is consistently outside its predicted band, "the impact model, not the trader, is wrong" — i.e., TCA is used to diagnose model misspecification, not just to grade traders.

**Q21. What is `slippage_attribution`, and what does it decompose per-bucket fill slippage into?**

`eq_algo/tca.py`'s `slippage_attribution(result)` breaks each filled bucket's slippage versus arrival price into three additive currency-per-share terms: `drift` (the bucket's pre-trade mid minus arrival — market noise plus the accumulated permanent impact of the parent's own earlier child orders), `spread` (the half-spread paid), and `temporary` (the square-root impact cost). These three sum exactly to the total per-bucket slippage, and a quantity-weighted `TOTAL` row aggregates across the whole order — validated to 1e-10 by `test_slippage_attribution_components_sum_exactly`.

**Q22. Where does this project's impact model documentedly understate true cost — the illiquid/thin-market edge case?**

`docs/VALIDATION.md` §8, failure mode 2 ("Impact model misspecification for illiquid names"), states this directly: the square-root law is calibrated on liquid equities, and for names traded above roughly 10% ADV participation, real costs are convex-worse than sqrt because of book depletion and information leakage — the capacity curve and POV caps understate the pain in that regime. The simulator's only guard against this is a blunt one: `IntradayMarket.execute` raises a `ValueError` if a child order would exceed 100% of a bucket's market volume, which prevents an impossible fill but does nothing to correct the cost *shape* below that hard limit.

**Q23. What happens mechanically in the simulator, per `docs/VALIDATION.md`, when volume in a bucket is zero but the schedule still routes an order there?**

`IntradayMarket.execute` raises an informative `ValueError` — "bucket {j} has zero market volume but the schedule routes {q[j]:.0f} shares there; reschedule around the halt/auction" — rather than silently producing a divide-by-zero or NaN fill (`test_zero_volume_bucket_raises_informative_error`). By contrast, a POV schedule handles the same zero-volume bucket gracefully by skipping it and continuing (`test_zero_volume_bucket_pov_skips_it`), since `pov_schedule` naturally trades zero when `participation * V_j = 0`. This distinction — hard failure for a schedule that assumed liquidity that wasn't there, versus graceful degradation for a schedule that adapts to realised volume — is exactly the operational argument for participation-based scheduling around thin or halted markets.

**Q24. Why does FX execution explicitly exclude "internalisation" from its cost model, and what does that mean for interpreting its cost numbers?**

`docs/METHODOLOGY.md` (FX) §4 states that major FX dealers internalise 60–90% of EURUSD flow against opposing client flow, so a real bank algo's realised impact can be far below what a pure external-liquidity model predicts. The project deliberately models only external liquidity consumption (assumption A7) and flags its own cost numbers as an *upper bound* for an internalising desk — "a documented simplification, not an accident." This is analogous to the equity project's illiquid-name caveat: both projects are explicit that their calibrated impact numbers are honest within the modeled regime but not a universal cost oracle outside it.

**Q25. What is the "last-look" mechanism modeled in the FX project, and why does it matter for algorithmic execution specifically?**

On dealer (non-firm) FX streams, `LastLookVenue` in `fx_algo` models a dealer holding a quoted order briefly (`hold_seconds = 2`) before accepting, with rejection probability rising logistically in how far the price has moved against the dealer during the hold (`p = expit((m − 0.6)/0.2)`), asymmetric by design — moves in the dealer's favor are filled. A rejected child resubmits at the post-move price plus an aggression penalty. METHODOLOGY.md §6 quantifies the resulting "trap": despite quoting a 40% tighter spread, the last-look stream is cheaper for uninformed flow (−0.018 pips) but strictly more expensive for informed flow (+0.275 pips), because adverse selection converts the quoted spread saving into rejection cost. An execution algorithm that routes purely on quoted spread, ignoring rejection dynamics, will systematically mis-rank venues for informed (alpha-bearing) flow.

**Q26. How does the WM/R 4pm fix benchmark illustrate the interaction between execution scheduling and benchmark design?**

`docs/METHODOLOGY.md` (FX) §2 explains that the pre-2015 fix was computed over just a 60-second window, which several dealers exploited by trading on client fix-order information inside that window to move the print in their favor ("The Cartel," ~$10bn in fines). The 2015 reform widened the window to 5 minutes specifically to make it harder to move. The project models the post-2015 mechanics: `sessions.fix_window_mask` and `tca.fix_benchmark` compute the fix as the TWAP of mids over a 5-minute window centered on 16:00 London, and a dedicated `fix_schedule` executes flat across exactly that window — measured tracking error is ≈0.00 pips versus ≈9.8 pips for a naive 3-hour TWAP. The lesson generalizes beyond FX: any execution schedule that trades *inside* the window used to compute its own benchmark can move that benchmark, which is why benchmark design (window length, what counts as "inside") is itself a control, not just a measurement choice.


## Round 13 — Regime-Switching Quant Strategy

**Q1. What does "regime-switching" mean as a trading-strategy premise, and why should it beat a strategy with one fixed rule set?**

The premise is that markets are not stationary: they spend long stretches in a calm, trending, low-correlation state and shorter stretches in a stressed, high-vol, high-correlation state, and the statistics that matter for a portfolio — mean return, volatility, cross-asset correlation — differ materially between the two. A single fixed rule (e.g. always fully invested, or a static vol target) is calibrated to the blend of both states and is therefore wrong in both. A regime-aware strategy infers which state is currently more likely and conditions its behavior — position size, which book it trades — on that inference, so it can behave differently in a 2022-style grinding bear than in a calm 2021 bull instead of applying one policy everywhere.

**Q2. Concretely, which regime-detection method does this project implement — is it a Markov-switching model, a volatility-threshold classifier, or something else?**

It is a Gaussian Hidden Markov Model (HMM) fit by Baum–Welch EM, implemented from scratch in `src/eq_regime/hmm.py`, sitting downstream of a features → PCA → HMM pipeline (`docs/METHODOLOGY.md` §1). A hidden state `s_t` follows a first-order Markov chain with transition matrix `A`, and observations are conditionally Gaussian, `x_t | s_t = k ~ N(μ_k, Σ_k)`. A volatility-threshold rule (the 200-day moving average) and a GMM (a "no-dynamics" special case of the HMM) are both present in the codebase, but only as an explicit benchmark and a k-selection/cross-check tool respectively, not as the trading model.

**Q3. Why an HMM instead of the two obvious alternatives — a simple threshold rule and a static Gaussian mixture model?**

`docs/METHODOLOGY.md` §2 lays out the comparison directly: a threshold rule (VIX > 25, price vs 200d MA) has no notion of persistence or uncertainty — it is a binary flag with an arbitrary cutoff. A GMM classifies every day independently (i.i.d.), so its posterior "flip-flops" day to day with no transition structure. The HMM adds exactly one thing on top of the GMM's emission model: a transition matrix, which gives regime persistence (expected duration `1/(1-p_ii)`), a filtered probability that blends today's evidence with yesterday's belief, and a principled way to avoid noise-driven flips. The threshold rule is kept in the codebase specifically as "the honest benchmark": if the HMM can't beat it net of costs, its extra complexity isn't earning its keep.

**Q4. What role does the GMM in `gmm.py` play if it isn't the trading model?**

Two roles, both stated explicitly in `docs/METHODOLOGY.md` §2. First, BIC-based selection of the number of hidden states `K`, since the GMM is algebraically the special case of the HMM where every row of the transition matrix is equal (no dynamics), which makes it a cheap way to test how many clusters the data supports before paying for a full Baum–Welch fit. Second, an independent cross-check of the HMM's emission parameters — the project validates the from-scratch GMM against `sklearn.mixture.GaussianMixture` (per-observation log-likelihood delta measured at 3.4e-14, `docs/VALIDATION.md` §1) and separately validates the HMM against `hmmlearn`, so the two implementations corroborate each other.

**Q5. What is a "filtered" probability, what is a "smoothed" probability, and why does this project insist that only filtered probabilities ever reach the trading strategy?**

Filtered: `α̂_t(k) = P(s_t = k | x_{1..t})`, computed causally from the forward recursion using only data up to and including `t`. Smoothed: `γ_t(k) = P(s_t = k | x_{1..T})`, computed from the backward pass over the *entire* sample, meaning it conditions on future observations. `docs/METHODOLOGY.md` calls this "the central honesty point of the project": smoothed probabilities look far cleaner around regime turns because they've effectively seen the future, so backtesting on them is lookahead bias that overstates performance. Every trading signal in `detection.py` and `backtest.py` is built exclusively from filtered probabilities, and this is not just a stated convention — it's test-enforced.

**Q6. How is the filtered-vs-smoothed causality claim actually tested, rather than just asserted in the docs?**

Via mutation tests. `docs/VALIDATION.md` §3 and §6.1 describe perturbing (mutating) observations *after* time `t` and checking what happens to the outputs at `t`: the filtered probability at `t` must be bit-identical (drift ≤ machine epsilon) because it mathematically cannot depend on future data, while the smoothed probability at `t` is expected to shift by "4+ orders of magnitude more," which is the quantitative proof it does use the future. This is implemented in `tests/test_detection.py::TestCriticalCausality` at the detection layer and `tests/test_backtest.py::TestNoLookahead` at the full-pipeline/ledger layer, so the "no lookahead" claim is checked end to end, not just at one module boundary.

**Q7. Show me the concrete numeric evidence that smoothed probabilities are "clairvoyant" around a regime turn.**

`docs/VALIDATION.md` §3 gives a specific table around the first true bull→bear transition (day index 320, 2016-03-25): at t−1 (still in the "transition" true state) the filtered `P_bear` is 0.103, but the smoothed `P_bear` is already 0.743 — three-quarters bear a full day *before* the true regime changes, because the smoother has read what happens next. The filtered probability only crosses into clearly-bear territory at t+1 (0.973). That gap between 0.103 and 0.743 at the same date is the entire lookahead-bias problem in one row.

**Q8. What is the core risk of any regime-based strategy — the failure mode that makes these strategies look great in a backtest and fail live?**

Overfitting to regime boundaries that were only obvious in hindsight. If you choose where the "bear" episodes start and end by eyeballing the whole price history, the classifier will look perfect in-sample — it's fit to data it already knows the outcome of — and can degrade sharply out-of-sample because a live detector has to infer the state from noisy, incomplete evidence in real time, without the benefit of already knowing how the story ends. This project's own methodology names the analogous statistical version of that risk explicitly: smoothed probabilities are exactly "peeking at the whole history" in miniature, applied per-day instead of per-regime-boundary, and the project treats using them for trading as the single worst mistake a regime strategy can make.

**Q9. How does this project's backtest guard against that overfitting risk structurally, not just at the level of "don't use smoothed probabilities"?**

Two structural defenses, per `docs/METHODOLOGY.md` and `backtest.py`. First, walk-forward, expanding-window refitting: `walk_forward_backtest()` refits the HMM every `refit_every=63` trading days (a quarter) on an expanding window, and the docstring in `backtest.py` states the timing convention explicitly — "the HMM used at `t` was last refitted on features strictly BEFORE its refit date and never sees data after `t`." Second, a null-data false-positive guard (`docs/VALIDATION.md` §4): on synthetic no-regime GBM panels, the walk-forward strategy is required to *lose* money relative to buy-and-hold (measured −4.4%, −7.6%, −3.1% CAGR excess across three seeds) rather than manufacture spurious alpha, and BIC on that null data is required to prefer K=1. Both are asserted in `tests/test_null_guard.py`, so "the model doesn't hallucinate regimes when there aren't any" is a testable property, not a hope.

**Q10. Isn't picking K (the number of regimes) itself a place where hindsight overfitting can creep back in?**

Yes, and the docs treat it as a governance problem rather than a purely statistical one. `docs/VALIDATION.md` §5 lists "overfitting k" as a named failure mode: BIC on a finite sample can select a K that fits noise, and more states mechanically produce shorter durations and more churn. The mitigation in `docs/DESK_GUIDE.md` §3 is procedural: K is fixed at 3 between annual reviews, and changing it requires BIC evidence *and* economically distinct state volatilities *and* out-of-sample confirmation — not just "BIC went up." `docs/METHODOLOGY.md` assumption 3 adds the failure-mode detail: if the true world has more states than the model allows, EM doesn't fail loudly, it silently absorbs the extra structure into existing states (e.g. a 2-state fit on a 3-state world merges "calm" and "transition," understating risk right before turns — quantified in `docs/VALIDATION.md` §5.4).

**Q11. Walk-forward validation sounds expensive — what specifically does `walk_forward_backtest()` do differently from a full-sample fit?**

Looking at `src/eq_regime/backtest.py::walk_forward_backtest`, it builds point-in-time features (`build_features`), then calls `expanding_fit_detect(..., min_train=252, refit_every=63)`, which fits the HMM only on an expanding training window and refits it periodically rather than once on the whole series. Contrast that with the "parameter recovery" exercise in `docs/VALIDATION.md` §2, which explicitly fits a full-sample HMM to check the model can recover known synthetic ground-truth parameters — that full-sample fit is a calibration sanity check, never the version used to generate the P&L numbers in §7. The walk-forward version is the only one whose returns are reported as "the backtest."

**Q12. Why are log returns used for return aggregation in this backtest instead of simple returns?**

Because log returns are additive across time and simple returns are not: the log return over two periods is exactly `r_1 + r_2`, while the two-period simple return is `(1+r_1)(1+r_2) - 1 ≈ r_1 + r_2` only approximately (the cross term `r_1 r_2` is dropped). That makes log returns the natural unit for anything that sums or averages returns over time or across assets — feature construction (`build_features`), the HMM's Gaussian emission model, and vol targeting in `strategy.py::vol_target_scale` all consume log returns, and `docs/METHODOLOGY.md` §6 states this as a project-wide convention: "Daily log-returns; vols annualised ACT/252." Simple returns are still needed elsewhere, because P&L compounding is a multiplicative, not additive, process — which is exactly what `np.expm1` bridges (see Q13).

**Q13. Where in the code does `np.expm1` show up, and what numerical problem does it solve?**

In `src/eq_regime/backtest.py::walk_forward_backtest`: `asset_log_ret = np.log(prices / prices.shift(1)).iloc[1:]` computes log returns once, and then both `log_ret = asset_log_ret.mean(axis=1)` (fed to the HMM/vol-targeting side) and `simple_ret = np.expm1(asset_log_ret).mean(axis=1)` (fed to `run_ledger` for compounding) are derived from that single computation. `expm1(x)` computes `exp(x) - 1` directly rather than as two separate floating-point operations, which matters because for small `x` (a typical daily log return is on the order of 1e-3 to 1e-2), `exp(x)` is very close to 1, and subtracting 1 from it afterward cancels almost all of the significant digits — catastrophic cancellation. `expm1` uses a numerically stable formula/series near zero so the small daily return is recovered to full precision instead of losing digits to rounding.

**Q14. The module docstring says returns are converted "internally" for compounding — walk through why the ledger needs simple returns specifically, not log returns.**

`run_ledger()` compounds P&L as `equity = initial_equity * np.cumprod(1.0 + net)`, i.e. it treats each period's return as multiplicative growth: a portfolio holding weight `w` earns `w * r_simple` of the asset's simple return, and equity multiplies by `(1 + net_ret)` each day. That identity — `equity_t = equity_{t-1} * (1 + r_t)` — is only true for simple returns; log returns would need to be exponentiated and summed instead, or the ledger's cost/weight arithmetic (which is linear in return, e.g. `cost = cost_rate * |Δw|`, `gross = w * next_ret`) would need to change form. So the pipeline uses log returns everywhere the additive property is wanted (features, vol estimation) and converts to simple returns, via the numerically careful `expm1`, exactly at the one place — ledger compounding — where the multiplicative identity is what's actually needed.

**Q15. The task description mentions this file was recently cleaned up around this log-return computation — what did that consolidation actually look like in the current code, and why does it matter?**

The current `walk_forward_backtest` computes `asset_log_ret` exactly once and derives both series it needs from that single array — `log_ret` by taking the mean directly, `simple_ret` by applying `np.expm1` to the same array before averaging — rather than computing log returns and simple returns via two independent expressions (e.g. a second, separately-written `prices / prices.shift(1) - 1` for the simple-return side). That's a small but meaningful numerical-hygiene fix: two independently written return computations can silently drift out of sync (different `shift`/`iloc` alignment, different NaN handling at the warm-up row) and are strictly more surface area to audit for correctness, whereas deriving both series from one already-log-returned array guarantees they are the same 252 dates with the same indexing by construction, with `expm1` doing the precision-preserving conversion rather than a second lossy log/subtract round-trip.

**Q16. What tests specifically defend the `run_ledger` arithmetic, beyond just checking it runs without error?**

`docs/VALIDATION.md` §6 lists "ledger arithmetic: exact hand-computed scenario, entry cost included" as a specific edge-case test in `tests/test_edge_cases.py`, meaning the test constructs a small scenario with known weights and returns and checks the ledger's `net_ret`/`equity` columns against a value computed by hand, not just against another run of the same code. `run_ledger`'s own docstring is precise about the two conventions that would otherwise be easy to get subtly wrong: `dw.iloc[0] = w.iloc[0]` (entering from flat charges a cost on day one) and the shift-by-one alignment where weight decided at `t` earns the return realized from `t` to `t+1`, both of which are exactly the kind of off-by-one that "hand-computed" tests are designed to catch.

**Q17. What is the hysteresis band in `strategy.py`, and what specific problem does it solve?**

`hysteresis_regime()` implements a two-threshold rule on the filtered bear probability: enter the bear state only when `p_bear` rises strictly above 0.70, and exit only when it falls strictly below 0.30; inside the band `[0.30, 0.70]` the previous state simply persists. This replaces a naive rule ("bear if `p_bear > 0.5`"), which flips every time noisy probability crosses one line — near a regime boundary the filtered probability oscillates around that line for days, so a single-threshold rule pays transaction costs on each crossing while capturing no real signal. `docs/METHODOLOGY.md` §3 quantifies the effect directly: on a noisy probability path, turnover falls from 263 trades to 48 (−82%) with hysteresis, while on a genuine two-switch path turnover falls from 24 to 8 and both real switches are still captured.

**Q18. Are the hysteresis boundary values themselves — exactly 0.70 or exactly 0.30 — tested, or only the general behavior?**

Tested precisely, and the convention is deliberately strict-inequality on both ends. `strategy.py::hysteresis_regime`'s docstring states: "at exactly `p == enter` the rule does NOT enter (strict `>`), at exactly `p == exit` it does NOT exit (strict `<`)"; the band is closed. `docs/VALIDATION.md` §6 lists this as its own edge case — "probabilities exactly at hysteresis thresholds: no flip (strict inequalities, documented convention)" — with test coverage in `tests/test_edge_cases.py::TestBoundaries` and `tests/test_strategy.py`. That level of care about a single inequality direction is the kind of detail that matters in practice: get it backwards and a probability sitting exactly on the boundary trades every day.

**Q19. How does the strategy translate a hysteresis-confirmed regime into a portfolio weight, including the "unconfirmed bear" case?**

`regime_target_weight()` in `strategy.py` maps the HMM's raw `argmax` label (bull/transition/bear) to a base weight (1.0/0.5/0.0), but the hysteresis flag *overrides* that label: whenever `bear_flag[t]` is True the weight is forced to `bear_weight` regardless of the raw label, and whenever it's False, an unconfirmed "bear" label (probability still inside the band, not yet past 0.70) is floored at `transition_weight` rather than allowed to fully de-risk. That means full de-risking only ever happens through the confirmed hysteresis band, never off a single noisy argmax flip — the docstring states this explicitly: "de-risking obeys the banded rule rather than raw argmax flips."

**Q20. What does vol targeting add on top of the regime weight, and why is it computed only from trailing (past) data?**

`vol_target_scale()` scales the regime-conditional weight by `target_vol / trailing_21d_realized_vol`, clipped at a maximum leverage of 1.5x. It's deliberately trailing-only — using returns "up to and including `t`" — so that the scale decided at the close of `t` is a genuine ex-ante forecast applied to the position held over `t → t+1`, consistent with the rest of the pipeline's no-lookahead discipline; using a centered or forward-looking vol estimate here would reintroduce exactly the kind of lookahead bias the filtered-vs-smoothed distinction is designed to eliminate elsewhere. `docs/VALIDATION.md` §6 separately validates that this actually hits its target: "vol targeting realizes the ex-ante target within 15% on constant-vol data."

**Q21. How does a regime-switching strategy differ from a plain trend-following or mean-reversion strategy applied on its own?**

A trend-following or mean-reversion rule applies one fixed logic to price action everywhere — e.g. always buy strength, or always fade extremes — regardless of the surrounding market context. A regime-switching strategy instead conditions its *entire policy* on an inferred state: `docs/DESK_GUIDE.md` §1 describes how the filtered probabilities become factor-rotation weights ("bull: momentum/size; bear: quality/low-vol; transition: blend"), meaning the strategy can effectively behave like a trend-follower in one state and like something far more defensive in another. This project even keeps the 200-day-MA rule — a classic trend rule — in the codebase specifically as a *benchmark* that the regime strategy must beat net of costs (`walk_forward_backtest` runs `ma_timing_weights` on the same dates with the same cost model), which makes the distinction concrete: one is a single always-on rule, the other is a policy that switches between behaviors based on a probabilistic state estimate.

**Q22. What is the FX-specific twist on "regime" in `python/fx/10-regime-switching`, and why does the FX project treat it as a distinct economic story rather than a generic reuse of the equity approach?**

FX regimes there are framed specifically as risk-on/risk-off (RORO): risk-on funds cheap carry trades (borrow JPY/CHF, lend AUD/NZD/EM) that grind higher with low daily vol; risk-off is that crowded position unwinding violently, with carry/EM currencies gapping down, havens bid, vol multiplying, and pairwise correlations spiking as "every USD pair trades the same one trade." `docs/METHODOLOGY.md` (FX) also carries a third, economically distinct state — a "USD squeeze" — where a dollar-funding scramble makes *everything* fall against USD, including normal safe havens, which is exactly what fails during 2008/March-2020-style episodes and is why the FX project sometimes prefers k=3 (RORO + squeeze) over a plain two-state risk-on/risk-off split.

**Q23. Given that framing, what specific book does the FX strategy trade in each detected regime?**

Per `docs/METHODOLOGY.md` (FX) §7: in `risk_on` it runs a rank-carry basket, long the top-3 yielding currencies and short the bottom-3, dollar-neutral, to harvest the rate differential while vol is low; in `risk_off` it flips to long JPY+CHF against the risk block (AUD, NZD, EM) — i.e. it explicitly cuts carry and owns the unwind rather than staying exposed to it; in `usd_squeeze` it goes long USD against everything, because that's the one state where even the usual haven currencies fail and only the dollar itself works. All books are vol-targeted at 10% annualized using a trailing 63-day covariance, capped at 4x leverage, and pay carry accrual plus pip-denominated transaction costs — this is a much richer regime-conditional action set than the equity project's single bull/transition/bear equity-weight dial.

**Q24. Does the FX project use the same filtered-probability, hysteresis-band discipline as the equity project, or something different?**

The core discipline is the same — filtered-only probabilities, enforced by an analogous mutation test (`tests/test_detection.py::test_filtered_only_past_mutation`, `tests/test_backtest.py::test_no_lookahead_mutation`) — but the confirmation rule is stricter in form: `docs/METHODOLOGY.md` (FX) §7 requires the challenger regime's filtered probability to be ≥0.70 (or the incumbent's <0.30) for **2 consecutive days**, not just a single-day threshold crossing. The doc is explicit that this buys lower turnover and flicker resistance at the price of additional detection lag, and frames it as "a governed, measured trade-off" documented quantitatively in `docs/VALIDATION.md`, i.e. the same honesty-about-costs posture as the equity project's hysteresis band, just with an added time-confirmation dimension appropriate to how fast FX risk episodes move.

**Q25. What is the real edge case unique to regime-switching strategies — the period right when the market actually flips state — and how is its cost documented here?**

The core failure mode is detection lag: a filter needs evidence to *accumulate* before it moves, so by construction it identifies a regime change some days after it has actually happened, and that gap is exactly when the strategy is still positioned for the old regime. `docs/VALIDATION.md` §5 quantifies this on the synthetic 3-state panel: mean bear-entry detection lag of 1.5 days (median 1) and mean bear-exit lag of 1.0 days, and then goes further with a "flip-aftermath" report (implemented as `flip_aftermath()` in `src/eq_regime/risk.py`) measuring the mean 10-day P&L immediately after a flip to bear: +0.2% on average but as bad as −3.2% in the worst case. `docs/DESK_GUIDE.md` §4 translates this to a real scenario: on a COVID-March-2020-style fast crash, the first de-risking trade would realistically trigger only after roughly −8 to −12% of drawdown had already happened — the doc's own framing is "the detection lag is the price of admission... a regime model is a loss limiter, not a crash predictor."

**Q26. Are there edge cases around the regime transition itself beyond simple lag — e.g. false alarms, or the model being fooled by a single noisy day?**

Yes, two distinct sub-cases are both documented and tested. First, "false alarms in corrections" (`docs/DESK_GUIDE.md` §4, citing 2015-Q3/2018-Q4 analogues): a sharp vol spike can trigger a bear entry right near the local low, and the recovery re-entry then lags — each false alarm costs "cost × 2 turns + missed rebound days," which is exactly what the flip-aftermath table is measuring. Second, single-day Gaussian-emission mis-assignment (`docs/METHODOLOGY.md` assumption 1, `docs/VALIDATION.md` §5.5): because daily returns are fatter-tailed than the Gaussian emissions the HMM assumes, a single extreme day can look overconfidently like a regime shift under the model, and the hysteresis band plus a minimum-duration filter (removing one/two-day flickers) are the two mechanisms that absorb these single-day flips before they reach the order pipe — with `docs/DESK_GUIDE.md` §3 stating explicitly that the governance response to a false alarm is *not* to widen the band ad hoc (which would invalidate the backtest) but to log the episode and review bands only on an annual cycle.


## Round 14 — The Foundations Trilogy

**Q1. `01-risk-metrics`'s README says "this is a single-asset project — portfolio-level risk (correlation, diversification) is out of scope." Concretely, what does staying single-asset let the project skip that `python/equity/03-var-es-engine` cannot?**

A single return series has one variance and no cross-terms, so every metric in F1 — historical/Gaussian/Cornish-Fisher VaR, Expected Shortfall, Sharpe/Sortino, the Kupiec backtest — is a function of one vector of numbers. `python/equity/03-var-es-engine` prices a book of correlated positions (`demo_portfolio()`, `demo_covariance()`), so its parametric and Monte Carlo VaR need a covariance (or scale) matrix and a Cholesky factorisation (`ev.parametric_var(pf.delta_exposures(), cov, ...)`, `ev.monte_carlo_var(pf, cov, ...)`) just to generate a jointly consistent set of scenarios before any VaR arithmetic starts. Assumption A9 in F1's `docs/METHODOLOGY.md` names this directly: "Real books hold multiple correlated positions; portfolio VaR/ES require a covariance (or copula) model and are *not* simply the sum of single-asset VaRs." F1 is a smaller problem by construction, and that smallness is the point — it isolates "which distributional assumption is right" from "how do positions co-move," so a reader can see the first question answered cleanly before the second one is layered on.

**Q2. If you wanted to turn F1 into a two-asset risk engine, what would actually have to change, and why is that not a small patch?**

It is not just "run the same functions twice and add them" — VaR and ES are not additive across positions (that's exactly why ES's sub-additivity is a *theorem* worth stating in F1's `docs/METHODOLOGY.md` §4, not a triviality). You would need a covariance matrix estimate (sample or EWMA), a way to draw or simulate correlated returns (Cholesky factorisation, as `03-var-es-engine`'s `monte_carlo_var.py` does), and a joint P&L definition before any of F1's historical/Gaussian/Cornish-Fisher logic could be reused on the *portfolio* return series rather than each asset's own. The single-asset version has no correlation to estimate, no factor mapping, and no joint-tail behavior to worry about — which is precisely the machinery `03-var-es-engine`'s `portfolio.py` and `monte_carlo_var.py` exist to carry. F1 answers "which VaR method is right on this one series"; the multi-asset engine answers that question *and* "how do these series interact," and the second question is a materially bigger modeling problem, not a bigger loop.

**Q3. What is the "no-look-ahead" bug class in backtesting, in general terms — what specifically goes wrong?**

It is using information on day t that would not actually have been available to a trader at the moment the trade was placed. The most common concrete form is computing a signal from day t's closing price and then trading as if you entered the position *at that same close* — but you cannot know a bar's close until it happens, so a strategy that "trades on the close it used to generate the signal" is silently assuming perfect foresight of the rest of that day. The bug is dangerous precisely because it is invisible in the code's arithmetic: `signal * return` on the same day looks like an ordinary vectorised backtest line, and it will produce a return series, a Sharpe ratio, an equity curve — all internally consistent, all wrong, because part of every day's "prediction" is really that day's own outcome leaking into its own evaluation.

**Q4. How exactly does F2's `run_backtest` avoid this, and where is the line?**

`src/eq_signal_backtest/engine.py` computes `position = signal.shift(1).fillna(0.0)` (line 156) — the position held on day t is literally the signal value computed on day t-1, one full row earlier in the DataFrame. `docs/METHODOLOGY.md` states the mechanics precisely: `position_t = signal_{t-1}`, with `position_0 = 0` since there is no `t=-1`, and `strategy_return_t = position_t * simple_return_t - cost_t`. Because the position used to weight day t's return was decided using only information through day t-1's close, there is no path by which the strategy's day-t P&L can depend on anything that happened on day t before its close — the lag is not a validation check bolted on afterward, it is baked into the one line that defines what "trading the strategy" means.

**Q5. `docs/VALIDATION.md` describes a "detector test" — `test_cheat_profits_from_a_jump_honest_engine_does_not`. What does it actually construct, and why is that a stronger proof than reading the code?**

It builds a price series that is flat and then jumps 30% overnight, with a signal engineered to fire exactly on the jump day. A same-day ("cheat") execution captures that jump (`cheat_equity.iloc[-1] > 1.25`); the honest engine's position on the jump day was set by the *previous* day's signal, decided before the jump happened, so its equity is unchanged (`== 1.0` to `1e-9`). This is stronger than inspecting `position = signal.shift(1)` and agreeing it looks right, because it constructs a case where the bug — if present — would be *profitable and visible*, and then shows the code under test does not take the free money available to a cheating implementation. A structural assertion (`position.iloc[t] == signal.iloc[t-1]` for every `t`, per `test_position_equals_prior_day_signal_structurally`) proves the mechanism; the jump test proves the mechanism actually matters.

**Q6. Why does it matter that `position.iloc[0]` is forced to `0.0` rather than left as `NaN`?**

There is no day `t=-1` to supply a signal for day 0, so the honest answer to "what position did the strategy hold before it had ever observed a signal" is "none" — `0.0`, not an undefined value that would need special-casing downstream. `test_first_day_position_is_zero_not_nan` in `tests/test_engine.py` pins this regardless of what `signal.iloc[0]` happens to be. It is a small detail, but it is the same discipline as the shift itself: an engine that let day 0 default to `NaN` would either crash on the first `NaN * return` multiplication or, worse, silently propagate `NaN` through the whole equity curve — exactly the kind of "runs fine, produces a wrong number" failure the no-look-ahead lag is designed to prevent elsewhere.

**Q7. F2's own README calls its out-of-sample result "the headline," and it is a losing number. Why is that the right outcome to publish rather than a failure of the project?**

A backtest's job is to tell you the truth about a strategy, not to produce an impressive number — the entire evaluation machinery (train/test split, walk-forward, cost accounting, no-look-ahead) exists to answer "would this have actually worked" as honestly as possible. If the honest answer is "no," reporting that *is* the deliverable; reporting only the flattering number the machinery was built to catch would defeat the purpose of building the machinery at all. F2 states this explicitly: "A reviewer should read this as the project's *result*, not its embarrassment. The alternative — a bundled dataset tuned until the strategy worked — would demonstrate nothing except that synthetic data can be made to say anything" (`docs/VALIDATION.md` §6). A strategy that only looks good in-sample is a warning sign about the evaluation, not a result worth publishing on its own.

**Q8. What specifically happened to the Sharpe ratio between in-sample and out-of-sample on the bundled data, and what does that number teach about parameter selection?**

In-sample Sharpe was 0.87 (parameters `fast=10, slow=125` chosen by `select_best_params` on the training window only); out-of-sample Sharpe was -0.12. `docs/VALIDATION.md` calls this "not decay, it is disappearance." The mechanism is Assumption A5 in `docs/METHODOLOGY.md`: picking the `argmax` Sharpe of a grid of noisy Sharpe estimates is, in expectation, biased upward relative to any individual cell's true Sharpe — a "winner's curse" of multiple testing. The in-sample number was not measuring a real edge; it was measuring how good the best of many noisy estimates looks by chance, and the out-of-sample test is precisely the mechanism that catches that inflation instead of reporting it as skill.

**Q9. F2 also runs a walk-forward variant that re-selects parameters every year and gets a similarly negative result (-0.06 Sharpe). Why does that matter more than either number alone?**

A single 70/30 split is one draw from history — a lucky or unlucky split boundary could flatter or damn the strategy on its own (Assumption A6). Walk-forward re-runs the same discipline across seven rolling formation/trading windows, each time selecting parameters using only that window's formation slice and scoring only its (out-of-sample) trading slice, then stitches the trading segments into one continuous curve. Getting a negative result from *both* a single split (-0.12 Sharpe) and from seven independently re-optimized windows (-0.06 Sharpe, with per-window Sharpes swinging from -1.55 to +1.76) is considerably stronger evidence than either protocol alone, because the two procedures could fail in different directions and instead agree. `docs/VALIDATION.md` §6 states it plainly: "Two independent evaluation protocols reaching the same negative conclusion is considerably stronger evidence than either alone."

**Q10. The in-sample parameter grid has a broad "plateau" — 39% of cells within 25% of the best Sharpe. Isn't a plateau usually read as evidence *against* overfitting?**

Conventionally yes — a single lucky cell in a sea of bad ones is the classic signature of curve-fitting, while a broad plateau is supposed to mean the result doesn't depend on one fragile parameter choice. F2's own data shows why that reading is incomplete: the plateau is real, and the out-of-sample Sharpe is still negative. `docs/VALIDATION.md` §6 states the correction directly: "A plateau says the in-sample result does not depend on one lucky parameter cell; it says nothing about whether the whole neighbourhood is fitted to the same noise." A wide plateau distinguishes "one lucky cell" from "a lucky neighbourhood" — it cannot distinguish either of those from "the whole in-sample surface is fitted to the same noise," which is exactly what the out-of-sample collapse shows happened here.

**Q11. How does F2 quantify the effect of transaction costs on a fast-trading moving-average crossover, and what's the general lesson?**

`docs/VALIDATION.md` §7 varies `(fast, slow)` and `cost_bps` together on the same data: the fast pair (10, 50), which trades 55 times over 10 years, loses 1.21 percentage points of CAGR and 0.071 of Sharpe going from 0 to 20 bps; the slow pair (60, 250), 12 trades, loses only 0.24pp and 0.015 — roughly a 5x difference in cost sensitivity. The selected parameters (10, 125), 32 trades, sit in between at 0.17pp of drag at 5 bps. The general lesson, stated in `docs/METHODOLOGY.md` A2: cost drag scales with turnover, and the bias a cost-free backtest introduces is *systematic*, always in the same direction — it flatters over-trading parameter pairs — which is why a first draft of any signal backtest that omits costs "silently favours over-trading parameter pairs that lose to realistic costs," per the README's design-decisions section.

**Q12. Why does F2 insist `parameter_grid`/`select_best_params` always run with `cost_bps` set (default 5.0), and never cost-free?**

Because the absence of costs is a known bias of unknown size on any new dataset, not a neutral default. `docs/VALIDATION.md` §7 is explicit that whether the bias is large enough to *reverse* a ranking depends on the sample — on the bundled data it does not (the fastest pair still has the highest Sharpe at every cost level tested, including 20 bps) — but that is a fact about this one seed, not a property you can assume holds elsewhere. Running the search cost-free would answer a different, easier question ("which parameters look best if trading were free") that has no connection to what a desk could actually capture; keeping costs in the search by default means the number the search optimizes for is the same number that would matter in production.

**Q13. F2's Assumption A9 says the bundled seed (32) "was chosen because its path is close to the model's central case... not because it is favourable — it produces a negative out-of-sample result." Why call that out explicitly?**

Because seed selection is itself a place backtest-overfitting can hide, one level removed from parameter selection. `docs/METHODOLOGY.md` A9 states it directly: "Reporting a number from a seed selected for its outcome would be the backtest-overfitting failure this project is about, committed one level up in the data-generating process." If a project quietly tried several seeds and reported the one where the strategy happened to work, it would be repeating exactly the selection bias the whole train/test and walk-forward apparatus exists to guard against — just applied to the data instead of the parameters. Stating that the seed was picked for representativeness (buy & hold CAGR/vol/drawdown resembling a real large-cap equity, per `docs/VALIDATION.md` §6) rather than for outcome is what makes the negative result trustworthy rather than merely convenient.

**Q14. Why does F3 implement Black-Scholes pricing and Greeks using only `math.erf`, instead of calling `scipy.stats.norm.cdf`?**

`docs/METHODOLOGY.md` §1.4 states the reasoning directly: the standard normal CDF is not a peripheral detail of Black-Scholes, it *is* the model — `N(d2)` is the risk-neutral exercise probability, `N(d1)` weights the expected stock receipt in the replicating portfolio. Calling `scipy.stats.norm.cdf` would compute the right number in one line but leave that piece as a black box; the point of a from-scratch replication is to understand every link from the SDE to the price. `math.erf` is chosen as the stopping point rather than building the CDF from even more primitive parts because it is a single, well-tested, general-purpose special function in the standard library (`N(x) = 0.5*(1 + erf(x/sqrt(2)))`) — general-purpose math, not option-pricing-specific machinery, so using it doesn't hide any model logic the way a library CDF or a hypothetical `black_scholes_call()` would.

**Q15. F3's closed form already checks put-call parity and its own Greeks against finite differences. What does agreement with an *independent* Monte Carlo pricer prove that those internal checks cannot?**

`docs/METHODOLOGY.md` §1.2 names the gap precisely: `call_price`/`put_price` share one `_d1_d2` helper, so a bug in `_d1_d2` — a wrong sign, a misremembered `+sigma^2/2` vs `-sigma^2/2`, a subtly wrong `N(x)` — would corrupt every internal check identically and still pass all of them. Put-call parity, monotonicity, and Greeks-vs-finite-differences are all checks *internal* to the same closed-form code, so they share its blind spots by construction. The Monte Carlo pricer in `monte_carlo.py` shares no code with the closed form: it draws standard normals and builds `S_T = S0 exp((r - sigma^2/2)T + sigma sqrt(T) Z)` directly from the SDE's solution, then averages discounted payoffs. If the closed form had a sign or constant error, the two would disagree by far more than Monte Carlo noise — they would have to be wrong in *exactly the same way* to agree by coincidence across a convergence table spanning four sample sizes, each within 3 standard errors of a error bar shrinking at the correct O(1/√n) rate. An independent implementation using a genuinely different numerical method is the only kind of check that rules out a whole class of "consistently wrong in the same way" bugs; a self-consistency check, no matter how many identities it verifies, cannot.

**Q16. F3's methodology says the antithetic-variate Monte Carlo standard error must be computed from "pair averages," not from the raw `2m` draws treated as independent. Why does that distinction matter for the cross-check in Q15 to mean anything?**

Antithetic sampling deliberately makes the mirrored draws negatively correlated, so treating the `2m` values as `2m` independent observations overstates the standard error — by about a third on an ATM contract, per `docs/METHODOLOGY.md` §1.2. That inflation is invisible to a test suite: every "agrees within 3 standard errors" check keeps passing, just against a bar that is too loose to mean anything. The project's own framing generalizes the point: "a tolerance expressed in units of your own error estimate is only as trustworthy as that estimate... A validation suite that certifies 'within 3 sigma' while computing sigma incorrectly is certifying its own arithmetic, not the model." The fix — computing the standard error from the `m` independent pair averages, and testing that reported error against the estimator's actual dispersion across 60 seeds (`docs/VALIDATION.md` §2.1) — is what makes the Monte Carlo cross-check in Q15 an honest witness rather than a check that would pass no matter what the closed form did wrong.

**Q17. F3 deliberately produces a volatility smile from a model that assumes flat volatility. What is that experiment, and why publish a demonstrated failure of your own model?**

Assumption A1 in `docs/METHODOLOGY.md` states Black-Scholes assumes geometric Brownian motion with constant volatility, so log-returns are exactly normal. The experiment prices a strike ladder by Monte Carlo under a Student-t(df=4), fat-tailed return distribution instead, then reads those prices back through the constant-vol closed form to recover an "implied" volatility per strike. On the reference numbers, that implied vol comes out higher at the wings than at the money (27.6% at K=70, 22.9% at K=105, 28.6% at K=140 — a 5.2-vol-point smile) instead of the flat line constant-vol GBM predicts, qualitatively the same shape real equity- and index-option markets have shown since the 1987 crash. The project publishes this because it is the honest completion of "prove the model right where theory says it should be right, and prove it wrong where reality says it should be wrong, with the same codebase" — a from-scratch rebuild that only ever showed the model succeeding would be demonstrating agreement with itself, not understanding of where the model's assumptions actually bind.

**Q18. F1's `docs/VALIDATION.md` says the Kupiec pass for historical VaR is "close to tautological" even though it's presented as a validation result. What does that mean, and what does the test actually falsify?**

Both VaR estimators in F1 are scored on the same window they were fitted to — but historical VaR's 99% quantile is defined as (approximately) the empirical 1% worst observation of that exact sample, so of course roughly 1% of that sample exceeds it; passing Kupiec is close to true by construction, not evidence the estimator is good at predicting *new* data. `docs/VALIDATION.md` §2 states this directly: "What the in-sample test *can* falsify is a model whose distributional assumption is wrong, which is exactly what happens to the Gaussian one" — the Gaussian estimator makes a real, falsifiable claim (returns are Normal(mu, sigma)) that the data can and does contradict (60 exceptions against 25.2 expected, LR = 35.02, p ≈ 0, rejected), while the historical estimator makes almost no claim to be wrong about in-sample. A genuine test of predictive power needs an out-of-sample VaR re-estimated on a trailing window and scored on the next day's return — which is exactly the discipline `02-trading-signal-backtest` builds and `01-risk-metrics` explicitly defers to `docs/DESK_GUIDE.md` §3.

**Q19. All three foundations projects ship at least one method they know is weaker, on purpose, next to a better one. What's the common teaching pattern?**

F1 computes Gaussian VaR specifically "to demonstrate its failure" against historical VaR (`docs/METHODOLOGY.md` §3.2); F2 always runs its parameter search *with* transaction costs because a cost-free version would silently flatter over-trading parameters, and reports the in-sample Sharpe next to the out-of-sample one specifically so the gap is visible rather than hidden; F3 ships a Monte Carlo pricer whose only job is to disagree with the closed form if the closed form is wrong, and deliberately engineers a case (the smile experiment) where the model's own assumption breaks. In each case the "worse" number or method is not a mistake left in the codebase — it's instrumentation, there so the gap between "what a naive approach would tell you" and "what's actually true" is a measured quantity instead of an assertion a reader has to take on faith. That is the shared lesson across the trilogy: doing quant work honestly means building the thing that could prove you wrong into the same pipeline as the thing that could make you look good, and reporting whichever one the data actually supports.

**Q20. Why do these three specific projects — not the equity or FX ones — form a natural starting trilogy for the whole portfolio?**

Each is deliberately the smallest version of a problem the rest of the portfolio solves at production scale: F1 is single-asset risk (no covariance matrix) versus `equity/03-var-es-engine`'s multi-asset book; F2 is a two-parameter, long/flat crossover with a hand-checkable sensitivity grid versus a full walk-forward/ML strategy stack; F3 is dependency-minimal, single-model, European-only Black-Scholes versus `equity/01-options-pricing`'s vectorised multi-model engine with C++/Rust twins. Every one of the three explicitly says so: F1 points to `equity/03-var-es-engine` for the multi-asset extension; F2's Assumption A5-A7 name exactly what a bigger project (deflated Sharpe, multi-asset testing) would add; F3's README calls itself "a warm-up for" `equity/01-options-pricing` and states outright it is "not a pricing library." They are the same ideas the rest of the portfolio scales up, stripped down to the point where the core mechanism — a quantile disagreement, an execution lag, an independent numerical witness — is the *whole* project rather than one module inside a much larger one.

**Q21. What does the production-grade successor to each foundations project add, and why would meeting that machinery first make the core idea harder to see?**

`equity/03-var-es-engine` adds a covariance/factor model, Cholesky-simulated correlated scenarios, Christoffersen's clustering test, Basel traffic-light zones, and Acerbi-Székely ES backtesting on top of F1's single-series VaR/ES/Kupiec. A bigger backtest engine (implied by F2's own "what this project deliberately does not do" list in `docs/METHODOLOGY.md` §4) would add a deflated Sharpe ratio correcting for the number of grid cells tried, multi-asset testing, volatility-scaled sizing, and statistical significance testing on the Sharpe gap. `equity/01-options-pricing` adds multiple pricing models, American exercise via a CRR tree, dividends, and vectorised performance. Every one of those additions is real and matters on a production desk — but each is also a second layer of correctness-checking wrapped around a mechanism (a quantile estimate, a lagged position, a closed-form price) that a reader needs to understand cleanly first. Meeting the covariance matrix before understanding why Gaussian VaR understates tail risk on one series, or meeting the deflated Sharpe correction before seeing an unadjusted in-sample Sharpe evaporate out-of-sample with your own eyes, means learning the fix before feeling the problem it fixes — the validation machinery ends up looking like ceremony instead of an answer to a question you've already asked.

**Q22. Practically, how should a student use these three projects relative to the portfolio's 10-area buildout?**

Read them first, and read them as complete stories, not reference material to skim past on the way to the "real" projects — each one is small enough to hold in your head in full: one return series, one signal with one execution rule, one closed-form price with one independent witness. The goal at this stage is to be able to explain, without looking anything up, why historical VaR beats Gaussian VaR on fat-tailed data, why `position = signal.shift(1)` is the one line standing between an honest backtest and a fabricated one, and what an independent Monte Carlo pricer proves that a self-consistency check cannot. Only after that intuition is solid does it make sense to move into `equity/03-var-es-engine`, the bigger backtest engine, and `equity/01-options-pricing` — at that point the covariance matrices, deflated Sharpe corrections, and CRR trees read as answers to problems you already understand, rather than machinery to memorize. Treat F1–F3 as the place the ideas get built, and the 10-area equity/FX buildout as the place they get scaled and hardened.


## Round 15 — Cross-Language Engineering

**Q1. Why does this portfolio validate its C++/Rust engines against golden vectors generated FROM the Python reference's output, rather than having each language's engineer independently re-derive the pricing/VaR formulas from the math?**

Independent re-derivation sounds more rigorous, but it isn't: if the C++ and Rust engineers both made the same conceptual mistake — say, both used the wrong sign convention for dividend yield, or both mis-derived the Cornish-Fisher correction term — their two implementations would still agree with each other perfectly while both being wrong. Cross-language *agreement* would then be indistinguishable from cross-language *correctness*, which is exactly the failure mode a validation scheme is supposed to catch. Generating golden vectors from the Python reference's own output instead proves a narrower, checkable claim: "the C++/Rust engine reproduces this specific, already-independently-validated reference," which only holds if the ported code actually mirrors the reference's semantics line for line.

**Q2. `DIAGRAMS.md` diagram 2 says golden-vector agreement proves the C++/Rust engines "compute the same numbers as the Python reference," but explicitly does NOT prove "that the Python reference itself is correct." Where does that second burden of proof live instead?**

It lives in each Python project's own analytic-identity and convergence tests — put-call parity, Greeks-vs-finite-difference, tree-to-Black-Scholes convergence, and (for VaR/ES) Kupiec/Basel backtests — per `docs/ARCHITECTURE.md` "Cross-language validation." Those tests check the reference against independent mathematical truths (an identity that must hold regardless of implementation, a limit that must be approached), not against another implementation. Golden vectors and analytic identities are deliberately different kinds of evidence: one says "matches a trusted source," the other says "the trusted source itself is not internally inconsistent."

**Q3. Walk through the actual golden-vector generation pipeline end to end, for the options engines.**

Per `docs/DIAGRAMS.md` diagram 2 and the files it names: (1) `python/equity/01-options-pricing/tests/golden/generate_golden.py` calls `bs_greeks` from the Python reference package `eq_options` on a fixed list of ~30 diverse cases (ATM/ITM/OTM, short/long-dated, negative rates, deep wings — see its `CASES` list) and writes `tests/golden/golden_vectors.json` at full double precision. (2) `cpp/equity-options-engine/tools/gen_golden_header.py` reads that JSON and emits `tests/golden_vectors.hpp`, a `constexpr std::array<GoldenCase, N>`. (3) `rust/equity-options-engine/tools/gen_golden_rs.py` reads the same JSON and emits `src/golden.rs`, a `pub const CASES: [GoldenCase; N]`. (4) Each engine's own test suite (GoogleTest / `cargo test`) asserts its computed values against the committed constants.

**Q4. Why are the golden vectors COMMITTED to the repo (as `golden_vectors.json`, `golden_vectors.hpp`, `golden.rs`) rather than regenerated at test time by invoking Python from the C++/Rust test run?**

A test that regenerates its own oracle at run time isn't testing against an independent reference at all — it's testing against whatever the reference computes *right now*, so a regression in the Python reference and a regression in the C++/Rust port could land at the same time and the test would still pass, having compared two simultaneously-broken outputs to each other. Committing the vectors freezes the oracle: `tests/golden_vectors.hpp` and `src/golden.rs` are fixed files under version control, so a C++/Rust test failure means "this engine's output moved," full stop, independent of whatever the Python reference happens to compute on a later run. `docs/ARCHITECTURE.md`'s design invariant "no golden-pinned value ever changes silently" is only meaningful because the vectors are static, diffable, committed artifacts.

**Q5. How does a golden vector cross the Python-to-C++ and Python-to-Rust boundary without losing precision?**

`gen_golden_header.py` and `gen_golden_rs.py` both format every double with Python's `repr()` via a small `fmt()` helper — the shortest decimal string that round-trips back to the exact same IEEE-754 bit pattern. That string is emitted as a C++/Rust floating-point literal (`fmt()` appends `.0` if the repr has no `.`/`e`/`E`, so it parses as a double/`f64` rather than an integer), and both languages' floating-point literal parsers reproduce the identical 64-bit value on read-back. So the committed header and Rust module carry the Python reference's outputs bit-for-bit, not merely "close."

**Q6. The VaR/ES golden values (e.g. in `cpp/equity-var-engine/tests/test_cross_language.cpp` and `cpp/fx-var-engine/tests/test_golden_python.cpp`) are hardcoded numeric literals inside the test file itself, not read from a generated header like the options engines' `golden_vectors.hpp`. Is that a different, less rigorous mechanism?**

Different mechanism, same discipline. `docs/ARCHITECTURE.md` notes the VaR engines use "direct `PYTHONPATH=src python3` invocation" rather than the JSON-plus-generator two-stage pipeline — `test_cross_language.cpp`'s header comment documents the exact reproduction command and the numpy/scipy version it was run against, and the resulting constants (e.g. `1.224129222375264e+02` for case A's 1% historical VaR) are pasted directly into the GoogleTest/`cargo test` source as literals. They are just as committed and just as frozen as `golden_vectors.hpp` — the difference is that there's no intermediate JSON/auto-generated-file layer, because there is only one target value per assertion rather than a large array of structurally identical cases worth generating programmatically.

**Q7. Why does each language use a completely different RNG — NumPy's PCG64 in Python, `std::mt19937_64` in C++, and a custom xoshiro256++ in Rust — instead of trying to synchronize them for bit-identical draws?**

No attempt is made at cross-language bit-identical seeding because it isn't achievable or valuable: PCG64, Mersenne Twister, and xoshiro256++ are structurally different generators that consume seeds and produce output streams in incompatible ways, so "same seed, same draws" across three algorithms is not a coherent goal. Each language instead picks the RNG that best fits its own standard-library situation — `cpp/equity-options-engine/include/eqopt/monte_carlo.hpp` uses `std::mt19937_64` because it's a `<random>` standard fixture; `rust/equity-options-engine/src/rng.rs` implements xoshiro256++ from scratch because "Rust's standard library ships no RNG at all." The cross-language contract is about the *statistical* answer, not the *path*.

**Q8. Given three different RNGs, what does "cross-language Monte Carlo agreement" actually mean in this portfolio?**

Statistical, not bitwise. `docs/ARCHITECTURE.md`'s "What the cross-language checks do *not* prove" section is explicit: "Monte Carlo paths are never bit-identical across languages... so MC agreement is statistical (within a documented number of standard errors), not exact." The actual random draws differ because the underlying bit streams differ, so a C++ and Rust engine pricing the same option with the same nominal seed will get different individual paths and a slightly different point estimate; what's checked is that each estimate falls within its own reported confidence interval of the analytic (or reference) value — `eqopt::MCResult::contains()` and the Rust `McResult::contains()` exist specifically to test that CI-membership property.

**Q9. What does "bitwise deterministic given a seed" mean WITHIN one engine, and why is that a different — and in some sense stronger — guarantee than cross-language agreement?**

Within a single engine, a fixed seed reproduces the exact same sequence of draws, hence the exact same output, on every run, on every machine, forever — `rust/equity-var-engine/src/rng.rs`'s doctest asserts `a.next_u64() == b.next_u64()` for two `Rng::new(42)` instances, and `cpp/equity-var-engine/include/eqvar/monte_carlo.hpp` documents `simulate_factor_returns` as "Bitwise deterministic in `seed`." That's a stronger claim than cross-language statistical agreement because it's an equality, not a tolerance band — no floating-point noise, no confidence interval, the same 64-bit pattern out to the last bit. It is a *different* claim, though, not a bigger version of the same one: cross-language agreement is about two different generators converging on the same underlying distribution; within-engine determinism is about one fixed generator being perfectly reproducible.

**Q10. Why do the C++ and Rust engines write their own inverse-CDF / Box-Muller normal transforms instead of using `std::normal_distribution` or a library normal sampler?**

Because the bitwise-determinism guarantee would break otherwise. `cpp/equity-options-engine/include/eqopt/monte_carlo.hpp` says it directly: normal deviates come from "an in-house inverse-CDF transform... rather than `std::normal_distribution`, whose algorithm is implementation-defined," so that "same (seed, threads) => bit-identical results, across standard libraries." `cpp/equity-var-engine/include/eqvar/monte_carlo.hpp`'s `RandomStream::gaussian()` does the same via `Phi^{-1}(uniform())`. The Rust RNGs (`rust/equity-options-engine/src/rng.rs`, `rust/equity-var-engine/src/rng.rs`) use Box-Muller instead, chosen in the module doc explicitly over inverse-CDF ("needs a high-degree rational approximation whose tail accuracy would dominate the error budget") and over Ziggurat ("large tables, harder to audit") — but the reasoning is the same: own the transform so the stream layout is fixed and reproducible rather than delegated to an unspecified standard-library algorithm.

**Q11. `eqvar::RandomStream` in `cpp/equity-var-engine/include/eqvar/monte_carlo.hpp` exposes four draw methods — `uniform()`, `gaussian()`, `gamma(shape)`, `chi_squared(df)`. Why does a VaR engine's RNG need gamma and chi-squared draws, not just normals?**

The multivariate Student-t factor model needs them: the file's header comment describes simulating Student-t returns via "the common-mixing-variable construction `Z / sqrt(W/df)`" where `Z` is multivariate normal and `W` is chi-squared — fatter tails than a pure Gaussian factor model without changing the target covariance (the scale matrix is pre-multiplied by `(df-2)/df` so the *realized* covariance still matches `cov` exactly). `gamma()` is implemented via Marsaglia-Tsang (with the `U^{1/a}` boost for shape < 1), and `chi_squared(df)` is literally `2 * gamma(df/2)` — both built on the same `uniform()`/`gaussian()` primitives so the whole chain stays inside one deterministic, from-scratch generator.

**Q12. `eqopt::mc_price` in `cpp/equity-options-engine/include/eqopt/monte_carlo.hpp` supports multiple `threads`. How does multithreading interact with the bitwise-determinism guarantee?**

It's preserved per thread count, not across thread counts. The header documents: "Multithreading partitions the paths into per-thread chunks with independently seeded (splitmix64-derived) RNG streams, so the result is deterministic for a given thread count," and the parameter doc is explicit that "different thread counts use different streams (statistically equivalent, not bit-equal)." So `mc_price(..., seed=42, threads=4)` reproduces bit-for-bit on every run with `threads=4`, but will not match `mc_price(..., seed=42, threads=1)` bit-for-bit — same statistical answer, different stream partition, same pattern as the cross-language case one level down.

**Q13. Why does the Rust `standard_normal()` in `rust/equity-options-engine/src/rng.rs` cache a value between calls, and why does that matter for reproducibility rather than just being a performance optimization?**

Box-Muller naturally produces two independent normal variates per pair of uniforms (`R cos(theta)`, `R sin(theta)`); `standard_normal()` returns the cosine term immediately and stashes the sine term in `cached_normal`, returning it on the *next* call without consuming any new uniforms. This is a performance win (half the `ln`/`sqrt`/`sin`/`cos` calls), but the doc comment frames it as a reproducibility property too: "consuming exactly two `u64` draws per pair makes the stream layout easy to reason about for bit-reproducibility tests" — the number of underlying RNG draws consumed by N calls to `standard_normal()` is fixed and predictable (`2 * ceil(N/2)`), which is exactly what a bitwise-determinism test needs to reason about.

**Q14. What does the Rust engines' Cargo.toml actually confirm about the "zero external dependencies" claim?**

`rust/equity-options-engine/Cargo.toml` has an empty `[dependencies]` block under the comment "ZERO external dependencies by design: determinism and auditability. RNG (SplitMix64 -> xoshiro256++) and the normal CDF (Cody's erfc) are implemented in-crate against std only." `rust/equity-var-engine/Cargo.toml` says the same for its own engine: "the deterministic RNG, special functions (erfc, inverse normal CDF, incomplete beta/gamma) and linear algebra are all implemented in-crate." Both crates' `[dependencies]` sections are literally empty — not "minimal," empty — so the claim is directly verifiable by reading the manifest rather than taking a docstring's word for it.

**Q15. Why does this portfolio treat "zero external crates" as a Rust convention worth calling out, rather than just using a well-tested `rand` or `statrs` crate?**

`docs/ARCHITECTURE.md`'s design invariants section states the reason directly: "the Rust engines in particular take zero external crates by convention... specifically so that a reviewer never has to trust an opaque dependency's numerics to trust this portfolio's numbers." A dependency on `rand` or `statrs` would mean the RNG algorithm, seeding procedure, and normal-CDF/PPF implementation could change silently on a crate version bump, and a reviewer auditing correctness would have to read someone else's code (or just trust it) rather than the ~150-200 lines actually shipped in `rng.rs`/`matrix.rs`. Owning the RNG, the normal CDF/PPF inverse transform, and the Cholesky factorization in-repo means every numerically load-bearing line in the engine is visible in the same codebase the golden-vector tests are validating.

**Q16. What does "warnings-as-errors" — `-Wall -Wextra -Werror` in C++, `RUSTFLAGS=-D warnings` / `#![deny(warnings)]` in Rust — actually buy beyond "the build succeeds"?**

`cpp/equity-options-engine/CMakeLists.txt` sets `add_compile_options(-Wall -Wextra -Werror -O2)`; `CONVENTIONS.md` states the Rust equivalent as `#![deny(warnings)]` in the CI profile, and `docs/DIAGRAMS.md` diagram 12 shows CI actually invoking `RUSTFLAGS='-D warnings'` before `cargo test --release`. The value isn't compilation success — a warning-free build with warnings merely printed would compile identically. It's that the class of bug a compiler warning flags (a signed/unsigned comparison that silently wraps, an unused-but-assigned variable that hints at a forgotten branch, a shadowed variable, an unreachable pattern arm) never gets the chance to sit in the tree as "just a warning nobody looked at" — it fails CI the same day it's introduced, at the point where the author still has full context, rather than surfacing later as a numerical bug that has to be root-caused from scratch.

**Q17. What numerical problem does the Cholesky-with-jitter-escalation technique solve, and where does it appear across the three languages?**

Plain Cholesky factorization requires the input covariance matrix to be strictly positive definite; it fails (or the code has to guard against a `sqrt` of a negative pivot) on a matrix that is only positive *semi*-definite — exactly singular or numerically indefinite due to floating-point noise. `cpp/equity-var-engine/include/eqvar/matrix.hpp`'s `cholesky()` (returning a `CholeskyResult{lower, jitter_added}`), Rust `rust/equity-var-engine/src/matrix.rs`'s `Matrix::cholesky_jitter()`, and Rust `rust/fx-var-engine/src/matrix.rs`'s `Matrix::cholesky_with_jitter()` all implement the same fix: add a small multiple of the identity to the diagonal and retry.

**Q18. How does the jitter escalation actually work, mechanically?**

Per `cpp/equity-var-engine/include/eqvar/matrix.hpp`: on failure, `jitter * mean(diag)` (default `jitter = 1e-10`) is added to the diagonal, and if factorization still fails the perturbation escalates geometrically (×10 each retry) up to `max_tries` (default 12) attempts. It's capped at `1e-6 * mean(diag)` — beyond that the code throws `std::runtime_error` rather than continuing to escalate, because "a matrix that needs more than that is materially indefinite (a genuine negative eigenvalue, not rounding)... factoring it would silently simulate a *different* covariance from the one supplied." The Rust equivalents mirror this: `rust/equity-var-engine/src/matrix.rs` names the cap `MAX_RELATIVE_JITTER` and its `cholesky_jitter()` error message spells out the same reasoning almost verbatim; `rust/fx-var-engine/src/matrix.rs`'s `cholesky_with_jitter()` returns `(L, jitter_used)` so a caller can log or alert on how much perturbation was actually needed.

**Q19. Why is a covariance matrix ever legitimately singular in this portfolio, rather than the jitter technique being purely a numerical-robustness hack for buggy input?**

Perfectly (or near-perfectly) correlated risk factors are a real market condition, not just bad data — the canonical example is a pegged FX pair, where two currencies' returns move in lockstep by policy, giving a correlation of essentially 1.0 and a covariance matrix with a near-zero eigenvalue by construction. `rust/fx-var-engine/src/monte_carlo.rs` has a test named exactly `pegged_currencies_trigger_cholesky_jitter`, with a comment noting "the second Cholesky pivot is exactly zero and the jitter path is" exercised — i.e. the jitter escalation is explicitly designed around, and tested against, a legitimate real-world input, not treated as a defensive catch-all for malformed data.

**Q20. Why does the jitter escalation stop and throw instead of continuing to add larger and larger perturbations until *something* factors?**

Because an unbounded escalation would eventually "succeed" on a matrix that isn't merely singular but genuinely indefinite — has a real negative eigenvalue, e.g. from a correlation-matrix typo (a correlation entered as 1.2, or a covariance built inconsistently) — and factoring it after enough jitter silently simulates a materially different covariance from the one the caller supplied, understating or overstating risk with no diagnostic that anything was wrong. `cpp/equity-var-engine/include/eqvar/matrix.hpp` is explicit that such matrices should be "repaired... upstream (eigenvalue clipping / nearest-PSD projection)" rather than patched over by the Cholesky call itself — the cap turns a silent risk-number distortion into a loud `std::runtime_error`/`EqVarError::Numerical`.

**Q21. Options golden vectors are checked to 1e-9; VaR/ES golden vectors to 1e-6..1e-8. Where are those numbers actually enforced, concretely?**

`cpp/equity-options-engine/tests/test_golden.cpp` defines `constexpr double kTol = 1e-9;` and checks every case in `golden_vectors.hpp` against it. On the VaR/ES side, `cpp/equity-var-engine/tests/test_cross_language.cpp` uses a mixed relative/absolute check (`1e-9 * abs(expected) + 1e-12`) for deterministic closed-form cases, while `cpp/fx-var-engine/tests/test_golden_python.cpp` and its Rust twin `rust/fx-var-engine/tests/test_golden_python.rs` use `EXPECT_NEAR(..., 1e-6)` for P&L-scale VaR/ES figures and `1e-8 * value` (relative) for a second batch of cases — the Rust file's header comment summarizes it as "P&L-scale figures agree to ~1e-6 absolute / ~1e-8 relative and probability-scale figures to ~1e-9 absolute."

**Q22. Why is the VaR/ES tolerance looser than the options tolerance, and why is that the honest choice rather than a way to paper over imprecision?**

`cpp/fx-var-engine/tests/test_golden_python.cpp`'s header explains the mechanism directly: "the only cross-language differences are libm sin/cos/exp rounding (~1 ulp per call) and the special-function implementations (scipy Cephes vs this library)." An option price is one closed-form formula evaluated once; a VaR/ES pipeline chains a synthetic return series through book revaluation (itself calling sin/cos), covariance estimation, sorting into order statistics, quantile interpolation, and a tail integral — each stage's ULP-level rounding difference between numpy/scipy's implementation and this codebase's own libm calls compounds through the next stage. A single-formula options price simply has fewer floating-point operations in series to accumulate noise across, so 1e-9 is achievable there in a way it structurally isn't for a multi-stage VaR pipeline; picking 1e-6/1e-8 for VaR/ES is admitting that arithmetic reality rather than hiding a real discrepancy behind a loose bound.

**Q23. How do we know the looser VaR/ES tolerance isn't quietly absorbing an actual numerical bug rather than legitimate rounding noise?**

Two structural facts rule that out. First, the discrepancy is explicitly attributed to a *named, bounded* source — "libm sin/cos/exp rounding (~1 ulp per call)" and "scipy Cephes vs this library" for special functions — not a vague "close enough" tolerance; ULP-level libm divergence is a known, quantifiable phenomenon, not a fudge factor. Second, `docs/ARCHITECTURE.md`'s design invariant "no golden-pinned value ever changes silently" means every numerics fix across the portfolio's two hardening passes was verified not to move a single pinned golden number — if the 1e-6/1e-8 tolerance were actually absorbing a real bug, tightening the engine's accuracy in a hardening pass would have been expected to shrink the residual, and that expectation was checked, not assumed.

**Q24. What does a golden-vector test failure actually tell an engineer — and what doesn't it tell them?**

It tells them the C++/Rust engine's output for that specific input diverged from the Python reference's committed value by more than the stated tolerance — i.e., the port's semantics drifted from the reference it's supposed to mirror. It does *not* by itself tell them which side is "right": per Diagram 2's own caveat, cross-language agreement never proved the Python reference was correct in the first place, so a failure could equally mean the C++/Rust port has a bug, or (much less likely, since the reference is separately validated) that the Python reference itself changed. In practice the diagnostic path is: check whether `golden_vectors.json` was regenerated (a deliberate, logged reference update) versus the engine code changed (the port drifted) — the committed-artifact design from Q4 is what makes that distinction checkable at all.

**Q25. Summarize the "validated three ways" structure `docs/ARCHITECTURE.md` describes for each of the four compiled-twin engine pairs.**

(1) The Python reference is validated against analytic identities and statistical backtests — put-call parity, Greeks-vs-finite-difference, tree-to-BS convergence, Kupiec/Basel — none of which involve the C++/Rust engines at all. (2) The C++ engine mirrors the Python semantics and is checked against golden vectors generated from the Python reference's output, to 1e-9 (options) or 1e-6..1e-8 (VaR/ES) tolerance via GoogleTest. (3) The Rust engine is checked against the same golden values via `cargo test`, to the same tolerances. The three legs carry different burdens of proof: leg 1 proves the *reference* is mathematically sound; legs 2 and 3 each prove their engine matches that reference — `docs/ARCHITECTURE.md` frames this as "the same discipline a real market-risk function uses when a new pricing library has to be proven against the incumbent before it is allowed to touch P&L."

**Q26. Is the golden-vector methodology the same generation mechanism for every one of the four engine pairs (equity/FX × options/VaR), or does it vary?**

It varies in mechanism but not in principle. The options engines (`cpp/equity-options-engine`, `rust/equity-options-engine`, and their FX twins) use the fully automated three-stage pipeline: `generate_golden.py` → `golden_vectors.json` → `gen_golden_header.py`/`gen_golden_rs.py` → committed `golden_vectors.hpp`/`golden.rs`. The VaR/ES engines instead run a one-off Python invocation whose printed `repr()` output is pasted directly as literals into `test_cross_language.cpp` / `test_golden_python.cpp` / `test_golden_python.rs`, with the exact reproduction command and library versions documented in the test file's header comment. Both mechanisms satisfy the same invariant — every consumed value is a frozen, committed constant traceable to a specific Python run — the difference is just whether there are enough structurally-identical cases (options: ~30 cases × 6 outputs) to make a code-generation step worth writing.

**Q27. What role does `RandomStream`/`Xoshiro256PlusPlus` play in `mc_bootstrap_se` / the bootstrap standard-error cross-check, and why does its determinism matter there specifically?**

`cpp/equity-var-engine/include/eqvar/monte_carlo.hpp`'s `mc_bootstrap_se` resamples the P&L sample with replacement `n_boot` times "via this file's own `RandomStream`, so the result is bitwise deterministic in `seed`," recomputing the same type-7 linear-interpolation VaR quantile on each resample. This is a distribution-free cross-check on the fixed-bandwidth order-statistic `var_se` estimator (documented elsewhere in the portfolio as biased low by 9-17% in deep tails / small samples — see Round 18 in `docs/LEARN.md`). Its determinism matters here for the same reason as everywhere else in the portfolio: a bootstrap SE that changed between identical runs would be indistinguishable from an actual change in estimated sampling error, defeating the point of having a second, independent-of-the-first estimator to compare against.

**Q28. Tie it together: why is "matches the reference" a different — and in some ways stronger — claim than "two independent teams re-derived the same formula," even though it sounds like a weaker bar?**

Independent re-derivation only proves *convergence*: two engineers ending up in the same place tells you nothing about whether that place is correct, because a shared conceptual error survives the comparison undetected. "Matches the reference to a stated tolerance," by contrast, is a claim with a named, separately-validated anchor: the C++/Rust engine is being measured against a specific artifact (`golden_vectors.json` / a documented `repr()` dump) that itself was produced by code carrying its *own* independent burden of proof — put-call parity, Kupiec backtests, tree-to-BS convergence — none of which depend on the C++/Rust ports existing at all. The chain of trust is linear and auditable (reference validated by math → engines validated against reference) rather than circular (two implementations validated by mutual agreement), which is precisely why a real market-risk function uses this pattern when proving a new pricing library against an incumbent before it's allowed to touch P&L.


## Round 16 — Testing, Validation & the NaN Defect Class

**Q1. What is the single IEEE 754 fact that this entire defect class hinges on?**

Under IEEE 754, NaN compares as unordered with everything, including itself: `NaN < x`, `NaN <= x`, `NaN > x`, `NaN >= x`, and even `NaN == NaN` all evaluate to `False`, for any `x`. There is no ordering comparison that returns `True` when one operand is NaN. That single fact is why a guard built purely out of ordering operators can never catch a NaN — not because NaN is "positive" or "in range", but because every comparison silently fails to fire.

**Q2. Concretely, why does `if x <= 0: raise ValueError` fail to catch a NaN input?**

The guard is written to reject non-positive numbers by checking `x <= 0`. When `x` is NaN, `x <= 0` evaluates to `False` (per Q1), so the `raise` branch is never taken — the same as if a genuinely valid positive number had been passed. The guard doesn't misclassify NaN as positive; it simply never gets a `True` answer from any comparison involving NaN, so it falls through the way any valid input would.

**Q3. Where in this portfolio was this defect class actually found, and how widespread was it?**

Per the README's "On input validation" section, a portfolio-wide hardening pass found the *same* defect class — `if x <= 0: raise` silently accepting NaN — in project after project, across all three languages (Python, C++, Rust). It wasn't a one-off bug in one function; it was a systemic pattern repeated everywhere validation had been written with ordering comparisons instead of explicit finiteness checks.

**Q4. What is the first documented consequence — a NaN spot reaching a pricer?**

A NaN spot price `S` reaching a Black-Scholes-style pricer that didn't reject it produces a NaN "price" as output. This is the mildest of the three consequences: a NaN output is visibly broken. Anyone consuming that number — a downstream aggregator, a P&L report, a human — sees `nan` and knows immediately something is wrong.

**Q5. What is the second documented consequence — NaN covariance into a VaR aggregator?**

A NaN entry in a factor covariance matrix reaching the VaR aggregation step in the Rust and C++ VaR engines produced a portfolio VaR of *exactly zero*, because the aggregation step computes `max(NaN, 0.0)`, and IEEE 754's `max`/`fmax` semantics propagate the non-NaN operand when one argument is NaN. The NaN doesn't survive to the output — it gets silently discarded and replaced by the other operand.

**Q6. Why is a VaR of exactly zero specifically called "more dangerous than a NaN" in the README?**

A NaN output is self-evidently broken and gets caught by any sanity check or human glance. A VaR of exactly zero on a hedged book is a *plausible* number — a well-hedged portfolio really can have very low VaR — so it looks like good news rather than a red flag. It is far less likely to be double-checked, which means the corrupted input can silently pass all the way through to a capital or hedging decision.

**Q7. What is the third documented consequence — NaN in a P&L feed reaching backtesting?**

A NaN in a historical P&L feed reaching VaR-exception counting caused the backtest to record zero breaches for that observation, because the breach test (P&L worse than VaR) is itself an ordering comparison against NaN, which evaluates `False`. The practical effect: a broken model with genuinely bad P&L data passed its Kupiec proportion-of-failures test and Basel traffic-light backtest green, because the corrupted days were silently excluded from the exception count rather than flagged.

**Q8. What do these three consequences have in common as a mechanism?**

In all three cases, the failure mode is not a crash — it's a *silent substitution*: a comparison (`<=`, `max`, an exception-count check) that is supposed to gate or aggregate a value instead quietly drops the NaN and behaves as if the input were absent or benign. The output in each case is a number that looks like a legitimate answer, not an error signal — which is exactly what makes the class dangerous rather than merely annoying.

**Q9. What is the actual fix pattern applied portfolio-wide?**

Replace ordering-comparison guards with explicit finiteness checks: `math.isfinite()` / `np.isfinite()` in Python, `std::isfinite()` in C++, `.is_finite()` in Rust. A finiteness check directly asks "is this value NaN or ±Infinity?" rather than inferring badness indirectly through a failed ordering comparison, so it catches NaN by construction instead of by accident.

**Q10. Show a real example of this fix in the equity options pricer.**

`validate_inputs()` in `python/equity/01-options-pricing/src/eq_options/black_scholes.py` iterates over `S`, `K`, `T`, `sigma` and, for each, does `if not math.isfinite(value): raise ValueError(...)` *before* the separate `if value < 0.0: raise ValueError(...)` non-negativity check. The finiteness check and the sign check are two distinct guards — the finiteness check exists specifically because the sign check alone cannot catch NaN.

**Q11. Show a real example of this fix in the VaR engine.**

`_validate_pnl()` in `python/equity/03-var-es-engine/src/eq_var/historical_var.py` calls `np.all(np.isfinite(arr))` on the P&L array and raises `ValueError("pnl contains NaN or infinite values")` if it fails, before any quantile computation touches the data. `historical_var.py`, `expected_shortfall.py`, and `parametric_var.py` in the same package all carry equivalent `np.isfinite(...)` guards.

**Q12. Is the fix confined to Python, or does it show up in the compiled engines too?**

It's cross-language. `cpp/equity-var-engine/src/returns.cpp` guards P&L with `if (!std::isfinite(v)) throw std::invalid_argument(...)`, and `parametric.cpp` checks weights, VaR, `z_range`, `skew`, and `excess_kurt` the same way. `rust/equity-var-engine/src/historical.rs`, `expected_shortfall.rs`, and `backtest.rs` use `.is_finite()` identically. The three languages independently reimplement the same finiteness-first validation contract.

**Q13. Does fixing this defect class change any golden-vector reference values?**

No — this is strictly an input-validation fix, not a change to any pricing or risk formula. It rejects a class of input (NaN/Inf) that a correct guard should never have let through in the first place; every previously-valid input still produces the same output. This is the same principle later applied to the implied-vol-solver and Cornish-Fisher fixes (Round 16's sibling defects, per `docs/DIAGRAMS.md` diagrams 10–11): robustness fixes are validated by *not* moving any pinned golden value.

**Q14. Why is this specific defect class more dangerous in a quant/risk codebase than in, say, a typical web application?**

In a web app, a validation gap that lets bad data through more often surfaces as a crash, a 500 error, or a visibly malformed page — something a user or monitoring system notices quickly. In a risk codebase, the same gap produces a *number* — a price, a VaR, a breach count — that looks exactly like every other legitimate number. Nothing crashes. The system keeps running and keeps reporting, and the wrong number quietly feeds a hedging or capital-allocation decision made by a human or another system that has no way to tell it apart from a correct one.

**Q15. Why is `max(NaN, 0.0)` propagating the non-NaN operand not itself a bug in IEEE 754 — and why does that matter here?**

It's specified behavior: IEEE 754 `fmax`/`max` are defined to return the non-NaN argument when exactly one operand is NaN, precisely so that a single missing/undefined data point doesn't poison an otherwise-valid aggregate elsewhere in a computation. That's a reasonable default for many uses of `max` — but it means `max` is not a safe substitute for an explicit NaN check when the *presence* of the NaN is itself the signal you need to detect, which is exactly the situation in VaR aggregation.

**Q16. What does a good test for this defect class actually look like, as opposed to a merely "adequate" one?**

A test that only exercises "normal" invalid inputs — negative spot, negative time-to-expiry, an out-of-range alpha — will pass against a guard that has this bug, because those inputs are exactly what ordering comparisons *do* catch. A test that actually catches this class explicitly constructs `float("nan")` (or `np.nan` / `f64::NAN` / `std::numeric_limits<double>::quiet_NaN()`) and passes it into every public entry point, asserting that it raises rather than silently succeeding or returning NaN.

**Q17. Point to a real test in this portfolio that does this.**

`test_negative_or_nan_inputs_raise_everywhere` in `python/equity/01-options-pricing/tests/test_edge_cases.py` is parametrized over both ordinary invalid inputs (negative `S`, `K`, `T`, `sigma`) *and* `(float("nan"), 100, 1, 0.2)`, and asserts `pytest.raises(ValueError)` against both `bs_price` and `crr_price` for every case in the same table — NaN is tested as a first-class member of the invalid-input set, not as an afterthought.

**Q18. Point to a real test on the VaR side that does this.**

`test_nan_raises` in `python/equity/03-var-es-engine/tests/test_historical_var.py` builds a 100-element zero P&L array, sets `pnl[3] = np.nan`, and asserts `pytest.raises(ValueError, match="NaN")` from `historical_var(pnl, 0.01)`. It doesn't just check that *some* invalid array is rejected — it checks that a single NaN buried inside an otherwise-valid array specifically is caught, which is the realistic failure mode (one bad tick in a feed, not a wholly corrupted series).

**Q19. Is there a complementary style of test — checking that NaN never *appears* in valid output, not just that it's rejected on input?**

Yes. `test_no_nans_price_sweep` and `test_no_nans_greeks_sweep` in the same `test_edge_cases.py` sweep a grid of legitimate `S`, `K`, `T`, `r`, `sigma` combinations (including extreme-but-valid values like `1e-3`/`1e5`) and assert every output is finite. This is the mirror image of Q17–Q18: rather than asserting bad input is rejected, it asserts that no combination of *good* input can accidentally manufacture a NaN downstream (e.g. via a `0/0` or `log(0)` inside the formula) — the other place NaN contamination can originate besides a bad upstream feed.

**Q20. What is this portfolio's broader testing philosophy, per `CONVENTIONS.md`'s "Testing contract"?**

Beyond happy paths and edge cases, the contract requires "analytic identities (put-call parity, Greeks vs finite differences, martingale checks), convergence (tree → BS, MC → BS), and failure behaviour (invalid inputs raise `ValueError` with informative messages)", plus "property-based checks where natural (monotonicity in vol, convexity in strike)". The emphasis throughout is on properties the model is *mathematically required* to have, not just a table of input/output pairs.

**Q21. What does it mean to test a PROPERTY rather than a fixed expected output, concretely?**

A fixed-output test asserts `bs_price(100, 100, 1, 0.05, 0.2, 0.0, "call") == 10.4506` (or close to it) — it only tells you that one specific input still produces one specific output. A property test asserts something that must hold for a whole *family* of inputs regardless of the specific numbers — e.g. `test_put_monotone_increasing_in_vol` and `test_call_delta_monotone_in_spot` in `python/equity/01-options-pricing/tests/test_properties.py` check that price/delta move in the required direction across a swept range of vol or spot values, and `test_parity_and_forward_consistency_stressed` checks that call price minus put price equals the forward-discounted difference across stressed parameter combinations.

**Q22. Why does property-based testing catch defect classes rather than just individual bugs?**

A fixed expected-output test only fails if the specific case it happened to encode is broken. A property test — "for every input in this swept range, X must hold" — fails on *any* input in the range that violates the property, including ones nobody specifically thought to hand-pick. Put-call parity, monotonicity in vol, and tree→BS convergence are structural truths of the model; if a NaN-guard bug, a solver-plateau bug, or a grid-resolution bug produces even one input where the property breaks, a property test sweeping that region will catch it — whereas a hand-picked expected-output table might simply never include that input.

**Q23. Is `test_no_nans_price_sweep` (Q19) itself a property test, and if so, what property does it encode?**

Yes — its property is "for every combination of valid inputs, the pricer's output is finite." That's exactly the kind of blanket, model-wide invariant that a single `if x <= 0: raise` guard bug would violate somewhere in the swept grid even if it happened to pass every individually-chosen edge case test. It's the direct test-side counterpart of the fix in Q9–Q13: the fix makes the guard reject non-finite input, and this test makes sure valid input can never accidentally *produce* non-finite output.

**Q24. How does `test_infinite_or_nan_price_rejected_by_implied_vol` extend this beyond simple parameter validation?**

That test (in `python/equity/01-options-pricing/tests/test_properties.py`) checks that the implied-vol solver itself rejects a NaN or Infinite *target price* — not a spot, strike, or vol, but the market-observed price the solver is trying to invert. It's a reminder that "every public entry point" in the README's fix description really does mean every one: an inversion routine consuming a market quote is just as much an attack surface for a silently-accepted NaN as a forward pricing call is.

**Q25. Does the finiteness-check fix generalize past NaN, or is it NaN-specific?**

It generalizes: `math.isfinite`, `np.isfinite`, `std::isfinite`, and `.is_finite()` all reject both NaN *and* ±Infinity in one check. An `if x <= 0: raise` guard has the same blind spot for `+Infinity` reaching code that assumes bounded input (it *would* be caught by a `<= 0` check only if it were non-positive infinity — positive infinity sails through a positivity guard exactly like NaN does, just for a different reason: it genuinely is `> 0`). The finiteness check closes both holes with one guard instead of needing separate NaN and Inf logic.

**Q26. How was this defect class found, according to the README's characterization of the review passes?**

The README frames it as the first of a sequence: "A portfolio-wide hardening pass found the same defect class in project after project" — i.e., a dedicated audit pass looking specifically for input-validation gaps, not something caught by the original per-project test suites during initial buildout. `docs/DIAGRAMS.md` diagram 13 places it as "Hardening pass 1" in the portfolio's review history, distinct from and prior to the later numerical-robustness pass (implied-vol plateau, Cornish-Fisher grid resolution) covered elsewhere in this portfolio's documentation.

**Q27. Why didn't the original test suites — which did test invalid inputs like negative numbers — catch this before the hardening pass?**

Because a test suite that asserts negative and out-of-range inputs raise `ValueError` will pass against a buggy `if x <= 0: raise` guard just as readily as against a correct one — negative numbers *are* correctly caught by that comparison. The gap only shows up if a test specifically constructs NaN and hands it to the same code path, which is a qualitatively different test input, not a more extreme version of "a bad number." This is precisely why Q16's distinction — NaN as a first-class test case, not an extrapolation of "very negative" — matters: no amount of testing more negative numbers would ever have found this bug.

**Q28. What is the general lesson this round is meant to teach about writing input guards?**

Never infer "this value is invalid" from a guard reaching `False` — a `False` comparison result can mean "the value is valid" or "the value is NaN, so no comparison could evaluate true," and ordering operators cannot distinguish the two. Any function whose validity range is expressed with `<`, `<=`, `>`, `>=`, or `==` needs an explicit, independent finiteness check ahead of those comparisons whenever the input could plausibly originate from a live feed, a prior calculation, or any other source that can produce NaN — which, in a quant/risk pipeline, is effectively everywhere.


## Round 17 — Business, Desk Usage & Model Risk Governance

**Q1. What is "model risk" as a formal risk category, and how is it different from market, credit, or operational risk?**

Model risk is the risk that a model itself is wrong, poorly specified, or misused — even when it is implemented flawlessly against its own spec. Market risk is the risk that prices move against you; credit risk is the risk a counterparty doesn't pay; operational risk is the risk a process or system fails. Model risk sits one level up: it's the risk that the *tool you use to measure* those other risks is itself miscalibrated — a VaR engine that assumes normal returns (`python/equity/03-var-es-engine`), a scorecard whose missing-data mechanism silently changed (`python/equity/06-credit-risk`), an options pricer whose no-jump assumption gets exploited by an earnings gap (`python/equity/01-options-pricing`). The code can pass every unit test and still be the wrong model for the market regime it's now facing.

**Q2. What is the standard model-governance lifecycle a real bank applies to a model like these, end to end?**

Development (build it, document assumptions), independent validation (a separate team stress-tests it against its own spec and against challengers), approval/sign-off (a model risk committee formally accepts it for a defined use), ongoing monitoring (automated triggers watch live performance), periodic re-validation (a scheduled full re-review, not just monitoring), and retirement (the model is formally decommissioned when it fails, or is superseded). The credit-risk `DESK_GUIDE.md` §6 spells this out almost verbatim under "Model governance — annual validation cycle," citing SR 11-7 / EBA expectations explicitly, with independent validation, hard monitoring triggers, change control, and a use test as the four numbered steps.

**Q3. Why does "independent" validation specifically matter — why can't the model's own developer validate it?**

A developer who built a model already believes its assumptions are reasonable; that belief is precisely what makes them the worst-positioned person to find where those assumptions break. They will test the paths they thought of, not the ones they didn't — someone who never held the belief has to actually interrogate it. The credit-risk project's governance section makes this concrete: independent validation includes "a challenger model (GBM benchmark) and the sklearn cross-check" — a second, differently-built model whose disagreement with the first is itself the validation signal, something the original developer has no incentive to go looking for.

**Q4. This portfolio's own documentation contract requires six things per project. How does that map onto a real model-validation submission package?**

CONVENTIONS.md's six items — why you chose the model, the assumptions register, how you validated it, where it fails, how a desk would use it, and edge cases — are almost exactly the sections a bank's model risk management team expects in a validation submission: model selection rationale and alternatives considered, a documented assumptions log with "what breaks if violated," independent validation evidence, a known-limitations register, a use-and-controls section, and a documented edge-case/boundary-behavior appendix. The equity options `DESK_GUIDE.md` §2 literally uses the phrase "known-limitations register" pointing back at the assumptions (A1–A8) and validation sections — the same artifact a real submission package would attach.

**Q5. Why is "how would a real desk use this?" one of the six mandatory questions, rather than an afterthought bolted on at the end?**

Because a model whose real-world usage was never thought through tends to have a specification gap that only shows up in production — never in a clean unit test suite, which only exercises the inputs someone already anticipated. The FX VaR engine's `DESK_GUIDE.md` makes this explicit as a design principle: the engine "refuses NaN" at every boundary — market snapshots, factor histories, scenario shocks — specifically because "every risk number this desk publishes is consumed by a limit check, a traffic light or a capital multiplier — and none of those fire on NaN." That behavior isn't something a math-correctness test would ever catch; it only surfaces once you ask who consumes the number and what happens when the pipeline hits a stale fixing at 06:00.

**Q6. What is P&L attribution, and why is it described as a live check on the risk model rather than just an accounting exercise?**

P&L attribution decomposes today's actual P&L move into the pieces the risk model says should explain it — for options, `dV ≈ delta·dS + ½gamma·dS² + vega·dsigma + theta·dt + rho·dr + residual`. If the attribution reconciles, the model's sensitivities are being borne out in reality; if there's a persistent unexplained residual, that's evidence the model or its calibration is wrong, not just noise. The options `DESK_GUIDE.md` §1.3 states this directly: "A persistent residual is the model-risk alarm (wrong q around ex-dates, smile move not captured by parallel vega...)." The algo-execution project extends the same idea to execution: the PM's daily sheet splits "signal P&L (gross), cost drag (ledger), execution quality (IS vs VWAP), each with its own owner" — attribution as an organizational control, not just a number.

**Q7. Give a concrete example of a persistent P&L residual acting as a model-risk red flag.**

The options desk guide ties the residual explicitly to specific mis-specifications: a wrong dividend yield `q` around ex-dividend dates, or a smile move that parallel vega (a single number) can't capture and needs splitting into vanna/volga buckets instead. Section 3.2's dividend-announcement scenario quantifies it — up to +1.15 mispricing on a 2.75-value call straddling the ex-date when the continuous-q assumption gets the direction right but the timing wrong. That's not a bug in the pricer; it's the assumption register (A-series in METHODOLOGY.md) manifesting as an actual, measurable residual on the P&L line, which is exactly why it's monitored rather than assumed away.

**Q8. What's an analogous "unexplained residual" check in the VaR/backtest world, distinct from options P&L attribution?**

Backtest exceptions and their clustering. The equity VaR desk guide §3 describes checking, on any exception (realized loss worse than VaR), whether losses cluster within a short window: `exception_cluster_table` flags gaps ≤5 days via the Christoffersen independence test, because "a cluster with acceptable count still fails conditional coverage (yellow flag on the model, not the desk)." The FX desk guide states the lesson even more sharply: "clustered exceptions... trigger a model review even in the green zone — clustering means the model is slow, not unlucky." An occasional, isolated exception is expected statistical noise at a 99% confidence level; a cluster is the residual-attribution equivalent of "the model stopped explaining reality."

**Q9. What are position limits and risk limits, and why do they function as controls that don't require anyone to fully trust the model's exact number?**

A limit is a hard cap on exposure (delta, VaR, ES, gross/net notional, participation rate) enforced regardless of how confident anyone is in the underlying model's precision — a limit only needs the number to be directionally useful, not exactly right. The FX VaR desk guide §3 lays out a full limit sheet (firm 99%/1d ES at $2.5m, per-pair delta caps, vega-per-pair, a separate peg-exposure notional limit) and explicitly notes peg exposure "gets its own notional limit precisely because it does not consume VaR limit" — i.e., the desk knows VaR is *wrong* for peg risk and layers an independent control on top rather than trying to fix the VaR number. That is model risk management in action: don't wait for the model to be perfect, bound the damage it can do if it's wrong.

**Q10. Give a second, different example of a limit compensating for a known model weakness rather than assuming the model is complete.**

The equity portfolio-optimization desk guide §7 lists a concentration floor (`effective_n` ≥ 4) and a turnover budget (≤2x/yr) as controls that exist independent of the optimizer's own math — an unconstrained raw-mean tangency portfolio in the same project's backtest race has effective N of only 1.6 (roughly two names) and 8.3x annual turnover, which "would blow any such budget; that alone disqualifies it operationally, before performance is even discussed." The limit doesn't ask whether the optimizer's covariance estimate is exactly right — it just refuses to let a bad estimate turn into a two-name book, full stop.

**Q11. Walk through the equity options desk's actual daily workflow — who uses the pricer, in what order, and for what?**

Per `python/equity/01-options-pricing/docs/DESK_GUIDE.md` §1: market-makers use the kernel intraday to generate theo prices and bid/ask quotes off the vol surface (one marked vol per strike/expiry, never a flat vol); the same kernel feeds an intraday risk pipeline that re-prices the whole book off current marks and aggregates to book-level delta/gamma/vega/theta/rho, with delta auto-hedged with futures inside a band; and overnight, an EOD batch re-prices off closing marks, produces the Greeks report, runs P&L attribution, and runs a nightly `comparison.py` harness checking BS vs tree vs MC agreement as an automated regression gate. Three different consumers (market-making, intraday risk, EOD governance) touch the same pure pricing functions for three different purposes.

**Q12. What triggers an escalation on the equity VaR desk, and who does it go to?**

Per `python/equity/03-var-es-engine/docs/DESK_GUIDE.md` §2, a limit breach on 99% 1d VaR or 97.5% ES triggers same-day escalation to the desk head and the market risk manager, and the desk must either cut risk or obtain a documented temporary limit increase — no quiet override. A VaR SE-adjusted breach (inside one standard error of the limit) is still treated as a breach out of conservatism, just flagged as statistically marginal rather than definitive. Separately, §3 describes a quarterly model review where a backtest method failing Kupiec/CC tests for two consecutive quarters is formally remediated or replaced, with the backtest evidence attached going through model-risk governance — an escalation path distinct from the daily breach path.

**Q13. What triggers an escalation on the credit-scoring pipeline, and how is it different in character from the VaR desk's triggers?**

Per `python/equity/06-credit-risk/docs/DESK_GUIDE.md` §6, three hard triggers govern ongoing monitoring: PSI > 0.25 on the score or any input forces recalibration/rebinning (0.10–0.25 is an investigate-band); realized-vs-predicted default rate falling outside the 95% binomial band, or a Hosmer-Lemeshow p-value < 0.01 on the latest vintage, forces at minimum an intercept recalibration; and an AUC drop of more than 5 points from development triggers a full redevelopment. Unlike the VaR desk's same-day breach escalation (a market event demands an immediate response), these are statistical monitoring triggers evaluated on a slower monthly/quarterly cadence against a stable population — the credit desk guide's §7 scenario on 2020 payment holidays even shows PSI staying "stable" (0.074) while calibration silently failed, underscoring that the two monitors catch different failure modes and neither substitutes for the other.

**Q14. What triggers an escalation on the algo-execution desk, and how many distinct kill-switches does it run?**

Per `python/equity/08-algo-execution/docs/DESK_GUIDE.md` §4, there are five automated, pre-trade-and-intraday kill-switches: a participation kill (a child order exceeding its cap on bucket volume is rejected outright), book limits (gross/net/per-name caps block new opening trades), a cost kill (realized slippage exceeding roughly 3x the modeled cost halts the algo and routes residuals manually), a data kill (stale prices or feature NaN spikes freeze the signal, with a stale-signal counter escalating after two days), and a drawdown kill (strategy MDD beyond backtest MDD plus a buffer forces a de-gross to half and a PM review). This is a materially higher density of automated, real-time triggers than either the VaR desk (same-day human escalation) or the credit desk (monthly/quarterly statistical triggers) — appropriate given execution risk compounds in minutes, not days.

**Q15. Who actually consumes a VaR number, and does everyone see the same thing?**

No — the equity VaR desk guide §7 lays out a consumer table with genuinely different numbers at different cadences: the desk head sees VaR/ES vs limits, ladders and exceptions daily pre-open; the CRO/risk committee sees top-of-house VaR/ES and the traffic-light zone daily-to-weekly; the regulator sees 99% VaR plus 97.5% ES and the 250-day exception count quarterly and ad hoc; model validation sees backtest p-values and method-disagreement flags quarterly; and finance/capital sees the capital multiplier k × VaR only monthly. The same underlying risk engine serves five audiences with five different slices and five different refresh rates — a single "VaR number" is really a family of derived views.

**Q16. Who consumes the credit scorecard's PD, and what does the "use test" require of them?**

Per `python/equity/06-credit-risk/docs/DESK_GUIDE.md` §1 and §6, origination uses the score plus cutoff and reason codes per application; pricing turns the PD into a risk-based spread per application; finance turns point-in-time PDs into IFRS 9/CECL expected credit losses monthly or quarterly; treasury computes Basel capital (K, RWA) monthly; risk appetite runs Vasicek economic capital and stress quarterly/annually; and model risk management owns the annual validation pack. The "use test" (§6.4) is the governance rule tying them together: "pricing, origination and provisioning must consume the same PD (one model, many uses) or divergences must be documented" — i.e., a bank isn't allowed to quietly run a friendlier PD for pricing than the one it reports for provisioning without that gap being an explicit, documented decision.

**Q17. How does model governance differ between a "champion/challenger" setup and a single blessed model, and where does this portfolio show a champion/challenger structure explicitly?**

A champion/challenger setup runs one model as the production number while one or more alternative models run in parallel purely to sanity-check it — disagreement between them is itself a monitored signal, not noise to be ignored. The FX VaR desk guide §8 states this as policy: "Champion model: FHS (headline VaR/ES); challengers: parametric normal/t (speed, decomposition), MC t/jump (EM and peg overlays). The daily report prints all of them; a >20% champion–challenger gap is a standing agenda item." The equity VaR desk guide's morning pack (§1) does the same thing under a different name — "the disagreement column" between FHS and parametric VaR — explicitly framed as "information, not a bug."

**Q18. Why does the portfolio-optimization project treat "beat equal weight net of costs" as a formal retirement test rather than just informal commentary?**

Because equal weight is cheap, has no estimation error, and is hard to beat after realistic costs — so a strategy that can't clear that bar isn't adding value, however sophisticated its math. `python/equity/07-portfolio-optimization/docs/DESK_GUIDE.md` §2 states it as governance: "the walk-forward race (EW benchmark included) is rerun quarterly; a strategy that cannot beat equal weight net of costs over rolling windows is a candidate for retirement. Equal weight is the null hypothesis of this business." That's the retirement stage of the model lifecycle (Q2) made concrete and mechanical — a scheduled, quantitative test rather than a subjective judgment call, run on the same cadence (quarterly) as the VaR desk's backtest method review.

**Q19. What does "change control" mean in a model-governance context, and how do at least two of these projects implement it in code rather than in a policy document?**

Change control means any modification to a model's calibrated parameters, binning tables, or constants is versioned, tracked, and re-approved rather than silently edited — the model in production today should be traceable to a specific signed-off version. The FX VaR desk guide §8 makes this literal: cross-language golden vectors pin specific numeric outputs so "a refactor here fails here rather than surfacing as a mystery failure in another language's CI... regenerating those constants is a deliberate, signed-off act, not a side effect." The credit-risk desk guide §6.3 states the same principle for its domain: "binning tables and coefficients are the model; any change is versioned and re-approved. The leakage deny-list (`FORBIDDEN_POST_OUTCOME_FIELDS`) is part of the model definition" — i.e., even the list of fields the model is *forbidden* to use is itself a governed artifact.

**Q20. The algo-execution desk guide describes a four-stage pipeline before a signal reaches real money — what are the stages, and what stops a signal from skipping ahead?**

Per `python/equity/08-algo-execution/docs/DESK_GUIDE.md` §1: research (a new feature needs a point-in-time mutation test and a hand-computed unit test before anyone even looks at its information coefficient), simulation (the candidate must clear an IC t-stat > 3 *and* a deflated Sharpe ratio > 0.9 at the honest trial count logged in the research diary — punishing exactly the p-hacking that comes from trying many variants), paper trading (4-8 weeks in shadow, with live IC compared against the backtest's confidence band — a divergence beyond 2 standard errors for 20 days "triggers a research review, not a quiet re-fit"), and production (where the signal's code hash, parameter set, trial count, and approval date are all recorded, and any parameter change restarts paper trading from scratch). Nothing skips a stage — the paper-trading rule specifically closes off the shortcut of a developer quietly re-tuning a live signal without new independent scrutiny.

**Q21. What is the difference between a model's "development team" believing an assumption and a model's actual assumptions register, and why does this portfolio insist on the latter being explicit and numbered?**

A developer's belief lives in their head and changes silently as they iterate; an assumptions register is a written, numbered, externally auditable list, each entry paired with what breaks if it's violated — which lets someone who never built the model check it against reality without re-deriving the model's logic. The credit-risk desk guide's scenarios section directly cites specific numbered assumptions failing in named historical episodes: "assumption 6" (the missingness mechanism) breaking during 2020 payment holidays, and single-name factor blindness described in the FX desk guide as "assumption A1's failure mode." Numbering assumptions turns "the model has limits" from a vague disclaimer into a checklist that ongoing monitoring and validation can be run against item by item.

**Q22. How does the FX VaR desk's three-hub, 24-hour operating cycle illustrate a governance problem that a single-timezone desk wouldn't face?**

Per `python/fx/03-var-es-engine/docs/DESK_GUIDE.md` §1, FX never closes, so "the official close is a convention, not a fact of nature" — the desk snaps positions at 17:00 NY by policy, and the EOD batch's NaN policy becomes a live governance decision every single night: "Tokyo and São Paulo holidays do not line up; the desk decides fill-vs-drop explicitly, the engine never imputes." A single-timezone equity desk has one unambiguous close and doesn't face this; the FX desk's governance has to specify, as policy, what "today's data" even means before any model can run on it — a data-completeness decision that in most desks is invisible because the market conveniently closes once.

**Q23. What does "netting across desks" reveal about model risk that a single desk's risk report can hide?**

The FX VaR desk guide §2 shows that because every position — spot, forward leg, option delta — maps to the same USD-pivot factor convention, firm-level risk is literally the sum of desk-level books, and diversification is reported as ΣVaR_desk − VaR_firm. But it flags the model risk this creates directly: "the engine's peg counterexample... is the one-slide explanation of why the desk-level 99% VaRs of two peg books can both be zero while the firm is carrying a full devaluation risk." A model that looks fine at the desk level can be systematically blind at the firm level if the risk factor decomposition itself has a gap — which is why peg exposure gets tracked as a separate notional limit (Q9) rather than being trusted to show up in VaR at any level of aggregation.

**Q24. What role does a stress-testing committee play that a daily VaR report and backtest cannot substitute for?**

VaR and its backtest tell you whether the model's *routine* quantile forecast has been reliable historically; a stress committee asks what happens in scenarios the historical window may never have contained. The equity VaR desk guide §5 makes the point explicit with numbers: on the demo book the historical replays (1987, 2008 Lehman, 2020 COVID) produce full-reval losses 2.5–4.7x the 99% VaR — "the committee's headline ('VaR is not a worst case')." The same section's reverse-stress exercise ("what joint 3-sigma move hurts most?") goes further than either VaR or forward stress by naming the book's actual concentration (AAPL −4.7%, JPM −3.6%, SPX −1.9%) rather than assuming a scenario and checking the loss — it's a control built specifically to compensate for the fact that no stress scenario library can be assumed complete.

**Q25. Give an example where a model's own governance section admits a scenario where its headline risk measure becomes actively useless, and what compensates for it.**

The equity VaR desk guide §6 describes a meme-stock/short-squeeze scenario where a short single-name position shows unbounded upside loss under a +100%/−60% two-sided shock, and states plainly: "VaR (calibrated on pre-squeeze vol) is useless, the ladder is the control." This is a rare and useful kind of honesty in a desk guide — rather than claiming the risk model covers this case, it identifies the case where the primary risk number breaks down and names the specific alternative control (the sensitivity ladder, which shows raw exposure rather than a probabilistic quantile) that catches what VaR structurally cannot.

**Q26. Tie it together: why is "model risk" ultimately a governance discipline about people and process, not just a mathematical property of the model?**

Because every mechanism surveyed above — independent validation, champion/challenger disagreement flags, escalation triggers, limits that don't require trusting the model exactly, retirement tests, versioned change control — exists to catch the case where the model is wrong in a way its own math cannot self-diagnose. A model cannot notice its own blind spot; only an external process (a different team, a different model, a limit, a scheduled re-test, a human escalation) can. That is why CONVENTIONS.md makes "how would it be used on a real desk" one of six mandatory documentation questions alongside the math: a model whose usage, consumers, and controls were never worked out on paper is a model whose governance gap will be discovered by a regulator or a loss, not by a code review.


## Round 18 — Real-World Postmortems & Judgment

**Q1. In plain language, what was the NaN-guard bug, and where in the code does it live?**

A guard written as `if x <= 0: raise ValueError(...)` looks like it rejects bad input, but under IEEE 754 every comparison against NaN — including `NaN <= 0` — evaluates to `False`. So instead of raising, the guard silently falls through and lets a NaN spot, rate, vol, or covariance entry continue into the pricer or risk calculation. This was found across nearly every project in all three languages (Python, C++, Rust) during a portfolio-wide hardening pass and is diagrammed in `docs/DIAGRAMS.md` diagram 9.

**Q2. Why is this specific failure mode more dangerous than the NaN itself?**

Because of a second IEEE 754 quirk one step downstream: `max(NaN, 0.0)` in the Rust and C++ VaR engines propagates the *non-NaN* operand, so a NaN covariance that reaches the portfolio VaR aggregator does not surface as a NaN — it comes out as a portfolio VaR of exactly zero. A NaN price is at least visibly broken; a VaR of exactly 0.0 is a plausible number for a well-hedged book, so nothing downstream (a risk report, a limit check) has any reason to question it. README.md's "On input validation" section calls this out explicitly as the most dangerous manifestation.

**Q3. The README also mentions a third symptom involving backtesting. What was it?**

A NaN in a P&L feed reaching the exception-counting logic caused VaR-exception counters to record zero breaches for that day, because the NaN comparison against the VaR threshold (another `<=`/`>` style check) silently failed to count as a breach. The practical effect: a broken model — one that was actually producing garbage risk numbers — passed its Kupiec proportion-of-failures and Basel traffic-light backtests green, the exact opposite of what those tests exist to catch.

**Q4. Why did the pre-fix test suites not catch this, given the CONVENTIONS.md testing contract calls for "failure behaviour... raise `ValueError`" tests?**

Those tests existed and passed — they fed in `-1.0`, `0.0`, or other ordinary invalid values and confirmed the raise fired, which it did. The blind spot was that nobody had a test case that specifically fed `float('nan')` through the same entry points and asserted that it *also* raises. A comparison-based guard and a finiteness-based guard produce identical behavior on every ordinary invalid input; they only diverge on NaN, which no example-based "does invalid input raise" test happened to include.

**Q5. What kind of test would have caught the NaN-guard bug sooner, and did the fix change any pinned numeric output?**

A test that explicitly parametrizes `float('nan')` (and `float('inf')` where relevant) as one of the "invalid input" cases at every public entry point, rather than relying on ordinary negative/zero examples to stand in for the whole invalid-input space. No — the fix (explicit `isfinite`/`is_finite` checks replacing bare comparisons) only changes what gets rejected before computation begins; it does not touch any formula, so it moved no golden-vector value, per ARCHITECTURE.md's "Design invariants" note that a numerics/robustness fix should never look, from the outside, like a silent formula change.

**Q6. Spot-checking `parametric_var.py`, how does `portfolio_sigma` guard against exactly this class of bug rather than relying on a bare `<=` check?**

`portfolio_sigma` computes `var = w @ Sigma @ w` and then does `float(np.sqrt(max(var, 0.0)))` — the same `max(x, 0.0)` shape that is dangerous elsewhere — but it first raises explicitly if `var < -1e-10 * max(1.0, |w|_max**2)`, a numerical-noise-tolerant negativity check that a NaN also fails (since `NaN < anything` is `False`, a NaN `var` slips past *that* check too — which is exactly why upstream, `robust_cholesky` in `monte_carlo_var.py` separately raises `ValueError` on any non-finite covariance entry via `np.isfinite(a).all()` before the matrix is used at all, catching the NaN before it ever reaches a `max()`).

---

**Q7. In plain language, what was the implied-vol solver plateau bug?**

All six options-pricing engines (Python, C++, Rust; equity and FX) used Newton-Raphson to invert price-vs-vol for implied volatility, and the old stopping rule exited as soon as the price residual stopped improving between iterations. Near a flat region of the price-vs-vol curve — deep ITM/OTM, long-dated, high-vol, anywhere vega is small — Newton can stall (tiny steps, or a step that overshoots and barely improves the residual) well before it has actually converged, and the old code took that stall as a signal to stop rather than as a signal to switch strategy.

**Q8. Why was this worse in the FX engines specifically?**

Per README.md, in the FX engines the stalled solver could saturate at the model's no-arbitrage upper bound on volatility and return that boundary value as "the" implied vol — not merely an imprecise answer but a materially wrong one, because that returned vol does not actually reprice the input option at all. Diagram 10 in DIAGRAMS.md frames this as the difference between "silently imprecise" and, worst case, "a vol that does not reprice the input."

**Q9. Why didn't the existing test suites catch this before the fix?**

Because the tests that existed were fixed-tolerance pass/fail checks: feed a price, solve for implied vol, assert the result is within some tolerance of the expected value. A solver that stops early but happens to land close enough to the true answer passes that test exactly the same way a solver that converged precisely does — the test has no way to distinguish "converged to full precision" from "stopped one or two Newton steps early but was lucky enough to be within tolerance." The plateau bug only shows up in the input regions (deep ITM/OTM, long-dated, high-vol) where "close enough" and "correct" diverge, and a test suite built mostly around typical at-the-money, moderate-expiry cases doesn't visit those regions often enough to expose it.

**Q10. What kind of test would have caught the plateau bug sooner?**

A convergence-rate test rather than a single pass/fail check: verify that the solver's error shrinks as the requested tolerance tightens (e.g. solving to `1e-6` should be measurably more accurate than solving to `1e-3`, and both should actually achieve their requested tolerance) — plus explicit round-trip tests planted specifically in the flat-vega regimes (deep ITM/OTM, long expiry, high vol) rather than only typical at-the-money cases. A solver that silently exits early on a stall will fail a convergence-rate check even when it happens to pass an isolated tolerance check.

**Q11. What was the actual fix, and did it change any golden-vector value?**

Per diagram 10 in DIAGRAMS.md, the fix adds a bracket-bisection refinement stage that always runs to full convergence tolerance rather than exiting whenever Newton progress stalls — the solver falls through to bisection/Brent on a maintained bracket instead of trusting a stalled Newton step. No golden-vector value changed: this is a robustness fix to how the solver behaves near flat vega, not a change to the pricing formula the golden vectors pin, per README.md's "On numerical robustness" section and ARCHITECTURE.md's design invariant that no golden-pinned value ever changes silently.

**Q12. Spot-checking the equity Python engine, how does `implied_vol` in `black_scholes.py` actually implement the "always fall through" behavior?**

`implied_vol` runs a bracketed Newton loop that never lets a candidate step outside a maintained `[lo, hi]` bracket — if a Newton step would leave the bracket, it substitutes the bisection midpoint `0.5 * (lo + hi)` instead of just quitting. Its docstring states directly: "the Newton loop always finishes with" a bracket-refinement stage — i.e. convergence is driven to completion via Brent/bisection on the maintained bracket regardless of whether Newton itself made clean progress, rather than treating a Newton stall as a reason to return early.

---

**Q13. In plain language, what was the Cornish-Fisher domain-check grid-resolution bug?**

The Cornish-Fisher expansion is only a valid quantile function where its derivative with respect to the standard normal quantile stays positive (monotone); for large skew/kurtosis inputs it can become non-monotone, which produces nonsense VaR (e.g. an implied 99% loss quantile smaller than the 95% one). All six VaR/ES engines checked this by scanning a finite grid of z-values and confirming the derivative stayed positive at each sampled point — which can miss a thin non-monotone dip that falls entirely between two grid nodes, silently accepting an invalid quantile as if it were fine.

**Q14. How narrow can that missed dip actually be? Is there a concrete documented example?**

Yes — the docstring in `parametric_var.py`'s `cornish_fisher_domain_ok` documents a specific case: `skew=0.122, excess_kurt=-0.427` is non-monotone on `|z| <= 4` (minimum derivative around -9e-4 near `z ~ 3.1`), yet an 801-point grid scan over that same range reports it as monotone. That is a genuinely thin dip — small enough that a fairly fine grid still steps over it entirely.

**Q15. Why didn't the pre-fix tests catch this?**

Because a grid-based domain check and a grid-based *test* of that check share the same blind spot: a test built around a fixed set of example (skew, kurtosis) pairs only verifies the grid scan's answer at the points the test happens to sample, and if those sample points are themselves not fine enough to straddle the dip, both the implementation and the test agree — wrongly — that the region is monotone. Testing the checker against a handful of "obviously fine" and "obviously bad" (skew, kurtosis) pairs never forces the test to land in the narrow band where the old grid-scan's own resolution failure actually bites.

**Q16. What kind of test would have caught this sooner?**

Property-based or randomized testing across many (skew, excess-kurtosis) combinations, rather than a fixed set of examples — ideally cross-checked against a resolution-independent method: for any given (skew, kurtosis) pair, verify the domain-check's monotone/non-monotone verdict agrees with a fine-grained numerical sweep (or, better, with the closed-form analytic derivative) at a resolution well beyond what the implementation itself uses, across a broad randomized sweep of inputs rather than a handful of hand-picked cases. That kind of sweep is far more likely to land inside a narrow dip like the documented `skew=0.122, excess_kurt=-0.427` case than any fixed example set chosen by hand.

**Q17. What was the fix, and does it still depend on grid resolution at all?**

Per `parametric_var.py`, the derivative of the CF expansion, `dz_cf/dz`, is itself an exact quadratic in z: `g(z) = A z^2 + B z + C` with `A = K/8 - S^2/6`, `B = S/3`, `C = 1 - K/8 + 5S^2/36`. The fix finds the closed-form global minimum of that quadratic on the domain directly — the vertex `-B/(2A)` when it falls inside the interval and `A > 0` (convex case), otherwise whichever endpoint is closer, and always an endpoint for the concave/linear case (`A <= 0`, since a concave function's minimum on a closed interval is always at an endpoint). There is no grid at all in the new check, so there is no spacing for a non-monotone region to hide between.

**Q18. Did the Cornish-Fisher fix change any golden-vector value, and what does the `n_grid` parameter still do in the fixed code?**

No — README.md is explicit that neither the implied-vol nor the Cornish-Fisher fix changed any golden-vector reference value; both are numerical-robustness corrections, not formula changes. In the fixed `cornish_fisher_domain_ok`, `n_grid` is retained purely for API compatibility (and is still validated as `>= 2`) but the docstring states plainly that it "no longer affects the result" — the exact closed-form check replaced the grid scan entirely rather than supplementing it.

---

**Q19. In plain language, what was the Monte Carlo VaR standard-error bias?**

The standard error reported alongside a Monte Carlo VaR figure was computed from a local density estimate at the loss quantile — a Gaussian KDE in Python and the FX C++/Rust engines, an order-statistic finite-difference estimate in the equity C++/Rust engines — using a bandwidth tuned to the bulk of the P&L distribution rather than to the tail where it's actually evaluated. Because the tail is sparsely sampled relative to the bulk, that fixed, bulk-tuned bandwidth under-resolves the tail density, and the resulting SE systematically understates the true sampling variability of the VaR estimate by roughly 9-17% in deep tails or with modest scenario counts.

**Q20. Why is an understated standard error a "directionally overconfident" bug rather than just noise, and where is that documented most precisely?**

Because a standard error is a claim about how much a VaR estimate might swing from one sample to the next — a desk uses it to size confidence intervals and to judge whether an observed VaR change is signal or noise. An SE that's too small every time, systematically, doesn't just add random error to that judgment; it consistently makes the estimate look more precise (and the desk more confident) than it actually is, exactly where the estimate matters most (deep tails, `alpha >= 0.995`). The `monte_carlo_var.py` module docstring quantifies this precisely: at `alpha=0.99` with 50,000 scenarios the KDE is within about 2% of the true SE, but at `alpha=0.999` (50,000 scenarios) or `alpha=0.99` with only 2,000 scenarios it underestimates by 9-13%; README.md gives the portfolio-wide range as roughly 9-17%.

**Q21. Why didn't the existing tests catch this before the fix?**

Because the existing convergence tests, per CONVENTIONS.md, check Monte Carlo results "to 3 standard errors" against a closed-form or reference value — that test only asks whether the point estimate lands inside a band built from the *reported* SE, and a systematically too-narrow SE just makes that band a bit tighter without ever definitively failing the check on any single run (a 9-17% understatement rarely turns a within-3-SE pass into a fail outright). Nothing in that test design compares the reported SE itself against an independent estimate of the true sampling variability — the SE was trusted as an input to the test rather than treated as a quantity under test.

**Q22. What kind of test would have caught the SE bias sooner, and what did the fix actually add?**

A test that benchmarks the reported SE against ground truth directly — e.g. many independent full Monte Carlo VaR replications, whose empirical standard deviation across replications is the true sampling SE, compared against what the KDE/order-statistic estimator reports on any single replication, averaged over many trials to separate systematic bias from trial-to-trial noise (this is exactly the benchmark described in the `monte_carlo_var.py` docstring). The fix does not replace the KDE estimator; it adds a second, distribution-free bootstrap estimator (`var_standard_error_bootstrap` in Python; `mc_bootstrap_se`/`var_standard_error_bootstrap` in C++; `var_bootstrap_se`/`var_standard_error_bootstrap` in Rust) as a cross-check, recommended whenever `alpha >= 0.995` or scenario counts are modest.

**Q23. Spot-checking `monte_carlo_var.py`, how does `var_standard_error_bootstrap` avoid needing a bandwidth at all?**

It resamples the realized scenario P&L with replacement `n_boot` times, applies the exact same order-statistic VaR rule used for the point estimate to each resample (the tail rank is computed once via `_tail()`, since it depends only on `n` and `alpha`, not the data, so it's identical across all resamples of the same size), and reports the standard deviation of the resulting VaR values across resamples — `float(np.std(boot_vars, ddof=1))`. There is no density estimate anywhere in that computation, so it cannot inherit a fixed-bandwidth bias; the tradeoff, per the module docstring, is higher trial-to-trial variance in the SE estimate itself unless `n_boot` is generous.

---

**Q24. In plain language, what was the antithetic-pairing bug in the FX Heston Monte Carlo?**

`simulate_terminal` in `heston_mc.py` draws `n_base = (n_paths + 1) // 2` base normals and mirrors them to `[z, -z]` to build antithetic pairs; the code even documents mechanically what happens if `n_paths` is odd: the mirrored array is truncated back to `n_paths`, so the *last* base draw loses its antithetic partner while every other draw keeps one. `mc_price`'s pairwise averaging — the step that removes the antithetic correlation from the SE calculation by averaging each `(z, -z)` pair before computing sample variance — then treats that broken, half-orphaned pairing as if every draw were still a clean independent pair, which silently understates the reported standard error rather than crashing or producing an obviously wrong price.

**Q25. Why is this the same defect class as the other four, even though it's much narrower in scope?**

Because the common thread across all five case studies is a plausible-looking wrong number rather than a crash: an odd `n_paths` doesn't error out, it just quietly degrades the pairing invariant that the SE calculation depends on, producing an SE that looks like a normal Monte Carlo standard error but is smaller than it should be — the same "confidently wrong, not visibly broken" shape as the NaN-to-zero-VaR bug and the KDE bandwidth bias, just localized to one engine (the FX vol-surface Heston Monte Carlo) instead of appearing across all engines that share an algorithm.

**Q26. What was the fix, and how is it different in kind from the other four fixes in this round?**

The fix in `simulate_terminal` is to reject the bad input explicitly: `if antithetic and n_paths % 2 != 0: raise ValueError(...)`, with a message that states the mechanism directly ("odd n_paths breaks pairing and silently understates the standard error"). This differs from the other four fixes, which each made an existing numerical procedure more correct/robust (finiteness checks, always-converge bisection, exact stationary-point check, a second SE estimator); here the fix instead makes an input that can't be handled correctly into a hard, explicit rejection — closer in spirit to the NaN-guard fix (validate before computing) than to the solver/domain-check fixes (compute more robustly).

---

**Q27. Across all five case studies, what is the one property they all share that makes them dangerous — and how would you check, in general, whether a given fix changed a formula or just its robustness?**

Every one of these five bugs produces a plausible-looking, silently wrong number rather than a crash or an obviously nonsensical output — a zero VaR that looks hedged, an implied vol that looks converged, a Cornish-Fisher quantile that looks like a quantile, an SE that looks appropriately small, a Monte Carlo SE that looks like a normal antithetic-pair SE. That is strictly more dangerous than a loud failure, because a loud failure gets investigated and a quiet one gets trusted. The general check for "formula fix vs. robustness fix" is exactly the one ARCHITECTURE.md's "Design invariants" section states as a hard constraint for this portfolio: every fix is verified against whether it moves any pinned golden-vector value — in all five cases here it does not, which is the evidence that what changed was how the code behaves at the edges of its input domain, not what it computes in the well-behaved middle of that domain.

**Q28. Why does "no golden-vector value changed" matter for how confidently a fix can be shipped, beyond just being a nice property?**

Because it collapses the review burden for the fix down to just the edge-case behavior instead of the whole formula. If a "robustness fix" also happened to move a pinned reference value, you could no longer tell, from the diff alone, whether you fixed the edge case or introduced a regression in the core computation — the two effects would be entangled in the same changed numbers. Confirming the golden vectors are untouched means the well-tested, already-validated middle of the input space is provably unaffected, so all that's left to review is the specific narrow behavior the fix targeted (does it now handle NaN / a stalled Newton step / a thin non-monotone dip / a modest-scenario deep tail / an odd path count correctly) — a much smaller and more checkable claim.

**Q29. Stated as a general principle, what is the transferable lesson across all five postmortems?**

A passing test suite is evidence of correctness only within the space of inputs and conditions it actually exercises — it says nothing about the inputs it never tried. Every one of these five bugs survived its original test suite not because the tests were sloppy, but because they were built, reasonably, around representative examples (typical valid/invalid inputs, typical moneyness and expiry, typical skew/kurtosis, typical scenario counts, even-numbered path counts) rather than around the full space the code actually has to handle in production, including its statistically rare or adversarial corners.

**Q30. What is the specific question this round wants you to walk away asking about any piece of numerical code you review — and how does it differ from asking "does this test pass"?**

Ask "what property should always hold here, regardless of implementation, and am I actually testing that property, or just a handful of examples?" — a NaN input should always be rejected, not just `-1` and `0`; convergence should always improve as tolerance tightens, not just hit one fixed tolerance; a domain check should agree with the exact analytic answer for *any* (skew, kurtosis) pair, not just a hand-picked set; a reported standard error should track the true sampling variability across scenario counts and tail depths, not just satisfy one 3-SE band; an invariant like antithetic pairing should hold for *any* valid input shape, not just the even counts someone happened to test with. "Does this test pass" only tells you the code agrees with the examples you already thought to write down; asking what property must always hold is what surfaces the examples you didn't think to write down.

