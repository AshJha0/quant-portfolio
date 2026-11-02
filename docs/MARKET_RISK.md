# Market Risk: The VaR/ES Workflow, End to End

This page walks through the workflow that `python/equity/03-var-es-engine` and
`python/fx/03-var-es-engine` support, from computing a VaR number to
backtesting and stress testing it. It summarises both projects'
`docs/DESK_GUIDE.md` files and notes where equity and FX differ. Read those two
files for the full version. All examples use synthetic data. None of this code
runs on a live book.

## Conventions

- `alpha` is the tail probability, in (0, 0.5). A 99% VaR is `alpha=0.01`. A
  97.5% ES is `alpha=0.025`.
- Historical VaR is the order statistic L(ceil(n * alpha)): the m-th largest
  loss with m = ceil(n * alpha). There is no interpolation.
- Expected Shortfall is the exact Acerbi-Tasche estimator, including the
  fractional atom at the VaR rank.
- EWMA volatility is seeded with the mean of r^2 over the first min(30, n)
  observations.
- Cholesky adds a diagonal jitter capped at 1e-6 times the mean diagonal and
  raises above that cap. `repair="clip"` or `repair="higham"` fixes a matrix
  that is not positive definite.

## The measures, and the three methods

Each VaR/ES number can come from three methods. The engines expose all three.

- **Historical simulation (`historical_var`)**, plain, age-weighted (BRW), or
  filtered (FHS). It takes the empirical tail of scenario P&L with no
  distributional assumption. FHS standardises each factor's returns by a
  conditional volatility before resampling, so it adapts to the current
  regime. Both desk guides use FHS as the primary method.
- **Parametric (`parametric_var`)**: delta-normal, Student-t, or a
  Cornish-Fisher expansion that corrects the normal quantile for skew and
  kurtosis. `cornish_fisher_valid(skew, excess_kurt, alpha)` reports whether the
  expansion is monotone at that point. The method is fast. It assumes a
  covariance matrix plus a shape correction describes the book, which fails for
  large option positions near the money or discontinuous payoffs.
- **Monte Carlo (`monte_carlo_var`)**: factor simulation with normal,
  Student-t or jump-mixture shocks. The Python FX engine reprices spot,
  forward and option positions in every scenario. The C++ and Rust VaR engines
  reprice European options in every scenario too, with vol held at today's
  level (no vol shocks). Every MC figure comes
  with a standard error.

When methods disagree, the gap is a signal. FHS well below parametric means the
recent window is quiet relative to the covariance matrix. A large
historical-versus-parametric gap on an FX book points at tail shape. The equity
desk guide calls this the method-disagreement flag.

## Expected Shortfall as the capital measure

VaR gives a loss threshold. It says nothing about losses beyond it. Expected
Shortfall is the average loss given that the threshold is breached. FRTB binds
capital to 97.5% ES. Both desk guides run 99% VaR alongside it for comparison.

ES is coherent and therefore subadditive: the ES of a combined portfolio never
exceeds the sum of its parts. The FX desk guide uses ES for cross-desk capital
allocation for that reason. It reports `sum(VaR_desk) - VaR_firm` as a
diversification benefit.

The FX engine's peg-currency example shows the limit of history-based VaR. Two
desks each hold a pegged-currency book. The peg held throughout the window, so
each desk shows a VaR of zero. The firm still carries the full devaluation
exposure. A notional limit on peg exposure, run alongside VaR, catches this.
See "Limits" in the FX section below.

## Standard errors

Every Monte Carlo VaR/ES figure ships with a standard error. The primary
estimator is a bootstrap over the simulated losses. It is distribution-free.
The KDE density-at-the-quantile estimate is kept as a diagnostic. It
underestimates the sampling error at deep tails and modest scenario counts. A
limit breach that sits inside one standard error of the limit is treated as a
breach and flagged as statistically marginal. [LEARN.md](LEARN.md) Round 7 has
the mechanics.

## Backtesting

Every VaR series is backtested against realised P&L.

- **`exceptions_from_pnl`** flags each day on which realised loss exceeded the
  ex-ante VaR.
- **`kupiec_pof`** tests whether the exception count is consistent with
  `alpha`. It takes keyword arguments only: `kupiec_pof(n_obs=..., n_exceptions=...,
  alpha=...)`. Its chi-squared(1) reference distribution is oversized at a
  250-day, 1% window. That is a property of the classical test.
  [LEARN.md](LEARN.md) Round 7 covers it.
- **`christoffersen_independence`** and **`christoffersen_cc`** test whether
  exceptions are independent over time. A model can pass Kupiec and fail
  independence when breaches cluster. `exception_cluster_table` shows the
  clusters. Clustering suggests a missing regime, not a wrong average.
- **`basel_traffic_light`** and **`basel_zone_probabilities`** map the
  exception count onto green, yellow and red zones. They work for any window
  length and any `alpha`. Zone edges come from the binomial CDF at 0.95 and
  0.9999. The plus factor is computed on the 250-day equivalent, so the
  multiplier runs from 3.00 in green to 4.00 in red.

The equity desk guide has a worked example on a synthetic GARCH stress period.
An unconditional VaR model lands in the red zone. The FHS model stays green.
The desk guide lists the exception counts.

Exception investigation is a human step. Each exception is attributed to a
market move, a data error or unmapped risk. Backtests run on clean P&L, with
fees and new deals removed. A method that fails Kupiec or Christoffersen for
two consecutive quarters goes through model-risk remediation. The equity desk
guide's usual fix is to replace an unconditional method with FHS.

## Stress testing

VaR and ES describe the likely range of outcomes under current conditions.
They are not worst cases. Both desk guides run a separate stress program. The
equity engine replays three historical episodes (1987, the 2008 Lehman
fortnight, 2020 COVID) against a synthetic demo book. Each replay loses a
multiple of the 99% VaR. The desk guide has the figures.

Stress testing also shows where delta-gamma approximation fails. On the equity
COVID replay, full revaluation and delta-gamma disagree materially on the same
book. A quadratic local approximation is unsafe for large gap moves. The
overnight batch therefore uses full revaluation, and the fast approximation is
used only for intraday limit monitoring.

**Reverse stress** asks for the worst plausible joint move instead of the loss
under a given scenario (`reverse_stress_delta` and `reverse_stress_delta_gamma`
in the equity engine). The answer is a direction. The delta case has a closed
form. The delta-gamma case works in whitened coordinates and raises when it
cannot find a feasible point.

## The FX risk day

FX markets do not close. The FX desk guide describes a 24-hour cycle across
three hubs: a 17:00 New York close, an 18:00 full VaR/ES/stress/backtest run, a
19:00 report, then parametric limit monitoring through Tokyo, London and New
York with a full re-run only after a large book move. Every FX position maps to
the same `FX:CCY`-vs-USD factor convention, so risk handed between hubs nets
without reconciliation.

**Limits.** FX limits differ from equity limits. There are per-pair delta
limits, tenor-bucket DV01 limits from forward legs, per-pair vega limits from
options, and a peg-exposure notional limit. The peg limit is separate from the
VaR limit and does not consume it, because a peg book can show zero VaR while
carrying full devaluation risk. A jump-mixture Monte Carlo VaR
(`monte_carlo_var(dist="jump")`) is run for currencies with peg-break risk. A
normal or Student-t factor model cannot represent a discrete jump.
[LEARN.md](LEARN.md) Round 6 covers the jump-mixture mechanics.

## Where the C++ and Rust engines fit

The C++ and Rust `equity-var-engine` and `fx-var-engine` are ports of the
Python VaR/ES and backtesting design. Their historical and Monte Carlo VaR
reprice European option legs in full in every scenario (BSM for equity,
Garman-Kohlhagen for FX), matched to the Python revaluation at 1e-11 for
historical VaR/ES; vol shocks and P&L explain stay in Python. They
follow the same conventions listed at the top of this page. Their golden
vectors come from the Python reference through the generator pipeline in
[ARCHITECTURE.md](ARCHITECTURE.md) "Cross-language testing". Latency figures
are not quoted yet. See the "Engine latency" table in ARCHITECTURE.md.

Fix history is in [CHANGELOG.md](../CHANGELOG.md).

## Further reading

- `python/equity/03-var-es-engine/docs/DESK_GUIDE.md` and
  `python/fx/03-var-es-engine/docs/DESK_GUIDE.md`: the full workflow and the
  synthetic demo-book numbers.
- `python/equity/03-var-es-engine/docs/METHODOLOGY.md` and
  `python/fx/03-var-es-engine/docs/METHODOLOGY.md`: method choices and the
  assumptions register.
- `python/equity/03-var-es-engine/docs/VALIDATION.md` and
  `python/fx/03-var-es-engine/docs/VALIDATION.md`: how each method is
  validated, with reproducible failure modes.
- [LEARN.md](LEARN.md) Rounds 6 and 7 for VaR/ES theory as a self-test.
- [COOKBOOK.md](COOKBOOK.md) "Market Risk: VaR, Expected Shortfall &
  Backtesting" for runnable code.
