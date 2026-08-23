# Market Risk: The VaR/ES Workflow, End to End

This is the desk-level walkthrough of what `python/equity/03-var-es-engine`
and `python/fx/03-var-es-engine` (plus their C++/Rust twins) are actually
built to support: not just "compute a VaR number" but the full daily cycle a
real market-risk function runs around that number. It draws directly from
both projects' own `docs/DESK_GUIDE.md` — read those two files in full for
the complete, unabridged version; this page synthesizes the cross-cutting
workflow both engines share and calls out where equity and FX genuinely
diverge.

## The measures, and why there are three ways to compute each

Every VaR/ES number in this portfolio can be produced three independent
ways, and a real risk function computes all three, not just one:

- **Historical simulation (`historical_var`, plain and age-weighted/BRW,
  or filtered — FHS)** — the empirical quantile of actual historical
  scenario P&L, no distributional assumption. FHS is the primary method on
  both desk guides specifically because it adapts to current volatility
  (returns are standardized by a conditional vol estimate before resampling)
  without assuming a parametric distribution.
- **Parametric / delta-normal (`parametric_var`)**, with a Cornish-Fisher
  expansion correcting the normal quantile for the portfolio's actual skew
  and kurtosis. Fast enough to re-run intraday on every quote request (see
  "The FX risk day" below) — the trade-off is the assumption that a
  covariance matrix plus a distributional correction is enough, which
  breaks down for genuinely nonlinear books (large option positions near
  the money, or discontinuous payoffs).
- **Monte Carlo (`monte_carlo_var`)** with full book revaluation — every
  scenario reprices the actual book (options via Garman-Kohlhagen/
  Black-Scholes, forwards via their deposit legs) rather than using a
  linear or delta-gamma approximation. This is the only one of the three
  that is correct for genuinely path-dependent or strongly convex risk, at
  the cost of being the slowest and needing a standard-error estimate
  attached to every number it produces (see "Standard errors" below).

Both desk guides are explicit that **the disagreement between methods is
itself information, not noise**: FHS running well below parametric says
"the recent window has been quiet relative to what the covariance matrix
implies"; a large historical-vs-parametric gap on an FX book specifically
flags a tail-shape issue the equity desk guide calls a "method-disagreement
flag." A market-risk report that only shows one number has thrown away a
diagnostic a report showing all three gets for free.

## Expected Shortfall, and why it — not VaR — is the capital-allocation measure

VaR answers "how much could we lose, at this confidence level" but says
nothing about *how bad* the loss gets beyond that threshold. Expected
Shortfall is the average loss *given* you're already past the VaR
threshold — a materially different number, and the one Basel's FRTB
framework actually binds capital to (both desk guides confirm 97.5% ES is
the primary regulatory measure, with 99% VaR run alongside as the classical
comparison point).

ES has a property VaR does not: it is coherent, meaning in particular that
it is subadditive — the ES of a combined portfolio never exceeds the sum of
its parts' ES. This is why the FX desk guide specifically uses ES, not VaR,
for cross-desk capital allocation (`ΣVaR_desk − VaR_firm` is reported as a
diversification benefit, but the actual allocation math uses ES because it
adds up coherently). The FX engine's own peg-currency example is the
sharpest illustration of why this matters in practice: two desks, each
running a pegged-currency book, can each show a VaR of exactly zero — the
peg has held throughout the historical window, so history sees no risk —
while the combined firm is carrying a full, uncompensated devaluation
exposure that no VaR number captured at any level. This is exactly the kind
of exposure a **notional limit on peg exposure**, run alongside VaR rather
than instead of it, exists to catch — see "Limits" below.

## Standard errors — an estimate without an error bar is a false precision

Every Monte Carlo VaR/ES figure in this portfolio ships with a standard
error, not just a point estimate — the equity desk guide's example quotes
figures as "$51.0k ± $0.4k," and a limit breach that sits inside one
standard error of the limit is treated as a breach (conservative) but
explicitly flagged as statistically marginal rather than presented with the
same confidence as a clear breach.

This portfolio's own review history found that the *default* standard-error
estimator — a local density-at-the-quantile estimate (Gaussian KDE, or an
order-statistic finite difference, depending on engine) — systematically
underestimates the true sampling error by roughly 9-17% at deep tails or
modest scenario counts, which is exactly the regime a desk relies on the
error bar most. Every Monte Carlo VaR engine in this portfolio (Python,
C++, and Rust, both equity and FX) now also exposes a distribution-free
bootstrap standard-error estimator as a cross-check — see
[LEARN.md](LEARN.md) Round 7 for the full mechanics, and reach for it
specifically whenever `alpha >= 0.995` or the scenario count is modest.

## Backtesting: proving the model is honest, continuously

A VaR number nobody checks against reality is an assumption dressed as a
measurement. Every VaR series in this portfolio is backtested against
realized P&L using the same machinery real regulators require:

- **`exceptions_from_pnl`** flags every day realized loss exceeded the
  ex-ante VaR — an "exception."
- **`kupiec_pof`** (proportion-of-failures) tests whether the exception
  *count* over a window is statistically consistent with the model's
  stated confidence level. Its reference distribution is chi-squared(1),
  which this portfolio's own documentation notes is provably oversized at
  the Basel 250-day/99% window (an honestly documented statistical
  property of the classical test, not a bug — see
  [LEARN.md](LEARN.md) Round 7).
- **`christoffersen_independence`** and **`christoffersen_cc`** go further
  than counting: they test whether exceptions are *independent* over time,
  not just correctly counted. A model can pass Kupiec (right number of
  breaches) while failing independence (all the breaches cluster in one bad
  month) — that clustering, via `exception_cluster_table`, is a "the model
  is missing a regime" signal distinct from "the model has the wrong
  average."
- **`basel_traffic_light`** and **`basel_zone_probabilities`** map the
  250-day exception count onto the regulatory green/amber/red zones and
  the corresponding capital multiplier escalation (3.00 at 0-4 exceptions,
  climbing to 4.00-plus with a presumption of model rejection at 10+). The
  equity desk guide's own worked example is concrete: on a GARCH stress
  period, an unconditional (non-adapting) VaR model lands in the red zone
  (13/250, 11/250 exceptions) while the FHS model that adapts to current
  volatility stays green (3/250) — the capital cost of a model that
  doesn't adapt to volatility is a directly measurable number, not an
  abstract concern.

**Exception investigation is not automatic** — both desk guides describe a
human step: attribute each exception to a real market move vs. a data
error vs. genuinely unmapped risk (dirty P&L containing fees or new deals
must be stripped before backtesting; backtests run on *clean* P&L only). A
method that fails Kupiec or Christoffersen for two consecutive quarters
goes through documented model-risk remediation — the equity desk guide's
typical fix is upgrading an unconditional method to FHS, which is exactly
the adaptivity story the Basel example above makes concrete.

## Stress testing: what VaR is not designed to answer

VaR and ES are statements about the *likely* range of outcomes under
current conditions; they are explicitly not worst-case numbers, and both
desk guides run a separate stress program to answer "what happens in a
scenario like nothing in the recent history." The equity engine's demo
book, run against three historical replays (1987, the 2008 Lehman
fortnight, 2020 COVID), loses 2.5-4.7× its 99% VaR in each — the point of
showing this to a stress committee every month (weekly in volatile regimes)
is precisely to keep "VaR is not a worst case" from being forgotten between
crises.

Stress testing also exposes a specific and important failure mode of
Greeks-based (delta-gamma) intraday risk: on the equity engine's COVID
replay, full revaluation and the delta-gamma approximation disagree by
$93k on the same book — evidence that a linear/quadratic local
approximation is unsafe for genuinely large, gap-like moves, which is why
the overnight batch always fully revalues rather than relying on the fast
approximation used for intraday limit monitoring.

**Reverse stress** inverts the question: rather than "what does scenario X
do to my book," it asks "what is the worst plausible joint move, and what
does it look like" (`reverse_stress_delta` / `reverse_stress_delta_gamma`
in the equity engine). The output is a *direction*, not just a number — the
equity desk guide's example names the specific concentrated exposure
(AAPL, JPM, SPX, and implied vol moving together) that a scenario-only
stress program might never have thought to test explicitly.

## The FX risk day — a genuinely different operating rhythm

FX markets never close, so the FX engine's desk guide describes a
three-hub, 24-hour follow-the-sun cycle rather than a single overnight
batch: an official close convention at 17:00 New York, an 18:00 full
VaR/ES/stress/backtest run, a 19:00 report to desk heads, and then live
parametric-proxy limit monitoring through the Tokyo, London, and New York
sessions with a full-reval re-run only triggered by a large book move. The
FX engine's every position (spot, cross, forward leg, option delta) maps to
the same `FX:CCY`-vs-USD factor convention specifically so that risk
transferred between hubs during handover nets correctly without a
reconciliation step — there is no "Tokyo EURJPY factor" distinct from a
"London EURJPY factor" to line up.

FX also carries a genuinely different limit structure from equity: per
currency-pair delta limits, tenor-bucket DV01 limits (from forward legs),
per-pair vega limits (from options), and — distinctively — a **peg exposure
notional limit that is separate from and does not consume the VaR limit at
all**, precisely because a peg book can show zero VaR while carrying full
devaluation risk (see "Expected Shortfall" above). An EM jump-mixture Monte
Carlo VaR (`monte_carlo_var(dist="jump")`) is run specifically for
currencies with devaluation/peg-break risk, because a normal or Student-t
factor model cannot represent a discrete jump event by construction — see
[LEARN.md](LEARN.md) Round 6 for the jump-mixture mechanics.

## Where the cross-language engines fit

The C++ and Rust equity-var-engine and fx-var-engine are not a separate risk
methodology — they are the same VaR/ES/backtesting math, validated to
1e-6-1e-8 tolerance against the Python engine's own golden vectors, built
specifically for the workloads in this workflow that are actually
latency-sensitive: an intraday parametric VaR re-run on every large quote
request, or a full-book Monte Carlo revaluation on a large book where
Python's per-scenario overhead would make the overnight batch too slow to
finish before the morning risk pack is due. See
[ARCHITECTURE.md](ARCHITECTURE.md) "Cross-language validation" for how that
guarantee is actually built and tested.

## Further reading

- `python/equity/03-var-es-engine/docs/DESK_GUIDE.md` and
  `python/fx/03-var-es-engine/docs/DESK_GUIDE.md` — the full, unabridged
  versions of the workflow summarized above, including the exact demo-book
  numbers referenced here.
- `python/equity/03-var-es-engine/docs/METHODOLOGY.md` and
  `python/fx/03-var-es-engine/docs/METHODOLOGY.md` — why each VaR method
  was chosen over its alternatives, and the full assumptions register.
- `python/equity/03-var-es-engine/docs/VALIDATION.md` and
  `python/fx/03-var-es-engine/docs/VALIDATION.md` — how every method is
  validated, and every documented failure mode with a reproducible example.
- [LEARN.md](LEARN.md) Rounds 6 and 7 for the underlying VaR/ES theory as a
  structured self-test, and Round 18 for the numerical-robustness bugs this
  portfolio's own review history found and fixed in these exact engines.
- [COOKBOOK.md](COOKBOOK.md) "Market Risk: VaR, Expected Shortfall &
  Backtesting" for runnable code covering every measure and backtest
  described on this page.
