# Service level objectives

## The idea

A service level objective says, in numbers, how good is good enough, and
over what window. The point is not the number itself but what happens when
it is missed: the time left before the objective is broken (the error
budget) decides whether the next week goes on features or on fixing
things. This is the approach of the Google SRE book, chapter 4 ("Service
Level Objectives"; Beyer et al., 2016).

This repository is a library, not a running service. Nobody calls it over
a network, so the usual availability SLO does not apply. What does run on
a schedule, and what users of the library depend on, is listed below, with
an objective for each that can be measured from data the repository
already produces.

## The formal statement

Each objective has a service level indicator (SLI: what is measured, and
where), a target, a window, and the action when the budget is spent.

| # | What | SLI (source) | Target | Window |
|---|---|---|---|---|
| S1 | Correctness of main | Fraction of `ci` runs on `main` where `ci-ok` succeeds, excluding runs cancelled by a newer push (Actions history) | >= 95% | rolling 28 days |
| S2 | Independent checks | Runs on `main` where the `quantlib`, `goldens` or `engine-differential` job fails (Actions history) | 0 | rolling 28 days |
| S3 | Forward log freshness | ECB business days with no line on the `forward-log` branch by 18:00 UTC the same day (branch history, once `FORWARD_LOG_ENABLED` is on) | <= 1 | calendar month |
| S4 | Fuzzing | Nightly `fuzz` findings not reproduced as a regression input or fixed within 7 days (Actions artefacts, `cpp/core/fuzz/regressions/`) | 0 | always |
| S5 | Model inventory | Tier-1 models past `next_review` (`docs/models/models.toml`) | 0 | always |
| S6 | Kernel latency | Per-call p99 and p99.9 of the gated kernels, same-runner A/B (`bench` job) | head p99.9 <= 1.25 x base p99.9 when the per-call bands separate | every pull request |

Budget arithmetic for S1: about 30 pushes to main in 28 days at the
current pace means a 95% target allows 1 red run (5% of 30 is 1.5). A
second red run in the window spends the budget.

### What happens when a budget is spent

- **S1 or S2.** Feature merges stop. The next pull request is the fix, or a
  revert of the change that broke main. S2 has no budget at all: an
  independent check failing on main means a number may be wrong, so the
  runbook entry "An independent check is red" applies at once.
- **S3.** Find the cause (feed late, runner failure, code). Missed days
  are not back-filled by hand (the log is append-only,
  `research/forward/README.md`); the gap is noted in the round handback.
- **S4.** The finding becomes a regression input in the next pull request,
  with a failing test first if it exposed a wrong answer.
- **S5.** Re-review the model (`docs/models/VALIDATION_TEMPLATE.md`) before
  any change to it is merged.
- **S6.** The bench gate already fails the pull request; see the runbook
  entry "The bench gate failed".

## Kernel latency (S6) in detail

Until the per-call timer lands (W12, `qp/bench/latency.hpp`), the harness
times batches of B back-to-back calls and its p99 and p99.9 are
percentiles of B-call batch means (`docs/BENCHMARKS.md`, "The method").
Those figures must not be used for S6: a rare 10 us stall averaged over 32
calls looks like 0.3 us. Until then S6 is measured on the medians only,
which is what the bench gate does today, and this table says so.

Once per-call timing is in place, S6 is computed from single-call samples
(B = 1, `tail_basis = "per-call"`). The acceptance test for that harness
in the plan is that an injected 1 ms stall at a known rate shows up in the
per-call p99.9 at that rate. No absolute nanosecond target is set: hosted
runners are shared virtual machines, so only the same-runner A/B
comparison is meaningful, and the objective is "no tail regression", not
"under N ns".

## Worked example

Suppose in the 28 days to 30 November there are 32 `ci` runs on main
and two fail `ci-ok`: one a flaky network fetch in `install_cxx.sh`, one a
real failure in `rust`. 30 of 32 is 93.75%, below 95%, so S1's budget is
spent. Per the rule above, the next merge is a fix: here, the retry
already used for apt.llvm.org extended to the failing fetch, plus the
`rust` fix. If instead `quantlib` had failed once, S2 would be broken
regardless of S1, and the runbook entry for independent checks applies.

## What is deliberately not here

- No availability SLO: there is no service to be available.
- No wire-to-wire or tick-to-trade latency objective: nothing in this
  repository is connected to a venue (see `docs/BENCHMARKS.md`).
- No objective that depends on paid infrastructure: every SLI above comes
  from free GitHub Actions history or files in the repository.

## References

- Beyer, Jones, Petoff and Murphy (eds.), *Site Reliability Engineering*,
  O'Reilly, 2016, ch. 4.
- Gil Tene, "How NOT to Measure Latency" (2015), on coordinated omission
  and why averaged tails hide stalls.
