# qp-exec

Execution research code: a reference limit order book, order flow, fills
and markouts, impact fitting, Hawkes order-flow models and execution
schedules. Distribution `qp-exec`, import `qp_exec`. Depends on numpy and
scipy only. The C++ fast book that mirrors the reference book is in
`cpp/exec-core`.

```
pip install -e python/execution
python -m pytest -p no:cacheprovider python/execution
```

## What is in it

| Module | What it does | Plan item |
|---|---|---|
| `qp_exec.lob` | `ReferenceBook`: price-time priority, limit, market, IOC, FOK, post-only, hidden, iceberg, midpoint and primary pegs, modify, open/close auction uncross. The rules are in [`lob/SPEC.md`](src/qp_exec/lob/SPEC.md). | W25 |
| `qp_exec.lob.events` | The event format shared with C++, a seeded random stream that uses every order type, replay with a digest per event | W25 |
| `qp_exec.lob.flow` | Zero-intelligence (Smith et al. 2003; Cont, Stoikov and Talreja 2010) and queue-reactive (Huang, Lehalle and Rosenbaum 2015) flow | W25 |
| `qp_exec.lob.fills` | A passive order replayed through flow with latency: queue position over time, partial fills; markouts at 1 s, 10 s, 1 min | W25 |
| `qp_exec.lob.impact` | Order-sign ACF exponent with a block-bootstrap CI, Lillo-Mike-Farmer sign generator, aggregate impact exponent with a bootstrap CI, signature-plot ratio, Binance aggTrades reader | W25 |
| `qp_exec.hawkes` | Exponential-kernel Hawkes, univariate and multivariate: Ogata simulation, O(N) likelihood, MLE with standard errors, time-rescaling KS test | W25c |
| `qp_exec.schedules.volume` | Calibrated intraday volume curves; VWAP and POV on a forecast with catch-up, a live participation cap and a closing-auction slice | W25d |
| `qp_exec.schedules.transient` | Optimal schedules under transient impact: Obizhaeva-Wang exponential and Gatheral power-law kernels; manipulation checks | W25/W26 |
| `qp_exec.schedules.placement` | Post or cross: fill probability from queue position, adverse selection, fees and drift | W25/W26 |

## The interface other work builds on

W24b (queue-aware fills), W26 (market making), W27 (venues and the router)
and W30 (the L3 book) use only the names exported by `qp_exec.lob`:
`ReferenceBook`, `Order`, `Fill`, `AddResult`, `Side`, `OrdType`, `Tif`,
`Peg`, `Phase`, `Reject`, and the event helpers. Prices are integer ticks and
quantities integer lots. A book in another language is correct when its
digest equals `ReferenceBook.digest` after every event of the same stream.

## The two books agree

Plain idea: the fast book is only trusted because it gives the same answer
as the slow, readable one, after every event, not just at the end.

Formal statement: for a seeded stream of events, `exec_core_replay` writes
the C++ book's digest after each event, and `replay_digests` computes the
reference book's. Every pair must be equal.

Worked example and evidence:

```
cmake -S cpp/exec-core -B build/exec-core -DCMAKE_BUILD_TYPE=Release
cmake --build build/exec-core -j
QP_EXEC_REPLAY=build/exec-core/exec_core_replay python -m pytest python/execution/tests/test_differential_cpp.py
python python/execution/tools/lob_differential.py 10000000 2026 <workdir> build/exec-core/exec_core_replay
```

The test runs 2 x 50,000 events by default (`QP_EXEC_DIFF_EVENTS` raises
it). The acceptance run of 10,000,000 events is recorded in the lane
handback (`review/2026-11-03/handbacks/r8x/r12-l1-book.md`) with its output.

## Stylised facts and their bands

| Fact | Band | Source of the band | Test |
|---|---|---|---|
| Order-sign ACF exponent | [0.3, 0.8] | Lillo and Farmer (2004) report 0.2-0.7 on LSE stocks; LMF (2005) gives tail - 1 | `test_sign_acf_exponent_in_named_band` (LMF signs, tail 1.5: about 0.59) |
| Signature-plot ratio RV(1)/RV(50) | [1.2, 3] | bid-ask bounce makes it above 1 (Andersen et al. 2000) | `test_signature_plot_shows_bounce` (ZI: about 1.8) |
| Spread distribution | mode 1-2 ticks, mean 1.5-3 ticks | the shape in Cont, Stoikov and Talreja (2010) | `test_zi_book_stays_valid_and_spread_distribution` |
| Impact exponent | CI compared with 0.5 | square-root law (Toth et al. 2011) | `test_impact_exponent_recovered_with_ci` (synthetic) |

Evidence class: all of these are on simulated flow. The real-data impact
fit on Binance aggTrades is not run in this package's tests (no network in
CI); `load_binance_aggtrades` reads the free daily files from
data.binance.vision. Crypto caveat: crypto venues are fragmented, trade all
day and have different participants, so exponents fitted there are not
evidence about equities.

## What is not here

* No licensed exchange or vendor order-book data (no accounts; data/README.md lists the excluded sources).
* Self-trade prevention is not modelled.
* The C++ book has no Python binding yet; Python talks to it through the
  replay tool and the shared event file.
