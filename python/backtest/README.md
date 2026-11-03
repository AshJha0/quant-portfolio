# qp-bt

One backtest core for the quant-portfolio strategy projects (W24a). The
distribution is `qp-bt` and the import name is `qp_bt`. It depends on
`qp-core`, numpy and pandas.

```
pip install -e python/core -e python/backtest
python -m pytest -p no:cacheprovider python/backtest
```

## The idea

Before this package, each strategy project had its own backtest loop, with
its own clock, fill rule and cost model, so fills could not be reconciled
from one project to the next. Every loop does the same four steps each bar:

1. carry the book held overnight up to the fill;
2. trade;
3. pay for the trade;
4. carry the new book to the close.

`qp_bt` does these steps once, in two modes:

* **Event mode** (`EventEngine`) holds shares and cash. Each bar runs a fixed
  order of typed events from one queue: `FILL`, then `DECIDE` (the strategy
  sees the close), then `CLOSE` (carry, financing, mark). Event mode is the
  reference. It supports share orders, target positions sized at the fill
  price, signal-to-fill lag and partial fills capped by participation.
* **Fast mode** (`run_fast`) does the same arithmetic on weights, one numpy
  pass per bar. It is for strategies that state a target weight path.

## The rule, formally

`qp_bt.core` states the full rule. For one period `s`, from close `s-1` to
close `s`:

* `h` is the book at close `s-1`, as weights of `NAV_{s-1}`.
* The fill is at `p_{s-1} + phi (p_s - p_{s-1})`.
* The trade is to target `w` as weights of the *post-trade* NAV `N`, where
  `N = m - C(w N - h (1 + a))` and `m = 1 + h . a`. In words, the cost comes
  out of the trade, so the book holds exactly the target after the fill.
* The cash held overnight is financed through `qp_core.stats.Financing`. The
  short book pays a borrow fee at the fill price.

## Worked example

One asset, starting flat. The target is `w = 1`, the cost is 10 bp and the
fill is half-way through a +2% day (`a = 1%`):

* `N = 1 / 1.001`, so the cost is 0.000999.
* `b = 1.02 / 1.01 - 1`.
* The gross return is `N b = 0.0098911` and the net return is `0.0088921`.

## What is checked, and how

| Claim | Evidence | Test |
|---|---|---|
| Both modes match a hand computation | A 3-asset, 10-day book with costs, borrow, financing and leverage, re-computed in exact rational arithmetic in shares and cash. Five fill variants (next bar f = 0.5 and 1, both close-auction bookings, a two-bar lag). Agreement to 1e-13. | `tests/test_hand_fixtures.py`, `tests/oracle.py` |
| Fast mode equals event mode | 20 random long-short fixtures with square-root impact, borrow, and spreads on financing, compared to 1e-10. | `tests/test_fast_equals_event.py` |
| Fills never use the decision close (except an auction, which is charged for) | Unit tests on fill bar and price, lag, and partial fills. | `tests/test_event_engine.py` |
| The migrated projects match the old code, or each difference is listed | A characterisation corpus captured before migration (1,050 arrays). | `tests/characterisation/` |

## Assumptions and limits

* A trade below 1e-12 of NAV is not sent. Anything that small is
  floating-point drift, not an order.
* If the costs of a trade exceed the NAV, the book is wiped out. It then stays
  at zero (an absorbing state).
* Partial fills apply to share orders in event mode only. Fast mode rejects a
  participation cap.

## Queue-aware fills and the portfolio layer

* `qp_bt.fills` holds `NextBarFill` (the bar-level fill above) and the W24b
  queue-aware model for resting limit orders: `QueueAwareFillModel` tracks
  the quantity ahead of a simulated order, attributes cancels in front of or
  behind it by the power rule, and fills on trades at or through its price;
  `fill_marketable` walks the visible book for an IOC order. The model reads
  a book only through the two-method `BookView` protocol.
* `qp_bt.fills.ReferenceBookView` binds the W25 reference book
  (`qp_exec.lob.ReferenceBook`) to `BookView`: ticks to prices, lots to
  floats, and our own order hidden from the public view. `qp_bt` does not
  import `qp_exec`; the book is duck-typed.
* Evidence: `tests/test_queue_fills.py` runs 200 random flows per cancel
  assumption against a minimal FIFO book and the same 400 against the W25
  book, checking fills and queue-ahead after every event. Telling the model
  the wrong cancel side makes seeds fail, so the check discriminates.
* `qp_bt.portfolio` (W24c) holds Ledoit-Wolf and PCA covariance,
  mean-variance with bounds and an L1 turnover cost, risk parity, volatility
  targeting and a walk-forward portfolio P&L.
* The old top-level imports `fills` and `portfolio` (with `python/backtest`
  on the path) still work through shims.
