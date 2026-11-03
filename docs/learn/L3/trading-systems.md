# Trading systems: backtest core, order path, feed and settlement (L3)

## Plain idea

A strategy is only as honest as the plumbing under it. Four pieces matter
most, and the repository now has one of each instead of one per package.

- **One backtest core.** Every backtester books trades the same way: a
  decision at the close, a fill later at a modelled price, costs paid out of
  the trade, and a book that drifts with the market until the next trade.
- **One order path with one kill switch.** Orders go through an order
  management system that keeps the books straight, and a kill switch can
  stop new risk anywhere from one symbol to the whole firm.
- **A feed handler that admits when it is wrong.** When market data has a
  gap, the local book is marked stale and quoting stops until it is rebuilt.
- **Settlement on real calendars.** A trade is not done until it settles,
  and the settlement date depends on two holiday calendars, not one.

## Formal statement

1. **Fill model (`qp_bt`).** A weight chosen at close `t` fills during day
   `t + 1` at `p_fill = p_t + f (p_{t+1} - p_t)`, with `f = 0.5` by default.
   A cost `c` per unit traded comes out of the trade: buying from flat into
   a weight of 1 leaves a post-trade value of `1 / (1 + c)`. Between trades
   the holdings drift; the trade back to target is a real trade with a real
   cost. Source: `python/backtest/README.md`.
2. **Kill switch (`docs/KILL_CONTRACT.md`).** A kill scope is
   `(firm, desk, strategy, symbol)` with wildcards only as a suffix. K1: a
   new order covered by an active kill is rejected. K2: engaging a kill
   returns every working order it covers, to be cancelled. K3: the kill
   becomes KILLED only when all of those orders are terminal. K4: a reset
   needs a request from one person and approval from a different one. The
   Python reference `python/oms/kill/contract.py` and the C++, Rust and Java
   gateways replay the same vectors, `python/oms/kill/vectors.json`.
3. **Sequenced feed (`cpp/ull/feed/README.md`).** With `next` the sequence
   number after the last one applied, an update with range `[first, last]`
   is a duplicate if `last < next`, is applied if `first <= next <= last`,
   and is held otherwise, which marks the book stale. A hole older than
   50 ms, or more than 64 held updates, triggers one snapshot request.
4. **Settlement (`qp_posttrade.settlement`).** US equities settle T+1 on
   days when both the NYSE and the Federal Reserve are open; the NYSE
   calendar is generated from Rule 7.2 for 2000-2099.

Latency figures from the low-latency code are software, in-process
measurements, not wire-to-wire; `docs/LATENCY_METHOD.md` says so in its first
line, and every table that quotes them must carry the same label.

## Worked example

**Next-bar fill.** One asset, bought at the close on a day before it rises
2 %. Under the v2.0 equity portfolio backtester the weight filled at the
decision close and the day's return was 0.02. Under `qp_bt` with `f = 0.5`
the fill is halfway through the day's move, so the position earns
`1.02 / 1.01 - 1 = 0.0099`, about half.

**Kill switch.** Three working orders: two on the rates desk (strategies MM1
and MM2), one on the FX desk. Alice kills the rates desk. The switch returns
orders 1 and 2 for cancellation and stays ENGAGED; a new MM1 order is
rejected with `KILLED:DESK:QP/RATES/*/*`, while the FX order is untouched.
After both cancels are acknowledged the kill is KILLED. Alice requests a
reset; her own approval is refused and the kill stays RESET_PENDING. Bob
approves, the kill is CLEARED, and MM1 can trade again.

**Settlement.** A US equity trade on Friday 9 October 2026 settles on
Tuesday 13 October: Monday is Columbus Day, when the NYSE is open but the
Fed is shut. A trade on Thursday 31 December 2026 settles on Monday
4 January 2027. (Before the Rounds 8-17 integration this date raised an error,
because the holiday tables stopped at the end of 2026.)

Source: all three are recomputed from the package code in
`docs/learn/tools/test_worked_examples.py::test_trading_systems`.

## Where the repository goes further

- The backtest core, its two modes (event and fast) and the hand-computed
  fixtures: `python/backtest/README.md`.
- The order management system (order book, FIX session, recovery from the
  event log, drop copy and reconciliation): `python/oms/README.md`.
- The C++ and Rust feed handlers and order gateways, and the binary feed
  format: `cpp/ull/`, `rust/ull/` and `docs/QPBF.md`.
- The Java order book and gateway: `java/README.md`.
- Post-trade: booking, transaction cost analysis, best execution and
  surveillance in `python/posttrade/`.

## Checks

1. With `f = 1`, what does the asset in the first example earn on the fill
   day?
2. Why must the approver of a reset be a different person?
3. A Binance depth update covers `[U, u] = [101, 105]` and the last applied
   id is 102. Is it applied, held or dropped?
4. Why does the settlement rule need the Fed calendar as well as the NYSE's?

Answers: nothing, because it fills at the next close and holds none of the
day's move; a kill is pulled when something has gone wrong, and one person
who is wrong, tired or compromised should not be able to restart the flow
alone; applied, since `101 <= 103 <= 105`; cash moves through the Fed's
payment system, so a day the exchange trades but the Fed is closed is not a
settlement day.
