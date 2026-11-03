# Trading-system design: answer key

Questions are in [`../questions.md`](../questions.md). Each case gives the
graded sizing answer in backticks with the shortest argument, then a
reference design. The reference design is one good answer, not the only
one; what matters is that each choice is tied to a requirement.
[`../verify.py`](../verify.py) recomputes every graded answer by a
different route and `career-prep/tests/test_final_packs.py` compares them.

## Market data and the book

- **X01** `2.56`. 5 x 10^6 x 64 x 8 bits = 2.56 x 10^9 bits a second. A
  10 GbE port carries it with headroom, a 1 GbE port does not.
  *Reference design.* Price levels in two arrays (bids, asks) indexed by
  tick offset from a reference price, each level holding a FIFO of orders
  (intrusive doubly linked list) and an aggregate size; a hash map from
  order id to its node so cancel and modify are O(1); best bid and ask kept
  as indices and moved only when a level empties. Recovery: on a sequence
  gap, buffer live messages, request a snapshot, apply buffered messages
  with sequence above the snapshot's, then resume. Test with a reference
  book (a slow, obviously correct sorted map) fed the same messages, and
  compare the top N levels after every message.

- **X08** `2048`. The queue grows at (1,000,000 - 400,000) per second for
  0.002 s, peaking at 1,200 messages. The next power of two is 2,048.
  *Reference design.* One isolated core per thread (decoder, book builder),
  busy-polling; the ring is preallocated, slot size fixed, head and tail on
  separate cache lines with acquire/release ordering. When the consumer
  falls behind, decide in advance: drop and mark the book stale (then
  resynchronise from a snapshot, as in X01), or apply back-pressure. A
  market-data ring must never block the network read, so the usual answer
  is to drop, count the drop and resynchronise.

## Liquidity provision

- **X02** `98`. r = s - q gamma sigma^2 tau = 100 - 10 x 0.1 x 4 x 0.5 = 98.
  A long position pulls both quotes down so that sells become more likely
  than buys.
  *Reference design.* Fair value from a microprice (size-weighted mid) or a
  short-horizon model; reservation price as above; half-spread from the
  Avellaneda-Stoikov optimum plus a floor for fees and adverse selection;
  hard inventory limits that switch to one-sided quoting. Controls: cancel
  all quotes when market data is older than a threshold, when the
  reference moves more than k ticks between updates, or on a gateway
  disconnect (cancel-on-disconnect at the venue).

- **X03** `34/49`. Gross 490 lots; net +150. Matched volume is gross minus
  net, 340 lots, so 340/490 = 34/49, about 69%. Only the 150 net lots go to
  the market.
  *Reference design.* Net in risk units, not lots: DV01 buckets for rates,
  delta and vega for options, beta-adjusted notional for equities. An RFQ
  is priced as the current hedge cost of the risk it adds or removes: a
  client trade that offsets the book is priced tighter than one that adds
  to it. Keep an audit trail of the book state at the time of each quote.

## Order routing and management

- **X04** `100003/1000`. Rank by all-in price: B at 100.001, A at 100.003,
  C at 100.008. Take 5,000 from B and 3,000 from A, and the last 2,000
  from C: (500,005 + 300,009 + 200,016) / 10,000 = 100.003.
  *Reference design.* Inputs: displayed size, price, fee schedule, fill
  probability and recent fill latency per venue, and the age of each
  quote. Avoid stale quotes by timing out venue states and sending
  immediate-or-cancel orders. Measure with post-trade mark-outs at several
  horizons and with fill rate against displayed size, split by venue.

- **X05** `19.998`. The worst case is a crash just before the next snapshot:
  9,999 events x 2 microseconds = 19,998 microseconds.
  *Reference design.* Orders move through a finite state machine (new,
  acknowledged, partially filled, filled, cancelled, rejected) and illegal
  transitions are rejected and logged, never silently applied. Every input
  is written to the log before it changes state; snapshots are taken off
  the hot path. On restart: load the last snapshot, replay the tail, then
  reconcile with the venue through the drop copy and raise a break for any
  difference.

- **X06** `300`. The bucket is empty after the burst. From then on it
  refills 50 a second while 80 arrive, so 30 a second are rejected for 10
  seconds.
  *Reference design.* Checks run in the gateway process, not the strategy:
  maximum order size and notional, price collars around a reference,
  position and loss limits, and order-rate limits. The kill hierarchy is a
  tree: tripping a node cancels all orders below it and blocks new ones
  until a human re-enables it. Test by injecting each failure and checking
  that no order leaves the gateway afterwards.

## After the trade

- **X07** `9`. Execution cost 6,000 x 0.05 + 3,000 x 0.10 = 600; opportunity
  cost 1,000 x 0.30 = 300; total 900 on a decision value of 1,000,000, or
  9 bp (Perold 1988).
  *Reference design.* Store every order event with timestamps from one
  synchronised clock, the decision price and the market state at arrival.
  Benchmark each fill against arrival, interval VWAP and a mark-out. Alerts:
  spoofing (large orders cancelled after small fills on the other side),
  marking the close, wash trades between related accounts, and outlier
  slippage by broker or algo.
