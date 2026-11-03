# Trading-system design: knowledge check (questions only)

Eight design cases, one per component of an electronic trading stack. Each
case has an open design question (talk it through for 20 to 30 minutes, the
way a design round runs) and one graded sizing question with a single
numeric answer. Reference designs and answers are in
[`keys/answers.md`](keys/answers.md). Every graded answer is recomputed by
[`verify.py`](verify.py) by simulation, enumeration or a numerical solve,
and `career-prep/tests/test_final_packs.py` fails if the key and the check
disagree.

Each case names the code in this repository that implements the component
(workstreams W25 to W30 of the v3.0.0 plan). Read the code after you have
sketched your own design, not before. Reference texts: Kleppmann,
*Designing Data-Intensive Applications* (2017), chs. 5, 9 and 11; Harris,
*Trading and Exchanges* (2003); FIX Trading Community, *FIX 4.4
specification*; Avellaneda and Stoikov (2008); Almgren and Chriss (2000).

How a design answer is judged: requirements stated first (throughput,
latency, correctness, recovery), then the data model, then the hot path,
then failure modes and how you would test them. A good answer names the
trade-off it is making and the number that decides it.

## Market data and the book

- **X01** Code: `python/execution/src/qp_exec/lob/`, `cpp/exec-core/`, `cpp/ull/book/`.
  Design a limit order book for one instrument that supports add, cancel,
  modify and execute, gives the best bid and ask in O(1), and can be
  rebuilt from an exchange feed after a gap.
  *Sizing:* the venue sends 5,000,000 messages a second at peak, 64 bytes
  each on the wire. What sustained bandwidth must the feed handler accept,
  in gigabits per second (10^9 bits)?

- **X08** Code: `cpp/ull/ring/`, `cpp/ull/feed/`, `cpp/ull/bench/`.
  Design the path from network card to strategy for a feed handler with a
  single-producer, single-consumer ring between the decoder and the book
  builder. Where do you pin threads, how do you avoid allocation, and what
  happens when the consumer falls behind?
  *Sizing:* a burst arrives at 1,000,000 messages a second for 2 ms while
  the consumer drains 400,000 a second. The ring capacity must be a power of
  two (so the index is a mask). What is the smallest capacity that loses no
  message, if the ring was empty when the burst began?

## Liquidity provision

- **X02** Code: `qp_exec/mm/` (W26).
  Design a quoting engine for a single-dealer market maker: fair value,
  inventory skew, quote width, and the controls that stop it quoting into a
  stale price.
  *Sizing:* use the Avellaneda-Stoikov indifference model. Mid 100,
  inventory q = 10 units long, risk aversion gamma = 0.1, price volatility
  sigma = 2 per square root of the time unit, horizon 0.5. What is the
  reservation price (the average of the bid and ask indifference prices)?

- **X03** Code: `qp_exec/crb/`, `qp_exec/rfq/` (W26).
  Design a central risk book that nets client flow across desks before
  hedging the residual in the market. What is the unit of netting, and how
  do you price an RFQ off the current book?
  *Sizing:* in one interval clients send signed flows of +300, -120, -50
  and +20 lots in the same instrument. What fraction of gross client
  volume is internalised (matched against other clients), as an exact
  fraction?

## Order routing and management

- **X04** Code: `qp_exec/sor/`, `python/execution/venues/` (W27).
  Design a smart order router for a lit market with three venues. What
  inputs does it rank on, how does it avoid chasing stale quotes, and how
  do you measure that it routed well?
  *Sizing:* buy 10,000 shares. Venue A shows 3,000 at 100.00 with a taker
  fee of 0.003 a share; venue B shows 5,000 at 100.00 with a fee of 0.001;
  venue C shows 4,000 at 100.01 with a rebate of 0.002 (a fee of -0.002).
  Displayed size is all that fills. What is the lowest achievable average
  all-in price per share?

- **X05** Code: `python/oms/` (`qp_oms`), `qp_core/eventlog.py` (W28).
  Design an order management system whose state survives a crash: an order
  state machine, an append-only event log, periodic snapshots, and a drop
  copy that reconciles against the venue.
  *Sizing:* the system snapshots every 10,000 events and replays events at
  2 microseconds each. What is the worst-case replay time after a crash, in
  milliseconds?

- **X06** Code: `python/oms/risk/`, `python/oms/kill/` (W28d).
  Design pre-trade risk checks and a kill-switch hierarchy (strategy, desk,
  firm) that cannot be bypassed by a misbehaving strategy.
  *Sizing:* an order-rate limit is a token bucket of capacity 100 that
  refills at 50 tokens a second, full at time 0. A strategy sends 100
  orders at time 0, then 80 orders a second, evenly spaced, for the next 10
  seconds. How many orders are rejected?

## After the trade

- **X07** Code: `python/posttrade/` (W29).
  Design a best-execution and surveillance pipeline: what is stored per
  order, how is each fill benchmarked, and what patterns raise an alert?
  *Sizing:* decision price 100.00, buy 10,000 shares. Fills: 6,000 at
  100.05 and 3,000 at 100.10. The last 1,000 never fill and the stock
  closes at 100.30. What is the implementation shortfall in basis points
  of the decision value, counting the unfilled shares at the close?
