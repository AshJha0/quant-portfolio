# Reference book semantics (W25)

This is the contract every book in the repository implements: the Python
reference book (`book.py`), the C++ fast book (`cpp/exec-core`), and later
the W30 L3 book and the W27 venues. Where this text and `book.py` disagree,
`book.py` is a bug and so is this text; fix both in one commit.

## Units

Prices are integer ticks, quantities integer lots, timestamps integers. No
floating point anywhere in matching.

## Priority

1. Price: better price first.
2. Within a price level, the **displayed queue** trades before the
   **non-displayed queue**. Displayed: plain limits, iceberg slices and
   primary pegs. Non-displayed: hidden orders and midpoint pegs.
3. Within each queue, time priority by a sequence number. A sequence number
   is assigned when an order rests, when an iceberg slice is refilled, when a
   quantity increase is applied, and when a peg is repriced to a new price.

## Order types

| Type | Rule |
|---|---|
| Limit DAY | Matches what it crosses, rests the remainder. |
| Limit IOC | Matches, the remainder is cancelled. |
| Limit FOK | Rejected with no fills unless the whole quantity is available at acceptable prices, counting hidden quantity and iceberg reserves. |
| Market | Must be IOC or FOK; no price limit. |
| Post-only | DAY limits only. Rejected if it would cross any resting order, displayed or not. Never slides. |
| Hidden | DAY limits only. Rests in the non-displayed queue. |
| Iceberg (`display_qty` > 0) | DAY limits only. Shows `min(display_qty, remaining)`; when the slice is filled it is refilled from the reserve and goes to the back of the displayed queue, inside the same sweep. |
| Midpoint peg | DAY only, no price. Non-displayed. Buy rests at floor((B+A)/2), sell at ceil((B+A)/2). |
| Primary peg | DAY only, no price. Displayed. Buy rests at B, sell at A. |

`B` and `A` are the best bid and ask among **non-pegged** resting orders,
displayed or hidden, read once at the end of each event before any peg moves.
A peg that cannot be priced (its reference side is empty) is inactive: it is
off the book, cannot trade, and can still be cancelled or modified.

Pegs reprice at the end of every event in continuous trading. A peg whose
target price is unchanged keeps its sequence number. Because pegs price off
non-pegged orders only, the only cross repricing can create is a midpoint buy
peg and a midpoint sell peg at the same integer midpoint; it trades at once,
the order with the lower sequence number is the maker.

Self-trade prevention is not modelled.

## Modify

`modify(id, new_qty)` sets the remaining quantity. A decrease keeps priority
and comes out of the iceberg reserve first. An increase loses priority; it is
added to the reserve of an iceberg and to the matchable quantity otherwise.
A price change is a cancel and a new order.

## Auctions

`start_auction()` enters the call phase. Only limit DAY orders that are not
post-only and not pegged are accepted (others: `NOT_IN_AUCTION`). Orders rest
without matching; the book may cross. Active pegs leave the book.

`uncross(ts, ref_price)` picks the price among resting limit prices by:
maximum executable volume, then minimum absolute imbalance, then closest to
`ref_price`, then the lower price. Buys at or above it in price-time order
are paired with sells at or below it in price-time order (displayed before
non-displayed within a level). Every fill is at the one auction price, with
the buy as `taker_id` and no aggressor. Then trading is continuous again and
pegs reprice.

## Digest

`digest(fills)` is FNV-1a over 64-bit words: for each fill
`taker, maker, price, qty`, then `-5`, then `state_words()` (see `book.py`).
The C++ book computes the same words; two books agree on an event when the
digests agree.
