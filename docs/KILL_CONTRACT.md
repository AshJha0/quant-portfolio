# Kill-switch contract (W28d, shared with W30)

This is the one kill switch for the whole order path. The Python OMS
(`qp_oms.kill`) is the reference implementation. The W30 gateways in C++,
Rust and Java implement the same contract and must replay the golden
vectors in `python/oms/kill/vectors.json` to the same outputs, byte for
byte after canonical JSON encoding.

## The plain idea

A kill switch stops new risk from leaving the building. When someone pulls
it, three things happen: new orders in that part of the business are
rejected, every working order there is cancelled, and nothing restarts
until two different people agree it should. Kills nest: killing a desk
kills every strategy and symbol on that desk.

## Formal statement

### Scope

A scope is a tuple `(firm, desk, strategy, symbol)`. A kill is placed at
one of four levels; the fields below the level are the wildcard `*`.

| Level    | Example scope                     |
|----------|-----------------------------------|
| FIRM     | `("QP", "*", "*", "*")`           |
| DESK     | `("QP", "RATES", "*", "*")`       |
| STRATEGY | `("QP", "RATES", "MM1", "*")`     |
| SYMBOL   | `("QP", "RATES", "MM1", "UST10Y")`|

An order with path `(f, d, s, y)` is *covered* by a kill scope when every
non-wildcard field of the scope equals the order's field. Wildcards may
only appear as a suffix; `("QP", "*", "MM1", "*")` is invalid.

### States of one kill

```
ENGAGED --(all cancels acked)--> KILLED --(reset requested by A)--> RESET_PENDING
RESET_PENDING --(approved by B != A)--> CLEARED   (kill removed)
RESET_PENDING --(approved by A)--> rejected, state unchanged
ENGAGED --(reset requested)--> rejected: cannot reset while cancels are outstanding
```

### Rules

- **K1 (block).** A new order is rejected if any active kill (any state other
  than CLEARED) covers it. The reject reason is `KILLED:<level>:<scope>`,
  naming the widest covering kill.
- **K2 (mass cancel).** Engaging a kill returns the ids of every working order
  it covers, sorted ascending. The caller must send a cancel for each.
- **K3 (acknowledged).** The kill moves to KILLED only after every one of those
  orders reports a terminal state (cancelled, filled, rejected or expired).
  A fill racing the cancel counts as terminal; the fill is still booked.
  If there is nothing to cancel, the kill is KILLED at once.
- **K4 (two-person reset).** A reset needs a request from actor A and an
  approval from a different actor B. Same-actor approval is rejected and
  recorded.
- **K5 (nesting).** Engaging a kill inside an active wider kill is recorded
  but does not change blocking. Clearing the wider kill leaves narrower kills
  active.
- **K6 (cancel on disconnect).** A session disconnect engages a SESSION-tagged
  kill at the STRATEGY level of every strategy on that session (actor
  `system:cod`). The same two-person reset applies.
- **K7 (idempotent).** Engaging a scope that already has an active kill
  returns the existing kill's id and an empty cancel list.

### Wire form (for parity)

Every transition is emitted as one canonical JSON object: keys sorted,
no spaces, UTF-8. Fields: `seq`, `event`, `kill_id`, `level`, `scope`
(list of four strings), `actor`, `state`, `cancel`, `reason`. The golden
vectors file lists commands and the expected event stream.

## Worked example

1. Order 7 is working on `("QP","RATES","MM1","UST10Y")`.
2. Risk officer `alice` engages DESK `("QP","RATES","*","*")`. Output:
   `cancel=[7]`, state ENGAGED.
3. A new order on `("QP","RATES","MM2","UST2Y")` is rejected:
   `KILLED:DESK:QP/RATES/*/*`.
4. Order 7 is acknowledged cancelled. State KILLED.
5. `alice` requests reset; `alice` approves: rejected (K4).
6. `bob` approves: CLEARED. Orders flow again.

## Sources

- FIA, *Best Practices for Automated Trading Risk Controls and System
  Safeguards* (2024): kill switch scope, cancel-on-disconnect.
- SEC Rule 15c3-5 (market access rule): pre-trade controls under the
  broker's direct and exclusive control.
- MiFID II RTS 6, Art. 12 (kill functionality: cancel all unexecuted orders).
- Two-person reset is this project's rule, following the four-eyes principle
  common on bank desks; it is a design choice, not a regulatory quote.
