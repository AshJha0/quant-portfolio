# cpp/ull/gateway: order gateway with the shared kill contract

This is part 2 of W30, the low-latency track. The code is header-only C++20.
The kill switch implements `docs/KILL_CONTRACT.md`, and the pre-trade checks
port `python/oms/risk/pretrade.py` (W28d).

## The plain idea

Every new order passes three gates in a fixed order. The first gate that
refuses names the reason:

1. **Kill switch.** If a kill covers the order's (firm, desk, strategy,
   symbol) path, the order is rejected with `KILLED:<level>:<scope>`.
2. **Stale book.** If the feed handler has flagged the symbol's book stale,
   the order is rejected with `STALE_BOOK`. When the flag goes up, the
   gateway also returns every working order on that symbol for
   cancellation. This is how a market maker "pulls quotes".
3. **Pre-trade risk**, in the reference's order: throttle, quantity and
   price sanity, restricted list, duplicate ClOrdID, duplicate content, fat
   finger, max notional, reference price, price band, self-trade, position
   limit, credit limit.

The kill gate runs first so that a killed strategy cannot use up its
throttle budget; a test checks this. The stale gate runs before risk
because a price band measured against a stale reference price means
nothing.

The gateway never sends anything itself. It returns the order ids to
cancel, and the caller owns the venue session.

## Formal statement

- `submit(o)` accepts iff `kill.check(o.path)` is empty, `o.intent.symbol`
  is not stale, and `risk.check_new(o.intent)` accepts. An accepted order is
  registered with both the kill switch (so a later kill returns it for
  cancellation, rule K2) and the risk book (so it counts as exposure).
- `kill.hpp` follows contract rules K1 to K7. It loops over kills and
  working orders in the same order as the Python reference: kills by
  ascending id, which is Python dict insertion order, and working orders by
  ascending id. The order of events is part of the contract.
- `pretrade.hpp` keeps working orders in insertion order, as the
  reference's dict does. It sums notional with the same compensated
  summation that CPython 3.12+ `sum()` uses (Neumaier). As a result, the
  exposure doubles are identical bit for bit, not just within a tolerance.

## Worked example

From `tests/test_gateway.cpp`, case `gateway_kill_then_reset`:

1. Order 1 on `QP/RATES/MM1/X` is accepted.
2. `alice` engages a DESK kill on `QP/RATES/*/*`. The gateway returns
   `[1]` to cancel.
3. A new order on `QP/RATES/MM2/Y` is rejected with
   `KILLED:DESK:QP/RATES/*/*`.
4. The cancel ack for order 1 arrives, and the kill moves to `KILLED`.
5. `alice` requests a reset. `alice` then tries to approve it, which is
   rejected by K4. `bob` approves, the kill clears, and orders flow again.

## Parity evidence

- `python/oms/kill/vectors.json` (3 cases, 24 commands): each output line
  of the C++ replay equals the expected line byte for byte. This is checked
  in `kill_vectors_replay_byte_for_byte`.
- `tests/pretrade_vectors.json`: 1,627 seeded operations generated from the
  Python reference by `tools/make_pretrade_vectors.py`. For each operation
  the test compares the reject reason, the worst-case long and short
  position, and the working notional. The doubles are compared with `==`.
  The vectors cover 13 reject reasons. `THROTTLE` is covered by a separate
  unit test.

The vectors depend on Python 3.12 or later, because `sum()` was not
compensated before 3.12. If they are regenerated under 3.11, the notional
values will differ in the last bits.

Build and run:

```
cmake -S cpp/ull/gateway -B build/ull-gateway && cmake --build build/ull-gateway && ctest --test-dir build/ull-gateway
```

## Sources

- `docs/KILL_CONTRACT.md` and its sources: FIA, *Best Practices for
  Automated Trading Risk Controls and System Safeguards* (2024); SEC Rule
  15c3-5; MiFID II RTS 6, Art. 12.
- CPython `Python/bltinmodule.c`, `builtin_sum_impl`: the compensated float
  summation added in 3.12.
