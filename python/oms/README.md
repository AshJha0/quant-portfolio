# qp-oms: order state machine, event log and FIX session

Order management for the quant-portfolio projects (workstream W28):
the order state machine, event log and FIX session (W28a, W28b), the
OMS/EMS with drop copy and reconciliation (W28c), and pre-trade risk, the
kill-switch hierarchy, crash recovery and the clock (W28d).

Nothing here connects to a real venue. Two FIX sessions talk to each other
over an in-memory link, which is enough to test every session rule.

## What is in it

| Module | What it does |
|---|---|
| `qp_core.eventlog` | Append-only log. Each record holds the SHA-256 of the one before it. Writes are fsynced per record by default. A torn last line is truncated on open; any other bad line raises. |
| `qp_oms.core.state` | One order as an immutable value, plus a pure `apply_report` that follows the FIX 4.4 OrdStatus/ExecType rules. A report the rules forbid raises `IllegalTransition` and changes nothing. |
| `qp_oms.core.book` | `OrderBook`: checks the event, logs it, then applies it. `OrderBook.replay(log)` rebuilds the same book from the log alone. |
| `qp_oms.fix.codec` | Tag=value encoder and strict decoder (BodyLength, CheckSum, field order). |
| `qp_oms.fix.session` | Logon with acceptor authentication, heartbeats and TestRequest, persisted sequence numbers, ResendRequest with PossDup resends and gap fill, SequenceReset, Logout, ExecID de-duplication. |
| `qp_oms.ems` | Parent orders, children cut by a `Slicer` (default whole-lot TWAP), ExecID de-duplication, an overfill guard, largest-remainder allocation at the parent average price, kill with mass cancel. |
| `qp_oms.risk` | Pre-trade checks: fat finger, notional, price band, credit and position limits including working-order exposure (a pending cancel counts in full, a pending amend at max(old, new)), ClOrdID and content duplicates, restricted list, self-trade prevention (cancel newest), a per-session throttle that rejects and never queues. |
| `qp_oms.kill` | The kill-switch hierarchy (firm, desk, strategy, symbol) of [docs/KILL_CONTRACT.md](../../docs/KILL_CONTRACT.md), rules K1-K7, with `replay()` and the golden `kill/vectors.json` that the C++, Rust and Java gateways replay byte for byte. |
| `qp_oms.dropcopy` | An independent position keeper fed from drop copies; a reused ExecID with different content is an error. |
| `qp_oms.recon` | Fill-level and position breaks, explained (in flight) or unexplained, written as a JSON-lines break report. One missing fill gives one break. |
| `qp_oms.recovery` | Hash-chained write-ahead journal with an fsync per record before send, torn-tail repair; `WallClock` (monotonic, with its source and a stated 5 ms error bound, the MiFID II RTS 25 context) and `SimClock`; `recover()` (order mass status and execution resend), tested over 1,000 seeded kill points and real process kills. |

## The rules, in one paragraph each

**Quantities.** While an order is working, `CumQty + LeavesQty = OrderQty`.
Once it is filled, canceled, expired, rejected or done for the day,
`LeavesQty = 0`. That is the FIX 4.4 rule (Vol. 4, "Order State Changes"),
so the identity holds only while the order works. Every transition checks
both, and a Hypothesis test drives random sequences of acks, fills,
cancels, replaces, rejects and trade busts through them.

**Status.** `OrdStatus` is computed, not stored, from the FIX 4.4
precedence order: Pending Cancel, Pending Replace, Done for Day, Filled,
Canceled/Expired, Partially Filled, New. A fill that arrives while a
cancel is pending leaves the status at Pending Cancel, as Appendix D
shows.

**Write-ahead.** `OrderBook` and `FixSession` write to the log or store
before anything leaves the process. If the process dies after the write
and before the send, the restarted session still has the message and
resends it when the peer asks for the gap
(`test_restart_after_crash_before_send_recovers_by_resend`).

**Sequence gaps.** A message numbered above the expected one is queued
and one ResendRequest goes out. Queued messages are processed in order
once the gap closes. A number below the expected one is ignored when it
carries `PossDupFlag=Y`; without the flag the session logs out, as FIX 4.4
Vol. 2 requires.

## Worked example

```python
from qp_core.eventlog import EventLog
from qp_oms.core import OrderBook, ExecReport, ExecType

book = OrderBook(EventLog("orders.jsonl"))          # fsync per record
book.submit("A1", side="1", symbol="XYZ", order_qty=1000, price=10.0)
book.on_report(ExecReport(ExecType.NEW, "e1", "A1"))
book.on_report(ExecReport(ExecType.TRADE, "e2", "A1", last_qty=400, last_px=10.0))
o = book.get("A1")
o.ord_status, o.cum_qty, o.leaves_qty               # PARTIALLY_FILLED, 400, 600
```

## Limits

- One outstanding cancel or replace per order. FIX allows a chain of them.
- A lost ResendRequest is not retried yet.
- The `ems`, `risk`, `kill`, `dropcopy`, `recon` and `recovery` folders sit
  outside `src/` and join `qp_oms` through their test `conftest.py`, so they
  are collected by pytest but not installed with the distribution. Moving
  them under `src/qp_oms/` waits on the Java parity test, which reads
  `python/oms/kill/vectors.json` by path.
- The journal fsyncs every record; group commit is not implemented.
- FIX 4.4 only, no encryption, no network transport. A QuickFIX
  interoperability job is planned as optional.
- The Appendix D tests follow the scenario groups (vanilla, cancel,
  replace, unsolicited, restatement, trade cancel). They do not yet
  reproduce every numbered table row.

## References

- FIX Trading Community, *FIX 4.4 Specification*, Vol. 2 (session protocol)
  and Vol. 4 Appendix D (order state change matrices), 2003.
- SEC Rule 17a-4(f), on records that cannot be rewritten or erased.
- Haber and Stornetta, "How to time-stamp a digital document", *J.
  Cryptology* 3 (1991), on hash chaining.

## Tests

```
cd python/oms
PYTHONPATH="src;../core/src" python -m pytest      # Windows path separator
```
