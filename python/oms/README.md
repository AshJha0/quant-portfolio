# qp-oms: order state machine, event log and FIX session

Order management for the quant-portfolio projects (workstream W28). This
release holds parts W28a and W28b. The OMS/EMS, drop copy and
reconciliation (W28c) and the pre-trade risk, kill hierarchy, crash recovery
and clock work (W28d) come next and build on these modules.

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
- A lost ResendRequest is not retried yet. That needs the W28d timers.
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
