"""Order management for quant-portfolio (W28).

Subpackages
-----------
core    FIX order state machine (OrdStatus / ExecType transition table,
        quantity invariants, ClOrdID chains) and the order book that
        journals every event to ``qp_core.eventlog`` before applying it.
fix     Pure-Python FIX 4.4 tag-value codec and session layer: logon and
        acceptor authentication, heartbeats and test requests, persisted
        sequence numbers, ResendRequest / SequenceReset gap fill, PossDup
        and ExecID de-duplication. No external venue: two sessions talk
        over an in-memory transport.

W28c (OMS/EMS, drop copy) and W28d (pre-trade risk, kill hierarchy,
crash recovery, clock) build on these modules in Round 13.
"""

from __future__ import annotations

__version__ = "3.0.0.dev0"

__all__ = ["__version__"]
