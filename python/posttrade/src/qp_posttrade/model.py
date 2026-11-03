"""Fill and order-event records the post-trade layer consumes.

The seams are deliberately small. ``Fill`` carries the fields of the OMS drop
copy (``qp_oms`` ``DropCopyFill``: exec_id, account, symbol, side, qty, px)
plus the three a post-trade desk also needs: trade time, venue and the
arrival (decision) price of the parent. ``OrderEvent`` is one line of the
attributable W28 audit log (who did what to which order, when).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Side = Literal["BUY", "SELL"]
EventKind = Literal["NEW", "CANCEL", "FILL"]


@dataclass(frozen=True)
class Fill:
    exec_id: str
    account: str
    symbol: str
    side: Side
    qty: float
    px: float
    ts: datetime
    venue: str
    parent_id: str = ""
    arrival_px: float = 0.0  # mid at the parent's decision time
    mid_at_fill: float = 0.0  # prevailing mid when the fill printed
    fee: float = 0.0  # currency units, positive = paid

    def __post_init__(self) -> None:
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {self.side!r}")
        if self.qty <= 0 or self.px <= 0:
            raise ValueError("qty and px must be positive")

    @property
    def sign(self) -> int:
        return 1 if self.side == "BUY" else -1


@dataclass(frozen=True)
class OrderEvent:
    """One audit-log event. ``trader`` makes the log attributable."""

    ts: float  # seconds since session start
    trader: str
    account: str
    symbol: str
    order_id: str
    kind: EventKind
    side: Side
    qty: float
    px: float
    beneficial_owner: str = ""  # for wash detection; defaults to account

    @property
    def owner(self) -> str:
        return self.beneficial_owner or self.account
