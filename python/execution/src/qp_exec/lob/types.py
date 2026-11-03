"""Value types of the reference book: the published interface.

Prices are integer ticks and quantities are integer lots. Integers keep the
Python reference book and the C++ fast book (``cpp/exec-core``) bit-identical:
there is no rounding to disagree about. Convert at the edge with
``price_ticks = round(price / tick_size)``.

The numeric values of every enum are part of the cross-language contract; the
C++ book and the event file format (``lob/events.py``) use the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Side(IntEnum):
    BUY = 1
    SELL = 2


class OrdType(IntEnum):
    LIMIT = 1
    MARKET = 2


class Tif(IntEnum):
    DAY = 1
    IOC = 2
    FOK = 3


class Peg(IntEnum):
    NONE = 0
    MID = 1  # non-displayed, prices at the midpoint, rounded passively
    PRIMARY = 2  # displayed, joins the same-side best price


class Phase(IntEnum):
    CONTINUOUS = 0
    AUCTION = 1


class Reject(IntEnum):
    NONE = 0
    INVALID = 1  # bad quantity, missing price, forbidden flag combination
    DUPLICATE_ID = 2
    POST_ONLY_WOULD_CROSS = 3
    FOK_UNFILLABLE = 4
    NOT_IN_AUCTION = 5  # order type not accepted during the auction call


@dataclass(frozen=True, slots=True)
class Order:
    """A new order as submitted.

    ``price`` is ignored (use 0) for market and pegged orders.
    ``display_qty`` > 0 makes an iceberg: that many lots show, the rest is
    reserve. ``hidden`` makes the whole order non-displayed. The two cannot be
    combined, and pegs take neither.
    """

    order_id: int
    side: Side
    qty: int
    price: int = 0
    ord_type: OrdType = OrdType.LIMIT
    tif: Tif = Tif.DAY
    post_only: bool = False
    hidden: bool = False
    display_qty: int = 0
    peg: Peg = Peg.NONE


@dataclass(frozen=True, slots=True)
class Fill:
    """One execution. ``aggressor`` is the taker side, or None in an auction."""

    taker_id: int
    maker_id: int
    price: int
    qty: int
    ts: int
    aggressor: Side | None


@dataclass(frozen=True, slots=True)
class AddResult:
    accepted: bool
    reject: Reject
    fills: tuple[Fill, ...]
    resting_qty: int  # quantity left on the book (visible + reserve), 0 if none
