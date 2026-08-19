"""Covered-interest arbitrage detector with bid/ask transaction costs.

Setup: pair BASE/QUOTE (EURUSD), domestic = quote ccy (USD), foreign = base
ccy (EUR).  Deposits are simple interest over accrual fraction ``tau``.
Two round trips, executed entirely at *adverse* sides of the market:

**Sell-forward arbitrage** (forward rich, F above the CIP band):
borrow 1 unit domestic at ``dom_rate_ask``, buy foreign spot at
``spot_ask``, invest at ``for_rate_bid``, sell the proceeds forward at
``fwd_bid``.  P&L at T per unit domestic borrowed:

    pnl = fwd_bid * (1 + r_f_bid * tau) / spot_ask - (1 + r_d_ask * tau)

**Buy-forward arbitrage** (forward cheap, F below the band): borrow foreign
at ``for_rate_ask``, sell spot at ``spot_bid``, invest domestic at
``dom_rate_bid``, buy the foreign repayment forward at ``fwd_ask``.  P&L at
T per unit *domestic* deployed (i.e. per ``spot_bid`` of foreign borrowed):

    pnl = (1 + r_d_bid * tau) - fwd_ask * (1 + r_f_ask * tau) / spot_bid

The implied no-arbitrage band for the forward is::

    F_lower = spot_bid * (1 + r_d_bid tau) / (1 + r_f_ask tau)
    F_upper = spot_ask * (1 + r_d_ask tau) / (1 + r_f_bid tau)

No detection fires while the tradeable forward sits inside the band —
post-2008 the *observed* mid-CIP deviation routinely exceeds bid/ask and
still is not free money once dealer balance-sheet costs are included
(Du–Tepper–Verdelhan; see docs/VALIDATION.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import require_finite

__all__ = ["CIPQuotes", "CIPArbitrageResult", "no_arb_bounds", "detect_cip_arbitrage"]


@dataclass(frozen=True)
class CIPQuotes:
    """Two-sided market for one CIP round trip at a single tenor.

    Rates are simple annualised deposit rates over ``tau`` (years, already
    day-counted, e.g. ACT/360).  Bids must not exceed asks; spot/forward
    quotes must be positive.  Deposit *rates* may be negative (EUR/CHF/JPY).
    """

    spot_bid: float
    spot_ask: float
    fwd_bid: float
    fwd_ask: float
    dom_rate_bid: float
    dom_rate_ask: float
    for_rate_bid: float
    for_rate_ask: float
    tau: float

    def __post_init__(self) -> None:
        # Finiteness first: with a NaN quote every comparison below is False,
        # so the detector would report a confident "no arbitrage".
        require_finite(
            spot_bid=self.spot_bid, spot_ask=self.spot_ask,
            fwd_bid=self.fwd_bid, fwd_ask=self.fwd_ask,
            dom_rate_bid=self.dom_rate_bid, dom_rate_ask=self.dom_rate_ask,
            for_rate_bid=self.for_rate_bid, for_rate_ask=self.for_rate_ask,
            tau=self.tau,
        )
        if self.tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if min(self.spot_bid, self.spot_ask, self.fwd_bid, self.fwd_ask) <= 0.0:
            raise ValueError("spot and forward quotes must be > 0")
        for label, bid, ask in (
            ("spot", self.spot_bid, self.spot_ask),
            ("forward", self.fwd_bid, self.fwd_ask),
            ("domestic deposit", self.dom_rate_bid, self.dom_rate_ask),
            ("foreign deposit", self.for_rate_bid, self.for_rate_ask),
        ):
            if bid > ask:
                raise ValueError(f"{label} bid {bid} exceeds ask {ask}")


@dataclass(frozen=True)
class CIPArbitrageResult:
    """Outcome of the detector.

    ``pnl`` is the riskless P&L at maturity in **domestic currency per unit
    of domestic notional deployed** (0.0 if no arbitrage).  ``direction`` is
    ``"sell_forward"``, ``"buy_forward"`` or ``"none"``.  ``f_lower`` /
    ``f_upper`` bound the no-arbitrage band for the forward.
    """

    is_arbitrage: bool
    direction: str
    pnl: float
    f_lower: float
    f_upper: float


def no_arb_bounds(q: CIPQuotes) -> tuple[float, float]:
    """No-arbitrage band (F_lower, F_upper) for the tradeable forward."""
    f_lower = q.spot_bid * (1.0 + q.dom_rate_bid * q.tau) / (1.0 + q.for_rate_ask * q.tau)
    f_upper = q.spot_ask * (1.0 + q.dom_rate_ask * q.tau) / (1.0 + q.for_rate_bid * q.tau)
    return f_lower, f_upper


def detect_cip_arbitrage(q: CIPQuotes, min_pnl: float = 0.0) -> CIPArbitrageResult:
    """Detect a CIP violation executable through the bid/ask.

    Parameters
    ----------
    q : CIPQuotes
        Two-sided quotes at one tenor.
    min_pnl : float
        Minimum P&L per unit domestic notional (at maturity) to flag —
        use this to encode residual costs (settlement, capital) beyond
        bid/ask.

    Returns
    -------
    CIPArbitrageResult
        Direction and P&L of the *better* round trip if it clears
        ``min_pnl``, else a no-arbitrage result.  Both directions cannot be
        simultaneously profitable when bid <= ask everywhere.
    """
    require_finite(min_pnl=min_pnl)
    f_lower, f_upper = no_arb_bounds(q)
    pnl_sell = q.fwd_bid * (1.0 + q.for_rate_bid * q.tau) / q.spot_ask - (
        1.0 + q.dom_rate_ask * q.tau
    )
    pnl_buy = (1.0 + q.dom_rate_bid * q.tau) - q.fwd_ask * (
        1.0 + q.for_rate_ask * q.tau
    ) / q.spot_bid
    best_pnl, direction = max((pnl_sell, "sell_forward"), (pnl_buy, "buy_forward"))
    if best_pnl > min_pnl:
        return CIPArbitrageResult(True, direction, best_pnl, f_lower, f_upper)
    return CIPArbitrageResult(False, "none", 0.0, f_lower, f_upper)
