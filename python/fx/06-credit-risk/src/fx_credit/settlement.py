"""FX settlement (Herstatt) risk: time-zone windows and gross vs PvP exposure.

Model
-----
An FX trade settles by two *independent* gross payments unless it goes through
a payment-versus-payment (PvP) system such as CLS.  If we irrevocably pay away
the currency we sold before we receive, with finality, the currency we bought,
then for that window we are exposed to the counterparty for the **full
principal of the bought currency** — not a mark-to-market amount.  This is
Herstatt risk, named after Bankhaus Herstatt (26 June 1974): German banks had
irrevocably paid DEM legs during the European morning; Herstatt's licence was
withdrawn at 15:30 CET, before the corresponding USD legs were paid out in New
York, and the DEM payers lost full principal.

Conventions
-----------
* All times are hours in UTC on the common value date (fractional hours,
  e.g. ``13.5`` = 13:30 UTC).  A single representative "winter" day is
  encoded; daylight-saving shifts are a documented simplification.
* We assume the sold currency is paid at the **opening** of its payment
  system (conservative: earliest irrevocability) and the bought currency is
  received with finality only at the **close** of its payment system
  (conservative: latest finality).
* FX pairs are quoted BASE/QUOTE per the portfolio conventions
  (``EURUSD`` = USD per 1 EUR).
* Exposures are reported in USD using a ``usd_rates`` map: USD per 1 unit of
  each currency.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import pandas as pd

__all__ = [
    "PAYMENT_SYSTEM_HOURS_UTC",
    "FXTrade",
    "SettlementExposure",
    "at_risk_window_hours",
    "time_zone_gap_matrix",
    "settlement_exposure",
    "gross_settlement_exposure",
    "net_settlement_exposure",
    "book_settlement_report",
]

#: Payment-system operating hours in UTC (open, close) on the value date.
#: Simplified single-day windows (RTGS business hours, winter time):
#:   JPY  BOJ-NET / FXYCS   09:00-17:00 JST  -> 00:00-08:00 UTC
#:   EUR  TARGET2 (T2)      07:00-18:00 CET  -> 06:00-17:00 UTC
#:   GBP  CHAPS             08:00-18:00 GMT  -> 08:00-18:00 UTC
#:   USD  Fedwire (core)    08:30-18:30 EST  -> 13:30-23:30 UTC
PAYMENT_SYSTEM_HOURS_UTC: dict[str, tuple[float, float]] = {
    "JPY": (0.0, 8.0),
    "EUR": (6.0, 17.0),
    "GBP": (8.0, 18.0),
    "USD": (13.5, 23.5),
}


@dataclass(frozen=True)
class FXTrade:
    """A single FX trade settling on a common value date.

    Parameters
    ----------
    trade_id : str
        Identifier.
    counterparty : str
        Counterparty name (netting/settlement key).
    pair : str
        6-letter pair, BASE/QUOTE convention, e.g. ``"EURUSD"``.
    notional_base : float
        Notional in units of the BASE currency (>= 0).
    rate : float
        Traded rate: QUOTE units per 1 BASE unit (> 0).
    we_buy_base : bool
        True if we buy the base currency (pay quote); False if we sell base.
    cls_settled : bool, default False
        True if the trade settles PvP through CLS (no principal risk).
    """

    trade_id: str
    counterparty: str
    pair: str
    notional_base: float
    rate: float
    we_buy_base: bool
    cls_settled: bool = False

    def __post_init__(self) -> None:
        if len(self.pair) != 6:
            raise ValueError(f"pair must be 6 letters BASE/QUOTE, got {self.pair!r}")
        if not (math.isfinite(self.notional_base) and self.notional_base >= 0):
            raise ValueError("notional_base must be finite and >= 0")
        if not (math.isfinite(self.rate) and self.rate > 0):
            raise ValueError("rate must be finite and > 0")

    @property
    def base(self) -> str:
        return self.pair[:3]

    @property
    def quote(self) -> str:
        return self.pair[3:]

    @property
    def bought_ccy(self) -> str:
        return self.base if self.we_buy_base else self.quote

    @property
    def sold_ccy(self) -> str:
        return self.quote if self.we_buy_base else self.base

    @property
    def bought_amount(self) -> float:
        """Amount receivable, in units of ``bought_ccy``."""
        return self.notional_base if self.we_buy_base else self.notional_base * self.rate

    @property
    def sold_amount(self) -> float:
        """Amount payable, in units of ``sold_ccy``."""
        return self.notional_base * self.rate if self.we_buy_base else self.notional_base


@dataclass(frozen=True)
class SettlementExposure:
    """Per-trade settlement (Herstatt) exposure result."""

    trade_id: str
    counterparty: str
    sold_ccy: str
    bought_ccy: str
    pay_time_utc: float
    receive_final_utc: float
    at_risk_hours: float
    exposure_usd: float
    cls_settled: bool


def _hours(ccy: str) -> tuple[float, float]:
    try:
        return PAYMENT_SYSTEM_HOURS_UTC[ccy]
    except KeyError as exc:
        raise ValueError(
            f"no payment-system hours encoded for {ccy!r}; "
            f"known: {sorted(PAYMENT_SYSTEM_HOURS_UTC)}"
        ) from exc


def at_risk_window_hours(sold_ccy: str, bought_ccy: str) -> float:
    """Hours of principal risk when paying ``sold_ccy`` and receiving ``bought_ccy``.

    We pay at the open of the sold currency's system and only obtain finality
    on the bought currency at the close of its system.  If the bought currency
    achieves finality *before* we pay (receive-before-pay), there is no
    Herstatt exposure for us (the counterparty bears it instead).  Same
    currency on both legs is not an FX settlement (zero window).

    Returns
    -------
    float
        ``max(0, close(bought) - open(sold))`` in hours; 0 if same currency.
    """
    if sold_ccy == bought_ccy:
        return 0.0
    pay_open, _ = _hours(sold_ccy)
    _, recv_close = _hours(bought_ccy)
    return max(0.0, recv_close - pay_open)


def time_zone_gap_matrix(currencies: tuple[str, ...] = ("JPY", "EUR", "GBP", "USD")) -> pd.DataFrame:
    """At-risk window (hours) for every (sold, bought) currency pair.

    Rows = currency paid away, columns = currency received.  The matrix is
    asymmetric: selling JPY against USD carries a near-full-day window while
    selling USD against JPY carries none — the essence of the time-zone gap.
    """
    mat = pd.DataFrame(
        [
            [at_risk_window_hours(s, b) for b in currencies]
            for s in currencies
        ],
        index=list(currencies),
        columns=list(currencies),
    )
    mat.index.name = "sold \\ bought"
    return mat


def _usd_value(amount: float, ccy: str, usd_rates: dict[str, float]) -> float:
    try:
        rate = usd_rates[ccy]
    except KeyError as exc:
        raise ValueError(f"missing USD rate for {ccy!r}") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"USD rate for {ccy!r} must be finite and > 0, got {rate}")
    return amount * rate


def settlement_exposure(trade: FXTrade, usd_rates: dict[str, float]) -> SettlementExposure:
    """Herstatt exposure of a single trade.

    Exposure is the **full principal of the bought currency** (BCBS 2013
    supervisory guidance definition), converted to USD, whenever the at-risk
    window is positive and the trade is not PvP-settled.

    Parameters
    ----------
    trade : FXTrade
    usd_rates : dict
        USD per 1 unit of currency, must cover both legs.
    """
    hours = at_risk_window_hours(trade.sold_ccy, trade.bought_ccy)
    pay_open = _hours(trade.sold_ccy)[0] if trade.sold_ccy in PAYMENT_SYSTEM_HOURS_UTC else 0.0
    recv_close = _hours(trade.bought_ccy)[1] if trade.bought_ccy in PAYMENT_SYSTEM_HOURS_UTC else 0.0
    if trade.sold_ccy == trade.bought_ccy:
        pay_open = recv_close = 0.0
    exposed = (not trade.cls_settled) and hours > 0.0
    exposure = _usd_value(trade.bought_amount, trade.bought_ccy, usd_rates) if exposed else 0.0
    return SettlementExposure(
        trade_id=trade.trade_id,
        counterparty=trade.counterparty,
        sold_ccy=trade.sold_ccy,
        bought_ccy=trade.bought_ccy,
        pay_time_utc=pay_open,
        receive_final_utc=recv_close,
        at_risk_hours=hours,
        exposure_usd=exposure,
        cls_settled=trade.cls_settled,
    )


def gross_settlement_exposure(trades: list[FXTrade], usd_rates: dict[str, float]) -> float:
    """Total gross Herstatt exposure (USD): sum of per-trade full principals at risk."""
    return sum(settlement_exposure(t, usd_rates).exposure_usd for t in trades)


def net_settlement_exposure(trades: list[FXTrade], usd_rates: dict[str, float]) -> float:
    """Herstatt exposure with same-counterparty, same-currency payment netting.

    For each counterparty, all non-CLS flows in the same currency on the value
    date are netted to a single payable or receivable per currency (bilateral
    payment netting, e.g. under an FXNET-style agreement).  Exposure is then
    the USD value of each net *receivable* whose finality time falls after the
    earliest net *payable* payment time — i.e. the receivable is still unpaid
    while our own money is already out of the door.

    Returns
    -------
    float
        Total netted exposure in USD across counterparties.
    """
    total = 0.0
    key = lambda t: t.counterparty  # noqa: E731
    for _, group in itertools.groupby(sorted(trades, key=key), key=key):
        flows: dict[str, float] = {}
        for t in group:
            if t.cls_settled:
                continue
            flows[t.bought_ccy] = flows.get(t.bought_ccy, 0.0) + t.bought_amount
            flows[t.sold_ccy] = flows.get(t.sold_ccy, 0.0) - t.sold_amount
        payables = {c: -a for c, a in flows.items() if a < 0}
        receivables = {c: a for c, a in flows.items() if a > 0}
        if not payables or not receivables:
            continue
        earliest_pay = min(_hours(c)[0] for c in payables)
        for ccy, amount in receivables.items():
            if _hours(ccy)[1] > earliest_pay:
                total += _usd_value(amount, ccy, usd_rates)
    return total


def book_settlement_report(trades: list[FXTrade], usd_rates: dict[str, float]) -> pd.DataFrame:
    """Per-trade settlement risk report with gross, CLS and window detail.

    Returns a DataFrame with one row per trade: legs, at-risk window, gross
    exposure (0 for CLS-settled trades) — plus the counterfactual exposure
    the trade *would* carry if it were not CLS-settled.
    """
    rows = []
    for t in trades:
        exp = settlement_exposure(t, usd_rates)
        counterfactual = settlement_exposure(
            FXTrade(t.trade_id, t.counterparty, t.pair, t.notional_base, t.rate,
                    t.we_buy_base, cls_settled=False),
            usd_rates,
        )
        rows.append(
            {
                "trade_id": t.trade_id,
                "counterparty": t.counterparty,
                "pair": t.pair,
                "sold": f"{exp.sold_ccy} {t.sold_amount:,.0f}",
                "bought": f"{exp.bought_ccy} {t.bought_amount:,.0f}",
                "cls": t.cls_settled,
                "pay_utc": exp.pay_time_utc,
                "final_utc": exp.receive_final_utc,
                "at_risk_h": exp.at_risk_hours,
                "exposure_usd": exp.exposure_usd,
                "exposure_if_gross_usd": counterfactual.exposure_usd,
            }
        )
    return pd.DataFrame(rows)
