"""Non-deliverable forwards (NDFs).

Plain idea: an NDF is an FX forward on a currency that cannot be delivered
offshore (BRL, INR, KRW, TWD and others). Nothing is exchanged at maturity
except the difference between the agreed rate and an official fixing, paid
in a deliverable currency, almost always USD.

Formal statement (EMTA NDF templates; Clark 2011, ch. 1). Take a pair
BASE/QUOTE quoted as QUOTE per BASE (USDBRL = BRL per USD), notional ``N``
in BASE, contract rate ``K`` and fixing ``X`` published on the fixing date
by the named source (for USDBRL, "BRL09", the central bank's PTAX rate).
On the settlement date the buyer of BASE receives

* settled in BASE (USD for USDBRL): ``N * (X - K) / X``;
* settled in QUOTE: ``N * (X - K)``.

The fixing is a spot rate for value on the settlement date, so under the
settlement-date forward measures ``E[X] = F`` in the quote currency and
``E[1/X] = 1/F`` in the base currency, where ``F`` is the forward to the
settlement date. Both are exact, with no convexity term. The value today,
in the settlement currency, is

* BASE-settled: ``N * P_base(T_s) * (1 - K / F)``;
* QUOTE-settled: ``N * P_quote(T_s) * (F - K)``, the same as a
  deliverable forward.

``P`` is the settlement currency's discount curve from the
:class:`~fx_rates.fxforward.MarketState` (basis-adjusted when basis quotes
are given), so the settlement currency is the discounting currency.

Fixing date: ``fixing_lag`` business days (2 in most EMTA templates)
before settlement, on the joint calendar of the pair and USD. Onshore
calendars for BRL, INR, KRW and similar are not shipped (see
:mod:`qp_core.holidays`); for those only weekends and the shipped centres
count, so check the published EMTA fixing calendar before relying on a
date.

Worked example: buy USD 10m of USDBRL at 5.55 for settlement on 8 Dec
2025, valued on 5 Jun 2025 (spot date 9 Jun) with spot 5.40 and flat 11%
BRL and 4.5% USD continuous rates. ``F = 5.40 exp(0.065 x 182/365) =
5.5779`` and the value is ``10m x exp(-0.045 x 186/365) x (1 - 5.55/5.5779)
= USD 48,900`` to the nearest 100 (the test computes it exactly).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from qp_core.holidays import CURRENCIES, HolidayCalendar, joint_calendar

from ._validation import require_finite
from .fxforward import MarketState, forward_for_date, year_fraction_365

__all__ = ["NDF", "ndf_fixing_date"]


def _fixing_calendar(pair: tuple[str, str]) -> HolidayCalendar:
    codes = [c for c in dict.fromkeys((*pair, "USD")) if c in CURRENCIES]
    return joint_calendar(*codes)


def ndf_fixing_date(settlement_date: dt.date, pair: tuple[str, str],
                    fixing_lag: int = 2) -> dt.date:
    """Fixing date ``fixing_lag`` business days before ``settlement_date``."""
    if isinstance(fixing_lag, bool) or int(fixing_lag) != fixing_lag or fixing_lag < 0:
        raise ValueError(f"fixing_lag must be a non-negative integer, got {fixing_lag!r}")
    cal = _fixing_calendar((pair[0].upper(), pair[1].upper()))
    d = settlement_date
    for _ in range(int(fixing_lag)):
        d = cal.adjust(d - dt.timedelta(days=1), "preceding")
    return d


@dataclass(frozen=True)
class NDF:
    """Non-deliverable forward, cash-settled against an official fixing.

    ``notional_base > 0`` buys the base currency (gains when the fixing is
    above ``strike``). ``settlement_ccy`` is one of the pair: by default USD
    when USD is in the pair, else the base currency.
    """

    notional_base: float
    strike: float
    fixing_date: dt.date
    settlement_date: dt.date
    pair: tuple[str, str] = ("USD", "BRL")
    settlement_ccy: str | None = None
    fixing_source: str = ""

    def __post_init__(self) -> None:
        require_finite(notional_base=self.notional_base, strike=self.strike)
        if self.strike <= 0.0:
            raise ValueError(f"strike must be > 0, got {self.strike}")
        if self.pair[0] == self.pair[1]:
            raise ValueError(f"pair needs two different currencies, got {self.pair}")
        if self.fixing_date > self.settlement_date:
            raise ValueError(f"fixing date {self.fixing_date} is after settlement "
                             f"{self.settlement_date}")
        ccy = self.settlement_ccy
        if ccy is None:
            ccy = "USD" if "USD" in self.pair else self.pair[0]
            object.__setattr__(self, "settlement_ccy", ccy)
        if ccy not in self.pair:
            raise ValueError(f"settlement_ccy {ccy!r} must be one of {self.pair}")

    def settlement_amount(self, fixing: float) -> float:
        """Cash paid on the settlement date, in ``settlement_ccy``."""
        require_finite(fixing=fixing)
        if fixing <= 0.0:
            raise ValueError(f"fixing must be > 0, got {fixing}")
        diff = self.notional_base * (fixing - self.strike)
        return diff / fixing if self.settlement_ccy == self.pair[0] else diff

    def value(self, market: MarketState) -> float:
        """Present value today in ``settlement_ccy`` (see the module notes)."""
        if market.pair != self.pair:
            raise ValueError(f"market pair {market.pair} does not match NDF pair {self.pair}")
        if market.valuation_date is None:
            raise ValueError("NDF valuation needs a MarketState with valuation_date")
        if self.settlement_date <= market.valuation_date:
            return 0.0
        f = forward_for_date(market, self.settlement_date)
        t = year_fraction_365(market.valuation_date, self.settlement_date)
        if self.settlement_ccy == self.pair[0]:
            df_base = float(market.foreign_curve_adjusted.df(t))
            return self.notional_base * df_base * (1.0 - self.strike / f)
        df_quote = float(market.domestic_curve_adjusted.df(t))
        return self.notional_base * df_quote * (f - self.strike)
