"""Native instruments priced on a :class:`MarketSnapshot`.

Each class states its pricing formula; all values are in the currency of
the returned :class:`Money`. Times are ACT/365F years from ``market.as_of``
(a float maturity is taken as already in years). A flow whose time is
negative has settled and is worth nothing today; a flow at ``t = 0`` is cash
today.

Formulas
--------
``Cash``            ``V = x P_c(t)``.
``FixedCashflows``  ``V = sum_i x_i P_c(t_i)``: zero-coupon and fixed-coupon
                    bonds, fixed legs, any known schedule.
``FXForward``       ``V = N (S P_d(t_s)/P_f(t_s)) P_f(T) - N K P_d(T)`` in the
                    quote currency (Clark 2011, ch. 1); identical to
                    ``fx_rates.FXForward.value(method="cashflows")``.
``FXSwap``          near leg plus far leg, each an ``FXForward``; legs already
                    settled drop out.
``EquityPosition``  ``V = S`` per share.
``EquityOption``    Black-Scholes-Merton with ``r = z_c(T)``, ``q`` from
                    ``DIV/<name>`` (0 if absent), ``sigma`` from surface
                    ``EQ/<name>`` at ``(T, K)`` (Hull 2022, ch. 15), via
                    :func:`qp_core.black_scholes.carry_price`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..black_scholes import carry_price
from ..errors import InputError
from ..market.snapshot import MarketSnapshot
from ..validation import require_finite
from .base import Cashflow, Maturity, Money, bump_risk, maturity_from_json, maturity_to_json

__all__ = ["Cash", "EquityOption", "EquityPosition", "FXForward", "FXSwap", "FixedCashflows"]


def _pair(pair: str) -> tuple[str, str]:
    if len(pair) != 6 or not pair.isalpha() or not pair.isupper():
        raise InputError(f"pair must look like 'EURUSD', got {pair!r}")
    if pair[:3] == pair[3:]:
        raise InputError(f"same-currency pair {pair!r}")
    return pair[:3], pair[3:]


def _alive(t: float) -> bool:
    return t >= 0.0


@dataclass(frozen=True)
class Cash:
    """``amount`` of ``ccy`` paid at ``when`` (default: today)."""

    ccy: str
    amount: float
    when: Maturity = 0.0

    def __post_init__(self) -> None:
        require_finite(amount=self.amount)

    def pv(self, market: MarketSnapshot) -> Money:
        t = market.time_to(self.when)
        if not _alive(t):
            return Money(0.0, self.ccy)
        return Money(self.amount if t == 0.0 else self.amount * market.df(self.ccy, t), self.ccy)

    def cashflows(self) -> list[Cashflow]:
        return [Cashflow(self.when, self.ccy, self.amount, "cash")]

    def risk(self, market: MarketSnapshot) -> dict[str, float]:
        t = market.time_to(self.when)
        return bump_risk(self.pv, market, curves=[self.ccy] if t > 0 else [])

    def to_dict(self) -> dict[str, Any]:
        return {"ccy": self.ccy, "amount": self.amount, "when": maturity_to_json(self.when)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Cash:
        return cls(d["ccy"], d["amount"], maturity_from_json(d["when"]))


@dataclass(frozen=True)
class FixedCashflows:
    """A known schedule of ``(when, amount)`` flows in one currency."""

    ccy: str
    flows: tuple[tuple[Maturity, float], ...]
    label: str = ""

    def __init__(self, ccy: str, flows: Sequence[tuple[Maturity, float]], label: str = "") -> None:
        f = tuple((w, float(a)) for w, a in flows)
        require_finite(**{f"flow{i}": a for i, (_, a) in enumerate(f)})
        object.__setattr__(self, "ccy", ccy)
        object.__setattr__(self, "flows", f)
        object.__setattr__(self, "label", label)

    @classmethod
    def bullet_bond(cls, ccy: str, notional: float, coupon: float,
                    pay_dates: Sequence[Maturity], accruals: Sequence[float]) -> FixedCashflows:
        """Fixed-coupon bullet: ``notional * coupon * accrual_i`` plus notional at the end."""
        if len(pay_dates) != len(accruals) or not pay_dates:
            raise InputError("pay_dates and accruals must be non-empty and equal length")
        flows = [(d, notional * coupon * a) for d, a in zip(pay_dates, accruals, strict=True)]
        flows[-1] = (flows[-1][0], flows[-1][1] + notional)
        return cls(ccy, flows, "bond")

    def pv(self, market: MarketSnapshot) -> Money:
        c = market.curve(self.ccy) if self.flows else None
        total = 0.0
        for w, a in self.flows:
            t = market.time_to(w)
            if _alive(t) and c is not None:
                total += a * c.df(t)
        return Money(total, self.ccy)

    def cashflows(self) -> list[Cashflow]:
        return [Cashflow(w, self.ccy, a, "fixed") for w, a in self.flows]

    def risk(self, market: MarketSnapshot) -> dict[str, float]:
        return bump_risk(self.pv, market, curves=[self.ccy])

    def to_dict(self) -> dict[str, Any]:
        return {"ccy": self.ccy, "label": self.label,
                "flows": [[maturity_to_json(w), a] for w, a in self.flows]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FixedCashflows:
        return cls(d["ccy"], [(maturity_from_json(w), a) for w, a in d["flows"]], d.get("label", ""))


@dataclass(frozen=True)
class FXForward:
    """Long ``notional_base`` of the base currency at ``strike`` on ``delivery``."""

    pair: str
    notional_base: float
    strike: float
    delivery: Maturity

    def __post_init__(self) -> None:
        _pair(self.pair)
        require_finite(notional_base=self.notional_base, strike=self.strike)
        if self.strike <= 0.0:
            raise InputError(f"strike must be > 0, got {self.strike}")

    @property
    def base(self) -> str:
        return self.pair[:3]

    @property
    def quote_ccy(self) -> str:
        return self.pair[3:]

    def pv(self, market: MarketSnapshot) -> Money:
        t = market.time_to(self.delivery)
        if not _alive(t):
            return Money(0.0, self.quote_ccy)
        n, k = self.notional_base, self.strike
        df_d = market.curve(self.quote_ccy).df(t)
        df_f = market.curve(self.base).df(t)
        return Money(n * market.fx_today(self.base, self.quote_ccy) * df_f - n * k * df_d, self.quote_ccy)

    def cashflows(self) -> list[Cashflow]:
        return [Cashflow(self.delivery, self.base, self.notional_base, "fx"),
                Cashflow(self.delivery, self.quote_ccy, -self.notional_base * self.strike, "fx")]

    def risk(self, market: MarketSnapshot) -> dict[str, float]:
        return bump_risk(self.pv, market, quotes=[f"FX/{self.pair}"], curves=[self.quote_ccy, self.base])

    def to_dict(self) -> dict[str, Any]:
        return {"pair": self.pair, "notional_base": self.notional_base, "strike": self.strike,
                "delivery": maturity_to_json(self.delivery)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FXForward:
        return cls(d["pair"], d["notional_base"], d["strike"], maturity_from_json(d["delivery"]))


@dataclass(frozen=True)
class FXSwap:
    """Buy ``notional_base`` at ``near_rate`` on ``near``, sell ``far_notional_base`` at ``far_rate`` on ``far``.

    ``far_notional_base`` defaults to ``notional_base`` (an even swap).
    """

    pair: str
    notional_base: float
    near_rate: float
    far_rate: float
    near: Maturity
    far: Maturity
    far_notional_base: float | None = None

    def legs(self) -> tuple[FXForward, FXForward]:
        nf = self.notional_base if self.far_notional_base is None else self.far_notional_base
        return (FXForward(self.pair, self.notional_base, self.near_rate, self.near),
                FXForward(self.pair, -nf, self.far_rate, self.far))

    def pv(self, market: MarketSnapshot) -> Money:
        a, b = self.legs()
        return a.pv(market) + b.pv(market)

    def cashflows(self) -> list[Cashflow]:
        a, b = self.legs()
        return a.cashflows() + b.cashflows()

    def risk(self, market: MarketSnapshot) -> dict[str, float]:
        return bump_risk(self.pv, market, quotes=[f"FX/{self.pair}"], curves=[self.pair[3:], self.pair[:3]])

    def to_dict(self) -> dict[str, Any]:
        return {"pair": self.pair, "notional_base": self.notional_base, "near_rate": self.near_rate,
                "far_rate": self.far_rate, "near": maturity_to_json(self.near),
                "far": maturity_to_json(self.far), "far_notional_base": self.far_notional_base}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FXSwap:
        return cls(d["pair"], d["notional_base"], d["near_rate"], d["far_rate"],
                   maturity_from_json(d["near"]), maturity_from_json(d["far"]), d.get("far_notional_base"))


@dataclass(frozen=True)
class EquityPosition:
    """One share of ``name`` quoted in ``ccy`` (spot quote ``EQ/<name>``)."""

    name: str
    ccy: str

    def pv(self, market: MarketSnapshot) -> Money:
        return Money(market.quote(f"EQ/{self.name}"), self.ccy)

    def cashflows(self) -> list[Cashflow]:
        return []

    def risk(self, market: MarketSnapshot) -> dict[str, float]:
        return bump_risk(self.pv, market, quotes=[f"EQ/{self.name}"])

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ccy": self.ccy}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EquityPosition:
        return cls(d["name"], d["ccy"])


@dataclass(frozen=True)
class EquityOption:
    """European option on one share of ``name``; see the module formula."""

    name: str
    ccy: str
    strike: float
    expiry: Maturity
    call: bool = True

    def __post_init__(self) -> None:
        require_finite(strike=self.strike)
        if self.strike <= 0.0:
            raise InputError("strike must be > 0")

    def pv(self, market: MarketSnapshot) -> Money:
        t = market.time_to(self.expiry)
        if not t > 0.0:
            if t < 0.0:
                return Money(0.0, self.ccy)
            s = market.quote(f"EQ/{self.name}")
            return Money(max(s - self.strike, 0.0) if self.call else max(self.strike - s, 0.0), self.ccy)
        s = market.quote(f"EQ/{self.name}")
        r = market.curve(self.ccy).zero_rate(t)
        q = market.quotes.get(f"DIV/{self.name}", 0.0)
        sigma = market.surface(f"EQ/{self.name}").vol(t, self.strike)
        return Money(carry_price(s, self.strike, t, r, q, sigma, 1.0 if self.call else -1.0, "EquityOption"),
                     self.ccy)

    def cashflows(self) -> list[Cashflow]:
        return []

    def risk(self, market: MarketSnapshot) -> dict[str, float]:
        return bump_risk(self.pv, market, quotes=[f"EQ/{self.name}"], curves=[self.ccy],
                         surfaces=[f"EQ/{self.name}"])

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ccy": self.ccy, "strike": self.strike,
                "expiry": maturity_to_json(self.expiry), "call": self.call}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EquityOption:
        return cls(d["name"], d["ccy"], d["strike"], maturity_from_json(d["expiry"]), d["call"])


