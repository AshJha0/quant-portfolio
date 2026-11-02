"""Euler VaR/ES attribution for an FX book: currency mapping, then Euler.

Plain idea
----------
:mod:`fx_var.risk_attribution` splits a risk number across exposures. For an
FX desk the first job is to say what the exposures are. Every position is
mapped to the currencies it is long or short, valued in the reporting
currency; the risk factors are the log returns of each currency against the
reporting currency (RiskMetrics 1996 Technical Document, 4th ed., s.6.2,
cash-flow mapping; Jorion 2007, *Value at Risk*, 3rd ed., s.11.2.1). Euler
then splits the book's VaR or ES across currencies exactly, and because
Euler components add up, any grouping (by pair, desk, region) is the sum of
its members.

Options need one FX convention decision. The first-order exposure of an
option to ``ln S`` is ``N S delta`` in the quote currency. When the book
reports in the pair's *base* currency (a USD book holding USDJPY options),
the value in base units is ``V/S``, and ``d(V/S)/d ln S = delta - V/S``: the
premium-adjusted spot delta (Reiswich and Wystup 2010, "A guide to FX
options quoting conventions", J. Derivatives 18(2), s.2.2; Clark 2011,
*Foreign Exchange Option Pricing*, s.3.2). So the exposure uses the
unadjusted delta for quote-currency reporting and the premium-adjusted
delta for base-currency reporting. This is the same rule that makes USDJPY
quote pa deltas.

Formal statement
----------------
Currency ``c`` held in amount ``a_c`` (signed), with ``X_c`` reporting units
per unit of ``c``. Exposure ``e_c = a_c X_c``; P&L ``~ sum_c e_c r_c`` with
``r_c = d ln X_c``. Then ``parametric_attribution(e, Sigma, alpha)`` gives
``total = sum_c e_c dR/de_c`` (Tasche 2000, s.4).

Option on BASE/QUOTE, notional ``N`` base units, GK value ``V`` (quote per
base), spot delta ``Delta``:

    reporting = quote:  e = N S Delta
    reporting = base :  e = N (Delta - V/S)

References: see :mod:`fx_var.risk_attribution` for the Euler results.
Worked example: ``tests/test_fx_attribution.py::test_eur_jpy_book_by_currency``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from . import _r3checks as chk
from .gk import gk_delta, gk_price
from .risk_attribution import Attribution, Measure, parametric_attribution

__all__ = [
    "FXAttribution",
    "currency_exposures",
    "fx_option_exposure",
    "fx_parametric_attribution",
    "group_components",
]


def currency_exposures(
    holdings: Mapping[str, float],
    rates_to_reporting: Mapping[str, float],
    currencies: Sequence[str],
) -> np.ndarray:
    """Reporting-currency exposure ``a_c X_c`` per currency, in ``currencies`` order.

    ``holdings`` maps currency to signed amount; a currency missing from
    ``holdings`` has zero exposure. The reporting currency itself carries no
    FX risk and must not appear in ``currencies``.
    """
    out = np.zeros(len(currencies))
    unknown = set(holdings) - set(currencies)
    if unknown:
        raise ValueError(f"holdings in currencies with no risk factor: {sorted(unknown)}")
    for i, c in enumerate(currencies):
        if c not in rates_to_reporting:
            raise ValueError(f"no rate to the reporting currency for {c}")
        x = chk.scalar(f"rate[{c}]", rates_to_reporting[c], lo=0.0, strict=True)
        a = chk.scalar(f"holding[{c}]", holdings.get(c, 0.0))
        out[i] = a * x
    return out


def fx_option_exposure(
    notional: float,
    spot: float,
    strike: float,
    expiry: float,
    r_d: float,
    r_f: float,
    vol: float,
    kind: str = "call",
    reporting: str = "quote",
) -> float:
    """First-order exposure of an FX option to ``d ln S``, in the reporting currency.

    ``notional`` is in base-currency units (negative for a short).
    ``reporting="quote"`` gives ``N S Delta`` (unadjusted spot delta);
    ``reporting="base"`` gives ``N (Delta - V/S)`` (premium-adjusted).
    """
    n = chk.scalar("notional", notional)
    s = chk.scalar("spot", spot, lo=0.0, strict=True)
    if reporting not in ("quote", "base"):
        raise ValueError(f"reporting must be 'quote' or 'base', got {reporting!r}")
    delta = float(gk_delta(s, strike, expiry, r_d, r_f, vol, kind))
    if reporting == "quote":
        return n * s * delta
    v = float(gk_price(s, strike, expiry, r_d, r_f, vol, kind))
    return n * (delta - v / s)


@dataclass(frozen=True)
class FXAttribution:
    """Euler attribution keyed by currency (or any factor label)."""

    labels: tuple[str, ...]
    exposures: np.ndarray
    result: Attribution

    @property
    def total(self) -> float:
        return self.result.total

    def component(self) -> dict[str, float]:
        return dict(zip(self.labels, self.result.component.tolist(), strict=True))

    def marginal(self) -> dict[str, float]:
        return dict(zip(self.labels, self.result.marginal.tolist(), strict=True))


def fx_parametric_attribution(
    exposures: Mapping[str, float],
    cov: np.ndarray,
    labels: Sequence[str],
    alpha: float = 0.01,
    *,
    measure: Measure = "var",
    mean: np.ndarray | None = None,
) -> FXAttribution:
    """Parametric Euler VaR/ES of a currency-mapped book.

    ``exposures`` maps label to reporting-currency exposure (sum several
    positions' exposures to the same currency before calling); ``cov`` is
    the covariance of the currencies' log returns against the reporting
    currency over the risk horizon, rows in ``labels`` order.
    """
    unknown = set(exposures) - set(labels)
    if unknown:
        raise ValueError(f"exposures for unknown labels: {sorted(unknown)}")
    e = np.array([chk.scalar(f"exposure[{c}]", exposures.get(c, 0.0)) for c in labels])
    res = parametric_attribution(e, cov, alpha, measure=measure, mean=mean)
    return FXAttribution(tuple(labels), e, res)


def group_components(attr: FXAttribution, groups: Mapping[str, Sequence[str]]) -> dict[str, float]:
    """Sum Euler components into groups; the groups must partition the labels.

    Valid because Euler components are additive (Tasche 2000, s.4): the risk
    contribution of a sub-portfolio is the sum of its members'.
    """
    seen: list[str] = [c for members in groups.values() for c in members]
    if sorted(seen) != sorted(attr.labels):
        raise ValueError("groups must use every label exactly once")
    comp = attr.component()
    return {g: float(sum(comp[c] for c in members)) for g, members in groups.items()}
