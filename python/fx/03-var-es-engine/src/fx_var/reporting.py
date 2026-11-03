"""FX VaR in a declared reporting currency, and multi-currency book risk (W14b, W14d).

The plain idea
--------------
"The book's VaR is 1.2 million" means nothing until you say in which
currency. A USD-based firm and a EUR-based subsidiary holding the same
positions see different P&L, because the exchange rate between their two
reporting currencies is itself one of the risk factors. So every function
here takes ``reporting_ccy`` as a required keyword, and states how a P&L
measured in one currency is turned into another.

There are two honest conversions:

* ``"scenario"``: value the book in the reporting currency before and after
  each scenario, converting at that scenario's own exchange rate. This is the
  loss a desk keeping its books in that currency actually suffers. A
  pure cash balance in the reporting currency has zero risk.
* ``"spot"``: measure P&L in USD and translate it at today's rate. This is
  what a consolidation does; the result scales linearly, so a VaR in USD
  converts to EUR by one multiplication.

The two differ by the cross term between the book's P&L and the move of the
reporting currency against USD. Neither is "the right one"; the report says
which one it is.

The formal statement
--------------------
With ``V(s)`` the book value in USD under scenario ``s``, ``X_c(s)`` the USD
price of one unit of currency ``c`` (``X_c(0)`` today)::

    scenario:  PnL_c(s) = V(s) / X_c(s) - V(0) / X_c(0)
    spot:      PnL_c(s) = (V(s) - V(0)) / X_c(0)

VaR and ES at tail probability ``alpha`` are the empirical quantile and tail
mean of ``-PnL_c`` (``fx_var.engine.empirical_var_es``). Under ``"spot"``,
``VaR_c = VaR_USD X_USD(0)/X_c(0)`` exactly, because the quantile commutes
with multiplication by a positive constant. Under ``"scenario"`` the
identity is instead "convert every scenario at its own rate, then take the
quantile", which :func:`var_report` does and the tests check by hand.

Crosses are triangulated through USD (``Market.cross``), so an EURJPY
position is long EUR and short JPY against USD and its risk lands on the two
currencies. :func:`triangulation_residuals` compares quoted crosses with the
triangulated ones.

Multi-currency book risk (:func:`fx_book_risk`) reports, in the reporting
currency, per currency: delta (P&L per unit log move of the currency's USD
price, i.e. the currency exposure), gamma (second derivative in the same
units); per currency and tenor bucket: rate sensitivity (P&L per 1bp); per
pair and tenor bucket: vega (P&L per vol point). A position at expiry ``T``
between bucket tenors ``T_a < T < T_b`` is split linearly between them
(weights sum to 1), so bucket totals sum to the book's total sensitivity.
Clark (2011) ch. 2-3 for FX conventions; BCBS MAR21.14 for FX delta by
currency and MAR21.8 for tenor buckets.

Worked example
--------------
A USD book that holds EUR 1m cash, at EURUSD 1.10, has USD delta to
``FX:EUR`` of 1.1m. In EUR (``"scenario"``) the same book has zero risk:
it is cash in the reporting currency. In EUR (``"spot"``) the USD P&L is
translated at 1/1.10, so the EUR VaR is the USD VaR divided by 1.10, not
zero: translation keeps the USD view of the risk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .book import Book, Market, Option, ShockLike
from .common import fx_factor, ir_factor, resolve_alpha, split_pair, vol_factor
from .engine import empirical_var_es

__all__ = [
    "DEFAULT_TENOR_BUCKETS",
    "VaRReport",
    "convert_report",
    "fx_book_risk",
    "reporting_pnl",
    "tenor_weights",
    "triangulation_residuals",
    "var_report",
]

Conversion = Literal["scenario", "spot"]
DEFAULT_TENOR_BUCKETS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0)


def _ccy(code: str) -> str:
    c = code.upper()
    if len(c) != 3 or not c.isalpha():
        raise ValueError(f"reporting_ccy must be a 3-letter currency code, got {code!r}")
    return c


def _check_conversion(conversion: str) -> Conversion:
    if conversion not in ("scenario", "spot"):
        raise ValueError(f"conversion must be 'scenario' or 'spot', got {conversion!r}")
    return conversion  # type: ignore[return-value]


def _fx_level(market: Market, ccy: str, shocks: Mapping[str, object] | None) -> float | np.ndarray:
    if ccy == "USD":
        return 1.0
    x0 = market.spot(ccy)
    if shocks is None:
        return x0
    move = np.asarray(shocks.get(fx_factor(ccy), 0.0), dtype=float)
    return x0 * np.exp(move)


def reporting_pnl(
    book: Book,
    market: Market,
    scenarios: ShockLike,
    *,
    reporting_ccy: str,
    conversion: Conversion = "scenario",
    option_method: str = "full",
) -> np.ndarray | float:
    """Scenario P&L of ``book`` in ``reporting_ccy`` (see module docstring).

    ``book.base`` is ignored: the reporting currency is the argument, so the
    same ``Book`` can be reported in any currency.
    """
    rep = _ccy(reporting_ccy)
    conv = _check_conversion(conversion)
    sh = Book._normalise_shocks(scenarios)
    v0 = book.value_usd(market, None)
    v1 = book.value_usd(market, sh, option_method=option_method)
    x0 = _fx_level(market, rep, None)
    if conv == "spot":
        out = (np.asarray(v1) - v0) / x0
    else:
        out = np.asarray(v1) / _fx_level(market, rep, sh) - v0 / x0
    return float(out) if np.ndim(out) == 0 else np.asarray(out, dtype=float)


@dataclass(frozen=True)
class VaRReport:
    """VaR and ES with the currency and conversion they are stated in."""

    var: float
    es: float
    alpha: float
    reporting_ccy: str
    conversion: Conversion
    n_scenarios: int


def var_report(
    book: Book,
    market: Market,
    scenarios: pd.DataFrame,
    *,
    reporting_ccy: str,
    alpha: float | None = None,
    conversion: Conversion = "scenario",
    option_method: str = "full",
) -> VaRReport:
    """Historical/MC VaR and ES of ``book`` over scenario rows, in ``reporting_ccy``."""
    if not isinstance(scenarios, pd.DataFrame) or scenarios.empty:
        raise ValueError("scenarios must be a non-empty DataFrame of factor shocks")
    a = resolve_alpha(alpha)
    pnl = np.atleast_1d(
        reporting_pnl(book, market, scenarios, reporting_ccy=reporting_ccy,
                      conversion=conversion, option_method=option_method)
    )
    var, es = empirical_var_es(pnl, a)
    return VaRReport(float(var), float(es), a, _ccy(reporting_ccy), conversion, int(pnl.size))


def convert_report(report: VaRReport, market: Market, to_ccy: str) -> VaRReport:
    """Translate a ``"spot"`` report into another currency at today's rate.

    A ``"scenario"`` report cannot be converted by a rate: the reporting
    currency's own move is inside every scenario. Recompute it with
    :func:`var_report` instead; this function raises ``ValueError``.
    """
    if report.conversion != "spot":
        raise ValueError(
            "a 'scenario' VaR is not a linear function of the exchange rate; "
            "recompute it with var_report(..., reporting_ccy=...)"
        )
    target = _ccy(to_ccy)
    factor = float(_fx_level(market, report.reporting_ccy, None)) / float(
        _fx_level(market, target, None)
    )
    return VaRReport(report.var * factor, report.es * factor, report.alpha, target,
                     "spot", report.n_scenarios)


def triangulation_residuals(market: Market, quoted: Mapping[str, float]) -> pd.Series:
    """Quoted cross minus USD-triangulated cross, relative (``quoted/tri - 1``)."""
    out = {}
    for pair, q in quoted.items():
        if not np.isfinite(q) or q <= 0:
            raise ValueError(f"quoted rate for {pair} must be positive and finite")
        out[pair.upper()] = q / market.cross(pair) - 1.0
    return pd.Series(out, dtype=float)


def tenor_weights(expiry: float, buckets: Sequence[float]) -> np.ndarray:
    """Linear split of ``expiry`` over bucket tenors (flat outside; sums to 1)."""
    b = np.asarray(buckets, dtype=float)
    if b.ndim != 1 or b.size == 0 or np.any(np.diff(b) <= 0):
        raise ValueError("tenor buckets must be a non-empty increasing sequence")
    w = np.zeros(b.size)
    if expiry <= b[0]:
        w[0] = 1.0
    elif expiry >= b[-1]:
        w[-1] = 1.0
    else:
        i = int(np.searchsorted(b, expiry, side="right")) - 1
        t = (expiry - b[i]) / (b[i + 1] - b[i])
        w[i], w[i + 1] = 1.0 - t, t
    return w


def _central(book: Book, market: Market, rep: str, conv: Conversion,
             factor: str, h: float) -> tuple[float, float]:
    """First and second central differences of reporting P&L in ``factor``."""
    up = float(reporting_pnl(book, market, {factor: h}, reporting_ccy=rep, conversion=conv))
    dn = float(reporting_pnl(book, market, {factor: -h}, reporting_ccy=rep, conversion=conv))
    return (up - dn) / (2 * h), (up + dn) / (h * h)  # pnl(0) = 0


def fx_book_risk(
    book: Book,
    market: Market,
    *,
    reporting_ccy: str,
    tenor_buckets: Sequence[float] = DEFAULT_TENOR_BUCKETS,
    conversion: Conversion = "scenario",
    fx_bump: float = 1e-4,
    rate_bump: float = 1e-5,
    vol_bump: float = 1e-5,
) -> dict[str, pd.DataFrame]:
    """Delta and gamma by currency, rate and vega by tenor bucket, in ``reporting_ccy``.

    Returns ``{"fx": DataFrame[ccy -> delta, gamma], "rates": DataFrame[ccy x
    bucket -> P&L per 1bp], "vega": DataFrame[pair x bucket -> P&L per vol
    point]}``. Delta is per unit log move of the currency's USD price
    (a currency exposure); a 1% move is ``0.01 x delta``. The reporting
    currency has no row under ``"scenario"`` conversion only if it is USD;
    otherwise its row carries the risk of everything else against it.
    """
    rep = _ccy(reporting_ccy)
    conv = _check_conversion(conversion)
    buckets = tuple(float(b) for b in tenor_buckets)
    fx_rows = {}
    for c in sorted(book.currencies() - {"USD"}):
        d, g = _central(book, market, rep, conv, fx_factor(c), fx_bump)
        fx_rows[c] = {"delta": d, "gamma": g}
    fx = pd.DataFrame.from_dict(fx_rows, orient="index", columns=["delta", "gamma"])

    rate_ccys = sorted({c for p in book.positions for c in p.rate_currencies()})
    pairs = sorted({q for p in book.positions for q in p.vol_pairs()})
    rates = pd.DataFrame(0.0, index=rate_ccys, columns=list(buckets))
    vega = pd.DataFrame(0.0, index=pairs, columns=list(buckets))
    for pos in book.positions:
        expiry = getattr(pos, "expiry", None)
        if expiry is None:
            continue
        w = tenor_weights(float(expiry), buckets)
        single = Book([pos], base=book.base)
        for c in pos.rate_currencies():
            d, _ = _central(single, market, rep, conv, ir_factor(c), rate_bump)
            rates.loc[c] += w * d * 1e-4
        if isinstance(pos, Option):
            pair = "".join(split_pair(pos.pair))
            d, _ = _central(single, market, rep, conv, vol_factor(pos.pair), vol_bump)
            vega.loc[pair] += w * d * 1e-2
    return {"fx": fx, "rates": rates, "vega": vega}

