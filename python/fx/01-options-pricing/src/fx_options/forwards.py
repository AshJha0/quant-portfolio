"""FX forwards via covered interest parity (CIP).

Covered interest parity:  ``F = S * exp((r_d - r_f) * (T - t_s))``.

A quoted spot rate is for delivery on the spot date ``t_s`` (T+2, or T+1
for USDCAD, USDTRY, USDRUB and USDPHP), so the carry runs from spot to
delivery (Clark 2011, ch. 1; Wystup 2017, ch. 1). ``t_spot`` defaults to 0,
the old reading in which spot is a price for today; that overstates a 1y
EURUSD forward by about 2.4 pips at a 2% rate differential. Callers with
dates should use :func:`forward_from_dates`, which finds the spot date on
the rule-based calendars of :mod:`qp_core.holidays`.

A domestic investor can replicate the forward by borrowing domestic cash,
buying spot foreign currency and depositing it at ``r_f``; absence of
arbitrage forces the forward to the CIP level (abstracting from the
cross-currency basis: see docs/METHODOLOGY.md assumption register).

Forward points are quoted as ``(F - S)`` scaled by the pair's pip factor
(1e4 for most pairs, 1e2 for JPY-quoted pairs).
"""

from __future__ import annotations

import datetime as dt
import math

from qp_core.holidays import fx_spot_date

from ._common import finite_result, safe_exp, validate_inputs

__all__ = [
    "PIP_FACTORS",
    "cip_forward",
    "forward_from_dates",
    "forward_points",
    "synthetic_forward_from_options",
]

#: Standard pip scaling: price decimal places quoted by convention.
PIP_FACTORS = {"default": 1e4, "jpy": 1e2}


def _check_t_spot(t_spot: float) -> float:
    if not math.isfinite(t_spot) or t_spot < 0.0:
        raise ValueError(f"t_spot must be finite and >= 0, got {t_spot!r}")
    return float(t_spot)


def cip_forward(S: float, T: float, r_d: float, r_f: float, *,
                t_spot: float = 0.0) -> float:
    """Covered-interest-parity forward rate.

    Parameters
    ----------
    S : float
        Spot rate, domestic per unit foreign, > 0.
    T : float
        Time to delivery in years, >= 0.
    r_d, r_f : float
        Domestic / foreign continuously compounded rates (ACT/365F).
    t_spot : float, keyword-only
        Time from today to the spot date in years (e.g. 4/365 for a
        Thursday EURUSD trade). Default 0 keeps the t = 0 reading.

    Returns
    -------
    float
        Forward rate ``F = S * exp((r_d - r_f) (T - t_spot))``.  ``F > S`` when the
        domestic rate exceeds the foreign rate (forward premium on the
        base currency), the classic carry relationship.
    """
    validate_inputs(S, S, T, r_d, r_f, 0.0)
    ts = _check_t_spot(t_spot)
    return finite_result(S * safe_exp((r_d - r_f) * (T - ts), "cip_forward"),
                         "cip_forward")


def forward_from_dates(S: float, trade_date: dt.date, value_date: dt.date,
                       r_d: float, r_f: float, pair: str) -> float:
    """CIP forward for a value date, with the carry run from the spot date.

    ``t_s`` and ``T`` are ACT/365F year fractions from ``trade_date`` to the
    pair's spot date (:func:`qp_core.holidays.fx_spot_date`) and to
    ``value_date``. Works for broken dates and for value dates before spot.

    Example: EURUSD traded Thu 5 Jun 2025 settles spot Mon 9 Jun; the
    forward to 9 Jun 2026 carries for 365 days, not 369.
    """
    if value_date < trade_date:
        raise ValueError(f"value date {value_date} precedes trade date {trade_date}")
    ts = (fx_spot_date(trade_date, pair) - trade_date).days / 365.0
    T = (value_date - trade_date).days / 365.0
    return cip_forward(S, T, r_d, r_f, t_spot=ts)


def forward_points(S: float, T: float, r_d: float, r_f: float,
                   pip_factor: float = 1e4, *, t_spot: float = 0.0) -> float:
    """Forward points ``(F - S) * pip_factor``.

    Parameters
    ----------
    S, T, r_d, r_f : float
        As in :func:`cip_forward`.
    pip_factor : float
        1e4 for e.g. EURUSD (pip = 0.0001), 1e2 for USDJPY (pip = 0.01).
    t_spot : float, keyword-only
        As in :func:`cip_forward`.

    Returns
    -------
    float
        Forward points in pips; positive when the base currency trades at
        a forward premium.
    """
    if not math.isfinite(pip_factor) or pip_factor <= 0:
        raise ValueError(
            f"pip_factor must be positive and finite, got {pip_factor!r}"
        )
    return finite_result((cip_forward(S, T, r_d, r_f, t_spot=t_spot) - S) * pip_factor,
                         "forward points")


def synthetic_forward_from_options(call_price: float, put_price: float,
                                   K: float, T: float, r_d: float) -> float:
    """Forward implied by put-call parity (a 'synthetic forward').

    Conversion/parity: ``C - P = e^{-r_d T} (F - K)``, hence
    ``F = K + (C - P) e^{r_d T}``.  Long call + short put at the same
    strike replicates a forward purchase of the base currency; desks call
    the position a *synthetic forward* (or a 'conversion' when run against
    an actual forward to lock in the mispricing).

    Parameters
    ----------
    call_price, put_price : float
        European premiums (domestic ccy per unit foreign) at strike ``K``.
    K : float
        Common strike, > 0.
    T : float
        Time to expiry in years, >= 0.
    r_d : float
        Domestic continuously compounded rate.

    Returns
    -------
    float
        The option-implied forward rate.
    """
    validate_inputs(K, K, T, r_d, 0.0, 0.0)
    for name, p in (("call_price", call_price), ("put_price", put_price)):
        if not math.isfinite(p):
            raise ValueError(f"{name} must be finite, got {p!r}")
    return finite_result(
        K + (call_price - put_price) * safe_exp(r_d * T, "synthetic forward"),
        "synthetic forward")
