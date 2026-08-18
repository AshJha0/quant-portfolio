"""Carry: deposit-rate differentials, forward points, and daily roll accrual.

Why carry is first-class in FX pairs trading
--------------------------------------------
A spot FX position is financed: holding long BASE/QUOTE means owning the base
currency (earning its deposit rate) funded in the quote currency (paying its
deposit rate).  In practice the position is held via forwards/swaps and the
financing shows up as **forward points** rolled daily (tom-next).  Backtest
P&L must therefore be

``total = spot P&L + carry accrual - transaction costs``.

A mean-reversion signal computed on spot alone can be systematically
**wrong-carry**: what looks like a spot spread reverting to its mean may be
nothing more than the forward premium playing out, and a strategy that keeps
buying the "cheap" high-yield currency's spot dip is earning carry, while one
that keeps selling it is paying carry every day it holds.

Conventions
-----------
* Deposit rates are annualised simple rates, ACT/365F day count.
* Covered interest parity (units QUOTE per BASE):
  ``F(tau) = S * (1 + r_quote * tau) / (1 + r_base * tau)``.
  Swap points ``= F - S``.  A high-yield base currency trades at a forward
  **discount** (F < S): the long earns the points back as time passes.
* Daily roll yield of a long base position, as a fraction of notional:
  ``(S - F(tau)) / S = (r_base - r_quote) * tau / (1 + r_base * tau)``
  which is ``(r_base - r_quote) * tau`` to first order.  ``method="linear"``
  uses the first-order form, ``method="swap"`` the exact swap-point form.
* Day-count fractions come from **actual calendar-day gaps / 365**, so a
  Friday-to-Monday step accrues 3 days.  Real tom-next rolls apply the
  3-day weekend charge on **Wednesday** (spot T+2 settles Monday); we apply
  it on Monday instead — same total accrual, shifted two business days.
  This T+2 wrinkle (and the triple swap on Wednesdays) is documented in
  docs/METHODOLOGY.md and DESK_GUIDE.md.
* Accrual timing: the carry earned over ``(t-1, t]`` uses the rates known at
  ``t-1`` (no lookahead) and is booked at ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "day_count_fractions",
    "forward_outright",
    "swap_points",
    "daily_roll_yield",
    "carry_accrual",
    "carry_ledger",
    "carry_adjusted_log_price",
]


def day_count_fractions(index: pd.Index, basis: float = 365.0) -> np.ndarray:
    """ACT/basis year fractions between consecutive index dates.

    Element 0 is 0 (nothing accrues before the first observation).  A
    Friday-to-Monday step yields 3/365 (weekend accrual — see module notes on
    the real-world Wednesday T+2 wrinkle).  For a non-datetime index, each
    step is treated as one business day on a 252-step grid (1/252).

    Parameters
    ----------
    index : pandas.Index
        DatetimeIndex (preferred) or any index.
    basis : float
        Days per year for the ACT day count (365 = ACT/365F).
    """
    if isinstance(index, pd.DatetimeIndex):
        days = np.diff(index.values.astype("datetime64[D]").astype(np.int64))
        out = np.concatenate([[0.0], days / basis])
    else:
        out = np.concatenate([[0.0], np.full(len(index) - 1, 1.0 / 252.0)])
    return out


def forward_outright(
    spot: float | np.ndarray, r_base: float | np.ndarray,
    r_quote: float | np.ndarray, tau: float | np.ndarray,
) -> float | np.ndarray:
    """Covered-interest-parity forward ``F = S (1 + r_q tau) / (1 + r_b tau)``.

    Units: QUOTE per BASE, same as spot.  Rates annualised simple, ACT/365F.
    """
    return spot * (1.0 + r_quote * tau) / (1.0 + r_base * tau)


def swap_points(
    spot: float | np.ndarray, r_base: float | np.ndarray,
    r_quote: float | np.ndarray, tau: float | np.ndarray,
) -> float | np.ndarray:
    """Forward points ``F - S``.  Negative for a high-yield base currency."""
    return forward_outright(spot, r_base, r_quote, tau) - spot


def daily_roll_yield(
    r_base: float | np.ndarray, r_quote: float | np.ndarray,
    tau: float | np.ndarray, method: str = "swap",
) -> float | np.ndarray:
    """Roll yield of a long-base position over ``tau`` years, per unit notional.

    ``method="swap"``: exact swap-point form
    ``(S - F)/S = (r_b - r_q) tau / (1 + r_b tau)`` (spot cancels).
    ``method="linear"``: first-order ``(r_b - r_q) tau``.

    A long position in a high-yield base currency has positive roll yield
    (it buys back the forward discount every day).
    """
    if method == "swap":
        return (r_base - r_quote) * tau / (1.0 + r_base * tau)
    if method == "linear":
        return (r_base - r_quote) * tau
    raise ValueError(f"method must be 'swap' or 'linear', got {method!r}")


def _as_rate_array(r: float | pd.Series | np.ndarray, n: int, name: str) -> np.ndarray:
    if np.isscalar(r):
        return np.full(n, float(r))
    arr = np.asarray(r, dtype=float)
    if len(arr) != n:
        raise ValueError(f"{name} has length {len(arr)}, expected {n}")
    return arr


def carry_accrual(
    r_base: float | pd.Series | np.ndarray,
    r_quote: float | pd.Series | np.ndarray,
    index: pd.Index,
    basis: float = 365.0,
    method: str = "swap",
) -> np.ndarray:
    """Per-period carry accrual for a long 1-unit base position, no lookahead.

    Element ``t`` (t >= 1) is the roll yield over ``(t-1, t]`` computed from
    the rates observed at ``t-1`` and the actual calendar-day gap.  Element 0
    is 0.

    Returns
    -------
    numpy.ndarray
        Fractional accrual per period (multiply by signed notional for P&L).
    """
    n = len(index)
    rb = _as_rate_array(r_base, n, "r_base")
    rq = _as_rate_array(r_quote, n, "r_quote")
    dt = day_count_fractions(index, basis)
    out = np.zeros(n)
    out[1:] = daily_roll_yield(rb[:-1], rq[:-1], dt[1:], method=method)
    return out


def carry_ledger(
    positions: pd.Series | np.ndarray,
    r_base: float | pd.Series | np.ndarray,
    r_quote: float | pd.Series | np.ndarray,
    index: pd.Index | None = None,
    notional: float = 1.0,
    basis: float = 365.0,
    method: str = "swap",
) -> pd.Series:
    """Daily carry P&L ledger for a single pair position.

    ``pnl_t = position_{t-1} * notional * accrual_t`` where the accrual over
    ``(t-1, t]`` uses rates at ``t-1`` (see :func:`carry_accrual`).  Long a
    high-yield base currency => positive carry.

    Parameters
    ----------
    positions : array-like
        Signed position (units of base notional) held from close ``t`` to
        close ``t+1``.
    index : pandas.Index, optional
        Dates; defaults to the index of ``positions`` when it is a Series.
    """
    if index is None:
        if not isinstance(positions, pd.Series):
            raise ValueError("index required when positions is not a Series")
        index = positions.index
    pos = np.asarray(positions, dtype=float)
    if len(pos) != len(index):
        raise ValueError("positions and index length mismatch")
    accr = carry_accrual(r_base, r_quote, index, basis=basis, method=method)
    pnl = np.zeros(len(pos))
    pnl[1:] = pos[:-1] * accr[1:] * notional
    return pd.Series(pnl, index=index, name="carry_pnl")


def carry_adjusted_log_price(
    price: pd.Series,
    r_base: float | pd.Series | np.ndarray,
    r_quote: float | pd.Series | np.ndarray,
    basis: float = 365.0,
    method: str = "swap",
) -> pd.Series:
    """Total-return log price: ``log S_t + cumulative carry accrual``.

    This is the log wealth of holding 1 unit of base currency financed in the
    quote currency, rolling daily.  Spreads built on carry-adjusted log
    prices measure *total-return* mean reversion; the difference between the
    spot spread and the carry-adjusted spread is exactly the cumulated rate
    differential — the carry drag/boost of the trade.

    Identity (tested): ``carry_adjusted - log(S)`` equals the cumulative sum
    of :func:`carry_accrual`.
    """
    if not isinstance(price, pd.Series):
        raise ValueError("price must be a pandas Series with a date index")
    accr = carry_accrual(r_base, r_quote, price.index, basis=basis, method=method)
    out = np.log(price.to_numpy(dtype=float)) + np.cumsum(accr)
    return pd.Series(out, index=price.index, name=f"{price.name}_tr")
