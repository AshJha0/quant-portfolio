"""Shared conventions, factor naming, validation and diagnostic warnings.

Conventions (see CONVENTIONS.md at portfolio root)
--------------------------------------------------
* FX pairs are quoted BASE/QUOTE: ``EURUSD`` = USD per 1 EUR.
* Internally every currency is mapped to a single **USD factor**:
  ``FX:CCY`` is the daily *log return* of the USD price of 1 unit of CCY
  (i.e. the log return of CCYUSD).  USD itself has no FX factor - its USD
  price is identically 1.  Cross pairs (EURJPY, ...) are *triangulated*
  through the two USD legs; there is no separate cross factor, which makes
  the factor set arbitrage-consistent by construction.
* ``IR:CCY`` is an *absolute* shock (decimal per annum) to the continuously
  compounded, ACT/365 zero rate of CCY (flat curve per currency).
* ``VOL:PAIR`` is an *absolute* shock (decimal, annualised) to the lognormal
  implied volatility of PAIR.

Rates are continuously compounded, annualised, ACT/365F.  Vols are quoted
on log-returns, annualised.  Time is in years; horizons in trading days
with ``TRADING_DAYS_PER_YEAR = 252``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252

#: Daily log-return standard deviation below which an FX factor is treated as
#: "peg-like".  0.0005/day is about 0.8% annualised - an order of magnitude
#: below any free float; HKD inside its band realises roughly this level.
PEG_VOL_THRESHOLD: float = 5e-4


class PegBlindnessWarning(UserWarning):
    """A risk factor has near-zero realised volatility (managed/pegged ccy).

    Historical simulation and variance-covariance methods will report a VaR
    of essentially zero for such a factor, even though the true risk is a
    rare, large revaluation jump (CHF 2015, classic EM peg breaks).  The
    engine flags this so the desk adds the peg-break stress add-on from
    :mod:`fx_var.stress_testing`.
    """


class NumericalWarning(UserWarning):
    """A numerical fallback was applied (e.g. Cholesky jitter on a singular
    correlation matrix)."""


def split_pair(pair: str) -> tuple[str, str]:
    """Split a 6-letter FX pair into (base, quote) currencies.

    Parameters
    ----------
    pair : str
        e.g. ``"EURUSD"`` (USD per 1 EUR -> base=EUR, quote=USD).

    Returns
    -------
    tuple of str
        ``(base_ccy, quote_ccy)``.

    Raises
    ------
    ValueError
        If ``pair`` is not 6 alphabetic characters or base == quote.
    """
    if not isinstance(pair, str) or len(pair) != 6 or not pair.isalpha():
        raise ValueError(f"FX pair must be 6 letters like 'EURUSD', got {pair!r}")
    base, quote = pair[:3].upper(), pair[3:].upper()
    if base == quote:
        raise ValueError(f"FX pair has identical legs: {pair!r}")
    return base, quote


def fx_factor(ccy: str) -> str:
    """Factor name for the log return of CCYUSD (USD price of 1 CCY)."""
    if ccy.upper() == "USD":
        raise ValueError("USD has no FX factor: its USD price is identically 1")
    return f"FX:{ccy.upper()}"


def ir_factor(ccy: str) -> str:
    """Factor name for an absolute shock to CCY's cc zero rate (ACT/365)."""
    return f"IR:{ccy.upper()}"


def vol_factor(pair: str) -> str:
    """Factor name for an absolute shock to PAIR's annualised implied vol."""
    base, quote = split_pair(pair)
    return f"VOL:{base}{quote}"


def validate_alpha(alpha: float) -> float:
    """Validate a VaR/ES confidence level, returning it unchanged.

    Raises
    ------
    ValueError
        If ``alpha`` is not strictly inside (0, 1).
    """
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha


def validate_horizon(horizon_days: float) -> float:
    """Validate a VaR horizon in trading days (must be > 0)."""
    horizon_days = float(horizon_days)
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be > 0, got {horizon_days}")
    return horizon_days


def validate_returns(
    returns: pd.DataFrame,
    required_factors: list[str] | None = None,
    min_obs: int = 60,
) -> pd.DataFrame:
    """Validate a factor-return history for use by any VaR method.

    NaN policy: the engine refuses NaNs outright (``ValueError``) rather
    than silently dropping or filling.  Missing FX fixings must be handled
    upstream (holiday calendars differ across time zones - see
    docs/DESK_GUIDE.md).

    Parameters
    ----------
    returns : pandas.DataFrame
        Rows = days, columns = factor names (``FX:*`` log returns,
        ``IR:*`` / ``VOL:*`` absolute daily changes).
    required_factors : list of str, optional
        Factors the book needs; a missing column raises ``ValueError``.
    min_obs : int
        Minimum history length; fewer rows raise ``ValueError``.

    Returns
    -------
    pandas.DataFrame
        The validated frame, restricted to ``required_factors`` if given.
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a pandas DataFrame of factor returns")
    if len(returns) < min_obs:
        raise ValueError(
            f"insufficient history: {len(returns)} rows < min_obs={min_obs}; "
            "VaR quantiles are meaningless on this sample"
        )
    if required_factors is not None:
        missing = [f for f in required_factors if f not in returns.columns]
        if missing:
            raise ValueError(
                f"returns is missing required factor columns: {missing}"
            )
        returns = returns[list(required_factors)]
    if returns.isna().to_numpy().any():
        bad = returns.columns[returns.isna().any()].tolist()
        raise ValueError(
            f"returns contains NaNs in columns {bad}; clean or drop them "
            "explicitly before calling the engine (NaN policy: refuse, "
            "never impute silently)"
        )
    return returns


def warn_peg_factors(returns: pd.DataFrame, factors: list[str]) -> list[str]:
    """Emit :class:`PegBlindnessWarning` for near-zero-vol FX factors.

    Returns the list of flagged factor names (empty if none).  Only ``FX:*``
    factors are screened - rate and vol factors are legitimately quiet.
    """
    import warnings as _w

    flagged: list[str] = []
    for f in factors:
        if not f.startswith("FX:") or f not in returns.columns:
            continue
        if float(returns[f].std(ddof=1)) < PEG_VOL_THRESHOLD:
            flagged.append(f)
    if flagged:
        _w.warn(
            f"factors {flagged} have daily vol < {PEG_VOL_THRESHOLD:.4%} "
            "(pegged/managed currency). Historical and parametric VaR are "
            "blind to peg-break risk; add the peg-break stress add-on "
            "(fx_var.stress_testing.peg_break_scenario).",
            PegBlindnessWarning,
            stacklevel=3,
        )
    return flagged
