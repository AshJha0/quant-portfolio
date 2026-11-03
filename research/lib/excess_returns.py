"""Currency excess returns via covered interest parity, with publication lags (W6).

Plain idea: holding a foreign currency for a day earns the spot move plus the
interest-rate gap between the two currencies. A study that uses spot moves
only (every FX study before Round 7) leaves out carry, which is the larger
part of the return for high-yield currencies. The rates must also be ones a
trader could have known on the day: a monthly average for March is not known
until March has ended and the series has been published.

Formal statement. With ``s_t`` the log price of one unit of foreign currency in
the base currency (base per foreign), foreign rate ``i*`` and base rate ``i``
(annualised, decimal) as known at ``t``, and the CIP forward
``f_t = s_t + i tau - i* tau*``, the log excess return of a long-foreign
position from ``t`` to ``t+1`` is

    rx_{t+1} = s_{t+1} - f_t = Delta s_{t+1} + i*_t tau*_t - i_t tau_t

with ``tau = calendar days / basis`` per currency (ACT/365 for GBP, JPY, AUD,
NZD, CAD, HKD, SGD, ZAR, INR, MYR and THB money markets; ACT/360 otherwise).
Spot, carry and total are returned separately (Lustig, Roussanov and
Verdelhan 2011, section I; Menkhoff, Sarno, Schmeling and Schrimpf 2012, JF,
section I.A).

Publication lags (``available_from``):

* ``monthly``: a monthly-average series for month m is used from the first
  business day of month m+2 (conservative for the OECD MEI release timing).
* ``daily``: an overnight fixing for day d is used from the next business day.
  The euro short-term rate (ECB), SONIA (Bank of England) and SOFR (New York
  Fed) are all published on the morning of the next business day.

Worked example (``test_excess_returns.py``): spot from 1.0000 to 1.0100 over
one calendar day, foreign rate 5%, base rate 1%, ACT/360 both sides:
``rx = log(1.01) + (0.05 - 0.01) / 360 = 0.00995033 + 0.00011111 = 0.01006144``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACT365 = frozenset({"GBP", "JPY", "AUD", "NZD", "CAD", "HKD", "SGD", "ZAR", "INR", "MYR", "THB"})


class LookAheadViolation(ValueError):
    """A rate was used before its publication date."""


def basis(ccy: str) -> float:
    """Money-market day-count denominator of ``ccy``."""
    return 365.0 if ccy.upper() in ACT365 else 360.0


def available_from(obs: pd.DatetimeIndex, kind: str) -> pd.DatetimeIndex:
    """First business date on which each observation is known."""
    obs = pd.DatetimeIndex(obs)
    if kind == "monthly":
        start = pd.DatetimeIndex(obs.to_period("M").to_timestamp()) + pd.offsets.MonthBegin(2)
        return pd.DatetimeIndex(start) + pd.offsets.BDay(0)
    if kind == "daily":
        return pd.DatetimeIndex(obs + pd.offsets.BDay(1))
    raise ValueError(f"kind must be 'monthly' or 'daily', got {kind!r}")


def as_known(rates: pd.Series, kind: str, dates: pd.DatetimeIndex) -> pd.Series:
    """The latest published value of ``rates`` on each of ``dates`` (NaN before the first)."""
    s = rates.dropna()
    known = pd.Series(s.to_numpy(dtype=float),
                      index=available_from(pd.DatetimeIndex(s.index), kind))
    known = known[~known.index.duplicated(keep="last")].sort_index()
    dates = pd.DatetimeIndex(dates)
    both = dates.union(pd.DatetimeIndex(known.index))
    return known.reindex(both).ffill().reindex(dates)


def assert_known(rates_used: pd.Series, rates: pd.Series, kind: str) -> None:
    """Raise :class:`LookAheadViolation` if any value of ``rates_used``
    (indexed by the date it was used) is not what had been published by then."""
    ref = as_known(rates, kind, pd.DatetimeIndex(rates_used.index))
    a, b = rates_used.to_numpy(dtype=float), ref.to_numpy(dtype=float)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    if not same.all():
        d = pd.Timestamp(rates_used.index[int(np.flatnonzero(~same)[0])])
        raise LookAheadViolation(f"rate used on {d.date()} was not yet published")


def check_series_release(series: str, obs_dates: list, used_on: list) -> None:
    """Look-ahead check through the shared release rules,
    ``qp_data.release.check_release(series, obs_dates, used_on)`` (lane L2,
    Round 7). That package holds the issuer release calendars; W6 studies use
    it for every named series and keep :func:`assert_known` only for the
    generic monthly/daily lag rules above."""
    from qp_data.release import check_release  # merged at integration

    check_release(series, obs_dates, used_on)


def excess_returns(spot: pd.Series, foreign_rate: pd.Series, base_rate: pd.Series,
                   foreign: str, base: str) -> pd.DataFrame:
    """Spot, carry and excess log return of long ``foreign`` against ``base``.

    ``spot`` is base per foreign on trading dates. The rates must already be
    lagged to their publication dates (:func:`as_known`). Row t is the return
    from row t-1 to row t; row 0 is NaN. ``days`` is the calendar-day accrual.
    """
    idx = pd.DatetimeIndex(spot.index)
    days = np.r_[np.nan, np.diff(idx.to_numpy().astype("datetime64[D]")).astype(float)]
    fr = foreign_rate.reindex(idx).to_numpy(dtype=float)
    br = base_rate.reindex(idx).to_numpy(dtype=float)
    carry = np.full(len(idx), np.nan)
    carry[1:] = fr[:-1] * days[1:] / basis(foreign) - br[:-1] * days[1:] / basis(base)
    ds = np.r_[np.nan, np.diff(np.log(spot.to_numpy(dtype=float)))]
    return pd.DataFrame({"spot": ds, "carry": carry, "excess": ds + carry, "days": days},
                        index=idx)


def invert_ecb(px: pd.DataFrame, base: str = "USD") -> pd.DataFrame:
    """ECB rates are units of currency per EUR. Return ``base`` per unit of
    each other currency, with EUR added: ``(base/EUR) / (ccy/EUR)``."""
    if base not in px.columns:
        raise ValueError(f"{base} not in the price file")
    out = px.rdiv(px[base], axis=0).drop(columns=[base])
    out["EUR"] = px[base]
    return out
