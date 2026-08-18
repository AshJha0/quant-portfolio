"""Seeded synthetic FX data generators (no network access, deterministic).

Two layers:

* **ticks-lite** intraday generator (:func:`generate_ticks`) — a regular
  sub-hourly grid of (time, bid, mid, ask) with session-dependent
  volatility and spread and a *planted* hourly momentum alpha
  (AR(1) autocorrelation ``phi`` in hourly log-mid increments).  The bar
  builder and signal layer consume this.
* **daily panel** (:func:`generate_daily_panel`) — per-day spot and
  base/quote deposit rates for the carry filter and overnight accrual.

All prices follow the CONVENTIONS.md FX quoting rules: BASE/QUOTE,
pips of the quote currency, ``pip_size`` price units per pip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..sessions import PairProfile, EURUSD

__all__ = ["generate_ticks", "generate_daily_panel"]


def _as_rng(seed: int | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def generate_ticks(
    n_days: int = 10,
    ticks_per_hour: int = 12,
    phi: float = 0.25,
    profile: PairProfile = EURUSD,
    seed: int | np.random.Generator = 0,
) -> pd.DataFrame:
    """Generate a seeded ticks-lite intraday series with planted momentum.

    Hourly mid increments follow an AR(1) in pips,

    .. math:: r_h = \\phi\\, r_{h-1} + \\sqrt{1-\\phi^2}\\,\\sigma_h\\,z_h,

    so the 1-hour momentum feature has population IC ``phi`` against the
    next hourly return (the *planted alpha*).  Within each hour, ticks
    interpolate the hourly endpoint via a Brownian bridge, so bars built
    from ticks recover the planted hourly dynamics exactly at hour
    boundaries.  Session vol and spread come from ``profile``.

    Parameters
    ----------
    n_days : int
        Number of 24h days.
    ticks_per_hour : int
        Ticks per hour (regular spacing; the last tick of hour ``h`` is at
        the hour boundary ``h+1`` minus one tick interval).
    phi : float
        AR(1) coefficient of hourly increments; 0 gives pure noise
        (no alpha), must satisfy ``|phi| < 1``.
    profile : PairProfile
        Session spread/vol profile.
    seed : int or numpy.random.Generator
        Deterministic seed.

    Returns
    -------
    pandas.DataFrame
        Columns ``time_hours`` (absolute hours from day-0 London
        midnight), ``mid``, ``bid``, ``ask``; one row per tick.

    Raises
    ------
    ValueError
        If ``abs(phi) >= 1`` or ``n_days < 1``.
    """
    if abs(phi) >= 1.0:
        raise ValueError(f"|phi| must be < 1, got {phi}")
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")
    rng = _as_rng(seed)
    pip = profile.pip_size

    n_hours = 24 * n_days
    hour_starts = np.arange(n_hours, dtype=float)
    sigma_h = profile.vol_at(hour_starts) * np.sqrt(60.0)  # pips per hour-step

    z = rng.standard_normal(n_hours)
    r = np.empty(n_hours)
    innov = np.sqrt(1.0 - phi**2) * sigma_h * z
    r[0] = innov[0]
    for h in range(1, n_hours):
        r[h] = phi * r[h - 1] + innov[h]

    hourly_mid = profile.s0 + pip * np.concatenate([[0.0], np.cumsum(r)])

    m = int(ticks_per_hour)
    frac = np.arange(m) / m  # tick offsets within the hour
    times = (hour_starts[:, None] + frac[None, :]).ravel()

    # Brownian bridge from hourly_mid[h] (first tick) to hourly_mid[h+1]
    # (last tick of the hour), so the bar close equals the planted hourly
    # endpoint exactly and bar returns carry the planted AR(1) undiluted.
    bridge_z = rng.standard_normal((n_hours, m - 1)) if m > 1 else np.zeros((n_hours, 0))
    mids = np.empty((n_hours, m))
    for h in range(n_hours):
        a, b = hourly_mid[h], hourly_mid[h + 1]
        mids[h, 0] = a if m > 1 else b
        w = 0.0
        for j in range(1, m):
            # standard bridge increment on the unit interval, pinned to b
            # at the last tick (j = m-1): tau = 1/(m-1) steps.
            tau = 1.0 / (m - 1)
            t_rem = 1.0 - (j - 1) * tau
            drift = (b - (a + pip * w)) * tau / t_rem
            volw = sigma_h[h] * np.sqrt(max(tau * (t_rem - tau) / t_rem, 0.0))
            w += drift / pip + volw * bridge_z[h, j - 1]
            mids[h, j] = a + pip * w
    mid = mids.ravel()

    half_spread = 0.5 * profile.spread_pips_at(times) * pip
    df = pd.DataFrame(
        {
            "time_hours": times,
            "mid": mid,
            "bid": mid - half_spread,
            "ask": mid + half_spread,
        }
    )
    return df


def generate_daily_panel(
    n_days: int = 10,
    r_base: float = 0.03,
    r_quote: float = 0.05,
    rate_vol: float = 0.001,
    profile: PairProfile = EURUSD,
    seed: int | np.random.Generator = 1,
) -> pd.DataFrame:
    """Generate a daily panel of spot fixes and deposit rates for carry.

    Rates are annualised, continuously compounded, ACT/365F.  The carry of
    a long BASE/QUOTE spot position rolled daily is approximately
    ``S * (r_base - r_quote) * dt`` quote-ccy units per unit of base.

    Parameters
    ----------
    n_days : int
        Number of days.
    r_base, r_quote : float
        Mean base- and quote-currency deposit rates (annualised).
    rate_vol : float
        Daily white noise added to each rate (annualised units).
    profile : PairProfile
        Pair (for the spot level).
    seed : int or numpy.random.Generator
        Deterministic seed.

    Returns
    -------
    pandas.DataFrame
        Index ``day`` (0..n_days-1); columns ``spot``, ``r_base``,
        ``r_quote``, ``carry`` (= r_base - r_quote).
    """
    rng = _as_rng(seed)
    days = np.arange(n_days)
    rb = r_base + rate_vol * rng.standard_normal(n_days)
    rq = r_quote + rate_vol * rng.standard_normal(n_days)
    spot = profile.s0 * np.exp(np.cumsum(0.002 * rng.standard_normal(n_days)))
    return pd.DataFrame(
        {"spot": spot, "r_base": rb, "r_quote": rq, "carry": rb - rq},
        index=pd.Index(days, name="day"),
    )
