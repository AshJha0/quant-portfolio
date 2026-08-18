"""Seeded synthetic market data with planted, known alpha.

Daily panel: cross-section of stocks whose next-day return loads on

- **momentum** (12-1): positive coefficient on the z-scored trailing
  252-21 day log return -> planted IC ~ ``mom_strength / sigma_daily``
  (0.04 for the defaults), and
- **short-term reversal**: negative coefficient on the trailing 21-day
  return.

Idiosyncratic vol follows a per-stock AR(1) log-vol process (vol
clustering).  Because the alpha is *planted with known strength*, the
feature/IC pipeline can be validated against ground truth: the momentum
feature must show IC in the planted range with t-stat > 2, and a pure-noise
feature must not (tests).

Intraday data: see :class:`eq_algo.intraday.IntradayMarket` (re-exported
here for convenience) — U-shaped volume, temporary + permanent impact.
Everything takes an explicit seed; no network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..intraday import IntradayConfig, IntradayMarket, u_shaped_profile

__all__ = ["DailyPanel", "generate_daily_panel",
           "IntradayConfig", "IntradayMarket", "u_shaped_profile"]


@dataclass
class DailyPanel:
    """Synthetic daily panel: prices, volumes (shares), simple returns."""

    prices: pd.DataFrame
    volumes: pd.DataFrame
    returns: pd.DataFrame = field(repr=False)

    @property
    def adv_dollars(self) -> pd.Series:
        """Full-sample average daily dollar volume per name (planning number)."""
        return (self.prices * self.volumes).mean()


def generate_daily_panel(n_stocks: int = 100, n_days: int = 1000, seed: int = 0,
                         mom_strength: float = 0.0008,
                         rev_strength: float = 0.0006,
                         sigma_daily: float = 0.02,
                         vol_persistence: float = 0.97,
                         vol_of_vol: float = 0.10,
                         burn_in: int = 280) -> DailyPanel:
    """Generate a seeded daily panel with planted momentum + reversal alpha.

    Return generating process (per stock ``i``, day ``t``):

        r_{i,t} = mom_strength * z(mom 12-1)_{i,t-1}
                  - rev_strength * z(ret 21d)_{i,t-1}
                  + sigma_{i,t} * eps_{i,t}

    where ``z`` is the cross-sectional z-score and ``log sigma`` follows an
    AR(1) around ``log(sigma_daily)`` with persistence ``vol_persistence``
    and innovation std ``vol_of_vol`` (vol clustering).  The planted next-day
    momentum IC is approximately ``mom_strength / sigma_daily`` (= 0.04 for
    the defaults, inside the realistic 0.03-0.05 band).

    Parameters
    ----------
    burn_in : int
        Extra leading days simulated (and discarded from the returned index
        only in the sense that features need warm-up); must be > 252 so the
        momentum signal is live from the first returned day.

    Returns
    -------
    DailyPanel
        ``n_days`` business days x ``n_stocks`` tickers ``S000..``.
    """
    if n_stocks < 2 or n_days < 2:
        raise ValueError("need n_stocks >= 2 and n_days >= 2")
    if burn_in < 253:
        raise ValueError("burn_in must be > 252 so planted momentum is live from day 1")
    rng = np.random.default_rng(seed)
    total = burn_in + n_days

    logvol = np.log(sigma_daily) * np.ones(n_stocks)
    rets = np.zeros((total, n_stocks))
    sig = np.zeros((total, n_stocks))
    for t in range(total):
        logvol = (1 - vol_persistence) * np.log(sigma_daily) + vol_persistence * logvol \
            + vol_of_vol * rng.standard_normal(n_stocks)
        sigma_t = np.exp(logvol)
        alpha = np.zeros(n_stocks)
        if t > 252:
            mom = rets[t - 252:t - 21].sum(axis=0)
            rev = rets[t - 21:t].sum(axis=0)
            alpha = mom_strength * _zs(mom) - rev_strength * _zs(rev)
        rets[t] = alpha + sigma_t * rng.standard_normal(n_stocks)
        sig[t] = sigma_t

    dates = pd.bdate_range("2015-01-02", periods=n_days)
    tickers = [f"S{i:03d}" for i in range(n_stocks)]
    r = rets[burn_in:]
    p0 = rng.uniform(20.0, 200.0, size=n_stocks)
    prices = pd.DataFrame(p0 * np.exp(np.cumsum(r, axis=0)),
                          index=dates, columns=tickers)
    adv = rng.lognormal(mean=np.log(1e6), sigma=0.8, size=n_stocks)
    vol_noise = rng.lognormal(mean=-0.5 * 0.3**2, sigma=0.3, size=(n_days, n_stocks))
    volumes = pd.DataFrame(adv * vol_noise, index=dates, columns=tickers)
    returns = prices.pct_change()
    return DailyPanel(prices=prices, volumes=volumes, returns=returns)


def _zs(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    if sd == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / sd
