"""Event-driven intraday FX backtester with session costs and carry.

No-lookahead is structural: the position indexed at bar ``t`` earns the
``t -> t+1`` close-to-close move, i.e. the P&L booked at bar ``t`` is
``pos_{t-1} * (close_t - close_{t-1})``.  Trading costs are charged at
the bar where the position *changes*, at half the session-dependent
quoted spread.  Carry accrues at the daily rollover (5pm NY = 21:00
London here) to the position held across it.

Units
-----
* Positions: base-currency notional units (1.0 = 1 unit of base).
* ``*_quote`` columns: quote-currency P&L per unit base notional.
* ``*_pips`` columns: quote P&L divided by ``pip_size`` — "pips earned
  per unit of position", the desk-standard intraday metric.
* Base-ccy P&L (CONVENTIONS.md: P&L reported in base ccy where needed)
  is ``net_quote / close`` and provided as ``net_base``.
* Rates annualised, continuously compounded, ACT/365F.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from .sessions import session_of_hour

__all__ = ["BacktestConfig", "IntradayBacktester", "information_coefficient"]


@dataclass(frozen=True)
class BacktestConfig:
    """Static configuration of the intraday backtester.

    Attributes
    ----------
    pip_size : float
        Price units per pip.
    spread_pips_by_session : Mapping[str, float]
        Full quoted spread (pips) by session name; cost per unit turnover
        is half of this.
    r_base, r_quote : float
        Annualised deposit rates (ACT/365F, continuously compounded)
        used when no per-bar carry series is supplied.
    rollover_hour : float
        Hour-of-day (London clock) of the overnight rollover.
    bar_hours : float
        Bar length in hours.
    days_per_year : float
        Carry day-count denominator (365 = ACT/365F).
    """

    pip_size: float
    spread_pips_by_session: Mapping[str, float] = field(
        default_factory=lambda: {"asia": 0.6, "london": 0.35, "overlap": 0.2, "ny": 0.4, "late": 1.0}
    )
    r_base: float = 0.0
    r_quote: float = 0.0
    rollover_hour: float = 21.0
    bar_hours: float = 1.0
    days_per_year: float = 365.0


class IntradayBacktester:
    """Vectorised event-driven intraday backtester (state = config only)."""

    def __init__(self, config: BacktestConfig) -> None:
        if config.pip_size <= 0:
            raise ValueError(f"pip_size must be > 0, got {config.pip_size}")
        self.config = config

    def run(
        self,
        bars: pd.DataFrame,
        positions: pd.Series,
        carry: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Run the backtest.

        Parameters
        ----------
        bars : pandas.DataFrame
            Output of ``features.build_bars`` (needs ``close`` and
            ``hour`` columns), indexed by bar end time.
        positions : pandas.Series
            Target position decided at each bar close (same index as
            ``bars``); applied to the *next* bar's return.
        carry : pandas.Series, optional
            Per-bar annualised rate differential ``r_base - r_quote``;
            defaults to the constant configured rates.

        Returns
        -------
        (ledger, summary) : (pandas.DataFrame, dict)
            ``ledger`` has per-bar columns ``close``, ``pos``, ``gross_quote``,
            ``cost_quote``, ``carry_quote``, ``net_quote``, ``gross_pips``,
            ``cost_pips``, ``carry_pips``, ``net_pips``, ``net_base``,
            ``cum_net_pips``.  ``summary`` aggregates in pips plus a
            per-bar Sharpe annualised at ``24*261`` bars/year.

        Raises
        ------
        ValueError
            If indices are misaligned or positions contain NaN.
        """
        cfg = self.config
        if not bars.index.equals(positions.index):
            raise ValueError("positions index must equal bars index")
        pos = positions.to_numpy(dtype=float)
        if np.any(~np.isfinite(pos)):
            raise ValueError("positions contain NaN/inf")
        close = bars["close"].to_numpy(dtype=float)
        hour = bars["hour"].to_numpy(dtype=float) % 24.0
        n = len(bars)

        pos_prev = np.concatenate([[0.0], pos[:-1]])
        dmid = np.concatenate([[0.0], np.diff(close)])
        gross_q = pos_prev * dmid

        # Trade at the close of bar t when the target changes; pay half the
        # session spread of the bar-t session.
        turnover = np.abs(pos - pos_prev)
        half_spread_price = 0.5 * self._spread_pips(hour) * cfg.pip_size
        cost_q = turnover * half_spread_price

        # Carry accrues at bar t if the holding interval (t-1, t] contains
        # the rollover instant, to the position held over that interval.
        hour_prev = np.concatenate([[np.nan], hour[:-1]])
        crossed = self._crosses_rollover(hour_prev, hour)
        if carry is None:
            carry_rate = np.full(n, cfg.r_base - cfg.r_quote)
        else:
            if not bars.index.equals(carry.index):
                raise ValueError("carry index must equal bars index")
            carry_rate = carry.to_numpy(dtype=float)
        close_prev = np.concatenate([[np.nan], close[:-1]])
        carry_q = np.where(
            crossed, pos_prev * close_prev * carry_rate / cfg.days_per_year, 0.0
        )
        carry_q = np.nan_to_num(carry_q)

        net_q = gross_q - cost_q + carry_q
        pip = cfg.pip_size
        ledger = pd.DataFrame(
            {
                "close": close,
                "pos": pos,
                "gross_quote": gross_q,
                "cost_quote": cost_q,
                "carry_quote": carry_q,
                "net_quote": net_q,
                "gross_pips": gross_q / pip,
                "cost_pips": cost_q / pip,
                "carry_pips": carry_q / pip,
                "net_pips": net_q / pip,
                "net_base": net_q / close,
            },
            index=bars.index,
        )
        ledger["cum_net_pips"] = ledger["net_pips"].cumsum()

        net = ledger["net_pips"].to_numpy()
        sd = net.std(ddof=0)
        summary = {
            "gross_pips": float(ledger["gross_pips"].sum()),
            "cost_pips": float(ledger["cost_pips"].sum()),
            "carry_pips": float(ledger["carry_pips"].sum()),
            "net_pips": float(ledger["net_pips"].sum()),
            "net_base": float(ledger["net_base"].sum()),
            "n_trades": int((turnover > 0).sum()),
            "turnover": float(turnover.sum()),
            "sharpe_ann": float(net.mean() / sd * np.sqrt(24.0 * 261.0)) if sd > 0 else 0.0,
            "hit_rate": float((net[net != 0] > 0).mean()) if (net != 0).any() else 0.0,
        }
        return ledger, summary

    def _spread_pips(self, hour: np.ndarray) -> np.ndarray:
        sess = session_of_hour(hour)
        return np.array([self.config.spread_pips_by_session[s] for s in sess], dtype=float)

    def _crosses_rollover(self, hour_prev: np.ndarray, hour: np.ndarray) -> np.ndarray:
        ro = self.config.rollover_hour
        with np.errstate(invalid="ignore"):
            same_day = (hour_prev < hour) & (hour_prev < ro) & (hour >= ro)
            wrapped = (hour_prev > hour) & ((hour_prev < ro) | (hour >= ro))
        out = same_day | wrapped
        out[np.isnan(hour_prev)] = False
        return out


def information_coefficient(
    feature: pd.Series, forward_returns: pd.Series
) -> tuple[float, float]:
    """Pearson IC of a feature against forward returns, with t-stat.

    ``forward_returns`` must already be the *next-bar* return aligned at
    the feature's timestamp (i.e. ``ret.shift(-1)`` of close-to-close
    returns); the caller owns the alignment so the causality is explicit.

    Returns
    -------
    (ic, t_stat) : tuple of float
        ``t = ic * sqrt(n - 2) / sqrt(1 - ic^2)``.
    """
    df = pd.concat([feature, forward_returns], axis=1).dropna()
    if len(df) < 3:
        raise ValueError("need at least 3 aligned observations")
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()
    if x.std(ddof=0) == 0 or y.std(ddof=0) == 0:
        return 0.0, 0.0
    ic = float(np.corrcoef(x, y)[0, 1])
    n = len(df)
    t = ic * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1.0 - ic**2))
    return ic, float(t)
