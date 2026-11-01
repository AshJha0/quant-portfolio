"""Event-driven intraday FX backtester with session costs and carry.

Execution convention (shared by every alpha project in the repository):
signals use data up to the close of period t. Orders fill at a modelled
t+1 price, ``p_t + f (p_{t+1} - p_t)`` with ``f = 0.5`` by default; a
close-auction variant fills at ``p_t`` and pays an auction cost.

In bar terms, with ``d = close_{t+1} - close_t``, the P&L booked at bar
``t+1`` is ``pos_{t-1} * f * d + pos_t * (1 - f) * d``: the old position
is still held until the fill, the new one from the fill onwards. The
trade pays half the session spread of bar ``t+1`` (the bar it fills in).
In ``close_auction`` mode the P&L is ``pos_t * d`` and the trade, booked
at bar ``t``, pays half the bar-``t`` spread plus ``auction_cost_bp`` of
the price. For an hourly FX bar "close auction" means filling at the
decision bar's close mid with an extra immediacy cost; there is no FX
auction. Carry accrues at the daily rollover (5pm NY = 21:00 London
here) to the position held across it, time-weighted by ``f`` when the
position changes inside the bar.

Units
-----
* Positions: base-currency notional units (1.0 = 1 unit of base).
* ``*_quote`` columns: quote-currency P&L per unit base notional.
* ``*_pips`` columns: quote P&L divided by ``pip_size``: "pips earned
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

from .research_stats import lo_sharpe_se

__all__ = [
    "BacktestConfig",
    "ExecutionConfig",
    "IntradayBacktester",
    "information_coefficient",
    "BARS_PER_YEAR",
]

#: Hourly bars per year used for annualisation: 24 bars x 261 weekdays.
#: Session-filtered zero-position bars are included in the count.
BARS_PER_YEAR = 24.0 * 261.0


@dataclass(frozen=True)
class ExecutionConfig:
    """When and at what price a decision taken at the close of bar t fills.

    Attributes
    ----------
    mode : {"next", "close_auction"}
        ``"next"`` fills at ``p_t + f (p_{t+1} - p_t)``. ``"close_auction"``
        fills at ``p_t`` and charges ``auction_cost_bp`` extra.
    fill_fraction : float
        ``f`` in (0, 1]. 1.0 is the next bar's close.
    auction_cost_bp : float
        Extra cost per unit traded notional in ``close_auction`` mode, in
        basis points of price. 0.5 bp is about 0.55 pips on EURUSD at 1.10.
    """

    mode: str = "next"
    fill_fraction: float = 0.5
    auction_cost_bp: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in ("next", "close_auction"):
            raise ValueError(f"mode must be 'next' or 'close_auction', got {self.mode!r}")
        f = float(self.fill_fraction)
        if not np.isfinite(f) or not (0.0 < f <= 1.0):
            raise ValueError(f"fill_fraction must be in (0, 1], got {self.fill_fraction}")
        c = float(self.auction_cost_bp)
        if not np.isfinite(c) or c < 0:
            raise ValueError(f"auction_cost_bp must be finite and >= 0, got {self.auction_cost_bp}")


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
        execution: ExecutionConfig | None = None,
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
        execution : ExecutionConfig, optional
            Fill convention; default ``ExecutionConfig()`` (next, f = 0.5).

        Returns
        -------
        (ledger, summary) : (pandas.DataFrame, dict)
            ``ledger`` has per-bar columns ``close``, ``pos``, ``gross_quote``,
            ``cost_quote``, ``carry_quote``, ``net_quote``, ``gross_pips``,
            ``cost_pips``, ``carry_pips``, ``net_pips``, ``net_base``,
            ``cum_net_pips``.  ``summary`` aggregates in pips plus a
            per-bar Sharpe annualised at ``BARS_PER_YEAR`` bars/year with
            its Lo/Newey-West standard error (``sharpe_se``).

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

        ex = execution if execution is not None else ExecutionConfig()
        if not np.all(np.isfinite(close)):
            raise ValueError("bars close contains NaN/inf")
        pos_prev = np.concatenate([[0.0], pos[:-1]])    # decided at t-1
        pos_prev2 = np.concatenate([[0.0, 0.0], pos[:-2]])[:n]  # decided at t-2
        dmid = np.concatenate([[0.0], np.diff(close)])
        spread_hs = 0.5 * self._spread_pips(hour) * cfg.pip_size
        close_prev = np.concatenate([[np.nan], close[:-1]])

        if ex.mode == "next":
            f = ex.fill_fraction
            # Bar t books the move close_{t-1} -> close_t. The position
            # decided at t-1 fills at close_{t-1} + f * dmid_t; until then
            # the position decided at t-2 is held.
            held = f * pos_prev2 + (1.0 - f) * pos_prev
            gross_q = pos_prev2 * f * dmid + pos_prev * (1.0 - f) * dmid
            turnover = np.abs(pos_prev - pos_prev2)
            cost_q = turnover * spread_hs
            fill_price = close_prev + f * dmid
        else:
            held = pos_prev
            gross_q = pos_prev * dmid
            turnover = np.abs(pos - pos_prev)
            cost_q = turnover * (spread_hs + ex.auction_cost_bp * 1e-4 * close)
            fill_price = close.copy()

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
        carry_q = np.where(
            crossed, held * close_prev * carry_rate / cfg.days_per_year, 0.0
        )
        carry_q = np.nan_to_num(carry_q)

        net_q = gross_q - cost_q + carry_q
        pip = cfg.pip_size
        ledger = pd.DataFrame(
            {
                "close": close,
                "pos": pos,
                "fill_price": np.where(turnover > 0, fill_price, np.nan),
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
            "sharpe_ann": float(net.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0,
            "sharpe_se": float(lo_sharpe_se(net, BARS_PER_YEAR)) if sd > 0 and n >= 3 else float("nan"),
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
