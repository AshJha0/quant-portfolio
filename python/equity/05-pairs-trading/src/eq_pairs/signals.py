"""Z-score computation, entry/exit/stop state machine, and position sizing.

Signal conventions
------------------
* z > 0: spread is rich (y expensive vs hedge) -> SHORT the spread
  (short y, long x). z < 0: spread is cheap -> LONG the spread.
* Position is quoted in spread units: +1 = long spread, -1 = short, 0 flat.
* Default rules (all configurable): enter when |z| >= 2, exit when z reverts
  through the exit band (z -> 0), hard stop when |z| >= 4, time stop after
  ``max_holding`` bars (a desk typically sets this at k x half-life; the
  helper :func:`time_stop_bars` computes it).
* Re-entry arming: after ANY exit the pair may not re-enter until |z| has
  first come back inside the entry band. Without this, a stop-loss at
  |z| > stop would be followed by an immediate re-entry at |z| > entry on
  the very next bar — re-fighting the trade the stop just cut.

The state machine emits, for each bar t, the position DECIDED with
information through t. It says nothing about execution timing: the
backtester applies its own strict lag (trade at t+1's close on the signal
from t). Keeping decision and execution separate is what makes the
no-lookahead property testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from .spread import OUFit

__all__ = [
    "zscore_rolling",
    "zscore_ou",
    "SignalRules",
    "generate_signals",
    "time_stop_bars",
    "size_positions",
]


def zscore_rolling(
    spread: pd.Series, window: int, min_periods: Optional[int] = None
) -> pd.Series:
    """Rolling z-score (s_t - mean_window) / std_window.

    The window at t uses observations up to AND INCLUDING t — the value at t
    is a statistic of data through t, so it may only drive trades at t+1
    (the backtester enforces the lag). Warm-up entries (< min_periods) are
    NaN. Windows with zero variance yield NaN, not inf.

    Parameters
    ----------
    spread : pandas.Series
        Spread in dollars.
    window : int
        Rolling window length (>= 3).
    min_periods : int, optional
        Minimum observations required (default = window).
    """
    if window < 3:
        raise ValueError(f"window must be >= 3, got {window}")
    mp = window if min_periods is None else min_periods
    mean = spread.rolling(window, min_periods=mp).mean()
    std = spread.rolling(window, min_periods=mp).std(ddof=1)
    std = std.where(std > 0.0)
    return (spread - mean) / std


def zscore_ou(spread: Union[pd.Series, np.ndarray], ou: OUFit) -> Union[pd.Series, np.ndarray]:
    """Z-score against the fitted OU stationary distribution.

    z_t = (s_t - mu) / (sigma / sqrt(2 kappa)). Unlike the rolling z-score
    this uses the model's stationary variance, so it has no warm-up NaNs —
    but it treats (mu, kappa, sigma) as frozen parameters; in walk-forward
    use they must come from the formation window only.

    Raises
    ------
    ValueError
        If the OU fit is not mean-reverting (stationary variance undefined).
    """
    if not ou.mean_reverting:
        raise ValueError(
            "OU fit is not mean-reverting (kappa ~ 0); z-score undefined — "
            "do not trade this spread"
        )
    return (spread - ou.mu) / ou.stationary_std


def time_stop_bars(half_life: float, k: float = 3.0, cap: int = 252) -> int:
    """Time stop in bars: ceil(k x half-life), capped.

    A convergence trade that has not converged after a few half-lives is
    evidence the model is wrong, not an invitation to wait longer.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not np.isfinite(half_life) or half_life <= 0:
        return cap
    return int(min(np.ceil(k * half_life), cap))


@dataclass(frozen=True)
class SignalRules:
    """Entry/exit/stop configuration.

    Attributes
    ----------
    entry_z : float
        Enter when |z| >= entry_z (default 2.0).
    exit_z : float
        Exit when the z-score has reverted through +/-exit_z towards zero
        (default 0.0 = full mean touch).
    stop_z : float
        Hard stop when |z| >= stop_z (default 4.0). Must be > entry_z.
    max_holding : int or None
        Time stop in bars from entry (None = disabled).
    """

    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 4.0
    max_holding: Optional[int] = None

    def __post_init__(self) -> None:
        if self.entry_z <= 0:
            raise ValueError(f"entry_z must be positive, got {self.entry_z}")
        if not 0 <= self.exit_z < self.entry_z:
            raise ValueError(
                f"need 0 <= exit_z < entry_z, got exit_z={self.exit_z}, "
                f"entry_z={self.entry_z}"
            )
        if self.stop_z <= self.entry_z:
            raise ValueError(
                f"stop_z must exceed entry_z, got stop_z={self.stop_z}, "
                f"entry_z={self.entry_z}"
            )
        if self.max_holding is not None and self.max_holding < 1:
            raise ValueError(f"max_holding must be >= 1, got {self.max_holding}")


def generate_signals(z: pd.Series, rules: SignalRules = SignalRules()) -> pd.DataFrame:
    """Run the entry/exit/stop state machine over a z-score series.

    Parameters
    ----------
    z : pandas.Series
        Z-score; NaNs (warm-up) force/keep the position flat and never
        propagate into the output.
    rules : SignalRules
        Thresholds; see class docstring.

    Returns
    -------
    DataFrame indexed like ``z`` with columns:
    ``position`` (int in {-1, 0, +1}, decided with info through t),
    ``event`` (str: "", "entry_long", "entry_short", "exit_mean",
    "exit_stop", "exit_time", "exit_nan").
    """
    n = len(z)
    pos = np.zeros(n, dtype=int)
    events = [""] * n
    state = 0
    bars_held = 0
    armed = True  # may enter; disarmed after an exit until |z| < entry_z
    zv = z.to_numpy(dtype=float)
    for t in range(n):
        zt = zv[t]
        if np.isnan(zt):
            if state != 0:
                events[t] = "exit_nan"
            state = 0
            bars_held = 0
            armed = True
            pos[t] = 0
            continue
        if state == 0:
            if not armed and abs(zt) < rules.entry_z:
                armed = True
            if armed and zt >= rules.entry_z:
                state, bars_held = -1, 0
                events[t] = "entry_short"
            elif armed and zt <= -rules.entry_z:
                state, bars_held = 1, 0
                events[t] = "entry_long"
        else:
            bars_held += 1
            stopped = abs(zt) >= rules.stop_z
            reverted = (state == 1 and zt >= -rules.exit_z) or (
                state == -1 and zt <= rules.exit_z
            )
            timed_out = (
                rules.max_holding is not None and bars_held >= rules.max_holding
            )
            if stopped:
                events[t] = "exit_stop"
            elif reverted:
                events[t] = "exit_mean"
            elif timed_out:
                events[t] = "exit_time"
            if stopped or reverted or timed_out:
                state = 0
                bars_held = 0
                armed = False
        pos[t] = state
    return pd.DataFrame({"position": pos, "event": events}, index=z.index)


def size_positions(
    price_y: float,
    price_x: float,
    direction: int,
    beta: float,
    gross: float = 1_000_000.0,
    mode: str = "dollar",
) -> tuple[float, float]:
    """Share quantities (q_y, q_x) for one spread unit of ``direction``.

    Parameters
    ----------
    price_y, price_x : float
        Current prices in dollars (must be positive).
    direction : int
        +1 long spread (long y / short x), -1 short spread, 0 flat.
    beta : float
        Hedge ratio (shares of x per share of y); used by "beta" mode.
        Must be positive for a conventional long-short pair.
    gross : float
        Target gross dollar exposure |q_y| P_y + |q_x| P_x (default $1mm).
    mode : {"dollar", "beta"}
        "dollar": equal dollars on each leg (long gross/2, short gross/2) —
        dollar-neutral at entry, but the hedge deviates from the
        cointegrating ratio. "beta": shares in the cointegrating proportion
        q_x = -beta q_y, scaled to the same gross — P&L tracks the spread
        exactly, but the trade carries a (usually small) net dollar
        exposure.

    Returns
    -------
    (q_y, q_x) : share quantities (signed; short is negative).
    """
    if direction == 0:
        return 0.0, 0.0
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1, 0 or +1, got {direction}")
    if price_y <= 0 or price_x <= 0:
        raise ValueError(f"prices must be positive, got {price_y}, {price_x}")
    if gross <= 0:
        raise ValueError(f"gross must be positive, got {gross}")
    if mode == "dollar":
        q_y = direction * (gross / 2.0) / price_y
        q_x = -direction * (gross / 2.0) / price_x
    elif mode == "beta":
        if beta <= 0 or not np.isfinite(beta):
            raise ValueError(
                f"beta must be positive and finite for beta-neutral sizing, got {beta}"
            )
        scale = gross / (price_y + beta * price_x)
        q_y = direction * scale
        q_x = -direction * beta * scale
    else:
        raise ValueError(f"mode must be 'dollar' or 'beta', got {mode!r}")
    return float(q_y), float(q_x)
