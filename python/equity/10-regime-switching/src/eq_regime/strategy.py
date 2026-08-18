"""Regime-conditional allocation with hysteresis and vol targeting.

Allocation rule
---------------
* **bull**       — fully long the equity factor (equal-weight index).
* **bear**       — de-risked to the defensive sleeve (cash; optionally a
  low-vol proxy via ``defensive_weight``).
* **transition** — scaled exposure between the two.

Hysteresis band — why it prevents whipsaw
-----------------------------------------
A naive rule ("bear if ``p_bear > 0.5``") flips every time the filtered
probability crosses one line.  Near a regime boundary the filtered
probability oscillates around that line for days, so the naive rule churns:
each crossing pays transaction costs twice and realises noise, not signal.

The hysteresis rule enters the bear state only when ``p_bear`` rises above
``enter`` (default 0.70) and leaves it only when ``p_bear`` falls below
``exit`` (default 0.30).  Inside the band ``[exit, enter]`` the PREVIOUS
state persists.  Small oscillations inside the band cause no trades at all;
only a decisive move through the whole band flips the position.  The test
suite verifies that hysteresis strictly reduces turnover versus the naive
threshold on noisy probability paths.

Boundary convention (test-enforced): at exactly ``p == enter`` the rule
does NOT enter (strict ``>``), at exactly ``p == exit`` it does NOT exit
(strict ``<``); the band is closed.

Vol targeting
-------------
Position size is scaled by ``target_vol / realized_vol`` (clipped to
``max_leverage``), using trailing realized vol only — the ex-ante forecast
at ``t`` never sees returns after ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252

__all__ = [
    "hysteresis_regime",
    "naive_threshold_regime",
    "regime_target_weight",
    "vol_target_scale",
    "build_weights",
    "turnover",
]


def hysteresis_regime(
    p_bear: pd.Series | np.ndarray,
    enter: float = 0.70,
    exit_: float = 0.30,
    initial_bear: bool = False,
) -> np.ndarray:
    """Two-threshold hysteresis on the bear probability.

    Parameters
    ----------
    p_bear : (T,) filtered bear-regime probability in [0, 1].
    enter : float
        Enter bear when ``p_bear > enter`` (strict).
    exit_ : float
        Exit bear when ``p_bear < exit_`` (strict).  Must satisfy
        ``exit_ < enter``.
    initial_bear : bool
        State before the first observation.

    Returns
    -------
    (T,) boolean array — True where the hysteresis state is bear.
    """
    if not 0.0 <= exit_ < enter <= 1.0:
        raise ValueError(f"need 0 <= exit < enter <= 1, got exit={exit_}, enter={enter}")
    p = np.asarray(p_bear, dtype=float)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_bear must lie in [0, 1]")
    out = np.empty(len(p), dtype=bool)
    state = initial_bear
    for t, pt in enumerate(p):
        if not state and pt > enter:
            state = True
        elif state and pt < exit_:
            state = False
        out[t] = state
    return out


def naive_threshold_regime(p_bear: pd.Series | np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Single-threshold rule (for comparison): bear iff ``p_bear > threshold``."""
    p = np.asarray(p_bear, dtype=float)
    return p > threshold


def regime_target_weight(
    regime: np.ndarray,
    bear_flag: np.ndarray | None = None,
    bull_weight: float = 1.0,
    transition_weight: float = 0.5,
    bear_weight: float = 0.0,
) -> np.ndarray:
    """Map regime labels (+ hysteresis bear flag) to a target equity weight.

    The hysteresis flag OVERRIDES the label: whenever ``bear_flag[t]`` is
    True the weight is ``bear_weight`` regardless of the argmax label, so
    de-risking obeys the banded rule rather than raw argmax flips.  When
    the flag is False, an unconfirmed ``bear`` label (probability still
    inside the band) is treated as ``transition`` — the weight is floored
    at ``transition_weight`` — so full de-risking happens only through the
    band.

    Parameters
    ----------
    regime : (T,) array of labels in {'bull', 'transition', 'bear'}.
    bear_flag : (T,) boolean hysteresis state, optional.
    bull_weight, transition_weight, bear_weight : float
        Equity weight per regime (1 - weight sits in the defensive sleeve).

    Returns
    -------
    (T,) float equity weights.
    """
    regime = np.asarray(regime)
    weights = np.where(
        regime == "bull",
        bull_weight,
        np.where(regime == "bear", bear_weight, transition_weight),
    ).astype(float)
    if bear_flag is not None:
        bear_flag = np.asarray(bear_flag, dtype=bool)
        if len(bear_flag) != len(weights):
            raise ValueError("bear_flag length mismatch")
        weights = np.where(bear_flag, bear_weight, np.maximum(weights, transition_weight))
    return weights


def vol_target_scale(
    index_returns: pd.Series,
    target_vol: float = 0.10,
    window: int = 21,
    max_leverage: float = 1.5,
) -> pd.Series:
    """Trailing-vol scaling factor ``target_vol / realized_vol`` (clipped).

    The scale at ``t`` uses returns up to and including ``t`` (trailing
    window) — applied to the position held over ``t+1`` it is ex-ante.

    Parameters
    ----------
    index_returns : (T,) daily log-returns of the traded factor.
    target_vol : float
        Annualised vol target.
    window : int
        Trailing window for realized vol.
    max_leverage : float
        Cap on the scale.

    Returns
    -------
    (T,) scale factors (NaN during warmup filled with 1.0).
    """
    if target_vol <= 0:
        raise ValueError(f"target_vol must be > 0, got {target_vol}")
    if max_leverage <= 0:
        raise ValueError(f"max_leverage must be > 0, got {max_leverage}")
    realized = index_returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    scale = (target_vol / realized).clip(upper=max_leverage)
    return scale.fillna(1.0)


def build_weights(
    detection: pd.DataFrame,
    index_returns: pd.Series,
    enter: float = 0.70,
    exit_: float = 0.30,
    bull_weight: float = 1.0,
    transition_weight: float = 0.5,
    bear_weight: float = 0.0,
    target_vol: float | None = 0.10,
    vol_window: int = 21,
    max_leverage: float = 1.5,
) -> pd.Series:
    """Full regime-conditional weight path from a detection table.

    Combines the regime target weight (with hysteresis on ``p_bear``) and
    optional regime-conditional vol targeting.  Everything is computed from
    information available at ``t``; the caller applies ``w_t`` to the return
    over ``t -> t+1`` (see :mod:`eq_regime.backtest`).

    Parameters
    ----------
    detection : pd.DataFrame
        Output of :func:`eq_regime.detection.expanding_fit_detect`
        (columns ``p_bear`` and ``regime``).
    index_returns : pd.Series
        Daily log-returns of the traded factor, superset of the detection
        index.
    Other parameters as in the component functions.

    Returns
    -------
    pd.Series
        Equity weight per detection date.
    """
    required = {"p_bear", "regime"}
    if not required.issubset(detection.columns):
        raise ValueError(f"detection table must have columns {sorted(required)}")
    bear_flag = hysteresis_regime(detection["p_bear"], enter=enter, exit_=exit_)
    base = regime_target_weight(
        detection["regime"].to_numpy(),
        bear_flag=bear_flag,
        bull_weight=bull_weight,
        transition_weight=transition_weight,
        bear_weight=bear_weight,
    )
    weights = pd.Series(base, index=detection.index, name="weight")
    if target_vol is not None:
        scale = vol_target_scale(
            index_returns, target_vol=target_vol, window=vol_window, max_leverage=max_leverage
        ).reindex(detection.index)
        weights = (weights * scale).clip(upper=max_leverage)
    return weights


def turnover(weights: pd.Series | np.ndarray) -> float:
    """Total one-sided turnover ``sum_t |w_t - w_{t-1}|`` (entry included)."""
    w = np.asarray(weights, dtype=float)
    if len(w) == 0:
        return 0.0
    return float(np.abs(np.diff(w, prepend=0.0)).sum())
