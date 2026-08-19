"""FX volatility risk premium (implied vs realized / forecast) calculator.

The volatility risk premium (VRP) is the spread between option-implied
volatility and subsequently realized (or model-forecast) volatility:

    VRP_t = IV_t - RV_{t -> t+tau}

Persistently positive VRP is one of the best-documented facts in FX options:
sellers of vol earn the premium as compensation for jump/crash risk (and for
G10 pairs the premium concentrates in the risk-reversal wings). The desk use
-- sizing a short-vol program, deciding when the premium is rich enough to
sell, and when a *negative* premium (realized above implied: crisis) says to
stand down -- is discussed in docs/DESK_GUIDE.md.

Units: all vols are **annualized, decimal** (0.10 = 10 vol points... note
10% = 0.10; "vol points" below means percentage points of annualized vol).
Variance-swap P&L uses variance units (vol squared), matching how variance
swaps actually settle.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "realized_vol_forward",
    "vol_risk_premium",
    "variance_swap_pnl",
    "premium_summary",
]


def _clean(v, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def realized_vol_forward(
    returns: Sequence[float] | np.ndarray,
    window: int,
    periods_per_year: int = 252,
) -> np.ndarray:
    """Forward-looking realized vol: RV over the *next* ``window`` returns.

    ``rv[t] = sqrt(ppy * mean(r_{t+1}^2 .. r_{t+window}^2))`` (zero-mean
    convention, matching variance-swap settlement). The last ``window``
    entries are NaN (window incomplete) -- callers must align on valid
    entries; :func:`vol_risk_premium` does this for you.
    """
    r = _clean(returns, "returns")
    if window < 2 or window > r.size:
        raise ValueError(f"window must be in [2, {r.size}], got {window}")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError(
            f"periods_per_year must be positive and finite, got {periods_per_year!r}"
        )
    r2 = r ** 2
    csum = np.concatenate([[0.0], np.cumsum(r2)])
    out = np.full(r.size, np.nan)
    # rv[t] covers returns t+1 .. t+window  (indices t+1 .. t+window in r, 0-based)
    valid_t = r.size - window - 1
    if valid_t >= 0:
        t = np.arange(valid_t + 1)
        out[t] = np.sqrt(periods_per_year * (csum[t + 1 + window] - csum[t + 1]) / window)
    return out


def vol_risk_premium(
    implied_vol: Sequence[float] | np.ndarray | pd.Series,
    realized_or_forecast_vol: Sequence[float] | np.ndarray | pd.Series,
) -> np.ndarray:
    """VRP series: implied minus realized (or model-forecast) vol.

    Both inputs annualized decimal vols, aligned element-wise; NaNs in the
    realized leg (incomplete forward windows) propagate to NaN premia, which
    downstream summaries ignore explicitly.
    """
    iv = np.asarray(implied_vol, dtype=float)
    rv = np.asarray(realized_or_forecast_vol, dtype=float)
    if iv.shape != rv.shape or iv.ndim != 1:
        raise ValueError(f"implied and realized must be 1-D and aligned, got {iv.shape} vs {rv.shape}")
    if not np.isfinite(iv).all():
        raise ValueError("implied vol contains NaN or infinite values")
    if (iv[np.isfinite(iv)] < 0).any() or (rv[np.isfinite(rv)] < 0).any():
        raise ValueError("volatilities must be non-negative")
    return iv - rv


def variance_swap_pnl(
    implied_vol: Sequence[float] | np.ndarray,
    realized_vol: Sequence[float] | np.ndarray,
    vega_notional: float = 1.0,
) -> np.ndarray:
    """Per-trade P&L of *selling* a variance swap at strike = implied vol.

    Short-variance payoff in vega-notional terms:

        pnl = N_vega * (K^2 - RV^2) / (2 K),  K = implied vol

    (the ``2K`` divisor converts variance units to vega/vol-point units, the
    standard market convention). Positive when realized comes in below the
    implied strike -- harvesting the premium; sharply negative in a depeg or
    crisis (the convexity of the variance payoff works *against* the seller).
    """
    iv = _clean(implied_vol, "implied_vol")
    rv = np.asarray(realized_vol, dtype=float)
    if iv.shape != rv.shape:
        raise ValueError("implied and realized must be aligned")
    if (iv <= 0).any():
        raise ValueError("implied vol strike must be strictly positive")
    if not np.isfinite(vega_notional):
        raise ValueError(f"vega_notional must be finite, got {vega_notional!r}")
    return vega_notional * (iv ** 2 - rv ** 2) / (2.0 * iv)


def premium_summary(premium: Sequence[float] | np.ndarray) -> dict:
    """Summary statistics of a VRP series, ignoring NaNs (incomplete windows).

    Returns mean/median/std (vol-point units of the input), the fraction of
    days with positive premium, and worst drawdown-style minimum -- the
    numbers a desk quotes when pitching or risk-limiting a vol-selling
    program.
    """
    arr = np.asarray(premium, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        raise ValueError("no finite premium observations")
    return {
        "n": int(valid.size),
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "std": float(valid.std(ddof=1)) if valid.size > 1 else np.nan,
        "frac_positive": float((valid > 0).mean()),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }
