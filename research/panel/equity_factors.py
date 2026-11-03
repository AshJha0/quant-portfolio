"""Equity factor signals on the Kenneth French library (W23, W9 extension).

Plain idea: the Fama-French factors (market, size, value, profitability,
investment, momentum) are long-short equity portfolios with a free, public
monthly history back to 1926 or 1963. They give the research programme a
second asset class with real data and no key. The signal here is factor
momentum: hold each factor long after a year in which it earned money and
short after a year in which it lost (Ehsani and Linnainmaa 2022, JF 77(3),
"Factor Momentum and the Momentum Factor"; Gupta and Kelly 2019, JPM).

Who pays: the factors are themselves compensated risks or anomalies; factor
momentum times them. Ehsani and Linnainmaa read it as the factors'
autocorrelation, which they report is positive for most factors.

Formal statement. For factor return ``f_m`` (a long-short, so already an
excess return; ``Mkt-RF`` is excess of the one-month bill),
``signal_m = sign(sum_{k=0}^{L-1} f_{m-k})`` and the strategy return is
``signal_{m-1} f_m``. ``L = 12`` by default. Results run behind the same
hold-out lock as the FX panel (``holdout_lock.HoldoutLock``), so until a
new signed registration exists only months before 2016 are seen.

Worked example (``tests/test_equity_factors.py``): a factor earns +1% in
months 0-11 and -1% in months 12-23. The trailing 12-month sum is +12% at
month 11, falls by 2% a month, is 0 at month 17 and negative from month 18.
So the signal is +1 for months 11-16, 0 at 17 and -1 from 18. The strategy
earns -1% in months 12-17, 0 in month 18 and +1% from month 19.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from holdout_lock import HoldoutLock


def factor_momentum_signal(factors: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """+1 / -1 / 0 by the sign of the trailing ``lookback``-month sum."""
    s = factors.rolling(lookback, min_periods=lookback).sum()
    return pd.DataFrame(np.sign(s.to_numpy()), index=s.index, columns=s.columns)


def factor_momentum(factors: pd.DataFrame, lookback: int = 12,
                    lock: HoldoutLock | None = None) -> dict:
    """Per-factor and equal-weight timed returns with summary statistics."""
    lock = lock or HoldoutLock()
    f = lock.guard(factors, "French factors")
    assert isinstance(f, pd.DataFrame)
    sig = factor_momentum_signal(f, lookback)
    timed = (sig.shift(1) * f).dropna(how="all")
    timed = timed.loc[sig.shift(1).notna().any(axis=1)]
    timed["EW"] = timed.mean(axis=1)
    stats = {}
    for c in timed.columns:
        x = timed[c].dropna()
        sd = float(x.std(ddof=1))
        stats[c] = {"months": len(x), "mean_monthly": float(x.mean()),
                    "sharpe_annual": float(x.mean() / sd * math.sqrt(12)) if sd > 0 else None}
    return {"label": "confirmatory" if lock.is_open else "exploratory",
            "returns": timed, "stats": stats}
