"""Cross-asset time-series momentum (W23 part 3, study F).

Plain idea: an asset that has gone up over the past year tends to keep
going up over the next month, and one that has gone down tends to keep
going down. Unlike cross-sectional momentum, each asset is judged only
against its own past, so the portfolio can be long everything or short
everything. Moskowitz, Ooi and Pedersen (2012, "Time series momentum", JFE
104(2), 228-250) find it in 58 futures across equity indices, currencies,
commodities and bonds; Hurst, Ooi and Pedersen (2017, "A century of
evidence on trend-following investing", JPM 44(1)) back to 1880. Who pays:
the authors read it as initial under-reaction and later over-reaction to
news, plus hedgers who pay to lay off risk. It is a known premium, so a
positive training result reproduces the literature and is not a discovery.

Formal statement. For asset ``s`` with monthly excess return ``r_{s,m}``
and ex-ante annualised volatility ``sigma_{s,m}`` known at the end of month
``m``:

    w_{s,m} = sign(sum_{k=0}^{L-1} r_{s,m-k}) * target / sigma_{s,m}
    TSMOM_{s,m+1} = w_{s,m} * r_{s,m+1}

with ``L = 12`` and ``target = 0.40`` as in the paper (40% per asset, which
gives a diversified portfolio of roughly 10-15% volatility). The portfolio
is the equal-weight mean over the assets that have a weight that month.
Excess returns must be financed: FX from ``research/panel/signals.py``
(CIP, calendar-day accrual), equities as ``Mkt-RF`` from the French library.
Volatility, when not supplied, is an exponentially weighted estimate on
monthly squared excess returns with a centre of mass of ``com`` months
(``sqrt(12 * EWMA(r^2))``). This differs from the paper, which uses daily
returns with a 60-day centre of mass; the difference is stated with every
result.

Every input passes the hold-out lock (``research/panel/holdout_lock.py``),
so only months before 2016 are seen until a new signed, dated registration
opens it. The draft registration is
``research/prereg/2026-11-03-tsmom-cross-asset.md`` (unsigned).

Worked example (``tests/test_tsmom.py``): an asset earns +1% a month for
twelve months, then +2%. With a supplied constant volatility of 10% the
scale is 0.40 / 0.10 = 4, the twelve-month sum is positive, so the weight
is +4 and month 13 earns 4 * 2% = 8%. If month 13 is -2% instead, it
earns -8%: momentum loses at turning points.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PANEL = Path(__file__).resolve().parent.parent / "panel"
if str(_PANEL) not in sys.path:
    sys.path.insert(0, str(_PANEL))

from holdout_lock import HoldoutLock  # noqa: E402
from study import nw_tstat  # noqa: E402

LOOKBACK = 12
TARGET_VOL = 0.40
VOL_COM = 12.0
VOL_NOTE = ("ex-ante volatility: EWMA of monthly squared excess returns, centre of mass "
            "{com:g} months (the paper uses daily returns, 60-day centre of mass)")


def ewma_vol(rx: pd.DataFrame, com: float = VOL_COM, min_periods: int = 12) -> pd.DataFrame:
    """Annualised ex-ante volatility known at each month end."""
    var = (rx.astype(float) ** 2).ewm(com=com, min_periods=min_periods).mean()
    return (12.0 * var) ** 0.5


def signal(rx: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """+1 / -1 / 0 by the sign of the trailing ``lookback``-month excess return sum."""
    s = rx.rolling(lookback, min_periods=lookback).sum()
    return pd.DataFrame(np.sign(s.to_numpy()), index=s.index, columns=s.columns)


def weights(rx: pd.DataFrame, vol: pd.DataFrame | None = None, lookback: int = LOOKBACK,
            target: float = TARGET_VOL, com: float = VOL_COM) -> pd.DataFrame:
    """Weights set at each month end, earning the next month's excess return."""
    sigma = ewma_vol(rx, com) if vol is None else vol.reindex_like(rx)
    return signal(rx, lookback) * target / sigma


def turnover(w: pd.DataFrame) -> pd.Series:
    """Sum of absolute weight changes per month (the first build counts)."""
    return w.fillna(0.0).diff().abs().sum(axis=1).iloc[1:]


def _stats(x: pd.Series) -> dict[str, float | int | None]:
    x = x.dropna()
    n = len(x)
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    ok = n > 1 and sd > 0
    return {"months": n,
            "mean_monthly": float(x.mean()) if n else None,
            "vol_annual": sd * math.sqrt(12) if ok else None,
            "sharpe_annual": float(x.mean()) / sd * math.sqrt(12) if ok else None,
            "nw_t": nw_tstat(x, lag=3)}


def tsmom(rx: pd.DataFrame, vol: pd.DataFrame | None = None, lookback: int = LOOKBACK,
          target: float = TARGET_VOL, com: float = VOL_COM,
          lock: HoldoutLock | None = None) -> dict:
    """Per-asset and portfolio TSMOM returns with summary statistics."""
    lock = lock or HoldoutLock()
    r = lock.guard(rx, "monthly excess returns")
    assert isinstance(r, pd.DataFrame)
    v = None
    if vol is not None:
        v = lock.guard(vol, "ex-ante volatility")
        assert isinstance(v, pd.DataFrame)
    w = weights(r, v, lookback, target, com)
    held = w.shift(1)
    timed = (held * r).loc[held.notna().any(axis=1)]
    port = timed.mean(axis=1, skipna=True).rename("TSMOM")
    stats = {c: _stats(timed[c]) for c in timed.columns}
    stats["TSMOM"] = _stats(port)
    live = w.loc[w.notna().any(axis=1)]
    return {
        "label": "confirmatory" if lock.is_open else "exploratory",
        "vol_note": "supplied ex-ante volatility" if vol is not None else VOL_NOTE.format(com=com),
        "weights": w,
        "returns": timed,
        "portfolio": port,
        "stats": stats,
        "mean_turnover": float(turnover(live).mean()) if len(live) > 1 else None,
    }


def assemble_assets(**blocks: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Join monthly excess-return blocks into one frame, prefixing asset names.

    ``assemble_assets(fx=fx_rx, eq=french[["Mkt-RF"]])`` gives columns
    ``fx:AUD``, ..., ``eq:Mkt-RF``. Dates are aligned to month end."""
    parts = []
    for name, b in blocks.items():
        f = (b.to_frame() if isinstance(b, pd.Series) else b).copy()
        f.index = pd.DatetimeIndex(f.index).to_period("M").to_timestamp(how="end").normalize()
        f.columns = [f"{name}:{c}" for c in f.columns]
        parts.append(f)
    return pd.concat(parts, axis=1).sort_index()
