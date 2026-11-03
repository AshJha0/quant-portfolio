"""VIX against realised variance: a descriptive study (W23 part 3, study E).

Plain idea: the VIX is the market's price of the next 30 days of S&P 500
variance, quoted as a volatility. Squared, it can be set against the
variance that then actually happened. On average the quote is higher than
the outcome: buyers of variance pay a premium to insure against the market
falling and volatility jumping together, and sellers of variance collect it
(Carr and Wu 2009, "Variance risk premiums", RFS 22(3), 1311-1341; Bollerslev,
Tauchen and Zhou 2009, RFS 22(11)). Who pays: investors who want crash
insurance; who earns: whoever writes it and bears the crash.

This study is descriptive. It uses no option prices: only the CBOE's
published closing VIX level (``cboe:VIX`` through ``qp_data``, keyless) and
the daily US market return from the Kenneth French library
(``Mkt-RF + RF``, the CRSP value-weighted market). That market is broader
than the S&P 500 the VIX is built on, so the gap is an approximation and is
labelled as one. No trading rule is tested and no premium is claimed as
tradable here; a confirmatory test would need a new dated registration.

Formal statement. With daily log market return ``r_t`` and ``h = 21``
trading days (the VIX's 30 calendar days):

* implied variance ``IV_t = (VIX_t / 100)^2`` (annualised);
* forward realised variance ``RV_t = (252 / h) * sum_{k=1}^{h} r_{t+k}^2``,
  the variance over the *next* ``h`` days, annualised the same way;
* the variance premium ``VRP_t = IV_t - RV_t``.

Overlapping windows make ``VRP_t`` autocorrelated to lag ``h - 1``, so its
mean is tested with a Newey-West t-statistic at that lag. The Mincer-Zarnowitz
regression ``RV_t = a + b IV_t + e`` says whether the quote is an unbiased
forecast (``a = 0, b = 1``) or a biased but informative one.

Every input passes the hold-out lock (``research/panel/holdout_lock.py``),
so while the lock is closed only data before 2016-01-01 is seen, and the
last ``h`` dates have no forward window rather than one that reads 2016.

Worked example (``tests/test_vix_rv.py``): a flat VIX of 20 gives
``IV = 0.04``. If every daily log return is +/-1% then each squared return
is 1e-4, ``RV = (252/21) * 21 * 1e-4 = 0.0252`` and ``VRP = 0.0148``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent / "panel",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from holdout_lock import HoldoutLock  # noqa: E402
from study import nw_tstat  # noqa: E402

HORIZON = 21
TRADING_DAYS = 252
LABEL = ("descriptive: CBOE published VIX level against realised variance of the "
         "CRSP value-weighted US market (French library); uses no option prices")


def implied_variance(vix: pd.Series) -> pd.Series:
    """Annualised implied variance ``(VIX / 100)^2``."""
    return (vix.astype(float) / 100.0) ** 2


def forward_realised_variance(log_ret: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Annualised variance over the next ``horizon`` days; NaN where incomplete."""
    sq = log_ret.astype(float) ** 2
    fwd = sq[::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1)
    return fwd * (TRADING_DAYS / horizon)


def market_log_returns(french_daily: pd.DataFrame) -> pd.Series:
    """Daily log return of the market, ``log(1 + Mkt-RF + RF)``."""
    simple = french_daily["Mkt-RF"].astype(float) + french_daily["RF"].astype(float)
    return pd.Series(np.log1p(simple.to_numpy()), index=simple.index, name="mkt")


def mincer_zarnowitz(y: pd.Series, x: pd.Series) -> dict[str, float]:
    """OLS ``y = a + b x``: intercept, slope and R^2."""
    df = pd.concat([y, x], axis=1).dropna()
    yy, xx = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    xm, ym = xx.mean(), yy.mean()
    b = float(((xx - xm) @ (yy - ym)) / ((xx - xm) @ (xx - xm)))
    a = float(ym - b * xm)
    resid = yy - a - b * xx
    r2 = float(1.0 - (resid @ resid) / ((yy - ym) @ (yy - ym)))
    return {"intercept": a, "slope": b, "r2": r2, "n": float(len(df))}


def vix_study(vix_close: pd.Series, log_ret: pd.Series, horizon: int = HORIZON,
              lock: HoldoutLock | None = None) -> dict:
    """Implied against forward realised variance; see the module docstring."""
    lock = lock or HoldoutLock()
    v = lock.guard(vix_close, "CBOE VIX")
    r = lock.guard(log_ret, "market returns")
    assert isinstance(v, pd.Series) and isinstance(r, pd.Series)
    common = v.index.intersection(r.index)
    iv = implied_variance(v.loc[common])
    rv = forward_realised_variance(r.loc[common], horizon)
    frame = pd.DataFrame({"iv": iv, "rv_fwd": rv})
    frame["vrp"] = frame["iv"] - frame["rv_fwd"]
    ok = frame.dropna()
    vrp = ok["vrp"]
    return {
        "label": LABEL + ("; confirmatory" if lock.is_open else "; exploratory"),
        "frame": frame,
        "days": len(ok),
        "first": str(ok.index.min().date()) if len(ok) else None,
        "last": str(ok.index.max().date()) if len(ok) else None,
        "mean_iv": float(ok["iv"].mean()),
        "mean_rv": float(ok["rv_fwd"].mean()),
        "mean_vrp": float(vrp.mean()),
        "share_positive": float((vrp > 0).mean()),
        "nw_t_vrp": nw_tstat(vrp, lag=horizon - 1),
        "mincer_zarnowitz": mincer_zarnowitz(ok["rv_fwd"], ok["iv"]),
        "vol_points": {"implied": 100 * math.sqrt(float(ok["iv"].mean())),
                       "realised": 100 * math.sqrt(float(ok["rv_fwd"].mean()))},
    }


def load_from_store(as_of: str | None = None) -> tuple[pd.Series, pd.Series]:
    """VIX close and market log returns from the local ``qp_data`` store.

    Fill the store first (network opt-in, no key)::

        QP_DATA_ALLOW_NETWORK=1 python -c "import qp_data; qp_data.fetch('cboe:VIX')"

    plus the French daily factors (``french:`` must be registered in
    ``qp_data.sources.SOURCES``; see ``qp_data.french``). Returns
    ``(vix_close, log_ret)``."""
    import qp_data

    vix = qp_data.get("cboe:VIX", as_of)["close"]
    ff = qp_data.get("french:F-F_Research_Data_Factors_daily", as_of)
    return vix, market_log_returns(ff)
