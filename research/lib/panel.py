"""Cross-sectional panel tools: the LRV portfolio sort and Fama-MacBeth slopes (W6).

Plain idea: instead of asking whether one basket earned money, ask whether,
month after month, currencies with a higher signal earned more than those
with a lower one. Pooling 30+ currencies gives many more independent bets than
one basket, which is what makes the power gate reachable.

``lrv_sort``: at each month-end, rank the currencies alive on that date
(``Universe.members``) by forward discount ``i* - i`` (equivalently the CIP
``s - f``), split them into ``n`` portfolios of as-equal-as-possible size,
and hold each equal-weighted over the next month. ``HML = P_n - P_1``
(Lustig, Roussanov and Verdelhan 2011, section I.B; their six portfolios on
developed and emerging currencies).

``fama_macbeth``: each month regress the next-month excess returns on the
cross-sectionally standardised signal; the slope series' mean over its
standard error is the t-statistic (Fama and MacBeth 1973). ``nw_lag`` > 0
uses a Newey-West standard error on the slope series.

Worked example (``test_panel.py``): six currencies with signals 1..6 and
next-month returns equal to the signals in percent give HML with two
portfolios of 3: ``(4+5+6)/3 - (1+2+3)/3 = 3`` percent.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def lrv_sort(signal: pd.DataFrame, next_ret: pd.DataFrame, n_port: int = 6,
             live: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monthly portfolio returns P1..Pn and HML.

    ``signal`` and ``next_ret`` are month x currency; row m of ``next_ret`` is
    the return over the month after the sort date m. ``live`` (same shape,
    bool) restricts each month to the point-in-time universe.
    """
    rows = []
    for m in signal.index:
        s = signal.loc[m]
        r = next_ret.loc[m]
        ok = s.notna() & r.notna()
        if live is not None:
            ok &= live.loc[m].reindex(s.index).fillna(False).astype(bool)
        s, r = s[ok], r[ok]
        if len(s) < n_port:
            rows.append([np.nan] * (n_port + 1))
            continue
        order = s.sort_values(kind="mergesort").index
        groups = np.array_split(np.asarray(order), n_port)
        ports = [float(r[list(g)].mean()) for g in groups]
        rows.append([*ports, ports[-1] - ports[0]])
    cols = [f"P{i + 1}" for i in range(n_port)] + ["HML"]
    return pd.DataFrame(rows, index=signal.index, columns=cols)


def fama_macbeth(signal: pd.DataFrame, next_ret: pd.DataFrame,
                 live: pd.DataFrame | None = None, nw_lag: int = 0,
                 min_cross: int = 5) -> dict:
    """Monthly cross-sectional slopes of ``next_ret`` on the z-scored signal."""
    slopes = {}
    for m in signal.index:
        s, r = signal.loc[m], next_ret.loc[m]
        ok = s.notna() & r.notna()
        if live is not None:
            ok &= live.loc[m].reindex(s.index).fillna(False).astype(bool)
        if ok.sum() < min_cross:
            continue
        x = s[ok].to_numpy(dtype=float)
        sd = x.std(ddof=1)
        if not sd > 0:
            continue
        x = (x - x.mean()) / sd
        y = r[ok].to_numpy(dtype=float)
        slopes[m] = float(x @ (y - y.mean()) / (x @ x))
    b = pd.Series(slopes, dtype=float)
    t_count = len(b)
    if t_count < 3:
        return {"slopes": b, "mean": None, "se": None, "t": None, "months": t_count}
    d = b.to_numpy() - b.mean()
    lrv = float(d @ d) / t_count
    for k in range(1, nw_lag + 1):
        lrv += 2.0 * (1.0 - k / (nw_lag + 1.0)) * float(d[k:] @ d[:-k]) / t_count
    se = math.sqrt(lrv / (t_count - 1)) if nw_lag else float(b.std(ddof=1)) / math.sqrt(t_count)
    return {"slopes": b, "mean": float(b.mean()), "se": se, "t": float(b.mean()) / se,
            "months": t_count}


def correlation_check(ours: pd.Series, published: pd.Series) -> dict:
    """Acceptance numbers against a published HML series on overlapping months:
    correlation, mean difference and its standard error."""
    j = pd.concat([ours, published], axis=1, join="inner").dropna()
    if len(j) < 12:
        raise ValueError(f"only {len(j)} overlapping months")
    a, b = j.iloc[:, 0], j.iloc[:, 1]
    diff = a - b
    se = float(diff.std(ddof=1)) / math.sqrt(len(diff))
    return {"months": len(j), "corr": float(a.corr(b)), "mean_diff": float(diff.mean()),
            "mean_diff_se": se, "pass": bool(a.corr(b) >= 0.8 and abs(diff.mean()) <= 2 * se)}
