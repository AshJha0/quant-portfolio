"""Out-of-sample hedge test on a daily index series with an implied-vol proxy.

Plain idea
----------
Cut a daily price history into non-overlapping windows of ``window`` days.
In each window, sell an at-the-money call that expires at the window end,
price it with the volatility proxy observed on the first day, and hedge it
daily with deltas computed from the proxy observed each day. Each rule
(time, band, Leland) sees the same windows, so their error variance and cost
can be compared, with standard errors from the spread across windows.

The volatility proxy (for SPX, the VIX index divided by 100) is **not an
option price**: VIX is a 30-day variance-swap rate, so premiums priced from
it are model premiums, and the test measures hedging error relative to
that model, not against traded option prices.

Formal statement
----------------
For windows ``w = 1..W`` and rule ``R``, ``PnL_w^R`` is the
``hedge_short_option`` P&L of one ATM call (``K = S_{w,0}``, ``T = window /
252``). Reported per rule: ``Var(PnL^R)`` with its SE (from the fourth
central moment, ``sqrt((m4 - s^4)/W)``), mean cost with SE, and mean trades.

Worked example
--------------
>>> import numpy as np
>>> from qp_core.hedging.backtest import oos_hedge_test
>>> from qp_core.hedging.simulate import gbm_paths
>>> s = gbm_paths(100.0, 0.0, 0.2, 2.0, 504, 1, seed=3)[0]
>>> rows = oos_hedge_test(s, np.full(s.size, 0.2), window=21, cost=0.0005)
>>> sorted(rows)
['band', 'leland', 'time']
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qp_core.errors import InputError
from qp_core.hedging.policies import HedgeRule, HedgeRuleSpec
from qp_core.hedging.simulate import hedge_short_option

__all__ = ["VOL_PROXY_LABEL", "RuleStats", "oos_hedge_test"]

VOL_PROXY_LABEL = "VIX/100 used as a volatility proxy; VIX is not an option price"


@dataclass(frozen=True)
class RuleStats:
    """Across-window statistics for one rule."""

    n_windows: int
    error_var: float
    error_var_se: float
    mean_pnl: float
    mean_pnl_se: float
    mean_cost: float
    mean_cost_se: float
    mean_trades: float
    label: str = VOL_PROXY_LABEL


def _var_se(x: npt.NDArray[np.float64]) -> tuple[float, float]:
    n = x.size
    d = x - x.mean()
    s2 = float((d**2).mean())
    m4 = float((d**4).mean())
    return float(x.var(ddof=1)), math.sqrt(max(m4 - s2 * s2, 0.0) / n)


def oos_hedge_test(
    spot: npt.ArrayLike,
    vol_proxy: npt.ArrayLike,
    *,
    window: int = 21,
    cost: float = 0.0005,
    risk_aversion: float = 1.0,
    rate: float = 0.0,
    days_per_year: float = 252.0,
) -> dict[str, RuleStats]:
    """Run the time, band and Leland rules over non-overlapping windows.

    ``spot`` and ``vol_proxy`` are aligned daily arrays; ``vol_proxy`` is an
    annualised decimal volatility (VIX / 100). Needs at least ten windows.
    """
    s = np.asarray(spot, dtype=np.float64)
    v = np.asarray(vol_proxy, dtype=np.float64)
    if s.ndim != 1 or s.shape != v.shape:
        raise InputError("spot and vol_proxy must be aligned 1-D arrays")
    if not (np.all(np.isfinite(s)) and np.all(s > 0.0) and np.all(np.isfinite(v)) and np.all(v > 0.0)):
        raise InputError("spot and vol_proxy must be finite and > 0")
    if window < 2:
        raise InputError("window must be >= 2")
    n_win = (s.size - 1) // window
    if n_win < 10:
        raise InputError(f"need at least 10 windows of {window} days, have {n_win}")
    paths = np.stack([s[w * window : w * window + window + 1] for w in range(n_win)])
    vols = np.stack([v[w * window : w * window + window] for w in range(n_win)])
    maturity = window / days_per_year
    out: dict[str, RuleStats] = {}
    for rule in (HedgeRule.TIME, HedgeRule.BAND, HedgeRule.LELAND):
        spec = HedgeRuleSpec(rule, cost=cost, risk_aversion=risk_aversion)
        pnls = np.empty(n_win)
        costs = np.empty(n_win)
        trades = np.empty(n_win)
        for w in range(n_win):
            # Each window is priced at its own day-0 proxy, so it runs alone.
            res = hedge_short_option(
                paths[w : w + 1] / paths[w, 0] * 100.0,
                100.0,
                maturity,
                rate,
                float(vols[w, 0]),
                True,
                spec,
                hedge_sigma=vols[w : w + 1],
            )
            pnls[w] = res.pnl[0]
            costs[w] = res.cost[0]
            trades[w] = res.trades[0]
        var, var_se = _var_se(pnls)
        out[rule.value] = RuleStats(
            n_windows=n_win,
            error_var=var,
            error_var_se=var_se,
            mean_pnl=float(pnls.mean()),
            mean_pnl_se=float(pnls.std(ddof=1) / math.sqrt(n_win)),
            mean_cost=float(costs.mean()),
            mean_cost_se=float(costs.std(ddof=1) / math.sqrt(n_win)),
            mean_trades=float(trades.mean()),
        )
    return out
