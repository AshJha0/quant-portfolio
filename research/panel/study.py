"""Run the carry / value / momentum panel on the point-in-time universe (W23 part 1).

Plain idea: for each signal, sort the currencies alive that month into
portfolios, hold them for a month, and measure the high-minus-low return and
the Fama-MacBeth slope. All of it runs behind the hold-out lock, so before a
new signed registration exists it can only see 1999-2015.

Formal statement. Inputs are month x currency frames: excess returns ``rx``
(row m = return over month m) and signals (row m = known at month-end m).
The next-month return used with signal row m is ``rx`` row m+1. The live mask
comes from ``research/lib/universe.Universe.mask``. For each signal the
result holds the LRV portfolio means, the HML mean, annualised Sharpe and
Newey-West t-statistic (lag 3, Bartlett), the Fama-MacBeth slope with its
NW t-statistic, the median live cross-section, and the power-gate verdict
for a panel of that size at the registration's IC and N_eff.

Labels: every run is ``exploratory`` unless the lock was opened by a signed
registration. A positive carry, value or momentum result is a known premium
(LRV 2011; AMP 2013; Menkhoff et al. 2012), not a discovery.

Worked example (``tests/test_panel_study.py``): six currencies whose
next-month return equals 1% times their signal rank give HML of 3% (two
portfolios) every month and a Fama-MacBeth t-statistic that is infinite in
the limit; with noise added the t-statistic stays far above 2.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))

import panel  # noqa: E402
import power_gate as pg  # noqa: E402
from holdout_lock import HoldoutLock  # noqa: E402

PREMIA = {
    "carry": "Lustig, Roussanov and Verdelhan (2011); Menkhoff et al. (2012, JF)",
    "value": "Asness, Moskowitz and Pedersen (2013); Menkhoff et al. (2017, RFS)",
    "momentum": "Menkhoff et al. (2012, JFE); Asness, Moskowitz and Pedersen (2013)",
}


def nw_tstat(x: pd.Series, lag: int = 3) -> float | None:
    """Mean over its Newey-West (Bartlett) standard error."""
    v = x.dropna().to_numpy(dtype=float)
    t = len(v)
    if t < lag + 3:
        return None
    d = v - v.mean()
    lrv = float(d @ d) / t
    for k in range(1, lag + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * float(d[k:] @ d[:-k]) / t
    if lrv <= 0:
        return None
    return float(v.mean() / math.sqrt(lrv / t))


def run_panel(rx: pd.DataFrame, signals: dict[str, pd.DataFrame], live: pd.DataFrame,
              lock: HoldoutLock | None = None, n_port: int = 5, nw_lag: int = 3,
              ic_monthly: float | None = None, n_eff: float | None = None,
              holdout_years: float = 11.0) -> dict:
    """Portfolio sorts and Fama-MacBeth slopes for each signal; see the module docstring."""
    lock = lock or HoldoutLock()
    lock.guard(rx, "excess returns")
    lock.guard(live, "universe mask")
    next_ret = rx.shift(-1)
    out: dict = {"label": "confirmatory" if lock.is_open else "exploratory",
                 "registration": lock.registration, "signals": {}}
    for name, sig in signals.items():
        lock.guard(sig, f"signal {name}")
        cols = [c for c in sig.columns if c in rx.columns]
        s, r = sig[cols].reindex(rx.index), next_ret[cols]
        lv = live.reindex(index=rx.index, columns=cols).fillna(False).astype(bool)
        ports = panel.lrv_sort(s, r, n_port=n_port, live=lv)
        hml = ports["HML"].dropna()
        fm = panel.fama_macbeth(s, r, live=lv, nw_lag=nw_lag)
        n_live = int(np.median((lv & s.notna() & r.notna()).sum(axis=1))) if len(lv) else 0
        gate = None
        if ic_monthly is not None and n_eff is not None:
            g = pg.gate("panel_fama_macbeth", n_currencies=n_live, ic_monthly=ic_monthly,
                        n_eff=n_eff, holdout_years=holdout_years)
            gate = {"verdict": g.verdict, "power": g.power, "reason": g.reason}
        sd = float(hml.std(ddof=1)) if len(hml) > 1 else float("nan")
        out["signals"][name] = {
            "months": len(hml),
            "portfolio_means": {k: float(v) for k, v in ports.drop(columns="HML").mean().items()},
            "hml_mean_monthly": float(hml.mean()) if len(hml) else None,
            "hml_sharpe_annual": float(hml.mean() / sd * math.sqrt(12)) if sd > 0 else None,
            "hml_t_nw": nw_tstat(hml, nw_lag),
            "fm_slope": fm["mean"], "fm_t": fm["t"], "fm_months": fm["months"],
            "median_live": n_live,
            "power_gate": gate,
            "known_premium": PREMIA.get(name),
        }
    return out
