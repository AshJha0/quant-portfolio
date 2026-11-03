"""FX exposure hedging policies with transaction cost.

Plain idea
----------
A desk accumulates a foreign-currency exposure over time (client flow,
coupons, a foreign-asset book). Hedging all of it every day removes FX risk
but pays the bid-offer on every change; hedging nothing pays no cost but
keeps the risk. A threshold rule re-hedges only when the open amount grows
past a limit, which trades a little risk for a lot less cost.

Formal statement
----------------
Exposure ``E_t`` (foreign units) and spot ``X_t`` (domestic per foreign).
The hedge ``H_t`` is a short forward-like position in foreign units; carry
is ignored (or folded into ``X``). Open amount ``O_t = E_t - H_t``. Policies:

* ``full``: ``H_t = ratio * E_t`` every period;
* ``threshold``: if ``|E_t - H_{t-1}| > band`` (or ``|ratio E_t - H_{t-1}|``),
  set ``H_t = ratio * E_t``, else ``H_t = H_{t-1}``;
* ``none``: ``H_t = 0``.

Period P&L is ``O_{t-1} (X_t - X_{t-1}) - c |H_t - H_{t-1}| X_t`` with ``c``
the one-side proportional cost. The report gives the P&L standard deviation,
total cost and number of trades; ``pnl_total == fx_pnl - cost`` exactly.

Worked example
--------------
>>> import numpy as np
>>> from qp_core.hedging.fx import FxHedgePolicy, run_fx_hedge
>>> e = np.array([100.0, 100.0, 150.0, 150.0]); x = np.array([1.0, 1.1, 1.0, 1.2])
>>> r = run_fx_hedge(e, x, FxHedgePolicy("full", cost=0.001))
>>> round(r.fx_pnl, 9), round(r.cost, 6)
(0.0, 0.15)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qp_core.errors import InputError

__all__ = ["FxHedgePolicy", "FxHedgeReport", "run_fx_hedge"]

_KINDS = ("none", "full", "threshold")


@dataclass(frozen=True)
class FxHedgePolicy:
    """``kind`` in {"none", "full", "threshold"}; ``band`` in foreign units."""

    kind: str
    ratio: float = 1.0
    band: float = 0.0
    cost: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise InputError(f"kind must be one of {_KINDS}, got {self.kind!r}")
        if not (0.0 <= self.ratio <= 1.0):
            raise InputError(f"ratio must be in [0, 1], got {self.ratio}")
        if self.band < 0.0 or self.cost < 0.0:
            raise InputError("band and cost must be >= 0")


@dataclass(frozen=True)
class FxHedgeReport:
    """Totals in domestic currency plus the per-period P&L series."""

    pnl: npt.NDArray[np.float64]
    pnl_total: float
    fx_pnl: float
    cost: float
    pnl_std: float
    n_trades: int
    hedge: npt.NDArray[np.float64]


def run_fx_hedge(exposure: npt.ArrayLike, spot: npt.ArrayLike, policy: FxHedgePolicy) -> FxHedgeReport:
    """Apply ``policy`` to aligned exposure and spot series (length >= 2)."""
    e = np.asarray(exposure, dtype=np.float64)
    x = np.asarray(spot, dtype=np.float64)
    if e.ndim != 1 or e.shape != x.shape or e.size < 2:
        raise InputError("exposure and spot must be 1-D, same length, at least 2")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(x)) and np.all(x > 0.0)):
        raise InputError("exposure must be finite and spot finite and > 0")
    n = e.size
    h = np.zeros(n)
    trades = 0
    prev = 0.0
    for t in range(n):
        target = policy.ratio * e[t]
        if policy.kind == "full":
            new = target
        elif policy.kind == "threshold":
            new = target if abs(target - prev) > policy.band else prev
        else:
            new = 0.0
        if new != prev:
            trades += 1
        h[t] = new
        prev = new
    open_amt = e - h
    fx = open_amt[:-1] * np.diff(x)
    dh = np.abs(np.diff(np.concatenate([[0.0], h])))
    costs = policy.cost * dh * x
    pnl = fx - costs[1:]
    pnl[0] -= costs[0]
    fx_total = float(fx.sum())
    cost_total = float(costs.sum())
    return FxHedgeReport(
        pnl=pnl,
        pnl_total=float(pnl.sum()),
        fx_pnl=fx_total,
        cost=cost_total,
        pnl_std=float(pnl.std(ddof=1)) if pnl.size > 1 else 0.0,
        n_trades=trades,
        hedge=h,
    )
