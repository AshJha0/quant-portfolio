"""Target redemption forward (TARF) by Monte Carlo.

Plain idea
----------
A TARF is a strip of forwards that dies once the client has earned a
target amount. The client buys (or sells) the notional at a better-than-
market strike at each fixing; when the fixing goes against them, the loss
is on a larger (leveraged) notional. Gains count towards the target;
losses do not. The structure is path-dependent and has no closed form.

Formal statement
----------------
Client long the base currency (a "buyer" TARF, as in Wystup 2017, *FX
Options and Structured Products*, 2nd ed., s.3.4). At fixing ``t_i``,
with forward-start notional ``N`` and leverage ``l``:

    gain_i = N max(S_i - K, 0),     loss_i = l N max(K - S_i, 0)

The accumulated gain ``G_i = sum_{j<=i} gain_j``. At the first ``i`` with
``G_i >= target`` the trade terminates; the final gain paid is the full
``gain_i`` (``"full"``), only ``target - G_{i-1}`` (``"capped"``), or
nothing (``"none"``). Each cash flow is discounted at ``r_d`` from its
fixing date (settlement lag ignored).

Worked example
--------------
With an unreachable target and leverage 1 a TARF is a strip of forwards:

>>> import math
>>> t = [0.25, 0.5, 0.75, 1.0]
>>> v, se = tarf_mc(1.10, 1.08, t, 0.03, 0.01, 0.08, 1e9, n_paths=20000, seed=1)
>>> exact = sum(math.exp(-0.03 * s) * (1.10 * math.exp(0.02 * s) - 1.08) for s in t)
>>> abs(v - exact) < 4 * se
True
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from qp_core.model import model

__all__ = ["tarf_mc"]


@model(id="CORE-EXO-TARF")
def tarf_mc(
    S: float, K: float, fixings: Sequence[float], rd: float, rf: float, sigma: float,
    target: float, leverage: float = 1.0, notional: float = 1.0, final: str = "full",
    n_paths: int = 100_000, seed: int = 0,
) -> tuple[float, float]:
    """Client-side value of a buyer TARF under GK dynamics; returns ``(value, se)``."""
    t = np.asarray(fixings, dtype=float)
    if t.ndim != 1 or t.size == 0 or np.any(t <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("fixings must be positive and increasing")
    if not (S > 0 and K > 0 and sigma > 0 and target > 0 and leverage >= 0 and notional > 0):
        raise ValueError("S, K, sigma, target, notional must be > 0 and leverage >= 0")
    if final not in ("full", "capped", "none"):
        raise ValueError("final must be 'full', 'capped' or 'none'")
    rng = np.random.default_rng(seed)
    dt = np.diff(np.concatenate([[0.0], t]))
    x = np.full(n_paths, math.log(S))
    alive = np.ones(n_paths, dtype=bool)
    acc = np.zeros(n_paths)
    pv = np.zeros(n_paths)
    for i in range(t.size):
        x = x + (rd - rf - 0.5 * sigma * sigma) * dt[i] + sigma * math.sqrt(dt[i]) * rng.standard_normal(n_paths)
        s = np.exp(x)
        gain = notional * np.maximum(s - K, 0.0)
        loss = leverage * notional * np.maximum(K - s, 0.0)
        hit = alive & (acc + gain >= target)
        pay_gain = gain.copy()
        if final == "capped":
            pay_gain = np.where(hit, target - acc, gain)
        elif final == "none":
            pay_gain = np.where(hit, 0.0, gain)
        flow = np.where(alive, pay_gain - loss, 0.0)
        pv += math.exp(-rd * t[i]) * flow
        acc = np.where(alive, acc + gain, acc)
        alive &= ~hit
    return float(pv.mean()), float(pv.std(ddof=1) / math.sqrt(n_paths))
