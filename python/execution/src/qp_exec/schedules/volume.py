"""VWAP, TWAP and POV on a volume forecast, with catch-up and slow-down (W25d).

Plain idea
----------
A VWAP order should trade in proportion to the day's volume, but the day's
volume is only known afterwards. So the schedule is built from a forecast
curve (the average shape of past days), and during the day the algorithm
compares what it has done with what the forecast said, catching up when
behind and slowing down when ahead, never above a participation cap.

Formal statement
----------------
Curve: for a history matrix ``V[d, j]`` (day ``d``, bucket ``j``), the
forecast share of bucket ``j`` is the trimmed mean over days of
``V[d, j] / sum_j V[d, j]``, shrunk towards flat by weight ``lam``
(Bialkowski, Darolles and Le Fol 2008 decompose volume the same way; the
shrinkage is the plain James-Stein-style guard against a short history).

Adaptive child size in bucket ``j`` given executed ``E`` and realised market
volume so far::

    target_j = X * (cum forecast share through j)    (VWAP)
    child_j  = clip(target_j - E, 0, cap * V_j)

where ``cap`` is the participation limit, enforced live against the bucket's
prints ``V_j`` as they arrive. POV sets ``target_j = E + rate * Vhat_j``,
where ``Vhat_j`` is the forecast bucket volume rescaled by the ratio of
realised to forecast volume over the buckets already finished (the day is
running hot or cold). A close benchmark reserves
a fraction for the closing auction, sent as one auction order.

Worked example
--------------
>>> import numpy as np
>>> hist = np.tile([3.0, 1.0, 1.0, 3.0], (20, 1))
>>> c = calibrate_curve(hist, shrink=0.0)
>>> c.tolist()
[0.375, 0.125, 0.125, 0.375]
>>> run = execute_adaptive(1000, c, forecast_day_volume=8000,
...                        realised=np.array([3000, 1000, 1000, 3000.]), cap=0.2)
>>> run.child.tolist(), round(run.slippage_vs_vwap_bucket_weights, 12)
([375.0, 125.0, 125.0, 375.0], 0.0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def calibrate_curve(history: FloatArray, shrink: float = 0.1, trim: float = 0.1) -> FloatArray:
    """Intraday volume shares from ``history[day, bucket]`` (rows need not sum to 1)."""
    h = np.asarray(history, float)
    if h.ndim != 2 or np.any(h < 0) or np.any(h.sum(axis=1) <= 0):
        raise ValueError("history must be days x buckets, non-negative, each day > 0")
    shares = h / h.sum(axis=1, keepdims=True)
    k = int(np.floor(trim * shares.shape[0]))
    s = np.sort(shares, axis=0)
    core = s[k: shares.shape[0] - k] if shares.shape[0] - 2 * k > 0 else s
    curve = core.mean(axis=0)
    curve = curve / curve.sum()
    flat = np.full_like(curve, 1.0 / curve.size)
    out: FloatArray = (1 - shrink) * curve + shrink * flat
    return out


def twap_curve(n: int) -> FloatArray:
    return np.full(n, 1.0 / n)


@dataclass(frozen=True)
class AdaptiveRun:
    child: FloatArray  # executed per bucket
    participation: FloatArray  # child / realised volume
    close_qty: float  # sent to the closing auction
    shortfall: float  # parent not completed (cap binding)
    slippage_vs_vwap_bucket_weights: float  # sum |child/X - realised share| / 2


def execute_adaptive(parent: float, curve: FloatArray, forecast_day_volume: float,
                     realised: FloatArray, cap: float, mode: str = "vwap",
                     pov_rate: float = 0.1, close_fraction: float = 0.0,
                     close_volume: float = 0.0) -> AdaptiveRun:
    """Run the schedule bucket by bucket against a realised volume path.

    ``mode`` is ``"vwap"`` (target the forecast cumulative curve) or
    ``"pov"`` (target ``pov_rate`` times the rescaled forecast volume). The
    realised volume of bucket ``j`` is only revealed after bucket ``j`` is
    decided: the rescale factor uses buckets ``< j`` only (no look-ahead).
    """
    c = np.asarray(curve, float)
    v = np.asarray(realised, float)
    if c.shape != v.shape or not np.isclose(c.sum(), 1.0):
        raise ValueError("curve must sum to 1 and match realised")
    if not 0 < cap <= 1 or not 0 <= close_fraction < 1:
        raise ValueError("cap in (0, 1], close_fraction in [0, 1)")
    cont = parent * (1 - close_fraction)
    fc = forecast_day_volume * c
    cum = np.cumsum(c)
    child = np.zeros_like(v)
    done = 0.0
    for j in range(v.size):
        seen_f = fc[:j].sum()
        heat = v[:j].sum() / seen_f if j > 0 and seen_f > 0 else 1.0
        vhat = fc[j] * heat
        if mode == "vwap":
            target = cont * cum[j]
        elif mode == "pov":
            target = min(cont, done + pov_rate * vhat)
        else:
            raise ValueError("mode is 'vwap' or 'pov'")
        # Catch up when behind. The participation cap is enforced live
        # against prints as they arrive (cap * v[j]); the size wanted is
        # decided from the forecast and past buckets only.
        want = max(target - done, 0.0)
        q = min(want, cap * v[j], cont - done)
        child[j] = q
        done += q
    close_qty = min(parent * close_fraction, cap * close_volume) if close_fraction else 0.0
    shortfall = parent - done - close_qty
    with np.errstate(divide="ignore", invalid="ignore"):
        part = np.where(v > 0, child / v, 0.0)
    w = child / max(done, 1e-300)
    slip = 0.5 * float(np.abs(w - v / v.sum()).sum())
    return AdaptiveRun(child, part, float(close_qty), float(max(shortfall, 0.0)), slip)
