"""Key-rate durations via localized triangular zero-curve bumps.

Method
------
For key tenor ``k`` with neighbours ``k_prev < k < k_next`` (the standard
2s/5s/10s/30s set by default, configurable), the bump applied to the curve's
zero rate at time ``t`` is a triangle::

    w(t) = 1                      t == k
    w(t) = (t - k_prev)/(k - k_prev)   k_prev <= t <= k
    w(t) = (k_next - t)/(k_next - k)   k <= t <= k_next
    w(t) = 0                      outside

The first key tenor extends flat to ``t = 0`` and the last extends flat to
infinity, so the triangle weights across all key tenors sum to exactly 1 at
every ``t`` — a partition of unity.  KRD_k is the central-difference
sensitivity of the PV to bumping the *curve pillars* by ``w(t_pillar) * h``.

Because the weights are a partition of unity, the sum of the key-rate bumps
equals a parallel bump of the pillar zeros, so::

    sum_k KRDV01_k == parallel DV01     (exactly, up to the tiny
                                         non-additivity of finite differences
                                         under non-linear repricing)

The residual is second-order in the bump size ``h`` (cross-gamma between key
rates); with ``h = 1bp`` it is far below 1e-6 of PV — measured and documented
in ``docs/VALIDATION.md``.  Under non-linear interpolation (PCHIP) each
pillar bump also perturbs neighbouring segments, which is why the match is
"within tolerance" rather than an algebraic identity.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .bond import FixedRateBond, dirty_price_from_curve
from .curve import DiscountCurve
from .risk import Position, portfolio_value

__all__ = [
    "DEFAULT_KEY_TENORS",
    "triangle_weights",
    "key_rate_dv01s",
    "key_rate_durations",
    "key_rate_convexities",
    "krd_report",
]

DEFAULT_KEY_TENORS: tuple[float, ...] = (2.0, 5.0, 10.0, 30.0)

PVFunc = Callable[[DiscountCurve], float]


def triangle_weights(
    key_tenors: Sequence[float], times: np.ndarray
) -> np.ndarray:
    """Matrix ``W[k, j]`` of triangular weights of key tenor ``k`` at
    ``times[j]``.  Columns sum to 1 (partition of unity)."""
    ks = np.asarray(key_tenors, dtype=float)
    if ks.ndim != 1 or ks.size == 0:
        raise ValueError("key_tenors must be a non-empty 1-D sequence")
    if np.any(np.diff(ks) <= 0):
        raise ValueError("key_tenors must be strictly increasing")
    t = np.asarray(times, dtype=float)
    n = ks.size
    w = np.zeros((n, t.size))
    for i, k in enumerate(ks):
        left = ks[i - 1] if i > 0 else None
        right = ks[i + 1] if i < n - 1 else None
        wi = np.zeros_like(t)
        # flat extension on the outer sides
        if left is None:
            wi = np.where(t <= k, 1.0, wi)
        else:
            up = (t - left) / (k - left)
            wi = np.where((t > left) & (t <= k), up, wi)
        if right is None:
            wi = np.where(t > k, 1.0, wi)
        else:
            down = (right - t) / (right - k)
            wi = np.where((t > k) & (t < right), down, wi)
        w[i] = wi
    return w


def _pv_func(
    target: FixedRateBond | list[Position],
    settlement: dt.date,
    z_spread: float,
) -> PVFunc:
    if isinstance(target, FixedRateBond):
        return lambda c: dirty_price_from_curve(target, settlement, c, z_spread)
    return lambda c: portfolio_value(target, settlement, c)


def key_rate_dv01s(
    target: FixedRateBond | list[Position],
    settlement: dt.date,
    curve: DiscountCurve,
    key_tenors: Sequence[float] = DEFAULT_KEY_TENORS,
    z_spread: float = 0.0,
    bump: float = 1e-4,
) -> np.ndarray:
    """Key-rate DV01s: PV gain for a 1bp *fall* localized at each key tenor.

    Central differences of full revaluation under triangular pillar bumps.
    ``target`` is a bond (with optional z-spread) or a list of positions.
    """
    pv = _pv_func(target, settlement, z_spread)
    w = triangle_weights(key_tenors, curve.times)
    out = np.empty(w.shape[0])
    for i in range(w.shape[0]):
        shift = w[i] * bump
        up = pv(curve.bumped_pillars(shift))
        dn = pv(curve.bumped_pillars(-shift))
        out[i] = (dn - up) / 2.0 * (1e-4 / bump)
    return out


def key_rate_durations(
    target: FixedRateBond | list[Position],
    settlement: dt.date,
    curve: DiscountCurve,
    key_tenors: Sequence[float] = DEFAULT_KEY_TENORS,
    z_spread: float = 0.0,
    bump: float = 1e-4,
) -> np.ndarray:
    """Key-rate durations: ``KRDV01_k / (PV * 1e-4)`` (years per unit yield).

    Sums approximately to the parallel effective duration (see module docs).
    """
    pv0 = _pv_func(target, settlement, z_spread)(curve)
    if pv0 == 0.0:
        raise ValueError("target has zero PV; key-rate durations undefined")
    return key_rate_dv01s(
        target, settlement, curve, key_tenors, z_spread, bump
    ) / (pv0 * 1e-4)


def key_rate_convexities(
    target: FixedRateBond | list[Position],
    settlement: dt.date,
    curve: DiscountCurve,
    key_tenors: Sequence[float] = DEFAULT_KEY_TENORS,
    z_spread: float = 0.0,
    bump: float = 1e-4,
) -> np.ndarray:
    """Diagonal key-rate convexities ``(1/PV) d2PV/dz_k^2`` (optional extra)."""
    pv = _pv_func(target, settlement, z_spread)
    pv0 = pv(curve)
    if pv0 == 0.0:
        raise ValueError("target has zero PV; key-rate convexity undefined")
    w = triangle_weights(key_tenors, curve.times)
    out = np.empty(w.shape[0])
    for i in range(w.shape[0]):
        shift = w[i] * bump
        up = pv(curve.bumped_pillars(shift))
        dn = pv(curve.bumped_pillars(-shift))
        out[i] = (up - 2.0 * pv0 + dn) / (bump * bump) / pv0
    return out


def krd_report(
    target: FixedRateBond | list[Position],
    settlement: dt.date,
    curve: DiscountCurve,
    key_tenors: Sequence[float] = DEFAULT_KEY_TENORS,
    z_spread: float = 0.0,
) -> pd.DataFrame:
    """KRD ladder DataFrame: per-tenor DV01 and duration + SUM row."""
    dv01s = key_rate_dv01s(target, settlement, curve, key_tenors, z_spread)
    krds = key_rate_durations(target, settlement, curve, key_tenors, z_spread)
    df = pd.DataFrame(
        {"key_rate_dv01": dv01s, "key_rate_duration": krds},
        index=[f"{k:g}y" for k in key_tenors],
    )
    df.loc["SUM"] = [dv01s.sum(), krds.sum()]
    return df
