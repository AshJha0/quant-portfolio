"""Bucketed vega on a volatility grid (expiry x strike, expiry x tenor x strike).

The plain idea
--------------
One "vega" number says how a book moves if every implied vol rises by the
same amount. A desk hedges with options at particular expiries and strikes,
so it needs the vega split by bucket: how much of the risk sits at 1y 25-delta
puts, how much at 5y-into-10y ATM swaptions. A vega grid bumps one node of
the vol grid at a time and reprices the book.

The formal statement
--------------------
The grid holds node vols ``s[i, j(, k)]`` on sorted axes. The vol at an
arbitrary point is multilinear in the node vols with flat extrapolation
beyond the first and last node on each axis, so the interpolation weights at
any point are non-negative and sum to 1 (a partition of unity). For a book
value ``P(s)`` the bucket vega is the central difference::

    V[i, j(, k)] = (P(s + h e_ijk) - P(s - h e_ijk)) / (2 h)

in value per unit of vol (multiply by 0.01 for "per vol point"). Because the
weights are a partition of unity and ``P`` depends on ``s`` only through the
interpolated vols, ``sum V = dP/dh`` for a parallel shift of every node, up
to the ``O(h^2)`` error of the differences. BCBS MAR21.25 defines vega
sensitivity per bucket in the same way (a bump of the implied vol at the
bucket's option maturity, and for rates also the underlying residual
maturity).

Multilinear interpolation in vol is a bucketing device, not a smile model:
it is neither smooth nor arbitrage-free. Pricing a book off this grid is
fine for risk aggregation; a pricing surface (SVI, SABR) should be bucketed
by passing its own repricing function.

Worked example
--------------
A single option at expiry 1.5y on a grid with expiries ``(1, 2)`` and its
strike on a node gets half of its vega in the 1y bucket and half in the 2y
bucket; the two buckets sum to the option's vega. ``tests/test_risk_vega.py``
checks this, the 3-D cube case, and the sum rule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from qp_core.model import model

__all__ = ["VolGrid", "vega_grid"]


@dataclass(frozen=True)
class VolGrid:
    """Node vols on a rectangular grid with named axes.

    Parameters
    ----------
    axes : sequence of 1-D arrays
        Strictly increasing coordinates, one per dimension (2 or 3). Typical
        names are ``("expiry", "strike")`` and ``("expiry", "tenor", "strike")``.
    vols : array
        Node vols, shape ``tuple(len(a) for a in axes)``; finite and >= 0.
    names : sequence of str
        One label per axis (used in reports only).
    """

    axes: tuple[np.ndarray, ...]
    vols: np.ndarray
    names: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        axes = tuple(np.asarray(a, dtype=float) for a in self.axes)
        vols = np.asarray(self.vols, dtype=float)
        if len(axes) not in (2, 3):
            raise ValueError(f"a vol grid has 2 or 3 axes, got {len(axes)}")
        for k, a in enumerate(axes):
            if a.ndim != 1 or a.size == 0:
                raise ValueError(f"axis {k} must be a non-empty 1-D array")
            if np.any(np.diff(a) <= 0) or not np.all(np.isfinite(a)):
                raise ValueError(f"axis {k} must be finite and strictly increasing")
        if vols.shape != tuple(a.size for a in axes):
            raise ValueError(f"vols shape {vols.shape} does not match axes {[a.size for a in axes]}")
        if not np.all(np.isfinite(vols)) or np.any(vols < 0):
            raise ValueError("node vols must be finite and >= 0")
        names = tuple(self.names) or tuple(f"axis{k}" for k in range(len(axes)))
        if len(names) != len(axes):
            raise ValueError("one name per axis")
        vols.setflags(write=False)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "vols", vols)
        object.__setattr__(self, "names", names)

    def weights(self, *coords: float) -> np.ndarray:
        """Interpolation weights over the nodes at ``coords`` (sum to 1)."""
        if len(coords) != len(self.axes):
            raise ValueError(f"expected {len(self.axes)} coordinates, got {len(coords)}")
        w = np.ones(())
        for a, x in zip(self.axes, coords, strict=True):
            wa = np.zeros(a.size)
            if a.size == 1 or x <= a[0]:
                wa[0] = 1.0
            elif x >= a[-1]:
                wa[-1] = 1.0
            else:
                i = int(np.searchsorted(a, x, side="right")) - 1
                t = (x - a[i]) / (a[i + 1] - a[i])
                wa[i], wa[i + 1] = 1.0 - t, t
            w = np.multiply.outer(w, wa)
        return w

    def vol(self, *coords: float) -> float:
        """Multilinear vol at ``coords`` with flat extrapolation."""
        return float(np.sum(self.weights(*coords) * self.vols))

    def with_vols(self, vols: np.ndarray) -> VolGrid:
        return VolGrid(self.axes, vols, self.names)

    def bumped(self, index: tuple[int, ...], h: float) -> VolGrid:
        v = np.array(self.vols)
        v[index] += h
        return self.with_vols(v)

    def shifted(self, h: float) -> VolGrid:
        return self.with_vols(np.array(self.vols) + h)


@model(id="CORE-RISK-VEGA-GRID")
def vega_grid(
    pv: Callable[[VolGrid], float], grid: VolGrid, h: float = 1e-4
) -> np.ndarray:
    """Bucket vegas (value per unit vol), same shape as ``grid.vols``.

    ``h`` must be below the smallest node vol so the down bump stays >= 0;
    a node with vol below ``h`` is bumped one-sided (forward difference) and
    that is the only place a first-order error enters.
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"bump must be > 0, got {h}")
    out = np.zeros(grid.vols.shape)
    for idx in np.ndindex(grid.vols.shape):
        up = pv(grid.bumped(idx, h))
        if grid.vols[idx] >= h:
            out[idx] = (up - pv(grid.bumped(idx, -h))) / (2.0 * h)
        else:
            out[idx] = (up - pv(grid)) / h
    return out
