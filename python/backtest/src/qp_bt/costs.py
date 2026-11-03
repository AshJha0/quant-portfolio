"""Cost models with named sources.

A weight-mode cost model is a callable ``cost(decision_bar, dv) -> float``:
``dv`` is the signed traded notional of each asset as a fraction of the
period's opening NAV, and the result is the cost in the same units. The core
adds the fill model's ``extra_cost_bp`` on top.

Every model carries ``source``: where its numbers come from and their
evidence class (MEASURED, PUBLISHED, ASSUMPTION), so a ledger can say why a
cost is what it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["ZERO_COST", "CostModel", "HalfSpreadCost", "LinearCost", "SqrtImpactCost"]

FloatArray = NDArray[np.float64]


class CostModel(Protocol):
    """Cost of a trade, as a fraction of the period's opening NAV."""

    def __call__(self, decision_bar: int, dv: FloatArray, /) -> float: ...


@dataclass(frozen=True)
class LinearCost:
    """``bps * 1e-4 * sum |dv|``: commission, fees and half-spread.

    Worked example: 5 bp on a 0.6 of NAV round of buys and sells costs
    ``5e-4 * 0.6 = 3e-4`` of NAV.
    """

    bps: float
    source: str = "ASSUMPTION: flat all-in cost per unit traded"

    def __post_init__(self) -> None:
        if not math.isfinite(self.bps) or self.bps < 0:
            raise ValueError(f"bps must be finite and >= 0, got {self.bps!r}")

    def __call__(self, decision_bar: int, dv: FloatArray) -> float:
        return float(self.bps * 1e-4 * np.abs(dv).sum())


@dataclass(frozen=True)
class HalfSpreadCost:
    """``sum_i |dv_i| * rate_i``: a per-asset (and optionally per-bar) rate.

    ``rate`` is a fraction of traded notional: shape ``(N,)`` for a fixed
    rate per asset, or ``(T, N)`` with row ``decision_bar`` used, so a cost
    quoted in pips can follow the price level (``pips * pip_size / spot``).

    Worked example: rates ``(1e-4, 4e-4)`` on trades ``(0.5, -0.25)`` cost
    ``0.5e-4 + 1e-4 = 1.5e-4`` of NAV.
    """

    rate: FloatArray
    source: str = "ASSUMPTION: half the quoted bid-ask spread per unit traded"

    def __post_init__(self) -> None:
        r = np.asarray(self.rate, dtype=float)
        if r.ndim not in (1, 2) or not np.isfinite(r).all() or (r < 0).any():
            raise ValueError("rate must be a finite, non-negative (N,) or (T, N) array")
        object.__setattr__(self, "rate", r)

    def __call__(self, decision_bar: int, dv: FloatArray) -> float:
        r = self.rate if self.rate.ndim == 1 else self.rate[decision_bar]
        return float(np.abs(dv) @ r)


@dataclass(frozen=True)
class SqrtImpactCost:
    """Square-root impact ``Y * sigma_i * sqrt(aum |dv_i| / ADV$_i) * |dv_i|``.

    ``sigma`` and ``adv_dollars`` are (T, N) arrays of estimates known at the
    decision close (row ``decision_bar`` is used). A non-finite or non-positive
    ADV$ is treated as infinite depth (zero impact); a NaN sigma as zero.

    Source: Almgren, Thum, Hauptmann and Li (2005); Toth et al. (2011),
    PUBLISHED; the coefficient ``Y`` in 0.5-1 is their range.
    """

    coef: float
    aum: float
    sigma: FloatArray
    adv_dollars: FloatArray
    source: str = "PUBLISHED: Almgren et al. (2005), Toth et al. (2011)"

    def __call__(self, decision_bar: int, dv: FloatArray) -> float:
        absw = np.abs(dv)
        sig = np.nan_to_num(self.sigma[decision_bar], nan=0.0)
        adv = self.adv_dollars[decision_bar]
        adv = np.where(np.isfinite(adv) & (adv > 0), adv, np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            part = np.where(absw > 0, self.aum * absw / adv, 0.0)
        return float(np.nansum(self.coef * sig * np.sqrt(part) * absw))


def _zero(decision_bar: int, dv: FloatArray) -> float:
    return 0.0


ZERO_COST: CostModel = _zero
