"""Exposure profiles from Monte Carlo values, and a swap exposure simulator.

Plain idea
----------
Exposure is what the bank would lose if the counterparty defaulted at time
t with nothing recovered: the netting-set value if positive, else nothing.
Averaging that over simulated futures gives the expected exposure EE(t);
averaging the negative part gives the expected negative exposure ENE(t)
(what the counterparty is exposed to us, used for DVA); a high quantile
gives the potential future exposure PFE(t).

Formal statement
----------------
For path values V_p(t_k), p = 1..N, and pathwise discount factors D_p(t_k):

    EE(t_k)  = mean_p max(V_p(t_k), 0),      EE*(t_k)  = mean_p D_p max(V_p, 0),
    ENE(t_k) = mean_p min(V_p(t_k), 0),      ENE*(t_k) = mean_p D_p min(V_p, 0),
    PFE_q(t_k) = the ceil(q N)-th smallest V_p(t_k), floored at 0.

Basel effective EPE (CRE53.17-53.19): EEE(t_k) = max(EEE(t_{k-1}), EE(t_k)),
EEPE = sum_{t_k <= 1y} EEE(t_k) dt_k / min(1y, maturity).

The swap simulator values a single-curve vanilla swap at its own reset dates
(just after each fixing and payment), where the floating leg is worth par
and the swap value is 1 - P(T_j, T_n) - K sum_{i>j} tau_i P(T_j, T_i). At
those dates the discounted EE of a payer swap is exactly the price of a
payer swaption on the remaining swap: that is the semi-analytic oracle the
tests use.

Worked example
--------------
>>> import numpy as np
>>> v = np.array([[0.0, 2.0, -1.0], [0.0, -2.0, 3.0]])
>>> prof = profile(np.array([0.0, 0.5, 1.0]), v)
>>> prof.ee.tolist(), prof.ene.tolist()
([0.0, 1.0, 1.5], [0.0, -1.0, -0.5])

Sources: Gregory, *The xVA Challenge*, 4th ed. (2020), ch. 11 (exposure
metrics); BCBS CRE53 (effective EPE). Evidence class: definitions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import InputError
from .hull_white import HullWhite, HullWhitePaths

__all__ = ["ExposureProfile", "SwapSpec", "eepe", "pfe_index", "profile", "swap_exposure"]


def pfe_index(n: int, q: float) -> int:
    """Zero-based order-statistic index ceil(q n) - 1 used for PFE.

    >>> pfe_index(1000, 0.95)
    949
    """
    if n < 1 or not 0.0 < q < 1.0:
        raise InputError("need n >= 1 and 0 < q < 1")
    return max(math.ceil(q * n) - 1, 0)


@dataclass(frozen=True)
class ExposureProfile:
    times: np.ndarray
    ee: np.ndarray
    ene: np.ndarray
    ee_disc: np.ndarray
    ene_disc: np.ndarray
    pfe: np.ndarray
    ee_se: np.ndarray
    ee_disc_se: np.ndarray


def profile(
    times: np.ndarray, values: np.ndarray, disc: np.ndarray | None = None, *, q: float = 0.95
) -> ExposureProfile:
    """Exposure metrics from ``values[path, k]`` (and ``disc`` of the same shape;
    all ones if omitted). Standard errors are of the plain mean over paths."""
    v = np.asarray(values, dtype=float)
    t = np.asarray(times, dtype=float)
    if v.ndim != 2 or v.shape[1] != t.size or v.shape[0] < 1:
        raise InputError("values must be (paths, len(times))")
    d = np.ones_like(v) if disc is None else np.asarray(disc, dtype=float)
    if d.shape != v.shape:
        raise InputError("disc must have the shape of values")
    pos, neg = np.maximum(v, 0.0), np.minimum(v, 0.0)
    n = v.shape[0]
    se = (lambda a: a.std(axis=0, ddof=1) / math.sqrt(n)) if n > 1 else (lambda a: np.zeros(a.shape[1]))
    k = pfe_index(n, q)
    pfe = np.maximum(np.sort(v, axis=0)[k], 0.0)
    return ExposureProfile(
        t,
        pos.mean(axis=0),
        neg.mean(axis=0),
        (d * pos).mean(axis=0),
        (d * neg).mean(axis=0),
        pfe,
        se(pos),
        se(d * pos),
    )


def eepe(times: Sequence[float], ee: Sequence[float], maturity: float) -> float:
    """Effective EPE over the first year (CRE53.19), left-point rectangle rule:
    sum_{k>=1, t_k <= h} EEE(t_k) (t_k - t_{k-1}) / h with h = min(1, maturity).

    >>> eepe([0.0, 0.5, 1.0, 2.0], [0.0, 2.0, 1.0, 5.0], 3.0)
    2.0
    """
    t = [float(x) for x in times]
    if len(t) != len(ee) or len(t) < 2 or t[0] != 0.0:
        raise InputError("times must start at 0 and match ee")
    h = min(1.0, float(maturity))
    if not h > 0.0:
        raise InputError("maturity must be positive")
    eee, acc = 0.0, 0.0
    for k in range(1, len(t)):
        eee = max(eee, float(ee[k]))
        if t[k] <= h + 1e-12:
            acc += eee * (t[k] - t[k - 1])
    return acc / h


@dataclass(frozen=True)
class SwapSpec:
    """Single-curve vanilla swap, unit notional unless ``notional`` is set.

    ``dates`` are T_0 < T_1 < ... < T_n (years): T_0 the start, the rest the
    payment dates of both legs; fixed accruals are T_i - T_{i-1}. ``payer``
    pays fixed ``strike``.
    """

    dates: tuple[float, ...]
    strike: float
    payer: bool = True
    notional: float = 1.0

    def __post_init__(self) -> None:
        d = self.dates
        if len(d) < 2 or d[0] < 0.0 or any(b <= a for a, b in zip(d, d[1:], strict=False)):
            raise InputError("dates must be >= 0 and strictly increasing, at least two")

    @property
    def accruals(self) -> list[float]:
        return [b - a for a, b in zip(self.dates, self.dates[1:], strict=False)]

    def grid(self) -> list[float]:
        """0 followed by T_0..T_{n-1} (T_0 dropped if it is 0)."""
        return [0.0] + [t for t in self.dates[:-1] if t > 0.0]


def _swap_value(hw: HullWhite, spec: SwapSpec, t: float, x: np.ndarray) -> np.ndarray:
    """Swap value at a grid time t (t = 0 or t = some T_j) given x(t)."""
    d = spec.dates
    acc = spec.accruals
    if t == 0.0 and d[0] > 0.0:
        start_df = hw.p0(d[0]) * np.ones_like(x)
        idx = range(1, len(d))
    else:
        j = d.index(t)
        start_df = np.ones_like(x)
        idx = range(j + 1, len(d))
    fixed = np.zeros_like(x)
    for i in idx:
        fixed += acc[i - 1] * np.exp(hw.log_a(t, d[i]) - hw.b(t, d[i]) * x)
    end_df = np.exp(hw.log_a(t, d[-1]) - hw.b(t, d[-1]) * x) if t > 0.0 else hw.p0(d[-1]) * np.ones_like(x)
    if t == 0.0:
        fixed = np.full_like(x, sum(acc[i - 1] * hw.p0(d[i]) for i in idx))
    pay = start_df - end_df - spec.strike * fixed
    return spec.notional * (pay if spec.payer else -pay)


def swap_exposure(
    hw: HullWhite, spec: SwapSpec, n_paths: int, seed: int, *, q: float = 0.95
) -> tuple[ExposureProfile, HullWhitePaths, np.ndarray]:
    """Simulate ``hw`` on ``spec.grid()`` and return the exposure profile,
    the paths and the value matrix ``values[path, k]``."""
    paths = hw.simulate(spec.grid(), n_paths, seed)
    vals = np.empty_like(paths.x)
    for k, t in enumerate(paths.times):
        vals[:, k] = _swap_value(hw, spec, float(t), paths.x[:, k])
    return profile(paths.times, vals, paths.disc, q=q), paths, vals
