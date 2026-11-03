"""Collateralised exposure: variation margin with a margin period of risk,
thresholds, minimum transfer amounts and initial margin.

Plain idea
----------
Under a CSA the counterparty posts cash as the trade moves in our favour, so
if it defaulted we would only lose what moved *since the last collateral we
actually received*. Three things leave a gap. The margin period of risk
(MPOR): from the last good margin call to closing out the trade takes days,
and the value keeps moving. The threshold H: the counterparty only posts
above an agreed amount. The minimum transfer amount (MTA): calls smaller
than this are not made. Initial margin (IM), posted up front and held
segregated, is a cushion against that gap.

Formal statement
----------------
On a grid t_0 < ... < t_n (each grid date is a margin call date), with path
values V(t) from the bank's side:

* Target VM (two-way CSA): T(V) = max(V - H_c, 0) - max(-V - H_b, 0), where
  H_c is the counterparty's threshold and H_b the bank's; positive means the
  bank holds collateral.
* Balance with MTA (path-dependent): B(t_0) = ``initial``;
  B(t_k) = T(V(t_k)) if |T(V(t_k)) - B(t_{k-1})| >= MTA, else B(t_{k-1}).
* MPOR: collateral available at t is C(t) = B(s*), s* the latest grid date
  with s* <= t - MPOR (B(t_0) if none). Put the lagged dates on the grid
  (``lagged_grid``) to make the lag exact.
* Exposure: E(t) = max(V(t) - C(t) - IM_r(t), 0); negative exposure
  N(t) = min(V(t) - C(t) + IM_p(t), 0). Segregated IM received reduces our
  exposure; IM posted is bankruptcy-remote and only cushions theirs
  (Gregory 2020, ch. 8-9; BCBS-IOSCO margin framework 2020).

Closed forms used as test oracles, for V a driftless Brownian motion with
volatility sigma, MPOR delta and dV = V(t) - V(t - delta) ~ N(0, s^2),
s = sigma sqrt(delta):

* H = 0, MTA = 0, no IM: EE = s phi(0) = s / sqrt(2 pi).
* Same with IM = k s: EE = s [phi(k) - k (1 - Phi(k))].
* MPOR = 0, threshold H, V ~ N(0, v^2): EE = E[min(V^+, H)]
  = v [phi(0) - phi(H/v)] + H (1 - Phi(H/v)).
* Limits: H = infinity gives the uncollateralised exposure; MPOR = 0, H = 0,
  MTA = 0 gives zero exposure; MPOR = 0, H = 0 gives 0 <= E <= MTA pathwise.

Worked example
--------------
A path 0 -> 10 -> 4 with H_c = 3, no MTA, no lag: target collateral is 7
then 1; exposures 0, 3, 3 - each capped at the threshold.

>>> import numpy as np
>>> e = collateralised_exposure(np.array([0.0, 1.0, 2.0]), np.array([[0.0, 10.0, 4.0]]), Csa(threshold_cpty=3.0))
>>> e.exposure.tolist()
[[0.0, 3.0, 3.0]]

Sources: Gregory, *The xVA Challenge*, 4th ed. (2020), ch. 8 (collateral)
and ch. 11.3 (the sigma sqrt(MPOR) phi(0) approximation); BCBS CRE53.23-53.26
(MPOR floors of 10 and 20 business days). Evidence class: definitions plus
derivations checked by the tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import InputError
from .exposure import ExposureProfile, profile

__all__ = [
    "CollateralisedExposure",
    "Csa",
    "brownian_ee_im",
    "brownian_ee_mpor",
    "brownian_ee_threshold",
    "collateral_balance",
    "collateralised_exposure",
    "collateralised_profile",
    "lagged_grid",
    "target_vm",
]

_EPS = 1e-12


@dataclass(frozen=True)
class Csa:
    """CSA terms. ``mpor`` in the grid's time unit (years: 10 business days
    is about 10/250). ``im_received``/``im_posted`` are scalars or arrays
    broadcastable to (paths, times). ``initial`` is the balance at t_0."""

    threshold_cpty: float = 0.0
    threshold_bank: float = 0.0
    mta: float = 0.0
    mpor: float = 0.0
    im_received: float | np.ndarray = 0.0
    im_posted: float | np.ndarray = 0.0
    initial: float = 0.0

    def __post_init__(self) -> None:
        if min(self.threshold_cpty, self.threshold_bank, self.mta, self.mpor) < 0.0:
            raise InputError("thresholds, MTA and MPOR must be non-negative")


def target_vm(v: np.ndarray, csa: Csa) -> np.ndarray:
    """Two-way target collateral T(V) (positive: the bank holds it)."""
    return np.maximum(v - csa.threshold_cpty, 0.0) - np.maximum(-v - csa.threshold_bank, 0.0)


def collateral_balance(values: np.ndarray, csa: Csa) -> np.ndarray:
    """Balance B(t_k) per path after each margin call, with the MTA rule."""
    v = np.asarray(values, dtype=float)
    tgt = target_vm(v, csa)
    b = np.empty_like(v)
    cur = np.full(v.shape[0], float(csa.initial))
    for k in range(v.shape[1]):
        move = np.abs(tgt[:, k] - cur) >= csa.mta if csa.mta > 0.0 else np.ones(v.shape[0], bool)
        cur = np.where(move, tgt[:, k], cur)
        b[:, k] = cur
    return b


def lagged_grid(times: Sequence[float], mpor: float) -> np.ndarray:
    """``times`` with every t - mpor >= t_0 added, sorted and de-duplicated,
    so the MPOR lag lands exactly on grid dates.

    >>> lagged_grid([0.0, 1.0, 2.0], 0.25).tolist()
    [0.0, 0.75, 1.0, 1.75, 2.0]
    """
    t = np.asarray(times, dtype=float)
    lag = t - mpor
    allt = np.concatenate([t, lag[lag >= t[0] - _EPS]])
    allt = np.unique(np.round(allt, 12))
    return allt


def _lag_index(times: np.ndarray, mpor: float) -> np.ndarray:
    """Index of the latest grid date <= t - mpor, or 0 if none."""
    idx = np.searchsorted(times, times - mpor + _EPS, side="right") - 1
    return np.maximum(idx, 0)


@dataclass(frozen=True)
class CollateralisedExposure:
    times: np.ndarray
    values: np.ndarray
    collateral: np.ndarray
    exposure: np.ndarray
    negative_exposure: np.ndarray


def collateralised_exposure(times: np.ndarray, values: np.ndarray, csa: Csa) -> CollateralisedExposure:
    """Pathwise collateral C(t), exposure E(t) and negative exposure N(t)."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    if v.ndim != 2 or v.shape[1] != t.size or t.size < 1:
        raise InputError("values must be (paths, len(times))")
    if np.any(np.diff(t) <= 0.0):
        raise InputError("times must be strictly increasing")
    bal = collateral_balance(v, csa)
    if csa.mpor > 0.0:
        lagged = _lag_index(t, csa.mpor)
        before = t - csa.mpor < t[0] - _EPS
        c = bal[:, lagged]
        c[:, before] = csa.initial
    else:
        c = bal
    im_r = np.broadcast_to(np.asarray(csa.im_received, dtype=float), v.shape)
    im_p = np.broadcast_to(np.asarray(csa.im_posted, dtype=float), v.shape)
    if np.any(im_r < 0.0) or np.any(im_p < 0.0):
        raise InputError("initial margin must be non-negative")
    e = np.maximum(v - c - im_r, 0.0)
    n = np.minimum(v - c + im_p, 0.0)
    return CollateralisedExposure(t, v, c, e, n)


def collateralised_profile(
    times: np.ndarray, values: np.ndarray, csa: Csa, disc: np.ndarray | None = None, *, q: float = 0.95
) -> ExposureProfile:
    """Exposure profile (EE, ENE, PFE, ...) of the collateralised position:
    the profile of V - C with IM applied, so ``ee``/``ene`` equal the means
    of E(t) and N(t) and the existing CVA/DVA code can use it unchanged."""
    ce = collateralised_exposure(times, values, csa)
    # E > 0 needs V - C > IM_r >= 0, which forces N = 0 (IM_p >= 0), so E + N
    # is a signed value whose positive part is E and negative part is N.
    signed = ce.exposure + ce.negative_exposure
    return profile(ce.times, signed, disc, q=q)


# ---------------------------------------------------------- closed forms


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def brownian_ee_mpor(sigma: float, mpor: float) -> float:
    """EE of a perfectly margined (H = MTA = 0) Brownian position: sigma
    sqrt(MPOR) phi(0).

    >>> round(brownian_ee_mpor(1e6, 10 / 250), 2)
    79788.46
    """
    return sigma * math.sqrt(mpor) * _phi(0.0)


def brownian_ee_im(sigma: float, mpor: float, im: float) -> float:
    """EE with IM: s [phi(k) - k (1 - Phi(k))], s = sigma sqrt(MPOR), k = IM / s."""
    s = sigma * math.sqrt(mpor)
    k = im / s
    return s * (_phi(k) - k * (1.0 - _Phi(k)))


def brownian_ee_threshold(v_sd: float, threshold: float) -> float:
    """E[min(V^+, H)] for V ~ N(0, v_sd^2): the no-lag EE with threshold H."""
    h = threshold / v_sd
    return v_sd * (_phi(0.0) - _phi(h)) + threshold * (1.0 - _Phi(h))
