"""Heston characteristic function, shared by equity and FX (plan W20, Round 18).

Plain idea
----------
Every Fourier pricer for the Heston model (P1/P2, Carr-Madan, Gil-Pelaez,
COS) needs the same thing: the characteristic function of the log price at
expiry. Equity and FX each had their own copy. This module is the one copy;
``eq_surface.heston.heston_cf`` and ``fx_surface.heston.heston_cf`` are thin
shims over it, and each package keeps its own pricers and its own
``HestonParams`` (their validation and deprecated positional orders differ,
so merging those would break callers for no gain).

Formal statement
----------------
Risk-neutral dynamics ``dS/S = mu dt + sqrt(v) dW_S``,
``dv = kappa (theta - v) dt + xi sqrt(v) dW_v``, ``d<W_S, W_v> = rho dt``,
with ``mu = r - q`` (equity) or ``r_d - r_f`` (FX). Then
``phi(u) = E[exp(i u ln S_T)] = exp(i u (x0 + mu T) + C(u, T) + D(u, T) v0)``
with, in the "little Heston trap" form of Albrecher, Mayer, Schoutens and
Tistaert (2007, Wilmott Magazine, Jan., 83-92), eq. (6)-(8)::

    b  = kappa - i rho xi u
    d  = sqrt(b^2 + xi^2 (i u + u^2))            (principal root)
    g  = (b - d) / (b + d)
    C  = (kappa theta / xi^2) ((b - d) T - 2 ln((1 - g e^{-dT}) / (1 - g)))
    D  = ((b - d) / xi^2) (1 - e^{-dT}) / (1 - g e^{-dT})

The decaying exponential keeps the log argument away from the branch cut,
so the principal log is continuous in ``u`` (Gatheral 2006, *The Volatility
Surface*, ch. 2). For ``xi < 1e-12`` the variance path is deterministic,
``v(t) = theta + (v0 - theta) e^{-kappa t}``, and ``phi`` is the Gaussian
characteristic function with integrated variance
``IV = theta T + (v0 - theta)(1 - e^{-kappa T}) / kappa``.

Bit compatibility. The arithmetic is written in the order both old copies
used (``xi * xi`` first, ``x0 + mu * T`` as one drift term), so the two
packages' prices did not move by a single bit when they switched to it;
``tests/test_kernel_corpus.py`` holds them to 1e-12 and
``python/core/tests/test_vol_heston.py`` checks equality outright.

Worked example
--------------
At ``u = 0`` every characteristic function is 1, and at ``u = -i`` it is
the forward ``E[S_T] = S e^{mu T}``:

>>> import math
>>> p = dict(v0=0.04, kappa=1.5, theta=0.05, xi=0.5, rho=-0.6)
>>> heston_cf(0.0, 1.0, **p, x0=math.log(100.0), mu=0.03)
(1+0j)
>>> round(heston_cf(-1j, 1.0, **p, x0=math.log(100.0), mu=0.03).real, 10)
103.0454533954
"""

from __future__ import annotations

import numpy as np

__all__ = ["DETERMINISTIC_XI", "heston_cf"]

#: Below this vol-of-vol the variance path is treated as deterministic.
DETERMINISTIC_XI = 1e-12


def heston_cf(
    u: np.ndarray | complex | float,
    T: float,
    *,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    x0: float = 0.0,
    mu: float = 0.0,
) -> np.ndarray | complex:
    """``E[exp(i u ln S_T)]`` with ``ln S_0 = x0`` and drift ``mu``.

    ``u`` may be real or complex, scalar or array (complex arguments are
    needed for the P1 shift and for damped integrands). Pass ``x0 = 0`` for
    the characteristic function of the log return ``ln(S_T / S_0)``.
    Returns a Python ``complex`` for scalar ``u``, else an ndarray.
    """
    if T < 0.0:
        raise ValueError(f"T must be non-negative, got {T}")
    u = np.asarray(u, dtype=complex)
    drift = x0 + mu * T

    if xi < DETERMINISTIC_XI:
        iv = theta * T + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / kappa
        out = np.exp(1j * u * (drift - 0.5 * iv) - 0.5 * u * u * iv)
        return out if out.ndim else complex(out)

    xi2 = xi * xi
    b = kappa - 1j * rho * xi * u
    d = np.sqrt(b * b + xi2 * (1j * u + u * u))
    g = (b - d) / (b + d)
    e_dT = np.exp(-d * T)
    C = (kappa * theta / xi2) * ((b - d) * T - 2.0 * np.log((1.0 - g * e_dT) / (1.0 - g)))
    D = ((b - d) / xi2) * (1.0 - e_dT) / (1.0 - g * e_dT)
    out = np.exp(1j * u * drift + C + D * v0)
    return out if out.ndim else complex(out)
