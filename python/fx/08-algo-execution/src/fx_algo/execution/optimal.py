"""Almgren-Chriss optimal execution adapted to FX session liquidity.

Classic Almgren-Chriss (2000) assumes *constant* temporary impact and
volatility over the execution horizon.  The FX day is anything but
constant: depth in the London-NY overlap is ~10x the late-NY session.
This module solves the discrete mean-variance execution problem with
**bucket-specific** impact and vol,

.. math::

    \\min_{n}\\; \\sum_{j=1}^{N} \\frac{\\eta_j}{\\tau} n_j^2
    \\;+\\; \\lambda \\sum_{j=1}^{N} \\sigma_j^2\\, \\tau\\, x_j^2,
    \\qquad x_j = X - \\sum_{i\\le j} n_i,\\; \\sum_j n_j = X,

a strictly convex equality-constrained QP solved *numerically* via its
KKT system (``scipy.linalg.solve``).  Linear permanent impact is
excluded from the objective because its cost is (to leading order)
schedule-independent — see docs/METHODOLOGY.md.

Two verifiable limits anchor the implementation (both unit-tested):

* constant ``eta``/``sigma``: the numerical solution equals the
  closed-form discrete AC trajectory
  ``x_j = X sinh(kappa (T - t_j)) / sinh(kappa T)`` with
  ``cosh(kappa tau) = 1 + lambda sigma^2 tau^2 / (2 eta)``;
* ``lambda -> 0``: the solution is ``n_j ∝ 1/eta_j``; with
  ``eta_j ∝ 1/depth_j`` (see :func:`eta_from_depth`) this is exactly the
  liquidity-weighted TWAP-analog schedule.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve

__all__ = [
    "eta_from_depth",
    "ac_closed_form_schedule",
    "piecewise_ac_schedule",
    "ac_expected_cost",
]


def eta_from_depth(
    depths: np.ndarray, k_eta: float = 1.0, ref_depth: float | None = None
) -> np.ndarray:
    """Quadratic temporary-impact coefficients from a depth profile.

    ``eta_j = k_eta * ref_depth / depth_j`` — trading cost per unit rate
    is inversely proportional to displayed session depth, the quadratic
    (AC-tractable) stand-in for the simulator's sqrt law, calibrated to
    match marginal cost at a reference participation (METHODOLOGY.md).

    Parameters
    ----------
    depths : numpy.ndarray
        Bucket depths (mm base per bucket), strictly positive.
    k_eta : float
        Overall impact scale (pips per (mm/bucket) at the reference depth).
    ref_depth : float, optional
        Normalising depth (default: mean of ``depths``).

    Returns
    -------
    numpy.ndarray
    """
    depths = np.asarray(depths, dtype=float)
    if not np.all(np.isfinite(depths)):
        raise ValueError("depths must be finite (no NaN/Inf)")
    if np.any(depths <= 0):
        raise ValueError("depths must be strictly positive")
    if ref_depth is None:
        ref_depth = float(depths.mean())
    return k_eta * ref_depth / depths


def ac_closed_form_schedule(
    parent_qty: float,
    n_buckets: int,
    eta: float,
    sigma: float,
    risk_aversion: float,
    tau: float = 1.0,
) -> np.ndarray:
    """Closed-form discrete Almgren-Chriss schedule (constant liquidity).

    Optimal holdings are ``x_j = X sinh(kappa(N-j)tau)/sinh(kappa N tau)``
    with ``cosh(kappa tau) = 1 + lambda sigma^2 tau^2 / (2 eta)``;
    trades are ``n_j = x_{j-1} - x_j``.  ``risk_aversion = 0`` gives
    TWAP.

    Parameters
    ----------
    parent_qty : float
        Signed parent quantity X.
    n_buckets : int
        Number of buckets N.
    eta : float
        Temporary impact coefficient (cost = eta * n^2 / tau per bucket).
    sigma : float
        Per-sqrt-bucket volatility (pips or price units — must simply be
        consistent with ``eta`` and ``risk_aversion``).
    risk_aversion : float
        Mean-variance lambda >= 0.
    tau : float
        Bucket length in the same time units as ``sigma``/``eta``.

    Returns
    -------
    numpy.ndarray
        Trades per bucket, summing exactly to ``parent_qty``.
    """
    _check_ac_args(risk_aversion, tau)
    if not (np.isfinite(eta) and np.isfinite(sigma) and np.isfinite(parent_qty)):
        raise ValueError("eta, sigma and parent_qty must be finite (no NaN/Inf)")
    if eta <= 0:
        raise ValueError(f"eta must be > 0, got {eta}")
    N = int(n_buckets)
    if N < 1:
        raise ValueError("n_buckets must be >= 1")
    if risk_aversion == 0 or sigma == 0:
        return np.full(N, parent_qty / N)
    kt = np.arccosh(1.0 + risk_aversion * sigma**2 * tau**2 / (2.0 * eta))
    j = np.arange(N + 1)
    x = parent_qty * np.sinh(kt * (N - j)) / np.sinh(kt * N)
    return -np.diff(x)


def piecewise_ac_schedule(
    parent_qty: float,
    eta: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    tau: float = 1.0,
    allow_sells: bool = False,
) -> np.ndarray:
    """Numerical piecewise-AC schedule under time-varying liquidity.

    Solves the strictly convex QP (module docstring) through its KKT
    conditions: ``[H 1; 1' 0] [n; mu] = [b; X]`` with
    ``H = 2 (diag(eta)/tau + lambda tau C' diag(sigma^2) C)``,
    ``b = 2 lambda tau C' sigma^2 X`` and ``C`` the lower-triangular
    cumulative-sum operator, via ``scipy.linalg.solve``.

    Parameters
    ----------
    parent_qty : float
        Signed parent quantity X.
    eta : numpy.ndarray
        Bucket temporary-impact coefficients (> 0), length N.
    sigma : numpy.ndarray
        Bucket vols (>= 0), length N.
    risk_aversion : float
        Mean-variance lambda >= 0.
    tau : float
        Bucket length.
    allow_sells : bool
        If False (default), one-sided execution is enforced: at high
        risk aversion the unconstrained optimum can briefly trade
        *against* the parent in very illiquid buckets; the bound
        ``sign(n_j) = sign(X)`` is then imposed by an exact active-set
        loop (clamp negative trades to zero, re-solve the equality QP on
        the free set, repeat).

    Returns
    -------
    numpy.ndarray
        Trades per bucket, summing to ``parent_qty`` to 1e-10 (then
        renormalised so the sum is *exact*).
    """
    _check_ac_args(risk_aversion, tau)
    eta = np.asarray(eta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if eta.ndim != 1 or len(eta) != len(sigma):
        raise ValueError("eta and sigma must be 1-D arrays of equal length")
    if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(sigma)):
        raise ValueError("eta and sigma must be finite (no NaN/Inf)")
    if not np.isfinite(parent_qty):
        raise ValueError("parent_qty must be finite (no NaN/Inf)")
    if np.any(eta <= 0):
        raise ValueError("eta must be strictly positive")
    if np.any(sigma < 0):
        raise ValueError("sigma must be non-negative")
    N = len(eta)
    if N == 1:
        return np.array([parent_qty], dtype=float)
    if parent_qty == 0.0:
        # Nothing to trade: the QP is degenerate (the exact-sum
        # renormalisation would divide by a zero total and return NaN).
        return np.zeros(N)

    sign = 1.0 if parent_qty >= 0 else -1.0
    X = abs(parent_qty)

    # Full-horizon quadratic form (kept fixed while clamping so the
    # variance of inventory carried *through* clamped buckets stays in
    # the objective): F(n) = n'(H/2)n - b'n + const.
    C = np.tril(np.ones((N, N)))
    d2 = risk_aversion * tau * sigma**2
    H = 2.0 * (np.diag(eta / tau) + C.T @ (d2[:, None] * C))
    b = 2.0 * (C.T @ (d2 * X))

    def solve_on(free: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(free)
        k = len(idx)
        kkt = np.zeros((k + 1, k + 1))
        kkt[:k, :k] = H[np.ix_(idx, idx)]
        kkt[:k, k] = 1.0
        kkt[k, :k] = 1.0
        rhs = np.concatenate([b[idx], [X]])
        n = np.zeros(N)
        n[idx] = solve(kkt, rhs, assume_a="sym")[:k]
        return n

    free = np.ones(N, dtype=bool)
    n = solve_on(free)
    if not allow_sells:
        for _ in range(N):
            if np.all(n >= -1e-12 * max(X, 1.0)):
                break
            free &= n >= 0.0  # clamp sell-back buckets, re-solve on the rest
            if not free.any():  # pragma: no cover - cannot happen for X > 0
                raise RuntimeError("active-set clamping removed all buckets")
            n = solve_on(free)
        n = np.clip(n, 0.0, None)
        # exact-sum against fp residue, preserving non-negativity: scale
        # multiplicatively, then absorb the remainder in the largest bucket
        n *= X / n.sum()
        n[int(np.argmax(n))] += X - n.sum()
    else:
        n += (X - n.sum()) / N
    return sign * n


def ac_expected_cost(
    trades: np.ndarray,
    eta: np.ndarray | float,
    sigma: np.ndarray | float,
    risk_aversion: float,
    tau: float = 1.0,
) -> float:
    """Mean-variance objective of a schedule under given liquidity.

    ``cost = sum eta_j n_j^2 / tau + lambda sum sigma_j^2 tau x_j^2``
    with ``x_j`` the post-trade holdings.  Used to certify optimality of
    the piecewise solution against any competitor schedule.

    Returns
    -------
    float
    """
    _check_ac_args(risk_aversion, tau)
    n = np.asarray(trades, dtype=float)
    eta_arr = np.broadcast_to(np.asarray(eta, dtype=float), n.shape)
    sig_arr = np.broadcast_to(np.asarray(sigma, dtype=float), n.shape)
    X = n.sum()
    x = X - np.cumsum(n)
    return float(np.sum(eta_arr * n**2) / tau + risk_aversion * tau * np.sum(sig_arr**2 * x**2))


def _check_ac_args(risk_aversion: float, tau: float) -> None:
    if risk_aversion < 0:
        raise ValueError(f"risk_aversion must be >= 0, got {risk_aversion}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
