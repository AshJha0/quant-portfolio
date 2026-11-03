"""Weights from forecasts and a covariance: MVO, risk parity, vol targeting.

Mean-variance with constraints and an L1 turnover penalty
---------------------------------------------------------
    maximise  mu'w - (gamma/2) w' S w - kappa * sum|w - w_prev|
    subject to lo <= w_i <= hi,  sum w = budget (optional),
               sum |w| <= gross (optional).

Plain idea: the turnover term is a linear cost per unit traded, so small
forecast changes no longer move the book (a no-trade region around
``w_prev``; Garleanu and Pedersen 2013 discuss the quadratic variant).
Unconstrained and with ``kappa = 0`` the answer is ``S^-1 mu / gamma``
(Markowitz 1952), which the tests use as the oracle. The L1 term is made
smooth by splitting ``w - w_prev = u - v`` with ``u, v >= 0``; SLSQP then
solves a smooth problem.

Risk parity
-----------
Equal risk contributions ``w_i (S w)_i = w' S w / N`` with ``w > 0`` and
``sum w = 1`` (Maillard, Roncalli and Teiletche 2010). Solved as the convex
problem ``min 1/2 y'Sy - (1/N) sum log y_i`` then ``w = y / sum y``
(Spinu 2013) by Newton's method. With a diagonal ``S`` the answer is
``w_i ∝ 1/sigma_i``.

Volatility targeting
--------------------
Scale weights by ``target / sqrt(w'Sw)``, capped at ``max_leverage``
(Moreira and Muir 2017 for the managed-volatility evidence).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

__all__ = ["mean_variance", "risk_contributions", "risk_parity", "vol_target"]


def _cov(cov: np.ndarray) -> np.ndarray:
    s = np.asarray(cov, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or not np.isfinite(s).all():
        raise ValueError("cov must be a finite square matrix")
    if not np.allclose(s, s.T, atol=1e-12):
        raise ValueError("cov must be symmetric")
    return s


def mean_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    gamma: float = 1.0,
    *,
    w_prev: np.ndarray | None = None,
    turnover_cost: float = 0.0,
    lower: float | np.ndarray = -np.inf,
    upper: float | np.ndarray = np.inf,
    budget: float | None = None,
    gross: float | None = None,
) -> np.ndarray:
    """Constrained mean-variance weights (see module docstring)."""
    s = _cov(cov)
    m = np.asarray(mu, dtype=float)
    n = len(m)
    if s.shape[0] != n:
        raise ValueError("mu and cov dimensions differ")
    if gamma <= 0 or turnover_cost < 0:
        raise ValueError("gamma must be > 0 and turnover_cost >= 0")
    w0 = np.zeros(n) if w_prev is None else np.asarray(w_prev, dtype=float)
    lo = np.broadcast_to(np.asarray(lower, dtype=float), (n,))
    hi = np.broadcast_to(np.asarray(upper, dtype=float), (n,))
    plain = (
        turnover_cost == 0
        and budget is None
        and gross is None
        and np.all(np.isinf(lo))
        and np.all(np.isinf(hi))
    )
    if plain:
        return np.linalg.solve(s, m) / gamma

    def split(z: np.ndarray) -> np.ndarray:
        return w0 + z[:n] - z[n:]

    def obj(z: np.ndarray) -> float:
        w = split(z)
        return float(-(m @ w) + 0.5 * gamma * w @ s @ w + turnover_cost * z.sum())

    def grad(z: np.ndarray) -> np.ndarray:
        g = -m + gamma * s @ split(z)
        return np.concatenate([g, -g]) + turnover_cost

    cons: list[Any] = []
    if budget is not None:
        cons.append({"type": "eq", "fun": lambda z: split(z).sum() - budget})
    for i in range(n):
        if np.isfinite(lo[i]):
            cons.append({"type": "ineq", "fun": lambda z, i=i: split(z)[i] - lo[i]})
        if np.isfinite(hi[i]):
            cons.append({"type": "ineq", "fun": lambda z, i=i: hi[i] - split(z)[i]})
    if gross is not None:
        # sum|w| <= gross, with |x| smoothed as sqrt(x^2 + 1e-12) so SLSQP sees
        # a differentiable constraint; the smoothing errs on the safe side.
        cons.append({"type": "ineq", "fun": lambda z: gross - np.sum(np.sqrt(split(z) ** 2 + 1e-12))})
    res: Any = minimize(
        obj, np.zeros(2 * n), jac=grad, method="SLSQP",
        bounds=[(0.0, None)] * (2 * n), constraints=cons,
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"mean_variance did not converge: {res.message}")
    return split(res.x)


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """``w_i (S w)_i``; they sum to the portfolio variance."""
    w = np.asarray(w, dtype=float)
    return w * (_cov(cov) @ w)


def risk_parity(cov: np.ndarray, tol: float = 1e-12, max_iter: int = 100) -> np.ndarray:
    """Long-only equal-risk-contribution weights summing to one (Spinu 2013)."""
    s = _cov(cov)
    n = s.shape[0]
    if np.any(np.diag(s) <= 0):
        raise ValueError("cov must have a positive diagonal")
    y = 1.0 / np.sqrt(np.diag(s))
    b = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        g = s @ y - b / y
        h = s + np.diag(b / y**2)
        step = np.linalg.solve(h, g)
        # damped step keeps y > 0
        a = 1.0
        while np.any(y - a * step <= 0):
            a *= 0.5
        y = y - a * step
        if np.max(np.abs(g)) < tol:
            break
    else:
        raise RuntimeError("risk_parity did not converge")
    return y / y.sum()


def vol_target(w: np.ndarray, cov: np.ndarray, target: float, max_leverage: float = np.inf) -> np.ndarray:
    """Scale ``w`` to ex-ante volatility ``target`` (same units as ``cov``)."""
    w = np.asarray(w, dtype=float)
    if target <= 0 or max_leverage <= 0:
        raise ValueError("target and max_leverage must be positive")
    vol = float(np.sqrt(w @ _cov(cov) @ w))
    if vol == 0:
        return w.copy()
    return w * min(target / vol, max_leverage)
