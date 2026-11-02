"""Tracking-error optimisation against a benchmark.

Ex-ante tracking error of weights w against benchmark b:
TE = sqrt((w - b)' Sigma (w - b)).

Closed form (Grinold and Kahn 2000, "Active Portfolio Management", 2nd ed.,
chapter 14; Roll 1992, "A mean/variance analysis of tracking error",
J. Portfolio Management 18(4)): maximise alpha' a subject to
a' Sigma a = TE^2 and 1' a = 0 (fully invested). With the budget-neutral
alpha alpha_c = alpha - (1' Sigma^{-1} alpha / 1' Sigma^{-1} 1) 1,

    a* = TE * Sigma^{-1} alpha_c / sqrt(alpha_c' Sigma^{-1} alpha_c),
    IR* = sqrt(alpha_c' Sigma^{-1} alpha_c).

With long-only or box constraints there is no closed form; the SLSQP
version below maximises alpha' a subject to the TE budget and the bounds,
and agrees with the closed form when no bound binds.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def tracking_error(w: np.ndarray, benchmark: np.ndarray, cov: np.ndarray) -> float:
    a = np.asarray(w, dtype=float) - np.asarray(benchmark, dtype=float)
    return float(np.sqrt(max(a @ np.asarray(cov, dtype=float) @ a, 0.0)))


def _centred_alpha(alpha: np.ndarray, cov: np.ndarray) -> np.ndarray:
    ones = np.ones(alpha.size)
    inv_a, inv_1 = np.linalg.solve(cov, alpha), np.linalg.solve(cov, ones)
    return alpha - (ones @ inv_a) / (ones @ inv_1) * ones


def max_ir_active_weights(alpha: np.ndarray, cov: np.ndarray, te_target: float) -> np.ndarray:
    """Closed-form active weights at a target TE (module docstring)."""
    al = np.asarray(alpha, dtype=float)
    c = np.asarray(cov, dtype=float)
    if c.shape != (al.size, al.size) or te_target <= 0.0:
        raise ValueError("cov must match alpha and te_target must be positive")
    ac = _centred_alpha(al, c)
    z = np.linalg.solve(c, ac)
    ir = float(np.sqrt(ac @ z))
    if ir == 0.0:
        return np.zeros(al.size)
    return te_target * z / ir


def information_ratio_bound(alpha: np.ndarray, cov: np.ndarray) -> float:
    al = np.asarray(alpha, dtype=float)
    c = np.asarray(cov, dtype=float)
    ac = _centred_alpha(al, c)
    return float(np.sqrt(ac @ np.linalg.solve(c, ac)))


def te_constrained_weights(
    alpha: np.ndarray,
    cov: np.ndarray,
    benchmark: np.ndarray,
    te_target: float,
    bounds: tuple[float, float] | None = (0.0, 1.0),
) -> np.ndarray:
    """Maximise alpha' w subject to TE <= te_target, 1' w = 1 and box bounds."""
    al = np.asarray(alpha, dtype=float)
    c = np.asarray(cov, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    n = al.size
    if b.shape != (n,) or abs(b.sum() - 1.0) > 1e-9 or te_target <= 0.0:
        raise ValueError("benchmark must sum to 1 and te_target be positive")
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)},
        {
            "type": "ineq",
            "fun": lambda w: te_target**2 - (w - b) @ c @ (w - b),
            "jac": lambda w: -2.0 * c @ (w - b),
        },
    ]
    res = minimize(
        lambda w: -(al @ w),
        b.copy(),
        jac=lambda w: -al,
        constraints=cons,
        bounds=None if bounds is None else [bounds] * n,
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"TE optimisation failed: {res.message}")
    return res.x
