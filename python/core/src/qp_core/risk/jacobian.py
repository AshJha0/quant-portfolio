"""Quote-to-node Jacobians, instrument-space risk and cross-gamma.

The plain idea
--------------
A curve is built from market quotes (deposit rates, swap rates). The curve
itself is stored as nodes (zero rates or discount factors at pillar times).
Bumping a node answers "what if the zero rate at 7y moved?", a question no
trader can hedge directly. Bumping a quote and rebuilding the curve answers
"what if the 7y swap rate moved?", which is hedgeable with a 7y swap. The
Jacobian links the two: it says how every quote moves when a node moves, and
its inverse turns node risk into quote (instrument-space) risk without
rebuilding the curve once per quote.

The formal statement
--------------------
Let ``z`` be the node vector and ``q = Q(z)`` the model quotes, with the curve
calibrated so that ``Q(z*) = q_mkt``. For a book value ``P(z)``::

    J = dQ/dz            (n_quotes x n_nodes, square for an exact bootstrap)
    dP/dq = (J^{-1})^T dP/dz      (implicit-function theorem)

so the quote delta vector solves ``J^T g_q = g_z``. Andersen and Piterbarg,
*Interest Rate Modeling* vol. 3 (2010) ch. 26, "Jacobian method"; Hagan and
West (2006) for the curve-construction side. The same identity maps a node
Hessian to a quote Hessian when ``Q`` is linear to second order; this module
computes cross-gamma directly by finite differences in whichever space the
caller passes, so no second-order term of ``Q`` is dropped silently.

Worked example
--------------
For a two-node curve where quote ``i`` depends only on node ``i`` with slope
``-0.5``, ``J = diag(-0.5, -0.5)`` and a book with ``dP/dz = (-2, -4)`` has
``dP/dq = (4, 8)``: each quote moves at half the speed of its node, so the
book is twice as sensitive per unit of quote. ``tests/test_risk_jacobian.py``
checks this and checks the Jacobian route against rebuilding the curve from
bumped quotes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from qp_core.model import model

__all__ = [
    "cross_gamma",
    "fd_gradient",
    "fd_jacobian",
    "node_to_quote_risk",
    "quote_to_node_risk",
]

VectorFn = Callable[[np.ndarray], np.ndarray]
ScalarFn = Callable[[np.ndarray], float]


def _as_vector(x: np.ndarray | list[float], name: str) -> np.ndarray:
    v = np.asarray(x, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D vector")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return v


def _check_step(h: float) -> float:
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"finite-difference step must be > 0, got {h}")
    return float(h)


def fd_jacobian(f: VectorFn, x: np.ndarray | list[float], h: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian ``J[i, j] = d f_i / d x_j``.

    Truncation error is ``O(h^2)`` per entry; with ``h = 1e-6`` on rates of
    order 1e-2 the round-off floor is about ``1e-10 / h = 1e-4`` relative to
    an entry of order 1, so callers checking to 1e-6 should use ``h`` near
    ``1e-5`` (the default ``1e-6`` is chosen for well-scaled discount-factor
    maps and is what the tests use).
    """
    x0 = _as_vector(x, "x")
    step = _check_step(h)
    cols = []
    for j in range(x0.size):
        e = np.zeros_like(x0)
        e[j] = step
        up = np.asarray(f(x0 + e), dtype=float)
        dn = np.asarray(f(x0 - e), dtype=float)
        cols.append((up - dn) / (2.0 * step))
    return np.column_stack(cols)


def fd_gradient(f: ScalarFn, x: np.ndarray | list[float], h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of a scalar function."""
    return np.asarray(fd_jacobian(lambda v: np.atleast_1d(f(v)), x, h)[0], dtype=float)


@model(id="CORE-RISK-JACOBIAN")
def cross_gamma(f: ScalarFn, x: np.ndarray | list[float], h: float = 1e-4) -> np.ndarray:
    """Full Hessian ``H[i, j] = d^2 f / dx_i dx_j`` by central differences.

    Diagonal: ``(f(x+h e_i) - 2 f(x) + f(x-h e_i)) / h^2``.
    Off-diagonal (the cross-gammas): the four-point stencil
    ``(f(++) - f(+-) - f(-+) + f(--)) / (4 h^2)``. Both are ``O(h^2)``.
    The result is symmetric by construction (the stencil is symmetric in
    ``i`` and ``j``), so ``0.5 dx^T H dx`` is the second-order Taylor term.
    """
    x0 = _as_vector(x, "x")
    step = _check_step(h)
    n = x0.size
    f0 = float(f(x0))
    hess = np.empty((n, n))
    eye = np.eye(n) * step
    for i in range(n):
        hess[i, i] = (float(f(x0 + eye[i])) - 2.0 * f0 + float(f(x0 - eye[i]))) / step**2
        for j in range(i + 1, n):
            pp = float(f(x0 + eye[i] + eye[j]))
            pm = float(f(x0 + eye[i] - eye[j]))
            mp = float(f(x0 - eye[i] + eye[j]))
            mm = float(f(x0 - eye[i] - eye[j]))
            hess[i, j] = hess[j, i] = (pp - pm - mp + mm) / (4.0 * step**2)
    return hess


@model(id="CORE-RISK-JACOBIAN")
def node_to_quote_risk(node_risk: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    """Map node sensitivities ``dP/dz`` to quote sensitivities ``dP/dq``.

    ``jacobian`` is ``J = dQ/dz`` (quotes by nodes). Solves ``J^T g_q = g_z``
    rather than forming ``J^{-1}``. ``node_risk`` may be a vector or a matrix
    whose columns are separate books.

    Raises ``ValueError`` if ``J`` is not square or is numerically singular
    (condition number above 1e12): a curve whose quotes do not pin down its
    nodes has no well-defined instrument-space risk.
    """
    jac = np.asarray(jacobian, dtype=float)
    g = np.asarray(node_risk, dtype=float)
    if jac.ndim != 2 or jac.shape[0] != jac.shape[1]:
        raise ValueError(f"Jacobian must be square, got shape {jac.shape}")
    if g.shape[0] != jac.shape[1]:
        raise ValueError(f"node risk has {g.shape[0]} rows, Jacobian has {jac.shape[1]} nodes")
    cond = np.linalg.cond(jac)
    if not np.isfinite(cond) or cond > 1e12:
        raise ValueError(f"Jacobian is numerically singular (condition number {cond:.3g})")
    return np.asarray(np.linalg.solve(jac.T, g), dtype=float)


@model(id="CORE-RISK-JACOBIAN")
def quote_to_node_risk(quote_risk: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    """Inverse map: ``dP/dz = J^T dP/dq``."""
    jac = np.asarray(jacobian, dtype=float)
    return np.asarray(jac.T @ np.asarray(quote_risk, dtype=float), dtype=float)
