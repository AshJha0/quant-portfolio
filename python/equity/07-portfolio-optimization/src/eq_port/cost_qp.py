"""Transaction-cost and turnover-penalised mean-variance rebalancing.

    max_w  mu' w - (lambda / 2) w' Sigma w
           - c' |w - w0|                 (proportional costs, e.g. half-spread)
           - (eta / 2) |w - w0|^2        (quadratic impact)
    s.t.   sum |w - w0| <= tau           (turnover cap, optional)
           1' w = 1                      (budget, optional)
           lo <= w <= hi                 (box, optional)

The absolute values are linearised with w = w0 + u - v, u, v >= 0, which
gives a smooth QP solved by SLSQP. References: Boyd et al. (2017),
"Multi-period trading via convex optimization", Foundations and Trends in
Optimization 3(1), section 2.3 (transaction cost model) and section 4.2;
Grinold and Kahn (2000), "Active Portfolio Management", chapter 16.

Oracle: with no budget and a diagonal Sigma the problem separates by asset
and has a closed-form no-trade band: with g_i = mu_i - lambda s_ii w0_i,
w_i = w0_i if |g_i| <= c_i, else
w_i = (mu_i - sign(g_i) c_i + eta w0_i) / (lambda s_ii + eta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class RebalanceResult:
    weights: np.ndarray
    turnover: float
    cost: float
    objective: float


def rebalance(
    mu: np.ndarray,
    cov: np.ndarray,
    w0: np.ndarray,
    risk_aversion: float,
    cost: np.ndarray | float = 0.0,
    impact: float = 0.0,
    max_turnover: float | None = None,
    budget: bool = True,
    bounds: tuple[float, float] | None = None,
) -> RebalanceResult:
    m = np.asarray(mu, dtype=float)
    c = np.asarray(cov, dtype=float)
    x0w = np.asarray(w0, dtype=float)
    n = m.size
    if c.shape != (n, n) or x0w.shape != (n,):
        raise ValueError("mu, cov and w0 must agree in size")
    kc = np.broadcast_to(np.asarray(cost, dtype=float), (n,)).copy()
    if risk_aversion <= 0.0 or impact < 0.0 or np.any(kc < 0.0):
        raise ValueError("risk_aversion must be positive; cost and impact non-negative")
    if max_turnover is not None and max_turnover < 0.0:
        raise ValueError("max_turnover must be non-negative")
    if budget and abs(x0w.sum() - 1.0) > 1e-9 and max_turnover is not None:
        raise ValueError("w0 must sum to 1 when a budget and a turnover cap are both set")

    def unpack(x: np.ndarray) -> np.ndarray:
        return x0w + x[:n] - x[n:]

    def obj(x: np.ndarray) -> float:
        w, d = unpack(x), x[:n] - x[n:]
        return float(-(m @ w) + 0.5 * risk_aversion * w @ c @ w + kc @ (x[:n] + x[n:]) + 0.5 * impact * d @ d)

    def jac(x: np.ndarray) -> np.ndarray:
        w, d = unpack(x), x[:n] - x[n:]
        gw = -m + risk_aversion * c @ w + impact * d
        return np.concatenate([gw + kc, -gw + kc])

    cons = []
    if budget:
        cons.append(
            {
                "type": "eq",
                "fun": lambda x: unpack(x).sum() - 1.0,
                "jac": lambda x: np.concatenate([np.ones(n), -np.ones(n)]),
            }
        )
    if max_turnover is not None:
        cons.append(
            {"type": "ineq", "fun": lambda x: max_turnover - x.sum(), "jac": lambda x: -np.ones(2 * n)}
        )
    if bounds is not None:
        lo, hi = bounds
        sel = np.hstack([np.eye(n), -np.eye(n)])
        cons.append({"type": "ineq", "fun": lambda x: unpack(x) - lo, "jac": lambda x: sel})
        cons.append({"type": "ineq", "fun": lambda x: hi - unpack(x), "jac": lambda x: -sel})
    start = np.zeros(2 * n)
    if budget and abs(x0w.sum() - 1.0) > 1e-12:
        gap = 1.0 - x0w.sum()
        start[:n] = max(gap, 0.0) / n
        start[n:] = max(-gap, 0.0) / n
    res = minimize(
        obj,
        start,
        jac=jac,
        bounds=[(0.0, None)] * (2 * n),
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 2000},
    )
    if not res.success:
        raise RuntimeError(f"rebalance QP failed: {res.message}")
    w = unpack(res.x)
    trade = np.abs(w - x0w)
    return RebalanceResult(w, float(trade.sum()), float(kc @ trade), -float(res.fun))


def no_trade_band_solution(
    mu: np.ndarray,
    var: np.ndarray,
    w0: np.ndarray,
    risk_aversion: float,
    cost: np.ndarray,
    impact: float = 0.0,
) -> np.ndarray:
    """Closed form for diagonal Sigma and no budget (module docstring oracle)."""
    m, s, x0, kc = (np.asarray(a, dtype=float) for a in (mu, var, w0, cost))
    g = m - risk_aversion * s * x0
    move = (m - np.sign(g) * kc + impact * x0) / (risk_aversion * s + impact)
    return np.where(np.abs(g) <= kc, x0, move)
