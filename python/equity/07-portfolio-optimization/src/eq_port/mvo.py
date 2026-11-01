"""Mean-variance optimization: closed forms, constrained SLSQP solvers,
and efficient-frontier tracing.

Conventions: ``mu`` is the (N,) per-period expected return vector, ``cov``
the (N, N) per-period covariance, ``rf`` the per-period risk-free rate.
Weights sum to 1 (fully invested) unless stated otherwise.

Optimization backend: closed forms where they exist (unconstrained
min-variance / tangency / frontier via the two-fund theorem) and
``scipy.optimize.minimize(method="SLSQP")`` for inequality-constrained
problems. cvxpy is not a dependency: the QPs here are small
(N ~ 10-100), smooth and well-scaled, SLSQP solves them to ~1e-9 KKT
tolerance in milliseconds, and every SLSQP solution is cross-checked
against closed forms in the tests, so a heavyweight conic-solver stack
buys nothing but install weight for this project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "min_variance_weights",
    "tangency_weights",
    "min_variance_constrained",
    "max_sharpe_constrained",
    "max_sharpe_qp",
    "MaxSharpeResult",
    "target_return_portfolio",
    "target_risk_portfolio",
    "efficient_frontier",
    "FrontierResult",
    "portfolio_vol",
    "portfolio_return",
]

_Bounds = tuple[float, float] | list[tuple[float, float]] | None


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """Per-period expected portfolio return w'mu."""
    return float(np.asarray(weights, float) @ np.asarray(mu, float))


def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    """Per-period portfolio volatility sqrt(w' Sigma w)."""
    w = np.asarray(weights, float)
    return float(np.sqrt(max(w @ np.asarray(cov, float) @ w, 0.0)))


def _validate(cov: np.ndarray, mu: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate/coerce (cov, mu); raise informative ValueError on bad input."""
    sigma = np.asarray(cov, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"cov must be a square matrix, got shape {sigma.shape}")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("cov contains NaN or infinite values")
    if np.max(np.abs(sigma - sigma.T), initial=0.0) > 1e-10 * np.max(np.abs(sigma), initial=0.0):
        raise ValueError("cov must be symmetric")
    m = None
    if mu is not None:
        m = np.asarray(mu, dtype=float).ravel()
        if m.shape[0] != sigma.shape[0]:
            raise ValueError(
                f"dimension mismatch: mu has {m.shape[0]} entries, cov is {sigma.shape}"
            )
        if not np.all(np.isfinite(m)):
            raise ValueError("mu contains NaN or infinite values")
    return sigma, m


def _solve_spd(sigma: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Sigma x = b via Cholesky; informative error if not PD."""
    try:
        c = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "cov is not positive definite (singular or has non-positive "
            "eigenvalues); repair it first with eq_port.covariance.psd_repair "
            "or use a shrunk estimator (ledoit_wolf_cc)."
        ) from exc
    y = np.linalg.solve(c, b)
    return np.linalg.solve(c.T, y)


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Closed-form unconstrained (budget-only) minimum-variance weights.

        w_mv = Sigma^{-1} 1 / (1' Sigma^{-1} 1)

    Parameters
    ----------
    cov : (N, N) array-like
        Positive-definite per-period covariance.

    Returns
    -------
    np.ndarray
        (N,) weights summing to 1 (short positions allowed).
    """
    sigma, _ = _validate(cov)
    ones = np.ones(sigma.shape[0])
    x = _solve_spd(sigma, ones)
    return x / float(ones @ x)


def tangency_weights(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Closed-form tangency (maximum-Sharpe) weights.

        w_tan = Sigma^{-1} (mu - rf 1) / (1' Sigma^{-1} (mu - rf 1))

    Parameters
    ----------
    mu : (N,) array-like
        Per-period expected returns.
    cov : (N, N) array-like
        Positive-definite per-period covariance.
    rf : float
        Per-period risk-free rate.

    Returns
    -------
    np.ndarray
        (N,) weights summing to 1 (short positions allowed).

    Raises
    ------
    ValueError
        If ``1' Sigma^{-1} (mu - rf) <= 0``: the tangency portfolio is
        then on the inefficient branch and the normalisation flips its
        sign, which is almost always a sign of garbage mean inputs.
    """
    sigma, m = _validate(cov, mu)
    assert m is not None
    ex = m - rf
    x = _solve_spd(sigma, ex)
    denom = float(np.ones_like(ex) @ x)
    if denom <= 0.0:
        raise ValueError(
            "1' Sigma^{-1} (mu - rf) <= 0: no fully-invested tangency portfolio "
            "with positive excess return exists for these inputs (check that "
            "expected returns exceed the risk-free rate on average)."
        )
    return x / denom


def _bounds_list(bounds: _Bounds, n: int) -> list[tuple[float, float]] | None:
    if bounds is None:
        return None
    if isinstance(bounds, tuple) and len(bounds) == 2 and np.isscalar(bounds[0]):
        return [(float(bounds[0]), float(bounds[1]))] * n
    lst = [(float(lo), float(hi)) for lo, hi in bounds]  # type: ignore[union-attr]
    if len(lst) != n:
        raise ValueError(f"bounds must have {n} entries, got {len(lst)}")
    return lst


def _scale(sigma: np.ndarray) -> float:
    """Scale factor making w'Sigma w O(1) for SLSQP (mean diagonal)."""
    s = float(np.trace(sigma)) / sigma.shape[0]
    return s if s > 0 else 1.0


def _feasible_start(bounds: list[tuple[float, float]] | None, n: int) -> np.ndarray:
    w0 = np.full(n, 1.0 / n)
    if bounds is not None:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        w0 = np.clip(w0, lo, hi)
        s = w0.sum()
        if s != 0:
            w0 = np.clip(w0 / s, lo, hi)
    return w0


def min_variance_constrained(
    cov: np.ndarray,
    bounds: _Bounds = (0.0, 1.0),
    budget: float = 1.0,
) -> np.ndarray:
    """Minimum-variance weights under box bounds and a budget constraint,
    solved with SLSQP.

    Parameters
    ----------
    cov : (N, N) array-like
        Per-period covariance (PSD; strict PD not required).
    bounds : (lo, hi) or list of (lo, hi) or None
        Per-asset box bounds; ``(0.0, 1.0)`` = long-only, None = unconstrained.
    budget : float
        Required sum of weights (1.0 = fully invested).

    Returns
    -------
    np.ndarray
        (N,) optimal weights with ``sum(w) == budget``.
    """
    sigma, _ = _validate(cov)
    n = sigma.shape[0]
    bl = _bounds_list(bounds, n)
    w0 = _feasible_start(bl, n) * budget
    s = _scale(sigma)
    sig_s = sigma / s  # rescale so the objective is O(1) for SLSQP

    def obj(w: np.ndarray) -> float:
        return float(w @ sig_s @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * sig_s @ w

    cons = [{"type": "eq", "fun": lambda w: w.sum() - budget, "jac": lambda w: np.ones(n)}]
    res = minimize(
        obj, w0, jac=jac, bounds=bl, constraints=cons, method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-14},
    )
    if not res.success:
        raise ValueError(f"SLSQP failed in min_variance_constrained: {res.message}")
    return res.x


def target_return_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    target: float,
    bounds: _Bounds = None,
    budget: float = 1.0,
) -> np.ndarray:
    """Minimum-variance portfolio achieving expected return ``target``.

    Solves min w'Sigma w s.t. w'mu = target, sum(w) = budget, box bounds.

    Parameters
    ----------
    mu, cov : array-like
        Per-period moments.
    target : float
        Required per-period expected return (equality constraint).
    bounds : see :func:`min_variance_constrained`.
    budget : float
        Required sum of weights.

    Returns
    -------
    np.ndarray
        (N,) optimal weights.

    Raises
    ------
    ValueError
        If ``target`` is outside the return range achievable within the box
        bounds (necessary-condition pre-check, ignoring the budget), or if
        SLSQP reports the constraint set infeasible.
    """
    sigma, m = _validate(cov, mu)
    assert m is not None
    n = sigma.shape[0]
    bl = _bounds_list(bounds, n)
    if bl is not None:
        lo = np.array([b[0] for b in bl])
        hi = np.array([b[1] for b in bl])
        # Necessary feasibility condition (box only, budget ignored): the
        # achievable w'mu range is [sum(min over box), sum(max over box)].
        r_hi = float(m @ np.where(m > 0, hi, lo))
        r_lo = float(m @ np.where(m > 0, lo, hi))
        if target > r_hi + 1e-12 or target < r_lo - 1e-12:
            raise ValueError(
                f"infeasible target return {target:g}: the box bounds only "
                f"allow expected returns in [{r_lo:g}, {r_hi:g}]"
            )
    w0 = _feasible_start(bl, n) * budget
    s = _scale(sigma)
    sig_s = sigma / s
    # scale the return-equality constraint to O(1) as well
    m_scale = float(np.max(np.abs(m))) or 1.0
    m_s = m / m_scale
    t_s = target / m_scale

    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - budget, "jac": lambda w: np.ones(n)},
        {"type": "eq", "fun": lambda w: float(w @ m_s) - t_s, "jac": lambda w: m_s},
    ]

    def obj(w: np.ndarray) -> float:
        return float(w @ sig_s @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * sig_s @ w

    res = minimize(
        obj, w0, jac=jac, bounds=bl, constraints=cons, method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-14},
    )
    if not res.success:
        raise ValueError(
            f"SLSQP failed in target_return_portfolio (target={target:g}): {res.message}"
        )
    return res.x


def target_risk_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    target_vol: float,
    bounds: _Bounds = (0.0, 1.0),
    budget: float = 1.0,
) -> np.ndarray:
    """Maximum-return portfolio subject to a volatility cap.

    Solves max w'mu s.t. sqrt(w'Sigma w) <= target_vol, sum(w) = budget,
    box bounds. The vol constraint is imposed on the variance (smooth).

    Parameters
    ----------
    target_vol : float
        Per-period volatility cap (> 0).

    Returns
    -------
    np.ndarray
        (N,) optimal weights.
    """
    if target_vol <= 0:
        raise ValueError(f"target_vol must be > 0, got {target_vol}")
    sigma, m = _validate(cov, mu)
    assert m is not None
    n = sigma.shape[0]
    bl = _bounds_list(bounds, n)
    # start at the (feasibility-friendly) constrained min-variance point
    try:
        w0 = min_variance_constrained(sigma, bounds=bounds, budget=budget)
    except ValueError:
        w0 = _feasible_start(bl, n) * budget
    s = _scale(sigma)
    sig_s = sigma / s
    v2_s = target_vol**2 / s
    m_scale = float(np.max(np.abs(m))) or 1.0
    m_s = m / m_scale

    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - budget, "jac": lambda w: np.ones(n)},
        {"type": "ineq", "fun": lambda w: v2_s - float(w @ sig_s @ w),
         "jac": lambda w: -2.0 * sig_s @ w},
    ]
    res = minimize(
        lambda w: -float(w @ m_s), w0, jac=lambda w: -m_s, bounds=bl,
        constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-14},
    )
    if not res.success:
        raise ValueError(
            f"SLSQP failed in target_risk_portfolio (target_vol={target_vol:g}): "
            f"{res.message}"
        )
    return res.x


@dataclass(frozen=True)
class MaxSharpeResult:
    """Outcome of :func:`max_sharpe_qp`.

    Attributes
    ----------
    weights : np.ndarray
        (N,) weights summing to 1.
    fallback : bool
        True when no feasible portfolio has a positive expected excess
        return. The maximum-Sharpe problem then has no meaningful solution
        and the constrained minimum-variance portfolio is returned.
    sharpe : float
        Per-period Sharpe ratio (w'mu - rf) / sqrt(w'Sigma w) of ``weights``.
    kkt_residual : float
        Largest violation of the first-order conditions of the
        fully-invested, box-constrained Sharpe problem, divided by the
        Sharpe ratio (0 for a fallback).
    """

    weights: np.ndarray
    fallback: bool
    sharpe: float
    kkt_residual: float


def _lp_max_excess(excess: np.ndarray, bl: list[tuple[float, float]]) -> np.ndarray:
    """argmax w'excess over sum(w) = 1 and the box.

    Exact greedy solution of this LP: every asset starts at its lower
    bound and the remaining budget goes to assets in decreasing order of
    excess return.
    """
    lo = np.array([b[0] for b in bl])
    hi = np.array([b[1] for b in bl])
    if lo.sum() > 1.0 + 1e-12 or hi.sum() < 1.0 - 1e-12:
        raise ValueError("box bounds are infeasible with sum(w) = 1")
    w = lo.copy()
    rest = 1.0 - lo.sum()
    for i in np.argsort(-excess, kind="stable"):
        if rest <= 0:
            break
        add = min(hi[i] - lo[i], rest)
        w[i] += add
        rest -= add
    return w


def _sharpe_kkt_residual(
    w: np.ndarray, excess: np.ndarray, sigma: np.ndarray,
    bl: list[tuple[float, float]] | None,
) -> float:
    """First-order optimality residual of max S(w) s.t. sum(w) = 1 and box.

    g = dS/dw = (excess * var - (w'excess) * Sigma w) / var^1.5. At the
    maximum there is a scalar nu with g_i = nu on free assets, g_i <= nu
    at a lower bound and g_i >= nu at an upper bound. Returns the largest
    violation relative to |S|.
    """
    var = float(w @ sigma @ w)
    sr = float(w @ excess) / np.sqrt(var)
    g = (excess * var - float(w @ excess) * (sigma @ w)) / var**1.5
    if bl is None:
        at_lo = np.zeros_like(w, dtype=bool)
        at_hi = at_lo.copy()
    else:
        lo = np.array([b[0] for b in bl])
        hi = np.array([b[1] for b in bl])
        at_lo = w <= lo + 1e-9
        at_hi = w >= hi - 1e-9
    free = ~(at_lo | at_hi)
    if free.any():
        nu = float(np.mean(g[free]))
    else:  # every asset at a bound: any nu in [max g_lo, min g_hi] works
        lo_nu = float(np.max(g[at_lo])) if at_lo.any() else -np.inf
        hi_nu = float(np.min(g[at_hi])) if at_hi.any() else np.inf
        if np.isfinite(lo_nu) and np.isfinite(hi_nu):
            nu = 0.5 * (lo_nu + hi_nu)
        else:
            nu = lo_nu if np.isfinite(lo_nu) else hi_nu
    viol = np.zeros_like(g)
    viol[free] = np.abs(g[free] - nu)
    only_lo = at_lo & ~at_hi
    only_hi = at_hi & ~at_lo
    viol[only_lo] = np.maximum(g[only_lo] - nu, 0.0)
    viol[only_hi] = np.maximum(nu - g[only_hi], 0.0)
    return float(np.max(viol)) / max(abs(sr), 1e-300)


def max_sharpe_qp(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
    bounds: _Bounds = (0.0, 1.0),
) -> MaxSharpeResult:
    """Maximum-Sharpe fully-invested portfolio, solved as a convex QP.

    The ratio max (w'mu - rf) / sqrt(w'Sigma w) is not convex in w. With
    excess returns e = mu - rf and the substitution y = kappa * w with
    kappa > 0 (Cornuejols and Tutuncu 2007, sec. 8.2) it becomes

        min y'Sigma y  s.t.  e'y = 1,  sum(y) = kappa,
                             lo_i kappa <= y_i <= hi_i kappa,  kappa >= 0.

    This is a convex QP in (y, kappa), so its KKT point is the global
    optimum. The substitution needs a feasible w with e'w > 0. When none
    exists (long-only: max(mu) <= rf) every feasible Sharpe ratio is
    non-positive, and maximising it would push the book into the most
    volatile corner. The function then returns the constrained
    minimum-variance portfolio with ``fallback=True``.

    Gradients are analytic. The solution is checked against the
    first-order conditions of the original ratio problem, and the
    function raises ``ValueError`` if the relative KKT residual exceeds
    1e-6.

    With ``bounds=None`` the closed-form tangency portfolio is returned
    (it needs 1'Sigma^-1 e > 0).
    """
    sigma, m = _validate(cov, mu)
    assert m is not None
    if not np.isfinite(rf):
        raise ValueError(f"rf must be finite, got {rf}")
    n = sigma.shape[0]
    bl = _bounds_list(bounds, n)
    excess = m - float(rf)

    if bl is None:
        x = _solve_spd(sigma, excess)
        if not float(x.sum()) > 0:
            raise ValueError(
                "unconstrained tangency is undefined: 1'Sigma^-1 (mu - rf) <= 0"
            )
        w = x / float(x.sum())
    else:
        w_lp = _lp_max_excess(excess, bl)
        if not float(w_lp @ excess) > 0.0:
            w = min_variance_constrained(sigma, bounds=bounds)
            var = float(w @ sigma @ w)
            sr = float(w @ excess) / float(np.sqrt(var)) if var > 0 else 0.0
            return MaxSharpeResult(w, True, sr, 0.0)
        lo = np.array([b[0] for b in bl])
        hi = np.array([b[1] for b in bl])
        s = _scale(sigma)
        sig_s = sigma / s
        # e'y = c for any c > 0 gives the same w = y / kappa (homogeneity).
        # c = e'w_lp makes (y, kappa) = (w_lp, 1) feasible and O(1).
        e_s = excess / float(w_lp @ excess)
        z0 = np.append(w_lp, 1.0)

        def obj(z: np.ndarray) -> float:
            y = z[:n]
            return float(y @ sig_s @ y)

        def jac(z: np.ndarray) -> np.ndarray:
            return np.append(2.0 * sig_s @ z[:n], 0.0)

        a_e = np.append(e_s, 0.0)
        a_b = np.append(np.ones(n), -1.0)
        cons = [
            {"type": "eq", "fun": lambda z: float(a_e @ z) - 1.0, "jac": lambda z: a_e},
            {"type": "eq", "fun": lambda z: float(a_b @ z), "jac": lambda z: a_b},
        ]
        rows = []
        idx = np.flatnonzero(np.isfinite(lo))
        if idx.size:  # y_i - lo_i kappa >= 0
            a = np.zeros((idx.size, n + 1))
            a[np.arange(idx.size), idx] = 1.0
            a[:, n] = -lo[idx]
            rows.append(a)
        idx = np.flatnonzero(np.isfinite(hi))
        if idx.size:  # hi_i kappa - y_i >= 0
            a = np.zeros((idx.size, n + 1))
            a[np.arange(idx.size), idx] = -1.0
            a[:, n] = hi[idx]
            rows.append(a)
        if rows:
            a_in = np.vstack(rows)
            cons.append({"type": "ineq", "fun": lambda z: a_in @ z, "jac": lambda z: a_in})
        zb = [(None, None)] * n + [(0.0, None)]
        res = minimize(obj, z0, jac=jac, bounds=zb, constraints=cons,
                       method="SLSQP", options={"maxiter": 1000, "ftol": 1e-15})
        if not res.success:
            raise ValueError(f"SLSQP failed in max_sharpe_qp: {res.message}")
        kappa = float(res.x[n])
        if not kappa > 0:
            raise ValueError("max_sharpe_qp: degenerate solution with kappa <= 0")
        w = np.clip(res.x[:n] / kappa, lo, hi)  # remove ~1e-15 bound overshoot
    var = float(w @ sigma @ w)
    sr = float(w @ excess) / float(np.sqrt(var))
    kkt = _sharpe_kkt_residual(w, excess, sigma, bl)
    if kkt > 1e-6:
        raise ValueError(f"max_sharpe_qp: KKT residual {kkt:.2e} exceeds 1e-6")
    return MaxSharpeResult(w, False, sr, kkt)


def max_sharpe_constrained(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
    bounds: _Bounds = (0.0, 1.0),
) -> np.ndarray:
    """Maximum-Sharpe fully-invested portfolio under box bounds.

    Returns the weights from :func:`max_sharpe_qp`. When no feasible
    portfolio has a positive excess return this is the constrained
    minimum-variance portfolio. Call :func:`max_sharpe_qp` to see whether
    that fallback fired.
    """
    return max_sharpe_qp(mu, cov, rf=rf, bounds=bounds).weights


@dataclass(frozen=True)
class FrontierResult:
    """Efficient frontier sample.

    Attributes
    ----------
    returns : np.ndarray
        (K,) per-period expected returns of the frontier portfolios.
    vols : np.ndarray
        (K,) per-period volatilities.
    weights : np.ndarray
        (K, N) frontier portfolio weights.
    """

    returns: np.ndarray
    vols: np.ndarray
    weights: np.ndarray


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 25,
    bounds: _Bounds = None,
    max_return: float | None = None,
) -> FrontierResult:
    """Trace the efficient frontier from the min-variance return upward.

    * ``bounds=None``: ANALYTIC frontier. With A = 1'S^{-1}1,
      B = 1'S^{-1}mu, C = mu'S^{-1}mu, D = AC - B^2, the min-variance
      portfolio for target m is w(m) = lam * S^{-1}1 + gam * S^{-1}mu with
      lam = (C - B m)/D, gam = (A m - B)/D. Weights are affine in m:
      the two-fund theorem: every frontier portfolio is a combination of
      any two distinct frontier portfolios.
    * ``bounds`` given: NUMERIC frontier via
      :func:`target_return_portfolio` at each target.

    Parameters
    ----------
    mu, cov : array-like
        Per-period moments (cov must be PD for the analytic branch).
    n_points : int
        Number of frontier points (>= 2).
    bounds : see :func:`min_variance_constrained`.
    max_return : float, optional
        Upper end of the target grid. Defaults to max(mu) for the
        constrained case and min-var return + 1.5 * (max(mu) - min-var
        return) for the analytic case.

    Returns
    -------
    FrontierResult
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    sigma, m = _validate(cov, mu)
    assert m is not None

    if bounds is None:
        ones = np.ones_like(m)
        si_one = _solve_spd(sigma, ones)
        si_mu = _solve_spd(sigma, m)
        a = float(ones @ si_one)
        b = float(ones @ si_mu)
        c = float(m @ si_mu)
        d = a * c - b * b
        if d <= 0:
            raise ValueError(
                "degenerate frontier: mu is proportional to 1 (all assets have "
                "the same expected return) or cov is ill-conditioned"
            )
        r_min = b / a
        r_max = max_return if max_return is not None else r_min + 1.5 * (m.max() - r_min)
        if r_max <= r_min:
            r_max = r_min + abs(r_min) + 1e-6
        targets = np.linspace(r_min, r_max, n_points)
        lam = (c - b * targets) / d
        gam = (a * targets - b) / d
        w = lam[:, None] * si_one[None, :] + gam[:, None] * si_mu[None, :]
    else:
        w_mv = min_variance_constrained(sigma, bounds=bounds)
        r_min = float(w_mv @ m)
        r_max = max_return if max_return is not None else float(m.max())
        r_max = r_min + 0.999 * (r_max - r_min)  # stay strictly feasible
        targets = np.linspace(r_min, r_max, n_points)
        w = np.empty((n_points, sigma.shape[0]))
        w[0] = w_mv
        for i, tgt in enumerate(targets[1:], start=1):
            w[i] = target_return_portfolio(m, sigma, tgt, bounds=bounds)

    rets = w @ m
    vols = np.sqrt(np.einsum("ki,ij,kj->k", w, sigma, w))
    return FrontierResult(rets, vols, w)
