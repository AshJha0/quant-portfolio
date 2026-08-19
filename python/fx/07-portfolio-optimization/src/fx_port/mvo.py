"""Mean-variance optimization: closed forms + SLSQP with FX-native constraints.

Closed forms (validated to 1e-12 in tests):

* :func:`min_variance_weights` — global minimum-variance, sum(w)=1.
* :func:`tangency_weights` — maximum-Sharpe, sum(w)=1.
* :func:`frontier_weights` — minimum variance for a target mean (two-fund).
* :func:`dollar_neutral_weights` — max utility subject to sum(w)=0, the
  natural constraint for a long-short currency book.

Numerical (SLSQP) counterparts add the FX-native constraint set: net budget
(sum-to-zero for dollar-neutral, sum-to-one for fully collateralised
long-only vs a USD basket), a gross-leverage budget ``sum|w| <= gross`` and
per-currency bounds.  Units: ``mu`` per-period mean returns, ``sigma``
per-period covariance; utility is ``mu'w - (gamma/2) w'Sigma w``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _to_arrays(
    mu: pd.Series | np.ndarray | None, sigma: pd.DataFrame | np.ndarray
) -> tuple[np.ndarray | None, np.ndarray, list]:
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError(f"sigma must be square, got shape {s.shape}")
    if s.size == 0:
        raise ValueError("sigma is empty: no assets to optimise over")
    if not np.all(np.isfinite(s)):
        raise ValueError("sigma contains NaN/Inf; clean the covariance estimate first")
    labels = (
        list(sigma.index) if isinstance(sigma, pd.DataFrame) else list(range(len(s)))
    )
    m = None
    if mu is not None:
        m = np.asarray(mu, dtype=float).ravel()
        if len(m) != len(s):
            raise ValueError("mu and sigma dimensions differ")
        if not np.all(np.isfinite(m)):
            raise ValueError("mu contains NaN/Inf; clean the return estimate first")
    return m, s, labels


def _solve(sigma: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(sigma, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "sigma is singular; apply psd_repair(min_eig>0) first "
            "(e.g. pegged currency in the universe)"
        ) from exc


def min_variance_weights(sigma: pd.DataFrame) -> pd.Series:
    """Global minimum-variance weights ``Sigma^-1 1 / (1' Sigma^-1 1)``."""
    _, s, labels = _to_arrays(None, sigma)
    ones = np.ones(len(s))
    w = _solve(s, ones)
    return pd.Series(w / (ones @ w), index=labels, name="min_var")


def tangency_weights(
    mu: pd.Series, sigma: pd.DataFrame, rf: float = 0.0
) -> pd.Series:
    """Tangency (max-Sharpe) weights ``Sigma^-1 (mu-rf) / 1'Sigma^-1 (mu-rf)``.

    Raises
    ------
    ValueError
        If ``1'Sigma^-1(mu-rf) <= 0`` (tangency lies on the short side of the
        frontier; the normalised formula is meaningless there).
    """
    m, s, labels = _to_arrays(mu, sigma)
    ex = m - rf
    z = _solve(s, ex)
    denom = float(np.ones(len(s)) @ z)
    if denom <= 0:
        raise ValueError(
            "1'Sigma^-1(mu-rf) <= 0: no long tangency portfolio exists"
        )
    return pd.Series(z / denom, index=labels, name="tangency")


def frontier_weights(
    mu: pd.Series, sigma: pd.DataFrame, target: float
) -> pd.Series:
    """Closed-form minimum-variance weights with mean = ``target``, sum(w)=1.

    Standard two-fund solution with scalars A=1'S^-1 mu, B=mu'S^-1 mu,
    C=1'S^-1 1, D=BC-A^2.
    """
    m, s, labels = _to_arrays(mu, sigma)
    ones = np.ones(len(s))
    si_mu, si_one = _solve(s, m), _solve(s, ones)
    a, b, c = float(ones @ si_mu), float(m @ si_mu), float(ones @ si_one)
    d = b * c - a * a
    if abs(d) < 1e-300:
        raise ValueError("degenerate frontier: mu proportional to ones")
    lam = (c * target - a) / d
    gam = (b - a * target) / d
    return pd.Series(lam * si_mu + gam * si_one, index=labels, name="frontier")


def dollar_neutral_weights(
    mu: pd.Series, sigma: pd.DataFrame, gamma: float = 1.0
) -> pd.Series:
    """Closed-form max-utility long-short weights subject to ``sum(w) = 0``.

    Maximise ``mu'w - (gamma/2) w'Sigma w`` s.t. ``1'w = 0``:

    ``w* = (1/gamma) [ Sigma^-1 mu - (1'Sigma^-1 mu / 1'Sigma^-1 1) Sigma^-1 1 ]``.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    m, s, labels = _to_arrays(mu, sigma)
    ones = np.ones(len(s))
    si_mu, si_one = _solve(s, m), _solve(s, ones)
    w = (si_mu - (ones @ si_mu) / (ones @ si_one) * si_one) / gamma
    return pd.Series(w, index=labels, name="dollar_neutral")


# ---------------------------------------------------------------------------
# SLSQP with net-budget, gross-leverage and box constraints
# ---------------------------------------------------------------------------


@dataclass
class MVOResult:
    """Optimizer output: weights plus ex-ante per-period stats."""

    weights: pd.Series
    expected_return: float
    volatility: float
    success: bool
    message: str


def _constraints(n: int, sum_to: float | None) -> list[dict]:
    cons: list[dict] = []
    if sum_to is not None:
        cons.append(
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - sum_to,
                "jac": lambda w: np.ones(n),
            }
        )
    return cons


def _solve_split_qp(
    m: np.ndarray,
    s: np.ndarray,
    gamma: float,
    sum_to: float | None,
    gross_limit: float,
    bounds: tuple[float | None, float | None] | None,
    extra_eq: list[tuple[np.ndarray, float]] | None = None,
):
    """Solve max m'w - (gamma/2) w'Sw with sum|w| <= gross via w = p - q.

    The split makes every constraint linear and the problem smooth, which
    SLSQP solves reliably (the raw |w| constraint has a kink that trips its
    line search).  p, q >= 0; sum(p+q) <= gross; optional box on w = p - q.
    """
    n = len(m)
    # Tiny penalty on sum(p+q) breaks the flat direction p,q -> p+d,q+d
    # (which leaves w unchanged); when the gross constraint binds the penalty
    # is constant on the feasible face, so the optimum is not distorted.
    eps = 1e-9 * max(1.0, float(np.abs(m).max()))

    def obj(x: np.ndarray) -> float:
        w = x[:n] - x[n:]
        return 0.5 * gamma * w @ s @ w - m @ w + eps * x.sum()

    def jac(x: np.ndarray) -> np.ndarray:
        w = x[:n] - x[n:]
        g = gamma * s @ w - m
        return np.concatenate([g + eps, -g + eps])

    cons: list[dict] = [
        {
            "type": "ineq",
            "fun": lambda x: gross_limit - x.sum(),
            "jac": lambda x: -np.ones(2 * n),
        }
    ]
    if sum_to is not None:
        row = np.concatenate([np.ones(n), -np.ones(n)])
        cons.append(
            {
                "type": "eq",
                "fun": lambda x: row @ x - sum_to,
                "jac": lambda x: row,
            }
        )
    if bounds is not None:
        lo, hi = bounds
        row = np.concatenate([np.eye(n), -np.eye(n)], axis=1)
        if hi is not None:
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda x: hi - row @ x,
                    "jac": lambda x: -row,
                }
            )
        if lo is not None:
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda x: row @ x - lo,
                    "jac": lambda x: row,
                }
            )
    for vec, val in extra_eq or []:
        srow = np.concatenate([vec, -vec])
        cons.append(
            {
                "type": "eq",
                "fun": lambda x, r=srow, v=val: r @ x - v,
                "jac": lambda x, r=srow: r,
            }
        )
    net = abs(sum_to) if sum_to is not None else 0.0
    slack = 0.25 * max(gross_limit - net, 0.0) / n
    x0 = np.full(2 * n, slack)
    if sum_to is not None:
        x0[:n] += max(sum_to, 0.0) / n
        x0[n:] += max(-sum_to, 0.0) / n
    res = minimize(
        obj,
        x0,
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, None)] * (2 * n),
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return res.x[:n] - res.x[n:], res


def _check_feasible(
    sum_to: float | None,
    gross_limit: float | None,
    bounds: tuple[float | None, float | None] | None = None,
    n: int | None = None,
) -> None:
    """Reject constraint sets with an empty feasible region, with a reason.

    Two ways an FX mandate becomes infeasible in practice:

    * a gross-leverage budget below the required net budget (you cannot hold
      net 100% with only 50% of gross allowed), and
    * a per-currency box that cannot reach the net budget (e.g. 8 currencies
      capped at 10% each cannot sum to 1.0).

    Silently returning the solver's failed iterate would put weights that
    violate the mandate in front of a trader, so both raise ``ValueError``.
    """
    if (
        gross_limit is not None
        and sum_to is not None
        and gross_limit < abs(sum_to) - 1e-12
    ):
        raise ValueError(
            f"infeasible: gross_limit={gross_limit} < |sum_to|={abs(sum_to)}"
        )
    if bounds is not None and sum_to is not None and n is not None:
        lo, hi = bounds
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"infeasible box: lower bound {lo} > upper bound {hi}")
        if hi is not None and n * hi < sum_to - 1e-12:
            raise ValueError(
                f"infeasible box: {n} weights capped at {hi} cannot sum to "
                f"{sum_to} (max attainable {n * hi})"
            )
        if lo is not None and n * lo > sum_to + 1e-12:
            raise ValueError(
                f"infeasible box: {n} weights floored at {lo} cannot sum to "
                f"{sum_to} (min attainable {n * lo})"
            )


def max_utility(
    mu: pd.Series,
    sigma: pd.DataFrame,
    gamma: float = 1.0,
    sum_to: float | None = 0.0,
    gross_limit: float | None = None,
    bounds: tuple[float | None, float | None] | None = None,
    x0: np.ndarray | None = None,
) -> MVOResult:
    """SLSQP maximisation of ``mu'w - (gamma/2) w'Sigma w`` under constraints.

    Parameters
    ----------
    mu, sigma : per-period mean vector and covariance.
    gamma : float
        Risk aversion > 0.
    sum_to : float or None
        Net-budget equality (0 = dollar-neutral, 1 = fully invested,
        None = unconstrained net).
    gross_limit : float or None
        Gross-leverage budget ``sum|w| <= gross_limit`` (None = off).
    bounds : (lo, hi) or None
        Per-currency box applied to every weight.
    x0 : np.ndarray, optional
        Starting point (default: feasible uniform point).

    Returns
    -------
    MVOResult
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    m, s, labels = _to_arrays(mu, sigma)
    n = len(s)
    _check_feasible(sum_to, gross_limit, bounds, n)
    if gross_limit is not None and gross_limit == 0:
        w = pd.Series(np.zeros(n), index=labels)
        return MVOResult(w, 0.0, 0.0, True, "gross_limit=0: zero portfolio")

    # Rescale mu and Sigma jointly (argmax invariant) so the objective is
    # O(1): daily FX variances are ~1e-5 and would starve SLSQP's ftol.
    scale = 1.0 / max(float(np.mean(np.diag(s))), 1e-300)
    s_sc, m_sc = s * scale, m * scale

    if gross_limit is not None:
        w_arr, res = _solve_split_qp(m_sc, s_sc, gamma, sum_to, gross_limit, bounds)
    else:
        def obj(w: np.ndarray) -> float:
            return 0.5 * gamma * w @ s_sc @ w - m_sc @ w

        def jac(w: np.ndarray) -> np.ndarray:
            return gamma * s_sc @ w - m_sc

        if x0 is None:
            # Warm start from the analytic optimum of the equality-constrained
            # problem — SLSQP then only has to polish.
            try:
                w_free = np.linalg.solve(gamma * s_sc, m_sc)
                if sum_to is not None:
                    ones = np.ones(n)
                    si_one = np.linalg.solve(gamma * s_sc, ones)
                    w_free = (
                        w_free + (sum_to - ones @ w_free) / (ones @ si_one) * si_one
                    )
                x0 = w_free
            except np.linalg.LinAlgError:
                x0 = np.full(n, (sum_to or 0.0) / n)
            if bounds is not None:
                lo = -np.inf if bounds[0] is None else bounds[0]
                hi = np.inf if bounds[1] is None else bounds[1]
                x0 = np.clip(x0, lo, hi)
        res = minimize(
            obj,
            x0,
            jac=jac,
            method="SLSQP",
            bounds=None if bounds is None else [bounds] * n,
            constraints=_constraints(n, sum_to),
            options={"maxiter": 1000, "ftol": 1e-14},
        )
        w_arr = res.x
    w = pd.Series(w_arr, index=labels, name="max_utility")
    return MVOResult(
        weights=w,
        expected_return=float(m @ w_arr),
        volatility=float(np.sqrt(max(w_arr @ s @ w_arr, 0.0))),
        success=bool(res.success),
        message=str(res.message),
    )


def min_variance_slsqp(
    sigma: pd.DataFrame,
    sum_to: float = 1.0,
    gross_limit: float | None = None,
    bounds: tuple[float | None, float | None] | None = None,
    target_return: float | None = None,
    mu: pd.Series | None = None,
) -> MVOResult:
    """SLSQP minimum variance, optionally with a target-mean equality.

    With no gross/box constraints this matches :func:`min_variance_weights`
    (or :func:`frontier_weights` when ``target_return`` is set) to high
    precision — used as a cross-check in the tests.
    """
    m, s, labels = _to_arrays(mu, sigma)
    n = len(s)
    _check_feasible(sum_to, gross_limit, bounds, n)
    # Scale-invariant objective (see max_utility): keeps SLSQP's ftol honest.
    scale = 1.0 / max(float(np.mean(np.diag(s))), 1e-300)
    s_sc = s * scale
    if target_return is not None and m is None:
        raise ValueError("target_return requires mu")

    if gross_limit is not None:
        extra = (
            [(m.copy(), float(target_return))] if target_return is not None else None
        )
        w_arr, res = _solve_split_qp(
            np.zeros(n), s_sc, 2.0, sum_to, gross_limit, bounds, extra_eq=extra
        )
    else:
        def obj(w: np.ndarray) -> float:
            return float(w @ s_sc @ w)

        def jac(w: np.ndarray) -> np.ndarray:
            return 2.0 * s_sc @ w

        cons = _constraints(n, sum_to)
        if target_return is not None:
            cons.append(
                {
                    "type": "eq",
                    "fun": lambda w: m @ w - target_return,
                    "jac": lambda w: m,
                }
            )
        x0 = np.full(n, sum_to / n)
        res = minimize(
            obj,
            x0,
            jac=jac,
            method="SLSQP",
            bounds=None if bounds is None else [bounds] * n,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-16},
        )
        w_arr = res.x
    w = pd.Series(w_arr, index=labels, name="min_var_slsqp")
    er = float(m @ w_arr) if m is not None else float("nan")
    return MVOResult(
        weights=w,
        expected_return=er,
        volatility=float(np.sqrt(max(w_arr @ s @ w_arr, 0.0))),
        success=bool(res.success),
        message=str(res.message),
    )


def efficient_frontier(
    mu: pd.Series,
    sigma: pd.DataFrame,
    n_points: int = 25,
    sum_to: float = 1.0,
    gross_limit: float | None = None,
    bounds: tuple[float | None, float | None] | None = None,
) -> pd.DataFrame:
    """Efficient frontier: min vol for a grid of target means.

    Targets run from the (constrained) minimum-variance portfolio's mean to
    the maximum single-currency mean (long-only style upper end).  Uses the
    closed form when unconstrained, SLSQP otherwise.

    Returns
    -------
    pd.DataFrame
        One row per target: columns ``target_return``, ``volatility`` and one
        weight column per currency.  Volatility is non-decreasing in the
        target above the min-var point (tested).
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    m, s, labels = _to_arrays(mu, sigma)
    # The closed-form two-fund frontier assumes sum(w) = 1; any other net
    # budget or extra constraints go through SLSQP.
    unconstrained = gross_limit is None and bounds is None and sum_to == 1.0
    if unconstrained:
        base = min_variance_weights(sigma)
        mu_lo = float(m @ base.to_numpy())
    else:
        mv = min_variance_slsqp(
            sigma, sum_to=sum_to, gross_limit=gross_limit, bounds=bounds
        )
        mu_lo = float(m @ mv.weights.to_numpy())
    mu_hi = float(np.max(m))
    if mu_hi <= mu_lo:
        mu_hi = mu_lo + abs(mu_lo) + 1e-6
    rows = []
    for tgt in np.linspace(mu_lo, mu_hi, n_points):
        if unconstrained:
            w = frontier_weights(mu, sigma, float(tgt))
            vol = float(np.sqrt(w.to_numpy() @ s @ w.to_numpy()))
            rows.append([tgt, vol, *w.to_numpy()])
        else:
            r = min_variance_slsqp(
                sigma,
                sum_to=sum_to,
                gross_limit=gross_limit,
                bounds=bounds,
                target_return=float(tgt),
                mu=mu,
            )
            if r.success:
                rows.append([tgt, r.volatility, *r.weights.to_numpy()])
    return pd.DataFrame(
        rows, columns=["target_return", "volatility", *labels]
    )
