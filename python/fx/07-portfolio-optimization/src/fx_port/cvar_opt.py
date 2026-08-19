"""CVaR-constrained portfolio optimization via the Rockafellar-Uryasev LP.

Definitions (losses are NEGATED returns, ``L = -r``):

* ``VaR_a`` = a-quantile of the loss distribution.
* ``CVaR_a = min_z  z + E[(L - z)+] / (1 - a)``  (Rockafellar & Uryasev 2000).
  At the optimum, z* is a VaR_a and CVaR_a is the expected loss in the
  (1-a) tail.  For an empirical distribution with S scenarios the inner
  minimisation is attained at one of the scenario losses, which is exactly
  how :func:`empirical_cvar` evaluates it — guaranteeing that the LP
  objective at the optimum EQUALS the empirical CVaR of the optimal weights
  (tested to 1e-9).

Because ``(L-z)+`` is piecewise linear, both "minimise CVaR" and "maximise
mean subject to CVaR <= limit" are LINEAR programs in
``(w, z, u_1..u_S)`` with ``u_s >= L_s(w) - z, u_s >= 0``:

    CVaR_a(w)  <=  z + (1/((1-a) S)) sum_s u_s .

We solve them with ``scipy.optimize.linprog`` (HiGHS).  To support a
gross-leverage budget ``sum|w| <= gross`` the weight is split
``w = w+ - w-`` with ``w+, w- >= 0`` — standard LP linearisation.

Why this matters for FX: mean-variance treats carry's premium as free money
because variance ignores skew.  The carry-crash tail lives exactly where
CVaR looks, so a CVaR constraint cuts carry sizing in a way vol targeting
cannot (demonstrated with numbers in the pipeline and VALIDATION.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def empirical_cvar(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    """Empirical CVaR (expected shortfall) at level ``alpha``, loss units.

    Evaluates the Rockafellar-Uryasev functional
    ``min_z z + mean((L - z)+)/(1-alpha)`` exactly by scanning the scenario
    losses (the minimiser is attained at a data point).  Positive output =
    expected LOSS in the tail (so a CVaR of 0.02 means -2% average tail
    return).

    Parameters
    ----------
    returns : array-like
        Return scenarios (not losses).
    alpha : float
        Confidence level in (0, 1), e.g. 0.95.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    losses = -np.asarray(returns, dtype=float).ravel()
    if losses.size == 0:
        raise ValueError("empty return sample")
    if not np.all(np.isfinite(losses)):
        raise ValueError("return scenarios contain NaN/Inf; clean the sample first")
    cand = np.unique(losses)
    vals = cand + np.maximum(losses[None, :] - cand[:, None], 0.0).mean(axis=1) / (
        1.0 - alpha
    )
    return float(vals.min())


def empirical_var(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    """Empirical VaR (loss units): the alpha-quantile of losses (higher interpolation)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    losses = -np.asarray(returns, dtype=float).ravel()
    if losses.size == 0:
        raise ValueError("empty return sample")
    if not np.all(np.isfinite(losses)):
        raise ValueError("return scenarios contain NaN/Inf; clean the sample first")
    return float(np.quantile(losses, alpha, method="higher"))


@dataclass
class CVaRResult:
    """LP output.

    Attributes
    ----------
    weights : pd.Series
        Optimal weights.
    cvar : float
        Rockafellar-Uryasev objective value at the optimum (== empirical
        CVaR of the optimal portfolio, tested).
    var : float
        The optimal auxiliary z (a VaR of the optimal portfolio).
    expected_return : float
        Scenario-mean return of the optimal portfolio (per period).
    success : bool
    message : str
    """

    weights: pd.Series
    cvar: float
    var: float
    expected_return: float
    success: bool
    message: str


def _build_lp(
    r: np.ndarray,
    alpha: float,
    sum_to: float | None,
    gross_limit: float | None,
    max_weight: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, int, int]:
    """Common constraint blocks for the RU LPs.

    Variable vector: ``[w+ (n), w- (n), z (1), u (S)]``.
    Returns (A_ub, b_ub, A_eq, b_eq, bounds, n, S); A_eq may be empty.
    """
    if r.ndim != 2 or r.size == 0:
        raise ValueError("scenarios must be a non-empty 2-D panel (rows=dates)")
    if not np.all(np.isfinite(r)):
        raise ValueError(
            "scenarios contain NaN/Inf; drop or fill missing scenario rows "
            "before building the CVaR LP"
        )
    s_n, n = r.shape
    n_var = 2 * n + 1 + s_n
    # u_s >= L_s - z  <=>  -r_s'(w+ - w-) - z - u_s <= 0
    a_tail = np.zeros((s_n, n_var))
    a_tail[:, :n] = -r
    a_tail[:, n : 2 * n] = r
    a_tail[:, 2 * n] = -1.0
    a_tail[:, 2 * n + 1 :] = -np.eye(s_n)
    rows = [a_tail]
    b_rows = [np.zeros(s_n)]
    if gross_limit is not None:
        if gross_limit < 0:
            raise ValueError(f"gross_limit must be >= 0, got {gross_limit}")
        g = np.zeros(n_var)
        g[: 2 * n] = 1.0
        rows.append(g[None, :])
        b_rows.append(np.array([gross_limit]))
    a_ub = np.vstack(rows)
    b_ub = np.concatenate(b_rows)
    if sum_to is not None:
        a_eq = np.zeros((1, n_var))
        a_eq[0, :n] = 1.0
        a_eq[0, n : 2 * n] = -1.0
        b_eq = np.array([sum_to])
    else:
        a_eq = np.zeros((0, n_var))
        b_eq = np.zeros(0)
    hi = None if max_weight is None else max_weight
    bounds = (
        [(0, hi)] * (2 * n) + [(None, None)] + [(0, None)] * s_n
    )
    return a_ub, b_ub, a_eq, b_eq, bounds, n, s_n


def _unpack(
    x: np.ndarray, n: int, s_n: int, alpha: float, r: np.ndarray, labels: list
) -> tuple[pd.Series, float, float, float]:
    w = x[:n] - x[n : 2 * n]
    z = float(x[2 * n])
    u = x[2 * n + 1 :]
    cvar = z + float(u.sum()) / ((1.0 - alpha) * s_n)
    port = r @ w
    return pd.Series(w, index=labels, name="weights"), cvar, z, float(port.mean())


def min_cvar(
    scenarios: pd.DataFrame,
    alpha: float = 0.95,
    sum_to: float | None = 1.0,
    gross_limit: float | None = None,
    max_weight: float | None = None,
    return_floor: float | None = None,
) -> CVaRResult:
    """Minimise portfolio CVaR over historical scenarios (RU linear program).

    Parameters
    ----------
    scenarios : pd.DataFrame
        Historical return scenarios, rows = dates, columns = assets.
    alpha : float
        CVaR confidence level.
    sum_to : float or None
        Net budget equality (1 = fully invested, 0 = dollar-neutral,
        None = free net).
    gross_limit : float or None
        ``sum|w| <= gross_limit``.
    max_weight : float or None
        Upper bound on each of w+ and w- (per-currency limit).
    return_floor : float or None
        Optional constraint ``mean(scenarios) @ w >= return_floor``.

    Returns
    -------
    CVaRResult
    """
    r = scenarios.to_numpy(dtype=float)
    a_ub, b_ub, a_eq, b_eq, bounds, n, s_n = _build_lp(
        r, alpha, sum_to, gross_limit, max_weight
    )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if return_floor is not None:
        mu = r.mean(axis=0)
        g = np.zeros(a_ub.shape[1])
        g[:n], g[n : 2 * n] = -mu, mu
        a_ub = np.vstack([a_ub, g])
        b_ub = np.concatenate([b_ub, [-return_floor]])
    c = np.zeros(a_ub.shape[1])
    c[2 * n] = 1.0
    c[2 * n + 1 :] = 1.0 / ((1.0 - alpha) * s_n)
    res = linprog(
        c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq if len(b_eq) else None,
        b_eq=b_eq if len(b_eq) else None, bounds=bounds, method="highs",
    )
    if not res.success:
        raise ValueError(f"CVaR LP infeasible or failed: {res.message}")
    w, cvar, z, er = _unpack(res.x, n, s_n, alpha, r, list(scenarios.columns))
    return CVaRResult(w, cvar, z, er, True, str(res.message))


def max_return_cvar_constrained(
    scenarios: pd.DataFrame,
    alpha: float = 0.95,
    cvar_limit: float = 0.02,
    mu: pd.Series | None = None,
    sum_to: float | None = 0.0,
    gross_limit: float | None = 2.0,
    max_weight: float | None = None,
) -> CVaRResult:
    """Maximise expected return subject to ``CVaR_alpha(w) <= cvar_limit``.

    This is the skew-aware sizing problem: the constraint prices the crash
    tail that variance ignores, so high-carry books get cut relative to
    unconstrained mean-variance sizing.

    Parameters
    ----------
    scenarios : pd.DataFrame
        Historical return scenarios (used BOTH for the tail constraint and,
        if ``mu`` is None, for expected returns).
    alpha, sum_to, gross_limit, max_weight : see :func:`min_cvar`.
    cvar_limit : float
        Per-period CVaR budget (loss units, e.g. 0.02 = 2%).
    mu : pd.Series, optional
        Expected returns overriding scenario means (e.g. shrunk means).

    Returns
    -------
    CVaRResult
        The RU objective (``result.cvar``) equals the empirical CVaR of the
        optimal weights whenever the constraint binds; it is always an upper
        bound on it.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    r = scenarios.to_numpy(dtype=float)
    a_ub, b_ub, a_eq, b_eq, bounds, n, s_n = _build_lp(
        r, alpha, sum_to, gross_limit, max_weight
    )
    m = (
        r.mean(axis=0)
        if mu is None
        else np.asarray(mu.reindex(scenarios.columns), dtype=float)
    )
    if np.any(np.isnan(m)):
        raise ValueError("mu missing some scenario columns")
    # CVaR budget row: z + sum(u)/((1-a)S) <= limit
    g = np.zeros(a_ub.shape[1])
    g[2 * n] = 1.0
    g[2 * n + 1 :] = 1.0 / ((1.0 - alpha) * s_n)
    a_ub = np.vstack([a_ub, g])
    b_ub = np.concatenate([b_ub, [cvar_limit]])
    c = np.zeros(a_ub.shape[1])
    c[:n], c[n : 2 * n] = -m, m  # maximise m'w
    res = linprog(
        c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq if len(b_eq) else None,
        b_eq=b_eq if len(b_eq) else None, bounds=bounds, method="highs",
    )
    if not res.success:
        raise ValueError(f"CVaR LP infeasible or failed: {res.message}")
    w, cvar, z, er_scen = _unpack(res.x, n, s_n, alpha, r, list(scenarios.columns))
    er = float(m @ w.to_numpy())
    return CVaRResult(w, cvar, z, er, True, str(res.message))


def carry_sizing(
    style_returns: pd.Series,
    alpha: float = 0.95,
    cvar_limit: float = 0.02,
    max_leverage: float = 10.0,
) -> tuple[float, float]:
    """Skew-aware sizing of a single style sleeve under a CVaR budget.

    CVaR is positively homogeneous: ``CVaR(s * r) = s * CVaR(r)`` for
    ``s >= 0``.  If the style has positive mean and positive CVaR, the
    mean-maximising size under ``CVaR <= limit`` is simply

        s* = min(max_leverage, cvar_limit / CVaR_alpha(r)) .

    Returns
    -------
    (float, float)
        Optimal size s* and the resulting portfolio CVaR ``s* * CVaR(r)``.
    """
    if max_leverage < 0:
        raise ValueError(f"max_leverage must be >= 0, got {max_leverage}")
    base = empirical_cvar(style_returns, alpha)
    if base <= 0:  # no tail loss at this level: constraint never binds
        s = max_leverage
    else:
        s = min(max_leverage, cvar_limit / base)
    return float(s), float(s * base)
