"""Order placement across venues: Cont and Kukanov's problem, solved as an LP.

Plain idea
----------
We must buy ``S`` lots soon. We can cross the spread now with a market order
(certain fill, cost ``h + f`` per lot against the mid) or post limit orders on
several venues (cheaper or even paid, but they fill only if enough sell flow
arrives to eat through the queue ahead of us). Whatever is still unfilled at
the horizon costs ``lambda_u`` per lot (we chase it later). The question is
how much to send where.

Formal statement
----------------
Cont and Kukanov (2017, "Optimal order placement in limit order markets",
Quantitative Finance 17(1)) pose: choose ``M >= 0`` (market order) and
``L_k >= 0`` (limit order on venue ``k``) to minimise the expected cost

    E[ (h+f) M + sum_k c_k min(L_k, D_k) + lambda_u (S - A)^+ + lambda_o (A - S)^+ ]

with ``A = M + sum_k min(L_k, D_k)`` the filled quantity and
``D_k = (xi_k - Q_k)^+`` the flow that reaches us behind the queue ``Q_k``.
For a lit venue ``c_k = -(h + r_k)`` (we earn the half spread and the maker
rebate ``r_k``); we also allow an adverse-selection charge so that
``c_k = -(h + r_k) + a_k``, and a dark midpoint venue enters with
``c_k = fee_k + a_k`` (no spread captured or paid against the mid).

**Departure from the paper, stated plainly.** We impose the no-oversubscription
rule ``M + sum_k L_k <= S``: a router never has more child quantity working
than the parent has left, which is how most desks run it to avoid overfills.
Then ``A <= S`` always, the ``lambda_o`` term is identically zero, and the
objective becomes

    (h+f-lambda_u) M + sum_k (c_k - lambda_u) E min(L_k, D_k) + lambda_u S.

When every ``c_k <= lambda_u`` this is convex, and with ``N`` scenarios for
``D`` (sample average approximation) it is exactly the linear program

    min (h+f-lambda_u) M + (1/N) sum_{n,k} (c_k - lambda_u) y_nk
    s.t. y_nk <= L_k, y_nk <= D_nk, y_nk >= 0, M + sum L_k <= S, M, L >= 0,

exact because negative coefficients push every ``y_nk`` up to
``min(L_k, D_nk)``. We solve it with HiGHS through ``scipy.optimize.linprog``.
The paper's oversubscribed solutions (``M + sum L > S``) are therefore out of
scope; the analytic special cases below are derived for this variant.

Worked example
--------------
One venue, ``h=0.5, f=0.3, r=0.2, lambda_u=2`` (cents per share), flow behind
the queue uniform on {0..9}. The single-venue rule (``single_venue_optimum``)
says post ``L*`` with ``F(L*) = (2h+f+r)/(lambda_u+h+r) = 1.5/2.7 = 0.556``,
so ``L* = 5`` and ``M* = S - 5``:

>>> import numpy as np
>>> D = np.arange(10.0).reshape(-1, 1)
>>> a = solve_allocation(S=8, h=0.5, f=0.3, c=[-0.7], lam_u=2.0, D=D)
>>> round(a.market, 6), [round(x, 6) for x in a.limits]
(3.0, [5.0])
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from qp_core.model import model
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Allocation:
    """Optimal split. ``cost`` is the expected cost per the objective above
    (in the units of ``h``), over the whole parent, against the mid."""

    market: float
    limits: tuple[float, ...]
    cost: float
    exact: bool  # every scenario fill sits at min(L_k, D_nk) to 1e-9


@model(id="EXEC-SOR-ALLOC")
def expected_cost(market: float, limits: Sequence[float], *, S: float, h: float, f: float,
                  c: Sequence[float], lam_u: float, D: ArrayLike) -> float:
    """The objective evaluated directly (no LP), for checks and plots."""
    d = np.asarray(D, dtype=float)
    lim = np.asarray(limits, dtype=float)
    fills = np.minimum(lim[None, :], d)
    a = market + fills.sum(axis=1)
    cost = (h + f) * market + fills @ np.asarray(c, float) + lam_u * np.maximum(S - a, 0.0)
    return float(cost.mean())


@model(id="EXEC-SOR-ALLOC")
def solve_allocation(*, S: float, h: float, f: float, c: Sequence[float], lam_u: float,
                     D: ArrayLike) -> Allocation:
    """Solve the no-oversubscription Cont-Kukanov LP on scenarios ``D`` (N x K).

    Raises ``ValueError`` if a venue has ``c_k > lambda_u``: the problem is
    then not convex (posting there is worse than not filling at all), and
    such a venue should simply be dropped by the caller.
    """
    d = np.asarray(D, dtype=float)
    if d.ndim != 2:
        raise ValueError("D must be N x K")
    n, k = d.shape
    cv = np.asarray(c, dtype=float)
    if cv.shape != (k,):
        raise ValueError("c must have one entry per venue")
    if np.any(cv > lam_u):
        raise ValueError("c_k > lambda_u makes the problem non-convex; drop that venue")
    if np.any(d < 0) or S <= 0:
        raise ValueError("D must be >= 0 and S > 0")
    # variables: [M, L_1..L_K, y_11..y_1K, ..., y_N1..y_NK]
    nv = 1 + k + n * k
    obj = np.empty(nv)
    obj[0] = h + f - lam_u
    obj[1:1 + k] = 0.0
    obj[1 + k:] = np.tile((cv - lam_u) / n, n)
    # y_nk - L_k <= 0
    rows = np.repeat(np.arange(n * k), 2)
    cols = np.empty(2 * n * k, dtype=int)
    cols[0::2] = 1 + k + np.arange(n * k)
    cols[1::2] = 1 + np.tile(np.arange(k), n)
    vals = np.tile([1.0, -1.0], n * k)
    a_ub = coo_matrix((vals, (rows, cols)), shape=(n * k + 1, nv)).tolil()
    a_ub[n * k, 0:1 + k] = 1.0  # M + sum L <= S
    b_ub = np.zeros(n * k + 1)
    b_ub[n * k] = S
    bounds = [(0, None)] * (1 + k) + [(0, float(x)) for x in d.ravel()]
    res = linprog(obj, A_ub=a_ub.tocsr(), b_ub=b_ub, bounds=bounds, method="highs")  # type: ignore[call-overload]
    if res.status != 0:  # pragma: no cover - the LP is always feasible and bounded
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x
    m, lim = float(x[0]), x[1:1 + k]
    y = x[1 + k:].reshape(n, k)
    exact = bool(np.allclose(y, np.minimum(lim[None, :], d), atol=1e-9))
    cost = float(res.fun) + lam_u * S
    return Allocation(m, tuple(float(v) for v in lim), cost, exact)


@model(id="EXEC-SOR-ALLOC")
def single_venue_optimum(*, S: float, h: float, f: float, r: float, lam_u: float,
                         D: ArrayLike) -> tuple[float, float]:
    """Closed form for one lit venue (the variant above), as ``(M*, L*)``.

    If ``lambda_u > h + f`` the whole budget is used (``M = S - L``) and the
    first-order condition ``-(h+f-lambda_u) + (c-lambda_u) P(D > L) = 0`` with
    ``c = -(h+r)`` gives the quantile rule ``F_D(L*) = (2h+f+r)/(lambda_u+h+r)``,
    capped at ``S``. For an empirical ``F`` we return the smallest sample
    point meeting the rule (the LP may return any point of the flat interval;
    tests use distributions where it is unique). If ``lambda_u <= h+f`` the
    market order is never worth it and ``M* = 0``; posting ``S`` is optimal.
    """
    d = np.sort(np.asarray(D, dtype=float).ravel())
    if lam_u <= h + f:
        return 0.0, float(S)
    q = (2 * h + f + r) / (lam_u + h + r)
    # smallest L with F(L) >= q, F(L) = P(D <= L)
    idx = int(np.ceil(q * len(d) - 1e-12)) - 1
    lstar = min(float(d[max(idx, 0)]), float(S))
    return float(S) - lstar, lstar


@model(id="EXEC-SOR-ALLOC")
def deterministic_optimum(*, S: float, h: float, f: float, c: Sequence[float],
                          lam_u: float, D: Sequence[float]) -> tuple[float, tuple[float, ...]]:
    """Closed form when the flow behind each queue is known: fill greedily
    from the cheapest venue (most negative ``c_k``) up to ``D_k``, then send
    the rest as a market order if ``h + f < lambda_u``."""
    left = float(S)
    lim = [0.0] * len(c)
    for i in sorted(range(len(c)), key=lambda j: c[j]):
        if c[i] - lam_u >= 0 or c[i] >= h + f:
            continue  # never better than the market order or the penalty
        take = min(left, float(D[i]))
        lim[i] = take
        left -= take
    m = left if h + f < lam_u else 0.0
    return m, tuple(lim)
