"""Rating migration matrices and lifetime PD.

Cohort estimator
----------------
Count obligors moving from grade i at the start of a year to grade j at the
end; P_ij = N_ij / N_i. Default is the last state and is absorbing.
Reference: Jafry and Schuermann (2004), "Measurement, estimation and
comparison of credit migration matrices", J. Banking & Finance 28, eq. (1).

Multi-period and lifetime PD
----------------------------
Under a time-homogeneous Markov chain the t-year matrix is P^t, and the
cumulative PD of grade i by year t is (P^t)_{i,D}. The marginal PD of year t
is the increment of the cumulative PD. This is the IFRS 9 lifetime PD
building block (IFRS 9 para 5.5.3 and B5.5.37).

Generator
---------
A continuous-time generator Q with exp(Q) = P gives PDs for fractional
horizons. The principal matrix logarithm can have negative off-diagonal
entries; we apply the Israel, Rosenthal and Wei (2001) fix (Mathematical
Finance 11, section 3.1, "weighted adjustment"): set negative off-diagonals
to zero and remove the added mass from the row's positive entries in
proportion to their size, keeping rows summing to zero.
Reference for the generator approach: Jarrow, Lando and Turnbull (1997),
Rev. Financial Studies 10, section 3.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, logm


def _check_stochastic(p: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    m = np.asarray(p, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("transition matrix must be square")
    if np.any(m < -tol) or not np.allclose(m.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("transition matrix rows must be non-negative and sum to 1")
    return m


def cohort_matrix(start: np.ndarray, end: np.ndarray, n_states: int) -> np.ndarray:
    """Cohort estimate of a one-period matrix; the last state (default) is absorbing.

    Rows with no observations are set to the identity row.
    """
    s = np.asarray(start, dtype=int)
    e = np.asarray(end, dtype=int)
    if s.shape != e.shape:
        raise ValueError("start and end must have the same shape")
    if s.size and (s.min() < 0 or e.min() < 0 or s.max() >= n_states or e.max() >= n_states):
        raise ValueError("states must lie in [0, n_states)")
    counts = np.zeros((n_states, n_states))
    np.add.at(counts, (s, e), 1.0)
    totals = counts.sum(axis=1)
    p = np.eye(n_states)
    nz = totals > 0
    p[nz] = counts[nz] / totals[nz, None]
    p[-1] = 0.0
    p[-1, -1] = 1.0
    return p


def cumulative_pd(p: np.ndarray, horizon: int) -> np.ndarray:
    """Cumulative PD by year: array (horizon, n_states - 1), row t-1 = (P^t)[:, D]."""
    m = _check_stochastic(p)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    out = np.empty((horizon, m.shape[0] - 1))
    pt = np.eye(m.shape[0])
    for k in range(horizon):
        pt = pt @ m
        out[k] = pt[:-1, -1]
    return out


def marginal_pd(p: np.ndarray, horizon: int) -> np.ndarray:
    """Unconditional default probability in each year (increments of cumulative PD)."""
    cum = cumulative_pd(p, horizon)
    return np.diff(np.vstack([np.zeros((1, cum.shape[1])), cum]), axis=0)


def generator(p: np.ndarray) -> np.ndarray:
    """Valid generator Q with exp(Q) close to P (principal log + IRW weighted fix)."""
    m = _check_stochastic(p)
    q = np.real(logm(m))
    n = q.shape[0]
    for i in range(n):
        row = q[i].copy()
        off = np.arange(n) != i
        neg = off & (row < 0.0)
        if not neg.any():
            continue
        b = float(-np.sum(row[neg]))
        row[neg] = 0.0
        g = abs(row[i]) + float(np.sum(row[off]))
        # Zeroing the negatives leaves the row summing to +b; take b back
        # from the remaining entries in proportion to |q_ij| (IRW 2001).
        row = row - np.abs(row) * b / g
        q[i] = row
    return q


def pd_at(q: np.ndarray, t: float) -> np.ndarray:
    """PD by grade at a fractional horizon t from a generator: exp(tQ)[:, D]."""
    if t < 0.0:
        raise ValueError("t must be non-negative")
    return expm(t * np.asarray(q, dtype=float))[:-1, -1]
