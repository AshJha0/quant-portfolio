"""One quantile and ES estimator for every VaR path (spec S2).

Losses ``L = -pnl`` sorted descending ``L(1) >= ... >= L(n)``, ``k = n * alpha``,
``m = ceil(k - 1e-9)`` clamped to ``[1, n]``.

* ``VaR = L(m)``, an order statistic (no interpolation).
* ``ES = [sum_{i<m} L(i) + (k - (m - 1)) L(m)] / k``, the exact integral of
  the empirical quantile function over the worst ``alpha`` of mass
  (Acerbi-Tasche, fractional atom on ``L(m)``).

The weighted version (BRW) normalises the weights, sorts them with the
losses (stable), takes ``m`` = first ``j`` with ``W_j >= alpha - 1e-12`` and
``ES = [sum_{i<m} w_i L(i) + (alpha - W_{m-1}) L(m)] / alpha``.
"""

from __future__ import annotations

import math

import numpy as np


def tail_rank(n: int, alpha: float) -> tuple[float, int]:
    """Return ``(k, m)`` for a sample of size ``n``."""
    k = n * alpha
    m = min(max(math.ceil(k - 1e-9), 1), n)
    return k, m


def var_es(pnl: np.ndarray, alpha: float) -> tuple[float, float]:
    """``(VaR, ES)`` of a finite, non-empty 1-D P&L sample. No validation here."""
    losses = -np.asarray(pnl, dtype=float)
    n = losses.size
    k, m = tail_rank(n, alpha)
    # The m largest losses, then ordered descending: O(n + m log m).
    top = np.sort(np.partition(losses, n - m)[n - m :])[::-1]
    var = float(top[m - 1])
    es = (float(top[: m - 1].sum()) + (k - (m - 1)) * var) / k
    return var, float(es)


def var_rows(pnl_rows: np.ndarray, alpha: float) -> np.ndarray:
    """Row-wise S2 VaR of a 2-D array (used by the bootstrap)."""
    losses = -np.asarray(pnl_rows, dtype=float)
    n = losses.shape[1]
    _, m = tail_rank(n, alpha)
    return np.partition(losses, n - m, axis=1)[:, n - m]


def weighted_var_es(pnl: np.ndarray, weights: np.ndarray, alpha: float) -> tuple[float, float]:
    """Weighted ``(VaR, ES)``. Weights must be non-negative with a positive sum."""
    losses = -np.asarray(pnl, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    order = np.argsort(-losses, kind="stable")
    ls, ws = losses[order], w[order]
    cum = np.cumsum(ws)
    m = int(np.searchsorted(cum, alpha - 1e-12, side="left"))
    m = min(m, ls.size - 1)  # 0-based index of L(m)
    w_before = float(cum[m - 1]) if m > 0 else 0.0
    var = float(ls[m])
    es = (float(ws[:m] @ ls[:m]) + (alpha - w_before) * var) / alpha
    return var, float(es)
