"""Extreme value theory: peaks over threshold (POT) with a generalised
Pareto tail, and the VaR and ES it implies.

Plain idea
----------
Historical VaR at 99.9 % rests on one or two data points. EVT says that
losses above a high threshold ``u`` behave like a generalised Pareto
distribution (GPD) whatever the parent distribution, so we fit two numbers
(shape ``xi`` and scale ``beta``) to the exceedances and read deep quantiles
off the fitted tail.

Formal statement
----------------
Pickands-Balkema-de Haan: for a large class of loss distributions,
``P(L - u <= y | L > u) -> G_{xi,beta}(y) = 1 - (1 + xi y / beta)^{-1/xi}``.
With ``n`` observations, ``n_u`` above ``u`` and tail probability ``alpha``
(McNeil and Frey 2000, eqs. 9-10; McNeil, Frey and Embrechts, *Quantitative
Risk Management*, 2nd ed. 2015, eqs. 5.18-5.19):

    VaR = u + (beta / xi) ( (n alpha / n_u)^{-xi} - 1 )
    ES  = (VaR + beta - xi u) / (1 - xi),            xi < 1

and at ``xi = 0`` (exponential tail): ``VaR = u + beta ln(n_u / (n alpha))``,
``ES = VaR + beta``. Parameters are fitted by maximum likelihood on the
exceedances (``scipy.stats.genpareto.fit`` with location fixed at 0).

Worked example: ``tests/test_evt.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto

from . import _r3checks as chk

_XI_ZERO = 1e-9


@dataclass(frozen=True)
class GPDTail:
    threshold: float
    xi: float
    beta: float
    n: int  # total observations
    n_u: int  # exceedances

    def var(self, alpha: float = 0.01) -> float:
        return pot_var(alpha, self.threshold, self.xi, self.beta, self.n, self.n_u)

    def es(self, alpha: float = 0.01) -> float:
        return pot_es(alpha, self.threshold, self.xi, self.beta, self.n, self.n_u)


def _tail_args(alpha, u, xi, beta, n, n_u):
    a = chk.alpha_(alpha)
    u = chk.scalar("threshold", u)
    xi = chk.scalar("xi", xi)
    beta = chk.scalar("beta", beta, lo=0.0, strict=True)
    n = chk.count("n", n)
    n_u = chk.count("n_u", n_u)
    if n_u > n:
        raise ValueError("n_u cannot exceed n")
    if a >= n_u / n:
        raise ValueError(f"alpha={a} is not in the fitted tail (n_u/n = {n_u / n:.4g})")
    return a, u, xi, beta, n, n_u


def pot_var(alpha: float, threshold: float, xi: float, beta: float, n: int, n_u: int) -> float:
    """POT VaR (McNeil-Frey eq. 9); ``alpha`` must be below ``n_u / n``."""
    a, u, xi, beta, n, n_u = _tail_args(alpha, threshold, xi, beta, n, n_u)
    ratio = n * a / n_u
    if abs(xi) < _XI_ZERO:
        return u - beta * np.log(ratio)
    return u + beta / xi * (ratio ** (-xi) - 1.0)


def pot_es(alpha: float, threshold: float, xi: float, beta: float, n: int, n_u: int) -> float:
    """POT ES (McNeil-Frey eq. 10); finite only for ``xi < 1``."""
    a, u, xi, beta, n, n_u = _tail_args(alpha, threshold, xi, beta, n, n_u)
    if xi >= 1.0:
        raise ValueError(f"ES is infinite for xi >= 1, got xi={xi}")
    v = pot_var(a, u, xi, beta, n, n_u)
    if abs(xi) < _XI_ZERO:
        return v + beta
    return (v + beta - xi * u) / (1.0 - xi)


def fit_pot(pnl, threshold_quantile: float = 0.95, *, threshold: float | None = None) -> GPDTail:
    """Fit a GPD to losses (``-pnl``) above a threshold by maximum likelihood.

    The threshold defaults to the ``threshold_quantile`` empirical quantile of
    losses. At least 30 exceedances are required; fewer gives an unstable fit.
    """
    losses = -chk.array("pnl", pnl, min_size=50)
    if threshold is None:
        q = chk.scalar("threshold_quantile", threshold_quantile)
        if not 0.5 <= q < 1.0:
            raise ValueError("threshold_quantile must be in [0.5, 1)")
        u = float(np.quantile(losses, q))
    else:
        u = chk.scalar("threshold", threshold)
    excess = losses[losses > u] - u
    if excess.size < 30:
        raise ValueError(f"need at least 30 exceedances, got {excess.size}")
    xi, _, beta = genpareto.fit(excess, floc=0.0)
    return GPDTail(u, float(xi), float(beta), losses.size, int(excess.size))


def mean_excess(pnl, thresholds) -> np.ndarray:
    """Empirical mean excess ``e(u) = E[L - u | L > u]`` at each threshold.

    For a GPD tail ``e(u)`` is linear in ``u`` with slope ``xi / (1 - xi)``,
    which is the usual visual check for choosing the threshold.
    """
    losses = -chk.array("pnl", pnl)
    t = chk.array("thresholds", thresholds)
    out = np.full(t.size, np.nan)
    for i, u in enumerate(t):
        ex = losses[losses > u]
        if ex.size:
            out[i] = ex.mean() - u
    return out
