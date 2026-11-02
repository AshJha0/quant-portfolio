"""Euler risk attribution: marginal, component and incremental VaR and ES.

Plain idea
----------
A desk wants to know which positions drive the book's risk. VaR and ES are
positively homogeneous of degree one in the position vector ``w``
(doubling every position doubles the risk), so Euler's theorem splits the
total exactly into per-position pieces:

    R(w) = sum_i w_i dR/dw_i.

``dR/dw_i`` is the *marginal* risk of position ``i`` and ``w_i dR/dw_i``
its *component* risk. *Incremental* risk is the full change
``R(w + dw) - R(w)`` from a trade ``dw``; for a small trade it is close to
``marginal . dw``.

Formal statement
----------------
Exposures ``w`` (currency P&L per unit factor return), factor returns
``r ~ N(mu, Sigma)``, book P&L ``w . r``, ``sigma_p = sqrt(w' Sigma w)``:

    VaR = z sigma_p - w . mu,           z = Phi^{-1}(1 - alpha)
    ES  = (phi(z)/alpha) sigma_p - w . mu
    dR/dw = k Sigma w / sigma_p - mu,   k = z (VaR) or phi(z)/alpha (ES)

For simulated or historical scenarios the ES gradient is the conditional
mean loss of each position on the tail scenarios (Tasche 2000, eq. 5.8;
Hallerbach 2003). The VaR gradient is the conditional mean on the VaR
scenario, which a single order statistic estimates badly, so we average a
symmetric window of ranks around it (the kernel estimator of Epperlein and
Smillie 2006 with a box kernel).

References
----------
* D. Tasche, "Risk contributions and performance measurement", TU Munich
  working paper (2000), sections 4-5.
* W. Hallerbach, "Decomposing portfolio Value-at-Risk: a general analysis",
  J. Risk 5(2) (2003).
* E. Epperlein and A. Smillie, "Cracking VaR with kernels", Risk (Aug 2006).
* P. Jorion, *Value at Risk*, 3rd ed. (2007), ch. 7.

Worked example: ``tests/test_risk_attribution.py::test_two_asset_hand_example``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

from . import _r3checks as chk

Measure = Literal["var", "es"]


@dataclass(frozen=True)
class Attribution:
    """Euler split of one risk number.

    ``total`` equals ``component.sum()`` to rounding for every method here.
    ``marginal`` is ``dR/dw``; ``component = w * marginal``;
    ``percent = component / total``.
    """

    total: float
    marginal: np.ndarray
    component: np.ndarray

    @property
    def percent(self) -> np.ndarray:
        return self.component / self.total


def _measure(measure: str) -> str:
    if measure not in ("var", "es"):
        raise ValueError(f"measure must be 'var' or 'es', got {measure!r}")
    return measure


def _cov_inputs(w, cov, mean):
    w = chk.array("w", w)
    cov = chk.array("cov", cov, ndim=2)
    if cov.shape != (w.size, w.size):
        raise ValueError(f"cov must be {w.size}x{w.size}, got {cov.shape}")
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("cov must be symmetric")
    mu = np.zeros(w.size) if mean is None else chk.array("mean", mean)
    if mu.shape != w.shape:
        raise ValueError("mean must have one entry per position")
    return w, cov, mu


def parametric_attribution(w, cov, alpha: float = 0.01, *, measure: Measure = "var",
                           mean=None) -> Attribution:
    """Closed-form Euler attribution under multivariate normal returns.

    See the module docstring for the formulas. Raises ``ValueError`` when the
    book has zero variance (the gradient is undefined there).
    """
    a = chk.alpha_(alpha)
    w, cov, mu = _cov_inputs(w, cov, mean)
    z = float(norm.ppf(1.0 - a))
    k = z if _measure(measure) == "var" else float(norm.pdf(z)) / a
    var_p = float(w @ cov @ w)
    if not var_p > 0.0:
        raise ValueError("book variance must be > 0 for an Euler attribution")
    sigma = np.sqrt(var_p)
    marginal = k * (cov @ w) / sigma - mu
    component = w * marginal
    return Attribution(total=float(k * sigma - w @ mu), marginal=marginal, component=component)


def parametric_incremental(w, dw, cov, alpha: float = 0.01, *, measure: Measure = "var",
                           mean=None) -> float:
    """Exact incremental risk ``R(w + dw) - R(w)`` under normal returns."""
    w, cov, mu = _cov_inputs(w, cov, mean)
    dw = chk.array("dw", dw)
    if dw.shape != w.shape:
        raise ValueError("dw must have one entry per position")
    after = parametric_attribution(w + dw, cov, alpha, measure=measure, mean=mu).total
    before = parametric_attribution(w, cov, alpha, measure=measure, mean=mu).total
    return after - before


def simulation_attribution(pnl_by_position, alpha: float = 0.01, *, measure: Measure = "es",
                           window: int | None = None) -> Attribution:
    """Euler attribution from scenario P&L, one column per position.

    ``pnl_by_position`` is ``(n_scenarios, n_positions)``; row sums are the
    book P&L. Unit exposures are implied, so ``marginal == component``.

    * ``measure="es"``: tail scenarios are the ``ceil(alpha n)`` worst book
      outcomes, with the same fractional weight on the boundary scenario as
      the package ES estimator, so ``total`` equals the empirical ES exactly.
    * ``measure="var"``: the mean of each position's loss over ``2 window + 1``
      ranks centred on the VaR order statistic. ``total`` is the window mean of
      book losses, which is the smoothed VaR. Default ``window`` is
      ``max(1, round(alpha n / 4))``: a window of half the tail count, which
      keeps the Monte Carlo noise of the contributions near 1 % of total on
      400,000 scenarios (tested).
    """
    a = chk.alpha_(alpha)
    x = chk.array("pnl_by_position", pnl_by_position, ndim=2, min_size=1)
    n = x.shape[0]
    if n < 10:
        raise ValueError(f"need at least 10 scenarios, got {n}")
    order = np.argsort(x.sum(axis=1), kind="stable")  # worst book P&L first
    k = a * n
    m = max(1, int(np.ceil(k - 1e-9)))
    if _measure(measure) == "es":
        weights = np.zeros(n)
        weights[order[: m - 1]] = 1.0
        weights[order[m - 1]] = k - (m - 1)
        comp = -(weights @ x) / k
    else:
        h = max(1, round(k / 4)) if window is None else chk.count("window", window)
        lo, hi = max(0, m - 1 - h), min(n, m + h)
        comp = -x[order[lo:hi]].mean(axis=0)
    return Attribution(total=float(comp.sum()), marginal=comp, component=comp.copy())


def simulation_incremental(pnl_by_position, dw, alpha: float = 0.01) -> tuple[float, float]:
    """Exact incremental (VaR, ES) of scaling position ``i`` by ``1 + dw_i``.

    Recomputes the empirical VaR and ES of the new book on the same
    scenarios; no linearisation.
    """
    a = chk.alpha_(alpha)
    x = chk.array("pnl_by_position", pnl_by_position, ndim=2)
    dw = chk.array("dw", dw)
    if dw.size != x.shape[1]:
        raise ValueError("dw must have one entry per position")
    v0, e0 = chk.loss_quantile(x.sum(axis=1), a)
    v1, e1 = chk.loss_quantile(x @ (1.0 + dw), a)
    return v1 - v0, e1 - e0
