"""Monte Carlo VaR and ES with a Student-t copula.

Plain idea
----------
A Gaussian Monte Carlo makes joint crashes too rare: under a Gaussian
dependence structure two assets almost never fall far together, however
correlated they are. A t copula keeps each asset's own marginal
distribution but draws the dependence from a multivariate t, whose shared
random scale makes extremes arrive together.

Formal statement
----------------
Draw ``Z ~ N(0, R)`` and an independent ``W ~ chi2_nu / nu``. Then
``T = Z / sqrt(W)`` is multivariate t and ``U_i = t_nu(T_i)`` has a t copula
with correlation ``R`` and ``nu`` degrees of freedom. Factor returns are
``X_i = F_i^{-1}(U_i)`` for any marginals ``F_i``. Two oracles hold exactly
(Demarta and McNeil 2005, eqs. 3 and 7; Embrechts, Lindskog and McNeil 2003):

* Kendall's tau: ``tau = (2 / pi) arcsin(rho)`` for every ``nu``.
* Tail dependence: ``lambda = 2 t_{nu+1}( -sqrt((nu + 1)(1 - rho) / (1 + rho)) )``.

References
----------
* S. Demarta and A. McNeil, "The t copula and related copulas",
  Int. Stat. Review 73(1) (2005) 111-129.
* P. Embrechts, F. Lindskog, A. McNeil, "Modelling dependence with copulas
  and applications to risk management", in *Handbook of Heavy Tailed
  Distributions in Finance* (2003), ch. 8.

Worked example: ``tests/test_copula_mc.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import t as student_t

from . import _r3checks as chk

Marginal = Callable[[np.ndarray], np.ndarray]


def t_copula_uniforms(corr, nu: float, n: int, seed: int | None = 0) -> np.ndarray:
    """``(n, d)`` uniforms from a t copula with correlation ``corr``."""
    r = chk.correlation("corr", corr)
    df = chk.scalar("nu", nu, lo=0.0, strict=True)
    n = chk.count("n", n)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, r.shape[0])) @ np.linalg.cholesky(r).T
    w = rng.chisquare(df, size=(n, 1)) / df
    return student_t.cdf(z / np.sqrt(w), df)


def t_copula_pnl(exposures, corr, nu: float, marginals: Sequence[Marginal], n: int = 100_000,
                 seed: int | None = 0) -> np.ndarray:
    """Book P&L ``sum_i exposures_i X_i`` with t-copula dependence.

    ``marginals[i]`` maps uniforms to factor returns (an inverse CDF, for
    example ``scipy.stats.norm(scale=0.01).ppf``).
    """
    w = chk.array("exposures", exposures)
    if len(marginals) != w.size:
        raise ValueError("need one marginal per exposure")
    u = t_copula_uniforms(corr, nu, n, seed)
    if u.shape[1] != w.size:
        raise ValueError("corr dimension must match exposures")
    x = np.column_stack([np.asarray(f(u[:, i]), dtype=float) for i, f in enumerate(marginals)])
    if not np.isfinite(x).all():
        raise ValueError("a marginal returned NaN or infinite values")
    return x @ w


def t_copula_var_es(exposures, corr, nu: float, marginals: Sequence[Marginal],
                    alpha: float = 0.01, n: int = 100_000, seed: int | None = 0) -> tuple[float, float]:
    """Empirical (VaR, ES) of :func:`t_copula_pnl` at tail ``alpha``."""
    a = chk.alpha_(alpha)
    return chk.loss_quantile(t_copula_pnl(exposures, corr, nu, marginals, n, seed), a)


def kendall_tau_to_rho(tau: float) -> float:
    """Invert ``tau = (2/pi) arcsin(rho)``: the standard copula calibration."""
    t = chk.scalar("tau", tau)
    if not -1.0 < t < 1.0:
        raise ValueError("tau must be in (-1, 1)")
    return float(np.sin(np.pi * t / 2.0))


def t_tail_dependence(rho: float, nu: float) -> float:
    """Closed-form upper (= lower) tail dependence of the t copula."""
    r = chk.scalar("rho", rho)
    df = chk.scalar("nu", nu, lo=0.0, strict=True)
    if not -1.0 < r < 1.0:
        raise ValueError("rho must be in (-1, 1)")
    return float(2.0 * student_t.cdf(-np.sqrt((df + 1.0) * (1.0 - r) / (1.0 + r)), df + 1.0))
