"""Equity VaR and ES with a choice of engine.

Each function wraps the reference function of the same name in ``eq_var``
and adds ``engine="python" | "cpp" | "rust"``. ``"python"`` (the default)
calls the reference unchanged; ``"cpp"`` and ``"rust"`` call the compiled
ports through ``qp_cpp`` / ``qp_rust``. Unknown names raise ``ValueError``;
a missing extension raises ``ImportError`` with an install hint (see
``qp_core.engines``).

Agreement with the reference (cross-language tolerances, checked over
10,000 random inputs per engine in ``tests/test_engine_differential.py``):

- historical VaR: exact (it is one order statistic of the sample);
- ES: 1e-13 relative (a tail sum in a different order);
- parametric VaR: 1e-12 relative (quadratic form times a quantile);
- Monte Carlo VaR on a linear book: statistical agreement only. Each engine
  has its own generator (NumPy PCG64, C++ mt19937_64, Rust xoshiro256++),
  so the same seed gives different paths; ``tests/test_engine_mc_var.py``
  checks each engine against the closed form within 4 standard errors of
  the sample quantile.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qp_core.engines import Engine, load_engine

from . import _tail
from ._validation import finite_array, validate_alpha
from .expected_shortfall import expected_shortfall as _expected_shortfall
from .historical_var import historical_var as _historical_var
from .monte_carlo_var import simulate_factor_returns
from .parametric_var import parametric_var as _parametric_var

__all__ = ["expected_shortfall", "historical_var", "monte_carlo_var", "parametric_var"]


def historical_var(pnl: np.ndarray, alpha: float = 0.01, *, engine: Engine = "python") -> float:
    """Historical-simulation VaR: the ceil(n alpha)-th largest loss."""
    mod = load_engine(engine)
    if mod is None:
        return _historical_var(pnl, alpha)
    return float(mod.eq_historical_var(np.ascontiguousarray(pnl, dtype=np.float64), alpha=alpha))


def expected_shortfall(pnl: np.ndarray, alpha: float = 0.01, *,
                       engine: Engine = "python") -> float:
    """Empirical ES (Acerbi and Tasche 2002, J. Banking and Finance 26(7),
    Definition 2.6)."""
    mod = load_engine(engine)
    if mod is None:
        return _expected_shortfall(pnl, alpha)
    return float(mod.eq_expected_shortfall(np.ascontiguousarray(pnl, dtype=np.float64),
                                           alpha=alpha))


def parametric_var(exposures: np.ndarray, cov: np.ndarray, alpha: float = 0.01,
                   dist: Literal["normal", "t"] = "normal", df: float = 6.0,
                   mean: float = 0.0, horizon_days: int = 1, *,
                   engine: Engine = "python") -> float:
    """Variance-covariance VaR (Jorion, Value at Risk, 3rd ed., ch. 7)."""
    mod = load_engine(engine)
    if mod is None:
        return _parametric_var(exposures, cov, alpha, dist, df, mean, horizon_days)
    return float(mod.eq_parametric_var(np.ascontiguousarray(exposures, dtype=np.float64),
                                       np.ascontiguousarray(cov, dtype=np.float64),
                                       alpha=alpha, dist=dist, df=df, mean=mean,
                                       horizon_days=horizon_days))


def monte_carlo_var(exposures: np.ndarray, cov: np.ndarray, alpha: float = 0.01,
                    n_paths: int = 100_000, dist: Literal["normal", "t"] = "normal",
                    df: float = 6.0, seed: int = 0, *, engine: Engine = "python") -> float:
    """Monte Carlo VaR of a linear book, P&L = exposures . r per path.

    The plain idea: draw ``n_paths`` factor-return vectors with covariance
    ``cov`` (normal, or Student-t scaled to the same covariance), take the
    book's P&L on each, and report the ``ceil(n alpha)``-th largest loss
    (spec S2). This is the linear-book model the C++ and Rust kernels
    implement; for options or full revaluation use
    :func:`eq_var.monte_carlo_var` with a ``Portfolio`` (Python only).

    Worked example: one factor with daily sigma 1 %, exposure 1,000,000 and
    alpha 1 %: the closed form is 2.326 * 10,000 = 23,263; 100,000 normal
    paths land within a few hundred of it on every engine.
    """
    mod = load_engine(engine)
    if mod is None:
        alpha = validate_alpha(alpha)
        w = finite_array("exposures", exposures, ndim=1)
        moves = simulate_factor_returns(cov, n_paths, dist=dist, df=df, seed=seed)
        if moves.shape[1] != w.size:
            raise ValueError(f"covariance is {moves.shape[1]}x{moves.shape[1]}, "
                             f"exposures has {w.size} entries")
        return _tail.var_es(moves @ w, alpha)[0]
    return float(mod.eq_monte_carlo_var(np.ascontiguousarray(exposures, dtype=np.float64),
                                        np.ascontiguousarray(cov, dtype=np.float64),
                                        alpha=alpha, n_paths=n_paths, dist=dist, df=df,
                                        seed=seed))
