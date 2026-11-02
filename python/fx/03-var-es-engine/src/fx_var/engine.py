"""FX VaR and ES with a choice of engine.

Each function wraps the reference function of the same name in ``fx_var``
and adds ``engine="python" | "cpp" | "rust"``. ``"python"`` (the default)
calls the reference unchanged; ``"cpp"`` and ``"rust"`` call the compiled
ports through ``qp_cpp`` / ``qp_rust``. Unknown names raise ``ValueError``;
a missing extension raises ``ImportError`` with an install hint (see
``qp_core.engines``).

Agreement with the reference (cross-language tolerances, checked over
10,000 random inputs per engine in ``tests/test_engine_differential.py``):

- empirical VaR exact, ES 1e-13 relative;
- normal and Student-t closed forms 1e-12 relative (1e-9 absolute);
- variance-covariance VaR and ES 1e-12 relative.

``empirical_var_es`` here is the equal-weight case only; weighted samples
stay on the reference function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qp_core.engines import Engine, load_engine

from .expected_shortfall import empirical_var_es as _empirical_var_es
from .expected_shortfall import normal_es as _normal_es
from .expected_shortfall import normal_var as _normal_var
from .expected_shortfall import t_es as _t_es
from .expected_shortfall import t_var as _t_var
from .parametric_var import var_covar as _var_covar

__all__ = ["empirical_var_es", "normal_es", "normal_var", "t_es", "t_var", "var_covar"]


def empirical_var_es(pnl, alpha: float = 0.01, *,
                     engine: Engine = "python") -> tuple[float, float]:
    """``(VaR, ES)`` of an equal-weight P&L sample."""
    mod = load_engine(engine)
    if mod is None:
        return _empirical_var_es(pnl, alpha)
    var, es = mod.fx_empirical_var_es(np.ascontiguousarray(pnl, dtype=np.float64), alpha=alpha)
    return float(var), float(es)


def normal_var(sigma: float, alpha: float = 0.01, mean: float = 0.0, *,
               engine: Engine = "python") -> float:
    """Normal VaR: ``-mean + sigma Phi^{-1}(1 - alpha)``."""
    mod = load_engine(engine)
    if mod is None:
        return _normal_var(sigma, alpha, mean)
    return float(mod.fx_normal_var(sigma, alpha=alpha, mean=mean))


def normal_es(sigma: float, alpha: float = 0.01, mean: float = 0.0, *,
              engine: Engine = "python") -> float:
    """Normal ES: ``-mean + sigma phi(z) / alpha``."""
    mod = load_engine(engine)
    if mod is None:
        return _normal_es(sigma, alpha, mean)
    return float(mod.fx_normal_es(sigma, alpha=alpha, mean=mean))


def t_var(sigma: float, alpha: float = 0.01, df: float = 6.0, mean: float = 0.0, *,
          engine: Engine = "python") -> float:
    """Unit-variance Student-t VaR with P&L standard deviation ``sigma``."""
    mod = load_engine(engine)
    if mod is None:
        return _t_var(sigma, alpha, df, mean)
    return float(mod.fx_t_var(sigma, alpha=alpha, df=df, mean=mean))


def t_es(sigma: float, alpha: float = 0.01, df: float = 6.0, mean: float = 0.0, *,
         engine: Engine = "python") -> float:
    """Unit-variance Student-t ES (McNeil, Frey and Embrechts, Quantitative
    Risk Management, 2nd ed., Example 2.15)."""
    mod = load_engine(engine)
    if mod is None:
        return _t_es(sigma, alpha, df, mean)
    return float(mod.fx_t_es(sigma, alpha=alpha, df=df, mean=mean))


def var_covar(exposures, cov, alpha: float = 0.01, horizon_days: float = 1.0,
              dist: str = "normal", df: float = 6.0, mean: float = 0.0, *,
              engine: Engine = "python") -> tuple[float, float]:
    """Variance-covariance ``(VaR, ES)``. Accepts arrays or a labelled
    ``pd.Series`` / ``pd.DataFrame`` (labels must then align)."""
    mod = load_engine(engine)
    if mod is None:
        if not isinstance(exposures, pd.Series):
            exposures = pd.Series(np.asarray(exposures, dtype=np.float64))
        if not isinstance(cov, pd.DataFrame):
            cov = pd.DataFrame(np.asarray(cov, dtype=np.float64), index=exposures.index,
                               columns=exposures.index)
        return _var_covar(exposures, cov, alpha, horizon_days, dist, df, mean)
    if isinstance(exposures, pd.Series) and isinstance(cov, pd.DataFrame):
        cov = cov.loc[exposures.index, exposures.index]
    var, es = mod.fx_var_covar(np.ascontiguousarray(exposures, dtype=np.float64),
                               np.ascontiguousarray(cov, dtype=np.float64), alpha=alpha,
                               horizon_days=horizon_days, dist=dist, df=df, mean=mean)
    return float(var), float(es)
