"""FX option pricing with a choice of engine.

Each function wraps the reference function of the same name in
``fx_options`` and adds ``engine="python" | "cpp" | "rust"``. ``"python"``
(the default) calls the reference unchanged; ``"cpp"`` and ``"rust"`` call
the compiled ports through ``qp_cpp`` / ``qp_rust``. Unknown names raise
``ValueError``; a missing extension raises ``ImportError`` with an install
hint (see ``qp_core.engines``).

Agreement with the reference is the same as for ``eq_options.engine`` with
(r, q) read as (r_d, r_f), and is checked over 10,000 random contracts per
engine in ``tests/test_engine_differential.py``. Greeks come back as
``GreeksResult`` from every engine.
"""

from __future__ import annotations

import numpy as np
from qp_core.engines import Engine, load_engine

from .binomial import binomial_price as _binomial_price
from .garman_kohlhagen import gk_price as _gk_price
from .garman_kohlhagen import implied_vol as _implied_vol
from .greeks import GreeksResult
from .greeks import analytic_greeks as _analytic_greeks

__all__ = ["analytic_greeks", "binomial_price", "gk_price", "gk_price_batch", "implied_vol"]


def gk_price(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float,
             option_type: str, *, engine: Engine = "python") -> float:
    """Garman-Kohlhagen price (Garman and Kohlhagen 1983, J. Int. Money and
    Finance 2(3), eq. 7-8)."""
    mod = load_engine(engine)
    if mod is None:
        return _gk_price(S, K, T, r_d, r_f, sigma, option_type)
    return float(mod.fx_gk_price(S, K, T, r_d, r_f, sigma, option_type))


def analytic_greeks(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float,
                    option_type: str, *, engine: Engine = "python") -> GreeksResult:
    """Closed-form Garman-Kohlhagen Greeks as ``GreeksResult``."""
    mod = load_engine(engine)
    if mod is None:
        return _analytic_greeks(S, K, T, r_d, r_f, sigma, option_type)
    return GreeksResult(**mod.fx_gk_greeks(S, K, T, r_d, r_f, sigma, option_type))


def implied_vol(price: float, S: float, K: float, T: float, r_d: float, r_f: float,
                option_type: str, *, engine: Engine = "python") -> float:
    """Garman-Kohlhagen implied volatility. Every engine raises
    ``ValueError`` for a quote outside the no-arbitrage bounds."""
    mod = load_engine(engine)
    if mod is None:
        return _implied_vol(price, S, K, T, r_d, r_f, option_type)
    return float(mod.fx_implied_vol(price, S, K, T, r_d, r_f, option_type))


def binomial_price(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float,
                   option_type: str, steps: int = 500, style: str = "american",
                   *, engine: Engine = "python") -> float:
    """CRR tree price with the foreign rate as the dividend yield."""
    mod = load_engine(engine)
    if mod is None:
        return _binomial_price(S, K, T, r_d, r_f, sigma, option_type, steps, style)
    if style not in ("american", "european"):
        raise ValueError(f"style must be 'american' or 'european', got {style!r}")
    return float(mod.fx_binomial_price(S, K, T, r_d, r_f, sigma, option_type, steps=steps,
                                       american=style == "american"))


def gk_price_batch(S, K, T, r_d, r_f, sigma, is_call, *, engine: Engine = "python") -> np.ndarray:
    """Vectorised price over equal-length arrays; ``is_call`` is boolean.
    The Python engine loops over the reference scalar function."""
    mod = load_engine(engine)
    cols = [np.ascontiguousarray(a, dtype=np.float64) for a in (S, K, T, r_d, r_f, sigma)]
    flags = np.ascontiguousarray(is_call, dtype=np.bool_)
    if mod is not None:
        return mod.fx_gk_price_batch(*cols, flags)
    if any(c.shape != flags.shape for c in cols) or flags.ndim != 1:
        raise ValueError("length mismatch: inputs must be equal-length 1-D arrays")
    return np.array([_gk_price(*(float(c[i]) for c in cols), "call" if flags[i] else "put")
                     for i in range(flags.size)], dtype=np.float64)
