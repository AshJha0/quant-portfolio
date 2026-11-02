"""Equity option pricing with a choice of engine.

Each function wraps the reference function of the same name in
``eq_options`` and adds ``engine="python" | "cpp" | "rust"``. ``"python"``
(the default) calls the reference unchanged; ``"cpp"`` and ``"rust"`` call
the compiled ports through ``qp_cpp`` / ``qp_rust``. Unknown names raise
``ValueError``; a missing extension raises ``ImportError`` with an install
hint (see ``qp_core.engines``).

Agreement with the reference (tolerances from the cross-language tests,
checked over 10,000 random contracts per engine in
``tests/test_engine_differential.py``):

- prices and Greeks: 64 eps (1 + d1^2) (|ref| + max(S, K)), Greeks times
  max(1, 1/T);
- implied vol: 4 max(eps max(S, K) / vega, 1e-12 sigma);
- CRR tree: 4 n eps max(S, K).

Greeks come back as ``BSGreeks`` from every engine.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qp_core.engines import Engine, load_engine

from .binomial import crr_price as _crr_price
from .black_scholes import OptionType
from .black_scholes import bs_price as _bs_price
from .black_scholes import implied_vol as _implied_vol
from .greeks import BSGreeks
from .greeks import bs_greeks as _bs_greeks

__all__ = ["bs_greeks", "bs_price", "bs_price_batch", "crr_price", "implied_vol"]


def bs_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
             option_type: OptionType = "call", *, engine: Engine = "python") -> float:
    """Black-Scholes-Merton price (Hull, Options, Futures and Other
    Derivatives, 11th ed., eq. 17.4)."""
    mod = load_engine(engine)
    if mod is None:
        return _bs_price(S, K, T, r, sigma, q, option_type)
    return float(mod.eq_bs_price(S, K, T, r, sigma, q, option_type))


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
              option_type: OptionType = "call", *, engine: Engine = "python") -> BSGreeks:
    """Closed-form Black-Scholes-Merton Greeks as ``BSGreeks``."""
    mod = load_engine(engine)
    if mod is None:
        return _bs_greeks(S, K, T, r, sigma, q, option_type)
    return BSGreeks(**mod.eq_bs_greeks(S, K, T, r, sigma, q, option_type))


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float = 0.0,
                option_type: OptionType = "call", *, engine: Engine = "python") -> float:
    """Black-Scholes implied volatility. Every engine raises ``ValueError``
    for a quote outside the no-arbitrage bounds."""
    mod = load_engine(engine)
    if mod is None:
        return _implied_vol(price, S, K, T, r, q, option_type)
    return float(mod.eq_implied_vol(price, S, K, T, r, q, option_type))


def crr_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
              option_type: OptionType = "call", style: Literal["american", "european"] = "american", n_steps: int = 500,
              *, engine: Engine = "python") -> float:
    """Cox-Ross-Rubinstein tree price (Cox, Ross and Rubinstein 1979)."""
    mod = load_engine(engine)
    if mod is None:
        return _crr_price(S, K, T, r, sigma, q, option_type, style, n_steps)
    if style not in ("american", "european"):
        raise ValueError(f"style must be 'american' or 'european', got {style!r}")
    return float(mod.eq_crr_price(S, K, T, r, sigma, q, option_type,
                                  american=style == "american", n_steps=n_steps))


def bs_price_batch(S, K, T, r, sigma, q, is_call, *, engine: Engine = "python") -> np.ndarray:
    """Vectorised price over equal-length arrays; ``is_call`` is boolean.
    The Python engine loops over the reference scalar function."""
    mod = load_engine(engine)
    cols = [np.ascontiguousarray(a, dtype=np.float64) for a in (S, K, T, r, sigma, q)]
    flags = np.ascontiguousarray(is_call, dtype=np.bool_)
    if mod is not None:
        return mod.eq_bs_price_batch(*cols, flags)
    if any(c.shape != flags.shape for c in cols) or flags.ndim != 1:
        raise ValueError("length mismatch: inputs must be equal-length 1-D arrays")
    s, k, t, rr, v, qq = cols
    return np.array([_bs_price(float(s[i]), float(k[i]), float(t[i]), float(rr[i]), float(v[i]),
                               float(qq[i]), "call" if flags[i] else "put")
                     for i in range(flags.size)], dtype=np.float64)
