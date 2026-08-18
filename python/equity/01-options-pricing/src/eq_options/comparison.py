"""Model-comparison harness: price one contract under every engine.

Prices the same European contract under Black-Scholes (closed form), the
CRR binomial tree, Black-76 on the model-consistent forward
``F = S exp((r - q) T)``, and exact-scheme Monte Carlo, then tabulates
prices, differences vs the analytic benchmark, and wall-clock runtimes.
Also produces convergence tables (tree steps -> BS, MC paths -> BS) used
verbatim in docs/VALIDATION.md.

Conventions: continuously compounded annualised ``r``, ``q`` (ACT/365F),
``T`` in years, ``sigma`` annualised.
"""

from __future__ import annotations

import math
import time

import pandas as pd

from .binomial import crr_price
from .black76 import black76_price
from .black_scholes import OptionType, bs_price, forward_price
from .monte_carlo import mc_price

__all__ = ["compare_models", "tree_convergence_table", "mc_convergence_table"]


def compare_models(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_steps: int = 1_000,
    n_paths: int = 200_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Price one European contract under BS, CRR tree, Black-76 and MC.

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type
        Contract and market inputs (see :func:`eq_options.black_scholes.bs_price`).
    n_steps : int
        Tree steps for the CRR engine.
    n_paths : int
        Monte Carlo paths (antithetic + control variate on).
    seed : int
        MC seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Indexed by model name with columns ``price``, ``abs_diff_vs_bs``,
        ``runtime_ms`` and ``std_error`` (NaN-free: 0.0 for exact models).
    """
    rows: list[dict[str, object]] = []

    t0 = time.perf_counter()
    p_bs = bs_price(S, K, T, r, sigma, q, option_type)
    t_bs = (time.perf_counter() - t0) * 1e3
    rows.append({"model": "black_scholes", "price": p_bs, "std_error": 0.0,
                 "runtime_ms": t_bs})

    t0 = time.perf_counter()
    p_tree = crr_price(S, K, T, r, sigma, q, option_type, "european", n_steps)
    t_tree = (time.perf_counter() - t0) * 1e3
    rows.append({"model": f"crr_tree_{n_steps}", "price": p_tree, "std_error": 0.0,
                 "runtime_ms": t_tree})

    t0 = time.perf_counter()
    F = forward_price(S, T, r, q)
    p_b76 = black76_price(F, K, T, r, sigma, option_type)
    t_b76 = (time.perf_counter() - t0) * 1e3
    rows.append({"model": "black76_on_forward", "price": p_b76, "std_error": 0.0,
                 "runtime_ms": t_b76})

    t0 = time.perf_counter()
    mc = mc_price(S, K, T, r, sigma, q, option_type, n_paths=n_paths, seed=seed)
    t_mc = (time.perf_counter() - t0) * 1e3
    rows.append({"model": f"monte_carlo_{n_paths}", "price": mc.value,
                 "std_error": mc.std_error, "runtime_ms": t_mc})

    df = pd.DataFrame(rows).set_index("model")
    df["abs_diff_vs_bs"] = (df["price"] - p_bs).abs()
    return df[["price", "abs_diff_vs_bs", "std_error", "runtime_ms"]]


def tree_convergence_table(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    steps: tuple[int, ...] = (10, 25, 50, 100, 250, 500, 1000, 2000),
) -> pd.DataFrame:
    """CRR European price vs Black-Scholes as tree steps grow.

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type
        Contract and market inputs.
    steps : tuple of int
        Tree step counts to evaluate.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``n_steps`` with columns ``tree_price``, ``bs_price``,
        ``abs_error`` and ``error_x_n`` (abs_error * n; roughly constant
        confirms O(1/n) convergence).
    """
    p_bs = bs_price(S, K, T, r, sigma, q, option_type)
    rows = []
    for n in steps:
        p = crr_price(S, K, T, r, sigma, q, option_type, "european", n)
        err = abs(p - p_bs)
        rows.append({"n_steps": n, "tree_price": p, "bs_price": p_bs,
                     "abs_error": err, "error_x_n": err * n})
    return pd.DataFrame(rows).set_index("n_steps")


def mc_convergence_table(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    paths: tuple[int, ...] = (1_000, 4_000, 16_000, 64_000, 256_000),
    seed: int = 42,
) -> pd.DataFrame:
    """MC price vs Black-Scholes as the path count grows.

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type
        Contract and market inputs.
    paths : tuple of int
        Path counts to evaluate (independent seeded streams).
    seed : int
        Base seed; run ``i`` uses ``seed + i``.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``n_paths`` with columns ``mc_price``, ``bs_price``,
        ``abs_error``, ``std_error`` and ``se_x_sqrt_n`` (roughly constant
        confirms O(1/sqrt(n)) convergence).
    """
    p_bs = bs_price(S, K, T, r, sigma, q, option_type)
    rows = []
    for i, n in enumerate(paths):
        res = mc_price(S, K, T, r, sigma, q, option_type, n_paths=n, seed=seed + i)
        rows.append({
            "n_paths": n,
            "mc_price": res.value,
            "bs_price": p_bs,
            "abs_error": abs(res.value - p_bs),
            "std_error": res.std_error,
            "se_x_sqrt_n": res.std_error * math.sqrt(n),
        })
    return pd.DataFrame(rows).set_index("n_paths")
