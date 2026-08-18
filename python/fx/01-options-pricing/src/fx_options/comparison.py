"""Cross-model comparison harness: GK vs Black-76 vs binomial vs Monte Carlo.

Purpose: model governance in miniature.  Every production pricing model is
benchmarked against independent implementations; agreement within
documented tolerances is the acceptance evidence (see docs/VALIDATION.md).
"""

from __future__ import annotations

import pandas as pd

from .binomial import binomial_price
from .black76 import black76_from_spot
from .garman_kohlhagen import gk_price
from .monte_carlo import mc_price

__all__ = ["compare_models", "binomial_convergence_table",
           "mc_convergence_table"]


def compare_models(S: float, K: float, T: float, r_d: float, r_f: float,
                   sigma: float, option_type: str, *,
                   binomial_steps: int = 1000, mc_paths: int = 200_000,
                   mc_seed: int = 42) -> pd.DataFrame:
    """Price one option with all four models.

    Returns
    -------
    pandas.DataFrame
        Indexed by model name with columns ``price``, ``abs_diff_vs_gk``
        and ``std_error`` (NaN for non-MC models).
    """
    gk = gk_price(S, K, T, r_d, r_f, sigma, option_type)
    b76 = black76_from_spot(S, K, T, r_d, r_f, sigma, option_type)
    tree = binomial_price(S, K, T, r_d, r_f, sigma, option_type,
                          steps=binomial_steps)
    mc = mc_price(S, K, T, r_d, r_f, sigma, option_type,
                  n_paths=mc_paths, rng=mc_seed)
    rows = {
        "garman_kohlhagen": (gk, 0.0, float("nan")),
        "black76_on_forward": (b76, abs(b76 - gk), float("nan")),
        f"binomial_{binomial_steps}": (tree, abs(tree - gk), float("nan")),
        f"monte_carlo_{mc_paths}": (mc.price, abs(mc.price - gk),
                                    mc.std_error),
    }
    return pd.DataFrame.from_dict(
        rows, orient="index",
        columns=["price", "abs_diff_vs_gk", "std_error"])


def binomial_convergence_table(S: float, K: float, T: float, r_d: float,
                               r_f: float, sigma: float, option_type: str,
                               step_grid: tuple[int, ...] = (10, 25, 50, 100,
                                                             200, 400, 800),
                               ) -> pd.DataFrame:
    """Binomial-vs-GK convergence as a DataFrame (steps, price, error)."""
    from .binomial import binomial_convergence
    rows = binomial_convergence(S, K, T, r_d, r_f, sigma, option_type,
                                step_grid=step_grid)
    df = pd.DataFrame(rows)
    df["steps"] = df["steps"].astype(int)
    return df.set_index("steps")


def mc_convergence_table(S: float, K: float, T: float, r_d: float,
                         r_f: float, sigma: float, option_type: str,
                         path_grid: tuple[int, ...] = (1_000, 10_000,
                                                       100_000, 500_000),
                         seed: int = 42) -> pd.DataFrame:
    """MC estimate, standard error and error-vs-GK per path count."""
    gk = gk_price(S, K, T, r_d, r_f, sigma, option_type)
    rows = []
    for n in path_grid:
        res = mc_price(S, K, T, r_d, r_f, sigma, option_type,
                       n_paths=n, rng=seed)
        rows.append({"n_paths": n, "mc_price": res.price,
                     "std_error": res.std_error,
                     "abs_error_vs_gk": abs(res.price - gk),
                     "error_in_se": (abs(res.price - gk) / res.std_error
                                     if res.std_error > 0 else 0.0)})
    return pd.DataFrame(rows).set_index("n_paths")
