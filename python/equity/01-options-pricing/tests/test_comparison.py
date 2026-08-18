"""Model-comparison harness and convergence tables."""

import numpy as np
import pytest

from eq_options import (
    bs_price,
    compare_models,
    mc_convergence_table,
    tree_convergence_table,
)


def test_compare_models_agree_within_tolerance() -> None:
    df = compare_models(100, 100, 1.0, 0.05, 0.2, 0.01, "call",
                        n_steps=1000, n_paths=100_000, seed=42)
    assert len(df) == 4
    bs = df.loc["black_scholes", "price"]
    assert df.loc["black76_on_forward", "abs_diff_vs_bs"] < 1e-10
    assert df.loc["crr_tree_1000", "abs_diff_vs_bs"] < 5e-3
    mc_row = df.loc["monte_carlo_100000"]
    assert abs(mc_row["price"] - bs) <= 3.0 * mc_row["std_error"]


def test_compare_models_no_nans_and_runtimes_positive() -> None:
    df = compare_models(100, 110, 0.5, 0.03, 0.3, 0.0, "put",
                        n_steps=500, n_paths=50_000, seed=1)
    assert not df.isna().any().any()
    assert (df["runtime_ms"] > 0).all()
    assert (df["price"] > 0).all()


def test_tree_convergence_table_errors_shrink() -> None:
    tbl = tree_convergence_table(100, 100, 1.0, 0.05, 0.2, 0.0, "call",
                                 steps=(10, 50, 250, 1000))
    errs = tbl["abs_error"].to_numpy()
    assert errs[-1] < errs[0] / 10.0  # O(1/n) over two decades
    assert (tbl["bs_price"] == bs_price(100, 100, 1.0, 0.05, 0.2, 0.0, "call")).all()


def test_tree_convergence_error_x_n_bounded() -> None:
    tbl = tree_convergence_table(100, 105, 0.5, 0.03, 0.25, 0.01, "put",
                                 steps=(100, 200, 400, 800, 1600))
    exn = tbl["error_x_n"].to_numpy()
    assert exn.max() < 50 * max(exn.min(), 1e-6)  # same order of magnitude


def test_mc_convergence_table_se_shrinks_like_sqrt_n() -> None:
    tbl = mc_convergence_table(100, 100, 1.0, 0.05, 0.2, 0.0, "call",
                               paths=(2_000, 32_000, 512_000), seed=5)
    se = tbl["std_error"].to_numpy()
    assert se[-1] < se[0] / 4.0  # 256x paths => 16x smaller SE ideally
    scaled = tbl["se_x_sqrt_n"].to_numpy()
    assert scaled.max() < 3.0 * scaled.min()  # roughly constant
    assert np.isfinite(tbl.to_numpy(dtype=float)).all()
