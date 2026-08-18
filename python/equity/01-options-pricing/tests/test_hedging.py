"""Discrete delta-hedging: error scaling, mean P&L, model risk, costs."""

import math

import pytest

from eq_options import pnl_std_vs_frequency, simulate_delta_hedge

BASE = dict(S0=100.0, K=100.0, T=0.25, r=0.02, sigma_realized=0.20)


def test_pnl_std_decreases_with_rebalance_frequency() -> None:
    stds = pnl_std_vs_frequency([4, 16, 64, 256], n_paths=3000, **BASE)
    values = [stds[n] for n in (4, 16, 64, 256)]
    assert all(b < a for a, b in zip(values, values[1:]))


def test_pnl_std_scales_like_one_over_sqrt_n() -> None:
    """Quadrupling N should roughly halve the std (allow 30% slack)."""
    stds = pnl_std_vs_frequency([16, 256], n_paths=5000, **BASE)
    ratio = stds[16] / stds[256]  # theoretical sqrt(256/16) = 4
    assert 2.0 < ratio < 8.0


def test_mean_pnl_near_zero_when_hedged_at_true_vol() -> None:
    res = simulate_delta_hedge(**BASE, n_rebalance=128, n_paths=8000, seed=17)
    assert abs(res.mean) <= 4.0 * res.mean_se
    assert abs(res.mean) < 0.05  # absolute sanity: ~5 cents on a ~$4 option


def test_mean_pnl_near_zero_with_real_world_drift() -> None:
    """Delta hedging removes the drift: mean ~ 0 even with mu != r."""
    res = simulate_delta_hedge(**BASE, mu=0.15, n_rebalance=128, n_paths=8000, seed=18)
    assert abs(res.mean) <= 5.0 * res.mean_se


def test_positive_expected_pnl_when_implied_above_realized() -> None:
    """Short at 25 vol, realized 15 vol => collect the gamma-weighted spread."""
    res = simulate_delta_hedge(
        S0=100, K=100, T=0.25, r=0.02, sigma_realized=0.15, sigma_hedge=0.25,
        n_rebalance=64, n_paths=4000, seed=19,
    )
    assert res.mean > 0.0
    assert res.mean > 5.0 * res.mean_se  # decisively positive


def test_negative_expected_pnl_when_implied_below_realized() -> None:
    res = simulate_delta_hedge(
        S0=100, K=100, T=0.25, r=0.02, sigma_realized=0.30, sigma_hedge=0.20,
        n_rebalance=64, n_paths=4000, seed=20,
    )
    assert res.mean < 0.0


def test_realized_pnl_matches_gamma_vol_spread_formula() -> None:
    """mean P&L ~ E[ sum (sigma_h^2 - sigma_r^2)/2 * S^2 * Gamma_h dt ]."""
    res = simulate_delta_hedge(
        S0=100, K=100, T=0.25, r=0.02, sigma_realized=0.15, sigma_hedge=0.25,
        n_rebalance=128, n_paths=8000, seed=21,
    )
    assert res.theory_pnl > 0.0
    # simulated mean within 4 SE of the same-path theory estimate
    assert abs(res.mean - res.theory_pnl) <= 4.0 * res.mean_se


def test_transaction_costs_reduce_mean_pnl() -> None:
    kwargs = dict(**BASE, n_rebalance=64, n_paths=3000, seed=23)
    free = simulate_delta_hedge(**kwargs)
    costly = simulate_delta_hedge(tc_rate=5e-4, **kwargs)
    assert costly.mean < free.mean
    assert costly.premium == free.premium  # same option sold


def test_transaction_cost_drag_grows_with_frequency() -> None:
    lo = simulate_delta_hedge(**BASE, n_rebalance=16, n_paths=3000,
                              tc_rate=5e-4, seed=29)
    hi = simulate_delta_hedge(**BASE, n_rebalance=256, n_paths=3000,
                              tc_rate=5e-4, seed=29)
    assert hi.mean < lo.mean


def test_put_hedging_also_flat_at_true_vol() -> None:
    res = simulate_delta_hedge(**BASE, option_type="put",
                               n_rebalance=128, n_paths=6000, seed=31)
    assert abs(res.mean) <= 5.0 * res.mean_se


def test_hedge_result_shapes_and_reproducibility() -> None:
    a = simulate_delta_hedge(**BASE, n_rebalance=16, n_paths=500, seed=1)
    b = simulate_delta_hedge(**BASE, n_rebalance=16, n_paths=500, seed=1)
    assert a.pnl.shape == (500,)
    assert a.mean == b.mean and a.std == b.std
    assert a.premium > 0.0
    assert a.mean_se == pytest.approx(a.std / math.sqrt(500), rel=1e-12)


def test_hedging_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        simulate_delta_hedge(S0=-1, K=100, T=1, r=0.02, sigma_realized=0.2)
    with pytest.raises(ValueError):
        simulate_delta_hedge(S0=100, K=100, T=0.0, r=0.02, sigma_realized=0.2)
    with pytest.raises(ValueError):
        simulate_delta_hedge(**BASE, n_rebalance=0)
    with pytest.raises(ValueError):
        simulate_delta_hedge(**BASE, tc_rate=-0.01)
