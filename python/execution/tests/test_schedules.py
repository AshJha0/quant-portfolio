"""Schedules: volume curves, adaptive VWAP/POV, transient impact, placement."""

from __future__ import annotations

import numpy as np
import pytest
from qp_exec.schedules.placement import decide, fill_probability
from qp_exec.schedules.transient import (
    expected_cost,
    exponential_kernel,
    gatheral_admissible,
    gaussian_kernel,
    min_round_trip_cost,
    optimal_schedule,
    power_law_kernel,
)
from qp_exec.schedules.volume import calibrate_curve, execute_adaptive, twap_curve


def u_shape_days(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([4, 2, 1.5, 1, 1, 1, 1.5, 2, 3, 5], float)
    return 1e4 * base * rng.lognormal(0, 0.3, (n, base.size)) * rng.lognormal(0, 0.4, (n, 1))


def test_curve_calibration_recovers_shape_and_shrinks() -> None:
    c = calibrate_curve(u_shape_days(250, 1), shrink=0.0)
    true = np.array([4, 2, 1.5, 1, 1, 1, 1.5, 2, 3, 5]) / 22
    assert np.max(np.abs(c - true)) < 0.01
    assert np.allclose(calibrate_curve(u_shape_days(5, 1), shrink=1.0), 0.1)


def test_forecast_vwap_beats_twap_out_of_sample() -> None:
    curve = calibrate_curve(u_shape_days(120, 2))
    test = u_shape_days(60, 3)
    s_v, s_t = [], []
    for day in test:
        fc = day.sum()  # day level known: isolates the shape
        s_v.append(execute_adaptive(1000, curve, fc, day, cap=0.5).slippage_vs_vwap_bucket_weights)
        s_t.append(execute_adaptive(1000, twap_curve(10), fc, day, cap=0.5).slippage_vs_vwap_bucket_weights)
    assert np.mean(s_v) < 0.6 * np.mean(s_t)


def test_cap_binds_and_reports_shortfall() -> None:
    day = np.full(10, 100.0)
    r = execute_adaptive(1000, twap_curve(10), 1000, day, cap=0.1)
    assert np.all(r.participation <= 0.1 + 1e-12)
    assert r.shortfall == pytest.approx(900)


def test_catch_up_after_cold_start_without_lookahead() -> None:
    # forecast says flat; the first half is cold, the second hot
    day = np.r_[np.full(5, 50.0), np.full(5, 150.0)]
    r = execute_adaptive(500, twap_curve(10), 1000, day, cap=0.5)
    assert r.child.sum() == pytest.approx(500)
    assert r.child[:5].sum() < 250  # fell behind while the cap bound
    # deciding bucket j never used volume j: changing the last bucket's
    # realised volume does not change earlier decisions
    day2 = day.copy()
    day2[-1] = 1e6
    r2 = execute_adaptive(500, twap_curve(10), 1000, day2, cap=0.5)
    assert np.allclose(r.child[:-1], r2.child[:-1])


def test_pov_and_close() -> None:
    day = np.full(10, 1000.0)
    r = execute_adaptive(2000, twap_curve(10), 10000, day, cap=0.3, mode="pov", pov_rate=0.1,
                         close_fraction=0.2, close_volume=5000)
    assert np.allclose(r.participation, 0.1)
    assert r.close_qty == pytest.approx(400)
    assert r.shortfall == pytest.approx(600)


def test_obizhaeva_wang_structure_and_cost() -> None:
    t = np.linspace(0, 10, 21)
    k = exponential_kernel(rho=0.3)
    x = optimal_schedule(1.0, t, k)
    assert x.sum() == pytest.approx(1.0)
    assert np.all(x > 0)
    assert np.allclose(x[1:-1], x[1])  # constant interior rate
    assert x[0] > x[1] and x[-1] > x[1]
    # closed form (Obizhaeva and Wang 2013, discrete): end/interior = 1/(1 - e^{-rho dt})
    a = np.exp(-0.3 * 0.5)
    assert x[0] / x[1] == pytest.approx(1 / (1 - a))
    twap = np.full(21, 1 / 21)
    assert expected_cost(x, t, k) < expected_cost(twap, t, k)


def test_power_law_kernel_optimum_is_valid_and_cheaper() -> None:
    t = np.linspace(0, 10, 21)
    k = power_law_kernel(gamma=0.5, tau=0.5)
    x = optimal_schedule(1.0, t, k)
    assert np.all(x > 0) and x[0] > x[10] < x[-1]
    assert min_round_trip_cost(t, k) > 0
    assert expected_cost(x, t, k) <= expected_cost(np.full(21, 1 / 21), t, k)


def test_non_convex_kernel_gives_sign_alternating_optimum() -> None:
    t = np.linspace(0, 10, 21)
    x = optimal_schedule(1.0, t, gaussian_kernel(width=2.0))
    assert np.any(x < 0)  # transaction-triggered manipulation (Alfonsi et al. 2012)


def test_gatheral_condition() -> None:
    assert gatheral_admissible(0.5, 0.5) and not gatheral_admissible(0.4, 0.5)


def test_fill_probability_monotone() -> None:
    ps = [fill_probability(q, 1, 1.0, 1.0, 5.0) for q in range(0, 20)]
    assert all(a >= b for a, b in zip(ps, ps[1:], strict=False))
    assert fill_probability(0, 1, 1.0, 1.0, 0.0) == 0.0


def test_placement_decision_regimes() -> None:
    kw = dict(qty=1, half_spread=0.5, take_fee=0.1, make_rebate=0.05, adverse=0.2,
              horizon=10.0, market_rate=0.5, mean_size=1.0)
    assert decide(queue_ahead=0, drift=0.0, **kw).action == "post"  # type: ignore[arg-type]
    assert decide(queue_ahead=40, drift=0.0, **kw).action == "cross"  # type: ignore[arg-type]
    assert decide(queue_ahead=6, drift=0.0, **kw).action == "post"  # type: ignore[arg-type]
    assert decide(queue_ahead=6, drift=0.5, **kw).action == "cross"  # type: ignore[arg-type]
    # adverse selection larger than the full spread plus rebate: never post
    kw2 = dict(kw, adverse=2.0)
    assert decide(queue_ahead=0, drift=0.0, **kw2).action == "cross"  # type: ignore[arg-type]
