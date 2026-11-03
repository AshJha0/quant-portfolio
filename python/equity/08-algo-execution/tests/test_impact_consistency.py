"""W3: the execution cost model is consistent across slicing and laws.

Wrong answers these tests pin (v2.0.0):

1. The simulator charged ``sigma * sqrt(q_j / (n_buckets * V_j))`` with
   ``V_j`` already a *bucket* volume, so the same continuous trading-rate
   profile cost less the finer it was sliced (cost ~ 1/sqrt(n)).
2. The Almgren-Chriss optimiser assumed linear temporary impact while the
   simulator charged square-root impact, and nothing solved the problem
   under the law the simulator uses.
3. POV took "forecast or realised" volumes through one argument, so the
   examples could hand it the realised tape (foresight).
4. ``expected_is_bps`` reads as an expectation over real markets; it is
   the cost implied by the model with the noise switched off.

Reference for the continuous cost: Almgren, Thum, Hauptmann and Li (2005),
"Direct estimation of equity market impact", Risk 18(7): temporary cost
rate ``eta * sigma * (v / V)^delta`` per share, ``v`` our trading rate and
``V`` the market volume rate, both per day.
"""

from __future__ import annotations

import numpy as np
import pytest
from eq_algo import (
    ACParams,
    IntradayConfig,
    IntradayMarket,
    ac_trajectory,
    compare_impact_laws,
    evaluate_schedules,
    pov_schedule,
    power_law_trajectory,
    twap_schedule,
)
from scipy.integrate import quad

MID, SIGMA, ETA, VDAY, X = 100.0, 0.02, 0.3, 1e6, 1e5


def rate(t):
    """Our trading rate, shares per day, integrating to X over [0, 1]."""
    return X * (1.0 + 0.5 * np.sin(2.0 * np.pi * t))


def flat_volume(t):
    return VDAY * np.ones_like(np.asarray(t, dtype=float))


def u_volume(t):
    """U-shaped market volume rate, shares per day, integrating to VDAY."""
    return VDAY * (1.0 + 3.0 * (2.0 * np.asarray(t, dtype=float) - 1.0) ** 2) / 2.0


def simulated_temp_cost(n, volume_rate, delta):
    cfg = IntradayConfig(mid0=MID, day_volume=VDAY, n_buckets=n, sigma_daily=SIGMA,
                         spread_bps=0.0, temp_coef=ETA, perm_coef=0.0,
                         price_noise=0.0, impact_exponent=delta)
    t = (np.arange(n) + 0.5) / n
    q = rate(t) / n                       # shares in each bucket
    bucket_vol = volume_rate(t) / n       # bucket volume scaled to bucket width
    res = IntradayMarket(cfg).execute(q, market_volumes=bucket_vol, seed=0)
    return float((res.fills["temp_cost"] * res.fills["qty"]).sum())


def quadrature_cost(volume_rate, delta):
    f = lambda t: MID * ETA * SIGMA * (rate(t) / volume_rate(t)) ** delta * rate(t)  # noqa: E731
    return quad(f, 0.0, 1.0, epsabs=1e-12, epsrel=1e-13, limit=200)[0]


@pytest.mark.parametrize("volume_rate", [flat_volume, u_volume], ids=["flat", "u_shape"])
@pytest.mark.parametrize("delta", [0.5, 0.6, 1.0])
def test_bucket_cost_converges_to_continuous_integral(volume_rate, delta):
    exact = quadrature_cost(volume_rate, delta)
    errs = [abs(simulated_temp_cost(n, volume_rate, delta) - exact) / exact
            for n in (10, 20, 40, 80)]
    # O(1/n) or better: n * err stays bounded, and the finest grid is close.
    assert max(n * e for n, e in zip((10, 20, 40, 80), errs, strict=True)) < 0.5
    assert errs[-1] < 5e-3
    assert errs[-1] < errs[0]


def test_twap_cost_does_not_fall_with_finer_slicing():
    """The v2.0.0 bug in one line: same order, more buckets, lower cost."""
    costs = []
    for n in (13, 26, 52, 104):
        cfg = IntradayConfig(n_buckets=n, price_noise=0.0, perm_coef=0.0, curvature=0.0,
                             spread_bps=0.0)
        res = IntradayMarket(cfg).execute(twap_schedule(X, n), seed=0)
        costs.append(float((res.fills["temp_cost"] * res.fills["qty"]).sum()))
    assert np.allclose(costs, costs[0], rtol=1e-12)


@pytest.mark.parametrize("lam", [0.0, 1e-7, 1e-6, 1e-5])
def test_power_law_solver_matches_closed_form_at_exponent_one(lam):
    p = ACParams(total_shares=X, n_slices=20, total_time=1.0, sigma=2.0, eta=1e-5)
    c = p.eta_tilde / p.tau                         # cost = c * n_k^2 per slice
    x = power_law_trajectory(X, c * np.ones(20), 1.0, lam, p.sigma**2 * p.tau)
    np.testing.assert_allclose(x, ac_trajectory(p, lam), rtol=0, atol=1e-8 * X)


def test_power_law_solver_is_optimal_under_its_own_law():
    """Under the square-root law the DP schedule beats the linear-law AC one."""
    cfg = IntradayConfig(price_noise=0.0, vol_noise=0.0)
    tab = compare_impact_laws(cfg, parent_qty=X, risk_aversion=1e-6)
    assert set(tab.index) == {"ac_linear", "power_law"}
    assert {"cost_linear_law_bps", "cost_power_law_bps",
            "objective_linear_law", "objective_power_law"} <= set(tab.columns)
    assert tab.loc["power_law", "objective_power_law"] <= tab.loc["ac_linear", "objective_power_law"] + 1e-9
    assert tab.loc["ac_linear", "objective_linear_law"] <= tab.loc["power_law", "objective_linear_law"] + 1e-9


def test_pov_takes_a_forecast_and_warns_on_the_old_keyword():
    fc = np.full(10, 1e5)
    np.testing.assert_allclose(pov_schedule(5e4, forecast_volumes=fc, participation=0.1),
                               np.r_[np.full(5, 1e4), np.zeros(5)])
    with pytest.warns(FutureWarning, match="forecast_volumes"):
        pov_schedule(5e4, market_volumes=fc, participation=0.1)


def test_model_implied_cost_column_is_labelled():
    cfg = IntradayConfig(n_buckets=6)
    tab = evaluate_schedules(IntradayMarket(cfg), {"twap": twap_schedule(1e4, 6)}, n_reps=3)
    assert "model_implied_is_bps" in tab.columns
    assert tab["model_implied_is_bps"].equals(tab["expected_is_bps"])
