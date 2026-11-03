"""W26 RFQ: closed-form optimum, skew direction, MLE recovery, analytics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qp_exec.rfq import (
    HitRatioModel,
    LastLookRecord,
    RfqRecord,
    expected_profit,
    fit_hit_ratio,
    lambert_w0,
    last_look_report,
    markouts,
    optimal_markup,
    skewed_markup,
    win_rates,
)


@pytest.mark.parametrize("x", [0.0, 1e-8, 0.5, 1.0, math.e, 10.0, 1e5])
def test_lambert_w_identity(x: float) -> None:
    w = lambert_w0(x)
    assert w * math.exp(w) == pytest.approx(x, rel=1e-14, abs=1e-300)


def test_omega_constant() -> None:
    assert lambert_w0(1.0) == pytest.approx(0.5671432904097838, rel=1e-15)


@pytest.mark.parametrize(("a", "b"), [(1.0, -200.0), (-2.0, -50.0), (3.0, -1000.0)])
def test_optimal_markup_is_argmax(a: float, b: float) -> None:
    hm = HitRatioModel(a, b)
    m = optimal_markup(hm)
    grid = np.linspace(0.0, 5 * m, 20001)
    best = grid[np.argmax([expected_profit(hm, g) for g in grid])]
    assert m == pytest.approx(best, abs=5 * m / 20000)
    # FOC: 1 + b m (1 - P) = 0.
    assert 1 + b * m * (1 - hm.p_win(m)) == pytest.approx(0.0, abs=1e-12)


def test_skew_reduces_to_optimal_with_no_inventory() -> None:
    hm = HitRatioModel(0.5, -100.0)
    m0 = skewed_markup(hm, inventory=0.0, size=1.0, firm_side=1, risk_aversion=1.0, sigma=0.1)
    assert m0 == pytest.approx(optimal_markup(hm), rel=1e-14)


def test_skew_direction_and_argmax() -> None:
    hm = HitRatioModel(0.5, -100.0)
    kw = {"size": 2.0, "risk_aversion": 5.0, "sigma": 0.05}
    sell_long = skewed_markup(hm, inventory=10.0, firm_side=-1, **kw)  # reduces risk
    buy_long = skewed_markup(hm, inventory=10.0, firm_side=1, **kw)  # adds risk
    m0 = optimal_markup(hm)
    assert sell_long < m0 < buy_long
    v = 5.0 * 0.05**2 * 1.0 * 10.0 * 2.0
    grid = np.linspace(-0.1, 0.2, 30001)
    best = grid[np.argmax([expected_profit(hm, g, v) for g in grid])]
    assert sell_long == pytest.approx(best, abs=2e-5)


def test_mle_recovers_parameters() -> None:
    rng = np.random.default_rng(0)
    true = HitRatioModel(1.0, -150.0)
    m = rng.uniform(0.0, 0.03, 20000)
    won = (rng.random(m.size) < np.array([true.p_win(x) for x in m])).astype(float)
    fit, se = fit_hit_ratio(m, won)
    assert abs(fit.intercept - 1.0) < 4 * se[0]
    assert abs(fit.slope + 150.0) < 4 * se[1]


def test_win_rates_and_markouts_hand() -> None:
    mids = [100.0, 100.5, 101.0, 99.0]
    recs = [
        RfqRecord("A", 1, True, 100.0, 2.0, 0),  # firm bought at 100
        RfqRecord("B", -1, True, 100.0, 1.0, 0),  # firm sold at 100
        RfqRecord("B", -1, False, 100.0, 1.0, 1),
    ]
    wr = win_rates(recs)
    assert wr["A"].rate == 1.0 and wr["B"].rate == 0.5 and wr["ALL"].n == 3
    mk = markouts(recs, mids, [1, 2])
    # h=1: A: +0.5/unit, B: -0.5/unit; size-weighted (2*0.5 - 1*0.5)/3.
    assert mk[("ALL", 1)].mean == pytest.approx(0.5 / 3)
    assert mk[("A", 2)].mean == pytest.approx(1.0)
    assert mk[("B", 2)].mean == pytest.approx(-1.0)


def test_last_look_report() -> None:
    rng = np.random.default_rng(3)
    recs = []
    for _ in range(4000):
        move = float(rng.normal())
        # Asymmetric dealer: rejects 30% when the move is against it, 2% otherwise.
        p_rej = 0.3 if move > 0 else 0.02
        recs.append(LastLookRecord(float(rng.uniform(1, 50)), bool(rng.random() >= p_rej), move))
    rep = last_look_report(recs, max_hold_ms=60.0)
    assert rep.asymmetry == pytest.approx(0.28, abs=4 * rep.asymmetry_se)
    assert rep.asymmetry / rep.asymmetry_se > 10
    assert rep.hold_max_ms <= 50.0 and rep.hold_median_ms < rep.hold_p95_ms
    with pytest.raises(ValueError):
        last_look_report(recs, max_hold_ms=10.0)
    with pytest.raises(ValueError):
        last_look_report([])
