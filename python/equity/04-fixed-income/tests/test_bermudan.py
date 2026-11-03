"""fi_rates.bermudan: Hull-White trinomial tree, bounds, convergence to the analytic European."""

from __future__ import annotations

import numpy as np
import pytest
from fi_rates.bermudan import bermudan_bounds, bermudan_swaption, bermudan_tree_core, build_time_grid
from fi_rates.curve import DiscountCurve
from fi_rates.hull_white import HullWhite

T = np.arange(1, 161) * 0.125
FLAT = DiscountCurve(T, np.exp(-0.03 * T))
STEEP = DiscountCurve(T, np.exp(-(0.01 + 0.004 * T) * T))
FIX, ACC = [1.0, 2.0, 3.0, 4.0, 5.0], [1.0] * 4


@pytest.mark.parametrize("curve", [FLAT, STEEP])
@pytest.mark.parametrize("payer", [True, False])
def test_between_bounds(curve, payer):
    res = bermudan_bounds(0.03, 0.01, curve, FIX, ACC, 0.03, FIX[:-1], payer)
    assert res.lower < res.price < res.upper


@pytest.mark.parametrize("payer", [True, False])
def test_single_exercise_converges_to_jamshidian(payer):
    # Flat curve: the analytic formula's f(0, t) (a central difference of
    # ln P) is exact there. On a log-linear curve f jumps at pillars and the
    # difference smears the jump, a 3.5e-6 gap that is the analytic side's.
    # The tree error oscillates with the step count (strike against node
    # spacing): -2.3e-6, +4.8e-6, -1.7e-6 at 50, 200, 400 steps a year, so
    # the bound is 1e-5 (0.1 bp of notional).
    hw = HullWhite(0.05, 0.012, FLAT)
    want = hw.swaption(2.0, 3.0, 0.025, 1, payer)
    got = bermudan_swaption(0.05, 0.012, FLAT, [2.0, 3.0, 4.0, 5.0], [1.0] * 3, 0.025, [2.0], payer,
                            steps_per_year=200)
    assert got == pytest.approx(want, abs=1e-5)


def test_tree_reprices_the_curve():
    # A cash flow of 2 at 5y, "exercised" as a receiver at t = 0: B - 1 = 2 P(0, 5) - 1.
    grid = build_time_grid([5.0], 20)
    cash = [0.0] * grid.size
    cash[-1] = 2.0
    dfs = [1.0, *np.asarray(STEEP.df(grid[1:]))]
    v = bermudan_tree_core(0.05, 0.01, list(grid), dfs, cash, [True] + [False] * (grid.size - 1),
                           payer=False)
    assert v == pytest.approx(2 * float(STEEP.df(5.0)) - 1.0, abs=1e-14)


def test_more_exercise_dates_are_worth_more():
    one = bermudan_swaption(0.03, 0.01, FLAT, FIX, ACC, 0.03, [1.0])
    two = bermudan_swaption(0.03, 0.01, FLAT, FIX, ACC, 0.03, [1.0, 3.0])
    four = bermudan_swaption(0.03, 0.01, FLAT, FIX, ACC, 0.03, FIX[:-1])
    assert one < two < four


def test_grid_and_validation():
    g = build_time_grid([0.5, 1.0], 4)
    assert g.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    with pytest.raises(ValueError):
        bermudan_swaption(0.03, 0.01, FLAT, FIX, ACC, 0.03, [1.5])
    with pytest.raises(ValueError):
        bermudan_swaption(0.03, 0.01, FLAT, FIX, ACC[:-1], 0.03, [1.0])
    with pytest.raises(ValueError):
        bermudan_tree_core(0.0, 0.01, [0.0, 1.0], [1.0, 0.97], [0.0, 1.0], [True, False], True)
