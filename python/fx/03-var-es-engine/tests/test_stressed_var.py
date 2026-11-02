"""Stressed VaR: planted stress window, capital formula by hand."""

import numpy as np
import pytest
from fx_var.stressed_var import basel25_capital, find_stress_window, stressed_var


def test_planted_stress_window():
    rng = np.random.default_rng(2)
    pnl = rng.standard_normal(1500)
    pnl[600:850] *= 4.0  # a 250-day crisis
    w = find_stress_window(pnl, 250, 0.01)
    # Every window holding the crisis tail has the same VaR; the earliest wins.
    assert 350 <= w.start <= 600 and w.stop - w.start == 250
    assert w.var > 3 * find_stress_window(pnl[:600], 250, 0.01).var
    assert w.es >= w.var
    assert stressed_var(pnl, 250, 0.01) == w.var


def test_window_var_is_historical_var():
    # alpha n = 3: VaR is the 3rd worst loss, ES the mean of the worst 3.
    p = np.random.default_rng(4).standard_normal(300)
    w = find_stress_window(p, 300, 0.01)
    worst = np.sort(-p)[::-1]
    assert w.var == worst[2]
    assert w.es == pytest.approx(worst[:3].mean(), rel=1e-14)


def test_capital_by_hand():
    v = np.full(60, 10.0)
    s = np.full(60, 25.0)
    assert basel25_capital(v, s) == pytest.approx(30.0 + 75.0)
    v[-1] = 40.0
    assert basel25_capital(v, s) == pytest.approx(40.0 + 75.0)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        basel25_capital(np.ones(59), np.ones(60))
    with pytest.raises(ValueError):
        basel25_capital(np.ones(60), np.ones(60), m_c=2.0)
    with pytest.raises(ValueError):
        find_stress_window(np.ones(100), 250)
