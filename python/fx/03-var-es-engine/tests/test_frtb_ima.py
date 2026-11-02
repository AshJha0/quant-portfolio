"""FRTB IMA: hand-computed cascade, stressed ES, IMCC, capital, backtests, PLA."""

import numpy as np
import pytest
from fx_var.frtb_ima import (
    desk_backtest,
    ima_capital,
    imcc,
    ks_statistic,
    liquidity_adjusted_es,
    pla_test,
    plus_factor,
    spearman,
    stressed_es,
)
from scipy.stats import ks_2samp, spearmanr


def test_cascade_hand_example():
    # sqrt(10^2 + (8 sqrt(1))^2 + (6 sqrt(2))^2 + (4 sqrt(2))^2 + (2 sqrt(6))^2)
    es = liquidity_adjusted_es([10.0, 8.0, 6.0, 4.0, 2.0])
    assert es == pytest.approx(np.sqrt(100 + 64 + 72 + 32 + 24), rel=1e-15)


def test_cascade_single_bucket_is_base_es():
    assert liquidity_adjusted_es([7.5]) == 7.5


def test_cascade_all_120_day_factors_is_sqrt_time():
    # One factor class with LH=120 shocked at every level: sqrt(sum of steps / 10)
    # telescopes to sqrt(120/10).
    assert liquidity_adjusted_es([1.0] * 5) == pytest.approx(np.sqrt(12.0), rel=1e-15)


def test_stressed_es_ratio_floor():
    assert stressed_es(30.0, 20.0, 10.0) == 60.0
    assert stressed_es(30.0, 5.0, 10.0) == 30.0  # ratio floored at 1


def test_imcc():
    assert imcc(100.0, [60.0, 50.0, 20.0]) == pytest.approx(0.5 * 100 + 0.5 * 130)


@pytest.mark.parametrize(("n", "pf"), [(0, 0.0), (4, 0.0), (5, 0.2), (9, 0.42), (10, 0.5), (40, 0.5)])
def test_plus_factor_table(n, pf):
    assert plus_factor(n) == pf


def test_capital_uses_average_when_larger():
    i = np.full(60, 100.0)
    s = np.full(60, 10.0)
    assert ima_capital(i, s, 0) == pytest.approx(1.5 * 100 + 10)
    assert ima_capital(i, s, 6) == pytest.approx(1.76 * 100 + 10)
    i[-1] = 1000.0
    assert ima_capital(i, s, 0) == pytest.approx(1010.0)


def test_desk_backtest_thresholds():
    pnl = np.zeros(250)
    v99, v975 = np.full(250, 1.0), np.full(250, 0.5)
    pnl[:12] = -2.0
    assert desk_backtest(pnl, v99, v975).eligible
    pnl[12] = -2.0
    r = desk_backtest(pnl, v99, v975)
    assert (r.exceptions_99, r.eligible) == (13, False)
    pnl = np.zeros(250)
    pnl[:31] = -0.7  # breaches only the 97.5 % line
    r = desk_backtest(pnl, v99, v975)
    assert (r.exceptions_99, r.exceptions_975, r.eligible) == (0, 31, False)


def test_spearman_and_ks_match_scipy():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(300)
    y = 0.8 * x + 0.3 * rng.standard_normal(300)
    y[:5] = y[5]  # ties
    assert spearman(x, y) == pytest.approx(spearmanr(x, y).statistic, rel=1e-12)
    assert ks_statistic(x, y) == pytest.approx(ks_2samp(x, y).statistic, rel=1e-12)


def test_pla_zones():
    rng = np.random.default_rng(9)
    h = rng.standard_normal(250)
    assert pla_test(h, h + 1e-3 * rng.standard_normal(250)).zone == "green"
    assert pla_test(h, rng.standard_normal(250)).zone == "red"
    # same ranks but a shifted distribution: Spearman 1, KS = 0.104 -> amber
    r = pla_test(h, h + 0.18)
    assert r.spearman == pytest.approx(1.0)
    assert 0.09 < r.ks <= 0.12 and r.zone == "amber"


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        liquidity_adjusted_es([1.0] * 6)
    with pytest.raises(ValueError):
        liquidity_adjusted_es([-1.0])
    with pytest.raises(ValueError):
        stressed_es(1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        pla_test(np.zeros(100), np.zeros(100))
