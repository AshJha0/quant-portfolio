"""Stress testing: historical replays, peg breaks, ladders, reverse stress."""

import numpy as np
import pandas as pd
import pytest

from fx_var import (
    Book,
    Option,
    PegBlindnessWarning,
    Spot,
    historical_scenarios,
    historical_var,
    peg_break_scenario,
    reverse_stress_linear,
    reverse_stress_numerical,
    run_stress,
    sensitivity_ladder,
    usd_broad_move,
)
from fx_var.data.synthetic import demo_book, demo_market


@pytest.fixture()
def market():
    return demo_market()


def test_scenario_library_contents():
    lib = historical_scenarios()
    assert {"brexit_2016", "chf_depeg_2015", "jpy_1998", "em_crisis"} <= set(lib)
    assert lib["brexit_2016"].shocks["FX:GBP"] == pytest.approx(np.log(1 - 0.081))
    assert lib["chf_depeg_2015"].shocks["FX:CHF"] == pytest.approx(np.log(1.149))
    for sc in lib.values():
        assert sc.description


def test_run_stress_ignores_unheld_factors(market):
    """A GBP-only book must not be moved by the EUR/JPY legs of Brexit."""
    book = Book([Spot("GBPUSD", -10e6)])  # short cable
    out = run_stress(book, market, historical_scenarios())
    brexit = out.loc["brexit_2016", "pnl"]
    # short 10m GBP, GBP -8.1%: pnl = -10e6*1.27*(0.919-1)
    assert brexit == pytest.approx(-10e6 * 1.27 * (np.exp(np.log(1 - 0.081)) - 1), rel=1e-9)
    assert brexit > 0


def test_run_stress_sorted_worst_first(market):
    out = run_stress(demo_book(), market, historical_scenarios())
    assert list(out["pnl"]) == sorted(out["pnl"])
    assert {"scenario", "pnl", "description"} == set(out.columns)


def test_usd_broad_move_signs(market):
    book = Book([Spot("EURUSD", 10e6), Spot("USDJPY", 5e6)])
    up = run_stress(book, market, [usd_broad_move(["EUR", "JPY"], +0.10)])
    dn = run_stress(book, market, [usd_broad_move(["EUR", "JPY"], -0.10)])
    # long EUR leg loses when USD +10%; long USDJPY gains: net = EUR dominates
    p_up, p_dn = up["pnl"].iloc[0], dn["pnl"].iloc[0]
    assert p_up < 0 < p_dn
    with pytest.raises(ValueError):
        usd_broad_move(["EUR"], -1.5)


def test_usd_broad_move_is_exact_simple_move(market):
    """USD +10% means every CCYUSD divides by 1.10 exactly."""
    sc = usd_broad_move(["EUR"], 0.10)
    book = Book([Spot("EURUSD", 1e6)])
    pnl = book.pnl(market, dict(sc.shocks))
    assert pnl == pytest.approx(1e6 * 1.08 * (1 / 1.1 - 1), rel=1e-12)


# ------------------------------------------------------------ peg break
def test_peg_break_supplies_the_loss_hs_missed(market):
    """The core peg story: HS VaR on the pegged book ~ 0 (with a warning),
    while the peg-break scenario produces the true regime-break loss."""
    hkd_usd_notional = 50e6
    book = Book([Spot("USDHKD", -hkd_usd_notional)])  # long HKD vs USD
    rng = np.random.default_rng(0)
    rets = pd.DataFrame({"FX:HKD": rng.normal(0, 1.2e-4, 500)})  # band noise
    with pytest.warns(PegBlindnessWarning):
        hs = historical_var(book, market, rets, 0.99)
    scen = peg_break_scenario("HKD", jump=-0.30)
    stress = run_stress(book, market, [scen])
    loss = -stress["pnl"].iloc[0]
    # exact: long N*X0 HKD in USD terms loses 30% of its USD value
    assert loss == pytest.approx(hkd_usd_notional * 0.30, rel=1e-9)
    assert hs.var < 0.002 * hkd_usd_notional  # HS saw essentially nothing
    assert loss > 100 * hs.var  # the stress add-on is the real risk number


def test_peg_break_contagion_and_vol(market):
    sc = peg_break_scenario("HKD", jump=-0.20, vol_spike=0.10,
                            vol_pairs=["USDHKD"], contagion={"SAR": -0.05})
    assert sc.shocks["FX:HKD"] == pytest.approx(np.log(0.8))
    assert sc.shocks["FX:SAR"] == pytest.approx(np.log(0.95))
    assert sc.shocks["VOL:USDHKD"] == 0.10
    with pytest.raises(ValueError):
        peg_break_scenario("HKD", jump=-1.2)


def test_chf_style_upward_break(market):
    """Positive jump: a short-CHF-style book vs a revaluation (SNB 2015)."""
    book = Book([Spot("USDCHF", 20e6)])  # long USD vs CHF = short CHF
    sc = peg_break_scenario("CHF", jump=+0.15)
    out = run_stress(book, market, [sc])
    assert out["pnl"].iloc[0] < 0  # short CHF loses on CHF revaluation


# ------------------------------------------------------------ ladders
def test_sensitivity_ladder_linear_monotone(market):
    book = Book([Spot("EURUSD", 10e6)])
    lad = sensitivity_ladder(book, market, "FX:EUR")
    assert (np.diff(lad["pnl"]) > 0).all()  # long EUR: monotone in shock
    assert lad.loc[lad["shock"] == 0.0, "pnl"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_sensitivity_ladder_option_convexity(market):
    """Long option: P&L ladder in the FX factor is convex (positive gamma)."""
    book = Book([Option("EURUSD", 30e6, 1.10, 0.25, "call")])
    grid = np.linspace(-0.06, 0.06, 13)
    lad = sensitivity_ladder(book, market, "FX:EUR", shocks=grid)
    second_diff = np.diff(lad["pnl"], 2)
    assert (second_diff > 0).all()


def test_sensitivity_ladder_default_grids(market):
    book = Book([Spot("EURUSD", 1e6)])
    assert len(sensitivity_ladder(book, market, "FX:EUR")) == 9
    with pytest.raises(ValueError):
        sensitivity_ladder(book, market, "XX:EUR")


# ------------------------------------------------------------ reverse stress
def _toy_linear():
    w = pd.Series({"FX:EUR": 1.2e7, "FX:JPY": -6e6, "FX:GBP": 4e6})
    sd = np.array([0.005, 0.006, 0.0055])
    corr = np.array([[1.0, 0.3, 0.6], [0.3, 1.0, 0.4], [0.6, 0.4, 1.0]])
    cov = pd.DataFrame(corr * np.outer(sd, sd), index=w.index, columns=w.index)
    return w, cov


def test_reverse_stress_closed_form_loss():
    """Loss at radius k is exactly k * sqrt(w' Sigma w)."""
    w, cov = _toy_linear()
    k = 3.0
    shocks, loss = reverse_stress_linear(w, cov, radius=k)
    sigma_p = float(np.sqrt(w @ cov @ w))
    assert loss == pytest.approx(k * sigma_p, rel=1e-12)
    # the shock indeed produces that loss on the linearised book
    assert -(w @ shocks) == pytest.approx(loss, rel=1e-12)
    # and lies exactly on the ellipsoid boundary
    m = float(shocks @ np.linalg.inv(cov.to_numpy()) @ shocks)
    assert np.sqrt(m) == pytest.approx(k, rel=1e-9)


def test_reverse_stress_numerical_confirms_closed_form():
    w, cov = _toy_linear()
    k = 2.5
    sh_cf, loss_cf = reverse_stress_linear(w, cov, radius=k)
    sh_num, loss_num = reverse_stress_numerical(w, cov, radius=k, seed=1)
    assert loss_num == pytest.approx(loss_cf, rel=1e-4)
    np.testing.assert_allclose(sh_num.to_numpy(), sh_cf.to_numpy(), rtol=5e-3,
                               atol=1e-6)


def test_reverse_stress_loss_target_inversion():
    w, cov = _toy_linear()
    target = 1_000_000.0
    shocks, loss = reverse_stress_linear(w, cov, loss_target=target)
    assert loss == pytest.approx(target, rel=1e-12)


def test_reverse_stress_validation():
    w, cov = _toy_linear()
    with pytest.raises(ValueError, match="exactly one"):
        reverse_stress_linear(w, cov)
    with pytest.raises(ValueError, match="exactly one"):
        reverse_stress_linear(w, cov, loss_target=1.0, radius=1.0)
    with pytest.raises(ValueError, match="positive"):
        reverse_stress_linear(w, cov, radius=-1.0)
    zero_w = pd.Series({"FX:EUR": 0.0})
    zero_cov = pd.DataFrame([[0.01]], index=["FX:EUR"], columns=["FX:EUR"])
    with pytest.raises(ValueError, match="zero linear risk"):
        reverse_stress_linear(zero_w, zero_cov, radius=1.0)


def test_reverse_stress_direction_hurts_the_book(market):
    """Applying the worst-direction shocks to the actual book produces a
    loss close to the linear prediction (spot book, small shocks)."""
    book = Book([Spot("EURUSD", 12e6), Spot("USDJPY", -6e6)])
    w = book.linear_exposures(market)
    sd = np.array([0.005, 0.006])
    cov = pd.DataFrame(np.diag(sd**2), index=w.index, columns=w.index)
    shocks, loss = reverse_stress_linear(w, cov, radius=2.33)
    pnl = book.pnl(market, shocks.to_dict())
    assert pnl < 0
    assert -pnl == pytest.approx(loss, rel=0.05)  # full reval vs linear
