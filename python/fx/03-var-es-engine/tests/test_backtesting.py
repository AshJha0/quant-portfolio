"""Kupiec, Christoffersen, Basel traffic light, ES backtest, rolling tests."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm

from fx_var import (
    basel_traffic_light,
    christoffersen_independence,
    conditional_coverage,
    es_backtest_acerbi_szekely,
    evaluate_var_backtest,
    kupiec_pof,
    normal_es,
    rolling_backtest,
)


# ------------------------------------------------------------ Kupiec
def test_kupiec_hand_computed_250_5():
    """n=250, x=5, p=0.01: LR = -2[(245 ln .99 + 5 ln .01)
    - (245 ln .98 + 5 ln .02)] = 1.9568..."""
    lr, p = kupiec_pof(5, 250, 0.99)
    hand = -2.0 * ((245 * np.log(0.99) + 5 * np.log(0.01))
                   - (245 * np.log(0.98) + 5 * np.log(0.02)))
    assert lr == pytest.approx(hand, abs=1e-12)
    assert lr == pytest.approx(1.9568, abs=1e-3)
    assert p == pytest.approx(chi2.sf(hand, 1), abs=1e-12)
    assert p > 0.05  # 5 exceptions in 250 days is not rejectable


def test_kupiec_zero_exceptions():
    lr, p = kupiec_pof(0, 250, 0.99)
    hand = -2.0 * 250 * np.log(0.99)
    assert lr == pytest.approx(hand, abs=1e-12)
    assert np.isfinite(p)


def test_kupiec_rejects_bad_model():
    _, p = kupiec_pof(15, 250, 0.99)
    assert p < 0.001


def test_kupiec_exact_coverage_is_minimum():
    """LR is ~0 when observed rate equals nominal."""
    lr, p = kupiec_pof(5, 500, 0.99)
    assert lr == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_kupiec_invalid_inputs():
    with pytest.raises(ValueError):
        kupiec_pof(-1, 250, 0.99)
    with pytest.raises(ValueError):
        kupiec_pof(251, 250, 0.99)
    with pytest.raises(ValueError):
        kupiec_pof(5, 250, 1.0)


# ------------------------------------------------------------ Christoffersen
def test_christoffersen_hand_computed():
    """Sequence 0,1,1,0,1,1,1,0,0,0: transition counts n00=2, n01=2,
    n10=2, n11=3, so pi01=1/2, pi11=3/5, pi=5/9 and
    LR = -2[(4 ln(4/9) + 5 ln(5/9)) - (4 ln(1/2) + 2 ln(2/5) + 3 ln(3/5))]
       = 0.0900139..."""
    e = [0, 1, 1, 0, 1, 1, 1, 0, 0, 0]
    lr, p = christoffersen_independence(e)
    hand = -2.0 * ((4 * np.log(4 / 9) + 5 * np.log(5 / 9))
                   - (4 * np.log(1 / 2) + 2 * np.log(2 / 5) + 3 * np.log(3 / 5)))
    assert lr == pytest.approx(hand, abs=1e-12)
    assert lr == pytest.approx(0.0900139, abs=1e-6)
    assert p == pytest.approx(chi2.sf(hand, 1), abs=1e-12)


def test_christoffersen_degenerate_symmetric_sequence_is_zero():
    """0,0,1,1,0,0,0,1,0,0 has pi01 = pi11 = pi = 1/3: LR exactly 0."""
    lr, p = christoffersen_independence([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    assert lr == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_christoffersen_detects_clustering():
    """All exceptions bunched: independence strongly rejected."""
    e = np.zeros(250, dtype=int)
    e[100:110] = 1
    _, p = christoffersen_independence(e)
    assert p < 0.001


def test_christoffersen_no_exceptions_degenerate():
    lr, p = christoffersen_independence(np.zeros(100, dtype=int))
    assert lr == 0.0
    assert p == 1.0


def test_christoffersen_invalid_inputs():
    with pytest.raises(ValueError):
        christoffersen_independence([1])
    with pytest.raises(ValueError):
        christoffersen_independence([0, 2, 1])


def test_conditional_coverage_is_sum():
    e = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
    lr_cc, p_cc = conditional_coverage(e, 0.9)
    lr_uc, _ = kupiec_pof(3, 10, 0.9)
    lr_ind, _ = christoffersen_independence(e)
    assert lr_cc == pytest.approx(lr_uc + lr_ind, abs=1e-12)
    assert p_cc == pytest.approx(chi2.sf(lr_cc, 2), abs=1e-12)


# ------------------------------------------------------------ Basel zones
def test_basel_boundaries_exact_4_5_and_9_10():
    """Regulatory table: green 0-4, yellow 5-9, red 10+ (250d, 99%)."""
    assert basel_traffic_light(4).zone == "green"
    assert basel_traffic_light(5).zone == "yellow"
    assert basel_traffic_light(9).zone == "yellow"
    assert basel_traffic_light(10).zone == "red"


@pytest.mark.parametrize("x,zone,mult", [
    (0, "green", 3.0), (1, "green", 3.0), (2, "green", 3.0),
    (3, "green", 3.0), (4, "green", 3.0),
    (5, "yellow", 3.40), (6, "yellow", 3.50), (7, "yellow", 3.65),
    (8, "yellow", 3.75), (9, "yellow", 3.85),
    (10, "red", 4.0), (12, "red", 4.0), (20, "red", 4.0),
])
def test_basel_multiplier_table(x, zone, mult):
    tl = basel_traffic_light(x)
    assert tl.zone == zone
    assert tl.multiplier == pytest.approx(mult)


def test_basel_cumulative_probability_values():
    """Cross-check the binomial cumulative probs behind the zone cuts."""
    assert basel_traffic_light(4).cumulative_prob == pytest.approx(0.8922, abs=1e-3)
    assert basel_traffic_light(5).cumulative_prob == pytest.approx(0.9588, abs=1e-3)
    assert basel_traffic_light(9).cumulative_prob < 0.9999
    assert basel_traffic_light(10).cumulative_prob >= 0.9999


def test_basel_invalid():
    with pytest.raises(ValueError):
        basel_traffic_light(-1)
    with pytest.raises(ValueError):
        basel_traffic_light(300, 250)


# ------------------------------------------------------------ evaluate
def test_evaluate_var_backtest_counts():
    pnl = np.array([-5.0, 1.0, -0.5, -3.0, 2.0, -1.1])
    var = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 1.0])
    res = evaluate_var_backtest(pnl, var, 0.99)
    np.testing.assert_array_equal(res.exceedances, [1, 0, 0, 1, 0, 1])
    assert res.n_exceptions == 3
    assert res.exception_rate == pytest.approx(0.5)


def test_evaluate_var_backtest_validation():
    with pytest.raises(ValueError):
        evaluate_var_backtest([1.0, 2.0], [1.0], 0.99)
    with pytest.raises(ValueError):
        evaluate_var_backtest([1.0, np.nan], [1.0, 1.0], 0.99)


# ------------------------------------------------------------ ES backtest
def test_es_backtest_correct_model_accepted():
    """Losses simulated from the model used for the forecasts: p large."""
    rng = np.random.default_rng(6)
    n, sigma, alpha = 1000, 1.0, 0.975
    pnl = rng.standard_normal(n) * sigma
    var = np.full(n, sigma * norm.ppf(alpha))
    es = np.full(n, normal_es(sigma, alpha))
    z, p = es_backtest_acerbi_szekely(pnl, var, es, alpha, seed=1)
    assert abs(z) < 0.5
    assert p > 0.05


def test_es_backtest_rejects_understated_es():
    """True vol double the forecast: Z >> 0 and p tiny."""
    rng = np.random.default_rng(7)
    n, alpha = 1000, 0.975
    pnl = rng.standard_normal(n) * 2.0  # true sigma 2
    var = np.full(n, 1.0 * norm.ppf(alpha))  # forecast sigma 1
    es = np.full(n, normal_es(1.0, alpha))
    z, p = es_backtest_acerbi_szekely(pnl, var, es, alpha, seed=2)
    assert z > 0.5
    assert p < 0.01


def test_es_backtest_validation():
    with pytest.raises(ValueError):
        es_backtest_acerbi_szekely([1.0], [1.0, 2.0], [1.0], 0.975)
    with pytest.raises(ValueError):
        es_backtest_acerbi_szekely([1.0], [1.0], [-1.0], 0.975)


# ------------------------------------------------------------ rolling
def test_rolling_backtest_shapes_and_window():
    from fx_var import Book, Spot
    from fx_var.data.synthetic import demo_market, simulate_history

    market = demo_market()
    book = Book([Spot("EURUSD", 10e6)])
    rets = simulate_history(book, market, 400, seed=3)

    def var_fn(bk, mkt, window):
        from fx_var import historical_var

        return historical_var(bk, mkt, window, 0.99).var

    out = rolling_backtest(book, market, rets, var_fn, window=250)
    assert len(out) == 150
    assert set(out.columns) == {"pnl", "var", "exceed"}
    assert (out["var"] > 0).all()
    with pytest.raises(ValueError, match="window"):
        rolling_backtest(book, market, rets.iloc[:100], var_fn, window=250)
    with pytest.raises(ValueError):
        rolling_backtest(book, market, rets, var_fn, window=10)


def test_garch_data_parametric_normal_fails_fhs_passes():
    """The flagship backtest story: on GARCH-simulated FX data over 500
    days, unconditional parametric-normal VaR gets clustered exceptions
    and fails conditional coverage; FHS passes."""
    from fx_var import Book, Spot, historical_var, parametric_var
    from fx_var.data.synthetic import demo_market, simulate_history

    market = demo_market()
    book = Book([Spot("EURUSD", 10e6), Spot("USDJPY", 8e6)])
    rets = simulate_history(book, market, 750, seed=29, garch=True,
                            regime_switching=True)

    def fn_param(bk, mkt, window):
        return parametric_var(bk, mkt, window, 0.99, min_obs=50).var

    def fn_fhs(bk, mkt, window):
        return historical_var(bk, mkt, window, 0.99, method="fhs", min_obs=50).var

    bt_p = rolling_backtest(book, market, rets, fn_param, window=250)
    bt_f = rolling_backtest(book, market, rets, fn_fhs, window=250)
    res_p = evaluate_var_backtest(bt_p["pnl"], bt_p["var"], 0.99)
    res_f = evaluate_var_backtest(bt_f["pnl"], bt_f["var"], 0.99)
    # parametric-normal: too many, *clustered* exceptions -> CC and
    # independence both rejected
    assert res_p.cc_p < 0.05
    assert res_p.independence_p < 0.05
    # FHS: conditional coverage acceptable
    assert res_f.cc_p > 0.05
    assert res_f.n_exceptions < res_p.n_exceptions
