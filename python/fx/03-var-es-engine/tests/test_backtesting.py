"""Kupiec, Christoffersen, Basel traffic light, ES backtest, rolling tests."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm

from fx_var import (
    acerbi_szekely_z2,
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
    lr, p = kupiec_pof(n_obs=250, n_exceptions=5, alpha=0.01)
    hand = -2.0 * ((245 * np.log(0.99) + 5 * np.log(0.01))
                   - (245 * np.log(0.98) + 5 * np.log(0.02)))
    assert lr == pytest.approx(hand, abs=1e-12)
    assert lr == pytest.approx(1.9568, abs=1e-3)
    assert p == pytest.approx(chi2.sf(hand, 1), abs=1e-12)
    assert p > 0.05  # 5 exceptions in 250 days is not rejectable


def test_kupiec_zero_exceptions():
    lr, p = kupiec_pof(n_obs=250, n_exceptions=0, alpha=0.01)
    hand = -2.0 * 250 * np.log(0.99)
    assert lr == pytest.approx(hand, abs=1e-12)
    assert np.isfinite(p)


def test_kupiec_rejects_bad_model():
    _, p = kupiec_pof(n_obs=250, n_exceptions=15, alpha=0.01)
    assert p < 0.001


def test_kupiec_exact_coverage_is_minimum():
    """LR is ~0 when observed rate equals nominal."""
    lr, p = kupiec_pof(n_obs=500, n_exceptions=5, alpha=0.01)
    assert lr == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_kupiec_invalid_inputs():
    with pytest.raises(ValueError):
        kupiec_pof(n_obs=250, n_exceptions=-1, alpha=0.01)
    with pytest.raises(ValueError):
        kupiec_pof(n_obs=250, n_exceptions=251, alpha=0.01)
    with pytest.raises(ValueError):
        kupiec_pof(n_obs=250, n_exceptions=5, alpha=0.0)


def _exact_kupiec_test_size(n_obs: int, alpha: float, nominal: float = 0.05) -> float:
    """Exact rejection probability of the chi2(1) Kupiec test under H0.

    No simulation needed: LR(x) depends only on the exception count x, and
    x ~ Binomial(n_obs, alpha) under a correctly calibrated model, so
    the true test size is the exact binomial probability mass on the
    rejection region {x : LR(x) > crit}.
    """
    from scipy.stats import binom, chi2

    p = alpha
    crit = chi2.ppf(1.0 - nominal, df=1)
    xs = np.arange(0, n_obs + 1)
    lrs = np.array([kupiec_pof(n_obs=n_obs, n_exceptions=int(x), alpha=alpha)[0]
                    for x in xs])
    return float(binom.pmf(xs[lrs > crit], n_obs, p).sum())


def test_kupiec_asymptotic_chi2_reference_is_oversized_at_regulatory_window():
    """The 250-day/99% Basel window is small-sample territory for the
    chi2(1) asymptotic reference: a nominally 5%-size test actually
    rejects a *correctly calibrated* model ~9.5% of the time here (the
    expected exception count is only 2.5). This is a property of the
    likelihood-ratio chi2 approximation for a rare (low tail-probability)
    binomial, not a bug in the LR formula -- documented in
    docs/VALIDATION.md F9 alongside the analogous eq_var finding. It
    attenuates quickly as the expected count grows: by n_obs=1000
    (expected count 10) the exact size is within a point of nominal.
    """
    size_250 = _exact_kupiec_test_size(250, 0.01)
    size_1000 = _exact_kupiec_test_size(1000, 0.01)
    assert size_250 == pytest.approx(0.0948, abs=0.005)
    assert size_1000 == pytest.approx(0.0551, abs=0.005)
    assert size_250 > 1.8 * 0.05
    assert size_1000 < 1.2 * 0.05


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
    lr_cc, p_cc = conditional_coverage(e, 0.10)
    lr_uc, _ = kupiec_pof(n_obs=10, n_exceptions=3, alpha=0.10)
    lr_ind, _ = christoffersen_independence(e)
    assert lr_cc == pytest.approx(lr_uc + lr_ind, abs=1e-12)
    assert p_cc == pytest.approx(chi2.sf(lr_cc, 2), abs=1e-12)


# ------------------------------------------------------------ Basel zones
def test_basel_boundaries_exact_4_5_and_9_10():
    """Regulatory table: green 0-4, yellow 5-9, red 10+ (250d, 99%)."""
    assert basel_traffic_light(n_obs=250, n_exceptions=4).zone == "green"
    assert basel_traffic_light(n_obs=250, n_exceptions=5).zone == "yellow"
    assert basel_traffic_light(n_obs=250, n_exceptions=9).zone == "yellow"
    assert basel_traffic_light(n_obs=250, n_exceptions=10).zone == "red"


@pytest.mark.parametrize("x,zone,mult", [
    (0, "green", 3.0), (1, "green", 3.0), (2, "green", 3.0),
    (3, "green", 3.0), (4, "green", 3.0),
    (5, "yellow", 3.40), (6, "yellow", 3.50), (7, "yellow", 3.65),
    (8, "yellow", 3.75), (9, "yellow", 3.85),
    (10, "red", 4.0), (12, "red", 4.0), (20, "red", 4.0),
])
def test_basel_multiplier_table(x, zone, mult):
    tl = basel_traffic_light(n_obs=250, n_exceptions=x)
    assert tl.zone == zone
    assert tl.multiplier == pytest.approx(mult)


def test_basel_cumulative_probability_values():
    """Cross-check the binomial cumulative probs behind the zone cuts."""
    assert basel_traffic_light(n_obs=250, n_exceptions=4).cumulative_prob == pytest.approx(0.8922, abs=1e-3)
    assert basel_traffic_light(n_obs=250, n_exceptions=5).cumulative_prob == pytest.approx(0.9588, abs=1e-3)
    assert basel_traffic_light(n_obs=250, n_exceptions=9).cumulative_prob < 0.9999
    assert basel_traffic_light(n_obs=250, n_exceptions=10).cumulative_prob >= 0.9999


def test_basel_invalid():
    with pytest.raises(ValueError):
        basel_traffic_light(n_obs=250, n_exceptions=-1)
    with pytest.raises(ValueError):
        basel_traffic_light(n_obs=250, n_exceptions=300)


# ------------------------------------------------------------ evaluate
def test_evaluate_var_backtest_counts():
    pnl = np.array([-5.0, 1.0, -0.5, -3.0, 2.0, -1.1])
    var = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 1.0])
    res = evaluate_var_backtest(pnl, var, 0.01)
    np.testing.assert_array_equal(res.exceedances, [1, 0, 0, 1, 0, 1])
    assert res.n_exceptions == 3
    assert res.exception_rate == pytest.approx(0.5)


def test_evaluate_var_backtest_validation():
    with pytest.raises(ValueError):
        evaluate_var_backtest([1.0, 2.0], [1.0], 0.01)
    with pytest.raises(ValueError):
        evaluate_var_backtest([1.0, np.nan], [1.0, 1.0], 0.01)


# ------------------------------------------------------------ ES backtest
def test_as_z2_correct_model_accepted():
    """Losses simulated from the model used for the forecasts: p large."""
    rng = np.random.default_rng(6)
    n, sigma, alpha = 1000, 1.0, 0.025
    pnl = rng.standard_normal(n) * sigma
    var = np.full(n, sigma * norm.isf(alpha))
    es = np.full(n, normal_es(sigma, alpha))
    r = acerbi_szekely_z2(pnl, var, es, alpha=alpha, seed=1)
    assert abs(r["z2"]) < 0.5
    assert r["p_value"] > 0.05


def test_as_z2_negative_when_es_understated():
    """Paper sign: true vol double the forecast gives Z2 << 0 and p tiny."""
    rng = np.random.default_rng(7)
    n, alpha = 1000, 0.025
    pnl = rng.standard_normal(n) * 2.0  # true sigma 2
    var = np.full(n, 1.0 * norm.isf(alpha))  # forecast sigma 1
    es = np.full(n, normal_es(1.0, alpha))
    r = acerbi_szekely_z2(pnl, var, es, alpha=alpha, seed=2)
    assert r["z2"] < -0.5
    assert r["p_value"] < 0.01
    assert r["n_exceptions"] == int(np.sum(pnl < -var))


def test_as_z2_hand_value():
    """Z2 = sum(pnl I / (T alpha ES)) + 1 on a three-day hand example."""
    pnl = np.array([-3.0, 1.0, -0.5])
    var = np.array([2.0, 2.0, 2.0])
    es = np.array([2.5, 2.5, 2.5])
    r = acerbi_szekely_z2(pnl, var, es, alpha=0.1, n_sims=10)
    assert r["z2"] == pytest.approx(-3.0 / (3 * 0.1 * 2.5) + 1.0, abs=1e-15)
    assert r["n_exceptions"] == 1


def test_as_z2_null_sampler_used():
    """A null sampler that always reproduces the observed path gives p = 1."""
    pnl = np.array([-3.0, 1.0, -0.5, 0.2])
    var = np.full(4, 2.0)
    es = np.full(4, 2.5)
    r = acerbi_szekely_z2(pnl, var, es, alpha=0.1, n_sims=50,
                          null_sampler=lambda rng, k: np.tile(pnl, (k, 1)))
    assert r["p_value"] == pytest.approx(1.0)


def test_deprecated_es_backtest_keeps_old_sign():
    rng = np.random.default_rng(7)
    n = 500
    pnl = rng.standard_normal(n) * 2.0
    var = np.full(n, norm.isf(0.025))
    es = np.full(n, normal_es(1.0, 0.025))
    with pytest.warns(DeprecationWarning):
        z_old, p_old = es_backtest_acerbi_szekely(pnl, var, es, 0.975, n_sim=200, seed=3)
    new = acerbi_szekely_z2(pnl, var, es, alpha=0.025, n_sims=200, seed=3)
    # 1 - 0.975 = 0.025000000000000022 in binary, so the two alphas differ
    # by 1 ulp of 0.025 and Z2 by a few ulps; rel 1e-12 bounds that.
    assert z_old == pytest.approx(-new["z2"], rel=1e-12)
    assert p_old == pytest.approx(new["p_value"])


def test_as_z2_validation():
    with pytest.raises(ValueError):
        acerbi_szekely_z2([1.0], [1.0, 2.0], [1.0], alpha=0.025)
    with pytest.raises(ValueError):
        acerbi_szekely_z2([1.0], [1.0], [-1.0], alpha=0.025)
    with pytest.raises(ValueError, match="tail probability"):
        acerbi_szekely_z2([1.0], [1.0], [1.0], alpha=0.975)


# ------------------------------------------------------------ rolling
def test_rolling_backtest_shapes_and_window():
    from fx_var import Book, Spot
    from fx_var.data.synthetic import demo_market, simulate_history

    market = demo_market()
    book = Book([Spot("EURUSD", 10e6)])
    rets = simulate_history(book, market, 400, seed=3)

    def var_fn(bk, mkt, window):
        from fx_var import historical_var

        return historical_var(bk, mkt, window, 0.01).var

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
        return parametric_var(bk, mkt, window, 0.01, min_obs=50).var

    def fn_fhs(bk, mkt, window):
        return historical_var(bk, mkt, window, 0.01, method="fhs", min_obs=50).var

    bt_p = rolling_backtest(book, market, rets, fn_param, window=250)
    bt_f = rolling_backtest(book, market, rets, fn_fhs, window=250)
    res_p = evaluate_var_backtest(bt_p["pnl"], bt_p["var"], 0.01)
    res_f = evaluate_var_backtest(bt_f["pnl"], bt_f["var"], 0.01)
    # parametric-normal: too many, *clustered* exceptions -> CC and
    # independence both rejected
    assert res_p.cc_p < 0.05
    assert res_p.independence_p < 0.05
    # FHS: conditional coverage acceptable
    assert res_f.cc_p > 0.05
    assert res_f.n_exceptions < res_p.n_exceptions


class TestKupiecExactSize:
    """TEST-5: exact finite-sample size of the 5 % Kupiec test at 250 / 99 %.

    Summing the Binomial(250, 0.01) pmf over the chi2(1) rejection set gives
    0.0948 -- the test is oversized (the README's "~9.5 %"), and ``x = 0``
    is itself a rejection.
    """

    N, ALPHA = 250, 0.01

    def _reject_set(self):
        crit = chi2.ppf(0.95, df=1)
        return [x for x in range(self.N + 1) if kupiec_pof(n_obs=self.N, n_exceptions=x, alpha=self.ALPHA)[0] > crit]

    def test_exact_size_is_9_5_percent(self):
        from scipy.stats import binom

        size = float(sum(binom.pmf(x, self.N, self.ALPHA) for x in self._reject_set()))
        assert size == pytest.approx(0.0948, abs=1e-4)
        assert size == pytest.approx(0.09476, abs=5e-5)

    def test_zero_exceptions_is_a_rejection(self):
        assert 0 in self._reject_set()
        assert kupiec_pof(n_obs=self.N, n_exceptions=0, alpha=self.ALPHA)[1] < 0.05

    def test_rejection_set_is_zero_or_upper_tail(self):
        rej = self._reject_set()
        assert rej[0] == 0
        assert rej[1:] == list(range(7, self.N + 1))
        assert kupiec_pof(n_obs=self.N, n_exceptions=6, alpha=self.ALPHA)[1] > 0.05
