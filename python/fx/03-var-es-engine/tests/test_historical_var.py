"""Historical simulation: plain HS, BRW age weights, FHS, peg blindness."""

import numpy as np
import pandas as pd
import pytest

from fx_var import (
    Book,
    Cash,
    PegBlindnessWarning,
    Spot,
    empirical_var,
    ewma_volatility,
    historical_var,
)
from fx_var.data.synthetic import demo_market, simulate_fx_returns


@pytest.fixture()
def market():
    return demo_market()


@pytest.fixture()
def eur_book():
    return Book([Spot("EURUSD", 10_000_000)])


def test_hs_quantile_exact_on_known_returns(market, eur_book):
    """HS VaR equals the hand-computed order-statistic of scenario P&L."""
    rng = np.random.default_rng(11)
    r = rng.normal(0, 0.006, 500)
    rets = pd.DataFrame({"FX:EUR": r})
    res = historical_var(eur_book, market, rets, alpha=0.01)
    pnl = 10e6 * 1.08 * (np.exp(r) - 1.0)
    losses = np.sort(-pnl)[::-1]
    assert res.var == pytest.approx(losses[4], rel=1e-12)  # 5th worst of 500
    assert res.es == pytest.approx(losses[:5].mean(), rel=1e-12)
    assert res.es >= res.var


def test_hs_full_reval_not_linear(market, eur_book):
    """HS revalues exp(r)-1, not r: exact equality with the direct formula."""
    rets = pd.DataFrame({"FX:EUR": np.linspace(-0.05, 0.05, 100)})
    res = historical_var(eur_book, market, rets, alpha=0.05)
    direct = empirical_var(10e6 * 1.08 * (np.exp(rets["FX:EUR"].to_numpy()) - 1), 0.05)
    assert res.var == pytest.approx(direct, rel=1e-12)


def test_age_weighted_reacts_faster(market, eur_book):
    """BRW: a fresh cluster of large losses lifts VaR above plain HS."""
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.003, 480)
    crisis = rng.normal(-0.01, 0.02, 20)  # most recent 20 days
    rets = pd.DataFrame({"FX:EUR": np.concatenate([calm, crisis])})
    plain = historical_var(eur_book, market, rets, 0.01, method="plain")
    aged = historical_var(eur_book, market, rets, 0.01, method="age", decay=0.97)
    assert aged.var > plain.var


def test_age_weights_normalised_and_monotone(market, eur_book):
    rets = pd.DataFrame({"FX:EUR": np.random.default_rng(0).normal(0, 0.005, 100)})
    res = historical_var(eur_book, market, rets, 0.05, method="age", decay=0.98)
    w = res.weights
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(w) > 0)  # older -> smaller weight, last is newest
    assert w[-1] / w[0] == pytest.approx((1 / 0.98) ** 99, rel=1e-9)


def test_ewma_volatility_recursion_hand_checked():
    """sigma^2_{t+1} = lam sigma^2_t + (1-lam) r_t^2, seeded at mean(r^2)
    over the first min(30, n) observations (zero-mean RiskMetrics)."""
    r = pd.DataFrame({"x": [0.01, -0.02, 0.005]})
    lam = 0.9
    sig, sig_next = ewma_volatility(r, lam)
    s0 = (0.01**2 + 0.02**2 + 0.005**2) / 3
    s1 = lam * s0 + 0.1 * 0.01**2
    s2 = lam * s1 + 0.1 * 0.02**2
    s3 = lam * s2 + 0.1 * 0.005**2
    assert sig["x"].iloc[0] == pytest.approx(np.sqrt(s0), rel=1e-12)
    assert sig["x"].iloc[1] == pytest.approx(np.sqrt(s1), rel=1e-12)
    assert sig["x"].iloc[2] == pytest.approx(np.sqrt(s2), rel=1e-12)
    assert sig_next["x"] == pytest.approx(np.sqrt(s3), rel=1e-12)


def test_fhs_tracks_conditional_volatility(market, eur_book):
    """FHS VaR responds to the *current* GARCH vol state; plain HS barely
    moves.  Ending the window in a high-vol vs low-vol state must change
    FHS VaR by more than it changes plain HS VaR."""
    rets, state = simulate_fx_returns(["EUR"], 1500, seed=14, garch=True,
                                      return_state=True)
    sig = state["FX:EUR"].to_numpy()
    lo_end = int(np.argmin(sig[500:1000])) + 500
    hi_end = int(np.argmax(sig[1000:])) + 1000
    win = 400
    out = {}
    for name, end in (("lo", lo_end), ("hi", hi_end)):
        window = rets.iloc[end - win:end]
        out[("plain", name)] = historical_var(eur_book, market, window, 0.01).var
        out[("fhs", name)] = historical_var(eur_book, market, window, 0.01,
                                            method="fhs").var
    fhs_ratio = out[("fhs", "hi")] / out[("fhs", "lo")]
    hs_ratio = out[("plain", "hi")] / out[("plain", "lo")]
    assert fhs_ratio > hs_ratio
    assert fhs_ratio > 1.5  # FHS clearly scales up in the high-vol state


def test_horizon_sqrt_time_scaling(market, eur_book):
    rets = pd.DataFrame({"FX:EUR": np.random.default_rng(1).normal(0, 0.006, 300)})
    v1 = historical_var(eur_book, market, rets, 0.01, horizon_days=1)
    v10 = historical_var(eur_book, market, rets, 0.01, horizon_days=10)
    assert v10.var == pytest.approx(np.sqrt(10) * v1.var, rel=1e-12)
    assert v10.es == pytest.approx(np.sqrt(10) * v1.es, rel=1e-12)


# ------------------------------------------------------------ peg blindness
def test_peg_factor_triggers_warning(market):
    """Near-zero-vol HKD factor: the engine flags peg blindness."""
    book = Book([Spot("USDHKD", -50_000_000)])
    rets = pd.DataFrame({"FX:HKD": np.random.default_rng(3).normal(0, 1.5e-4, 500)})
    with pytest.warns(PegBlindnessWarning, match="FX:HKD"):
        res = historical_var(book, market, rets, 0.01)
    assert res.flagged_peg_factors == ("FX:HKD",)


def test_peg_warning_suppressible(market):
    import warnings

    book = Book([Spot("USDHKD", -50_000_000)])
    rets = pd.DataFrame({"FX:HKD": np.random.default_rng(3).normal(0, 1.5e-4, 500)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = historical_var(book, market, rets, 0.01, warn_pegs=False)
    assert res.flagged_peg_factors == ()


def test_free_float_not_flagged(market, eur_book):
    import warnings

    rets = pd.DataFrame({"FX:EUR": np.random.default_rng(4).normal(0, 0.006, 300)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", PegBlindnessWarning)
        res = historical_var(eur_book, market, rets, 0.01)
    assert res.flagged_peg_factors == ()


# ------------------------------------------------------------ edges / policy
def test_insufficient_history_raises(market, eur_book):
    rets = pd.DataFrame({"FX:EUR": np.zeros(30)})
    with pytest.raises(ValueError, match="insufficient history"):
        historical_var(eur_book, market, rets, 0.01)


def test_nan_policy_raises(market, eur_book):
    r = np.random.default_rng(0).normal(0, 0.005, 300)
    r[100] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        historical_var(eur_book, market, pd.DataFrame({"FX:EUR": r}), 0.01)


def test_missing_factor_column_raises(market):
    book = Book([Spot("EURUSD", 1e6), Spot("USDJPY", 1e6)])
    rets = pd.DataFrame({"FX:EUR": np.random.default_rng(0).normal(0, 0.005, 300)})
    with pytest.raises(ValueError, match="FX:JPY"):
        historical_var(book, market, rets, 0.01)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.7])
def test_alpha_edges_raise(market, eur_book, alpha):
    rets = pd.DataFrame({"FX:EUR": np.random.default_rng(0).normal(0, 0.005, 300)})
    with pytest.raises(ValueError, match="alpha"):
        historical_var(eur_book, market, rets, alpha)


def test_bad_method_and_decay_raise(market, eur_book):
    rets = pd.DataFrame({"FX:EUR": np.random.default_rng(0).normal(0, 0.005, 300)})
    with pytest.raises(ValueError, match="method"):
        historical_var(eur_book, market, rets, 0.01, method="bootstrap")
    with pytest.raises(ValueError, match="decay"):
        historical_var(eur_book, market, rets, 0.01, method="age", decay=1.5)


def test_empty_book_raises(market):
    rets = pd.DataFrame({"FX:EUR": np.random.default_rng(0).normal(0, 0.005, 300)})
    with pytest.raises(ValueError, match="book has no positions"):
        historical_var(Book([]), market, rets, 0.01)


def test_base_ccy_cash_book_raises(market):
    """Book holding only base-ccy cash has no risk factors: refuse, not 0."""
    book = Book([Cash("USD", 100e6)], base="USD")
    rets = pd.DataFrame({"FX:EUR": np.random.default_rng(0).normal(0, 0.005, 300)})
    with pytest.raises(ValueError, match="no risk factors"):
        historical_var(book, market, rets, 0.01)


def test_ewma_seed_uses_only_the_seed_window():
    """No look-ahead: data after the first 30 rows cannot move the seed."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.01, 100)
    b = a.copy()
    b[60:] *= 5.0  # a later volatility regime
    sa, _ = ewma_volatility(pd.DataFrame({"x": a}))
    sb, _ = ewma_volatility(pd.DataFrame({"x": b}))
    assert sa["x"].iloc[0] == sb["x"].iloc[0] == pytest.approx(np.sqrt(np.mean(a[:30] ** 2)))
    np.testing.assert_array_equal(sa["x"].iloc[:61], sb["x"].iloc[:61])
