"""Cross-method consistency and disagreement: the three engines must agree
on a plain-vanilla G10 book and disagree in the documented direction on an
EM fat-tailed book."""

import numpy as np
import pytest

from fx_var import (
    Book,
    Spot,
    historical_var,
    monte_carlo_var,
    parametric_var,
    sample_cov,
)
from fx_var.data.synthetic import demo_em_book, demo_market, simulate_fx_returns, simulate_history


@pytest.fixture(scope="module")
def market():
    return demo_market()


def test_methods_agree_on_gaussian_g10_book(market):
    """Normal, constant-vol world: HS, parametric and MC VaR within 15% of
    each other (they estimate the same quantile)."""
    book = Book([Spot("EURUSD", 10e6), Spot("GBPUSD", -4e6), Spot("USDJPY", 6e6)])
    rets = simulate_history(book, market, 2000, seed=3, garch=False)
    hs = historical_var(book, market, rets, 0.99).var
    pa = parametric_var(book, market, rets, 0.99).var
    mc = monte_carlo_var(book, market, sample_cov(rets), 0.99,
                         n_scenarios=100_000, seed=1).var
    assert hs == pytest.approx(pa, rel=0.15)
    assert mc == pytest.approx(pa, rel=0.15)


def test_em_book_method_disagreement(market):
    """EM fat tails: historical 99% VaR on t-shaped EM data exceeds the
    normal parametric figure - the documented method disagreement that
    motivates t/jump MC for EM books."""
    book = demo_em_book()
    # heavy-tailed EM history: a *common* t(3) mixing variable (variance-
    # matched) - EM tail events hit the whole block at once, so the fat
    # tail survives portfolio aggregation
    df, seed = 3.0, 7
    base = simulate_fx_returns(["MXN", "BRL", "TRY", "ZAR"], 2000, seed=seed)
    rng = np.random.default_rng(seed)
    w = rng.chisquare(df, len(base)) / df
    mix = np.sqrt((df - 2.0) / df) / np.sqrt(w)
    rets = base.mul(mix, axis=0)
    hs = historical_var(book, market, rets, 0.99).var
    pa = parametric_var(book, market, rets, 0.99).var
    assert hs > pa * 1.10  # normal parametric underestimates the 99% tail
    # at 95% the ordering reverses (t borrows mass from the shoulders)
    hs95 = historical_var(book, market, rets, 0.95).var
    pa95 = parametric_var(book, market, rets, 0.95).var
    assert hs95 < pa95


def test_es_var_ratio_reveals_tail_shape(market):
    """ES/VaR at 97.5% is a tail-shape diagnostic: higher for the EM
    t-world than for the Gaussian G10 world."""
    g10 = Book([Spot("EURUSD", 10e6)])
    rets_g = simulate_history(g10, market, 2000, seed=5)
    r_g = historical_var(g10, market, rets_g, 0.975)
    em = demo_em_book()
    rng = np.random.default_rng(9)
    base = simulate_fx_returns(["MXN", "BRL", "TRY", "ZAR"], 2000, seed=9)
    t_shocks = rng.standard_t(3, size=base.shape) / np.sqrt(3.0)
    rets_e = base * 0 + base.std(ddof=1).to_numpy() * t_shocks
    r_e = historical_var(em, market, rets_e, 0.975)
    assert r_e.es / r_e.var > r_g.es / r_g.var
