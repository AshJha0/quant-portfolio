"""Garman-Kohlhagen internals and delta-vega(-gamma) mapping behaviour."""

import numpy as np
import pytest

from fx_var import Book, Market, Option
from fx_var.gk import gk_delta, gk_gamma, gk_price, gk_vega

S, K, T, RD, RF, SIG = 1.08, 1.10, 0.25, 0.053, 0.039, 0.075


@pytest.fixture()
def market():
    return Market(
        spot_usd={"EUR": S, "JPY": 1.0 / 149.0},
        rates={"USD": RD, "EUR": RF, "JPY": 0.001},
        vols={"EURUSD": SIG, "EURJPY": 0.095},
    )


def test_put_call_parity():
    c = gk_price(S, K, T, RD, RF, SIG, "call")
    p = gk_price(S, K, T, RD, RF, SIG, "put")
    parity = S * np.exp(-RF * T) - K * np.exp(-RD * T)
    assert c - p == pytest.approx(parity, abs=1e-10)


def test_delta_bounds_and_signs():
    dc = gk_delta(S, K, T, RD, RF, SIG, "call")
    dp = gk_delta(S, K, T, RD, RF, SIG, "put")
    assert 0.0 < dc < np.exp(-RF * T)
    assert -np.exp(-RF * T) < dp < 0.0
    assert dc - dp == pytest.approx(np.exp(-RF * T), abs=1e-10)


def test_greeks_vs_finite_differences():
    h = 1e-5
    fd_delta = (gk_price(S + h, K, T, RD, RF, SIG) - gk_price(S - h, K, T, RD, RF, SIG)) / (2 * h)
    assert gk_delta(S, K, T, RD, RF, SIG) == pytest.approx(fd_delta, abs=1e-7)
    fd_gamma = (gk_price(S + h, K, T, RD, RF, SIG) - 2 * gk_price(S, K, T, RD, RF, SIG)
                + gk_price(S - h, K, T, RD, RF, SIG)) / h**2
    assert gk_gamma(S, K, T, RD, RF, SIG) == pytest.approx(fd_gamma, rel=1e-4)
    hv = 1e-5
    fd_vega = (gk_price(S, K, T, RD, RF, SIG + hv) - gk_price(S, K, T, RD, RF, SIG - hv)) / (2 * hv)
    assert gk_vega(S, K, T, RD, RF, SIG) == pytest.approx(fd_vega, abs=1e-7)


def test_zero_vol_and_zero_expiry_intrinsic():
    # T = 0: pure intrinsic
    assert gk_price(1.20, K, 0.0, RD, RF, SIG, "call") == pytest.approx(0.10, abs=1e-12)
    assert gk_price(1.00, K, 0.0, RD, RF, SIG, "call") == pytest.approx(0.0, abs=1e-12)
    # vol = 0: discounted forward intrinsic
    v = gk_price(S, K, T, RD, RF, 0.0, "call")
    f = S * np.exp((RD - RF) * T)
    assert v == pytest.approx(np.exp(-RD * T) * max(f - K, 0.0), abs=1e-12)


def test_gk_input_validation():
    with pytest.raises(ValueError):
        gk_price(S, -1.0, T, RD, RF, SIG)
    with pytest.raises(ValueError):
        gk_price(S, K, -0.1, RD, RF, SIG)
    with pytest.raises(ValueError):
        gk_price(S, K, T, RD, RF, -0.2)
    with pytest.raises(ValueError):
        gk_price(S, K, T, RD, RF, SIG, kind="digital")


def test_vega_positive_gamma_positive():
    assert gk_vega(S, K, T, RD, RF, SIG) > 0
    assert gk_gamma(S, K, T, RD, RF, SIG) > 0


# ------------------------------------------------------- mapping vs full reval
def test_delta_vega_mapping_small_shock_accuracy(market):
    """For small FX/vol shocks the delta-vega mapping tracks full GK reval."""
    book = Book([Option("EURUSD", 30e6, K, T, "call")])
    shocks = {"FX:EUR": 0.003, "VOL:EURUSD": 0.002}
    full = book.pnl(market, shocks, option_method="full")
    dv = book.pnl(market, shocks, option_method="delta_vega")
    assert dv == pytest.approx(full, rel=0.05)


def test_gamma_term_improves_mapping(market):
    """For a pure FX shock, adding the gamma term must shrink the mapping
    error, and the delta-vega error must grow ~quadratically with the shock."""
    book = Book([Option("EURUSD", 30e6, K, T, "call")])
    errs_dv, errs_dvg = [], []
    for shock in (0.01, 0.02, 0.04):
        sh = {"FX:EUR": shock}
        full = book.pnl(market, sh, option_method="full")
        errs_dv.append(abs(book.pnl(market, sh, option_method="delta_vega") - full))
        errs_dvg.append(abs(book.pnl(market, sh, option_method="delta_vega_gamma") - full))
    for dv, dvg in zip(errs_dv, errs_dvg):
        assert dvg < dv
    # doubling the shock ~quadruples the delta-only error (convexity)
    assert errs_dv[1] / errs_dv[0] == pytest.approx(4.0, rel=0.35)
    assert errs_dv[2] / errs_dv[1] == pytest.approx(4.0, rel=0.35)


def test_mapping_large_shock_underestimates_long_option_loss(market):
    """Long-option books: delta-only P&L is always below full reval
    (convexity works for you); a VaR built on delta-vega therefore
    overstates long-gamma losses and understates short-gamma losses."""
    long_call = Book([Option("EURUSD", 30e6, K, T, "call")])
    for shock in (-0.06, 0.06):
        full = long_call.pnl(market, {"FX:EUR": shock}, option_method="full")
        dv = long_call.pnl(market, {"FX:EUR": shock}, option_method="delta_vega")
        assert full > dv  # long gamma: truth beats linearisation both ways


def test_invalid_option_method(market):
    book = Book([Option("EURUSD", 1e6, K, T)])
    with pytest.raises(ValueError, match="option_method"):
        book.pnl(market, {"FX:EUR": 0.01}, option_method="taylor9")


def test_cross_pair_option_triangulates(market):
    """A EURJPY option revalues off the triangulated cross and JPY leg."""
    book = Book([Option("EURJPY", 10e6, 165.0, 0.25, "call")])
    de, dj = 0.02, -0.01
    v1 = book.value_usd(market, {"FX:EUR": de, "FX:JPY": dj})
    s_e, s_j = S * np.exp(de), (1.0 / 149.0) * np.exp(dj)
    x = s_e / s_j
    price_jpy = gk_price(x, 165.0, 0.25, 0.001, RF, 0.095, "call")
    assert v1 == pytest.approx(10e6 * price_jpy * s_j, rel=1e-10)
