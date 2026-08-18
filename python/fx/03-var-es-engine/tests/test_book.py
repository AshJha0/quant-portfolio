"""Book, Market, triangulation and base-currency P&L tests."""

import numpy as np
import pandas as pd
import pytest

from fx_var import Book, Cash, Forward, Market, Option, Spot, split_pair


@pytest.fixture()
def market():
    return Market(
        spot_usd={"USD": 1.0, "EUR": 1.08, "JPY": 1.0 / 149.0, "GBP": 1.27,
                  "MXN": 1.0 / 18.5},
        rates={"USD": 0.053, "EUR": 0.039, "JPY": 0.001, "GBP": 0.052},
        vols={"EURUSD": 0.075},
    )


# ---------------------------------------------------------------- pairs
def test_split_pair():
    assert split_pair("EURUSD") == ("EUR", "USD")
    assert split_pair("usdjpy") == ("USD", "JPY")


@pytest.mark.parametrize("bad", ["EUR", "EURUSD1", "EUR/US", "", "EUREUR"])
def test_split_pair_invalid(bad):
    with pytest.raises(ValueError):
        split_pair(bad)


# ---------------------------------------------------------------- market
def test_market_usd_spot_must_be_one():
    with pytest.raises(ValueError, match="USD"):
        Market(spot_usd={"USD": 1.1, "EUR": 1.08})


def test_market_rejects_nonpositive_spot():
    with pytest.raises(ValueError):
        Market(spot_usd={"EUR": -1.08})


def test_market_cross_by_triangulation(market):
    # EURJPY = EURUSD / JPYUSD = 1.08 * 149
    assert market.cross("EURJPY") == pytest.approx(1.08 * 149.0, rel=1e-12)
    assert market.cross("EURUSD") == pytest.approx(1.08)


def test_market_cip_forward(market):
    t = 0.5
    f = market.forward("EURUSD", t)
    assert f == pytest.approx(1.08 * np.exp((0.053 - 0.039) * t), rel=1e-12)


def test_market_missing_data_raises(market):
    with pytest.raises(KeyError):
        market.spot("SEK")
    with pytest.raises(KeyError):
        market.rate("MXN")
    with pytest.raises(KeyError):
        market.vol("USDMXN")


# ---------------------------------------------------------------- triangulation
def test_triangulation_identity_eurjpy(market):
    """Direct EURJPY P&L == engine's USD-leg decomposition, any shocks."""
    n = 10_000_000.0
    book = Book([Spot("EURJPY", n)])
    rng = np.random.default_rng(42)
    for _ in range(25):
        de, dj = rng.normal(0, 0.02, 2)
        pnl_engine = book.pnl(market, {"FX:EUR": de, "FX:JPY": dj})
        # direct: P&L in JPY = N (X1 - X0), converted to USD at shocked JPYUSD
        s_e1 = 1.08 * np.exp(de)
        s_j1 = (1.0 / 149.0) * np.exp(dj)
        x0 = 1.08 * 149.0
        x1 = s_e1 / s_j1
        pnl_direct = n * (x1 - x0) * s_j1
        assert pnl_engine == pytest.approx(pnl_direct, abs=1e-6)


def test_triangulation_cross_equals_two_usd_positions(market):
    """EURJPY position == EURUSD position + USDJPY position (USD legs cancel)."""
    n = 10_000_000.0
    e0 = market.spot("EUR")  # EURUSD entry
    book_cross = Book([Spot("EURJPY", n)])
    book_legs = Book([Spot("EURUSD", n), Spot("USDJPY", n * e0)])
    rng = np.random.default_rng(7)
    shocks = pd.DataFrame({"FX:EUR": rng.normal(0, 0.02, 50),
                           "FX:JPY": rng.normal(0, 0.02, 50)})
    p1 = book_cross.pnl(market, shocks)
    p2 = book_legs.pnl(market, shocks)
    np.testing.assert_allclose(p1, p2, atol=1e-6)


# ---------------------------------------------------------------- P&L basics
def test_spot_zero_initial_value(market):
    book = Book([Spot("EURUSD", 1e6)])
    assert book.value_usd(market) == pytest.approx(0.0, abs=1e-9)


def test_spot_pnl_long_eur(market):
    book = Book([Spot("EURUSD", 1e6)])
    pnl = book.pnl(market, {"FX:EUR": 0.01})
    assert pnl == pytest.approx(1e6 * 1.08 * (np.exp(0.01) - 1.0), rel=1e-12)
    assert book.pnl(market, {"FX:EUR": -0.01}) < 0


def test_short_position_sign(market):
    book = Book([Spot("GBPUSD", -1e6)])
    assert book.pnl(market, {"FX:GBP": -0.02}) > 0


def test_base_ccy_cash_has_zero_risk(market):
    """A base-currency balance carries no FX risk in base-ccy P&L."""
    book = Book([Cash("EUR", 25_000_000)], base="EUR")
    pnl = book.pnl(market, {"FX:EUR": -0.10, "FX:JPY": 0.05, "FX:GBP": -0.05})
    assert pnl == pytest.approx(0.0, abs=1e-9)


def test_non_usd_base_translation(market):
    """USD cash in a EUR-base book gains when EUR falls."""
    book = Book([Cash("USD", 1_000_000)], base="EUR")
    assert book.pnl(market, {"FX:EUR": -0.05}) > 0
    assert book.pnl(market, {"FX:EUR": +0.05}) < 0


def test_base_change_consistency(market):
    """USD-base and EUR-base books agree on the USD value of the P&L when
    the P&L is converted at the shocked EUR rate."""
    positions = [Spot("USDJPY", 5e6), Spot("GBPUSD", 2e6)]
    shocks = {"FX:JPY": -0.03, "FX:GBP": 0.01, "FX:EUR": 0.02}
    pnl_usd = Book(positions, base="USD").pnl(market, shocks)
    pnl_eur = Book(positions, base="EUR").pnl(market, shocks)
    s_eur1 = 1.08 * np.exp(0.02)
    assert pnl_eur * s_eur1 == pytest.approx(pnl_usd, rel=1e-9)


# ---------------------------------------------------------------- factors
def test_factors_enumeration(market):
    book = Book([Spot("EURJPY", 1e6), Forward("GBPUSD", 1e6, 0.5),
                 Option("EURUSD", 1e6, 1.10, 0.25)])
    f = book.factors(market)
    assert set(f) == {"FX:EUR", "FX:JPY", "FX:GBP", "IR:GBP", "IR:USD",
                      "IR:EUR", "VOL:EURUSD"}
    assert f == sorted(f)  # FX:* < IR:* < VOL:* lexicographically


def test_empty_book(market):
    book = Book([])
    assert book.factors(market) == []
    assert book.pnl(market, {"FX:EUR": 0.05}) == 0.0
    assert book.value_usd(market) == 0.0


def test_single_ccy_book(market):
    book = Book([Spot("EURUSD", 1e6)])
    assert book.factors(market) == ["FX:EUR"]


def test_fx_usd_shock_rejected(market):
    book = Book([Spot("EURUSD", 1e6)])
    with pytest.raises(ValueError, match="FX:USD"):
        book.pnl(market, {"FX:USD": 0.01})


def test_linear_exposures_spot(market):
    """dPnL/dlog(EURUSD) for a long-EUR spot = notional x spot in USD."""
    book = Book([Spot("EURUSD", 1e6)])
    w = book.linear_exposures(market)
    assert w["FX:EUR"] == pytest.approx(1e6 * 1.08, rel=1e-6)


def test_linear_exposures_cross(market):
    """EURJPY exposure splits +N*S_eur on EUR and -N*X0*S_jpy on JPY."""
    n = 1e6
    book = Book([Spot("EURJPY", n)])
    w = book.linear_exposures(market)
    assert w["FX:EUR"] == pytest.approx(n * 1.08, rel=1e-6)
    assert w["FX:JPY"] == pytest.approx(-n * 1.08, rel=1e-6)  # X0*S_jpy = S_eur


def test_dataframe_and_series_shocks(market):
    book = Book([Spot("EURUSD", 1e6)])
    df = pd.DataFrame({"FX:EUR": [0.01, -0.01, 0.0]})
    out = book.pnl(market, df)
    assert out.shape == (3,)
    assert out[2] == pytest.approx(0.0, abs=1e-9)
    ser = pd.Series({"FX:EUR": 0.01})
    assert book.pnl(market, ser) == pytest.approx(out[0], rel=1e-12)


def test_position_validation():
    with pytest.raises(ValueError):
        Spot("EURUSD", 1e6, entry_rate=-1.0)
    with pytest.raises(ValueError):
        Forward("EURUSD", 1e6, expiry=-0.5)
    with pytest.raises(ValueError):
        Forward("EURUSD", 1e6, 0.5, strike=0.0)
    with pytest.raises(ValueError):
        Option("EURUSD", 1e6, strike=-1.1, expiry=0.25)
    with pytest.raises(ValueError):
        Option("EURUSD", 1e6, 1.1, 0.25, kind="straddle")


def test_negative_shocked_vol_rejected(market):
    book = Book([Option("EURUSD", 1e6, 1.10, 0.25)])
    with pytest.raises(ValueError, match="negative"):
        book.pnl(market, {"VOL:EURUSD": -0.10})
