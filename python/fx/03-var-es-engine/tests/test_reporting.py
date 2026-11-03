"""W14b/W14d: FX VaR in a declared reporting currency; multi-currency book risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fx_var.book import Book, Cash, Forward, Market, Option, Spot
from fx_var.engine import empirical_var_es
from fx_var.reporting import (
    convert_report,
    fx_book_risk,
    reporting_pnl,
    tenor_weights,
    triangulation_residuals,
    var_report,
)

MKT = Market(
    spot_usd={"EUR": 1.10, "GBP": 1.27, "JPY": 1 / 150.0},
    rates={"USD": 0.05, "EUR": 0.03, "GBP": 0.045, "JPY": 0.001},
    vols={"EURUSD": 0.08, "GBPJPY": 0.11},
)


def _three_ccy_book() -> Book:
    return Book([
        Cash("EUR", 2e6),
        Spot("GBPUSD", -1.5e6),
        Spot("EURJPY", 1e6),  # a cross: long EUR, short JPY via USD
        Forward("GBPJPY", 5e5, 0.75),
        Option("EURUSD", 3e6, 1.12, 0.4, "call"),
        Option("GBPJPY", -1e6, 190.0, 1.5, "put"),
    ])


def _scenarios(n: int = 2000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = ["FX:EUR", "FX:GBP", "FX:JPY", "IR:USD", "IR:EUR", "IR:GBP", "IR:JPY",
            "VOL:EURUSD", "VOL:GBPJPY"]
    scale = np.array([0.006, 0.007, 0.008, 5e-4, 5e-4, 5e-4, 2e-4, 0.004, 0.006])
    return pd.DataFrame(rng.standard_t(5, size=(n, len(cols))) * scale, columns=cols)


def test_var_same_in_usd_and_eur_after_conversion():
    # Plan acceptance: three-currency book, VaR in USD and EUR agree after
    # conversion. "spot": one rate converts the whole report exactly.
    book, sc = _three_ccy_book(), _scenarios()
    usd = var_report(book, MKT, sc, reporting_ccy="USD", alpha=0.01, conversion="spot")
    eur = var_report(book, MKT, sc, reporting_ccy="EUR", alpha=0.01, conversion="spot")
    conv = convert_report(usd, MKT, "EUR")
    assert conv.var == pytest.approx(eur.var, rel=1e-12)
    assert conv.es == pytest.approx(eur.es, rel=1e-12)
    assert eur.reporting_ccy == "EUR" and eur.conversion == "spot"


def test_scenario_conversion_equals_per_scenario_hand_conversion():
    # "scenario": convert each scenario at its own EURUSD, then the quantile.
    book, sc = _three_ccy_book(), _scenarios()
    v0 = book.value_usd(MKT)
    v1 = book.value_usd(MKT, {c: sc[c].to_numpy() for c in sc.columns})
    x1 = MKT.spot("EUR") * np.exp(sc["FX:EUR"].to_numpy())
    by_hand = v1 / x1 - v0 / MKT.spot("EUR")
    var, es = empirical_var_es(by_hand, 0.01)
    rep = var_report(book, MKT, sc, reporting_ccy="EUR", alpha=0.01)
    assert rep.var == pytest.approx(var, rel=1e-12)
    assert rep.es == pytest.approx(es, rel=1e-12)
    # It agrees with Book(base="EUR").pnl, the package's existing convention.
    eur_book = Book(book.positions, base="EUR")
    np.testing.assert_allclose(eur_book.pnl(MKT, sc), by_hand, rtol=1e-12, atol=1e-6)
    with pytest.raises(ValueError, match="not a linear"):
        convert_report(rep, MKT, "USD")


def test_worked_example_reporting_ccy_cash():
    book = Book([Cash("EUR", 1e6)])
    sc = pd.DataFrame({"FX:EUR": np.linspace(-0.03, 0.03, 501)})
    usd = reporting_pnl(book, MKT, sc, reporting_ccy="USD")
    eur = reporting_pnl(book, MKT, sc, reporting_ccy="EUR")
    eur_spot = reporting_pnl(book, MKT, sc, reporting_ccy="EUR", conversion="spot")
    np.testing.assert_allclose(eur, 0.0, atol=1e-6)
    np.testing.assert_allclose(eur_spot, np.asarray(usd) / 1.10, rtol=1e-12)
    risk = fx_book_risk(book, MKT, reporting_ccy="USD")
    assert risk["fx"].loc["EUR", "delta"] == pytest.approx(1.1e6, rel=1e-6)


def test_reporting_ccy_is_required_and_validated():
    with pytest.raises(TypeError):
        var_report(_three_ccy_book(), MKT, _scenarios(10))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="3-letter"):
        reporting_pnl(_three_ccy_book(), MKT, {}, reporting_ccy="EURO")
    with pytest.raises(ValueError, match="conversion"):
        reporting_pnl(_three_ccy_book(), MKT, {}, reporting_ccy="EUR", conversion="avg")
    with pytest.raises(ValueError, match="non-empty"):
        var_report(_three_ccy_book(), MKT, pd.DataFrame(), reporting_ccy="EUR")


def test_triangulation_residuals():
    quoted = {"EURJPY": 1.10 * 150 * (1 + 2e-4), "EURGBP": 1.10 / 1.27}
    res = triangulation_residuals(MKT, quoted)
    assert res["EURJPY"] == pytest.approx(2e-4, rel=1e-9)
    assert abs(res["EURGBP"]) < 1e-15


def test_tenor_weights_partition_of_unity():
    b = (0.25, 0.5, 1.0, 2.0)
    for t in (0.1, 0.25, 0.4, 0.75, 1.5, 3.0):
        w = tenor_weights(t, b)
        assert w.sum() == pytest.approx(1.0) and np.all(w >= 0)
        assert np.dot(w, b) == pytest.approx(min(max(t, 0.25), 2.0))
    with pytest.raises(ValueError):
        tenor_weights(1.0, (1.0, 0.5))


@pytest.mark.parametrize("rep", ["USD", "EUR", "JPY"])
def test_bucketed_risk_sums_to_whole_book_sensitivity(rep):
    book = _three_ccy_book()
    risk = fx_book_risk(book, MKT, reporting_ccy=rep)
    for c in ("EUR", "GBP", "JPY", "USD"):
        h = 1e-5
        whole = (reporting_pnl(book, MKT, {f"IR:{c}": h}, reporting_ccy=rep)
                 - reporting_pnl(book, MKT, {f"IR:{c}": -h}, reporting_ccy=rep)) / (2 * h) * 1e-4
        got = risk["rates"].loc[c].sum() if c in risk["rates"].index else 0.0
        assert got == pytest.approx(whole, rel=1e-6, abs=1e-6)
    for pair in ("EURUSD", "GBPJPY"):
        h = 1e-5
        whole = (reporting_pnl(book, MKT, {f"VOL:{pair}": h}, reporting_ccy=rep)
                 - reporting_pnl(book, MKT, {f"VOL:{pair}": -h}, reporting_ccy=rep)) / (2 * h) * 1e-2
        assert risk["vega"].loc[pair].sum() == pytest.approx(whole, rel=1e-6)
    # Delta-gamma by currency explains a small joint FX move.
    move = {"FX:EUR": 0.004, "FX:GBP": -0.003, "FX:JPY": 0.002}
    fx = risk["fx"]
    approx = sum(fx.loc[c, "delta"] * m + 0.5 * fx.loc[c, "gamma"] * m * m
                 for c, m in ((k[3:], v) for k, v in move.items()))
    full = reporting_pnl(book, MKT, move, reporting_ccy=rep)
    assert approx == pytest.approx(full, rel=2e-2)


def test_eur_reporting_moves_risk_onto_the_other_currencies():
    # In EUR, EUR cash is riskless; a GBP balance worth 1.27m USD is long GBP
    # and short EUR against USD, each by its EUR value.
    book = Book([Cash("EUR", 1e6), Cash("GBP", 1e6)])
    usd = fx_book_risk(book, MKT, reporting_ccy="USD")["fx"]
    eur = fx_book_risk(book, MKT, reporting_ccy="EUR")["fx"]
    assert usd.loc["EUR", "delta"] == pytest.approx(1.1e6, rel=1e-6)
    assert eur.loc["GBP", "delta"] == pytest.approx(1e6 * 1.27 / 1.10, rel=1e-6)
    assert eur.loc["EUR", "delta"] == pytest.approx(-1e6 * 1.27 / 1.10, rel=1e-6)
