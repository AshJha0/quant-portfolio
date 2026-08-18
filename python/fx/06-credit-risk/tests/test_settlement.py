"""FX settlement (Herstatt) risk: windows, gross vs PvP, netting — hand-checked."""

import numpy as np
import pytest

from fx_credit.data.synthetic import generate_fx_trade_book
from fx_credit.settlement import (
    PAYMENT_SYSTEM_HOURS_UTC,
    FXTrade,
    at_risk_window_hours,
    book_settlement_report,
    gross_settlement_exposure,
    net_settlement_exposure,
    settlement_exposure,
    time_zone_gap_matrix,
)

USD_RATES = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 1.0 / 148.0}


def test_window_pay_jpy_receive_usd_hand_checked():
    """Pay JPY at Tokyo open 00:00 UTC; USD final at Fedwire close 23:30 UTC.

    At-risk window = 23.5 hours — the classic Herstatt direction.
    """
    assert at_risk_window_hours("JPY", "USD") == pytest.approx(23.5)


def test_window_pay_usd_receive_jpy_is_zero():
    """JPY finality (08:00 UTC) precedes USD payment (13:30 UTC): no exposure."""
    assert at_risk_window_hours("USD", "JPY") == 0.0


def test_window_herstatt_1974_direction():
    """Pay EUR (06:00 UTC) vs receive USD (23:30 UTC): 17.5h — DEM/USD in 1974."""
    assert at_risk_window_hours("EUR", "USD") == pytest.approx(17.5)


def test_window_same_currency_zero():
    assert at_risk_window_hours("USD", "USD") == 0.0


def test_window_unknown_currency_raises():
    with pytest.raises(ValueError, match="payment-system hours"):
        at_risk_window_hours("XAU", "USD")


def test_gap_matrix_asymmetry_and_diagonal():
    m = time_zone_gap_matrix()
    assert np.all(np.diag(m.to_numpy()) == 0.0)
    assert m.loc["JPY", "USD"] == pytest.approx(23.5)
    assert m.loc["USD", "JPY"] == 0.0
    assert m.loc["JPY", "USD"] != m.loc["USD", "JPY"]  # time-zone asymmetry


def test_exposure_full_principal_hand_checked():
    """Buy USD 50m vs JPY at 148: pay JPY 7.4bn at 00:00, receive USD 50m.

    Exposure = FULL bought principal = USD 50,000,000 for 23.5 hours.
    """
    t = FXTrade("T1", "TokyoBank", "USDJPY", 50e6, 148.0, we_buy_base=True)
    e = settlement_exposure(t, USD_RATES)
    assert e.sold_ccy == "JPY" and e.bought_ccy == "USD"
    assert e.at_risk_hours == pytest.approx(23.5)
    assert e.exposure_usd == pytest.approx(50e6)
    assert e.pay_time_utc == 0.0 and e.receive_final_utc == 23.5


def test_pvp_cls_zeroes_exposure():
    t = FXTrade("T1", "TokyoBank", "USDJPY", 50e6, 148.0, we_buy_base=True, cls_settled=True)
    assert settlement_exposure(t, USD_RATES).exposure_usd == 0.0


def test_exposure_converted_to_usd():
    """Buy EUR 40m vs USD: bought principal 40m EUR = USD 43.2m at 1.08."""
    t = FXTrade("T2", "EuroBank", "EURUSD", 40e6, 1.08, we_buy_base=True)
    e = settlement_exposure(t, USD_RATES)
    assert e.bought_ccy == "EUR"
    assert e.exposure_usd == pytest.approx(40e6 * 1.08)


def test_receive_before_pay_no_exposure_trade():
    """Sell USD, buy JPY: we receive JPY before paying USD — zero exposure."""
    t = FXTrade("T", "X", "USDJPY", 10e6, 148.0, we_buy_base=False)
    e = settlement_exposure(t, USD_RATES)
    assert e.sold_ccy == "USD" and e.bought_ccy == "JPY"
    assert e.exposure_usd == 0.0 and e.at_risk_hours == 0.0


def test_zero_notional_trade_zero_exposure():
    t = FXTrade("T", "X", "USDJPY", 0.0, 148.0, we_buy_base=True)
    assert settlement_exposure(t, USD_RATES).exposure_usd == 0.0


def test_gross_exposure_is_sum_of_trades():
    book = generate_fx_trade_book()
    total = gross_settlement_exposure(book, USD_RATES)
    parts = sum(settlement_exposure(t, USD_RATES).exposure_usd for t in book)
    assert total == pytest.approx(parts)
    assert total > 0.0


def test_offsetting_trades_net_to_zero():
    a = FXTrade("A", "CP", "EURUSD", 25e6, 1.09, we_buy_base=True)
    b = FXTrade("B", "CP", "EURUSD", 25e6, 1.09, we_buy_base=False)
    assert gross_settlement_exposure([a, b], USD_RATES) > 0.0
    assert net_settlement_exposure([a, b], USD_RATES) == 0.0


def test_netting_never_exceeds_gross_on_book():
    book = generate_fx_trade_book()
    assert net_settlement_exposure(book, USD_RATES) <= gross_settlement_exposure(book, USD_RATES) + 1e-9


def test_netting_reduces_for_partial_offset():
    a = FXTrade("A", "CP", "EURUSD", 40e6, 1.08, we_buy_base=True)
    b = FXTrade("B", "CP", "EURUSD", 25e6, 1.08, we_buy_base=False)
    gross = gross_settlement_exposure([a, b], USD_RATES)
    net = net_settlement_exposure([a, b], USD_RATES)
    assert net < gross
    # net EUR receivable = 15m EUR, still at risk vs USD payable
    assert net == pytest.approx(15e6 * 1.08)


def test_netting_across_counterparties_not_allowed():
    a = FXTrade("A", "CP1", "EURUSD", 25e6, 1.09, we_buy_base=True)
    b = FXTrade("B", "CP2", "EURUSD", 25e6, 1.09, we_buy_base=False)
    assert net_settlement_exposure([a, b], USD_RATES) > 0.0


def test_single_currency_book_no_settlement_risk():
    """Same currency both legs (e.g. an internal USD/USD cash move): no FX settlement risk."""
    t = FXTrade("T", "X", "USDUSD", 10e6, 1.0, we_buy_base=True)
    assert settlement_exposure(t, USD_RATES).exposure_usd == 0.0
    assert gross_settlement_exposure([t], USD_RATES) == 0.0


def test_book_report_cls_counterfactual():
    book = generate_fx_trade_book()
    rep = book_settlement_report(book, USD_RATES)
    assert len(rep) == 6
    cls_rows = rep[rep["cls"]]
    assert (cls_rows["exposure_usd"] == 0.0).all()
    # at least one CLS trade would carry exposure if settled gross
    assert (cls_rows["exposure_if_gross_usd"] > 0.0).any()
    non_cls = rep[~rep["cls"]]
    assert (non_cls["exposure_usd"] == non_cls["exposure_if_gross_usd"]).all()


def test_missing_usd_rate_raises():
    t = FXTrade("T", "X", "USDJPY", 10e6, 148.0, we_buy_base=True)
    with pytest.raises(ValueError, match="USD rate"):
        settlement_exposure(t, {"JPY": 1 / 148.0})


def test_invalid_trade_parameters_raise():
    with pytest.raises(ValueError, match="6 letters"):
        FXTrade("T", "X", "EUR/USD", 1e6, 1.1, True)
    with pytest.raises(ValueError, match="notional"):
        FXTrade("T", "X", "EURUSD", -1e6, 1.1, True)
    with pytest.raises(ValueError, match="rate"):
        FXTrade("T", "X", "EURUSD", 1e6, 0.0, True)
