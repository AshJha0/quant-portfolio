"""W6d: a long-short book's cash earns the financing rate; Sharpe uses excess."""

import pandas as pd
import pytest
from eq_algo import BacktestConfig, run_backtest
from qp_core.stats import Financing, FinancingWarning

CFG = BacktestConfig(n_quantiles=2, gross_exposure=2.0, max_weight=1.0,
                     linear_cost_bps=0.0, impact_coef=0.0)


def _flat():
    dates = pd.DatetimeIndex(["2026-10-30", "2026-11-02", "2026-11-03"])  # Fri, Mon, Tue
    prices = pd.DataFrame(100.0, index=dates, columns=list("ABCD"))
    signal = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 3, index=dates, columns=list("ABCD"))
    return prices, signal


def test_dollar_neutral_book_earns_rf_on_its_capital_hand_fixture():
    prices, signal = _flat()
    res = run_backtest(prices, signal, CFG, financing=Financing(rate=0.04, basis=360.0))
    led = res.ledger
    assert led["net_exposure"].abs().max() == pytest.approx(0.0)
    assert led["financing"].iloc[1] == pytest.approx(0.04 * 3 / 360, rel=1e-12)
    assert led["financing"].iloc[2] == pytest.approx(0.04 * 1 / 360, rel=1e-12)
    assert led["net_ret"].iloc[1] == pytest.approx(0.04 * 3 / 360, rel=1e-12)
    assert res.excess_returns.abs().max() == pytest.approx(0.0, abs=1e-18)


def test_no_financing_warns_and_excess_equals_net():
    prices, signal = _flat()
    with pytest.warns(FinancingWarning):
        res = run_backtest(prices, signal, CFG)
    assert (res.ledger["financing"] == 0).all()
    assert res.excess_returns.equals(res.ledger["net_ret"].rename("excess_ret"))
