"""W6d: the cash residual of a levered or under-invested book is financed."""

import numpy as np
import pandas as pd
import pytest
from eq_port.backtest import make_static_strategy, run_backtest
from qp_core.execution import ExecutionConfig
from qp_core.stats import Financing, FinancingWarning

# Filled at Thursday's close, so the levered book is held over Friday night.
SAME_CLOSE = ExecutionConfig("close_auction", auction_cost_bp=0.0)


def _flat(days):
    return pd.DataFrame(0.0, index=pd.DatetimeIndex(days), columns=["A", "B"])


def test_levered_book_pays_borrow_over_a_weekend_hand_fixture():
    # Thu (estimation), Fri (trade), Mon: 150% invested, 4% + 50 bp, ACT/360
    ret = _flat(["2026-10-29", "2026-10-30", "2026-11-02"])
    fin = Financing(rate=0.04, borrow_spread_bp=50.0, basis=360.0)
    res = run_backtest(ret, make_static_strategy(np.array([0.75, 0.75])), window=1,
                       financing=fin, execution=SAME_CLOSE)
    # Friday: trade day, accrual Thu->Fri 1 day, cash -0.5
    g_fri = -0.5 * 0.045 * 1 / 360
    assert res.financing.iloc[0] == pytest.approx(g_fri, rel=1e-12)
    # Monday: 3 days; Friday's financing loss drifts the weights to 1.5 / (1 + g_fri),
    # so the borrowed cash is slightly larger than 0.5 (-1.875e-4 before the drift)
    cash_mon = 1.0 - 1.5 / (1.0 + g_fri)
    fin_mon = cash_mon * 0.045 * 3 / 360
    assert fin_mon == pytest.approx(-1.875e-4, rel=1e-3)
    assert res.financing.iloc[1] == pytest.approx(fin_mon, rel=1e-12)
    assert res.risk_free.iloc[1] == pytest.approx(0.04 * 3 / 360)
    assert res.excess_returns.iloc[1] == pytest.approx(fin_mon - 0.04 * 3 / 360, rel=1e-12)


def test_under_invested_book_earns_on_cash():
    ret = _flat(["2026-10-29", "2026-10-30", "2026-11-02"])
    res = run_backtest(ret, make_static_strategy(np.array([0.25, 0.25])), window=1,
                       financing=Financing(rate=0.04, lend_spread_bp=10.0),
                       execution=SAME_CLOSE)
    g_fri = 0.5 * 0.039 / 360
    cash_mon = 1.0 - 0.5 / (1.0 + g_fri)
    assert res.net_returns.iloc[1] == pytest.approx(cash_mon * 0.039 * 3 / 360, rel=1e-12)


def test_no_financing_warns():
    ret = _flat(["2026-10-29", "2026-10-30", "2026-11-02"])
    with pytest.warns(FinancingWarning):
        res = run_backtest(ret, make_static_strategy(np.array([0.5, 0.5])), window=1)
    assert (res.financing == 0).all() and res.excess_returns.equals(res.net_returns.rename("excess"))
