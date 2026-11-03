"""W6: per-leg P&L in a stated numeraire (two-day hand example) and the
financing of posted capital (W6d)."""

import math

import pandas as pd
import pytest
from fx_pairs.backtest import run_backtest
from qp_core.execution import ExecutionConfig
from qp_core.stats import Financing, FinancingWarning

AUCTION = ExecutionConfig(mode="close_auction", auction_cost_bp=0.0)


def _two_days():
    idx = pd.DatetimeIndex(["2026-10-30", "2026-11-02"])  # Fri, Mon
    return idx, pd.Series([1.00, 1.01], index=idx), pd.Series([2.00, 1.98], index=idx)


def test_two_day_leg_pnl_hand_example():
    """Long 1 unit of pair 1, short 0.5 of pair 2, quote-currency units:
    leg1 = log(1.01) = 0.00995033; leg2 = -0.5 log(1.98/2.00) = +0.00502513."""
    _, p1, p2 = _two_days()
    res = run_backtest(p1, p2, [1.0, 1.0], 0.5, pip_spread_1=0.0, pip_spread_2=0.0,
                       execution=AUCTION, financing=Financing(rate=0.0))
    assert res.leg1_pnl.iloc[1] == pytest.approx(math.log(1.01), rel=1e-12)
    assert res.leg2_pnl.iloc[1] == pytest.approx(-0.5 * math.log(0.99), rel=1e-12)
    assert (res.leg1_pnl + res.leg2_pnl).iloc[1] == pytest.approx(res.total_pnl.iloc[1], rel=1e-12)


def test_capital_earns_financing_and_excess_excludes_it():
    _, p1, p2 = _two_days()
    res = run_backtest(p1, p2, [0.0, 0.0], 1.0, execution=AUCTION, notional=1e6,
                       financing=Financing(rate=0.05, basis=360.0))
    assert res.financing_pnl.iloc[1] == pytest.approx(1e6 * 0.05 * 3 / 360, rel=1e-12)
    assert res.total_pnl.iloc[1] == pytest.approx(1e6 * 0.05 * 3 / 360, rel=1e-12)
    assert res.excess_pnl.iloc[1] == 0.0


def test_no_financing_warns():
    _, p1, p2 = _two_days()
    with pytest.warns(FinancingWarning):
        res = run_backtest(p1, p2, [0.0, 0.0], 1.0)
    assert (res.financing_pnl == 0).all()
