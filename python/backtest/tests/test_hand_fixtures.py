"""Hand fixtures: the oracle for both modes (W24 acceptance).

1. The one-asset worked example of ``qp_bt.core``, with the numbers written
   out by hand.
2. A 3-asset, 10-day book with known P&L, costs, borrow and financing,
   checked against ``oracle.py``: an exact rational re-computation in shares
   and cash that shares no code with the engine.
"""

from __future__ import annotations

from fractions import Fraction as F

import numpy as np
import pandas as pd
import pytest
from qp_bt import EventEngine, LinearCost, NextBarFill, WeightPath, run_fast
from qp_core.stats import Financing

from . import oracle


def test_one_asset_worked_example_by_hand():
    # p: 100 -> 100 (decide w = 1 at close 0) -> 102 (fill at 101 with f = 0.5)
    prices = pd.DataFrame({"A": [100.0, 102.0]})
    tgt = np.array([[1.0], [np.nan]])
    res = run_fast(prices.pct_change(), tgt, NextBarFill(0.5, 1), LinearCost(10.0))
    row = res.ledger.iloc[1]
    n_post = 1.0 / 1.001                       # N = 1 - 0.001 N
    b = 102.0 / 101.0 - 1.0
    assert row["cost"] == pytest.approx(1.0 - n_post, abs=1e-15)      # 0.000999001
    assert row["gross_ret"] == pytest.approx(n_post * b, abs=1e-15)   # 0.0098911
    assert row["net_ret"] == pytest.approx(n_post * (1 + b) - 1, abs=1e-15)
    assert row["net_ret"] == pytest.approx(0.0088921, abs=1e-7)
    assert row["turnover"] == pytest.approx(n_post, abs=1e-15)
    # closing weight is exactly the target: costs are paid out of the trade
    assert res.weights.iloc[1, 0] == pytest.approx(1.0, abs=1e-15)


PRICES = [
    [100, 50, 20], [101, 49, 20], [103, 50, 21], [102, 52, 21], [104, 51, 22],
    [103, 53, 22], [105, 52, 21], [106, 54, 23], [104, 55, 22], [107, 54, 24],
]
TARGETS = {
    0: [F(1, 2), F(3, 10), F(2, 5)],      # 120% gross long: borrows cash
    3: [F(3, 5), F(-1, 5), F(1, 5)],      # a short in B: pays borrow
    6: [F(0), F(1, 2), F(-1, 4)],
    8: [F(1, 4), F(1, 4), F(1, 4)],       # 75% invested: cash earns
}


def _fixture(phi, lag, cost_bp, extra_bp, rate, borrow_bp):
    t_n = len(PRICES)
    prices = pd.DataFrame(np.array(PRICES, dtype=float), columns=list("ABC"))
    tgt = np.full((t_n, 3), np.nan)
    for k, v in TARGETS.items():
        tgt[k] = [float(x) for x in v]
    fin = Financing(rate=rate, periods_per_year=252.0)
    fill = NextBarFill(phi, lag, extra_bp)
    kw = dict(fill=fill, cost=LinearCost(cost_bp), financing=fin,
              borrow_bps_annual=borrow_bp)
    fast = run_fast(prices.pct_change(), tgt, **kw)
    event = EventEngine(prices, **kw).run(WeightPath(tgt))
    exact = oracle.run([[F(x) for x in row] for row in PRICES], TARGETS, F(phi).limit_denominator(),
                       lag, cost_bp, extra_bp, F(rate).limit_denominator(10**9), F(1, 252),
                       F(borrow_bp).limit_denominator() / 10_000 / 252)
    return fast, event, exact


CASES = [
    (0.5, 1, 5.0, 0.0, 0.04, 50.0),
    (1.0, 1, 10.0, 0.0, 0.05, 25.0),
    (0.0, 1, 2.0, 10.0, 0.03, 0.0),   # close auction booked next day
    (1.0, 0, 2.0, 10.0, 0.03, 40.0),  # close auction booked on the decision day
    (0.25, 2, 7.0, 0.0, 0.0, 100.0),  # two-bar signal-to-fill lag
]


@pytest.mark.parametrize("case", CASES)
def test_three_asset_ten_day_fixture_matches_exact_oracle(case):
    fast, event, exact = _fixture(*case)
    want = {
        "net_ret": [float(e["net"]) for e in exact],
        "cost": [float(e["cost"]) for e in exact],
        "borrow": [float(e["borrow"]) for e in exact],
        "financing": [float(e["fin"]) for e in exact],
        "gross_ret": [float(e["gross"]) for e in exact],
        "turnover": [float(e["turnover"]) for e in exact],
        "nav": [float(e["nav"]) for e in exact],
    }
    for mode in (fast, event):
        for col, vals in want.items():
            np.testing.assert_allclose(mode.ledger[col].to_numpy(), vals, rtol=0, atol=1e-13,
                                       err_msg=col)
    np.testing.assert_allclose(event.positions.to_numpy(),
                               [[float(x) for x in e["q"]] for e in exact], rtol=1e-13, atol=0)


def test_fixture_exercises_every_ledger_component():
    fast, _, _ = _fixture(*CASES[0])
    led = fast.ledger
    assert (led["cost"] > 0).sum() == 4           # four rebalances
    assert (led["borrow"] > 0).any()               # the short in B
    assert (led["financing"] < 0).any()            # 120% gross borrows
    assert (led["financing"] > 0).any()            # 75% invested earns
    ident = led["gross_ret"] - led["cost"] - led["borrow"] + led["financing"]
    np.testing.assert_allclose(ident, led["net_ret"], atol=1e-16)
