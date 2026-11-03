"""Tests for the walk-forward backtester, pip costs and base-ccy conversion."""

import numpy as np
import pandas as pd
import pytest
from fx_port import (
    base_conversion_returns,
    convert_base,
    pips_to_bps,
    run_backtest,
    sharpe_ratio,
    total_log_returns,
)
from fx_port.data import make_panel
from qp_core.execution import ExecutionConfig

DT = 1.0 / 252


def _tiny_market():
    idx = pd.bdate_range("2021-01-04", periods=8)
    spot = pd.DataFrame(
        {"AUD": [0.01, -0.02, 0.005, 0.01, -0.005, 0.002, 0.003, -0.001],
         "JPY": [-0.002, 0.001, 0.0, 0.004, -0.002, 0.001, 0.0, 0.002]},
        index=idx,
    )
    carry = pd.DataFrame(
        {"AUD": 0.03 * DT, "JPY": -0.02 * DT}, index=idx
    )
    return spot, carry


AUCTION = ExecutionConfig("close_auction", auction_cost_bp=0.0)


def test_ledger_hand_computed_with_carry():
    """Close-auction fill (the old fill price), 10 bp, one allocation.

    Decided at close 2 and filled there, booked in row 3. Costs come out of
    the trade: N = 1 / (1 + 10e-4 * 1.0). The book held overnight into day 3
    is the new one, so day 3 earns N (w . R) and N (w . d), with
    R = exp(spot) - 1 (simple returns).
    """
    spot, carry = _tiny_market()
    w = pd.Series({"AUD": 0.5, "JPY": -0.5})
    res = run_backtest(spot, carry, lambda hist: w, est_window=3, rebalance_every=100,
                       cost_bps=10.0, execution=AUCTION)
    led = res.ledger
    assert led.index[0] == spot.index[3]
    wv = w.to_numpy()
    n_post = 1.0 / (1.0 + 10e-4)
    r3 = np.expm1(spot.iloc[3].to_numpy())
    exp_spot = n_post * float(wv @ r3)
    exp_carry = n_post * float(wv @ carry.iloc[3].to_numpy())
    exp_cost = 1.0 - n_post
    assert led.iloc[0]["spot_pnl"] == pytest.approx(exp_spot, abs=1e-15)
    assert led.iloc[0]["carry_pnl"] == pytest.approx(exp_carry, abs=1e-15)
    assert led.iloc[0]["cost"] == pytest.approx(exp_cost, abs=1e-15)
    assert led.iloc[0]["net"] == pytest.approx(exp_spot + exp_carry - exp_cost, abs=1e-15)
    assert led.iloc[0]["turnover"] == pytest.approx(n_post, abs=1e-15)
    # no further rebalance: no cost, and the book drifts with prices
    assert (led.iloc[1:]["cost"] == 0).all()
    h3 = wv * n_post * (1 + r3) / (1 + led.iloc[0]["net"])
    np.testing.assert_allclose(res.weights.iloc[0].to_numpy(), h3, rtol=0, atol=1e-15)
    r4 = np.expm1(spot.iloc[4].to_numpy())
    assert led.iloc[1]["spot_pnl"] == pytest.approx(float(h3 @ r4), abs=1e-15)
    assert led.iloc[1]["carry_pnl"] == pytest.approx(float(h3 @ carry.iloc[4].to_numpy()),
                                                     abs=1e-15)
    assert (led["carry_pnl"] != 0).all()


def test_default_fill_is_next_bar_not_the_decision_close():
    """Wrong answer fixed in W24b: the weights decided at close t used to earn
    all of day t+1 for free. Now the fill is half-way through day t+1."""
    spot, carry = _tiny_market()
    w = pd.Series({"AUD": 1.0, "JPY": 0.0})
    zero = carry * 0.0
    res = run_backtest(spot, zero, lambda hist: w, est_window=3, rebalance_every=100,
                       cost_bps=0.0)
    r = np.expm1(spot.iloc[3]["AUD"])
    # f = 0.5: the fill is at p_2 (1 + r/2); the book earns (1+r)/(1+r/2) - 1
    assert res.ledger.iloc[0]["net"] == pytest.approx((1 + r) / (1 + 0.5 * r) - 1, abs=1e-15)
    old = run_backtest(spot, zero, lambda hist: w, est_window=3, rebalance_every=100,
                       cost_bps=0.0, execution=AUCTION)
    assert old.ledger.iloc[0]["net"] == pytest.approx(r, abs=1e-15)


def test_cost_charged_on_weight_changes_only():
    spot, carry = _tiny_market()
    calls = []

    def wf(hist):
        calls.append(len(hist))
        w = 0.3 if len(calls) % 2 else 0.6
        return pd.Series({"AUD": w, "JPY": -w})

    res = run_backtest(spot, carry, wf, est_window=3, rebalance_every=2,
                       cost_bps=100.0, execution=AUCTION)
    led = res.ledger
    # rebalances on rows 2,4,6 -> filled at those closes, booked on 3,5,7
    cost_days = led[led["cost"] > 0].index
    assert list(cost_days) == [spot.index[3], spot.index[5], spot.index[7]]
    # second rebalance: trade from the drifted book h back up to 0.6 / -0.6
    # of the post-trade NAV N, paying 100 bp: N = 1 - 0.01 * |0.6 N - h|_1
    h = res.weights.loc[spot.index[4]].to_numpy()
    target = np.array([0.6, -0.6])
    n_post = 1.0
    for _ in range(100):
        n_post = 1.0 - 0.01 * float(np.abs(target * n_post - h).sum())
    assert led.loc[spot.index[5], "cost"] == pytest.approx(1.0 - n_post, abs=1e-15)
    assert led.loc[spot.index[5], "turnover"] == pytest.approx(
        float(np.abs(target * n_post - h).sum()), abs=1e-15)


def test_rebalance_frequency_respected():
    panel = make_panel(seed=1, n_days=300, currencies=["EUR", "JPY", "AUD"])
    dec = total_log_returns(panel.spots, panel.rates)
    n_calls = []

    def wf(hist):
        n_calls.append(hist.index[-1])
        return pd.Series(1.0 / 3, index=dec.total.columns)

    run_backtest(dec.spot, dec.carry, wf, est_window=100, rebalance_every=21)
    # calls at rows 99, 120, 141, ...
    expected = list(range(99, len(dec.total), 21))
    assert len(n_calls) == len(expected)
    assert n_calls[0] == dec.total.index[99]
    assert n_calls[1] == dec.total.index[120]


def test_no_lookahead_weight_func_sees_only_history():
    panel = make_panel(seed=1, n_days=250, currencies=["EUR", "JPY"])
    dec = total_log_returns(panel.spots, panel.rates)
    seen: list[pd.Timestamp] = []

    def wf(hist):
        seen.append(hist.index[-1])
        return pd.Series(0.5, index=dec.total.columns)

    run_backtest(dec.spot, dec.carry, wf, est_window=50, rebalance_every=10)
    # every history endpoint strictly precedes the first day those weights earn P&L
    for i, t_seen in enumerate(seen):
        applied_from = dec.total.index[49 + 10 * i + 1]
        assert t_seen < applied_from


def test_no_lookahead_future_spike_does_not_change_early_weights():
    panel = make_panel(seed=6, n_days=260, currencies=["EUR", "JPY", "AUD"])
    dec = total_log_returns(panel.spots, panel.rates)

    def wf(hist):
        mu = hist.tail(60).mean()
        w = (mu - mu.mean())
        return w / max(w.abs().sum(), 1e-12)

    base = run_backtest(dec.spot, dec.carry, wf, est_window=100, rebalance_every=21)
    spot2 = dec.spot.copy()
    spot2.iloc[-5:] += 0.5  # absurd future move
    bumped = run_backtest(spot2, dec.carry, wf, est_window=100, rebalance_every=21)
    cut = dec.spot.index[-30]
    pd.testing.assert_frame_equal(
        base.weights.loc[:cut], bumped.weights.loc[:cut]
    )


def test_pips_to_bps_hand_computed():
    # EURUSD at 1.2500, 25 pip cost: 25*0.0001/1.25 = 0.002 = 20 bps
    assert pips_to_bps(25, 1.25) == pytest.approx(20.0, rel=1e-12)
    # USDJPY at 110.00, pip size 0.01, 3 pips: 3*0.01/110 = 2.727... bps
    assert pips_to_bps(3, 110.0, pip_size=0.01) == pytest.approx(
        3 * 0.01 / 110.0 * 1e4, rel=1e-12
    )
    with pytest.raises(ValueError, match="pips"):
        pips_to_bps(-1, 1.25)
    with pytest.raises(ValueError, match="spot"):
        pips_to_bps(1, 0.0)


def test_backtest_input_validation():
    spot, carry = _tiny_market()
    with pytest.raises(ValueError, match="aligned"):
        run_backtest(spot, carry.iloc[:-1], lambda h: pd.Series(), est_window=3)
    with pytest.raises(ValueError, match="est_window"):
        run_backtest(spot, carry, lambda h: pd.Series(), est_window=1)
    with pytest.raises(ValueError, match="rebalance"):
        run_backtest(spot, carry, lambda h: pd.Series(), est_window=3,
                     rebalance_every=0)
    with pytest.raises(ValueError, match=r"NaN|missing"):
        run_backtest(spot, carry, lambda h: pd.Series({"AUD": 1.0}), est_window=3)
    with pytest.raises(ValueError, match="cost"):
        run_backtest(spot, carry,
                     lambda h: pd.Series({"AUD": 0.5, "JPY": -0.5}),
                     est_window=3, cost_bps=-1.0)


# ---------------------------------------------------------------------------
# Base-currency conversion (GBP reporting)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gbp_setup():
    panel = make_panel(seed=13, n_days=400)
    dec = total_log_returns(panel.spots, panel.rates)  # USD base
    conv = base_conversion_returns(panel.spots["GBP"], panel.rates, base="GBP")
    return panel, dec, conv.reindex(dec.total.index)


def test_base_switch_adds_base_ccy_own_return(gbp_setup):
    """Identity: EUR total return in GBP == EUR total in USD + USD total in GBP.

    Cross-check against the direct construction from the EURGBP cross rate:
    log-diff(EURUSD/GBPUSD) + (i_EUR - i_GBP)*dt, exactly.
    """
    panel, dec, conv = gbp_setup
    converted = convert_base(dec.total["EUR"], conv)
    cross = panel.spots["EUR"] / panel.spots["GBP"]  # EURGBP spot
    direct = (
        np.log(cross).diff()
        + ((panel.rates["EUR"] - panel.rates["GBP"]) * DT).shift(1)
    ).iloc[1:]
    assert np.max(np.abs(converted - direct)) < 1e-14


def test_dollar_neutral_log_returns_invariant_to_base(gbp_setup):
    """sum(w)=0 kills the common conversion term exactly (log-return version)."""
    _panel, dec, conv = gbp_setup
    rng = np.random.default_rng(2)
    w = pd.Series(rng.standard_normal(dec.total.shape[1]), index=dec.total.columns)
    w -= w.mean()  # dollar-neutral
    port_usd = dec.total @ w
    port_gbp = convert_base(dec.total, conv) @ w
    assert np.max(np.abs(port_usd - port_gbp)) < 1e-14
    assert sharpe_ratio(port_usd) == pytest.approx(sharpe_ratio(port_gbp), rel=1e-9)


def test_net_long_portfolio_shifts_by_conversion(gbp_setup):
    _panel, dec, conv = gbp_setup
    w = pd.Series(1.0 / dec.total.shape[1], index=dec.total.columns)  # sum = 1
    port_usd = dec.total @ w
    port_gbp = convert_base(dec.total, conv) @ w
    assert np.max(np.abs((port_gbp - port_usd) - conv)) < 1e-14


def test_conversion_validation(gbp_setup):
    panel, dec, conv = gbp_setup
    with pytest.raises(ValueError, match="cover"):
        convert_base(dec.total, conv.iloc[10:])
    with pytest.raises(ValueError, match="missing"):
        base_conversion_returns(panel.spots["GBP"], panel.rates.drop(columns="GBP"))
    with pytest.raises(ValueError, match="positive"):
        base_conversion_returns(panel.spots["GBP"] * -1.0, panel.rates)
