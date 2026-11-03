"""Backtest tests: exact ledger identity and no-lookahead mutation."""

import numpy as np
import pandas as pd
import pytest
from fx_regime import (
    StrategyConfig,
    oracle_regimes,
    run_backtest,
    static_carry_regimes,
)


def test_ledger_identity_exact(backtests2):
    """net = spot + carry - cost, row by row, to 1e-15."""
    for bt in backtests2.values():
        led = bt.ledger
        assert np.allclose(
            led["net"], led["spot"] + led["carry"] - led["cost"], atol=1e-15
        )
        assert (led["cost"] >= 0).all()
        assert (led["turnover"] >= 0).all()


def test_cumulative_consistency(backtests2):
    led = backtests2["filtered"].ledger
    assert np.isclose(led["net"].cumsum().iloc[-1], led["net"].sum(), atol=1e-12)


def test_weights_aligned_with_ledger(backtests2):
    bt = backtests2["filtered"]
    assert (bt.weights.index == bt.ledger.index).all()
    assert np.isfinite(bt.weights.to_numpy()).all()


def test_no_lookahead_mutation(panel2, det2):
    """CRITICAL: perturbing future returns leaves the past ledger intact."""
    cfg = StrategyConfig()
    bt1 = run_backtest(panel2.returns, panel2.deposit_rates, det2.regimes, cfg)
    cut = 900
    cut_date = panel2.returns.index[cut]
    rets2 = panel2.returns.copy()
    rets2.iloc[cut + 1:] *= -5.0
    bt2 = run_backtest(rets2, panel2.deposit_rates, det2.regimes, cfg)
    keep = bt1.ledger.index <= cut_date
    pd.testing.assert_frame_equal(bt1.ledger.loc[keep], bt2.ledger.loc[keep])
    assert not np.allclose(
        bt1.ledger.loc[~keep, "net"], bt2.ledger.loc[~keep, "net"]
    )


def test_position_uses_previous_day_regime():
    """Constructed case: the regime decided at t drives the day t+1 book."""
    idx = pd.bdate_range("2020-01-01", periods=80)
    ccys = ["AUD", "NZD", "JPY", "CHF"]
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.standard_normal((80, 4)) * 0.006,
                        index=idx, columns=ccys)
    rates = pd.DataFrame(
        {"AUD": 0.04, "NZD": 0.05, "JPY": 0.0, "CHF": -0.01, "USD": 0.02},
        index=idx,
    )
    regimes = pd.Series("risk_on", index=idx)
    flip = idx[70]
    regimes.loc[flip:] = "risk_off"
    cfg = StrategyConfig(cov_window=30, n_carry_long=1, n_carry_short=1,
                         rebalance_every=1000)
    bt = run_backtest(rets, rates, regimes, cfg)
    # day AFTER the flip is the first day traded on the risk_off book
    day_of_flip = bt.ledger.index.get_loc(flip)
    assert bt.ledger["regime"].iloc[day_of_flip] == "risk_on"
    assert bt.ledger["regime"].iloc[day_of_flip + 1] == "risk_off"
    w_after = bt.weights.iloc[day_of_flip + 1]
    assert w_after["JPY"] > 0 and w_after["AUD"] < 0


def test_spot_pnl_hand_check():
    """On a day with no trade the book held at the previous close, h, earns
    h . R (simple returns) and the carry h . d; it is not reset to target."""
    idx = pd.bdate_range("2020-01-01", periods=80)
    ccys = ["AUD", "NZD", "JPY", "CHF"]
    rets = pd.DataFrame(0.001, index=idx, columns=ccys)
    rates = pd.DataFrame(
        {"AUD": 0.04, "NZD": 0.05, "JPY": 0.0, "CHF": -0.01, "USD": 0.02},
        index=idx,
    )
    regimes = pd.Series("risk_off", index=idx)
    cfg = StrategyConfig(cov_window=30)
    bt = run_backtest(rets, rates, regimes, cfg)
    t = int(np.flatnonzero(bt.ledger["turnover"].to_numpy() == 0.0)[1])  # no trade
    h = bt.weights.iloc[t - 1]
    date = bt.weights.index[t]
    assert bt.ledger["spot"].iloc[t] == pytest.approx(
        float(h @ np.expm1(rets.loc[date])), abs=1e-15
    )
    # risk_off book: equal returns cancel (long havens = short risk), and
    # equal returns keep the dollar-neutral book dollar-neutral
    assert bt.ledger["spot"].iloc[t] == pytest.approx(0.0, abs=1e-12)
    # carry: differential vs USD 0.5*(-0.02-0.03) - 0.5*(0.02+0.03) = -0.05
    # p.a. per unit of |w|/0.5, ACT/365 on the calendar gap
    scale = h["JPY"] / 0.5
    gap = (date - bt.weights.index[t - 1]).days
    assert bt.ledger["carry"].iloc[t] == pytest.approx(-0.05 * gap / 365 * scale, rel=1e-9)
    # the book drifts: h_t = h_{t-1} (1 + R) / (1 + net), R = exp(0.001) - 1
    np.testing.assert_allclose(
        bt.weights.iloc[t], h * np.exp(0.001) / (1 + bt.ledger["net"].iloc[t]), rtol=0, atol=1e-14)


def test_static_baseline_is_always_risk_on(backtests2):
    assert (backtests2["static"].ledger["regime"] == "risk_on").all()


def test_oracle_uses_true_states(panel2, backtests2):
    led = backtests2["oracle"].ledger
    true = pd.Series(
        [panel2.state_names[s] for s in panel2.states],
        index=panel2.returns.index,
    )
    # regime driving day t is the TRUE state at t-1
    pos = panel2.returns.index.get_indexer(led.index)
    expected = true.iloc[pos - 1].to_numpy()
    assert (led["regime"].to_numpy() == expected).all()


def test_turnover_and_drift(backtests2, panel2):
    """Turnover is the notional traded at the fill; on days without a trade
    the book drifts, h_t = h_{t-1} (1 + R_t) / (1 + net_t), and pays nothing."""
    bt = backtests2["filtered"]
    led = bt.ledger
    assert led["turnover"].iloc[0] > 0
    assert ((led["turnover"] > 0) == (led["cost"] > 0)).all()
    quiet = np.flatnonzero(led["turnover"].to_numpy() == 0.0)
    quiet = quiet[quiet > 0]
    assert len(quiet) > 10
    r = np.expm1(panel2.returns.loc[led.index].to_numpy())
    h = bt.weights.to_numpy()
    net = led["net"].to_numpy()
    for k in quiet:
        np.testing.assert_allclose(h[k], h[k - 1] * (1 + r[k]) / (1 + net[k]), rtol=0,
                                   atol=1e-14)


def test_invalid_inputs_raise(panel2, det2):
    with pytest.raises(ValueError, match="USD"):
        run_backtest(
            panel2.returns,
            panel2.deposit_rates.drop(columns=["USD"]),
            det2.regimes,
        )
    bad = det2.regimes.copy()
    bad.index = bad.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="subset"):
        run_backtest(panel2.returns, panel2.deposit_rates, bad)
    with pytest.raises(ValueError, match="equal length"):
        oracle_regimes(panel2.returns.index, panel2.states[:-5],
                       panel2.state_names)


def test_static_carry_regimes_helper(panel2):
    reg = static_carry_regimes(panel2.returns.index)
    assert (reg == "risk_on").all()
    assert len(reg) == len(panel2.returns)
