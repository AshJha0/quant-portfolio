"""W26 hedging: closed forms, identities and the orderings the literature predicts."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qp_core.black_scholes import carry_greeks
from qp_core.errors import InputError
from qp_core.hedging import (
    FxHedgePolicy,
    HedgeRule,
    HedgeRuleSpec,
    OptionGreeks,
    effectiveness,
    gamma_vega_hedge,
    gbm_paths,
    hedge_cost,
    hedge_short_option,
    leland_number,
    leland_sigma,
    oos_hedge_test,
    run_fx_hedge,
    whalley_wilmott_halfwidth,
)

S0, K, T, SIG = 100.0, 100.0, 0.25, 0.2


@pytest.fixture(scope="module")
def paths() -> np.ndarray:
    return gbm_paths(S0, 0.0, SIG, T, 63, 1500, seed=11)


def test_leland_hand_values() -> None:
    # Le = sqrt(2/pi) * 2 * 0.001 / (0.2 * sqrt(1/52)) by hand.
    le = math.sqrt(2 / math.pi) * 0.002 / (0.2 * math.sqrt(1 / 52))
    assert leland_number(0.2, 0.001, 1 / 52) == pytest.approx(le, rel=1e-15)
    assert leland_sigma(0.2, 0.001, 1 / 52) == pytest.approx(0.2 * math.sqrt(1 + le), rel=1e-15)
    assert leland_sigma(0.2, 0.0, 1 / 52) == 0.2


def test_ww_halfwidth_hand_value_and_scaling() -> None:
    h = whalley_wilmott_halfwidth(100.0, 0.02, 0.001, 1.0, 0.0, 1.0)
    assert h == pytest.approx((1.5 * 0.001 * 100 * 0.0004) ** (1 / 3), rel=1e-14)
    # Cube-root law: 8x the cost doubles the band.
    assert whalley_wilmott_halfwidth(100.0, 0.02, 0.008, 1.0, 0.0, 1.0) == pytest.approx(2 * h, rel=1e-12)
    assert whalley_wilmott_halfwidth(100.0, 0.02, 0.0, 1.0, 0.0, 1.0) == 0.0


def test_spec_validation() -> None:
    with pytest.raises(InputError):
        HedgeRuleSpec(HedgeRule.TIME, cost=-1.0)
    with pytest.raises(InputError):
        HedgeRuleSpec(HedgeRule.BAND, risk_aversion=0.0)
    with pytest.raises(InputError):
        HedgeRuleSpec(HedgeRule.TIME, every=0)


@pytest.mark.parametrize("rule", list(HedgeRule))
def test_pnl_identity_1e9(paths: np.ndarray, rule: HedgeRule) -> None:
    res = hedge_short_option(paths, K, T, 0.03, SIG, True, HedgeRuleSpec(rule, cost=0.001))
    assert np.max(np.abs(res.pnl - (res.hedge_error - res.cost))) < 1e-9
    assert np.all(res.cost >= 0.0)


def test_frictionless_daily_hedge_replicates(paths: np.ndarray) -> None:
    res = hedge_short_option(paths, K, T, 0.0, SIG, True, HedgeRuleSpec(HedgeRule.TIME))
    eff = effectiveness(res)
    # Discrete hedging error std ~ sqrt(pi/4) * vega * sigma / sqrt(N) (Bertsimas-Kogan-Lo 2000).
    vega = carry_greeks(S0, K, T, 0.0, 0.0, SIG, True).vega
    approx = math.sqrt(math.pi / 4) * vega * SIG / math.sqrt(63)
    assert eff.pnl_std == pytest.approx(approx, rel=0.15)
    assert abs(res.pnl.mean()) < 4 * eff.pnl_std / math.sqrt(res.pnl.size)
    assert eff.value > 0.9 and eff.se < 0.02
    assert eff.mean_cost == 0.0


def test_put_and_zero_vol_cases(paths: np.ndarray) -> None:
    res = hedge_short_option(paths, K, T, 0.0, SIG, False, HedgeRuleSpec(HedgeRule.TIME))
    assert effectiveness(res).value > 0.9


def test_cost_ordering(paths: np.ndarray) -> None:
    k = 0.002
    time_ = hedge_short_option(paths, K, T, 0.0, SIG, True, HedgeRuleSpec(HedgeRule.TIME, cost=k))
    band = hedge_short_option(
        paths, K, T, 0.0, SIG, True, HedgeRuleSpec(HedgeRule.BAND, cost=k, risk_aversion=1.0)
    )
    lel = hedge_short_option(paths, K, T, 0.0, SIG, True, HedgeRuleSpec(HedgeRule.LELAND, cost=k))
    # The band trades less and spends less than daily rebalancing.
    assert band.cost.mean() < time_.cost.mean()
    assert band.trades.mean() < time_.trades.mean()
    # Leland's raised vol overprices the premium hedge; mean P&L moves up versus time.
    assert lel.pnl.mean() > time_.pnl.mean()
    # Leland mean P&L is closer to zero than the time rule's cost drag.
    assert abs(lel.pnl.mean()) < abs(time_.pnl.mean())


def test_hedge_sigma_shape_check(paths: np.ndarray) -> None:
    with pytest.raises(InputError):
        hedge_short_option(paths, K, T, 0.0, SIG, True, HedgeRuleSpec(HedgeRule.TIME), hedge_sigma=np.ones(3))


def test_gamma_vega_hedge_neutralises() -> None:
    book = carry_greeks(100, 105, 0.5, 0.01, 0.0, 0.2, True)
    o1 = carry_greeks(100, 100, 0.25, 0.01, 0.0, 0.2, True)
    o2 = carry_greeks(100, 95, 1.0, 0.01, 0.0, 0.2, False)
    b = OptionGreeks(-10 * book.delta, -10 * book.gamma, -10 * book.vega)
    h = gamma_vega_hedge(
        b, OptionGreeks(o1.delta, o1.gamma, o1.vega), OptionGreeks(o2.delta, o2.gamma, o2.vega)
    )
    assert b.gamma + h.n1 * o1.gamma + h.n2 * o2.gamma == pytest.approx(0.0, abs=1e-12)
    assert b.vega + h.n1 * o1.vega + h.n2 * o2.vega == pytest.approx(0.0, abs=1e-10)
    assert b.delta + h.n1 * o1.delta + h.n2 * o2.delta + h.n_underlying == pytest.approx(0.0, abs=1e-12)
    assert hedge_cost(h, 0.05, 0.1, 0.01) == pytest.approx(
        abs(h.n1) * 0.05 + abs(h.n2) * 0.1 + abs(h.n_underlying) * 0.01
    )


def test_gamma_vega_singular_same_expiry() -> None:
    o = OptionGreeks(0.5, 0.02, 20.0)
    with pytest.raises(InputError):
        gamma_vega_hedge(OptionGreeks(0, 1, 1), o, OptionGreeks(0.4, 0.04, 40.0))


def test_fx_hedge_policies() -> None:
    rng = np.random.default_rng(5)
    x = 1.1 * np.exp(np.cumsum(0.006 * rng.standard_normal(500)))
    e = 1e6 + np.cumsum(2e4 * rng.standard_normal(500))
    none = run_fx_hedge(e, x, FxHedgePolicy("none", cost=1e-4))
    full = run_fx_hedge(e, x, FxHedgePolicy("full", cost=1e-4))
    thr = run_fx_hedge(e, x, FxHedgePolicy("threshold", band=1e5, cost=1e-4))
    for r in (none, full, thr):
        assert r.pnl_total == pytest.approx(r.fx_pnl - r.cost, abs=1e-6)
    assert none.cost == 0.0 and none.n_trades == 0
    assert full.pnl_std < thr.pnl_std < none.pnl_std
    assert thr.cost < full.cost and thr.n_trades < full.n_trades
    with pytest.raises(InputError):
        FxHedgePolicy("partial")


def test_oos_hedge_test_synthetic() -> None:
    s = gbm_paths(100.0, 0.0, 0.2, 4.0, 1008, 1, seed=8)[0]
    rows = oos_hedge_test(s, np.full(s.size, 0.2), window=21, cost=0.0005)
    assert rows["time"].n_windows == 48
    assert rows["band"].mean_cost < rows["time"].mean_cost
    for r in rows.values():
        assert r.error_var_se > 0 and "not an option price" in r.label
    with pytest.raises(InputError):
        oos_hedge_test(s[:100], np.full(100, 0.2))
