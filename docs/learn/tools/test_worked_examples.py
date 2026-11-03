"""Recompute every number printed in a worked example under docs/learn/.

If a page changes a figure, or the code behind it changes, this fails. The
package source folders are put on ``sys.path`` here so the test runs from the
repository root without installing anything.
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom, norm

REPO = Path(__file__).resolve().parents[3]
for src in (
    "python/foundations/03-black-scholes-replication/src",
    "python/foundations/01-risk-metrics/src",
    "python/equity/04-fixed-income/src",
    "python/equity/06-credit-risk/src",
    "python/equity/03-var-es-engine/src",
    "python/core/src",
    "python/execution/src",
    "python/posttrade/src",
):
    sys.path.insert(0, str(REPO / src))

from eq_bs_replication.black_scholes import call_greeks, call_price  # noqa: E402

LEARN = REPO / "docs" / "learn"


def _page(rel: str) -> str:
    return (LEARN / rel).read_text(encoding="utf-8")


def test_l0_normal_var() -> None:
    z = norm.ppf(0.99)
    assert round(z, 4) == 2.3263
    assert round(z * 0.02 * 1_000_000) == 46_527
    assert round(2 * z * 0.02 * 1_000_000) == 93_054
    assert "$46,527" in _page("L0/README.md")


def test_l1_returns_and_scaling() -> None:
    assert math.isclose(1.10 * 0.90 - 1, -0.01)
    up, down = math.log(1.10), math.log(0.90)
    assert (round(up, 5), round(down, 5), round(up + down, 5)) == (0.09531, -0.10536, -0.01005)
    assert round(0.0126 * math.sqrt(252), 3) == 0.200
    assert math.ceil(1000 * 0.01) == 10 and math.ceil(250 * 0.01) == 3
    assert "2.506" in (REPO / "docs" / "MARKET_RISK.md").read_text(encoding="utf-8")


def test_l2_black_scholes() -> None:
    args = (100.0, 100.0, 0.05, 0.20, 1.0)
    price = call_price(*args)
    g = call_greeks(*args)
    assert round(price, 4) == 10.4506
    assert round(g.delta, 4) == 0.6368
    assert round(g.gamma, 6) == 0.018762
    assert round(g.vega, 3) == 37.524
    assert round(g.vega / 100, 4) == 0.3752
    assert round(g.theta, 4) == -6.4140
    assert round(price - 100 + 100 * math.exp(-0.05), 4) == 5.5735
    page = _page("L2/README.md")
    for figure in ("10.4506", "0.6368", "0.018762", "37.524", "0.3752", "-6.4140", "5.5735"):
        assert figure in page


def test_rates_bond() -> None:
    from fi_rates.bond import FixedRateBond, price_from_ytm
    from fi_rates.risk import convexity, dv01, macaulay_duration, modified_duration

    bond = FixedRateBond(dt.date(2026, 1, 15), dt.date(2036, 1, 15), 0.05, frequency=2)
    s = dt.date(2026, 1, 15)
    p5, p6, p4 = (price_from_ytm(bond, s, y) for y in (0.05, 0.06, 0.04))
    assert (round(p5, 4), round(p6, 4), round(p4, 4)) == (100.0, 92.5613, 108.1757)
    dmod, cvx = modified_duration(bond, s, 0.05), convexity(bond, s, 0.05)
    assert round(macaulay_duration(bond, s, 0.05), 4) == 7.9894
    assert round(dmod, 4) == 7.7946
    assert round(cvx, 3) == 73.629
    assert round(dv01(bond, s, 0.05), 5) == 0.07795
    first = -dmod * 0.01 * 100
    second = 0.5 * cvx * 0.0001 * 100
    assert round(first, 3) == -7.795
    assert round(first + second, 3) == -7.426
    assert round(p6 - 100, 3) == -7.439
    assert round(-first + second, 3) == 8.163
    assert round(p4 - 100, 3) == 8.176
    page = _page("L2/rates.md")
    for figure in (
        "92.5613",
        "108.1757",
        "7.9894",
        "7.7946",
        "73.629",
        "0.07795",
        "-7.426",
        "-7.439",
        "+8.176",
    ):
        assert figure in page


def test_credit_cds() -> None:
    from eq_credit.cds import HazardCurve, cds_legs, par_spread

    curve = HazardCurve(np.array([10.0]), np.array([0.02]))
    prot, annuity = cds_legs(curve, 5.0, 0.03, 0.4)
    spread = par_spread(curve, 5.0, 0.03, 0.4)
    assert round(prot, 6) == 0.053088
    assert round(annuity, 4) == 4.4074
    assert round(spread * 1e4, 2) == 120.45
    assert round(0.02 * 0.6 * 1e4, 1) == 120.0
    assert round(math.exp(-0.1) * 100, 2) == 90.48
    page = _page("L2/credit.md")
    for figure in ("0.053088", "4.4074", "120.45 bp", "90.48 %"):
        assert figure in page


def test_stochastic_calculus() -> None:
    rng = np.random.default_rng(7)
    sigma, n = 0.2, 200_000
    ratio = np.exp(-0.5 * sigma**2 + sigma * rng.standard_normal(n))
    se = ratio.std(ddof=1) / math.sqrt(n)
    assert abs(ratio.mean() - 1.0) < 3 * se
    assert round(float(np.median(ratio)), 3) == 0.980
    assert round(math.exp(-0.02), 4) == 0.9802

    S, r = 100.0, 0.05
    v = call_price(S, 100.0, r, sigma, 1.0)
    g = call_greeks(S, 100.0, r, sigma, 1.0)
    terms = (g.theta, 0.5 * sigma**2 * S**2 * g.gamma, r * S * g.delta, -r * v)
    assert [round(t, 4) for t in terms] == [-6.4140, 3.7524, 3.1842, -0.5225]
    assert abs(sum(terms)) < 1e-9
    assert round(g.delta, 5) == 0.63683


def test_microstructure() -> None:
    assert math.isclose(2 * math.sqrt(0.0001), 0.02)
    kyle = 2.0 / (2 * 1_000_000)
    assert math.isclose(kyle, 0.000001)
    assert math.isclose(kyle * 50_000, 0.05)
    assert math.isclose(0.02 * math.sqrt(0.01), 0.002)
    assert math.isclose(0.02 * math.sqrt(0.04), 0.004)
    assert round(2 * math.sqrt(2), 2) == 2.83


def test_l4_basel_binomial() -> None:
    from eq_var.backtesting import basel_zone_probabilities

    pmf = [binom.pmf(k, 250, 0.01) for k in range(6)]
    cdf = [binom.cdf(k, 250, 0.01) for k in range(6)]
    assert [round(p, 4) for p in pmf] == [0.0811, 0.2047, 0.2574, 0.2149, 0.1341, 0.0666]
    assert [round(c, 4) for c in cdf] == [0.0811, 0.2858, 0.5432, 0.7581, 0.8922, 0.9588]
    assert round(binom.sf(4, 250, 0.02), 2) == 0.56
    table = basel_zone_probabilities(250, 0.01)
    assert table.shape[0] >= 6
    numeric = table.select_dtypes("number").to_numpy()
    assert np.any(np.isclose(numeric, cdf[4], atol=5e-5))


def _in(page: str, *figures: str) -> None:
    text = _page(page)
    for figure in figures:
        assert figure in text, f"{page} lacks {figure}"


def test_market_making_and_routing() -> None:
    from qp_exec.crb.book import CentralRiskBook, Fill
    from qp_exec.mm.quotes import GLFT, ASParams, GLFTParams, avellaneda_stoikov, glft_asymptotic
    from qp_exec.rfq.quote import HitRatioModel, optimal_markup
    from qp_exec.sor.allocation import solve_allocation

    q = avellaneda_stoikov(100.0, 2, 1.0, ASParams(gamma=0.1, sigma=2.0, k=1.5))
    assert (round(q.reservation, 4), round(q.bid, 4), round(q.ask, 4)) == (99.2, 98.3546, 100.0454)
    assert round(q.ask - q.bid, 4) == 1.6908
    p = GLFTParams(gamma=0.1, sigma=2.0, k=1.5, A=140.0, q_max=20)
    exact0, approx0 = GLFT(p).quote(100.0, 0), glft_asymptotic(100.0, 0, p)
    exact2 = GLFT(p).quote(100.0, 2)
    assert (round(exact0.bid, 4), round(exact0.ask, 4)) == (99.3284, 100.6716)
    assert (round(approx0.bid, 4), round(approx0.ask, 4)) == (99.3288, 100.6712)
    assert round(exact2.reservation, 4) == 99.8952
    a = solve_allocation(S=8, h=0.5, f=0.3, c=[-0.7], lam_u=2.0, D=np.arange(10.0).reshape(-1, 1))
    assert (round(a.market, 6), [round(x, 6) for x in a.limits]) == (3.0, [5.0])
    assert round(optimal_markup(HitRatioModel(intercept=1.0, slope=-200.0)), 6) == 0.007836
    crb = CentralRiskBook(["ES", "NQ"])
    crb.book(Fill("eq_cash", "ES", 10.0, 100.0, "A"))
    crb.book(Fill("eq_deriv", "ES", -7.0, 100.0, "B"))
    assert (crb.net()["ES"], crb.gross()["ES"]) == (3.0, 17.0)
    _in("L3/market-making-and-routing.md", "1.6908", "98.3546", "100.0454", "99.3284",
        "100.6716", "99.3288", "100.6712", "99.8952", "0.007836", "Net 3, gross 17")


def test_trading_systems() -> None:
    import importlib.util

    from qp_posttrade.settlement import us_equity_settle

    assert round(1.02 / 1.01 - 1, 4) == 0.0099
    spec = importlib.util.spec_from_file_location(
        "kill_contract_learn", REPO / "python" / "oms" / "kill" / "contract.py"
    )
    assert spec is not None and spec.loader is not None
    kc = importlib.util.module_from_spec(spec)
    sys.modules["kill_contract_learn"] = kc
    spec.loader.exec_module(kc)
    ks = kc.KillSwitch()
    mm1, mm2 = ("QP", "RATES", "MM1", "UST10Y"), ("QP", "RATES", "MM2", "UST2Y")
    fx = ("QP", "FX", "SPOT", "EURUSD")
    for i, path in ((1, mm1), (2, mm2), (3, fx)):
        ks.on_order_working(i, path)
    kid, cancels = ks.engage(("QP", "RATES", "*", "*"), "alice")
    assert cancels == [1, 2] and ks.state(kid) == "ENGAGED"
    assert ks.check(mm1) == "KILLED:DESK:QP/RATES/*/*" and ks.check(fx) is None
    ks.on_order_terminal(1)
    ks.on_order_terminal(2)
    assert ks.state(kid) == "KILLED"
    assert ks.request_reset(kid, "alice") and not ks.approve_reset(kid, "alice")
    assert ks.state(kid) == "RESET_PENDING"
    assert ks.approve_reset(kid, "bob") and ks.state(kid) == "CLEARED"
    assert ks.check(mm1) is None
    assert us_equity_settle(dt.date(2026, 10, 9)) == dt.date(2026, 10, 13)
    assert us_equity_settle(dt.date(2026, 12, 31)) == dt.date(2027, 1, 4)
    _in("L3/trading-systems.md", "1.02 / 1.01 - 1 = 0.0099", "KILLED:DESK:QP/RATES/*/*",
        "Tuesday 13 October", "4 January 2027")


def test_rates_vol_and_explain() -> None:
    from qp_core.risk.pnl_explain import explain
    from qp_core.scenarios.historical import scenario
    from qp_core.vol.sabr import sabr_vol

    args = (5.0, 0.02, 0.5, -0.3, 0.4)
    atm_o = sabr_vol(0.03, 0.03, *args)
    atm_h = sabr_vol(0.03, 0.03, *args, variant="hagan")
    assert round(atm_o * 100, 4) == round(atm_h * 100, 4) == 12.1209
    assert round(sabr_vol(0.03, 0.01, *args) * 100, 3) == 28.346
    assert round(sabr_vol(0.03, 0.01, *args, variant="hagan") * 100, 3) == 28.182
    e = explain(hpl=9.25, time={"theta": -1.0}, delta=[100.0], move=[0.1], labels=["x"],
                gamma=np.array([[40.0]]))
    assert math.isclose(e.risk["x"], 10.2) and math.isclose(e.rtpl, 9.2)
    assert math.isclose(e.unexplained, 0.05, abs_tol=1e-12)
    shock = scenario("chf_2015").shocks["EURCHF"]
    assert round(shock, 4) == -0.1555
    assert round(1.2010 * math.exp(shock), 4) == 1.0280
    _in("L3/rates-vol-and-pnl-explain.md", "12.1209", "28.346", "28.182", "9.2", "0.05",
        "-0.1555", "1.0280")


def test_dated_swaps_and_bond_futures() -> None:
    from fi_rates import fixings, legs
    from fi_rates.bond_futures import cme_conversion_factor

    store = fixings.FixingStore()
    store.add("SOFR", dt.date(2024, 1, 4), 0.0530)
    store.add("SOFR", dt.date(2024, 1, 5), 0.0531)
    rate, fixed = legs.compounded_rate(
        fixings.SOFR, dt.date(2024, 1, 4), dt.date(2024, 1, 8), None, store, dt.date(2024, 1, 9)
    )
    assert round(rate * 100, 4) == 5.3081 and fixed == 1.0
    assert round((0.0530 + 3 * 0.0531) / 4 * 100, 4) == 5.3075
    assert cme_conversion_factor(0.0375, dt.date(2018, 11, 15), dt.date(2008, 12, 1)) == 0.8357
    assert math.isclose(124 / 125, 0.992) and math.isclose(0.008 * 0.7, 0.0056)
    assert math.isclose(123 / 125, 0.984) and math.isclose(0.008 * 0.6, 0.0048)
    _in("L2/dated-swaps-and-bond-futures.md", "5.3081 %", "5.3075 %", "0.8357", "0.992",
        "0.0056", "0.984", "0.0048")


def test_counterparty_and_capital() -> None:
    from qp_core.capital.saccr import (
        IRTrade,
        NettingSet,
        ead,
        supervisory_delta_option,
        supervisory_duration,
    )
    from qp_core.xva.adjustments import HazardCurve, cva

    profile = cva([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1.0] * 6, HazardCurve.flat(0.02), 0.4)
    assert round(profile, 4) == 0.0571
    assert round(1 - math.exp(-0.1), 4) == 0.0952
    delta = supervisory_delta_option(0.06, 0.05, 0.5, 1.0, call=False, bought=True)
    trades = [
        IRTrade("USD", 10_000, 0.0, 10.0, delta=1.0),
        IRTrade("USD", 10_000, 0.0, 4.0, delta=-1.0),
        IRTrade("EUR", 5_000, 1.0, 11.0, delta=delta),
    ]
    r = ead(NettingSet(mtm=30 - 20 + 50, ir=trades))
    assert (round(r.addon), round(r.rc), round(r.ead)) == (347, 60, 569)
    assert round(delta, 4) == -0.2694 and round(supervisory_duration(0.0, 10.0), 4) == 7.8694
    _in("L2/counterparty-and-capital.md", "0.0571", "0.0952", "347", "569", "-0.2694", "7.8694")


def test_aad() -> None:
    from qp_core.aad.black_scholes import bs_greeks_aad
    from qp_core.aad.tape import Tape, exp
    from qp_core.black_scholes import carry_greeks

    tape = Tape()
    x0, x1 = tape.var(1.0), tape.var(2.0)
    y = x0 * x1 + exp(x0)
    g = tape.gradient(y, [x0, x1])
    assert (round(y.value, 6), round(g[0], 6), round(g[1], 6)) == (4.718282, 4.718282, 1.0)
    a = bs_greeks_aad(100.0, 100.0, 1.0, 0.05, 0.02, 0.2)
    c = carry_greeks(100.0, 100.0, 1.0, 0.05, 0.02, 0.2, True)
    assert (round(a.price, 4), round(a.delta, 4), round(a.vega, 2)) == (9.2270, 0.5869, 37.90)
    assert math.isclose(a.delta, c.delta, abs_tol=1e-12)
    assert math.isclose(a.vega, c.vega, rel_tol=1e-12)
    readme = (REPO / "rust" / "aad" / "README.md").read_text(encoding="utf-8")
    for cell in ("1.7-2.1x", "1.6-1.8x", "2.0-6.7x", "2.1-2.4x", "13-34x"):
        assert cell in readme
    _in("L4/aad.md", "4.718282", "9.2270", "0.5869", "37.90", "1.6 to 2.1", "2.0 to 6.7",
        "13 to 34")


@pytest.mark.parametrize(
    "page", ["L0/README.md", "L1/README.md", "L2/README.md", "L3/README.md", "L4/README.md"]
)
def test_every_level_page_has_the_four_parts(page: str) -> None:
    text = _page(page)
    for heading in ("## Plain idea", "## Formal statement", "## Worked example", "## Checks"):
        assert heading in text, f"{page} lacks {heading}"
