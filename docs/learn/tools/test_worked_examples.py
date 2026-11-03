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


@pytest.mark.parametrize(
    "page", ["L0/README.md", "L1/README.md", "L2/README.md", "L3/README.md", "L4/README.md"]
)
def test_every_level_page_has_the_four_parts(page: str) -> None:
    text = _page(page)
    for heading in ("## Plain idea", "## Formal statement", "## Worked example", "## Checks"):
        assert heading in text, f"{page} lacks {heading}"
