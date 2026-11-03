"""Dated OIS and multi-curve bootstrap against QuantLib (W13b).

The oracle is ``tests/data/dated_bootstrap_usd.txt``, written by QuantLib
1.44 (``tests/oracle/gen_dated_bootstrap_fixture.py``): every instrument's
periods come from QuantLib's own schedules and the expected numbers are
QuantLib's pillar discount factors. Tolerances:

* pillar discount factors to 1e-12 absolute (QuantLib's iterative bootstrap
  stops at an accuracy of 1e-12; ours bisects to rounding);
* every input quote reprices to 1e-10 in rate (the W13 acceptance bar); in
  practice about 1e-15.

The Rust and C++ ports run the same file in ``rust/rates-curves`` and
``cpp/rates-curves``.
"""

from __future__ import annotations

import math
import pathlib

import pytest
from fi_rates.bootstrap import (
    DatedCurve,
    FixedPeriod,
    FloatPeriod,
    ForwardQuote,
    SwapQuote,
    bootstrap_dated,
    bootstrap_ois_dated,
    dated_quote_residual,
    futures_rate,
    hull_white_convexity,
)
from fi_rates.multicurve import bootstrap_projection_dated

FIXTURE = pathlib.Path(__file__).parent / "data" / "dated_bootstrap_usd.txt"


def _parse() -> dict:
    blocks: dict = {"hwconv": []}
    lines = [ln.split("#", 1)[0].split() for ln in FIXTURE.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    i = 0
    cur = None
    while i < len(lines):
        tok = lines[i]
        i += 1
        if tok[0] in ("build", "given"):
            opts = dict(o.split("=") for o in tok[2:])
            cur = {"kind": tok[0], "opts": opts, "quotes": [], "expect": [], "nodes": []}
            blocks[tok[1]] = cur
        elif tok[0] == "end":
            cur = None
        elif tok[0] == "node":
            cur["nodes"].append((int(tok[1]), float(tok[2])))
        elif tok[0] == "expect":
            cur["expect"].append((int(tok[1]), float(tok[2])))
        elif tok[0] == "fwd":
            s, e, tau, r = int(tok[1]), int(tok[2]), float(tok[3]), float(tok[4])
            cur["quotes"].append(ForwardQuote(s, e, tau, r, f"fwd {s}-{e}"))
        elif tok[0] == "swap":
            k, nf, nl = float(tok[1]), int(tok[2]), int(tok[3])
            fx = tuple(FixedPeriod(int(a[1]), float(a[2])) for a in lines[i:i + nf])
            fl = tuple(FloatPeriod(int(a[1]), int(a[2]), int(a[3]), float(a[4]))
                       for a in lines[i + nf:i + nf + nl])
            i += nf + nl
            cur["quotes"].append(SwapQuote(k, fx, fl, f"swap {k} to {fl[-1].end}"))
        elif tok[0] == "xbs":
            i += int(tok[4]) + int(tok[5])  # cross-currency: tested in fx_rates
        elif tok[0] == "hwconv":
            blocks["hwconv"].append(tuple(float(x) for x in tok[1:]))
        else:  # pragma: no cover - fixture format guard
            raise ValueError(f"unknown fixture line {tok}")
    return blocks


FX = _parse()


@pytest.fixture(scope="module")
def usd_ois():
    return bootstrap_ois_dated(FX["usd_ois"]["quotes"])


def test_ois_pillars_match_quantlib(usd_ois):
    exp = FX["usd_ois"]["expect"]
    assert list(usd_ois.curve.days) == [d for d, _ in exp]
    for (_, want), got in zip(exp, usd_ois.curve.dfs, strict=True):
        assert got == pytest.approx(want, rel=0, abs=1e-12)


def test_ois_reprices_every_quote(usd_ois):
    assert len(usd_ois.diagnostics) == 18
    assert usd_ois.max_abs_residual < 1e-10
    assert "error bp" in usd_ois.report()


def test_projection_curve_matches_quantlib(usd_ois):
    res = bootstrap_projection_dated(FX["term3m"]["quotes"], usd_ois.curve)
    exp = FX["term3m"]["expect"]
    assert list(res.curve.days) == [d for d, _ in exp]
    for (_, want), got in zip(exp, res.curve.dfs, strict=True):
        assert got == pytest.approx(want, rel=0, abs=1e-12)
    assert res.max_abs_residual < 1e-10


def test_projection_differs_from_single_curve(usd_ois):
    """Multi-curve matters: the same 10Y quote on one curve gives another curve."""
    multi = bootstrap_projection_dated(FX["term3m"]["quotes"], usd_ois.curve).curve
    single = bootstrap_dated(FX["term3m"]["quotes"]).curve
    assert abs(multi.df(3657) - single.df(3657)) > 1e-5


def test_hull_white_convexity_matches_quantlib():
    assert len(FX["hwconv"]) == 3
    for price, t1, t2, sigma, a, want in FX["hwconv"]:
        assert hull_white_convexity(price, t1, t2, sigma, a) == pytest.approx(want, rel=1e-13)


def test_hull_white_convexity_small_a_limit():
    lim = hull_white_convexity(96.0, 2.0, 2.25, 0.01, 0.0)
    near = hull_white_convexity(96.0, 2.0, 2.25, 0.01, 1e-7)
    assert lim == pytest.approx(near, rel=1e-6)
    # Ho-Lee: about sigma^2 t1 (t2 + dt) / 2 for small a
    assert lim == pytest.approx(0.5 * 0.01**2 * 2.0 * 2.5, rel=0.02)
    with pytest.raises(ValueError):
        hull_white_convexity(96.0, 2.0, 1.0, 0.01, 0.03)


def test_futures_rate():
    assert futures_rate(95.0) == pytest.approx(0.05)
    assert futures_rate(95.0, 1e-4) == pytest.approx(0.0499)


def test_extrapolation_holds_last_forward_flat(usd_ois):
    c = usd_ois.curve
    last = c.days[-1]
    f_in = c.forward_rate(last - 10)
    f_out = c.forward_rate(last + 1000)
    assert f_out == pytest.approx(f_in, rel=1e-12)  # no jump at the last pillar


def test_seasoned_quote_residual_on_built_curve(usd_ois):
    """A swap off the pillars still prices on the built curve (no exception, finite)."""
    q = SwapQuote(0.04, (FixedPeriod(400, 1.0), FixedPeriod(765, 1.0)),
                  (FloatPeriod(35, 400, 400, 1.0), FloatPeriod(400, 765, 765, 1.0)))
    r = dated_quote_residual(q, usd_ois.curve, usd_ois.curve)
    assert math.isfinite(r)


def test_input_order_does_not_matter(usd_ois):
    rev = bootstrap_ois_dated(list(reversed(FX["usd_ois"]["quotes"])))
    assert rev.curve.dfs == usd_ois.curve.dfs


def test_duplicate_pillar_raises():
    q = ForwardQuote(0, 30, 30 / 360, 0.05, "a")
    with pytest.raises(ValueError, match="share pillar"):
        bootstrap_ois_dated([q, ForwardQuote(10, 30, 20 / 360, 0.05, "b")])


def test_impossible_quote_names_the_quote():
    with pytest.raises(ValueError, match="crazy"):
        bootstrap_ois_dated([ForwardQuote(0, 30, 30 / 360, -50.0, "crazy")])


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        bootstrap_ois_dated([])
    with pytest.raises(ValueError):
        bootstrap_ois_dated([ForwardQuote(5, 5, 0.1, 0.05)])
    with pytest.raises(ValueError):
        bootstrap_ois_dated([SwapQuote(0.05, (), (FloatPeriod(0, 10, 10, 0.1),))])
    with pytest.raises(ValueError):
        DatedCurve([10, 5], [0.99, 0.98])
    with pytest.raises(ValueError):
        DatedCurve([10], [-1.0])
    with pytest.raises(ValueError):
        DatedCurve([10], [0.99]).df(-1)
    c = DatedCurve([10], [0.99])
    with pytest.raises(ValueError):
        bootstrap_dated([ForwardQuote(0, 5, 0.1, 0.05)], discount=c, projection=c)


def test_to_discount_curve_agrees_inside(usd_ois):
    dc = usd_ois.curve.to_discount_curve()
    for d in (30, 400, 2000):
        assert float(dc.df(d / 365.0)) == pytest.approx(usd_ois.curve.df(d), rel=1e-13)
