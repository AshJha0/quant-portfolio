"""Cross-currency discount curve from EURUSD basis swaps against QuantLib (W13b).

Oracle: the ``eur_xccy`` block of the shared fixture
``python/equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt``,
written by QuantLib 1.44's ``ConstNotionalCrossCurrencyBasisSwapRateHelper``
(constant notionals, 3M/3M, spread on the EUR leg, USD collateral). The
USD OIS curve and the EUR OIS projection curve are taken as QuantLib built
them (the ``expect`` lines of ``usd_ois`` and the ``eur_ois`` nodes), so
this test checks the cross-currency step alone. Pillar discount factors to
1e-12; every basis quote reprices to 1e-10.
"""

from __future__ import annotations

import pathlib

import pytest
from fx_rates.xccy import (
    XccyBasisQuote,
    XccyCurve,
    XccyFloatPeriod,
    bootstrap_xccy_discount,
    xccy_basis_residual,
)

FIXTURE = (pathlib.Path(__file__).resolve().parents[3] / "equity" / "04-fixed-income"
           / "tests" / "data" / "dated_bootstrap_usd.txt")


def _parse():
    rows = [ln.split("#", 1)[0].split() for ln in FIXTURE.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r]
    out: dict = {}
    block = None
    i = 0
    while i < len(rows):
        r = rows[i]
        i += 1
        if r[0] in ("build", "given"):
            block = out.setdefault(r[1], {"expect": [], "nodes": [], "quotes": []})
        elif r[0] == "swap":
            i += int(r[2]) + int(r[3])
        elif r[0] == "expect":
            block["expect"].append((int(r[1]), float(r[2])))
        elif r[0] == "node":
            block["nodes"].append((int(r[1]), float(r[2])))
        elif r[0] == "xbs":
            nf, nd = int(r[4]), int(r[5])

            def per(a):
                return XccyFloatPeriod(int(a[1]), int(a[2]), int(a[3]), float(a[4]))
            fl = tuple(per(a) for a in rows[i:i + nf])
            dl = tuple(per(a) for a in rows[i + nf:i + nf + nd])
            i += nf + nd
            block["quotes"].append(XccyBasisQuote(float(r[1]), int(r[2]), int(r[3]), fl, dl,
                                                  f"basis to {r[3]}"))
    return out


FX = _parse()


def _curve(pairs):
    return XccyCurve([d for d, _ in pairs], [x for _, x in pairs])


USD = _curve(FX["usd_ois"]["expect"])
EUR_OIS = _curve(FX["eur_ois"]["nodes"])


def test_xccy_curve_matches_quantlib():
    res = bootstrap_xccy_discount(FX["eur_xccy"]["quotes"], EUR_OIS, USD)
    exp = FX["eur_xccy"]["expect"]
    assert len(exp) == 6
    assert list(res.curve.days) == [d for d, _ in exp]
    for (_, want), got in zip(exp, res.curve.dfs, strict=True):
        assert got == pytest.approx(want, rel=0, abs=1e-12)
    assert res.max_abs_residual < 1e-10


def test_negative_basis_raises_eur_discount_factors():
    """A negative basis on the EUR leg means the EUR receiver accepts less
    than EUR OIS flat to swap into USD: EUR cash flows under USD collateral
    are discounted at rates below EUR OIS, so the discount factors are
    higher (Fujii, Shimada and Takahashi 2010, sec. 4)."""
    res = bootstrap_xccy_discount(FX["eur_xccy"]["quotes"], EUR_OIS, USD)
    for d, x in zip(res.curve.days, res.curve.dfs, strict=True):
        assert x > EUR_OIS.df(d)


def test_zero_basis_recovers_projection_curve():
    """With no basis the collateral-adjusted curve is the projection curve.

    Exact on a flat curve, which every log-linear curve reproduces between
    any pillars; on a general curve the pillars differ and so do the
    interpolated coupons, so this identity is only exact here."""
    import math

    flat = XccyCurve([1, 20000], [math.exp(-0.03 / 365), math.exp(-0.03 * 20000 / 365)])
    zero = [XccyBasisQuote(0.0, q.start, q.end, q.foreign, q.domestic, q.label)
            for q in FX["eur_xccy"]["quotes"]]
    res = bootstrap_xccy_discount(zero, flat, USD)
    for d, x in zip(res.curve.days, res.curve.dfs, strict=True):
        assert x == pytest.approx(flat.df(d), rel=0, abs=1e-13)


def test_residual_and_validation():
    q = FX["eur_xccy"]["quotes"][0]
    assert abs(xccy_basis_residual(q, EUR_OIS, EUR_OIS, USD)) > 1e-5
    with pytest.raises(ValueError):
        bootstrap_xccy_discount([], EUR_OIS, USD)
    with pytest.raises(ValueError, match="share pillar"):
        bootstrap_xccy_discount([q, q], EUR_OIS, USD)
    with pytest.raises(ValueError):
        XccyBasisQuote(0.0, 10, 5, q.foreign, q.domestic)
    with pytest.raises(ValueError):
        XccyCurve([5, 5], [0.9, 0.8])
