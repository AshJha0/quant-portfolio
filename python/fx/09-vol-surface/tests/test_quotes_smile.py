"""Quotes <-> five-vol smile: exact linear relations and conventions."""

import numpy as np
import pytest

from fx_surface import SmileQuotes, quotes_from_vols, vols_from_quotes


def test_quotes_to_vols_exact_relations():
    q = SmileQuotes(atm=0.10, rr25=-0.02, bf25=0.0035, rr10=-0.036, bf10=0.0116)
    v = vols_from_quotes(q)
    assert v["25c"] == pytest.approx(0.10 + 0.0035 - 0.01, abs=1e-15)
    assert v["25p"] == pytest.approx(0.10 + 0.0035 + 0.01, abs=1e-15)
    assert v["10c"] == pytest.approx(0.10 + 0.0116 - 0.018, abs=1e-15)
    assert v["10p"] == pytest.approx(0.10 + 0.0116 + 0.018, abs=1e-15)
    assert v["atm"] == 0.10


@pytest.mark.parametrize(
    "atm,rr25,bf25,rr10,bf10",
    [
        (0.072, -0.0045, 0.0020, -0.0080, 0.0066),
        (0.101, -0.0170, 0.0030, -0.0306, 0.0087),
        (0.350, 0.0600, 0.0120, 0.1080, 0.0400),
        (0.055, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_round_trip_quotes_vols_quotes(atm, rr25, bf25, rr10, bf10):
    q = SmileQuotes(atm, rr25, bf25, rr10, bf10)
    q2 = quotes_from_vols(vols_from_quotes(q))
    for field in ("atm", "rr25", "bf25", "rr10", "bf10"):
        assert getattr(q2, field) == pytest.approx(getattr(q, field), abs=1e-14)


def test_round_trip_vols_quotes_vols():
    vols = {"10p": 0.1296, "25p": 0.1135, "atm": 0.10, "25c": 0.0935, "10c": 0.0936}
    v2 = vols_from_quotes(quotes_from_vols(vols))
    for k in vols:
        assert v2[k] == pytest.approx(vols[k], abs=1e-14)


def test_positive_rr_means_calls_richer():
    v = vols_from_quotes(SmileQuotes(0.10, +0.02, 0.003, +0.03, 0.008))
    assert v["25c"] > v["25p"]
    assert v["10c"] > v["10p"]


def test_negative_rr_means_puts_richer():
    v = vols_from_quotes(SmileQuotes(0.10, -0.02, 0.003, -0.03, 0.008))
    assert v["25c"] < v["25p"]
    assert v["10c"] < v["10p"]


def test_typical_bf_nonnegative_gives_convex_wings():
    v = vols_from_quotes(SmileQuotes(0.10, -0.01, 0.004, -0.018, 0.012))
    assert 0.5 * (v["25c"] + v["25p"]) >= v["atm"]
    assert 0.5 * (v["10c"] + v["10p"]) >= v["atm"]


def test_negative_bf_warns_but_proceeds():
    with pytest.warns(UserWarning, match="negative butterfly"):
        v = vols_from_quotes(SmileQuotes(0.10, 0.0, -0.002, 0.0, 0.001))
    assert v["25c"] == pytest.approx(0.098)


def test_quote_set_implying_negative_vol_raises():
    with pytest.raises(ValueError, match="non-positive vol"):
        vols_from_quotes(SmileQuotes(0.02, -0.10, 0.001, -0.15, 0.002))


def test_nonpositive_atm_raises():
    with pytest.raises(ValueError, match="ATM vol"):
        SmileQuotes(0.0, 0.0, 0.0, 0.0, 0.0)


def test_missing_pillar_raises():
    with pytest.raises(ValueError, match="missing pillar"):
        quotes_from_vols({"atm": 0.1, "25c": 0.11, "25p": 0.1, "10c": 0.12})


def test_zero_rr_bf_flat_smile():
    v = vols_from_quotes(SmileQuotes(0.08, 0.0, 0.0, 0.0, 0.0))
    assert len({round(x, 15) for x in v.values()}) == 1


def test_preset_round_trips(eurusd, usdjpy, em_market):
    for market in (eurusd, usdjpy, em_market):
        for sl in market.slices:
            q2 = quotes_from_vols(vols_from_quotes(sl.quotes))
            assert q2.rr25 == pytest.approx(sl.quotes.rr25, abs=1e-14)
            assert q2.bf10 == pytest.approx(sl.quotes.bf10, abs=1e-14)
