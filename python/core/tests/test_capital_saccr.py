"""SA-CCR against the BCBS 279 Annex 4a worked examples.

Every expected number below is copied from BCBS 279 (March 2014), Annex 4a,
pages 22-30. The paper rounds intermediate values for display only
(footnote 20), so each check is to the published rounding.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest
from qp_core.capital import saccr as s
from qp_core.errors import InputError

SWAPTION_DELTA = s.supervisory_delta_option(0.06, 0.05, 0.5, 1.0, call=False, bought=True)


def ex1_trades() -> list[s.IRTrade]:
    return [
        s.IRTrade("USD", 10_000, 0.0, 10.0, delta=1.0),
        s.IRTrade("USD", 10_000, 0.0, 4.0, delta=-1.0),
        s.IRTrade("EUR", 5_000, 1.0, 11.0, delta=SWAPTION_DELTA, maturity=11.0),
    ]


def ex2_trades() -> list[s.SingleFactorTrade]:
    return [
        s.SingleFactorTrade("CR_AA", "Firm A", 10_000, 0.0, 3.0, delta=1.0),
        s.SingleFactorTrade("CR_BBB", "Firm B", 10_000, 0.0, 6.0, delta=-1.0),
        s.SingleFactorTrade("CR_IDX_IG", "CDX.IG", 10_000, 0.0, 5.0, delta=1.0),
    ]


def ex3_trades() -> list[s.CommodityTrade]:
    return [
        s.CommodityTrade("CO_OIL_GAS", "crude oil", 10_000, 0.75, delta=1.0),
        s.CommodityTrade("CO_OIL_GAS", "crude oil", 20_000, 2.0, delta=-1.0),
        s.CommodityTrade("CO_METALS", "silver", 10_000, 5.0, delta=1.0),
    ]


def test_example1_interest_rates() -> None:
    sd = [round(s.supervisory_duration(a, b), 2) for a, b in [(0, 10), (0, 4), (1, 11)]]
    assert sd == [7.87, 3.63, 7.49]
    adj = [round(t.adjusted_notional) for t in ex1_trades()]
    assert adj == [78_694, 36_254, 37_428]
    assert round(SWAPTION_DELTA, 2) == -0.27
    assert round(s.ir_effective_notional(0, -36_254.0 * 1.0, 78_694.0)) == 59_270
    assert round(SWAPTION_DELTA * 37_428) == -10_083
    r = s.ead(s.NettingSet(mtm=30 - 20 + 50, ir=ex1_trades()))
    assert (round(r.rc), round(r.addon), r.multiplier, round(r.ead)) == (60, 347, 1.0, 569)


def test_example2_credit() -> None:
    adj = [round(t.adjusted_notional) for t in ex2_trades()]
    assert adj == [27_858, 51_836, 44_240]
    r = s.ead(s.NettingSet(mtm=20 - 40 + 0, credit=ex2_trades()))
    assert round(r.rc) == 0
    assert round(r.addon) == 282
    assert round(r.multiplier, 3) == 0.965
    assert round(r.ead) == 381


def test_example3_commodity() -> None:
    assert round(10_000 * s.maturity_factor(0.75) - 20_000) == -11_340
    r = s.ead(s.NettingSet(mtm=-50 - 30 + 100, commodity=ex3_trades()))
    assert (round(r.rc), round(r.addon), r.multiplier) == (20, 3_841, 1.0)
    assert round(r.ead) == 5_406


def test_example4_ir_plus_credit() -> None:
    r = s.ead(s.NettingSet(mtm=60 - 20, ir=ex1_trades(), credit=ex2_trades()))
    assert (round(r.rc), round(r.addon), round(r.ead)) == (40, 629, 936)


def test_example5_margined() -> None:
    mpor = s.mpor_days(5)
    assert mpor == 14
    ns = s.NettingSet(
        mtm=80.0,
        collateral=200.0,
        ir=ex1_trades(),
        commodity=ex3_trades(),
        mpor=mpor,
        th=0.0,
        mta=5.0,
        nica=150.0,
    )
    r = s.ead(ns)
    assert round(r.rc) == 0
    assert round(r.addon_by_class["IR"]) == 123
    assert round(r.addon_by_class["CO"]) == 1_278
    assert round(r.addon) == 1_401
    assert round(r.multiplier, 3) == 0.958
    assert round(r.ead) == 1_879


def test_supervisory_table_checksum() -> None:
    # Any edit to the CRE52.72 table must be deliberate: update this digest
    # together with the source line in saccr.py.
    blob = json.dumps(s.SUPERVISORY, sort_keys=True).encode()
    assert hashlib.sha256(blob).hexdigest()[:16] == _DIGEST


_DIGEST = "084d34dfeae7386c"


def test_table_spot_values() -> None:
    assert s.SUPERVISORY["IR"] == (0.005, None, 0.50)
    assert s.SUPERVISORY["FX"][0] == 0.04
    assert s.SUPERVISORY["CR_BBB"][0] == 0.0054
    assert s.SUPERVISORY["CO_ELECTRICITY"] == (0.40, 0.40, 1.50)
    assert s.SUPERVISORY["EQ_INDEX"] == (0.20, 0.80, 0.75)


def test_multiplier_limits() -> None:
    assert s.multiplier(10.0, 0.0, 100.0) == 1.0
    assert s.multiplier(-1e12, 0.0, 1.0) == pytest.approx(0.05)
    assert s.multiplier(0.0, 0.0, 0.0) == 1.0


def test_fx_addon_nets_within_pair() -> None:
    t = [
        s.FXTrade("EURUSD", 1_000, 2.0, 1.0),
        s.FXTrade("EURUSD", 400, 3.0, -1.0),
        s.FXTrade("GBPUSD", 100, 0.5, -1.0),
    ]
    assert s.fx_addon(t) == pytest.approx(0.04 * 600 + 0.04 * 100 * math.sqrt(0.5))


def test_margined_capped_at_unmargined() -> None:
    # A huge threshold makes margined RC exceed the unmargined EAD; CRE52.2 caps it.
    ns = s.NettingSet(mtm=0.0, ir=ex1_trades(), mpor=10, th=1e6)
    un = s.ead(s.NettingSet(mtm=0.0, ir=ex1_trades()))
    assert s.ead(ns).ead == pytest.approx(un.ead)


def test_bad_inputs() -> None:
    with pytest.raises(InputError):
        s.supervisory_duration(5.0, 1.0)
    with pytest.raises(InputError):
        s.supervisory_delta_option(-0.01, 0.01, 0.5, 1.0, call=True, bought=True)
    with pytest.raises(InputError):
        s.ead(s.NettingSet(mtm=0.0, equity=[s.SingleFactorTrade("XX", "a", 1, 0, 1)]))
    with pytest.raises(InputError):
        s.mpor_days(0)
