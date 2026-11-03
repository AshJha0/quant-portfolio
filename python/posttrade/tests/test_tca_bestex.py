from __future__ import annotations

import random
from datetime import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qp_posttrade import STAMP
from qp_posttrade.bestex import build, reconcile
from qp_posttrade.model import Fill, Side
from qp_posttrade.tca import fill_costs, tca


def _fill(
    i: int, side: Side, qty: float, px: float, arr: float, mid: float, venue: str, fee: float = 0.0
) -> Fill:
    return Fill(
        f"e{i}",
        "A",
        "XYZ",
        side,
        qty,
        px,
        datetime(2026, 11, 2, 15),
        venue,
        arrival_px=arr,
        mid_at_fill=mid,
        fee=fee,
    )


def test_sell_signs() -> None:
    c = fill_costs(_fill(1, "SELL", 100, 9.98, 10.0, 9.99, "V"))
    assert c.shortfall == pytest.approx(2.0)
    assert c.spread == pytest.approx(1.0)


def test_missing_reference_prices_raise() -> None:
    with pytest.raises(ValueError, match="arrival"):
        fill_costs(_fill(1, "BUY", 100, 10, 0.0, 10, "V"))


def test_tca_by_venue_sums_to_total() -> None:
    fs = [_fill(1, "BUY", 100, 10.02, 10, 10.01, "V1", 0.3), _fill(2, "BUY", 200, 10.0, 10, 10.0, "V2", 0.2)]
    tot, by = tca(fs)
    assert tot.stamp == STAMP
    assert tot.shortfall == pytest.approx(2.0)
    assert tot.total_cost == pytest.approx(2.5)
    assert sum(v.qty for v in by.values()) == tot.qty == 300


def test_rts28_ranking_and_605_buckets() -> None:
    fs = [
        _fill(1, "BUY", 50, 10.01, 10, 10.0, "SMALL"),
        _fill(2, "BUY", 5000, 10.0, 10, 10.0, "BIG"),
        _fill(3, "BUY", 600, 10.0, 10, 10.0, "MID"),
    ]
    r = build(fs)
    assert r.stamp == STAMP
    assert [x.venue for x in r.rts28] == ["BIG", "MID", "SMALL"]
    assert sum(x.pct_volume for x in r.rts28) == pytest.approx(100)
    b = {x.bucket: x for x in r.rule605}
    assert b["0-99"].fills == 1
    assert b["0-99"].avg_eff_spread_bps == pytest.approx(20.0)
    assert b["2000-9999"].qty == 5000


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 80))
def test_bestex_reconciles_to_tca(seed: int, n: int) -> None:
    rng = random.Random(seed)
    fs = []
    for i in range(n):
        mid = 100 + rng.gauss(0, 1)
        side: Side = rng.choice(["BUY", "SELL"])
        s = 1 if side == "BUY" else -1
        qty = rng.choice([0.5, 10, 150, 800, 3000, 20000])
        fs.append(
            _fill(
                i,
                side,
                qty,
                mid + s * rng.uniform(-0.02, 0.05),
                100.0,
                mid,
                rng.choice("ABCDEFG"),
                rng.uniform(0, 2),
            )
        )
    tot, _ = tca(fs)
    assert reconcile(build(fs), tot) == []


def test_reconcile_reports_a_difference() -> None:
    fs = [_fill(1, "BUY", 100, 10.02, 10, 10.01, "V1")]
    tot, _ = tca(fs)
    r = build([*fs, _fill(2, "BUY", 1, 10, 10, 10, "V1")])
    assert any(d.startswith("rts28.qty") for d in reconcile(r, tot))
