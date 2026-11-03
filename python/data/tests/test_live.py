"""Live checks against the real endpoints. Opt-in: QP_DATA_LIVE_TESTS=1.
CI never sets it. Each test fetches a few days and checks the parser still
understands the response."""

from __future__ import annotations

import datetime as dt
import os

import pytest
import qp_data

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("QP_DATA_LIVE_TESTS") != "1", reason="live network tests are opt-in"),
]

KEYS = ["fred:DGS10", "ecb:EXR:USD", "ecb:ESTR", "frankfurter:USD", "nyfed:SOFR",
        "boe:IUDSOIA", "ust:par", "cboe:VIX", "coinbase:BTC-USD:1d", "binance:BTCUSDT:1d"]


@pytest.mark.parametrize("key", KEYS)
def test_live_source_parses(key):
    f, payload, _url = qp_data.fetch_frame(
        key, dt.date(2024, 7, 1), dt.date(2024, 7, 5), allow_network=True
    )
    assert len(f) > 0 and payload
