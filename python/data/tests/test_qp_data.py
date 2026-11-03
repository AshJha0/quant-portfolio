"""Offline tests for qp_data. Recorded responses live in ``fixtures/``
(downloaded 2026-11-03 from the public endpoints named in each test).
Nothing here touches the network; ``test_live.py`` does, opt-in only."""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest
import qp_data
from qp_data import release, sources
from qp_data.errors import (
    FetchError,
    NetworkDisabled,
    NoVintage,
    SchemaError,
    SourceNotPermitted,
    UnknownReleaseRule,
)
from qp_data.net import download
from qp_data.store import Store

FIX = Path(__file__).parent / "fixtures"
D = dt.date


def fx(name: str) -> bytes:
    return (FIX / name).read_bytes()


def recorded(name: str):
    seen: list[str] = []

    def opener(url: str, timeout: float) -> bytes:
        seen.append(url)
        return fx(name)

    opener.seen = seen  # type: ignore[attr-defined]
    return opener


# ------------------------------------------------------------------ net --
def test_offline_by_default(monkeypatch):
    monkeypatch.delenv("QP_DATA_ALLOW_NETWORK", raising=False)
    with pytest.raises(NetworkDisabled, match="disabled"):
        download("https://example.org")


def test_retry_with_exponential_backoff():
    waits: list[float] = []
    n = {"calls": 0}

    def flaky(url, timeout):
        n["calls"] += 1
        if n["calls"] <= 2:
            raise urllib.error.HTTPError(url, 503, "busy", {}, None)  # type: ignore[arg-type]
        return b"x"

    assert download("u", opener=flaky, backoff=0.5, sleep=waits.append) == b"x"
    assert waits == [0.5, 1.0]


def test_no_retry_on_client_error():
    n = {"calls": 0}

    def gone(url, timeout):
        n["calls"] += 1
        raise urllib.error.HTTPError(url, 404, "nope", {}, None)  # type: ignore[arg-type]

    with pytest.raises(FetchError, match="permanent"):
        download("u", opener=gone, sleep=lambda s: None)
    assert n["calls"] == 1


def test_gives_up_after_retries():
    waits: list[float] = []

    def down(url, timeout):
        raise TimeoutError("t")

    with pytest.raises(FetchError, match="4 attempts"):
        download("u", opener=down, retries=3, backoff=1.0, sleep=waits.append)
    assert waits == [1.0, 2.0, 4.0]


def test_429_is_retried():
    calls = []

    def limited(url, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 429, "slow down", {}, None)  # type: ignore[arg-type]
        return b"ok"

    assert download("u", opener=limited, sleep=lambda s: None) == b"ok"


# --------------------------------------------------------------- parsers --
def test_fred_csv_keyless_empty_cell_is_missing():
    # fredgraph.csv, id=DGS10, 2024-06-24..2024-07-08; 4 July holiday is an empty cell
    f = sources.parse_fred_csv(fx("fred_DGS10.csv"), "DGS10")
    assert len(f) == 10
    assert pd.Timestamp("2024-07-04") not in f.index
    assert f.loc["2024-07-01", "value"] == 4.48


def test_fred_old_dot_convention():
    f = sources.parse_fred_csv(b"DATE,DEXUSEU\n2021-01-01,.\n2021-01-04,1.2296\n")
    assert list(f["value"]) == [1.2296]


def test_fred_wrong_series_and_layout():
    with pytest.raises(SchemaError):
        sources.parse_fred_csv(fx("fred_DGS10.csv"), "DGS2")
    with pytest.raises(SchemaError):
        sources.parse_fred_csv(b"a,b,c\n1,2,3\n")


def test_fred_restricted_series_refused():
    with pytest.raises(SourceNotPermitted):
        sources.build_url("fred:SP500")


def test_fred_url_is_keyless():
    url = sources.build_url("fred:DGS10", D(2024, 1, 1), D(2024, 2, 1))
    assert "api_key" not in url and "fredgraph.csv?id=DGS10&cosd=2024-01-01&coed=2024-02-01" in url


def test_ecb_exr():
    f = sources.parse_ecb_csv(fx("ecb_exr_USD_JPY.csv"))
    assert list(f.columns) == ["JPY", "USD"]
    assert f.loc["2024-07-01", "JPY"] == 173.15
    assert len(f) == 5


def test_ecb_estr_single_column():
    f = sources.parse_ecb_csv(fx("ecb_estr.csv"))
    assert list(f.columns) == ["value"]
    assert f.loc["2024-07-01", "value"] == 3.665


def test_ecb_bad_layout():
    with pytest.raises(SchemaError):
        sources.parse_ecb_csv(b"A,B\n1,2\n")


def test_frankfurter():
    f = sources.parse_frankfurter(fx("frankfurter_USD_EUR_JPY.json"))
    assert f.loc["2024-07-01", "EUR"] == 0.93067
    with pytest.raises(SchemaError):
        sources.parse_frankfurter(b'{"rates": {}}')


def test_nyfed_sofr_sorted_ascending():
    f = sources.parse_nyfed_sofr(fx("nyfed_sofr.json"))
    assert f.index.is_monotonic_increasing
    assert f.loc["2024-07-01", "value"] == 5.40
    assert f.loc["2024-07-05", "volume_bn"] == 2126


def test_boe_sonia():
    f = sources.parse_boe_csv(fx("boe_sonia.csv"))
    assert f.loc["2024-07-01", "value"] == 5.2 and len(f) == 5


def test_ust_par_tenor_columns():
    f = sources.parse_ust_par(fx("ust_par.csv"))
    assert "0.0833333" in f.columns and "10" in f.columns
    assert f.loc["2024-07-01", "10"] == 4.48  # agrees with FRED DGS10 for the same day


def test_cboe_vix():
    f = sources.parse_cboe_vix(fx("cboe_vix.csv"))
    assert f.loc["2024-07-01", "close"] == 12.22


def test_coinbase_candles_reordered():
    f = sources.parse_coinbase_candles(fx("coinbase_BTC-USD_1d.json"))
    assert f.index.is_monotonic_increasing
    # row is [time, low, high, open, close, volume]; open must not be swapped with low
    assert f.loc["2024-07-01", "open"] == 62669.14 and f.loc["2024-07-01", "low"] == 62448.46
    with pytest.raises(SchemaError):
        sources.parse_coinbase_candles(b"[[1,2,3]]")


def test_binance_klines():
    f = sources.parse_binance_klines(fx("binance_BTCUSDT_1d.json"))
    assert f.loc["2024-07-01", "close"] == 62899.99
    assert f["open"].dtype == float


def test_unknown_source():
    with pytest.raises(KeyError, match="unknown source"):
        sources.build_url("nosuchsource:FOO")


# --------------------------------------------------------------- release --
@pytest.mark.parametrize(
    ("key", "obs", "expect"),
    [
        ("fred:DGS10", D(2024, 7, 1), D(2024, 7, 2)),
        ("fred:DGS10", D(2024, 7, 3), D(2024, 7, 5)),  # 4 July
        ("fred:DGS10", D(2024, 7, 5), D(2024, 7, 8)),  # Friday -> Monday
        ("fred:DEXUSEU", D(2024, 7, 1), D(2024, 7, 8)),  # weekly H.10
        ("fred:DEXUSEU", D(2024, 7, 5), D(2024, 7, 8)),
        ("fred:DEXUSEU", D(2024, 1, 10), D(2024, 1, 16)),  # MLK Monday -> Tuesday
        ("fred:CPIAUCSL", D(2024, 3, 1), D(2024, 4, 22)),
        ("fred:CPIAUCSL", D(2025, 9, 1), D(2025, 10, 24)),  # shutdown override
        ("fred:CPIAUCSL", D(2024, 12, 1), D(2025, 1, 21)),  # Jan 20 2025 is MLK day
        ("fred:FEDFUNDS", D(2024, 6, 1), D(2024, 7, 1)),
        ("fred:FEDFUNDS", D(2024, 8, 1), D(2024, 9, 3)),  # 1 Sep Sunday, 2 Sep Labor Day
        ("nyfed:SOFR", D(2024, 7, 3), D(2024, 7, 5)),
        ("ecb:ESTR", D(2024, 3, 28), D(2024, 4, 2)),  # Good Friday and Easter Monday
        ("boe:IUDSOIA", D(2024, 5, 3), D(2024, 5, 7)),  # early May bank holiday
        ("ecb:EXR:USD", D(2024, 7, 1), D(2024, 7, 1)),
        ("ust:par", D(2024, 7, 1), D(2024, 7, 1)),
        ("binance:BTCUSDT:1d", D(2024, 7, 6), D(2024, 7, 7)),
    ],
)
def test_release_dates(key, obs, expect):
    assert release.release_date(key, obs) == expect


def test_unknown_release_rule_blocks_fetch():
    with pytest.raises(UnknownReleaseRule):
        release.release_date("fred:UNRATE", D(2024, 1, 1))
    with pytest.raises(UnknownReleaseRule):
        qp_data.fetch("fred:UNRATE", opener=recorded("fred_DGS10.csv"))


def test_check_release_catches_look_ahead():
    release.check_release("fred:DGS10", [D(2024, 7, 1)], [D(2024, 7, 2)])
    with pytest.raises(ValueError, match="before release"):
        release.check_release("fred:DGS10", [D(2024, 7, 1)], [D(2024, 7, 1)])
    # the in-month use of a monthly average (plan finding quant G2)
    with pytest.raises(ValueError, match="FEDFUNDS"):
        release.check_release("fred:FEDFUNDS", [D(2024, 6, 1)], [D(2024, 6, 28)])


# ----------------------------------------------------------------- store --
@pytest.fixture
def store(tmp_path):
    return Store(tmp_path, fmt="csv")


def test_fetch_stores_bytes_manifest_and_release(store):
    e = qp_data.fetch(
        "fred:DGS10", D(2024, 6, 24), D(2024, 7, 8), store=store,
        opener=recorded("fred_DGS10.csv"), retrieved_at="2024-07-09T12:00:00Z",
    )
    assert e.n_rows == 10 and e.source == "fred" and e.release_date == "2024-07-09"
    assert store.get_bytes(e.sha256) == fx("fred_DGS10.csv")
    assert "cosd=2024-06-24" in e.url
    line = json.loads(store.manifest_path.read_text().splitlines()[0])
    assert {"source", "url", "retrieved_at", "as_of", "release_date", "schema_version"} <= set(line)


def test_same_bytes_not_duplicated(store):
    for _ in range(2):
        qp_data.fetch("fred:DGS10", store=store, opener=recorded("fred_DGS10.csv"))
    assert len(store.manifest("fred:DGS10")) == 1


def test_as_of_hides_unreleased_observations(store):
    qp_data.fetch("fred:DGS10", store=store, opener=recorded("fred_DGS10.csv"),
                  retrieved_at="2024-06-01T00:00:00Z")
    # Morning of 2 July: 1 July (released 2 July) is known, 2 July is not
    f = store.get("fred:DGS10", as_of=D(2024, 7, 2))
    assert f.index.max() == pd.Timestamp("2024-07-01")


def _cpi(store, value_march: float, retrieved: str) -> None:
    body = f"observation_date,CPIAUCSL\n2024-02-01,310.967\n2024-03-01,{value_march}\n".encode()
    qp_data.fetch("fred:CPIAUCSL", store=store, opener=lambda u, t: body, retrieved_at=retrieved)


def test_as_of_returns_value_known_then_after_revision(store):
    _cpi(store, 312.332, "2024-04-30T00:00:00Z")  # first print
    _cpi(store, 312.345, "2024-06-30T00:00:00Z")  # revised
    early = store.get("fred:CPIAUCSL", as_of="2024-05-15")
    late = store.get("fred:CPIAUCSL", as_of="2024-07-01")
    assert early.loc["2024-03-01", "value"] == 312.332
    assert late.loc["2024-03-01", "value"] == 312.345
    assert store.get("fred:CPIAUCSL").loc["2024-03-01", "value"] == 312.345
    # before the March release date (22 April) March is absent in either vintage
    assert pd.Timestamp("2024-03-01") not in store.get("fred:CPIAUCSL", as_of="2024-04-21").index


def test_as_of_before_first_vintage(store):
    _cpi(store, 312.345, "2024-06-30T00:00:00Z")
    f = store.get("fred:CPIAUCSL", as_of="2024-05-15")
    assert f.attrs["revision_risk"] is True
    with pytest.raises(NoVintage):
        store.get("fred:CPIAUCSL", as_of="2024-05-15", strict=True)
    with pytest.raises(NoVintage):
        store.get("fred:DGS2")


def test_corrupt_object_detected(store):
    e = qp_data.fetch("fred:DGS10", store=store, opener=recorded("fred_DGS10.csv"))
    store.object_path(e.sha256).write_bytes(b"tampered")
    with pytest.raises(SchemaError, match="corrupt"):
        store.get_bytes(e.sha256)


def test_parquet_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    st = Store(tmp_path, fmt="parquet")
    qp_data.fetch("ecb:EXR:USD", store=st, opener=recorded("ecb_exr_USD_JPY.csv"))
    assert st.get("ecb:EXR:USD").loc["2024-07-01", "JPY"] == 173.15


def test_every_source_has_a_release_rule_for_its_example_key():
    for key in ["fred:DGS10", "ecb:EXR:USD", "ecb:ESTR", "frankfurter:USD", "nyfed:SOFR",
                "boe:IUDSOIA", "ust:par", "cboe:VIX", "coinbase:BTC-USD:1d", "binance:BTCUSDT:1d"]:
        release.rule_for(key)
        sources.build_url(key, D(2024, 7, 1), D(2024, 7, 5))


def test_frankfurter_latest_layout():
    body = b'{"amount":1.0,"base":"EUR","date":"2024-07-05","rates":{"USD":1.0826,"JPY":174.06}}'
    f = sources.parse_frankfurter(body)
    assert f.loc["2024-07-05", "USD"] == 1.0826 and len(f) == 1


# ------------------------------------------- deprecated live.py wrappers --
# Each wrapper is run against a recorded response through the real download
# path (the HTTP call itself is replaced), and checked against the numbers
# the old code computed from the same payload.
@pytest.fixture
def fake_http(monkeypatch):
    import qp_data.net as net

    served: list[str] = []

    def serve(name: str) -> None:
        def _open(url: str, timeout: float) -> bytes:
            served.append(url)
            return fx(name)

        monkeypatch.setattr(net, "_urlopen", _open)

    monkeypatch.setenv("QP_DATA_ALLOW_NETWORK", "1")
    serve.served = served  # type: ignore[attr-defined]
    return serve


def test_fx_pairs_legs_wrapper(fake_http):
    live = pytest.importorskip("fx_pairs.data.live")
    fake_http("frankfurter_USD_EUR_JPY.json")
    with pytest.warns(DeprecationWarning, match="v4.0.0"):
        legs = live.fetch_frankfurter_legs(("EUR", "JPY"), "2024-07-01", "2024-07-05")
    assert legs.loc["2024-07-01", "EUR"] == pytest.approx(1 / 0.93067, rel=1e-15)
    assert legs.index.name == "date"
    assert "2024-07-01..2024-07-05?base=USD&symbols=EUR,JPY" in fake_http.served[0]


def test_fx_port_and_fx_var_wrappers(fake_http, monkeypatch):
    port = pytest.importorskip("fx_port.data.live")
    var = pytest.importorskip("fx_var.data.live")
    fake_http("frankfurter_USD_EUR_JPY.json")
    with pytest.warns(DeprecationWarning):
        a = port.fetch_spots(["EUR", "JPY"], "2024-07-01", "2024-07-05")
    with pytest.warns(DeprecationWarning):
        b = var.load_frankfurter(["EUR", "JPY"], "2024-07-01", "2024-07-05", allow_network=True)
    assert a.loc["2024-07-03", "JPY"] == pytest.approx(1 / 161.91, rel=1e-15)
    pd.testing.assert_frame_equal(a, b)


def test_fx_vol_wrapper(fake_http):
    pytest.importorskip("fx_vol")
    from fx_vol.data import live

    fake_http("frankfurter_USD_EUR_JPY.json")
    with pytest.warns(DeprecationWarning):
        s = live.load_frankfurter("USDJPY", "2024-07-01", "2024-07-05")
    assert s.name == "USDJPY" and s.iloc[0] == 161.14 and len(s) == 5


def test_fx_options_wrapper(fake_http):
    live = pytest.importorskip("fx_options.data.live")
    fake_http("frankfurter_USD_EUR_JPY.json")
    with pytest.warns(DeprecationWarning):
        r = live.fetch_ecb_rates("USD", ("EUR", "JPY"))
    assert r["EUR"] == 0.92387 and r["_date"] == "2024-07-05"  # last fixing in the payload


def test_fx_rates_fred_wrapper_is_keyless(fake_http, monkeypatch):
    live = pytest.importorskip("fx_rates.data.live")
    monkeypatch.setenv("FX_RATES_ALLOW_NETWORK", "1")
    fake_http("fred_DGS10.csv")
    with pytest.warns(DeprecationWarning):
        obs = live.load_fred_series("DGS10")
    assert obs[0] == ("2024-07-08", 4.28) and len(obs) == 10
    assert "api_key" not in fake_http.served[0]


def test_fi_rates_cmt_wrapper(fake_http):
    live = pytest.importorskip("fi_rates.data.live")
    fake_http("fred_DGS10.csv")  # same body for every tenor: shape check only
    with pytest.warns(DeprecationWarning):
        df = live.load_fred_cmt(allow_network=True, start="2024-06-24")
    assert list(df.columns) == sorted(live.FRED_CMT_SERIES.values())
    assert df.loc["2024-07-01", 10.0] == 4.48


def test_fi_rates_refuses_without_opt_in(monkeypatch):
    live = pytest.importorskip("fi_rates.data.live")
    monkeypatch.delenv("QP_DATA_ALLOW_NETWORK", raising=False)
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        live.load_fred_cmt()
