"""Forward tracker: guards, idempotence, causality and parity with fx_studies.

All data here is synthetic, dated in the forward window; nothing touches the
network or the 2022-2025 hold-out.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "research" / "forward"), str(REPO / "research" / "studies")]

import forward as fw  # noqa: E402
import fx_studies  # noqa: E402

NOW = dt.datetime(2030, 1, 1, tzinfo=dt.UTC)


def synthetic_eur(start: str, end: str, seed: int = 7) -> pd.DataFrame:
    """EUR-quoted G10 rates on business days. GBP and CHF share a stochastic
    trend with USD so some pairs cointegrate."""
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    n = len(idx)
    common = np.cumsum(rng.normal(0, 0.004, n))
    vol = np.where((np.arange(n) // 120) % 2 == 0, 0.003, 0.009)  # alternating calm and stormy spells
    cols = {}
    for k, c in enumerate(fw.CURRENCIES):
        own = np.cumsum(rng.normal(0, 1, n) * vol)
        level = {"USD": 1.1, "JPY": 160.0, "SEK": 11.0, "NOK": 11.5}.get(c, 1.5)
        mix = common if c in ("USD", "GBP", "CHF") else 0.0
        noise = rng.normal(0, 0.002, n) if c in ("GBP", "CHF") else own
        cols[c] = level * np.exp(mix + noise + 0.0001 * k)
    return pd.DataFrame(cols, index=idx)[list(fw.CURRENCIES)]


def to_csv(eur: pd.DataFrame, path: Path) -> Path:
    rows = [(c, d.strftime("%Y-%m-%d"), v) for d, r in eur.iterrows() for c, v in r.items()]
    pd.DataFrame(rows, columns=["CURRENCY", "TIME_PERIOD", "OBS_VALUE"]).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- guards

def test_request_before_forward_start_is_refused():
    with pytest.raises(fw.ForwardGuardError):
        fw.check_request(dt.date(2026, 11, 2), dt.date(2026, 11, 10))
    with pytest.raises(fw.ForwardGuardError):
        fw.check_request(dt.date(2023, 6, 1), dt.date(2023, 6, 30))  # inside the hold-out
    with pytest.raises(fw.ForwardGuardError):
        fw.check_request(dt.date(2026, 11, 10), dt.date(2026, 11, 3))  # reversed window
    fw.check_request(fw.FORWARD_START, fw.FORWARD_START)  # the first allowed day passes


def test_fetch_refuses_before_any_download(monkeypatch):
    called = []
    monkeypatch.setattr(fw.qp_fetch, "_download", lambda url: called.append(url) or b"")
    with pytest.raises(fw.ForwardGuardError):
        fw.fetch(dt.date(2025, 12, 31), dt.date(2026, 11, 5))
    assert called == []


def test_rows_before_forward_start_are_refused(tmp_path):
    eur = synthetic_eur("2025-12-29", "2026-11-06")
    with pytest.raises(fw.ForwardGuardError):
        fw.check_rows(eur)
    with pytest.raises(fw.ForwardGuardError):
        fw.read_csv(to_csv(eur, tmp_path / "bad.csv"))
    with pytest.raises(fw.ForwardGuardError):
        fw.append(tmp_path / "log.jsonl", eur, "x", "test", now=NOW)
    assert not (tmp_path / "log.jsonl").exists()


def test_fetch_refuses_a_response_with_old_rows(monkeypatch, tmp_path):
    body = to_csv(synthetic_eur("2026-11-02", "2026-11-04"), tmp_path / "r.csv").read_bytes()
    monkeypatch.setattr(fw.qp_fetch, "_download", lambda url: body)
    with pytest.raises(fw.ForwardGuardError):
        fw.fetch(dt.date(2026, 11, 3), dt.date(2026, 11, 4))


def test_parameters_match_the_registered_study_code():
    for k, v in fw.PAIRS.items():
        assert fx_studies.PAIRS[k] == v, k
    for k, v in fw.REGIME.items():
        assert fx_studies.REGIME[k] == v, k
    assert fx_studies.REGIME["first_test_year"] - 1999 == fw.REGIME_HISTORY_YEARS


# ---------------------------------------------------------------- log

def test_append_is_idempotent_and_first_value_wins(tmp_path):
    log = tmp_path / "log.jsonl"
    eur = synthetic_eur("2026-11-03", "2026-11-20")
    first = fw.append(log, eur, "a" * 64, "test", now=NOW)
    assert len(first.appended) == len(eur) and first.skipped == []
    before = log.read_bytes()
    again = fw.append(log, eur * 1.01, "b" * 64, "test", now=NOW)  # a "revision"
    assert again.appended == [] and len(again.skipped) == len(eur)
    assert log.read_bytes() == before


def test_record_fields_and_provenance(tmp_path):
    log = tmp_path / "log.jsonl"
    fw.append(log, synthetic_eur("2026-11-03", "2026-11-05"), "c" * 64, "file:x.csv", now=NOW)
    recs = fw.read_log(log)
    assert [r["date"] for r in recs] == ["2026-11-03", "2026-11-04", "2026-11-05"]
    r = recs[0]
    assert r["schema"] == fw.SCHEMA
    assert set(r["rates_per_eur"]) == set(fw.CURRENCIES)
    assert set(r["usd_per_unit"]) == set(fw.CURRENCIES) - {"USD"} | {"EUR"}
    assert r["regime"] == {"status": "warming_up", "scorable_from_year": 2031}
    assert r["pairs"] == {"status": "warming_up", "scorable_from_year": 2030}
    prov = r["provenance"]
    assert prov["data_sha256"] == "c" * 64 and prov["written_utc"] == "2030-01-01T00:00:00Z"
    assert set(prov["prereg_sha256"]) == set(fw.PREREG_FILES)
    assert prov["prereg_sha256"]["pairs-fx-g10.md"] == fx_studies.SIGNED_SHA256["pairs-fx-g10.md"]


def test_next_start(tmp_path):
    log = tmp_path / "log.jsonl"
    assert fw.next_start(log) == fw.FORWARD_START
    fw.append(log, synthetic_eur("2026-11-03", "2026-11-06"), "d", "t", now=NOW)
    assert fw.next_start(log) == dt.date(2026, 11, 7)


def test_cli_reads_csv(tmp_path, capsys):
    csv = to_csv(synthetic_eur("2026-11-03", "2026-11-10"), tmp_path / "e.csv")
    log = tmp_path / "log.jsonl"
    assert fw.main(["update", "--log", str(log), "--csv", str(csv)]) == 0
    assert fw.main(["update", "--log", str(log), "--csv", str(csv)]) == 0
    assert len(fw.read_log(log)) == 6
    assert "appended 0 day(s)" in capsys.readouterr().out


def test_cli_with_a_current_log_does_not_fetch(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    fw.append(log, synthetic_eur("2026-11-03", "2026-11-06"), "d", "t", now=NOW)
    monkeypatch.setattr(fw.qp_fetch, "_download", lambda url: pytest.fail("fetched"))
    assert fw.main(["update", "--log", str(log), "--today", "2026-11-06"]) == 0


# ---------------------------------------------------------------- rules

def test_first_scorable_years():
    assert fw.pairs_first_year(pd.Timestamp("2026-11-03")) == 2030
    assert fw.regime_first_year(pd.Timestamp("2026-11-03")) == 2031
    assert fw.pairs_first_year(pd.Timestamp("2027-01-04")) == 2030  # first business day counts as whole
    assert fw.regime_first_year(pd.Timestamp("2027-01-11")) == 2032


def test_pair_path_matches_fx_studies_pair_returns():
    px = fw.usd_prices(synthetic_eur("2026-11-03", "2030-12-31"))
    logp = np.log(px)
    idx = pd.DatetimeIndex(logp.index)
    form, test = fw.fold_masks(idx, 2030)
    f2, t2 = fx_studies.pairs_fold_masks(idx, 2030)
    assert (form == f2).all() and (test == t2).all()
    w = form | test
    for a, b in [("GBP", "CHF"), ("EUR", "GBP"), ("JPY", "NOK")]:
        mine = fw.pair_path(logp[a].to_numpy()[w], logp[b].to_numpy()[w], form[w], test[w])
        ref = fx_studies.pair_returns(logp[a][w], logp[b][w], test[w], fx_studies.PAIRS)
        np.testing.assert_array_equal(mine["net"][test[w]], ref)


@pytest.fixture(scope="module")
def long_run(tmp_path_factory):
    """Synthetic forward log through March 2031, written in two sittings."""
    d = tmp_path_factory.mktemp("fwd")
    eur = synthetic_eur("2026-11-03", "2031-03-31")
    log = d / "log.jsonl"
    cut = pd.Timestamp("2030-06-28")
    fw.append(log, eur[eur.index <= cut], "s1", "t", now=NOW)
    fw.append(log, eur[eur.index > cut], "s2", "t", now=NOW)
    return eur, fw.read_log(log)


def test_pairs_are_scored_from_2030_and_match_the_study(long_run):
    eur, recs = long_run
    by = {r["date"]: r for r in recs}
    assert by["2029-12-31"]["pairs"]["status"] == "warming_up"
    y30 = [r for r in recs if r["date"].startswith("2030")]
    assert all(r["pairs"]["status"] == "scored" for r in y30)
    assert len({tuple(r["pairs"]["selected"]) for r in y30}) == 1  # one selection per year
    assert y30[0]["pairs"]["selected"], "the synthetic GBP/CHF pair should be selected"
    assert any(r["pairs"]["held"] for r in y30)
    # the daily net returns summed over 2030 equal the study's walk-forward fold
    px = fx_studies.usd_prices(eur.loc[eur.index <= "2030-12-31", list(fw.ORDER)])  # parse_ecb_csv order
    study = fx_studies.pairs_study(px, years=(2030,))
    mine = sum(r["pairs"]["net_return"] for r in y30)
    assert y30[0]["pairs"]["selected"] == study["folds"][0]["pairs"]
    assert mine == pytest.approx(study["folds"][0]["net_return"], abs=1e-12)


def test_regime_is_scored_from_2031_and_is_causal(long_run):
    eur, recs = long_run
    by = {r["date"]: r for r in recs}
    assert by["2030-12-31"]["regime"]["status"] == "warming_up"
    y31 = [r for r in recs if r["date"].startswith("2031")]
    assert all(r["regime"]["status"] == "scored" for r in y31)
    assert {r["regime"]["fit_through"] for r in y31} == {"2030-12-31"}
    assert all(0.0 <= r["regime"]["p_high"] <= 1.0 for r in y31)
    # recomputing one day with all later data removed gives the same answer
    day = pd.Timestamp(y31[20]["date"])
    px = fw.usd_prices(eur.loc[eur.index <= day, list(fw.ORDER)])
    again = fw.regime_signal(px, day)
    assert again["p_high"] == pytest.approx(y31[20]["regime"]["p_high"], abs=1e-12)
    assert again["high_vol"] == y31[20]["regime"]["high_vol"]


def test_input_column_order_does_not_matter(tmp_path):
    eur = synthetic_eur("2026-11-03", "2026-11-12")
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    fw.append(a, eur, "x", "t", now=NOW)
    fw.append(b, eur[eur.columns[::-1]], "x", "t", now=NOW)
    assert a.read_bytes() == b.read_bytes()


def test_two_sittings_equal_one(long_run, tmp_path):
    """Writing the log in two runs gives the same signals as one run (no look-ahead, no state drift)."""
    eur, recs = long_run
    part = eur[eur.index <= "2030-07-31"]
    log = tmp_path / "one.jsonl"
    fw.append(log, part, "s1", "t", now=NOW)
    one = fw.read_log(log)
    two = recs[: len(one)]
    for a, b in zip(one, two, strict=True):
        assert a["date"] == b["date"]
        assert a["pairs"] == b["pairs"] and a["regime"] == b["regime"]
    assert json.dumps(one[-1]["pairs"])  # serialisable
