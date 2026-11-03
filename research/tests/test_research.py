"""Tests for the Phase 4 research tooling. No network: fetchers are tested
on their parsers with inline fixtures."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "research" / "tools"), str(REPO / "research" / "studies"), str(REPO / "data" / "fetch")]

import prereg  # noqa: E402
import qp_fetch  # noqa: E402
from models import HoldoutViolation, johansen_trace, kalman_hedge, training_only  # noqa: E402

REG_TEXT = "# R\n\nbody\n\n## Signature\n\nsigned_by:\nsigned_on:\nsha256:\n"


# ---- registration hash and the CI check ---------------------------------

def _repo(tmp_path: Path, reg_text: str = REG_TEXT) -> Path:
    (tmp_path / "research" / "prereg").mkdir(parents=True)
    (tmp_path / "research" / "results").mkdir()
    (tmp_path / "research" / "prereg" / "r.md").write_text(reg_text, newline="\n")
    return tmp_path


def _result(repo: Path, sha: str, end: str) -> Path:
    p = repo / "research" / "results" / "x.json"
    p.write_text(json.dumps({"prereg": "research/prereg/r.md", "prereg_sha256": sha,
                             "period": {"start": "1999-01-04", "end": end}}))
    return p


def test_hash_ignores_signature_and_line_endings():
    signed = REG_TEXT.replace("signed_by:", "signed_by: A").replace("sha256:", "sha256: x")
    h = prereg.registration_hash(REG_TEXT)
    assert prereg.registration_hash(signed) == h
    assert prereg.registration_hash(REG_TEXT.replace("\n", "\r\n")) == h
    assert prereg.registration_hash(REG_TEXT.replace("body", "body.")) != h


def test_missing_signature_heading_is_an_error():
    with pytest.raises(ValueError):
        prereg.registration_hash("# no heading\n")


def test_training_result_with_matching_hash_passes(tmp_path):
    repo = _repo(tmp_path)
    h = prereg.registration_hash(REG_TEXT)
    assert prereg.check_result(_result(repo, h, "2021-12-31"), repo) == []


def test_hash_mismatch_is_refused(tmp_path):
    repo = _repo(tmp_path)
    errs = prereg.check_result(_result(repo, "0" * 64, "2021-12-31"), repo)
    assert any("mismatch" in e for e in errs)


def test_holdout_result_refused_until_signed(tmp_path):
    repo = _repo(tmp_path)
    h = prereg.registration_hash(REG_TEXT)
    assert any("hold-out" in e for e in prereg.check_result(_result(repo, h, "2022-01-03"), repo))
    signed = REG_TEXT.replace("signed_by:", "signed_by: coordinator").replace(
        "signed_on:", "signed_on: 2026-11-03").replace("sha256:", f"sha256: {h}")
    (repo / "research" / "prereg" / "r.md").write_text(signed, newline="\n")
    assert prereg.check_result(_result(repo, h, "2025-12-31"), repo) == []


def test_signature_with_wrong_hash_does_not_count(tmp_path):
    bad = REG_TEXT.replace("signed_by:", "signed_by: c").replace(
        "signed_on:", "signed_on: 2026-11-03").replace("sha256:", "sha256: deadbeef")
    repo = _repo(tmp_path, bad)
    h = prereg.registration_hash(bad)
    assert prereg.check_result(_result(repo, h, "2023-06-30"), repo)


def test_cli_exit_codes(tmp_path):
    repo = _repo(tmp_path)
    _result(repo, "0" * 64, "2021-12-31")
    assert prereg.check_dir(repo / "research" / "results", repo)


def test_committed_preregs_are_drafts_and_results_pass():
    regs = sorted((REPO / "research" / "prereg").glob("*.md"))
    assert {p.name for p in regs} >= {"pairs-fx-g10.md", "regime-fx-basket.md"}
    for p in regs:
        prereg.load(p)  # every file has a Signature heading
    results = REPO / "research" / "results"
    if results.exists():
        assert prereg.check_dir(results) == {}


def test_study_parameters_match_prereg_text():
    import fx_studies

    pairs = (REPO / "research" / "prereg" / "pairs-fx-g10.md").read_text()
    p = fx_studies.PAIRS
    for needle in [f"delta = {p['delta']:.0e}".replace("e-0", "e-"), f"|z| > {p['entry_z']}",
                   f"|z| < {p['exit_z']}", f"{p['formation_years']} calendar years",
                   f"{p['embargo_days']} business days", f"{p['cost_bp']:g} bp"]:
        assert needle in pairs, needle
    regime = (REPO / "research" / "prereg" / "regime-fx-basket.md").read_text()
    r = fx_studies.REGIME
    assert f"seed {r['seed']}, {r['n_init']} restarts" in regime
    assert f"above {r['threshold']}" in regime
    assert f"First test year in training: {r['first_test_year']}" in regime


# ---- hold-out guard ------------------------------------------------------

def test_training_only_guard():
    ok = pd.Series([1.0], index=pd.to_datetime(["2021-12-31"]))
    assert training_only(ok) is ok
    with pytest.raises(HoldoutViolation):
        training_only(pd.Series([1.0, 2.0], index=pd.to_datetime(["2021-12-31", "2022-01-03"])))


# ---- Johansen: oracle is statsmodels ------------------------------------

@pytest.mark.parametrize("k", [0, 1, 3])
def test_johansen_matches_statsmodels(k):
    vecm = pytest.importorskip("statsmodels.tsa.vector_ar.vecm")
    rng = np.random.default_rng(7 + k)
    w = np.cumsum(rng.standard_normal(500))
    x = np.column_stack([w + rng.standard_normal(500), 0.5 * w + rng.standard_normal(500)])
    ours = johansen_trace(x, k_ar_diff=k)
    ref = vecm.coint_johansen(x, 0, k)
    np.testing.assert_allclose(ours.trace, ref.lr1, rtol=1e-9)
    np.testing.assert_allclose(ours.eigenvalues, ref.eig, rtol=1e-9)
    np.testing.assert_allclose(ours.cv95, ref.cvt[:, 1])


def test_johansen_rank_on_cointegrated_and_independent_walks():
    rng = np.random.default_rng(1)
    w = np.cumsum(rng.standard_normal(1000))
    assert johansen_trace(np.column_stack([w + rng.standard_normal(1000), w])).rank == 1
    walks = np.cumsum(rng.standard_normal((1000, 2)), axis=0)
    assert johansen_trace(walks).rank == 0


# ---- Kalman: delta = 0 equals recursive least squares --------------------

def test_kalman_delta_zero_is_recursive_ols():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(300).cumsum()
    y = 0.3 + 1.7 * x + 0.1 * rng.standard_normal(300)
    kf = kalman_hedge(y, x, delta=0.0, obs_var=0.01, prior_var=1e6)  # ridge obs_var/prior = 1e-8; a 1e10 prior loses digits in P
    for t in (10, 100, 299):
        a = np.column_stack([np.ones(t + 1), x[: t + 1]])
        coef, *_ = np.linalg.lstsq(a, y[: t + 1], rcond=None)
        np.testing.assert_allclose([kf.alpha[t], kf.beta[t]], coef, rtol=1e-6, atol=1e-6)


def test_kalman_tracks_a_moving_beta():
    rng = np.random.default_rng(4)
    n = 2000
    x = rng.standard_normal(n).cumsum() + 50
    beta = np.linspace(0.5, 1.5, n)
    y = beta * x + 0.05 * rng.standard_normal(n)
    kf = kalman_hedge(y, x, delta=1e-5, obs_var=0.0025)
    assert abs(kf.beta[-1] - 1.5) < 0.05


# ---- fetcher parsers ----------------------------------------------------

def test_parse_fred_csv_drops_missing():
    s = qp_fetch.parse_fred_csv("observation_date,DEXUSEU\n2021-01-01,.\n2021-01-04,1.2296\n", "DEXUSEU")
    assert list(s.values) == [1.2296] and s.index[0] == pd.Timestamp("2021-01-04")


def test_fred_refuses_restricted_series(tmp_path):
    with pytest.raises(qp_fetch.SourceNotPermitted):
        qp_fetch.fetch_fred("SP500", raw_dir=tmp_path)


def test_parse_ecb_csv_pivots():
    text = ("KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n"
            "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2021-12-30,1.1300\n"
            "EXR.D.JPY.EUR.SP00.A,D,JPY,EUR,SP00,A,2021-12-30,130.0\n"
            "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2021-12-31,1.1326\n")
    df = qp_fetch.parse_ecb_csv(text)
    assert list(df.columns) == ["JPY", "USD"] and df.loc["2021-12-31", "USD"] == 1.1326
    assert np.isnan(df.loc["2021-12-31", "JPY"])


def test_raw_data_is_gitignored():
    gi = (REPO / "data" / ".gitignore").read_text()
    assert re.search(r"^/?raw/?$", gi, re.M)
