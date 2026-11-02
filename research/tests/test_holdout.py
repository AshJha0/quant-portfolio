"""Tests for the hold-out entry point. Synthetic data only: nothing here
reads real data dated on or after 2022-01-01, and the network is blocked."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "research" / "tools"), str(REPO / "research" / "studies"), str(REPO / "data" / "fetch")]

import fx_studies  # noqa: E402
import holdout as ho  # noqa: E402
import prereg  # noqa: E402
import qp_fetch  # noqa: E402
from models import HoldoutViolation  # noqa: E402
from qp_core.stats import deflated_sharpe_ratio  # noqa: E402

CLEAN = {"commit": "0" * 40, "dirty": False, "dirty_paths": []}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("network access in a test")
    monkeypatch.setattr(qp_fetch, "_download", refuse)


@pytest.fixture(scope="module")
def synth_px() -> pd.DataFrame:
    return fx_studies.usd_prices(qp_fetch.parse_ecb_csv(ho.synthetic_ecb_csv(seed=1)))


# ---- deflated Sharpe -----------------------------------------------------

def test_deflated_sharpe_matches_the_papers_worked_example():
    # Bailey and Lopez de Prado (2014), section 4: annualised SR 2.5 over
    # 1,250 daily returns, skewness -3, kurtosis 10, 100 trials whose
    # annualised Sharpe variance is 1/2. The paper reports E[max SR] = 0.1132
    # (per day) and DSR = 0.9004.
    out = ho.deflated_sharpe(sr=2.5 / math.sqrt(250), n_obs=1250, skew=-3.0, kurtosis=10.0,
                             n_trials=100, trial_sr_variance=0.5 / 250)
    assert out["sr0_per_period"] == pytest.approx(0.1132, abs=5e-5)
    assert out["dsr"] == pytest.approx(0.9004, abs=5e-4)


def test_deflated_sharpe_of_returns_matches_qp_core():
    r = np.random.default_rng(5).standard_normal(1000) * 0.01 + 0.0008
    ours = ho.deflated_sharpe_of(r, n_trials=3)
    assert ours["dsr"] == pytest.approx(deflated_sharpe_ratio(r, 3), rel=1e-12)
    assert ours["n_trials"] == 3


def test_deflated_sharpe_one_trial_has_zero_benchmark_and_falls_with_trials():
    one = ho.deflated_sharpe(0.05, 1000, 0.0, 3.0, 1)
    assert one["sr0_per_period"] == 0.0
    assert ho.deflated_sharpe(0.05, 1000, 0.0, 3.0, 3)["dsr"] < one["dsr"]


def test_deflated_sharpe_of_flat_series_is_none():
    assert ho.deflated_sharpe_of(np.zeros(100), 3)["dsr"] is None


# ---- folds: formation and fitting use only earlier data ------------------

@pytest.mark.parametrize("year", ho.HOLDOUT_YEARS)
def test_pairs_fold_masks_end_before_the_test_year(year):
    idx = pd.bdate_range("2012-01-02", "2025-12-31")
    form, test = fx_studies.pairs_fold_masks(idx, year)
    first_test = np.flatnonzero(test)[0]
    assert idx[test].year.unique().tolist() == [year]
    assert np.flatnonzero(form).max() == first_test - fx_studies.PAIRS["embargo_days"] - 1
    assert idx[form].min().year == year - fx_studies.PAIRS["formation_years"]
    assert not (form & test).any()


def test_pairs_fold_ignores_later_data(synth_px):
    full = fx_studies.pairs_study(synth_px, years=(2023,))
    cut = fx_studies.pairs_study(synth_px[synth_px.index <= "2023-12-31"], years=(2023,))
    scrambled = synth_px.copy()
    scrambled.loc[scrambled.index > "2023-12-31"] *= 1.5  # any change after the fold
    later = fx_studies.pairs_study(scrambled, years=(2023,))
    assert full["folds"] == cut["folds"] == later["folds"]
    assert full["sharpe"] == cut["sharpe"] == later["sharpe"]


def test_regime_fold_ignores_later_data(synth_px):
    full = fx_studies.regime_study(synth_px, years=(2022,))
    scrambled = synth_px.copy()
    scrambled.loc[scrambled.index > "2022-12-31"] *= 0.7
    assert fx_studies.regime_study(scrambled, years=(2022,)) == full


def test_training_mode_still_refuses_holdout_dates(synth_px):
    with pytest.raises(HoldoutViolation):
        fx_studies.pairs_study(synth_px)
    with pytest.raises(HoldoutViolation):
        fx_studies.regime_study(synth_px)


def test_holdout_prices_refuses_short_or_late_data(synth_px):
    eur = qp_fetch.parse_ecb_csv(ho.synthetic_ecb_csv(end="2025-06-30"))
    with pytest.raises(ho.HoldoutRefused, match="December 2025"):
        fx_studies.holdout_prices(eur)
    eur = qp_fetch.parse_ecb_csv(ho.synthetic_ecb_csv(start="2020-01-01"))
    with pytest.raises(ho.HoldoutRefused, match="starts after 2018"):
        fx_studies.holdout_prices(eur)


# ---- guards ---------------------------------------------------------------

REG = "# R\n\nbody\n\n## Signature\n\nsigned_by:\nsigned_on:\nsha256:\n"


def _sign(text: str) -> str:
    h = prereg.registration_hash(text)
    return text.replace("signed_by:", "signed_by: c").replace("signed_on:", "signed_on: 2026-11-02") \
               .replace("sha256:", f"sha256: {h}")


def test_signed_registration_refuses_unsigned_and_edited(tmp_path):
    f = tmp_path / "r.md"
    f.write_text(REG, newline="\n")
    h = prereg.registration_hash(REG)
    with pytest.raises(ho.HoldoutRefused, match="not signed"):
        ho.signed_registration(f, h)
    f.write_text(_sign(REG), newline="\n")
    assert ho.signed_registration(f, h).signed
    with pytest.raises(ho.HoldoutRefused, match="edited"):
        ho.signed_registration(f, "f" * 64)


def test_committed_registrations_match_the_code():
    for name, sha in fx_studies.SIGNED_SHA256.items():
        assert ho.signed_registration(REPO / "research" / "prereg" / name, sha).signed


def test_dirty_tree_is_refused_before_data_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(ho, "git_state", lambda _repo: {**CLEAN, "dirty": True, "dirty_paths": ["x"]})
    with pytest.raises(ho.HoldoutRefused, match="dirty"):
        fx_studies.run_holdout("all", tmp_path / "missing.csv", tmp_path, dry_run=False)


def test_unsigned_registration_is_refused_before_data_is_read(tmp_path, monkeypatch):
    regs = tmp_path / "prereg"
    shutil.copytree(REPO / "research" / "prereg", regs)
    text = (regs / "pairs-fx-g10.md").read_text(encoding="utf-8")
    (regs / "pairs-fx-g10.md").write_text(text.replace("sha256: dfca", "sha256: 0fca"), encoding="utf-8")
    monkeypatch.setattr(ho, "git_state", lambda _repo: CLEAN)
    with pytest.raises(ho.HoldoutRefused, match="not signed"):
        fx_studies.run_holdout("pairs", tmp_path / "missing.csv", tmp_path, False, prereg_dir=regs)


def test_existing_result_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(ho, "git_state", lambda _repo: CLEAN)
    target = tmp_path / "regime_fx_basket_holdout.json"
    target.write_text("{}")
    with pytest.raises(ho.HoldoutRefused, match="single-use"):
        fx_studies.run_holdout("regime", tmp_path / "missing.csv", tmp_path, dry_run=False)
    with pytest.raises(ho.HoldoutRefused, match="never"):
        ho.write_once(target, {"a": 1})
    assert target.read_text() == "{}"


def test_dry_run_refuses_the_results_directory():
    with pytest.raises(ho.HoldoutRefused):
        fx_studies.main(["all", "--dry-run", "--out", str(fx_studies.RESULTS)])


# ---- full dry run: provenance and the prereg check -------------------------

@pytest.fixture(scope="module")
def dry_run(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("dry")
    assert fx_studies.main(["all", "--dry-run", "--out", str(out)]) == 0
    return out


def test_dry_run_writes_both_results_with_provenance(dry_run):
    data = (dry_run / "synthetic_ecb_2012-2025.csv").read_bytes()
    for stem, metric in [("pairs_fx_g10", "decision"), ("regime_fx_basket", "decision")]:
        doc = json.loads((dry_run / f"{stem}_holdout_dryrun.json").read_text())
        prov = doc["provenance"]
        assert set(prereg.PROVENANCE_FIELDS) <= set(prov)
        assert prov["dry_run"] is True and len(prov["git_commit"]) == 40
        assert prov["input_sha256"] == {"synthetic_ecb_2012-2025.csv": ho.sha256_bytes(data)}
        assert prov["prereg_sha256"][doc["prereg"]] == doc["prereg_sha256"]
        assert doc["period"] == {"start": "2022-01-03", "end": "2025-12-31"}
        assert doc["mode"] == "holdout" and metric in doc
    pairs = json.loads((dry_run / "pairs_fx_g10_holdout_dryrun.json").read_text())
    assert [f["year"] for f in pairs["folds"]] == list(ho.HOLDOUT_YEARS)
    assert pairs["deflated_sharpe"]["n_trials"] == 3


def test_second_dry_run_is_refused(dry_run):
    with pytest.raises(ho.HoldoutRefused, match="single-use"):
        fx_studies.main(["regime", "--dry-run", "--out", str(dry_run)])


def test_prereg_check_refuses_dry_runs_and_dirty_trees_and_accepts_clean(dry_run, tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(REPO / "research" / "prereg", repo / "research" / "prereg")
    res = repo / "research" / "results"
    res.mkdir(parents=True)
    doc = json.loads((dry_run / "regime_fx_basket_holdout_dryrun.json").read_text())
    (res / "regime_fx_basket_holdout_dryrun.json").write_text(json.dumps(doc))
    assert any("dry-run" in e for e in prereg.check_result(res / "regime_fx_basket_holdout_dryrun.json", repo))
    doc["provenance"].update(dry_run=False, git_dirty=False)
    clean = res / "regime_fx_basket_holdout.json"
    clean.write_text(json.dumps(doc))
    assert prereg.check_result(clean, repo) == []
    doc["provenance"]["git_dirty"] = True
    clean.write_text(json.dumps(doc))
    assert any("dirty" in e for e in prereg.check_result(clean, repo))
    del doc["provenance"]
    clean.write_text(json.dumps(doc))
    assert prereg.check_result(clean, repo) == ["hold-out result has no provenance block"]
