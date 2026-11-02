"""Tests for the trials log, selection calibration and cost/event stress. No network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "research" / "tools"), str(REPO / "research" / "studies"), str(REPO / "data" / "fetch")]

import forward_studies as fs  # noqa: E402
import prereg  # noqa: E402
import robustness as rb  # noqa: E402
import trials  # noqa: E402

CCY = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK"]


def _walks(n: int, seed: int, start: str = "2004-01-01", coint: bool = False) -> pd.DataFrame:
    """Nine independent random walks; with ``coint`` currencies 0-3 share one
    trend and 4-6 another, each off its trend by a fast AR(1) spread, so 9
    of the 36 pairs are cointegrated."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.006, (n, len(CCY))), axis=0)
    if coint:
        for base, members in ((0, (1, 2, 3)), (4, (5, 6))):
            for j in members:
                u = np.zeros(n)
                e = rng.normal(0, 0.003, n)
                for t in range(1, n):
                    u[t] = 0.8 * u[t - 1] + e[t]
                x[:, j] = x[:, base] + u
    return pd.DataFrame(np.exp(x), index=pd.bdate_range(start, periods=n), columns=CCY)


# ---- trials log ---------------------------------------------------------------

def test_append_is_append_only_and_complete(tmp_path):
    log = tmp_path / "t.jsonl"
    a = trials.append("s1", "train", "research/prereg/x.md", "ab" * 32, "r1.json", {"f": np.mean, "k": 2},
                      {"sharpe": np.float64(0.5)}, git=("c0ffee", False), path=log)
    first = log.read_text(encoding="utf-8")
    trials.append("s2", "stress", "research/prereg/x.md", "ab" * 32, "r2.json", {}, {}, git=("c0ffee", True),
                  path=log)
    text = log.read_text(encoding="utf-8")
    assert text.startswith(first) and text.count("\n") == 2
    rows = trials.read(log)
    assert [r["study"] for r in rows] == ["s1", "s2"]
    assert set(trials.FIELDS) <= set(rows[0]) and rows[0] == a
    assert rows[0]["params"] == {"f": "mean", "k": 2} and rows[0]["metric"] == {"sharpe": 0.5}
    assert trials.count("research/prereg/x.md", "train", log) == 1


def test_write_logs_every_training_run(tmp_path, monkeypatch):
    log = tmp_path / "trials.jsonl"
    monkeypatch.setenv(trials.ENV, str(log))
    monkeypatch.setattr(fs, "RESULTS", tmp_path)
    for _ in range(2):
        fs._write("momentum_fx_g10_train", "momentum-fx-g10.md", ("2000-01-03", "2021-12-31"),
                  {"sharpe": 0.1, "sharpe_se_lo": 0.2, "trial_count": 1}, params=fs.MOMENTUM)
    rows = trials.read(log)
    reg = prereg.load(REPO / "research" / "prereg" / "momentum-fx-g10.md")
    assert len(rows) == 2
    assert rows[1]["prereg_sha256"] == reg.sha256 and rows[1]["kind"] == "train"
    assert rows[1]["params"]["lookback_days"] == 63 and rows[1]["metric"]["sharpe"] == 0.1
    assert rows[1]["git_commit"] is None or len(rows[1]["git_commit"]) == 40


def test_committed_trials_log_is_well_formed():
    rows = trials.read(trials.LOG)
    for r in rows:
        assert set(trials.FIELDS) <= set(r)
        assert (REPO / r["prereg"]).is_file() and len(r["prereg_sha256"]) == 64
        assert r["kind"] in ("train", "stress", "calibration")


# ---- selection calibration ---------------------------------------------------

def test_bh_matches_hand_example_and_statsmodels():
    assert rb.bh_reject(np.array([0.01, 0.02, 0.03, 0.5])).tolist() == [True, True, True, False]
    assert rb.bh_reject(np.array([0.04, 0.03, 0.02, 0.001])).tolist() == [True, True, True, True]
    assert not rb.bh_reject(np.array([0.02, 0.5, 0.6, 0.7])).any()  # 0.02 > 0.05/4
    assert rb.bh_reject(np.array([np.nan, 0.001])).tolist() == [False, True]
    multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests
    p = np.random.default_rng(3).uniform(0, 0.2, 36) ** 2
    assert rb.bh_reject(p, 0.05).tolist() == multipletests(p, 0.05, method="fdr_bh")[0].tolist()


def test_eg_pvalue_is_five_percent_at_the_critical_value():
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.adfvalues import mackinnoncrit

    cv5 = mackinnoncrit(N=2, regression="c", nobs=np.inf)[1]
    assert rb.eg_pvalue(cv5) == pytest.approx(0.05, abs=0.003)
    assert rb.eg_pvalue(-5.0) < rb.eg_pvalue(-3.0) < rb.eg_pvalue(-1.0)


def test_null_resamples_whole_days_and_keeps_start():
    x = np.cumsum(np.random.default_rng(0).normal(size=(50, 3)), axis=0)
    y = rb.null_levels(x, np.random.default_rng(1))
    assert np.allclose(y[0], x[0]) and y.shape == x.shape
    dx, dy = np.diff(x, axis=0), np.diff(y, axis=0)
    rows = {tuple(np.round(r, 12)) for r in dx}
    assert all(tuple(np.round(r, 12)) in rows for r in dy)  # cross-section kept day by day


def test_calibration_separates_real_cointegration_from_null():
    pytest.importorskip("statsmodels")
    real = rb.selection_calibration(_walks(1300, 4, coint=True), n_perms=8, years=(2008,))
    f = real["folds"][0]
    true = {"AUD/CAD", "AUD/CHF", "AUD/EUR", "CAD/CHF", "CAD/EUR", "CHF/EUR", "GBP/JPY", "GBP/NOK", "JPY/NOK"}
    assert f["candidates"] == 36
    assert true <= set(f["selected_pairs"]) and true <= set(f["bh_selected_pairs"])
    assert f["null_p95"] < f["selected"] and f["perm_pvalue"] == pytest.approx(1 / 9)
    null = rb.selection_calibration(_walks(1300, 5), n_perms=8, years=(2008,))
    g = null["folds"][0]
    assert g["selected"] <= 4 and g["bh_selected"] <= 1 and 0 < g["perm_pvalue"] <= 1


# ---- cost and event stress ----------------------------------------------------

def test_event_days_flag_jumps_and_gaps_ex_ante():
    px = _walks(400, 6)
    jump_day = px.index[300]
    px.loc[jump_day:, "CHF"] *= 1.15  # a 14 % one-day move, SNB-style
    ev = rb.event_days(px)
    assert ev[jump_day] and not ev.iloc[299]
    assert not ev.iloc[:rb.EVENT_MIN_ROWS].any()  # no flag before the trailing window fills
    gapped = px.drop(px.index[200:204])  # five calendar days or more between two fixes
    assert rb.event_days(gapped)[px.index[204]]
    same = rb.event_days(px.iloc[:300])  # the flag on a day never depends on later days
    assert same.equals(ev.iloc[:300])


def test_stress_table_variants():
    idx = pd.bdate_range("2010-01-01", periods=6)
    r = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01, 0.02], index=idx)
    ev = pd.Series([False, False, True, False, True, False], index=idx)
    t = rb.stress_table(r, ev)
    assert t["event_days"] == 2 and t["event_days_exposed"] == 2 and t["event_pnl"] == pytest.approx(0.02)
    assert t["exclude"]["n_days"] == 4 and t["adverse"]["n_days"] == 6
    assert t["adverse"]["net_return_ann"] < t["base"]["net_return_ann"]


def test_cost_stress_is_monotone_in_cost():
    px = _walks(1300, 7, coint=True)
    p = {**fs.PAIRS_V2, "first_test_year": 2007}
    lo = rb.pairs_v2_daily(px, {**p, "cost_bp": 1.0})
    hi = rb.pairs_v2_daily(px, {**p, "cost_bp": 5.0})
    assert lo.index.equals(hi.index) and (hi <= lo + 1e-15).all() and hi.sum() < lo.sum()
    res = rb.cost_event_stress(px, p=p)
    for study in ("pairs_v2", "momentum"):
        s = [res[study][f"{c:g}bp"]["base"]["sharpe"] for c in rb.COSTS_BP]
        assert s[0] > s[1] > s[2]
    json.dumps(res)


def test_robustness_refuses_holdout_data():
    from models import HoldoutViolation

    with pytest.raises(HoldoutViolation):
        rb.cost_event_stress(_walks(300, 8, start="2021-06-01"))
