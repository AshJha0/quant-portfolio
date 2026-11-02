"""Tests for the forward evaluation and the power gate on signing. Synthetic data, no network."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "research" / "forward"), str(REPO / "research" / "tools"),
                str(REPO / "research" / "studies"), str(REPO / "data" / "fetch")]

import evaluate as ev  # noqa: E402
import forward_studies as fs  # noqa: E402
import prereg  # noqa: E402

ECB = ["USD", "AUD", "CAD", "CHF", "GBP", "JPY", "NOK", "NZD", "SEK"]
LEVEL = {"USD": 1.1, "AUD": 1.6, "CAD": 1.5, "CHF": 0.95, "GBP": 0.85, "JPY": 160.0,
         "NOK": 11.5, "NZD": 1.8, "SEK": 11.2}


def _rates(start: str, end: str, seed: int = 0) -> pd.DataFrame:
    days = [d for d in pd.date_range(start, end) if not ev.target_closed(d.date())]
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.005, (len(days), len(ECB)))
    return pd.DataFrame(np.exp(np.cumsum(steps, axis=0)) * np.array([LEVEL[c] for c in ECB]),
                        index=pd.DatetimeIndex(days), columns=ECB)


def _write_inputs(tmp: Path, log_end: str = "2028-11-02", seed: int = 0) -> tuple[Path, Path, str]:
    eur = _rates("2022-01-03", log_end, seed)
    hist = eur[eur.index < "2026-11-03"]
    fwd = eur[eur.index >= "2026-11-03"]
    long = hist.stack().rename("OBS_VALUE").reset_index()
    long.columns = ["TIME_PERIOD", "CURRENCY", "OBS_VALUE"]
    long["TIME_PERIOD"] = long["TIME_PERIOD"].dt.strftime("%Y-%m-%d")
    hpath = tmp / "history.csv"
    hpath.write_bytes(long.to_csv(index=False).encode("utf-8"))
    lpath = tmp / "log.jsonl"
    with lpath.open("w", encoding="utf-8", newline="\n") as f:
        for day, row in fwd.iterrows():
            usd = {c: row["USD"] / row[c] for c in ECB if c != "USD"} | {"EUR": row["USD"]}
            f.write(json.dumps({"schema": 1, "date": day.strftime("%Y-%m-%d"),
                                "rates_per_eur": {c: row[c] for c in ECB}, "usd_per_unit": usd,
                                "provenance": {}}) + "\n")
    return lpath, hpath, hashlib.sha256(hpath.read_bytes()).hexdigest()


def _signed_copies(tmp: Path, signed: bool = True) -> Path:
    fake = tmp / ("repo" if signed else "repo_unsigned")
    (fake / "research" / "prereg").mkdir(parents=True)
    for name, _ in ev.REGS.values():
        src = (REPO / "research" / "prereg" / name).read_text(encoding="utf-8")
        body, _ = prereg._split(src)
        h = prereg.registration_hash(src)
        sig = f"signed_by: t\nsigned_on: 2026-11-02\nsha256: {h}\n" if signed else "signed_by:\nsigned_on:\nsha256:\n"
        (fake / "research" / "prereg" / name).write_text(f"{body}## Signature\n\n{sig}", encoding="utf-8")
    return fake


def test_easter_and_target_days():
    assert ev._easter(2027) == dt.date(2027, 3, 28)
    assert ev._easter(2028) == dt.date(2028, 4, 16)
    assert ev.target_closed(dt.date(2027, 3, 26)) and ev.target_closed(dt.date(2027, 3, 29))
    assert ev.target_closed(dt.date(2027, 12, 25)) and not ev.target_closed(dt.date(2027, 3, 30))


def test_interim_before_evaluation_date_gives_no_verdict(tmp_path):
    lpath, hpath, sha = _write_inputs(tmp_path, log_end="2027-06-30")
    fake = _signed_copies(tmp_path)  # even signed, the date blocks a verdict
    reps = ev.evaluate(lpath, hpath, sha, "abc", today=dt.date(2027, 7, 1), repo=fake)
    for rep in reps.values():
        assert rep["final"] is False and rep["reject_h0"] is None
        assert rep["label"].startswith("INTERIM")
        assert rep["period"] == {"start": "2026-11-03", "end": "2027-06-30"}
        assert any("evaluation date" in w for w in rep["not_final_because"])
    out = tmp_path / "out"
    assert ev.main(["--log", str(lpath), "--history", str(hpath), "--history-sha256", sha,
                    "--log-commit", "abc", "--today", "2027-07-01", "--out", str(out)]) == 0
    assert not out.exists()


def test_full_window_after_date_still_not_final_while_unsigned(tmp_path):
    lpath, hpath, sha = _write_inputs(tmp_path)
    reps = ev.evaluate(lpath, hpath, sha, "abc", today=dt.date(2028, 11, 13),
                       repo=_signed_copies(tmp_path, signed=False))
    for rep in reps.values():
        assert rep["not_final_because"] == ["registration is not signed"]
        assert rep["period"]["end"] == "2028-11-02"
    assert len(reps["pairs"]["folds"]) == 2


def test_final_verdict_with_signed_copies(tmp_path):
    lpath, hpath, sha = _write_inputs(tmp_path)
    reps = ev.evaluate(lpath, hpath, sha, "abc", today=dt.date(2028, 11, 13), repo=_signed_copies(tmp_path))
    for rep in reps.values():
        assert rep["final"] is True and isinstance(rep["reject_h0"], bool)
        assert rep["mode"] == "forward" and rep["history_sha256"] == sha
    assert reps["momentum"]["n_days"] == len(ev.expected_dates(ev.FORWARD_START, ev.WINDOW_END))


def test_missing_day_blocks_verdict(tmp_path):
    lpath, hpath, sha = _write_inputs(tmp_path)
    lines = lpath.read_text(encoding="utf-8").splitlines()
    lpath.write_text("\n".join(lines[:10] + lines[11:]) + "\n", encoding="utf-8")
    rep = ev.evaluate(lpath, hpath, sha, "abc", today=dt.date(2029, 1, 1), repo=_signed_copies(tmp_path))
    assert rep["momentum"]["reject_h0"] is None
    assert any("missing" in w for w in rep["momentum"]["not_final_because"])


def test_refusals(tmp_path):
    lpath, hpath, _sha = _write_inputs(tmp_path, log_end="2026-12-31")
    with pytest.raises(ev.EvaluationError, match="sha256"):
        ev.evaluate(lpath, hpath, "0" * 64, "abc", today=dt.date(2027, 1, 1))
    rec = json.loads(lpath.read_text(encoding="utf-8").splitlines()[0])
    cases = ((lambda r: r["usd_per_unit"].update(GBP=9.9), "disagrees"),
             (lambda r: r.update(schema=2), "schema"),
             (lambda r: r.update(date="2026-11-02"), "before"))
    for mutate, msg in cases:
        bad = json.loads(json.dumps(rec))
        mutate(bad)
        p = tmp_path / "bad.jsonl"
        p.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        with pytest.raises(ev.EvaluationError, match=msg):
            ev.read_log(p)
    dup = tmp_path / "dup.jsonl"
    dup.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ev.EvaluationError, match="duplicate"):
        ev.read_log(dup)
    assert ev.main(["--log", str(lpath), "--history", str(hpath), "--history-sha256", "0" * 64,
                    "--log-commit", "x"]) == 1


def test_only_forward_days_scored_and_build_cost(tmp_path):
    lpath, hpath, sha = _write_inputs(tmp_path, log_end="2027-03-31")
    px = ev.combine(ev.read_history(hpath, sha), ev.read_log(lpath))
    r = ev.momentum_scored(px, dt.date(2027, 3, 31))
    assert r.index.min() == pd.Timestamp("2026-11-03")
    raw = fs.momentum_returns(px)
    assert r.iloc[0] == pytest.approx(raw.loc["2026-11-03"] - 1e-4)  # unit-gross build, 1 bp
    assert (r.iloc[1:] == raw.loc[r.index[1:]]).all()
    pr, folds = ev.pairs_returns(px, dt.date(2027, 3, 31))
    assert pr.index.min() >= pd.Timestamp("2026-11-03") and len(folds) == 1


def test_pairs_scores_a_planted_cointegrated_pair(tmp_path):
    """Ground truth: GBP built as a mean-reverting spread on CHF; the pair is selected."""
    eur = _rates("2022-01-03", "2027-06-30", seed=3)
    rng = np.random.default_rng(7)
    u = np.zeros(len(eur))
    for t in range(1, len(u)):
        u[t] = 0.93 * u[t - 1] + rng.normal(0, 0.004)
    usd_chf = eur["USD"] / eur["CHF"]
    eur["GBP"] = eur["USD"] / (usd_chf * np.exp(u))  # log usd_gbp = log usd_chf + u
    idx = eur.index
    px = ev.usd_prices(eur)
    pr, folds = ev.pairs_returns(px[sorted(px.columns)], dt.date(2027, 6, 30))
    assert "CHF/GBP" in folds[0]["pairs"] and folds[0]["entries"] > 0
    assert pr.index.min() == idx[idx >= "2026-11-03"][0]


# ---- power gate on signing -----------------------------------------------

def _draft(tmp: Path, ref: str | None) -> Path:
    p = tmp / "research" / "prereg" / "x.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    extra = f"Power: `{ref}`.\n" if ref else ""
    p.write_text(f"# X\n\nH1.\n{extra}\n## Signature\n\nsigned_by:\nsigned_on:\nsha256:\n", encoding="utf-8")
    return p


def test_sign_refuses_without_power_result(tmp_path):
    p = _draft(tmp_path, None)
    with pytest.raises(prereg.SigningRefused, match="no simulated power"):
        prereg.sign(p, "t", "2026-11-02", repo=tmp_path)
    p = _draft(tmp_path, "research/power/x.json")
    with pytest.raises(prereg.SigningRefused, match="unreadable"):
        prereg.sign(p, "t", "2026-11-02", repo=tmp_path)
    (tmp_path / "research" / "power").mkdir(parents=True)
    (tmp_path / "research" / "power" / "x.json").write_text(json.dumps({"cells": [{"reps": 0}]}))
    with pytest.raises(prereg.SigningRefused, match="cell 0"):
        prereg.sign(p, "t", "2026-11-02", repo=tmp_path)
    assert not prereg.load(p).signed
    assert prereg.main(["prereg.py", "sign", str(p), "t", "2026-11-02"]) == 1


def test_sign_with_power_result_keeps_hash(tmp_path):
    p = _draft(tmp_path, "research/power/x.json")
    (tmp_path / "research" / "power").mkdir(parents=True)
    (tmp_path / "research" / "power" / "x.json").write_text(
        json.dumps({"cells": [{"reps": 200, "level": {"rejection_rate": 0.5}}]}))
    before = prereg.load(p).sha256
    assert prereg.sign(p, "t", "2026-11-02", repo=tmp_path) == before
    assert prereg.load(p).signed
    with pytest.raises(prereg.SigningRefused, match="already"):
        prereg.sign(p, "t", "2026-11-02", repo=tmp_path)


def test_real_drafts_power_gate(tmp_path):
    for name in ("pairs-fx-g10-v2.md", "momentum-fx-g10.md"):
        shutil.copy(REPO / "research" / "prereg" / name, tmp_path / name)
    assert prereg.power_errors((tmp_path / "pairs-fx-g10-v2.md").read_text(encoding="utf-8")) == []
    assert prereg.power_errors((tmp_path / "momentum-fx-g10.md").read_text(encoding="utf-8")) == []
