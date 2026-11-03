"""Tests for research/cli/qp_run.py: sweep, query, rerun (W23 part 2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "research" / "cli"), str(Path(__file__).parent)]

import qp_run  # noqa: E402

FN = "sweep_fixture:study"
DS = "research/cli/tests/sweep_fixture.py"
GIT = ("abc", False)


def _spec(**kw):
    s = {"study": "toy", "fn": FN, "fixed": {"c": 1.0}, "grid": {"a": [1, 2], "b": [0.5]},
         "datasets": [DS]}
    s.update(kw)
    return s


def test_sweep_logs_every_point_with_hashes(tmp_path):
    log = tmp_path / "t.jsonl"
    rows = qp_run.sweep(_spec(), log, git=GIT)
    assert [r["params"] for r in rows] == [{"a": 1, "b": 0.5, "c": 1.0},
                                           {"a": 2, "b": 0.5, "c": 1.0}]
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 2
    r = lines[0]
    assert r["prereg"] == "exploratory" and r["kind"] == "sweep" and r["fn"] == FN
    assert r["datasets"][DS] == qp_run.sha256_file(REPO / DS)
    assert r["lockfile_sha256"] == qp_run.sha256_file(REPO / "uv.lock")
    assert r["metric"] == {"sharpe": 1.5, "n": 3}


def test_dry_run_writes_nothing(tmp_path):
    log = tmp_path / "t.jsonl"
    assert len(qp_run.sweep(_spec(), log, dry_run=True, git=GIT)) == 2
    assert not log.exists()


def test_sweep_refuses_bad_specs(tmp_path):
    with pytest.raises(ValueError):
        qp_run.sweep(_spec(kind="holdout"), tmp_path / "x", git=GIT)
    with pytest.raises(ValueError):
        qp_run.sweep(_spec(fixed={"a": 1}), tmp_path / "x", git=GIT)
    with pytest.raises(ValueError):
        qp_run.sweep({"study": "x"}, tmp_path / "x", git=GIT)
    with pytest.raises(ValueError):
        qp_run.resolve("no_colon")


def test_query_and_rerun(tmp_path, capsys):
    log = tmp_path / "t.jsonl"
    qp_run.sweep(_spec(), log, git=GIT)
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(qp_run.query(rows, study="toy", kind="sweep")) == 2
    assert qp_run.query(rows, study="other") == []
    assert qp_run.main(["trials", "rerun", "--random", "7", "--log", str(log)]) == 0
    rows[1]["metric"]["sharpe"] += 1e-9
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert qp_run.main(["trials", "rerun", "--index", "1", "--log", str(log)]) == 1
    assert qp_run.main(["trials", "query", "--study", "toy", "--log", str(log)]) == 0
    assert "toy" in capsys.readouterr().out


def test_rerun_refuses_changed_data_and_old_rows():
    row = {"fn": FN, "params": {"a": 1, "b": 1, "c": 0}, "metric": {},
           "datasets": {DS: "0" * 64}}
    with pytest.raises(ValueError, match="changed"):
        qp_run.rerun(row)
    with pytest.raises(ValueError):
        qp_run.rerun({"params": {}})


def test_compare_nan_and_missing():
    assert qp_run.compare({"x": float("nan")}, {"x": float("nan")}, 1e-12) == []
    assert qp_run.compare({"x": 1.0}, {}, 1e-12)
