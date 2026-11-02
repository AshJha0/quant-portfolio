"""Unit tests for the CI helper scripts in .github/scripts.

Covers bench_compare.py (the regression rule and both committed schemas),
aggregate_counts.py (artefact names for presets, python/core and bindings)
and discover_projects.py (python/core is found and kept out of the 23).

Run from the repository root::

    python -m pytest -p no:cacheprovider -q tests/test_ci_scripts.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / ".github" / "scripts"
FIX = Path(__file__).resolve().parent / "fixtures" / "bench"
COMMITTED = sorted(REPO.glob("*/*/bench/results.json"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


bc = _load("bench_compare")
agg = _load("aggregate_counts")
disc = _load("discover_projects")


def _write(p: Path, recs: object) -> Path:
    p.write_text(json.dumps(recs), encoding="utf-8")
    return p


# ------------------------------------------------------------ bench_compare --
class TestBenchLoad:
    def test_cpp_schema_uses_after_records_and_median(self):
        p50 = bc.load_p50(FIX / "cpp_committed.json")
        assert p50["bs_price"] == 100.0           # median of 90 and 110; "before" 500 ignored
        assert p50["mc_price n=10000"] == 5000.0  # median of 4000 and 6000
        assert p50["only_committed"] == 3000.0    # 3 us converted to ns

    def test_rust_schema_without_label(self):
        p50 = bc.load_p50(FIX / "rust_committed.json")
        assert p50 == {"pnl_one_scenario_250pos": 2026.0, "normal_var_closed_form": 12.5}

    def test_wrapped_dict_is_accepted(self, tmp_path):
        p = _write(tmp_path / "w.json", {"results": [{"kernel": "k", "p50": 7}]})
        assert bc.load_p50(p) == {"k": 7.0}

    def test_unknown_unit_is_an_input_error(self, tmp_path):
        p = _write(tmp_path / "u.json", [{"kernel": "k", "p50": 1, "unit": "cycles"}])
        with pytest.raises(bc.BenchInputError):
            bc.load_p50(p)

    @pytest.mark.parametrize("path", COMMITTED, ids=[p.parent.parent.relative_to(REPO).as_posix() for p in COMMITTED])
    def test_every_committed_results_file_loads(self, path):
        p50 = bc.load_p50(path)
        assert p50, path
        assert all(v > 0 for v in p50.values()), path


class TestBenchRule:
    def test_regression_above_floor_fails(self):
        rows = bc.compare("e", {"slow": 5000.0}, {"slow": 10001.0})
        assert [r.status for r in rows] == ["fail"]

    def test_exactly_two_times_passes(self):
        assert bc.compare("e", {"slow": 5000.0}, {"slow": 10000.0})[0].status == "ok"

    def test_kernel_at_or_below_one_microsecond_is_report_only(self):
        rows = bc.compare("e", {"fast": 1000.0, "tiny": 12.5}, {"fast": 9000.0, "tiny": 500.0})
        assert {r.kernel: r.status for r in rows} == {"fast": "report", "tiny": "report"}

    def test_new_and_missing_kernels_are_reported(self):
        rows = bc.compare("e", {"gone": 5000.0}, {"added": 5000.0})
        assert {r.kernel: r.status for r in rows} == {"gone": "missing", "added": "new"}


class TestBenchMain:
    def test_end_to_end_fail_and_summary(self, tmp_path):
        ci = _write(tmp_path / "ci.json", [
            {"kernel": "bs_price", "label": "after", "p50": 900.0},           # 9x, committed <= 1 us
            {"kernel": "mc_price n=10000", "label": "after", "p50": 10100.0},  # 2.02x of 5000: fail
        ])
        md, js = tmp_path / "s.md", tmp_path / "c.json"
        rc = bc.main(["--pair", "cpp/eqopt", str(FIX / "cpp_committed.json"), str(ci),
                      "--summary", str(md), "--json", str(js)])
        assert rc == 1
        text = md.read_text(encoding="utf-8")
        assert "**1 failing kernel(s)**" in text and "**FAIL**" in text
        assert "not the published numbers" in text
        rows = {r["kernel"]: r["status"] for r in json.loads(js.read_text(encoding="utf-8"))}
        assert rows == {"bs_price": "report", "mc_price n=10000": "fail", "only_committed": "missing"}

    def test_end_to_end_pass_with_rust_schema(self, tmp_path):
        ci = _write(tmp_path / "ci.json", [
            {"engine": "fx-var-engine", "kernel": "pnl_one_scenario_250pos", "unit": "ns", "p50": 3900},
            {"engine": "fx-var-engine", "kernel": "normal_var_closed_form", "unit": "ns", "p50": 80},
        ])
        assert bc.main(["--pair", "rust/fxvar", str(FIX / "rust_committed.json"), str(ci)]) == 0

    def test_missing_ci_file_is_an_error(self, tmp_path):
        assert bc.main(["--pair", "x", str(FIX / "rust_committed.json"), str(tmp_path / "none.json")]) == 2

    def test_missing_committed_file_is_report_only(self, tmp_path):
        ci = _write(tmp_path / "ci.json", [{"kernel": "k", "p50": 1e9}])
        assert bc.main(["--pair", "x", str(tmp_path / "absent.json"), str(ci)]) == 0


# --------------------------------------------------------- aggregate_counts --
JUNIT = """<?xml version="1.0"?><testsuites><testsuite name="s">
<testcase name="a"/><testcase name="b"/><testcase name="c"><skipped/></testcase>
</testsuite></testsuites>"""


def _artefact(root: Path, name: str, fname: str = "junit.xml", body: str = JUNIT, sub: str = "") -> None:
    d = root / name / sub
    d.mkdir(parents=True)
    (d / fname).write_text(body, encoding="utf-8")


class TestAggregate:
    def test_presets_core_rust_and_bindings(self, tmp_path):
        _artefact(tmp_path, "junit-py-core-3.12", sub="junit")
        _artefact(tmp_path, "junit-py-core-3.10")
        _artefact(tmp_path, "junit-py-equity__01-options-pricing-3.12")
        presets = {"gcc", "clang", "asan", "tsan", "gcc16-cxx26", "msvc"}
        for preset in presets:
            _artefact(tmp_path, f"junit-cpp-fx-var-engine-{preset}")
        _artefact(tmp_path, "rust-core", "cargo-test.log",
                  "test result: ok. 4 passed; 0 failed; 1 ignored; 0 measured\n")
        _artefact(tmp_path, "junit-bind-cpp-nanobind-3.12")
        _artefact(tmp_path, "junit-bind-rust-pyo3-3.12")
        _artefact(tmp_path, "junit-bind-rust-pyo3-3.13")
        out = agg.aggregate(tmp_path, "3.12", "gcc")
        assert out["python"]["core"] == {"passed": 2, "failed": 0, "errors": 0, "skipped": 1}
        assert set(out["python_by_version"]["core"]) == {"3.10", "3.12"}
        assert set(out["cpp_by_compiler"]["fx-var-engine"]) == presets
        assert set(out["cpp"]) == {"fx-var-engine"}
        assert out["rust"]["core"]["passed"] == 4
        assert set(out["bindings"]) == {"cpp-nanobind", "rust-pyo3"}
        assert set(out["bindings_by_version"]["rust-pyo3"]) == {"3.12", "3.13"}
        # python 2 projects x 2 passed, cpp gcc 2, rust 4, bindings 2 x 2
        assert out["totals"]["all_passed"] == 4 + 2 + 4 + 4

    def test_unsplittable_cpp_name_is_an_error(self, tmp_path):
        _artefact(tmp_path, "junit-cpp-something-gcc")
        with pytest.raises(ValueError):
            agg.aggregate(tmp_path, "3.12", "gcc")


# -------------------------------------------------------- discover_projects --
class TestDiscover:
    def test_core_is_separate_from_the_projects(self, tmp_path):
        for rel in ("python/core", "python/equity/01-a", "python/fx/02-b"):
            (tmp_path / rel).mkdir(parents=True)
            (tmp_path / rel / "pyproject.toml").write_text(
                '[project]\nname = "x"\n[project.optional-dependencies]\ntest = ["pytest"]\n', encoding="utf-8")
        assert [p["slug"] for p in disc.discover(tmp_path)] == ["equity__01-a", "fx__02-b"]
        core = disc.discover_core(tmp_path)
        assert core == {"path": "python/core", "slug": "core", "extra": "test", "spec": ".[test]"}

    def test_core_absent_returns_none(self, tmp_path):
        assert disc.discover_core(tmp_path) is None
