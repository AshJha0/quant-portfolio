"""Unit tests for the CI helper scripts in .github/scripts.

Covers bench_ab.py (the same-runner regression rule and the results schema),
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
COMMITTED = sorted(REPO.glob("*/*/bench/results.json"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


ab = _load("bench_ab")
agg = _load("aggregate_counts")
disc = _load("discover_projects")


def _write(p: Path, recs: object) -> Path:
    p.write_text(json.dumps(recs), encoding="utf-8")
    return p


# ----------------------------------------------------------------- bench_ab --
def _rec(kernel, p10, p50, p90, unit="ns"):
    return {"kernel": kernel, "unit": unit, "p10": p10, "p50": p50, "p90": p90}


class TestBenchAbRule:
    B = ab.Band(90.0, 100.0, 110.0)

    def test_clear_regression_fails(self):
        assert ab.classify(self.B, ab.Band(140.0, 150.0, 160.0))[1] == "FAIL"

    def test_large_ratio_with_overlapping_bands_is_report_only(self):
        # +50% on the median, but head p10 (105) is inside the base band.
        assert ab.classify(self.B, ab.Band(105.0, 150.0, 400.0))[1] == "slower"

    def test_small_shift_with_separate_bands_is_report_only(self):
        # Bands do not overlap, but +20% is under the 25% threshold.
        assert ab.classify(self.B, ab.Band(115.0, 120.0, 125.0))[1] == "same"

    def test_speedup_new_and_removed(self):
        assert ab.classify(self.B, ab.Band(40.0, 50.0, 60.0))[1] == "faster"
        assert ab.classify(None, self.B)[1] == "new"
        assert ab.classify(self.B, None)[1] == "removed"


class TestBenchAbLoad:
    def test_units_and_missing_band(self, tmp_path):
        p = _write(tmp_path / "r.json", {"results": [
            _rec("a", 1, 2, 3, unit="us"),
            {"kernel": "old", "p50": 7},
        ]})
        bands = ab.load_bands(p)
        assert bands["a"] == ab.Band(1000.0, 2000.0, 3000.0)
        assert bands["old"] == ab.Band(7.0, 7.0, 7.0)

    def test_pool_takes_median_per_percentile(self):
        rounds = [{"k": ab.Band(1, 10, 20)}, {"k": ab.Band(3, 30, 90)}, {"k": ab.Band(2, 20, 40)}]
        assert ab.pool(rounds)["k"] == ab.Band(2, 20, 40)

    @pytest.mark.parametrize("path", COMMITTED, ids=lambda p: str(p.relative_to(REPO)))
    def test_committed_results_parse(self, path):
        assert ab.load_bands(path)


class TestBenchAbMain:
    def _rounds(self, tmp_path, side, p50s):
        paths = []
        for i, v in enumerate(p50s):
            paths.append(str(_write(tmp_path / f"{side}{i}.json", [
                _rec("slow_kernel", 0.95 * v, v, 1.05 * v), _rec("steady", 9, 10, 11)])))
        return paths

    def test_interleaved_regression_fails_and_writes_reports(self, tmp_path):
        base = self._rounds(tmp_path, "b", [1000, 1010, 990])
        head = self._rounds(tmp_path, "h", [1500, 1490, 1520])
        md, js = tmp_path / "s.md", tmp_path / "s.json"
        rc = ab.main(["--engine", "cpp/x", "--base", *base, "--head", *head,
                      "--summary", str(md), "--json", str(js)])
        assert rc == 1
        rows = {r["kernel"]: r["status"] for r in json.loads(js.read_text(encoding="utf-8"))}
        assert rows == {"slow_kernel": "FAIL", "steady": "same"}
        assert "same runner" in md.read_text(encoding="utf-8")

    def test_noise_passes(self, tmp_path):
        base = self._rounds(tmp_path, "b", [1000, 1300, 900])
        head = self._rounds(tmp_path, "h", [1100, 950, 1250])
        assert ab.main(["--engine", "x", "--base", *base, "--head", *head]) == 0

    def test_missing_head_is_an_error_and_missing_base_report_only(self, tmp_path):
        head = self._rounds(tmp_path, "h", [10])
        assert ab.main(["--engine", "x", "--base", str(tmp_path / "no.json"),
                        "--head", str(tmp_path / "none.json")]) == 2
        assert ab.main(["--engine", "x", "--base", str(tmp_path / "no.json"), "--head", *head]) == 0


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
        assert core == {"path": "python/core", "slug": "core", "extra": "test", "spec": ".[test]", "deps": ""}

    def test_workspace_deps_are_listed_transitively(self, tmp_path):
        def proj(rel, name, deps, dev=()):
            (tmp_path / rel).mkdir(parents=True)
            text = "\n".join([
                "[project]",
                f'name = "{name}"',
                "dependencies = " + json.dumps(list(deps)),
                "[project.optional-dependencies]",
                "dev = " + json.dumps(list(dev)),
                "",
            ])
            (tmp_path / rel / "pyproject.toml").write_text(text, encoding="utf-8")
        proj("python/core", "qp-core", [])
        proj("python/oms", "qp-oms", ["qp-core"])
        proj("python/execution", "qp-exec", ["qp-core", "numpy>=1.24"], dev=["qp-oms>=3"])
        proj("python/backtest", "qp-bt", ["qp-core"], dev=["qp-exec"])
        proj("python/equity/05-a", "eq-a", ["qp-bt>=3.0.0.dev0"], dev=["pytest"])
        got = {p["path"]: p["deps"] for p in disc.discover(tmp_path)}
        # the dev extra is followed only for the project itself
        assert got["python/equity/05-a"] == "python/backtest"
        assert got["python/backtest"] == "python/execution"
        assert got["python/execution"] == "python/oms"
        assert got["python/oms"] == ""

    def test_core_absent_returns_none(self, tmp_path):
        assert disc.discover_core(tmp_path) is None
