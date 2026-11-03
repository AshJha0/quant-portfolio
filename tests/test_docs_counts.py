"""TEST-12: the counts quoted in the portfolio docs match the repository.

Test counts are not typed into the top-level docs. CI's ``counts`` job
aggregates the JUnit XML and cargo logs of every test job into
``counts.json`` (pointed at by ``QP_COUNTS_JSON``; a ``counts.json`` at the
repository root is also picked up) and publishes it in the run summary. This
file checks:

* the repository structure the README describes (31 sub-projects);
* every LEARN.md question count and COOKBOOK.md recipe count quoted in the
  docs equals the count in those files;
* no hand-typed test total appears in the top-level docs;
* the golden-vector pipeline exists and reproduces the committed constants;
* with ``counts.json`` present: every project (and ``python/core``), every
  engine under every CMake preset, every Rust crate (and ``rust/core``) and
  both binding packages reported, with no failures or errors. Without it (a local run) those tests skip.

Run from the repository root::

    python -m pytest -p no:cacheprovider -q tests/test_docs_counts.py
    QP_COUNTS_JSON=counts.json python -m pytest -p no:cacheprovider -q tests/test_docs_counts.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
DOCS = REPO / "docs"

PY_AREAS = {
    "equity": sorted((REPO / "python/equity").glob("[0-9][0-9]-*")),
    "fx": sorted((REPO / "python/fx").glob("[0-9][0-9]-*")),
    "foundations": sorted((REPO / "python/foundations").glob("[0-9][0-9]-*")),
}
# Shared Python stacks CI tests as projects of their own (discover_projects.py
# SHARED); their slug is the folder name.
PY_SHARED = sorted(p.name for p in (REPO / "python").glob("*")
                   if p.name in {"backtest", "oms", "execution", "posttrade"} and (p / "pyproject.toml").is_file())
# Engines are the */*-engine directories. cpp/core and rust/core are shared
# libraries, not sub-projects; rust/ also holds the workspace Cargo.toml.
CPP_ENGINES = sorted(p for p in (REPO / "cpp").glob("*-engine") if p.is_dir())
RUST_ENGINES = sorted(p for p in (REPO / "rust").glob("*-engine") if p.is_dir())
# CMake presets every engine runs in CI (cpp/<engine>/CMakePresets.json).
CPP_PRESETS = {"gcc", "clang", "asan", "tsan", "gcc16-cxx26", "msvc"}
BINDINGS = {"cpp-nanobind", "rust-pyo3"}


def _counts_path() -> Path | None:
    env = os.environ.get("QP_COUNTS_JSON")
    if env:
        p = Path(env)
        assert p.is_file(), f"QP_COUNTS_JSON={env} does not exist"
        return p
    p = REPO / "counts.json"
    return p if p.is_file() else None


@cache
def executed_counts() -> dict | None:
    """counts.json written by the CI ``counts`` job, or None outside CI."""
    p = _counts_path()
    return json.loads(p.read_text(encoding="utf-8")) if p else None


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _num(s: str) -> int:
    return int(s.replace(",", ""))


def count_questions() -> int:
    return len(re.findall(r"^\*\*Q\d+\.", _read(DOCS / "LEARN.md"), re.M))


def count_sub_projects() -> int:
    return sum(len(v) for v in PY_AREAS.values()) + len(CPP_ENGINES) + len(RUST_ENGINES)


def count_recipes() -> int:
    heads = re.findall(r"^### (.+)$", _read(DOCS / "COOKBOOK.md"), re.M)
    return sum(1 for h in heads if h.startswith(("How do I", "What happens")))


@pytest.fixture(scope="module")
def readme() -> str:
    return _read(README)


DOC_FILES = (README, DOCS / "README.md", DOCS / "ARCHITECTURE.md", DOCS / "DIAGRAMS.md",
             DOCS / "LEARN.md", DOCS / "COOKBOOK.md", DOCS / "MARKET_RISK.md", DOCS / "index.html")


# -------------------------------------------------------------------- tests --
class TestStructure:
    def test_sub_project_count_is_derived_everywhere(self, readme):
        # Generated, not typed: the number comes from the directory layout
        # and every "<n> sub-projects" in the docs must equal it.
        n = count_sub_projects()
        assert f"{n} sub-projects" in readme
        for f in (*DOC_FILES, DOCS / "START_HERE.md"):
            for q in re.findall(r"(\d+) sub-projects", _read(f)):
                assert int(q) == n, (f, q, n)

    def test_31_sub_projects(self):
        # The layout the README describes; a change here is a structural
        # change and needs a docs review, not just a new number.
        assert count_sub_projects() == 31

    def test_23_python_projects(self):
        assert sum(len(v) for v in PY_AREAS.values()) == 23
        assert len(PY_AREAS["equity"]) == len(PY_AREAS["fx"]) == 10
        assert len(PY_AREAS["foundations"]) == 3

    def test_every_python_project_collects_from_its_root_without_install(self):
        # CONVENTIONS.md: `pytest -q` passes offline from the project root.
        for projects in PY_AREAS.values():
            for p in projects:
                toml = _read(p / "pyproject.toml")
                has_pythonpath = 'pythonpath = ["src"]' in toml
                has_conftest = (p / "conftest.py").exists()
                assert has_pythonpath or has_conftest, p


class TestLearnAndCookbook:
    """Question and recipe counts quoted anywhere in the docs equal the files."""

    def test_question_bank_count_everywhere(self):
        n = count_questions()
        learn = _read(DOCS / "LEARN.md")
        assert f"**{n} questions across" in learn
        assert f"| **Total** | | **{n}** |" in learn
        assert f"{n} self-test questions" in learn
        for f in DOC_FILES:
            for q in re.findall(r"(\d[\d,]*)(?:-question| self-test questions| questions across)", _read(f)):
                assert _num(q) == n, (f, q, n)

    def test_learn_rounds_add_up(self):
        learn = _read(DOCS / "LEARN.md")
        rows = re.findall(r"^\| (\d+) \| [^|]+ \| (\d+) \|$", learn, re.M)
        assert len(rows) == 19, "round table not found"
        assert sum(int(n) for _, n in rows) == count_questions()
        per_round = re.split(r"^#{2,4} Round (\d+)\b.*$", learn, flags=re.M)
        if len(per_round) > 1:
            counts = {int(per_round[i]): len(re.findall(r"^\*\*Q\d+\.", per_round[i + 1], re.M))
                      for i in range(1, len(per_round), 2)}
            assert counts == {int(r): int(n) for r, n in rows}, counts

    def test_recipe_count_everywhere(self):
        n = count_recipes()
        assert f"{n} recipes" in _read(DOCS / "README.md")
        assert f"{n} recipes" in _read(DOCS / "index.html")
        for f in DOC_FILES:
            for q in re.findall(r"(\d+) (?:copy-pasteable |verified )?recipes", _read(f)):
                assert int(q) == n, (f, q, n)


class TestNoHandTypedTestCounts:
    """Test counts are generated by CI (counts.json), never typed into the
    top-level docs, so they cannot go stale."""

    def test_docs_point_at_counts_job(self, readme):
        assert "counts.json" in readme
        assert "counts.json" in _read(DOCS / "index.html")

    def test_no_test_totals_in_prose(self):
        pat = re.compile(r"\b\d[\d,]* (?:passing )?tests? (?:across|in total|all passing)|passing tests\b"
                         r"|\b(?:6,446|5,963|5,943|5,913|5,734)\b", re.I)
        for f in DOC_FILES:
            hits = pat.findall(_read(f))
            assert not hits, (f, hits)


class TestGoldenPipeline:
    """The golden-vector pipeline documented in ARCHITECTURE.md exists for all
    four engine pairs and reproduces the committed constants."""

    @pytest.mark.parametrize("engine", [e.name for e in CPP_ENGINES])
    def test_cpp_header_matches_json(self, engine):
        r = subprocess.run([sys.executable, str(REPO / "cpp" / engine / "tools/gen_golden_header.py"), "--check"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr

    @pytest.mark.parametrize("engine", [e.name for e in RUST_ENGINES])
    def test_rust_module_matches_json(self, engine):
        r = subprocess.run([sys.executable, str(REPO / "rust" / engine / "tools/gen_golden_rs.py"), "--check"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr

    @pytest.mark.parametrize("project", ["python/equity/01-options-pricing", "python/fx/01-options-pricing",
                                         "python/equity/03-var-es-engine", "python/fx/03-var-es-engine"])
    def test_python_project_has_generator_json_and_lock(self, project):
        p = REPO / project
        assert (p / "tests/golden/generate_golden.py").exists()
        assert (p / "tests/golden/golden_vectors.json").exists()
        assert any(p.glob("tests/test_golden*.py")), project


class TestExecutedCounts:
    """Only meaningful in CI, where counts.json exists."""

    @pytest.fixture(autouse=True)
    def _need_counts(self):
        if executed_counts() is None:
            pytest.skip("counts.json not present (local run)")

    def test_every_project_and_engine_reported(self):
        c = executed_counts()
        want_py = {f"{a}__{p.name}" for a, ps in PY_AREAS.items() for p in ps} | {"core"} | set(PY_SHARED)
        assert set(c["python"]) == want_py
        assert set(c["cpp"]) == {e.name for e in CPP_ENGINES}
        for e in CPP_ENGINES:
            assert set(c["cpp_by_compiler"].get(e.name, {})) == CPP_PRESETS, e.name
        assert set(c["rust"]) == {e.name for e in RUST_ENGINES} | {"core"}
        assert set(c["bindings"]) == BINDINGS

    def test_no_failures_or_errors_anywhere(self):
        c = executed_counts()
        bad = {}
        for section in ("python_by_version", "cpp_by_compiler", "bindings_by_version"):
            for name, per in c[section].items():
                for variant, r in per.items():
                    if r["failed"] or r["errors"]:
                        bad[f"{section}:{name}:{variant}"] = r
        for name, r in c["rust"].items():
            if r["failed"] or r["errors"]:
                bad[f"rust:{name}"] = r
        assert not bad, bad
