"""TEST-12: the numbers quoted in the portfolio docs match the test runs.

Every headline figure (sub-projects, tests per area, questions in the
LEARN.md self-test bank, recipes in COOKBOOK.md, the per-engine
cross-validation table, the per-project README test counts) is compared
with the prose, so a stale count fails CI instead of surviving in a README.

Where the test counts come from:

* CI (authoritative). The ``counts`` job aggregates the JUnit XML and cargo
  logs of every test job into ``counts.json`` and points ``QP_COUNTS_JSON``
  at it (a ``counts.json`` at the repository root is also picked up). The
  counts are tests that were executed and passed: Python on the reference
  version (3.12), C++ with the reference compiler (GCC), Rust from
  ``cargo test --release``.
* Local fallback (approximate, for a quick check only). Without
  ``counts.json`` the counts are estimated: Python by
  ``pytest --collect-only`` per project with ``PYTHONPATH=src``, C++ by
  counting ``TEST`` / ``TEST_F`` macros (``TEST_P`` times the golden case
  count), Rust by counting ``#[test]`` functions plus rustdoc doctests.
  Collected is not passed: a skipped test is collected but not passed, so
  the fallback can disagree with CI. A warning says so.

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
import warnings
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
CPP_ENGINES = sorted(p for p in (REPO / "cpp").iterdir() if p.is_dir())
RUST_ENGINES = sorted(p for p in (REPO / "rust").iterdir() if p.is_dir())


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


def _executed_passed(section: str, key: str) -> int:
    counts = executed_counts()
    assert counts is not None
    entry = counts[section].get(key)
    assert entry is not None, f"counts.json has no {section} entry for {key}: the job did not run or upload"
    return int(entry["passed"])


def _warn_fallback() -> None:
    warnings.warn("counts.json not found: using LOCAL FALLBACK counts (collected / source-scanned, "
                  "not executed). CI uses executed pass counts.", stacklevel=2)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _num(s: str) -> int:
    return int(s.replace(",", ""))


# ------------------------------------------- local fallback counters --
@cache
def collect_python(project: Path) -> int:
    """LOCAL FALLBACK ONLY: collected (not executed) tests in one project."""
    env = dict(os.environ, PYTHONPATH="src")
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=",
         "--collect-only", "-q"],
        cwd=project, env=env, capture_output=True, text=True, timeout=600,
    )
    m = re.search(r"(\d+) tests? collected", out.stdout + out.stderr)
    assert m, f"could not parse collection output for {project}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
    assert "error" not in out.stdout.lower().split("collected")[-1], out.stdout[-2000:]
    return int(m.group(1))


def count_cpp(engine: Path) -> int:
    n = 0
    for f in sorted((engine / "tests").glob("*.cpp")):
        src = _read(f)
        plain = len(re.findall(r"^\s*TEST(?:_F)?\(", src, re.M))
        param = len(re.findall(r"^\s*TEST_P\(", src, re.M))
        n += plain
        if param:
            assert "INSTANTIATE_TEST_SUITE_P" in src and "kCases.size()" in src, f
            # instantiated over the golden case list of the twin Python project
            twin = {
                "equity-options-engine": "python/equity/01-options-pricing",
                "fx-options-engine": "python/fx/01-options-pricing",
                "equity-var-engine": "python/equity/03-var-es-engine",
                "fx-var-engine": "python/fx/03-var-es-engine",
            }[engine.name]
            golden = json.loads(_read(REPO / twin / "tests/golden/golden_vectors.json"))
            n += param * golden["n_cases"]
    return n


def count_rust(engine: Path) -> tuple[int, int]:
    files = list((engine / "src").rglob("*.rs")) + list((engine / "tests").rglob("*.rs"))
    n_test = sum(len(re.findall(r"^\s*#\[test\]", _read(f), re.M)) for f in files)
    n_doc = 0
    skip_langs = {"ignore", "text", "no_run", "sh", "bash", "toml", "json", "console"}
    for f in (engine / "src").rglob("*.rs"):
        in_block = False
        for ln in _read(f).splitlines():
            m = re.match(r"\s*//[/!]\s?(.*)", ln)
            if not m:
                continue
            body = m.group(1).strip()
            if body.startswith("```"):
                if not in_block:
                    in_block = True
                    info = {t.strip() for t in body[3:].split(",")} - {""}
                    if not (info & skip_langs):
                        n_doc += 1
                else:
                    in_block = False
    return n_test, n_doc


def count_questions() -> int:
    return len(re.findall(r"^\*\*Q\d+\.", _read(DOCS / "LEARN.md"), re.M))


def count_recipes() -> int:
    heads = re.findall(r"^### (.+)$", _read(DOCS / "COOKBOOK.md"), re.M)
    return sum(1 for h in heads if h.startswith(("How do I", "What happens")))


# ----------------------------------------------------------------- fixtures --
@pytest.fixture(scope="module")
def py_counts() -> dict[str, dict[str, int]]:
    if executed_counts() is not None:
        return {area: {p.name: _executed_passed("python", f"{area}__{p.name}") for p in projects}
                for area, projects in PY_AREAS.items()}
    _warn_fallback()
    return {area: {p.name: collect_python(p) for p in projects} for area, projects in PY_AREAS.items()}


@pytest.fixture(scope="module")
def cpp_counts() -> dict[str, int]:
    if executed_counts() is not None:
        return {e.name: _executed_passed("cpp", e.name) for e in CPP_ENGINES}
    _warn_fallback()
    return {e.name: count_cpp(e) for e in CPP_ENGINES}


@pytest.fixture(scope="module")
def rust_counts() -> dict[str, int]:
    if executed_counts() is not None:
        return {e.name: _executed_passed("rust", e.name) for e in RUST_ENGINES}
    _warn_fallback()
    return {e.name: sum(count_rust(e)) for e in RUST_ENGINES}


@pytest.fixture(scope="module")
def readme() -> str:
    return _read(README)


def _readme_area_table(readme: str) -> dict[str, int]:
    rows = re.findall(r"^\| (Python — equity|Python — FX|Python — foundations|C\+\+ engines|Rust engines|\*\*Total\*\*) \| \*?\*?(\d+)\*?\*? \| \*?\*?([\d,]+)\*?\*? \|",
                      readme, re.M)
    assert len(rows) == 6, rows
    return {name: _num(tests) for name, _, tests in rows}


def _readme_engine_table(readme: str) -> dict[str, tuple[int, int, int]]:
    rows = re.findall(r"^\| (Equity options/Greeks|FX options/Greeks|Equity VaR/ES|FX VaR/ES) \| `([^`]+)` — ([\d,]+) \| ([\d,]+) \| ([\d,]+) \|",
                      readme, re.M)
    assert len(rows) == 4, rows
    return {path: (_num(py), _num(cpp), _num(rs)) for _, path, py, cpp, rs in rows}


# -------------------------------------------------------------------- tests --
class TestStructure:
    def test_31_sub_projects(self, readme):
        n = sum(len(v) for v in PY_AREAS.values()) + len(CPP_ENGINES) + len(RUST_ENGINES)
        assert n == 31
        assert "31 sub-projects" in readme
        assert "| **Total** | **31** |" in readme

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
    def test_question_bank_count_everywhere(self):
        n = count_questions()
        learn = _read(DOCS / "LEARN.md")
        assert f"**{n} questions across" in learn
        assert f"| **Total** | | **{n}** |" in learn
        assert f"{n} self-test questions" in learn
        for f in (README, DOCS / "README.md", DOCS / "ARCHITECTURE.md", DOCS / "index.html"):
            assert f"{n}-question" in _read(f) or f"{n} self-test questions" in _read(f), f

    def test_learn_rounds_add_up(self):
        learn = _read(DOCS / "LEARN.md")
        # "| 1 | Quant & Math Foundations | 28 |" ... "| **Total** | | **481** |"
        rows = re.findall(r"^\| (\d+) \| [^|]+ \| (\d+) \|$", learn, re.M)
        assert len(rows) == 18, "round table not found"
        assert sum(int(n) for _, n in rows) == count_questions()
        # and the per-round headings contain exactly that many questions each
        per_round = re.split(r"^#{2,4} Round (\d+)\b.*$", learn, flags=re.M)
        if len(per_round) > 1:
            counts = {int(per_round[i]): len(re.findall(r"^\*\*Q\d+\.", per_round[i + 1], re.M))
                      for i in range(1, len(per_round), 2)}
            assert counts == {int(r): int(n) for r, n in rows}, counts

    def test_recipe_count_everywhere(self):
        n = count_recipes()
        assert f"{n} copy-pasteable" in _read(DOCS / "README.md")
        assert f"{n} recipes" in _read(DOCS / "index.html")
        assert "61 recipes" not in _read(DOCS / "index.html")


class TestTestCounts:
    def test_python_area_totals_match_readme(self, py_counts, readme):
        table = _readme_area_table(readme)
        assert table["Python — equity"] == sum(py_counts["equity"].values())
        assert table["Python — FX"] == sum(py_counts["fx"].values())
        assert table["Python — foundations"] == sum(py_counts["foundations"].values())

    def test_cpp_and_rust_totals_match_readme(self, cpp_counts, rust_counts, readme):
        table = _readme_area_table(readme)
        assert table["C++ engines"] == sum(cpp_counts.values()), cpp_counts
        assert table["Rust engines"] == sum(rust_counts.values()), rust_counts

    def test_grand_total_is_consistent_everywhere(self, py_counts, cpp_counts, rust_counts, readme):
        total = (sum(sum(v.values()) for v in py_counts.values())
                 + sum(cpp_counts.values()) + sum(rust_counts.values()))
        table = _readme_area_table(readme)
        assert table["**Total**"] == total
        pretty = f"{total:,}"
        assert f"**{pretty} passing tests**" in readme or f"{pretty} passing tests" in readme
        assert f"{pretty} tests across the portfolio" in readme
        for f in (DOCS / "index.html", DOCS / "README.md", DOCS / "LEARN.md",
                  DOCS / "DIAGRAMS.md", DOCS / "COOKBOOK.md"):
            assert pretty in _read(f), f
        # no stale total anywhere in the docs layer
        stale = {"5,963", "5,943", "5,913"} - {pretty}
        for f in (README, *DOCS.glob("*.md"), DOCS / "index.html"):
            txt = _read(f)
            for s in stale:
                assert s not in txt, (f, s)

    def test_per_engine_table_matches(self, py_counts, cpp_counts, rust_counts, readme):
        table = _readme_engine_table(readme)
        expect = {
            "python/equity/01-options-pricing": (py_counts["equity"]["01-options-pricing"],
                                                 cpp_counts["equity-options-engine"],
                                                 rust_counts["equity-options-engine"]),
            "python/fx/01-options-pricing": (py_counts["fx"]["01-options-pricing"],
                                             cpp_counts["fx-options-engine"],
                                             rust_counts["fx-options-engine"]),
            "python/equity/03-var-es-engine": (py_counts["equity"]["03-var-es-engine"],
                                               cpp_counts["equity-var-engine"],
                                               rust_counts["equity-var-engine"]),
            "python/fx/03-var-es-engine": (py_counts["fx"]["03-var-es-engine"],
                                           cpp_counts["fx-var-engine"],
                                           rust_counts["fx-var-engine"]),
        }
        assert table == expect

    def test_index_html_area_numbers(self, py_counts, cpp_counts, rust_counts):
        html = _read(DOCS / "index.html")
        for label, n in (
            ("equity", sum(py_counts["equity"].values())),
            ("FX", sum(py_counts["fx"].values())),
            ("foundations", sum(py_counts["foundations"].values())),
            ("C++", sum(cpp_counts.values())),
            ("Rust", sum(rust_counts.values())),
        ):
            assert f"{n:,}" in html, (label, n)

    def test_per_project_readme_counts(self, py_counts, cpp_counts, rust_counts):
        """The per-project READMEs quote a test count next to their test
        command; each must equal what the tree actually contains."""
        checks = {
            REPO / "python/equity/01-options-pricing/README.md": py_counts["equity"]["01-options-pricing"],
            REPO / "python/equity/03-var-es-engine/README.md": py_counts["equity"]["03-var-es-engine"],
            REPO / "python/fx/01-options-pricing/README.md": py_counts["fx"]["01-options-pricing"],
            REPO / "python/fx/03-var-es-engine/README.md": py_counts["fx"]["03-var-es-engine"],
            REPO / "cpp/equity-options-engine/README.md": cpp_counts["equity-options-engine"],
            REPO / "cpp/equity-var-engine/README.md": cpp_counts["equity-var-engine"],
            REPO / "cpp/fx-options-engine/README.md": cpp_counts["fx-options-engine"],
            REPO / "cpp/fx-var-engine/README.md": cpp_counts["fx-var-engine"],
        }
        for f, n in checks.items():
            txt = _read(f)
            quoted = {int(x) for x in re.findall(r"#\s*(\d+) tests", txt)}
            assert quoted == {n}, (f, quoted, n)
        for eng in RUST_ENGINES:
            n_test, n_doc = count_rust(eng)
            txt = _read(eng / "README.md")
            m = re.search(r"#\s*(\d+) (?:integration )?tests \+ (\d+) doctests \((\d+)\)", txt)
            assert m, eng
            assert (int(m.group(1)), int(m.group(2)), int(m.group(3))) == (n_test, n_doc, n_test + n_doc), (eng, m.groups())

    def test_cookbook_sweep_numbers(self, py_counts, cpp_counts, rust_counts):
        cb = _read(DOCS / "COOKBOOK.md")
        eq, fx, fo = (sum(py_counts[a].values()) for a in ("equity", "fx", "foundations"))
        assert f"equity `{eq:,}`, FX `{fx:,}`" in cb
        assert f"foundations `{fo:,}`" in cb
        assert f"**total `{eq + fx + fo:,}`** Python tests" in cb
        assert f"plus the {sum(cpp_counts.values())} C++" in cb
        assert f"and {sum(rust_counts.values())} Rust engine tests" in cb


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
        want_py = {f"{a}__{p.name}" for a, ps in PY_AREAS.items() for p in ps}
        assert set(c["python"]) == want_py
        assert set(c["cpp"]) == {e.name for e in CPP_ENGINES}
        assert set(c["rust"]) == {e.name for e in RUST_ENGINES}

    def test_no_failures_or_errors_anywhere(self):
        c = executed_counts()
        bad = {}
        for section in ("python_by_version", "cpp_by_compiler"):
            for name, per in c[section].items():
                for variant, r in per.items():
                    if r["failed"] or r["errors"]:
                        bad[f"{section}:{name}:{variant}"] = r
        for name, r in c["rust"].items():
            if r["failed"] or r["errors"]:
                bad[f"rust:{name}"] = r
        assert not bad, bad
