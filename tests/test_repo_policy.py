"""Repository policy checks (W8, W12 Round-6 slice).

* One version: every package, both bindings and every ``__version__`` equal
  the top-level ``VERSION`` file (docs/VERSIONING.md).
* No Nasdaq-family data source (Nasdaq Data Link, LOBSTER, ITCH sample
  files) is named outside the excluded-sources note and the history files.
* The ``ci`` Hypothesis profile is deterministic: two runs on one commit
  draw identical examples (conftest.py).

Run from the repository root::

    python -m pytest -p no:cacheprovider -q tests/test_repo_policy.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

REPO = Path(__file__).resolve().parents[1]
VERSION = (REPO / "VERSION").read_text(encoding="utf-8").strip()


def _git_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


# ------------------------------------------------------------------ version --
def _pyprojects() -> list[Path]:
    return sorted(p for p in REPO.glob("**/pyproject.toml") if ".venv" not in p.parts and p.parent != REPO)


# __init__.py files whose package belongs to another lane this round; the
# integration lane sets their __version__ line when the lanes merge
# (review/2026-11-03/handbacks/r6/L4.md). Remove each entry once done.
VERSION_PENDING: set[str] = set()


class TestOneVersion:
    def test_version_file_is_pep440(self):
        assert str(Version(VERSION)) == VERSION

    def test_every_pyproject_has_the_repo_version(self):
        projects = _pyprojects()
        # python/core, python/data, the four shared stacks (backtest, oms,
        # execution, posttrade), 23 projects, 2 bindings
        assert len(projects) == 31
        bad = {}
        for p in projects:
            v = tomllib.loads(p.read_text(encoding="utf-8"))["project"]["version"]
            if Version(v) != Version(VERSION):
                bad[str(p.relative_to(REPO))] = v
        assert not bad, bad

    def test_rust_binding_crate_has_the_repo_version(self):
        cargo = tomllib.loads((REPO / "bindings/rust-pyo3/Cargo.toml").read_text(encoding="utf-8"))
        # Cargo spells the pre-release SemVer style (2.1.0-dev.0); compare as PEP 440.
        assert Version(cargo["package"]["version"]) == Version(VERSION)
        lock = (REPO / "bindings/rust-pyo3/Cargo.lock").read_text(encoding="utf-8")
        m = re.search(r'name = "qp-rust"\nversion = "([^"]+)"', lock.replace("\r\n", "\n"))
        assert m and m.group(1) == cargo["package"]["version"]

    def test_every_dunder_version_matches(self):
        bad, pending_ok = {}, set()
        for rel in _git_files():
            if not rel.endswith("__init__.py") or not rel.startswith(("python/", "bindings/")):
                continue
            m = re.search(r'^__version__ = "([^"]+)"', (REPO / rel).read_text(encoding="utf-8"), re.M)
            if not m:
                continue
            if Version(m.group(1)) == Version(VERSION):
                if rel in VERSION_PENDING:
                    pending_ok.add(rel)
            elif rel not in VERSION_PENDING:
                bad[rel] = m.group(1)
        assert not bad, bad
        # A pending entry that is already fixed must be removed from the list.
        assert not pending_ok, f"remove from VERSION_PENDING: {sorted(pending_ok)}"

    def test_changelog_has_an_unreleased_section_while_the_version_is_a_dev_release(self):
        changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
        if Version(VERSION).is_devrelease:
            assert "## [Unreleased]" in changelog
        else:
            assert f"## [{VERSION}]" in changelog

    def test_migration_notes_cover_the_version(self):
        migration = (REPO / "docs/MIGRATION.md").read_text(encoding="utf-8")
        v = Version(VERSION)
        if v.is_devrelease:
            assert f"## Unreleased (toward v{v.base_version})" in migration
        else:
            assert f"## {v.base_version}" in migration


# ------------------------------------------------------- excluded sources --
# The owner excluded Nasdaq Data Link, LOBSTER and Nasdaq ITCH sample files
# (plan section 2). The names may appear only in the excluded-sources note,
# in history that must not be rewritten, and in this test.
EXCLUDED_SOURCES = re.compile(r"nasdaq|lobster|\bITCH\b", re.I)
ALLOWED = {
    "data/README.md",  # the excluded-sources note
    "CHANGELOG.md",  # release history is not rewritten
    "tests/test_repo_policy.py",
}
# Files that still name an excluded source and are owned by other work
# (removal is tracked in review/2026-11-03/handbacks/r6/L4.md). The list
# only shrinks: an entry that no longer matches fails the test.
PENDING_REMOVAL: set[str] = set()


def test_excluded_data_sources_are_not_named_elsewhere():
    hits = set()
    for rel in _git_files():
        p = REPO / rel
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        if EXCLUDED_SOURCES.search(rel) or EXCLUDED_SOURCES.search(text):
            hits.add(rel)
    new = sorted(hits - ALLOWED - PENDING_REMOVAL)
    assert not new, f"excluded data source named in: {new}"
    cleaned = sorted(PENDING_REMOVAL - hits)
    assert not cleaned, f"no longer names an excluded source; remove from PENDING_REMOVAL: {cleaned}"


# ------------------------------------------------------ hypothesis profiles --
PROBE = '''
import os
from hypothesis import given, strategies as st

@given(st.floats(-1e6, 1e6), st.integers(0, 10**9))
def test_probe(x, n):
    with open(os.environ["QP_PROBE_OUT"], "a", encoding="utf-8") as fh:
        fh.write(f"{x!r} {n}\\n")
'''


def _run_probe(tmp: Path, out: Path, profile: str) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k not in {"CI", "QP_HYPOTHESIS_PROFILE"}}
    env.update(QP_HYPOTHESIS_PROFILE=profile, QP_PROBE_OUT=str(out))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", f"--confcutdir={REPO}",
         f"--rootdir={tmp}", str(tmp / "test_probe.py")],
        cwd=tmp, env=env, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    lines = out.read_text(encoding="utf-8").splitlines()
    out.unlink()
    return lines


@pytest.fixture
def probe_dir() -> Path:
    # Inside the repository so the root conftest.py is an ancestor.
    d = REPO / "tests" / "_hypothesis_probe"
    d.mkdir(exist_ok=True)
    (d / "test_probe.py").write_text(PROBE, encoding="utf-8")
    yield d
    shutil.rmtree(d)


class TestHypothesisProfiles:
    def test_ci_profile_draws_identical_examples_twice(self, probe_dir):
        out = probe_dir / "examples.txt"
        first = _run_probe(probe_dir, out, "ci")
        second = _run_probe(probe_dir, out, "ci")
        assert len(first) >= 50
        assert first == second

    def test_nightly_profile_is_random(self, probe_dir):
        out = probe_dir / "examples.txt"
        assert _run_probe(probe_dir, out, "nightly") != _run_probe(probe_dir, out, "nightly")

    def test_profile_choice(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("qp_root_conftest", REPO / "conftest.py")
        assert spec and spec.loader
        root_conftest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(root_conftest)
        choose = root_conftest.chosen_profile
        assert choose({}) == "dev"
        assert choose({"CI": "true"}) == "ci"
        assert choose({"CI": "true", "QP_HYPOTHESIS_PROFILE": "nightly"}) == "nightly"
        with pytest.raises(ValueError):
            choose({"QP_HYPOTHESIS_PROFILE": "fast"})


# ------------------------------------------------------------- api diff --
def _api_diff():
    import importlib.util

    spec = importlib.util.spec_from_file_location("qp_api_diff", REPO / ".github/scripts/api_diff.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestApiDiff:
    def test_dunder_all_reads_assign_and_augmented_assign(self):
        src = '__all__ = ["a", "b"]\n__all__ += ["c"]\nx = 1\n'
        assert _api_diff().dunder_all(src) == {"a", "b", "c"}

    def test_dunder_all_is_none_when_computed_or_absent(self):
        mod = _api_diff()
        assert mod.dunder_all("__all__ = sorted(names)\n") is None
        assert mod.dunder_all("x = 1\n") is None

    def test_every_package_has_a_static_dunder_all(self):
        # The removed-name check reads __all__ statically; a computed one
        # would silently fall back to griffe alone.
        mod = _api_diff()
        pkgs = mod.packages()
        assert len(pkgs) == 24
        for name, src in pkgs:
            init = (REPO / src / name / "__init__.py").read_text(encoding="utf-8")
            assert mod.dunder_all(init), name


def test_docs_are_valid_utf8():
    """GitHub Pages builds docs/ with Jekyll; one bad byte fails the whole site."""
    root = Path(__file__).resolve().parents[1] / "docs"
    bad = []
    for path in root.rglob("*"):
        if path.suffix.lower() in {".md", ".html"} and path.is_file():
            try:
                path.read_bytes().decode("utf-8")
            except UnicodeDecodeError as exc:
                bad.append(f"{path.relative_to(root)}: {exc}")
    assert not bad, bad
