"""Tests for the hash-checked CI install path.

Covers .github/scripts/check_constraints_lock.py (pin parsing, agreement and
disagreement), the argument checks in .github/scripts/hashed_install.sh, and
a guard that every pip install in the workflows either goes through that
script or uses --no-deps (so nothing unhashed is resolved from the index).

Run from the repository root::

    python -m pytest -p no:cacheprovider -q tests/test_ci_install.py
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / ".github" / "scripts"
WORKFLOWS = REPO / ".github" / "workflows"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ccl = _load("check_constraints_lock")

EXPORT = """\
numpy==2.4.6 ; python_full_version < '3.12' \\
    --hash=sha256:aa
numpy==2.5.4 ; python_full_version >= '3.12' \\
    --hash=sha256:bb
Scikit_Learn==1.9.1 \\
    --hash=sha256:cc
    # via eq-regime
"""


def test_pins_reads_markers_hashes_and_normalises_names() -> None:
    got = ccl.pins(EXPORT)
    assert got == {
        ("numpy", "python_full_version < '3.12'"): "2.4.6",
        ("numpy", "python_full_version >= '3.12'"): "2.5.4",
        ("scikit-learn", ""): "1.9.1",
    }


def test_pins_ignores_crlf_and_double_quoted_markers() -> None:
    text = 'numpy==2.5.4 ; python_full_version >= "3.12"\r\n    # via x\r\n'
    assert ccl.pins(text) == {("numpy", "python_full_version >= '3.12'"): "2.5.4"}


def _run_main(monkeypatch, tmp_path: Path, constraints: str, exported: str) -> int:
    cons = tmp_path / "ci-constraints.txt"
    cons.write_text(constraints, encoding="utf-8")
    monkeypatch.setattr(ccl, "CONSTRAINTS", cons)

    class Done:
        stdout = exported

    monkeypatch.setattr(ccl.subprocess, "run", lambda *a, **k: Done())
    return ccl.main()


def test_main_passes_when_the_files_agree(monkeypatch, tmp_path, capsys) -> None:
    cons = "numpy==2.4.6 ; python_full_version < '3.12'\n" \
           "numpy==2.5.4 ; python_full_version >= '3.12'\nscikit-learn==1.9.1\n"
    assert _run_main(monkeypatch, tmp_path, cons, EXPORT) == 0
    assert "agree on 3 pins" in capsys.readouterr().out


def test_main_fails_on_a_version_or_membership_difference(monkeypatch, tmp_path, capsys) -> None:
    cons = "numpy==2.4.6 ; python_full_version < '3.12'\n" \
           "numpy==2.5.3 ; python_full_version >= '3.12'\nsetuptools==84.2.1\n"
    assert _run_main(monkeypatch, tmp_path, cons, EXPORT) == 1
    out = capsys.readouterr().out
    assert "numpy ; python_full_version >= '3.12': ci-constraints.txt 2.5.3, uv.lock 2.5.4" in out
    assert "scikit-learn: ci-constraints.txt -, uv.lock 1.9.1" in out
    assert "setuptools: ci-constraints.txt 84.2.1, uv.lock -" in out


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_committed_constraints_match_the_committed_lock(monkeypatch) -> None:
    monkeypatch.chdir(REPO)
    assert ccl.main() == 0


BASH = shutil.which("bash")


@pytest.mark.skipif(BASH is None or sys.platform == "win32", reason="needs a POSIX bash")
def test_hashed_install_rejects_an_unknown_option() -> None:
    r = subprocess.run([BASH, str(SCRIPTS / "hashed_install.sh"), "--bogus"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 2
    assert "unknown option --bogus" in r.stderr


@pytest.mark.skipif(BASH is None or sys.platform == "win32", reason="needs a POSIX bash")
def test_hashed_install_rejects_a_path_without_a_project_name(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.x]\n", encoding="utf-8")
    r = subprocess.run([BASH, str(SCRIPTS / "hashed_install.sh"), str(tmp_path)],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 1
    assert "no project name" in r.stderr


def test_every_workflow_pip_install_is_hash_checked_or_no_deps() -> None:
    """A bare `pip install x` would resolve x and its deps from the index unhashed."""
    offenders = []
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        lines = wf.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith(("- name:", "#")):
                continue
            if re.search(r"\b(pip|uv pip) install\b", line):
                cmd = line
                j = i
                while cmd.rstrip().endswith("\\") and j + 1 < len(lines):
                    j += 1
                    cmd += lines[j]
                if "--no-deps" not in cmd and "--require-hashes" not in cmd:
                    offenders.append(f"{wf.name}:{i + 1}: {line.strip()}")
    assert offenders == []


def test_hashed_install_script_requires_hashes_wheels_and_no_deps() -> None:
    text = (SCRIPTS / "hashed_install.sh").read_text(encoding="utf-8")
    assert "--require-hashes --only-binary :all: --no-deps" in text
    assert "--no-deps --no-build-isolation" in text
    assert "pip check" in text
