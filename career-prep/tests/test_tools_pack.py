"""Tools pack: run each keyed command against its scenario and check the result."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1] / "tools"
KEY = dict(re.findall(r"^- \*\*(T\d\d)\*\* `([^`]+)`",
                      (PACK / "keys" / "answers.md").read_text("utf-8"), re.M))
GIT = shutil.which("git")


def _bash() -> str | None:
    # On Windows prefer Git Bash over the WSL launcher in System32.
    if GIT and os.name == "nt":
        cand = Path(GIT).resolve().parents[1] / "bin" / "bash.exe"
        if cand.exists():
            return str(cand)
    return shutil.which("bash")


BASH = _bash()
pytestmark = pytest.mark.skipif(GIT is None or BASH is None, reason="needs git and bash")

ENV = {**os.environ, "GIT_AUTHOR_NAME": "Ann", "GIT_AUTHOR_EMAIL": "ann@example.invalid",
       "GIT_COMMITTER_NAME": "Ann", "GIT_COMMITTER_EMAIL": "ann@example.invalid",
       "GIT_CONFIG_NOSYSTEM": "1", "HOME": "", "GIT_CONFIG_GLOBAL": os.devnull}


def git(repo: Path, *args: str) -> str:
    assert GIT
    return subprocess.run([GIT, *args], cwd=repo, env=ENV, capture_output=True, text=True,
                          check=True).stdout.strip()


def commit(repo: Path, name: str, text: str, msg: str) -> str:
    (repo / name).write_text(text, "utf-8", newline="\n")
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def run_key(repo: Path, tid: str) -> str:
    assert BASH
    res = subprocess.run([BASH, "-c", KEY[tid]], cwd=repo, env=ENV, capture_output=True,
                         text=True, check=False)
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "core.autocrlf", "false")
    commit(tmp_path, "notes.txt", "one\n", "first")
    return tmp_path


def t01(r: Path) -> None:
    first = git(r, "rev-parse", "HEAD")
    commit(r, "a.txt", "x\n", "second")
    run_key(r, "T01")
    assert git(r, "rev-parse", "HEAD") == first
    assert git(r, "diff", "--cached", "--name-only") == "a.txt"


def t02(r: Path) -> None:
    (r / "notes.txt").write_text("edited\n", "utf-8")
    (r / "other.txt").write_text("keep\n", "utf-8")
    run_key(r, "T02")
    assert (r / "notes.txt").read_text("utf-8") == "one\n"
    assert (r / "other.txt").exists()


def t03(r: Path) -> None:
    commit(r, "notes.txt", "two\n", "bad change")
    n = int(git(r, "rev-list", "--count", "HEAD"))
    run_key(r, "T03")
    assert int(git(r, "rev-list", "--count", "HEAD")) == n + 1
    assert (r / "notes.txt").read_text("utf-8") == "one\n"


def t04(r: Path) -> None:
    git(r, "switch", "-q", "-c", "feat")
    tip = commit(r, "f.txt", "f\n", "feature work")
    git(r, "switch", "-q", "main")
    git(r, "branch", "-q", "-D", "feat")
    run_key(r, "T04")
    assert git(r, "rev-parse", "feat") == tip


def t05(r: Path) -> None:
    (r / "check.sh").write_text("grep -q good value.txt\n", "utf-8", newline="\n")
    git(r, "add", "check.sh")
    commit(r, "value.txt", "good\n", "v1")
    git(r, "tag", "v1")
    for i in range(3):
        commit(r, f"n{i}.txt", "x\n", f"noise {i}")
    bad = commit(r, "value.txt", "broken\n", "the bug")
    for i in range(3, 6):
        commit(r, f"n{i}.txt", "x\n", f"noise {i}")
    out = run_key(r, "T05")
    # the wording of the final message varies across git versions; the ref does not
    assert git(r, "rev-parse", "refs/bisect/bad") == bad, out
    git(r, "bisect", "reset")


def t06(r: Path) -> None:
    commit(r, "prices.csv", "a,1\nb,2\nc,3\n", "prices")
    changed = commit(r, "prices.csv", "a,1\nb,2\nc,4\n", "fix c")
    assert run_key(r, "T06").split()[0] == changed


def t07(r: Path) -> None:
    (r / "notes.txt").write_text("half done\n", "utf-8")
    run_key(r, "T07")
    assert git(r, "status", "--porcelain") == ""
    assert "wip" in git(r, "stash", "list")


def t08(r: Path) -> None:
    (r / "x.txt").write_text("x\n", "utf-8")
    (r / "y.txt").write_text("y\n", "utf-8")
    git(r, "add", "x.txt", "y.txt")
    git(r, "commit", "-q", "-m", "two files")
    assert run_key(r, "T08").split() == ["x.txt", "y.txt"]


TRADES = "AAA,100\nBBB,50\nAAA,200\nCCC,10\nBBB,70\nAAA,30\n"


def _trades(r: Path) -> None:
    (r / "trades.csv").write_text(TRADES, "utf-8", newline="\n")


def t09(r: Path) -> None:
    _trades(r)
    assert run_key(r, "T09").split() == ["3", "AAA"]


def t10(r: Path) -> None:
    _trades(r)
    assert run_key(r, "T10") == str(sum(int(ln.split(",")[1]) for ln in TRADES.split()))


def t11(r: Path) -> None:
    (r / "logs" / "sub").mkdir(parents=True)
    (r / "logs" / "b.log").write_text("INFO\nERROR x\n", "utf-8")
    (r / "logs" / "a.log").write_text("INFO\n", "utf-8")
    (r / "logs" / "sub" / "c.log").write_text("ERROR y\nERROR z\n", "utf-8")
    assert run_key(r, "T11").split() == ["logs/b.log", "logs/sub/c.log"]


def t12(r: Path) -> None:
    _trades(r)
    assert run_key(r, "T12") == "30"


SCENARIOS: dict[str, Callable[[Path], None]] = {
    "T01": t01, "T02": t02, "T03": t03, "T04": t04, "T05": t05, "T06": t06,
    "T07": t07, "T08": t08, "T09": t09, "T10": t10, "T11": t11, "T12": t12,
}


def test_every_task_is_keyed_and_tested() -> None:
    q = re.findall(r"^- \*\*(T\d\d)\*\*", (PACK / "questions.md").read_text("utf-8"), re.M)
    assert len(q) == 12 and set(q) == set(KEY) == set(SCENARIOS)


@pytest.mark.parametrize("tid", sorted(SCENARIOS))
def test_key_command_does_the_task(tid: str, repo: Path) -> None:
    SCENARIOS[tid](repo)
