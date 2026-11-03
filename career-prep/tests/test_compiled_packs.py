"""Rust and Java packs: compile every program and compare with the key.

A key that is an error code (``E0382``) or ``compile error`` must fail to
compile (with that code, for Rust); any other key is the program's exact
stdout with lines joined by " | ". The solutions files run their own tests.
Each language is skipped when its toolchain is not on PATH.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUSTC = shutil.which("rustc")
JAVA = shutil.which("java")


def _programs(pack: str, prefix: str, lang: str) -> dict[str, str]:
    text = (ROOT / pack / "questions.md").read_text("utf-8")
    return dict(re.findall(rf"^### ({prefix}\d\d) [^\n]*\n```{lang}\n(.*?)```", text, re.M | re.S))


def _key(pack: str, prefix: str) -> dict[str, str]:
    text = (ROOT / pack / "keys" / "answers.md").read_text("utf-8")
    return dict(re.findall(rf"^- \*\*({prefix}\d\d)\*\* `([^`]+)`", text, re.M))


RUST = _programs("rust", "R", "rust")
RUST_KEY = _key("rust", "R")
JAVA_SRC = _programs("java", "J", "java")
JAVA_KEY = _key("java", "J")


def _joined(out: str) -> str:
    return " | ".join(out.strip().splitlines())


def test_every_program_is_keyed() -> None:
    assert len(RUST) == 14 and set(RUST) == set(RUST_KEY)
    assert len(JAVA_SRC) == 12 and set(JAVA_SRC) == set(JAVA_KEY)


@pytest.mark.skipif(RUSTC is None, reason="rustc not installed")
@pytest.mark.parametrize("pid", sorted(RUST))
def test_rust_program_matches_key(pid: str, tmp_path: Path) -> None:
    assert RUSTC
    src = tmp_path / f"{pid.lower()}.rs"
    src.write_text(RUST[pid], "utf-8")
    exe = tmp_path / pid.lower()
    res = subprocess.run([RUSTC, "--edition", "2021", "-A", "warnings", "-o", str(exe), str(src)],
                         capture_output=True, text=True, timeout=300, check=False)
    want = RUST_KEY[pid]
    if re.fullmatch(r"E\d{4}", want):
        assert res.returncode != 0, f"{pid} compiled but the key says {want}"
        codes = set(re.findall(r"error\[(E\d{4})\]", res.stderr))
        assert want in codes, (pid, codes)
        return
    assert res.returncode == 0, res.stderr
    try:
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60, check=True)
    except OSError as err:  # Windows Smart App Control can refuse a fresh, unsigned binary
        if getattr(err, "winerror", None) == 4551:
            pytest.skip("binary blocked by the OS application-control policy")
        raise
    assert _joined(run.stdout) == want


@pytest.mark.skipif(RUSTC is None, reason="rustc not installed")
def test_rust_solutions(tmp_path: Path) -> None:
    assert RUSTC
    exe = tmp_path / "solutions_test"
    subprocess.run([RUSTC, "--edition", "2021", "--test", "-o", str(exe),
                    str(ROOT / "rust" / "solutions.rs")], check=True, timeout=300)
    res = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120, check=False)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "3 passed" in res.stdout


@pytest.mark.skipif(JAVA is None, reason="java not installed")
@pytest.mark.parametrize("pid", sorted(JAVA_SRC))
def test_java_program_matches_key(pid: str, tmp_path: Path) -> None:
    assert JAVA
    src = tmp_path / "Main.java"
    src.write_text(JAVA_SRC[pid], "utf-8")
    res = subprocess.run([JAVA, str(src)], capture_output=True, text=True, timeout=120,
                         check=False)
    want = JAVA_KEY[pid]
    if want == "compile error":
        assert res.returncode != 0 and "error:" in res.stderr, (pid, res.stdout, res.stderr)
        return
    assert res.returncode == 0, res.stderr
    assert _joined(res.stdout) == want


@pytest.mark.skipif(JAVA is None, reason="java not installed")
def test_java_solutions() -> None:
    assert JAVA
    res = subprocess.run([JAVA, "-ea", str(ROOT / "java" / "Solutions.java")],
                         capture_output=True, text=True, timeout=300, check=False)
    assert res.returncode == 0 and res.stdout.strip() == "ok", res.stdout + res.stderr
