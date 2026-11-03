"""Build and run the C++ and Rust replay tools on a recording.

``build_cpp()`` compiles tools/recorder/cpp/replay_dump.cpp against the
header-only cpp/ull/feed with g++ (native on Linux/macOS, through WSL on
Windows). ``build_rust()`` builds tools/recorder/rust with cargo. Each
returns a callable ``run(venue, recording, symbol) -> dump text`` or None
when the toolchain is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
Runner = Callable[[str, Path, str], str]


def _wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}"


def build_cpp(out_dir: Path) -> Runner | None:
    src = HERE / "cpp" / "replay_dump.cpp"
    inc = REPO / "cpp" / "ull" / "feed" / "include"
    flags = ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
    if sys.platform == "win32":
        if not shutil.which("wsl"):
            return None
        exe = _wsl_path(out_dir / "replay_dump")
        cmd = ["wsl", "-e", "g++", *flags, "-I", _wsl_path(inc), _wsl_path(src), "-o", exe]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return None

        def run_wsl(venue: str, rec: Path, sym: str) -> str:
            r = subprocess.run(
                ["wsl", "-e", exe, venue, _wsl_path(rec), sym], capture_output=True, check=True
            )
            return r.stdout.decode()

        return run_wsl
    gxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not gxx:
        return None
    exe_p = out_dir / "replay_dump"
    subprocess.run([gxx, *flags, "-I", str(inc), str(src), "-o", str(exe_p)], check=True)

    def run(venue: str, rec: Path, sym: str) -> str:
        return subprocess.run(
            [str(exe_p), venue, str(rec), sym], capture_output=True, check=True
        ).stdout.decode()

    return run


def build_rust(out_dir: Path) -> Runner | None:
    cargo = shutil.which("cargo")
    if not cargo:
        return None
    crate = HERE / "rust"
    target = out_dir / "cargo-target"
    r = subprocess.run(
        [
            cargo,
            "build",
            "--release",
            "--quiet",
            "--manifest-path",
            str(crate / "Cargo.toml"),
            "--target-dir",
            str(target),
        ],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode())
    exe = target / "release" / ("qp-recorder-replay.exe" if sys.platform == "win32" else "qp-recorder-replay")

    def run(venue: str, rec: Path, sym: str) -> str:
        return subprocess.run(
            [str(exe), venue, str(rec), sym], capture_output=True, check=True
        ).stdout.decode()

    return run
