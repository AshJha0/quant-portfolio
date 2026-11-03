"""Compile and run the C++ files of the career-prep packs.

Uses ``g++`` from PATH (CI, Linux, macOS with GCC). On a Windows machine
without g++ it falls back to ``wsl g++`` and runs the binary inside WSL.
When no compiler is found the C++ checks are skipped locally, but on CI
(``CI`` set in the environment) a missing compiler is a failure, so the
checks can never silently stop running there.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PureWindowsPath

import pytest

STD_FLAGS = [
    "-std=c++20",
    "-O1",
    "-g",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Werror",
    "-Wconversion",
    "-Wshadow",
]
SAN_FLAGS = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]


def _wsl_path(p: Path) -> str:
    w = PureWindowsPath(p)
    drive = w.drive.rstrip(":").lower()
    return "/mnt/" + drive + "/" + "/".join(w.parts[1:])


def _toolchain() -> list[str] | None:
    if shutil.which("g++"):
        return []
    if os.name == "nt" and shutil.which("wsl"):
        try:
            ok = (
                subprocess.run(["wsl", "-e", "g++", "--version"], capture_output=True, timeout=60).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        if ok:
            return ["wsl", "-e"]
    return None


_PREFIX = _toolchain()


def require_compiler() -> list[str]:
    if _PREFIX is None:
        if os.environ.get("CI"):
            pytest.fail("g++ is required on CI for the career-prep C++ checks")
        pytest.skip("no g++ (or wsl g++) on this machine")
    return _PREFIX


def _host(p: Path) -> str:
    return _wsl_path(p) if _PREFIX else str(p)


_BUILD = Path(tempfile.gettempdir()) / "qp-career-prep-cxx"


def compile_cpp(
    sources: list[Path], extra: list[str] | None = None, include: list[Path] | None = None
) -> Path:
    """Compile ``sources`` into one executable; cached on the content hash."""
    prefix = require_compiler()
    flags = STD_FLAGS + (extra if extra is not None else SAN_FLAGS)
    inc = [f"-I{_host(d)}" for d in (include or [])]
    h = hashlib.sha256(" ".join(flags + inc).encode())
    for s in sources:
        h.update(s.read_bytes())
    for d in include or []:
        for f in sorted(d.glob("*.hpp")):
            h.update(f.read_bytes())
    _BUILD.mkdir(exist_ok=True)
    exe = _BUILD / f"{sources[0].stem}-{h.hexdigest()[:16]}"
    if not exe.exists():
        cmd = [*prefix, "g++", *flags, *inc, *[_host(s) for s in sources], "-o", _host(exe), "-pthread"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert res.returncode == 0, f"compile failed:\n{' '.join(cmd)}\n{res.stderr}"
    return exe


def run(exe: Path, stdin: str = "", timeout: float = 300) -> str:
    cmd = [*(_PREFIX or []), _host(exe)]
    res = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout)
    assert res.returncode == 0, f"{exe.name} exited {res.returncode}:\n{res.stderr[-4000:]}"
    return res.stdout.replace("\r\n", "\n")
