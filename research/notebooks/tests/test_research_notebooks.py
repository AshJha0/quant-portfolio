"""The research notebooks: fresh, deterministic, checks pass, solutions hidden."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_notebooks  # noqa: E402

NAMES = sorted(build_notebooks.NOTEBOOKS)


def _code(path: Path) -> list[str]:
    nb = json.loads(path.read_text("utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _run(path: Path) -> str:
    ns: dict[str, object] = {"__name__": "__main__"}
    out = io.StringIO()
    old = Path.cwd()
    os.chdir(HERE)
    try:
        with contextlib.redirect_stdout(out):
            for src in _code(path):
                exec(compile(src, str(path), "exec"), ns)
    finally:
        os.chdir(old)
    return out.getvalue()


def test_committed_notebooks_match_the_builder() -> None:
    assert build_notebooks.main(["--check"]) == 0


@pytest.mark.parametrize("name", NAMES)
def test_solution_runs_and_is_deterministic(name: str) -> None:
    path = HERE / f"{name}-solution.ipynb"
    first = _run(path)
    assert first == _run(path)


@pytest.mark.parametrize("name", NAMES)
def test_exercise_hides_solutions(name: str) -> None:
    cells = build_notebooks.NOTEBOOKS[name]
    ex = "\n".join(_code(HERE / f"{name}-exercise.ipynb"))
    shared = "\n".join(c[1] for c in cells if c[0] in ("code", "check"))
    tasks = [c for c in cells if c[0] == "task"]
    assert tasks
    for _, solution, stub in tasks:
        assert stub in ex
        for line in solution.splitlines():
            if line.strip() and line not in stub and line not in shared:
                assert line not in ex, f"solution line leaked: {line!r}"
