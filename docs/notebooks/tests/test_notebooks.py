"""The learning notebooks: fresh, deterministic, and their checks pass.

CI also runs the solution notebooks through a real kernel with nbmake (see
the handback for the workflow step); this test needs no Jupyter install. It
executes every code cell of each solution notebook in order, in one
namespace, with the notebook folder as the working directory, which is what
a kernel does for code without magics.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_notebooks  # noqa: E402

NAMES = sorted(build_notebooks.NOTEBOOKS)
TIME_BUDGET_S = 60.0


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


def test_eight_notebooks() -> None:
    assert [n[3:] for n in NAMES] == [
        "black-scholes-implied-vol", "value-at-risk", "yield-curve-bootstrap",
        "optimal-execution", "aad-greeks", "market-making", "cva", "risk-parity"]


def test_committed_notebooks_match_the_builder() -> None:
    assert build_notebooks.main(["--check"]) == 0


@pytest.mark.parametrize("name", NAMES)
def test_solution_runs_and_is_deterministic(name: str) -> None:
    path = HERE / f"{name}-solution.ipynb"
    t0 = time.perf_counter()
    first = _run(path)
    assert time.perf_counter() - t0 < TIME_BUDGET_S
    assert first == _run(path), "two runs printed different output"


@pytest.mark.parametrize("name", NAMES)
def test_exercise_hides_solutions_but_keeps_checks(name: str) -> None:
    cells = build_notebooks.NOTEBOOKS[name]
    ex = "\n".join(_code(HERE / f"{name}-exercise.ipynb"))
    tasks = [c for c in cells if c[0] == "task"]
    checks = [c for c in cells if c[0] == "check"]
    assert tasks and checks
    shared = "\n".join(c[1] for c in cells if c[0] in ("code", "check"))
    for _, solution, stub in tasks:
        assert stub in ex
        for line in solution.splitlines():
            if line.strip() and line not in stub and line not in shared:
                assert line not in ex, f"solution line leaked: {line!r}"
    for _, text, _ in checks:
        assert "assert" in text and text in ex


@pytest.mark.parametrize("name", NAMES)
def test_exercise_stub_fails_its_check(name: str) -> None:
    """An untouched exercise must not pass: the checks really test something."""
    with pytest.raises(Exception):  # noqa: B017
        _run(HERE / f"{name}-exercise.ipynb")


def test_notebooks_use_library_code_and_no_network() -> None:
    lib = {"01": "eq_options", "02": "eq_var", "03": "fi_rates", "04": "eq_algo",
           "05": "qp_core.aad", "06": "qp_exec.mm", "07": "qp_core.xva",
           "08": "qp_bt.portfolio"}
    for name in NAMES:
        src = "\n".join(_code(HERE / f"{name}-solution.ipynb"))
        assert f"from {lib[name[:2]]}" in src
        for bad in ("requests", "urllib", "http://", "https://", "yfinance", "read_csv("):
            assert bad not in src, (name, bad)
