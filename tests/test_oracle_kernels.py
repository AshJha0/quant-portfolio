"""Static check of ``rust/ORACLE_KERNELS.md`` (plan W20 target 4).

The cargo half (``--run``) needs a Rust toolchain and runs in the Rust CI job.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "rust" / "tools" / "check_oracle_kernels.py"
_spec = importlib.util.spec_from_file_location("check_oracle_kernels", _PATH)
assert _spec is not None and _spec.loader is not None
ck = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ck  # dataclasses look the module up by name
_spec.loader.exec_module(ck)


def _rows() -> list:
    return ck.parse(ck.DOC.read_text(encoding="utf-8"))


def test_oracle_kernel_list_is_consistent() -> None:
    rows = _rows()
    assert len(rows) >= 5
    assert ck.check_static(rows) == []


def test_plan_d1_kernels_are_listed() -> None:
    ids = {r.id for r in _rows()}
    for need in ("bs-price-greeks", "bs-implied-vol", "eq-hist-var", "eq-cholesky", "eq-mc-var"):
        assert need in ids


def test_checker_catches_a_renamed_test_and_a_dropped_call() -> None:
    row = _rows()[0]
    bad = [replace(row, test="no_such_test"), replace(row, id="x", symbols=("no_such_fn",))]
    errors = ck.check_static(bad)
    assert any("no #[test] fn no_such_test" in e for e in errors)
    assert any("never calls no_such_fn(" in e for e in errors)


def test_checker_catches_a_constants_module_from_another_source() -> None:
    row = replace(_rows()[0], oracle="python/core/pyproject.toml")
    assert any("header does not name" in e for e in ck.check_static([row]))
