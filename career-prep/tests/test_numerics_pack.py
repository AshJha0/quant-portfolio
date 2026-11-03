"""The Numerics pack key must agree with independent computation and simulation."""

from __future__ import annotations

import importlib.util
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1] / "numerics"
_spec = importlib.util.spec_from_file_location("cp_verify_numerics", PACK / "verify.py")
assert _spec and _spec.loader
verify = importlib.util.module_from_spec(_spec)
sys.modules["cp_verify_numerics"] = verify
_spec.loader.exec_module(verify)

KEY_LINE = re.compile(r"^- \*\*(N\d\d)\*\* `([^`]+)`", re.M)
REL_TOL: dict[str, float] = getattr(verify, "REL_TOL", {})


def _key() -> dict[str, str]:
    return dict(KEY_LINE.findall((PACK / "keys" / "answers.md").read_text("utf-8")))


def _questions() -> list[str]:
    return re.findall(r"^- \*\*(N\d\d)\*\*", (PACK / "questions.md").read_text("utf-8"), re.M)


def test_twenty_five_problems_each_with_one_key_and_one_check() -> None:
    q = _questions()
    assert len(q) == 25 and len(set(q)) == 25
    assert set(q) == set(_key()) == set(verify.EXACT)


def test_questions_file_holds_no_answers() -> None:
    text = (PACK / "questions.md").read_text("utf-8")
    for pid, ans in _key().items():
        line = next(ln for ln in text.splitlines() if f"**{pid}**" in ln)
        assert f"`{ans}`" not in line, pid


@pytest.mark.parametrize("pid", sorted(verify.EXACT))
def test_key_matches_independent_computation(pid: str) -> None:
    written = Fraction(_key()[pid])
    computed = verify.EXACT[pid]()
    if isinstance(computed, Fraction):
        assert written == computed, (pid, written, computed)
    else:
        assert float(written) == pytest.approx(computed, rel=REL_TOL.get(pid, 1e-9), abs=1e-12), pid

