"""Risk, portfolio and alpha keys must agree with independent computation.

Each pack has ``questions.md`` (no answers), ``keys/answers.md`` (graded
lines carry the answer in backticks; discussion lines do not) and
``verify.py`` (``EXACT`` and ``MONTE_CARLO`` dicts, optional ``TOLERANCE``).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKS = {"risk": ("K", 20), "portfolio": ("O", 15), "alpha": ("A", 14)}
REL = 1e-5  # keys carry at least six significant figures


def _load(pack: str) -> ModuleType:
    name = f"cp_{pack}_verify"
    spec = importlib.util.spec_from_file_location(name, ROOT / pack / "verify.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MODS = {p: _load(p) for p in PACKS}


def _graded_key(pack: str) -> dict[str, str]:
    prefix = PACKS[pack][0]
    text = (ROOT / pack / "keys" / "answers.md").read_text("utf-8")
    return dict(re.findall(rf"^- \*\*({prefix}\d\d)\*\* `([^`]+)`", text, re.M))


def _ids(pack: str, path: Path, discussion: bool = False) -> list[str]:
    prefix = PACKS[pack][0] + ("D" if discussion else "")
    pat = rf"^- \*\*({prefix}\d)\*\*" if discussion else rf"^- \*\*({prefix}\d\d)\*\*"
    return re.findall(pat, path.read_text("utf-8"), re.M)


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_every_question_has_one_key_and_one_check(pack: str) -> None:
    q = _ids(pack, ROOT / pack / "questions.md")
    assert len(q) == PACKS[pack][1] == len(set(q))
    assert set(q) == set(_graded_key(pack)) == set(MODS[pack].EXACT)
    dq = _ids(pack, ROOT / pack / "questions.md", discussion=True)
    dk = _ids(pack, ROOT / pack / "keys" / "answers.md", discussion=True)
    assert dq and dq == dk, "every discussion prompt has a note"


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_questions_file_holds_no_answers(pack: str) -> None:
    text = (ROOT / pack / "questions.md").read_text("utf-8")
    for pid, ans in _graded_key(pack).items():
        line = next(ln for ln in text.splitlines() if f"**{pid}**" in ln)
        assert f"`{ans}`" not in line, pid


CASES = [(p, pid) for p in sorted(PACKS) for pid in sorted(MODS[p].EXACT)]


@pytest.mark.parametrize(("pack", "pid"), CASES)
def test_key_matches_independent_computation(pack: str, pid: str) -> None:
    written = Fraction(_graded_key(pack)[pid])
    computed = MODS[pack].EXACT[pid]()
    if isinstance(computed, Fraction):
        assert written == computed, (pid, written, computed)
    else:
        rel = getattr(MODS[pack], "TOLERANCE", {}).get(pid, REL)
        assert float(written) == pytest.approx(computed, rel=rel), (pid, float(written), computed)


MC_CASES = [(p, pid) for p in sorted(PACKS) for pid in sorted(MODS[p].MONTE_CARLO)]


@pytest.mark.parametrize(("pack", "pid"), MC_CASES)
def test_key_matches_seeded_simulation(pack: str, pid: str) -> None:
    estimator, tol = MODS[pack].MONTE_CARLO[pid]
    assert abs(estimator() - float(Fraction(_graded_key(pack)[pid]))) < tol, pid
