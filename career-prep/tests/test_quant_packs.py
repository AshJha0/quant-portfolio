"""The stochastic, derivatives and fixed-income keys must agree with independent checks.

Each pack's ``verify.py`` recomputes every answer without the closed form the
key uses (see each module docstring). This test parses the key, compares it
with that computation (exactly for rationals, to a relative tolerance for
floats) and with the seeded simulations, and checks that the questions file
gives nothing away.
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
PACKS = {"stochastic": "S", "derivatives": "D", "fixed-income": "F"}
SIZE = 22


def _load(pack: str) -> ModuleType:
    name = f"cp_verify_{pack.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, ROOT / pack / "verify.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


VERIFY = {p: _load(p) for p in PACKS}


def _key(pack: str) -> dict[str, str]:
    pat = re.compile(rf"^- \*\*({PACKS[pack]}\d\d)\*\* `([^`]+)`", re.M)
    return dict(pat.findall((ROOT / pack / "keys" / "answers.md").read_text("utf-8")))


def _questions(pack: str) -> list[str]:
    text = (ROOT / pack / "questions.md").read_text("utf-8")
    return re.findall(rf"^- \*\*({PACKS[pack]}\d\d)\*\*", text, re.M)


@pytest.mark.parametrize("pack", PACKS)
def test_every_problem_has_one_key_and_one_check(pack: str) -> None:
    q = _questions(pack)
    assert len(q) == SIZE and len(set(q)) == SIZE
    assert set(q) == set(_key(pack)) == set(VERIFY[pack].EXACT)
    assert set(getattr(VERIFY[pack], "MONTE_CARLO", {})) <= set(q)


@pytest.mark.parametrize("pack", PACKS)
def test_questions_file_holds_no_answers(pack: str) -> None:
    text = (ROOT / pack / "questions.md").read_text("utf-8")
    for pid, ans in _key(pack).items():
        line = next(ln for ln in text.splitlines() if f"**{pid}**" in ln)
        assert f"`{ans}`" not in line, pid
        assert ans not in line or len(ans) <= 2, pid


CASES = [(p, pid) for p in PACKS for pid in sorted(VERIFY[p].EXACT)]


@pytest.mark.parametrize(("pack", "pid"), CASES)
def test_key_matches_independent_computation(pack: str, pid: str) -> None:
    mod = VERIFY[pack]
    written = Fraction(_key(pack)[pid])
    computed = mod.EXACT[pid]()
    if isinstance(computed, Fraction):
        assert written == computed, (pid, written, computed)
    else:
        rel = getattr(mod, "TOL", {}).get(pid, 1e-9)
        assert float(written) == pytest.approx(computed, rel=rel, abs=1e-12), pid


SECOND = [(p, pid) for p in PACKS for pid in sorted(getattr(VERIFY[p], "SECOND_ROUTE", {}))]


@pytest.mark.parametrize(("pack", "pid"), SECOND)
def test_key_matches_second_route(pack: str, pid: str) -> None:
    got = VERIFY[pack].SECOND_ROUTE[pid]()
    assert float(Fraction(_key(pack)[pid])) == pytest.approx(got, rel=1e-8), pid


MC = [(p, pid) for p in PACKS for pid in sorted(getattr(VERIFY[p], "MONTE_CARLO", {}))]


@pytest.mark.parametrize(("pack", "pid"), MC)
def test_key_matches_seeded_simulation(pack: str, pid: str) -> None:
    estimator, tol = VERIFY[pack].MONTE_CARLO[pid]
    assert abs(estimator() - float(Fraction(_key(pack)[pid]))) < tol, pid


def test_a_wrong_key_is_caught() -> None:
    """The comparison is not vacuous: a key off by one part in 10^6 fails."""
    mod = VERIFY["derivatives"]
    written = float(Fraction(_key("derivatives")["D01"])) * (1 + 1e-6)
    assert written != pytest.approx(mod.EXACT["D01"](), rel=1e-9)
