"""Checks for the mental-maths generator and the market-making game."""

from __future__ import annotations

import importlib.util
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

GAMES = Path(__file__).resolve().parents[1] / "games"


def _load(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(f"cp_{name}", GAMES / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"cp_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


mm = _load("mental_maths")
mk = _load("market_making")


def _recompute(q: str) -> Fraction:
    """Independent parser: derive the answer from the question text alone."""
    if m := re.fullmatch(r"(\d+) \* (\d+)", q):
        return Fraction(int(m[1])) * int(m[2])
    if m := re.fullmatch(r"([\d.]+)% of (\d+)", q):
        return Fraction(m[1]) / 100 * int(m[2])
    if m := re.fullmatch(r"(\d+)/(\d+) as a decimal", q):
        return Fraction(int(m[1]), int(m[2]))
    if m := re.fullmatch(r"(\d+)\^2", q):
        return Fraction(int(m[1]) ** 2)
    if m := re.fullmatch(r"expected sum of (\d+) fair dice", q):
        k = int(m[1])
        return Fraction(sum(range(1, 7)), 6) * k
    if m := re.fullmatch(r"1 \+ 2 \+ \.\.\. \+ (\d+)", q):
        return Fraction(sum(range(1, int(m[1]) + 1)))
    raise AssertionError(f"unparsed question {q!r}")


@pytest.mark.parametrize("seed", range(25))
def test_every_drill_answer_is_right(seed: int) -> None:
    for d in mm.generate(seed, 40):
        assert _recompute(d.question) == d.answer, d


def test_drill_is_reproducible_and_covers_all_kinds() -> None:
    assert mm.generate(7, 30) == mm.generate(7, 30)
    assert mm.generate(7, 30) != mm.generate(8, 30)
    assert {d.kind for d in mm.generate(1, 300)} == set(mm.KINDS)
    with pytest.raises(ValueError):
        mm.generate(1, 0)
    with pytest.raises(ValueError):
        mm.generate(1, 3, ("nope",))


def test_parse_answer_accepts_decimal_and_fraction() -> None:
    assert mm.parse_answer("0.375") == mm.parse_answer("3/8") == Fraction(3, 8)


def test_cli_prints_questions_then_answers(capsys: pytest.CaptureFixture[str]) -> None:
    assert mm.main(["--seed", "3", "--n", "5"]) == 0
    out = capsys.readouterr().out
    assert out.index("Answers") > out.index(" 5. ")


def test_dice_sum_moments_by_enumeration() -> None:
    mean, var = mk.exact_moments()
    assert (mean, var) == (Fraction(21, 2), Fraction(35, 4))
    # cross-check: three independent dice, each mean 7/2 and variance 35/12
    assert var == 3 * Fraction(sum((f - Fraction(7, 2)) ** 2 for f in range(1, 7)), 6)


def test_zero_width_quote_loses_three_halves_per_round() -> None:
    res = mk.simulate(lambda: (10.5, 10.5), 200_000, seed=11)
    mean = sum(r.pnl for r in res) / len(res)
    exact = Fraction(sum(abs(Fraction(d) - Fraction(7, 2)) for d in range(1, 7)), 6)
    assert exact == Fraction(3, 2)
    assert mean == pytest.approx(-1.5, abs=0.03)


def test_quote_as_wide_as_her_information_is_never_hit() -> None:
    res = mk.simulate(lambda: (8.0, 13.0), 5_000, seed=2)
    assert all(r.side == 0 and r.pnl == 0 for r in res)


def test_crossed_quote_rejected() -> None:
    import random

    with pytest.raises(ValueError):
        mk.play_round(random.Random(0), 11.0, 10.0)
