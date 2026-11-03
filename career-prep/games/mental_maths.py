"""Mental-maths drill generator (career prep).

Trading desks test fast, exact arithmetic: two-digit products, percentages,
fractions to decimals, and simple expected values. This script prints a
reproducible drill (same seed, same questions) with the answers held back
until the end, or checks your typed answers.

    python career-prep/games/mental_maths.py --seed 7 --n 20
    python career-prep/games/mental_maths.py --seed 7 --n 20 --interactive

Every answer is an exact ``Fraction``; the CI test recomputes each one from
the question text by a separate parser, so a wrong key cannot ship.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from fractions import Fraction

KINDS = ("product", "percent", "fraction", "square", "ev_dice", "sum_series")


@dataclass(frozen=True)
class Drill:
    kind: str
    question: str
    answer: Fraction


def _product(rng: random.Random) -> Drill:
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    return Drill("product", f"{a} * {b}", Fraction(a * b))


def _percent(rng: random.Random) -> Drill:
    p = rng.choice([5, 12.5, 15, 20, 25, 35, 40, 60, 75])
    x = rng.randint(4, 96) * 10
    return Drill("percent", f"{p}% of {x}", Fraction(str(p)) * x / 100)


def _fraction(rng: random.Random) -> Drill:
    d = rng.choice([8, 16, 20, 25, 40, 32])
    n = rng.randint(1, d - 1)
    return Drill("fraction", f"{n}/{d} as a decimal", Fraction(n, d))


def _square(rng: random.Random) -> Drill:
    a = rng.randint(11, 99)
    return Drill("square", f"{a}^2", Fraction(a * a))


def _ev_dice(rng: random.Random) -> Drill:
    k = rng.randint(2, 6)
    return Drill("ev_dice", f"expected sum of {k} fair dice", Fraction(7 * k, 2))


def _sum_series(rng: random.Random) -> Drill:
    n = rng.randint(10, 60)
    return Drill("sum_series", f"1 + 2 + ... + {n}", Fraction(n * (n + 1), 2))


_MAKERS = {"product": _product, "percent": _percent, "fraction": _fraction,
           "square": _square, "ev_dice": _ev_dice, "sum_series": _sum_series}


def generate(seed: int, n: int, kinds: tuple[str, ...] = KINDS) -> list[Drill]:
    """Return ``n`` drills; identical for identical ``seed`` and ``kinds``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    unknown = set(kinds) - set(_MAKERS)
    if unknown:
        raise ValueError(f"unknown kinds: {sorted(unknown)}")
    rng = random.Random(seed)
    return [_MAKERS[rng.choice(kinds)](rng) for _ in range(n)]


def parse_answer(text: str) -> Fraction:
    """Accept '0.375', '3/8' or '1209'."""
    return Fraction(text.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args(argv)
    drills = generate(args.seed, args.n)
    if not args.interactive:
        for i, d in enumerate(drills, 1):
            print(f"{i:2d}. {d.question}")
        print("\nAnswers")
        for i, d in enumerate(drills, 1):
            print(f"{i:2d}. {float(d.answer):g}")
        return 0
    right, start = 0, time.perf_counter()
    for i, d in enumerate(drills, 1):
        reply = input(f"{i:2d}. {d.question} = ")
        try:
            ok = parse_answer(reply) == d.answer
        except (ValueError, ZeroDivisionError):
            ok = False
        right += ok
        print("  correct" if ok else f"  no: {float(d.answer):g}")
    print(f"{right}/{len(drills)} in {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
