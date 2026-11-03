"""Market-making game: quote on the sum of three dice (career prep).

The hidden value V is the sum of three fair dice. Each round you quote a
bid and an ask for one unit of V. A counterparty who has seen one of the
three dice (the "informed" die) trades against you when her conditional
fair value, ``d + 7``, is strictly outside your quote: she buys at your ask
if ``d + 7 > ask`` and sells at your bid if ``d + 7 < bid``. Then all dice
are shown and the position settles at V.

What the game teaches (each fact is checked in ``career-prep/tests``):

* E[V] = 10.5 and Var[V] = 35/4: the unconditional fair value and risk.
* Quoting a zero-width market at 10.5 to this counterparty loses
  E|d - 3.5| = 3/2 per round on average. That is adverse selection: you
  only trade when she knows you are wrong.
* Widening to bid 8, ask 13 (half-width 2.5 = max |d - 3.5|) means she
  never trades: no loss, but no spread earned either. A real desk sets the
  width between these two by also trading with uninformed flow (Glosten and
  Milgrom 1985).

    python career-prep/games/market_making.py --rounds 10 --seed 1
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

Quote = tuple[float, float]
QuoteFn = Callable[[], Quote]


@dataclass(frozen=True)
class RoundResult:
    dice: tuple[int, int, int]
    bid: float
    ask: float
    side: int  # +1 we sold at the ask, -1 we bought at the bid, 0 no trade
    pnl: float


def exact_moments() -> tuple[Fraction, Fraction]:
    """E[V] and Var[V] by enumerating all 216 outcomes."""
    sums = [a + b + c for a in range(1, 7) for b in range(1, 7) for c in range(1, 7)]
    mean = Fraction(sum(sums), len(sums))
    var = Fraction(sum((s - mean) ** 2 for s in sums), len(sums))
    return mean, var


def play_round(rng: random.Random, bid: float, ask: float) -> RoundResult:
    if not bid <= ask:
        raise ValueError("bid must not exceed ask")
    dice = (rng.randint(1, 6), rng.randint(1, 6), rng.randint(1, 6))
    value = sum(dice)
    informed = dice[0] + 7.0
    if informed > ask:  # she buys from us at the ask
        return RoundResult(dice, bid, ask, +1, ask - value)
    if informed < bid:  # she sells to us at the bid
        return RoundResult(dice, bid, ask, -1, value - bid)
    return RoundResult(dice, bid, ask, 0, 0.0)


def simulate(quote: QuoteFn, rounds: int, seed: int) -> list[RoundResult]:
    rng = random.Random(seed)
    return [play_round(rng, *quote()) for _ in range(rounds)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    total = 0.0
    for i in range(1, args.rounds + 1):
        bid, ask = (float(x) for x in input(f"round {i}, bid ask: ").split())
        r = play_round(rng, bid, ask)
        total += r.pnl
        trade = {1: "she bought", -1: "she sold", 0: "no trade"}[r.side]
        print(f"  dice {r.dice} (she saw {r.dice[0]}), {trade}, P&L {r.pnl:+.1f}, "
              f"total {total:+.1f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
