"""Compare measured line coverage with .github/coverage-floors.json (W12).

Usage: python coverage_floor.py FLOORS_JSON KEY PERCENT

Exit 1 when PERCENT is below the floor for KEY. A floor of null records
the value and passes, so a new package can join before its floor is set.
The floor only moves up, by hand, in a reviewed pull request.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def check(floors: dict[str, object], key: str, percent: float) -> tuple[bool, str]:
    if key not in floors:
        return False, f"{key}: no entry in coverage-floors.json; add one (null to record only)"
    floor = floors[key]
    if floor is None:
        return True, f"{key}: {percent:.2f}% (no floor yet; set it to {math.floor(percent)})"
    assert isinstance(floor, int | float)
    if percent < floor:
        return False, f"{key}: {percent:.2f}% is below the floor {floor}%"
    note = f"; raise the floor to {math.floor(percent)}" if percent >= floor + 1 else ""
    return True, f"{key}: {percent:.2f}% >= floor {floor}%{note}"


def main(argv: list[str]) -> int:
    floors = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    ok, msg = check(floors, argv[2], float(argv[3]))
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
