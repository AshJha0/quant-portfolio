"""Smoke test run against each built wheel: python wheel_smoke.py <module>.

The full binding suites compare against the Python reference packages, which
are not installed next to a wheel under test. This checks the narrower claim
that the wheel imports, its compiled extension loads, and one price comes out
right: Black-Scholes call, S=K=100, T=1, r=5%, sigma=20%, no dividend, is
10.4505835721856 (the standard textbook case).
"""

from __future__ import annotations

import importlib
import sys


def main(module: str) -> int:
    qp = importlib.import_module(module)
    if not qp.available():
        print(f"{module}: compiled extension failed to import")
        return 1
    got = qp.eq_bs_price(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "call")
    want = 10.4505835721856
    if abs(got - want) > 1e-10:
        print(f"{module}: eq_bs_price = {got!r}, want {want!r}")
        return 1
    print(f"{module} {qp.__version__}: ok (eq_bs_price = {got!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
