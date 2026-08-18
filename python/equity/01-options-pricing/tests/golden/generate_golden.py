"""Generate golden test vectors for cross-validating the C++/Rust engines.

Writes ``tests/golden/golden_vectors.json``: ~30 cases mapping
``{S, K, T, r, q, sigma, type}`` to analytic Black-Scholes-Merton
``{price, delta, gamma, vega, theta, rho}`` at full double precision
(JSON round-trips IEEE-754 doubles exactly via repr).

Units/conventions match the library: continuously compounded annualised
``r``/``q`` (ACT/365F), ``T`` in years, ``sigma`` annualised; theta per
year; vega/rho per unit of vol/rate.

Run from the project root::

    PYTHONPATH=src python tests/golden/generate_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eq_options import bs_greeks  # noqa: E402

# Deliberately diverse: ATM/ITM/OTM, short/long dated, low/high vol,
# negative rates, dividend yields, deep wings.
CASES: list[tuple[float, float, float, float, float, float, str]] = [
    # (S, K, T, r, q, sigma, type)
    (100.0, 100.0, 1.00, 0.05, 0.00, 0.20, "call"),
    (100.0, 100.0, 1.00, 0.05, 0.00, 0.20, "put"),
    (42.0, 40.0, 0.50, 0.10, 0.00, 0.20, "call"),      # Hull
    (42.0, 40.0, 0.50, 0.10, 0.00, 0.20, "put"),
    (100.0, 110.0, 0.25, 0.03, 0.01, 0.30, "call"),
    (100.0, 110.0, 0.25, 0.03, 0.01, 0.30, "put"),
    (100.0, 90.0, 2.00, 0.02, 0.03, 0.15, "call"),
    (100.0, 90.0, 2.00, 0.02, 0.03, 0.15, "put"),
    (50.0, 100.0, 1.00, 0.05, 0.00, 0.40, "call"),     # deep OTM call
    (50.0, 100.0, 1.00, 0.05, 0.00, 0.40, "put"),      # deep ITM put
    (200.0, 100.0, 0.50, 0.04, 0.02, 0.25, "call"),    # deep ITM call
    (200.0, 100.0, 0.50, 0.04, 0.02, 0.25, "put"),
    (100.0, 100.0, 0.02, 0.02, 0.00, 0.20, "call"),    # ~1 week
    (100.0, 100.0, 0.02, 0.02, 0.00, 0.20, "put"),
    (100.0, 100.0, 5.00, 0.03, 0.02, 0.22, "call"),    # long dated
    (100.0, 100.0, 5.00, 0.03, 0.02, 0.22, "put"),
    (100.0, 105.0, 1.00, -0.01, 0.00, 0.20, "call"),   # negative rate
    (100.0, 105.0, 1.00, -0.01, 0.00, 0.20, "put"),
    (100.0, 95.0, 0.75, 0.02, 0.05, 0.18, "call"),     # heavy dividend
    (100.0, 95.0, 0.75, 0.02, 0.05, 0.18, "put"),
    (100.0, 100.0, 1.00, 0.05, 0.00, 0.01, "call"),    # near-zero vol
    (100.0, 100.0, 1.00, 0.05, 0.00, 0.01, "put"),
    (100.0, 100.0, 1.00, 0.05, 0.00, 1.50, "call"),    # very high vol
    (100.0, 100.0, 1.00, 0.05, 0.00, 1.50, "put"),
    (5.0, 5.0, 0.30, 0.06, 0.00, 0.45, "call"),        # small notional
    (5.0, 7.5, 0.30, 0.06, 0.00, 0.45, "put"),
    (1000.0, 1200.0, 1.50, 0.045, 0.015, 0.28, "call"),  # index-like
    (1000.0, 800.0, 1.50, 0.045, 0.015, 0.28, "put"),
    (75.0, 70.0, 0.50, 0.10, 0.05, 0.35, "call"),      # Haug carry example
    (75.0, 70.0, 0.50, 0.10, 0.05, 0.35, "put"),
    (100.0, 60.0, 0.10, 0.00, 0.00, 0.50, "call"),     # short-dated deep ITM
    (100.0, 160.0, 0.10, 0.00, 0.00, 0.50, "put"),
]


def build_vectors() -> list[dict[str, object]]:
    """Compute the golden vectors (full double precision)."""
    out: list[dict[str, object]] = []
    for S, K, T, r, q, sigma, otype in CASES:
        g = bs_greeks(S, K, T, r, sigma, q, otype)  # type: ignore[arg-type]
        out.append({
            "inputs": {"S": S, "K": K, "T": T, "r": r, "q": q,
                       "sigma": sigma, "type": otype},
            "outputs": {
                "price": g.price,
                "delta": g.delta,
                "gamma": g.gamma,
                "vega": g.vega,
                "theta": g.theta,
                "rho": g.rho,
            },
        })
    return out


def main() -> None:
    target = Path(__file__).resolve().parent / "golden_vectors.json"
    payload = {
        "description": (
            "Golden Black-Scholes-Merton vectors for cross-validating the "
            "C++/Rust engines against the eq_options Python reference. "
            "Conventions: continuously compounded annualised r/q (ACT/365F), "
            "T in years, sigma annualised; theta per year; vega per unit vol; "
            "rho per unit rate. Tolerance for consumers: 1e-10 absolute."
        ),
        "generator": "tests/golden/generate_golden.py",
        "n_cases": len(CASES),
        "cases": build_vectors(),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(CASES)} golden vectors to {target}")


if __name__ == "__main__":
    main()
