"""Generate golden test vectors for cross-language validation.

Produces ``golden_vectors.json``: ~30 Garman-Kohlhagen cases spanning
moneyness, tenor, rate signs (incl. negative rates) and vol regimes.
Each case maps inputs {S, K, T, r_d, r_f, sigma, type} to outputs
{price, delta_spot, delta_fwd, gamma, vega, theta, rho_d, rho_f} at full
double precision (repr round-trip, i.e. ~1e-16 relative; consumers should
match to 1e-10 absolute).  The committed JSON is the reference for the
C++ / Rust engines (CONVENTIONS.md).

Run from the project root:  python tests/golden/generate_golden.py
"""

from __future__ import annotations

import itertools
import json
import pathlib

from fx_options import analytic_greeks

OUT = pathlib.Path(__file__).parent / "golden_vectors.json"


def build_cases() -> list[dict]:
    """Deterministic grid + hand-picked stress cases (30 total)."""
    cases: list[dict] = []
    # Grid: EURUSD-like base scenario across moneyness / tenor / type.
    for K, T, opt in itertools.product((1.00, 1.10, 1.21), (0.25, 1.0),
                                       ("call", "put")):
        cases.append(dict(S=1.10, K=K, T=T, r_d=0.045, r_f=0.030,
                          sigma=0.095, type=opt))
    # USDJPY-like level (pip 0.01), carry-heavy.
    for K, opt in itertools.product((140.0, 147.5, 155.0), ("call", "put")):
        cases.append(dict(S=147.50, K=K, T=0.5, r_d=0.005, r_f=0.052,
                          sigma=0.115, type=opt))
    # Negative rates (EURCHF era), high vol (EM), long-dated, tiny tenor.
    stress = [
        dict(S=1.08, K=1.08, T=1.0, r_d=-0.0075, r_f=-0.005, sigma=0.065),
        dict(S=1.08, K=1.00, T=1.0, r_d=-0.0075, r_f=-0.005, sigma=0.065),
        dict(S=18.50, K=20.00, T=0.25, r_d=0.1125, r_f=0.045, sigma=0.35),
        dict(S=1.25, K=1.25, T=5.0, r_d=0.04, r_f=0.02, sigma=0.10),
        dict(S=1.25, K=1.40, T=7 / 365, r_d=0.04, r_f=0.02, sigma=0.10),
        dict(S=0.65, K=0.60, T=2.0, r_d=0.035, r_f=0.041, sigma=0.13),
    ]
    for base, opt in itertools.product(stress, ("call", "put")):
        cases.append({**base, "type": opt})
    return cases


def main() -> None:
    vectors = []
    for case in build_cases():
        g = analytic_greeks(case["S"], case["K"], case["T"], case["r_d"],
                            case["r_f"], case["sigma"], case["type"])
        vectors.append({
            "inputs": case,
            "outputs": {
                "price": float(g.price),
                "delta_spot": float(g.delta_spot),
                "delta_fwd": float(g.delta_forward),
                "gamma": float(g.gamma),
                "vega": float(g.vega),
                "theta": float(g.theta),
                "rho_d": float(g.rho_domestic),
                "rho_f": float(g.rho_foreign),
            },
        })
    payload = {
        "description": "Garman-Kohlhagen golden vectors for cross-language "
                       "validation (fx_options v1). Conventions: BASE/QUOTE "
                       "quotation, r_d = quote ccy, r_f = base ccy, rates "
                       "continuously compounded ACT/365F, theta per year.",
        "tolerance": 1e-10,
        "n_cases": len(vectors),
        "cases": vectors,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(vectors)} cases to {OUT}")


if __name__ == "__main__":
    main()
