"""Generate golden test vectors from an independent oracle (mpmath, 50 digits).

Writes ``tests/golden/golden_vectors.json``. The reference values do not
come from the code under test: closed forms and Greeks are evaluated in
mpmath at 50 significant digits, implied-vol cases invert an
mpmath-computed quote whose true vol is known by construction, and CRR
trees run the same recursion in mpmath arithmetic. The Python, C++ and Rust
engines all consume this file (C++ and Rust through
``tools/gen_golden_header.py`` and ``tools/gen_golden_rs.py``).

Each record is ``{"fn", "in", "type", "flag", "out": {name: [value, tol]}}``
with inputs in the fixed order given by ``schema`` and an absolute
tolerance per output computed by the rules in ``tolerance_rules``.

Run from the project root (needs mpmath)::

    python tests/golden/generate_golden.py          # rewrite the JSON
    python tests/golden/generate_golden.py --check  # exit 1 if it is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp

mp.mp.dps = 50
EPS = 2.220446049250313e-16
TARGET = Path(__file__).resolve().parent / "golden_vectors.json"

SCHEMA = {
    "bs": {"in": ["S", "K", "T", "r", "q", "sigma"],
           "out": ["price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]},
    "iv": {"in": ["price", "S", "K", "T", "r", "q"], "out": ["sigma"]},
    "crr": {"in": ["S", "K", "T", "r", "q", "sigma", "steps"], "out": ["price"],
            "flag": "1 = American exercise, 0 = European"},
    "b76": {"in": ["F", "K", "T", "r", "sigma"],
            "out": ["price", "delta", "gamma", "vega", "theta", "rho"]},
}

RULES = {
    "bs": "64 eps (1 + d1^2) (|v| + max(S, K)): a sum of terms of size max(S, K), each "
          "a few ulps off after erfc/exp, with the Gaussian density amplifying an "
          "argument error by (1 + d1^2).",
    "b76": "64 eps (1 + d1^2) (|v| + max(F, K)), as for bs.",
    "iv": "4 max(eps max(S, K) / vega, 1e-12 sigma): one rounding of the quote "
          "(and of the parity shift) mapped through vega, plus the solver's 1e-12 "
          "relative floor.",
    "crr": "4 n eps max(S, K): n steps of backward induction, each a convex "
           "combination (or a max) that adds at most an ulp of the node values.",
    "edge": "Exact limits (sigma = 0 with an overflowing forward, S = 0 American put "
            "with r < 0): 4 eps |v|.",
}


def _d1(S, K, T, b, sigma):
    v = sigma * mp.sqrt(T)
    return (mp.log(S / K) + b * T) / v + v / 2


def bs_oracle(S, K, T, r, q, sigma, call):
    S, K, T, r, q, sigma = map(mp.mpf, (S, K, T, r, q, sigma))
    v = sigma * mp.sqrt(T)
    d1 = _d1(S, K, T, r - q, sigma)
    d2 = d1 - v
    dq, dr = mp.exp(-q * T), mp.exp(-r * T)
    n1 = mp.npdf(d1)
    if call:
        price = S * dq * mp.ncdf(d1) - K * dr * mp.ncdf(d2)
        delta = dq * mp.ncdf(d1)
        theta = -S * dq * n1 * sigma / (2 * mp.sqrt(T)) + q * S * dq * mp.ncdf(d1) - r * K * dr * mp.ncdf(d2)
        rho = K * T * dr * mp.ncdf(d2)
    else:
        price = K * dr * mp.ncdf(-d2) - S * dq * mp.ncdf(-d1)
        delta = -dq * mp.ncdf(-d1)
        theta = -S * dq * n1 * sigma / (2 * mp.sqrt(T)) - q * S * dq * mp.ncdf(-d1) + r * K * dr * mp.ncdf(-d2)
        rho = -K * T * dr * mp.ncdf(-d2)
    gamma = dq * n1 / (S * v)
    vega = S * dq * n1 * mp.sqrt(T)
    vanna = -dq * n1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta,
            "rho": rho, "vanna": vanna, "volga": volga}, d1


def b76_oracle(F, K, T, r, sigma, call):
    F, K, T, r, sigma = map(mp.mpf, (F, K, T, r, sigma))
    v = sigma * mp.sqrt(T)
    d1 = _d1(F, K, T, 0, sigma)
    d2 = d1 - v
    df = mp.exp(-r * T)
    sgn = 1 if call else -1
    price = df * sgn * (F * mp.ncdf(sgn * d1) - K * mp.ncdf(sgn * d2))
    n1 = mp.npdf(d1)
    return {"price": price, "delta": df * sgn * mp.ncdf(sgn * d1), "gamma": df * n1 / (F * v),
            "vega": df * F * n1 * mp.sqrt(T), "theta": r * price - df * F * n1 * sigma / (2 * mp.sqrt(T)),
            "rho": -T * price}, d1


def crr_oracle(S, K, T, r, q, sigma, steps, american, call):
    S, K, T, r, q, sigma = map(mp.mpf, (S, K, T, r, q, sigma))
    dt = T / steps
    log_u = sigma * mp.sqrt(dt)
    u = mp.exp(log_u)
    d = 1 / u
    p = (mp.exp((r - q) * dt) - d) / (u - d)
    disc = mp.exp(-r * dt)
    sgn = 1 if call else -1
    spot = lambda m: S * mp.exp(m * log_u)  # noqa: E731
    vals = [max(sgn * (spot(2 * j - steps) - K), 0) for j in range(steps + 1)]
    for i in range(steps - 1, -1, -1):
        vals = [disc * (p * vals[j + 1] + (1 - p) * vals[j]) for j in range(i + 1)]
        if american:
            vals = [max(vals[j], sgn * (spot(2 * j - i) - K)) for j in range(i + 1)]
    return vals[0]


def rec(fn, inputs, otype, out, flag=0):
    names = SCHEMA[fn]["out"]
    return {"fn": fn, "in": [float(x) for x in inputs], "type": otype, "flag": flag,
            "out": {k: [float(out[k][0]), float(out[k][1])] for k in names}}


def bs_records():
    cases = [
        (100, 100, 1.0, 0.05, 0.0, 0.2), (42, 40, 0.5, 0.10, 0.0, 0.2), (100, 110, 0.25, 0.03, 0.01, 0.3),
        (100, 90, 2.0, 0.02, 0.03, 0.15), (50, 100, 1.0, 0.05, 0.0, 0.4), (200, 100, 0.5, 0.04, 0.02, 0.25),
        (100, 100, 0.02, 0.02, 0.0, 0.2), (100, 100, 5.0, 0.03, 0.02, 0.22), (100, 105, 1.0, -0.01, 0.0, 0.2),
        (100, 95, 0.75, 0.02, 0.05, 0.18), (100, 100, 1.0, 0.05, 0.0, 0.01), (100, 100, 1.0, 0.05, 0.0, 1.5),
        (5, 7.5, 0.3, 0.06, 0.0, 0.45), (1000, 1200, 1.5, 0.045, 0.015, 0.28), (75, 70, 0.5, 0.1, 0.05, 0.35),
        (100, 60, 0.1, 0.0, 0.0, 0.5), (100, 160, 0.1, 0.0, 0.0, 0.5),
        # edge regions: very short, very long, tiny and huge vol, |d1| > 8
        (100, 100, 1e-6, 0.03, 0.0, 0.2), (100, 101, 1e-4, 0.03, 0.0, 0.3), (100, 100, 30.0, 0.03, 0.01, 0.25),
        (100, 100, 1.0, 0.03, 0.0, 0.001), (100, 100, 1.0, 0.03, 0.0, 3.0), (100, 300, 0.25, 0.03, 0.0, 0.2),
        (100, 30, 0.25, 0.03, 0.0, 0.2), (100, 100, 2.0, -0.02, -0.01, 0.3),
    ]
    out = []
    for S, K, T, r, q, sig in cases:
        for call in (True, False):
            vals, d1 = bs_oracle(S, K, T, r, q, sig, call)
            M = max(S, K)
            amp = 1 + d1 * d1
            o = {k: (v, 64 * EPS * amp * (abs(v) + M)) for k, v in vals.items()}
            out.append(rec("bs", (S, K, T, r, q, sig), "call" if call else "put", o))
    return out


def iv_records():
    out = []
    S, r, q = 100.0, 0.03, 0.01
    for lk in (-3.0, -1.0, -0.3, -0.05, -1e-6, 0.0, 1e-6, 0.05, 0.3, 1.0, 3.0):
        for T in (1e-5, 0.02, 0.25, 1.0, 10.0, 50.0):
            for sig in (1e-3, 0.01, 0.1, 0.5, 2.0):
                K = float(mp.mpf(S) * mp.exp(lk))
                for call in (True, False):
                    vals, _d1 = bs_oracle(S, K, T, r, q, sig, call)
                    price = float(vals["price"])
                    # Keep quotes whose time value and distance to the upper
                    # bound both exceed 64 ulps of the quote: inside that the
                    # vol is not identifiable from a double.
                    Sm, Km, Tm = mp.mpf(S), mp.mpf(K), mp.mpf(T)
                    F = Sm * mp.exp((r - q) * Tm)
                    lower = mp.exp(-r * Tm) * max((F - Km) if call else (Km - F), 0)
                    upper = Sm * mp.exp(-q * Tm) if call else Km * mp.exp(-r * Tm)
                    gap = 64 * EPS * max(price, float(upper))
                    if not (price > 1e-300 and float(vals["price"] - lower) > gap and float(upper - vals["price"]) > gap):
                        continue
                    tol = 4 * max(float(EPS * max(S, K) / vals["vega"]), 1e-12 * sig)
                    if tol > 1e-3 * sig:
                        continue  # information-limited quote; covered by backward-stability tests
                    out.append(rec("iv", (price, S, K, T, r, q), "call" if call else "put",
                                   {"sigma": (sig, tol)}))
    # The hard regime from the review: K=90, T=0.02, sigma=10%, OTM put.
    vals, _ = bs_oracle(100, 90, 0.02, 0.03, 0.0, 0.1, False)
    out.append(rec("iv", (float(vals["price"]), 100, 90, 0.02, 0.03, 0.0), "put", {"sigma": (0.1, 1e-12)}))
    return out


def crr_records():
    out = []
    cases = [(100, 100, 1.0, 0.05, 0.0, 0.2), (100, 110, 0.5, 0.03, 0.02, 0.3), (90, 100, 1.0, 0.05, 0.0, 0.2),
             (100, 100, 2.0, -0.02, 0.0, 0.25), (100, 95, 0.75, 0.02, 0.06, 0.18)]
    for (S, K, T, r, q, sig) in cases:
        for steps in (50, 200):
            for call in (True, False):
                for american in (False, True):
                    v = crr_oracle(S, K, T, r, q, sig, steps, american, call)
                    out.append(rec("crr", (S, K, T, r, q, sig, steps), "call" if call else "put",
                                   {"price": (v, 4 * steps * EPS * max(S, K))}, int(american)))
    return out


def b76_records():
    out = []
    for F, K, T, r, sig in [(100, 100, 1.0, 0.05, 0.2), (100, 120, 0.5, 0.02, 0.3), (2500, 2400, 0.25, 0.04, 0.18),
                            (50, 30, 2.0, -0.01, 0.4), (80, 80, 0.01, 0.03, 0.6)]:
        for call in (True, False):
            vals, d1 = b76_oracle(F, K, T, r, sig, call)
            amp = 1 + d1 * d1
            o = {k: (v, 64 * EPS * amp * (abs(v) + max(F, K))) for k, v in vals.items()}
            out.append(rec("b76", (F, K, T, r, sig), "call" if call else "put", o))
    return out


def edge_records():
    out = []
    # sigma = 0 with (r - q) T = 800: the forward overflows; the value is
    # S e^{-qT} - K e^{-rT} = 100 - 100 e^{-800} = 100 to double precision.
    nan = float("nan")
    out.append({"fn": "bs", "in": [100.0, 100.0, 1000.0, 0.8, 0.0, 0.0], "type": "call", "flag": 0,
                "out": {"price": [100.0, 4 * EPS * 100]},
                "note": "sigma = 0, overflowing forward; only price is checked"})
    for r in (-0.02, -0.005):
        v = mp.mpf(100) * mp.exp(-mp.mpf(r))
        out.append(rec("crr", (0.0, 100.0, 1.0, r, 0.0, 0.2, 100), "put",
                       {"price": (v, 4 * EPS * float(v))}, 1))
    _ = nan
    return out


def build() -> dict:
    records = bs_records() + iv_records() + crr_records() + b76_records() + edge_records()
    return {
        "description": ("Golden vectors for the equity options engines (Python, C++, Rust). "
                        "Reference values from mpmath at 50 digits, not from the code under "
                        "test. Units: r/q continuously compounded, T in years, theta per year, "
                        "vega per unit vol, rho per unit rate. Each output carries its own "
                        "absolute tolerance; see tolerance_rules."),
        "generator": "tests/golden/generate_golden.py",
        "oracle": f"mpmath {mp.__version__}, mp.dps = 50",
        "schema": SCHEMA,
        "tolerance_rules": RULES,
        "n_records": len(records),
        "records": records,
    }


def main() -> int:
    text = json.dumps(build(), indent=1) + "\n"
    if "--check" in sys.argv[1:]:
        if not TARGET.exists() or TARGET.read_text() != text:
            print(f"STALE: {TARGET}")
            return 1
        print(f"OK: {TARGET}")
        return 0
    TARGET.write_text(text)
    print(f"wrote {json.loads(text)['n_records']} records to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
