"""Generate FX golden test vectors from an independent oracle (mpmath, 50 digits).

Writes ``tests/golden/golden_vectors.json``. Reference values are computed
in mpmath, not by the code under test: Garman-Kohlhagen price and Greeks
(both rhos, vanna, volga, the four delta conventions), implied vol from an
mpmath quote with known vol, strike-from-delta in all four conventions
(premium-adjusted strikes solved by bisection in mpmath on the branch
above the fold), DNS ATM strikes, CIP forwards and CRR trees run in
mpmath arithmetic. Python, C++ and Rust all consume this file.

Each record is ``{"fn", "in", "type", "flag", "out": {name: [value, tol]}}``
with inputs in the order given by ``schema`` and an absolute tolerance per
output from ``tolerance_rules``.

Run from the project root (needs mpmath)::

    python tests/golden/generate_golden.py          # rewrite the JSON
    python tests/golden/generate_golden.py --check  # exit 1 if stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp

mp.mp.dps = 50
EPS = 2.220446049250313e-16
TARGET = Path(__file__).resolve().parent / "golden_vectors.json"
CONVENTIONS = ["spot", "forward", "spot_pa", "forward_pa"]

SCHEMA = {
    "gk": {"in": ["S", "K", "T", "rd", "rf", "sigma"],
           "out": ["price", "delta_spot", "delta_fwd", "delta_spot_pa", "delta_fwd_pa", "gamma",
                   "vega", "theta", "rho_d", "rho_f", "vanna", "volga"]},
    "iv": {"in": ["price", "S", "K", "T", "rd", "rf"], "out": ["sigma"]},
    "strike": {"in": ["delta", "S", "T", "rd", "rf", "sigma"], "out": ["K"],
               "flag": "convention index: 0 spot, 1 forward, 2 spot_pa, 3 forward_pa"},
    "dns_atm": {"in": ["S", "T", "rd", "rf", "sigma"], "out": ["K"],
                "flag": "1 = premium-adjusted, 0 = unadjusted"},
    "crr": {"in": ["S", "K", "T", "rd", "rf", "sigma", "steps"], "out": ["price"],
            "flag": "1 = American exercise, 0 = European"},
    "fwd": {"in": ["S", "T", "rd", "rf"], "out": ["forward"]},
}

RULES = {
    "gk": "64 eps (1 + d1^2) (|v| + max(S, K)): terms of size max(S, K), a few ulps "
          "each after erfc/exp, with the Gaussian density amplifying an argument "
          "error by (1 + d1^2).",
    "iv": "4 max(eps max(S, K) / vega, 1e-12 sigma): one rounding of the quote mapped "
          "through vega, plus the solver's 1e-12 relative floor.",
    "strike": "2e-14 + 16 eps (K + |delta / (d delta / dK)|) (1 + v^2): Brent's 1e-14 "
              "absolute stopping width, plus a few ulps of the target delta mapped "
              "through the slope of delta in K.",
    "dns_atm": "8 eps K (1 + sigma^2 T + |(rd - rf) T|): two exps of small arguments.",
    "crr": "4 n eps max(S, K): n backward-induction steps of convex combinations.",
    "fwd": "8 eps F (1 + |(rd - rf) T|): one exp and one product.",
    "edge": "Exact limit (sigma = 0 with an overflowing forward): 4 eps |v|.",
}


def gk_oracle(S, K, T, rd, rf, sigma, call):
    S, K, T, rd, rf, sigma = map(mp.mpf, (S, K, T, rd, rf, sigma))
    v = sigma * mp.sqrt(T)
    d1 = (mp.log(S / K) + (rd - rf) * T) / v + v / 2
    d2 = d1 - v
    df, ff = mp.exp(-rd * T), mp.exp(-rf * T)
    F = S * ff / df
    p = 1 if call else -1
    n1 = mp.npdf(d1)
    price = p * (S * ff * mp.ncdf(p * d1) - K * df * mp.ncdf(p * d2))
    out = {
        "price": price,
        "delta_spot": p * ff * mp.ncdf(p * d1),
        "delta_fwd": p * mp.ncdf(p * d1),
        "delta_spot_pa": p * ff * (K / F) * mp.ncdf(p * d2),
        "delta_fwd_pa": p * (K / F) * mp.ncdf(p * d2),
        "gamma": ff * n1 / (S * v),
        "vega": S * ff * n1 * mp.sqrt(T),
        "theta": -S * ff * n1 * sigma / (2 * mp.sqrt(T)) + p * (rf * S * ff * mp.ncdf(p * d1)
                                                                 - rd * K * df * mp.ncdf(p * d2)),
        "rho_d": p * K * T * df * mp.ncdf(p * d2),
        "rho_f": -p * S * T * ff * mp.ncdf(p * d1),
        "vanna": -ff * n1 * d2 / sigma,
        "volga": S * ff * n1 * mp.sqrt(T) * d1 * d2 / sigma,
    }
    return out, d1


def _bisect(f, lo, hi, n=400):
    flo = f(lo)
    for _ in range(n):
        mid = (lo + hi) / 2
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


def strike_oracle(delta, S, T, rd, rf, sigma, call, conv):
    S, T, rd, rf, sigma, delta = map(mp.mpf, (S, T, rd, rf, sigma, delta))
    F = S * mp.exp((rd - rf) * T)
    v = sigma * mp.sqrt(T)
    p = 1 if call else -1
    fwd_delta = delta * mp.exp(rf * T) if conv in ("spot", "spot_pa") else delta
    if conv in ("spot", "forward"):
        z = mp.sqrt(2) * mp.erfinv(2 * (p * fwd_delta) - 1)
        K = F * mp.exp(-p * z * v + v * v / 2)
        d1 = (mp.log(F / K)) / v + v / 2
        slope = -mp.npdf(d1) / (K * v) * (mp.exp(-rf * T) if conv == "spot" else 1)
        return K, slope

    def pa(K):
        d2 = (mp.log(F / K) - v * v / 2) / v
        return p * (K / F) * mp.ncdf(p * d2)

    if call:
        x = _bisect(lambda x: mp.log(mp.ncdf(x)) + mp.log(v) - mp.log(mp.npdf(x)), -(v + 1), v + 10)
        k_lo = F * mp.exp(-x * v - v * v / 2)
        K = _bisect(lambda K: pa(K) - fwd_delta, k_lo, F * mp.exp(20 * v))
    else:
        K = _bisect(lambda K: pa(K) - fwd_delta, F * mp.exp(-20 * v), F * mp.exp(20 * v))
    h = K * mp.mpf("1e-20")
    slope = (pa(K + h) - pa(K - h)) / (2 * h) * (mp.exp(-rf * T) if conv == "spot_pa" else 1)
    return K, slope


def crr_oracle(S, K, T, rd, rf, sigma, steps, american, call):
    S, K, T, rd, rf, sigma = map(mp.mpf, (S, K, T, rd, rf, sigma))
    dt = T / steps
    u = mp.exp(sigma * mp.sqrt(dt))
    d = 1 / u
    p = (mp.exp((rd - rf) * dt) - d) / (u - d)
    disc = mp.exp(-rd * dt)
    sgn = 1 if call else -1
    vals = [max(sgn * (S * u ** (2 * j - steps) - K), 0) for j in range(steps + 1)]
    for i in range(steps - 1, -1, -1):
        vals = [disc * (p * vals[j + 1] + (1 - p) * vals[j]) for j in range(i + 1)]
        if american:
            vals = [max(vals[j], sgn * (S * u ** (2 * j - i) - K)) for j in range(i + 1)]
    return vals[0]


def rec(fn, inputs, otype, out, flag=0):
    return {"fn": fn, "in": [float(x) for x in inputs], "type": otype, "flag": flag,
            "out": {k: [float(v[0]), float(v[1])] for k, v in out.items()}}


MARKETS = [  # (S, K, T, rd, rf, sigma)
    (1.10, 1.10, 0.5, 0.0425, 0.0290, 0.0825), (1.10, 1.25, 1.0, 0.0425, 0.0290, 0.0825),
    (147.5, 150.0, 0.5, 0.0050, 0.0525, 0.1075), (1.08, 1.05, 1.0, -0.0075, -0.0050, 0.065),
    (0.65, 0.60, 2.0, 0.045, 0.040, 0.12), (1.27, 1.40, 0.25, 0.05, 0.04, 0.09),
    (7.20, 7.0, 3.0, 0.02, 0.05, 0.05), (1.10, 0.80, 0.02, 0.04, 0.03, 0.10),
    (1.10, 1.10, 10.0, 0.03, 0.01, 0.30), (150.0, 100.0, 0.5, 0.0, 0.05, 0.6),
    (1.10, 1.10, 1e-5, 0.04, 0.02, 0.10), (1.10, 1.11, 1.0, 0.04, 0.02, 0.002),
]


def gk_records():
    out = []
    for S, K, T, rd, rf, sig in MARKETS:
        for call in (True, False):
            vals, d1 = gk_oracle(S, K, T, rd, rf, sig, call)
            amp = 1 + d1 * d1
            o = {k: (v, 64 * EPS * amp * (abs(v) + max(S, K))) for k, v in vals.items()}
            out.append(rec("gk", (S, K, T, rd, rf, sig), "call" if call else "put", o))
    return out


def iv_records():
    out = []
    S, rd, rf = 1.10, 0.04, 0.02
    for lk in (-1.0, -0.3, -0.05, 0.0, 0.05, 0.3, 1.0):
        for T in (1e-4, 0.02, 0.25, 1.0, 10.0):
            for sig in (0.005, 0.05, 0.1, 0.5, 2.0):
                K = float(mp.mpf(S) * mp.exp(lk))
                for call in (True, False):
                    vals, _ = gk_oracle(S, K, T, rd, rf, sig, call)
                    price = float(vals["price"])
                    Sm, Km, Tm = mp.mpf(S), mp.mpf(K), mp.mpf(T)
                    F = Sm * mp.exp((rd - rf) * Tm)
                    lower = mp.exp(-rd * Tm) * max((F - Km) if call else (Km - F), 0)
                    upper = Sm * mp.exp(-rf * Tm) if call else Km * mp.exp(-rd * Tm)
                    gap = 64 * EPS * max(price, float(upper))
                    if not (price > 1e-300 and float(vals["price"] - lower) > gap
                            and float(upper - vals["price"]) > gap):
                        continue
                    tol = 4 * max(float(EPS * max(S, K) / vals["vega"]), 1e-12 * sig)
                    if tol > 1e-3 * sig:
                        continue
                    out.append(rec("iv", (price, S, K, T, rd, rf), "call" if call else "put",
                                   {"sigma": (sig, tol)}))
    vals, _ = gk_oracle(100, 90, 0.02, 0.03, 0.0, 0.1, False)
    out.append(rec("iv", (float(vals["price"]), 100, 90, 0.02, 0.03, 0.0), "put", {"sigma": (0.1, 1e-12)}))
    return out


def strike_records():
    out = []
    for S, _, T, rd, rf, sig in MARKETS[:7]:
        for ci, conv in enumerate(CONVENTIONS):
            for call, d in ((True, 0.10), (True, 0.25), (False, -0.10), (False, -0.25), (False, -0.40)):
                K, slope = strike_oracle(d, S, T, rd, rf, sig, call, conv)
                v = sig * T ** 0.5
                tol = 2e-14 + 16 * EPS * float(K + abs(mp.mpf(d) / slope)) * (1 + v * v)
                out.append(rec("strike", (d, S, T, rd, rf, sig), "call" if call else "put",
                               {"K": (K, tol)}, ci))
    return out


def dns_records():
    out = []
    for S, _, T, rd, rf, sig in MARKETS[:7]:
        for pa in (0, 1):
            Sm, Tm, sm = mp.mpf(S), mp.mpf(T), mp.mpf(sig)
            F = Sm * mp.exp((mp.mpf(rd) - rf) * Tm)
            K = F * mp.exp((-1 if pa else 1) * sm * sm * Tm / 2)
            tol = 8 * EPS * float(K) * (1 + sig * sig * T + abs((rd - rf) * T))
            out.append(rec("dns_atm", (S, T, rd, rf, sig), "call", {"K": (K, tol)}, pa))
    return out


def crr_records():
    out = []
    for S, K, T, rd, rf, sig in MARKETS[:5]:
        for steps in (50, 200):
            for call in (True, False):
                for american in (False, True):
                    v = crr_oracle(S, K, T, rd, rf, sig, steps, american, call)
                    out.append(rec("crr", (S, K, T, rd, rf, sig, steps), "call" if call else "put",
                                   {"price": (v, 4 * steps * EPS * max(S, K))}, int(american)))
    return out


def fwd_records():
    out = []
    for S, _, T, rd, rf, _ in MARKETS:
        F = mp.mpf(S) * mp.exp((mp.mpf(rd) - rf) * mp.mpf(T))
        out.append(rec("fwd", (S, T, rd, rf), "call",
                       {"forward": (F, 8 * EPS * float(F) * (1 + abs((rd - rf) * T)))}))
    return out


def edge_records():
    return [{"fn": "gk", "in": [1.0, 1.0, 1000.0, 0.8, 0.0, 0.0], "type": "call", "flag": 0,
             "out": {"price": [1.0, 4 * EPS]},
             "note": "sigma = 0, overflowing forward; only price is checked"}]


def build() -> dict:
    records = (gk_records() + iv_records() + strike_records() + dns_records() + crr_records()
               + fwd_records() + edge_records())
    return {
        "description": ("Golden vectors for the FX options engines (Python, C++, Rust). "
                        "Reference values from mpmath at 50 digits, not from the code under "
                        "test. Prices in domestic currency per unit foreign notional; rates "
                        "continuously compounded; T in years; theta per year. Each output "
                        "carries its own absolute tolerance; see tolerance_rules."),
        "generator": "tests/golden/generate_golden.py",
        "oracle": f"mpmath {mp.__version__}, mp.dps = 50",
        "schema": SCHEMA,
        "conventions": CONVENTIONS,
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
