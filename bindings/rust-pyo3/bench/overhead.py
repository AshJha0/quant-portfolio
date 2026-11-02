"""Per-call overhead of scalar vs batch calls from Python (not a test).

Usage: python bench/overhead.py [--module qp_rust|qp_cpp] [--n 100000]

Prints the median time per option of: the Python reference engine, the
binding's scalar call in a Python loop, and one batch call over n options.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import statistics
import time

import numpy as np


def _median_ns(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="qp_rust")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--repeat", type=int, default=7)
    a = ap.parse_args()
    qp = importlib.import_module(a.module)
    import eq_options

    rng = np.random.default_rng(0)
    n = a.n
    S = rng.uniform(80, 120, n)
    K = rng.uniform(80, 120, n)
    T = rng.uniform(0.1, 2.0, n)
    r = rng.uniform(0.0, 0.05, n)
    sig = rng.uniform(0.1, 0.5, n)
    q = np.zeros(n)
    is_call = rng.random(n) < 0.5
    side = np.where(is_call, "call", "put").tolist()
    m = min(n, 20_000)  # Python loops run on a slice to keep the run short
    Sl, Kl, Tl, rl, sl = (x[:m].tolist() for x in (S, K, T, r, sig))

    def py_ref():
        f = eq_options.bs_price
        for i in range(m):
            f(Sl[i], Kl[i], Tl[i], rl[i], sl[i], 0.0, side[i])

    def scalar():
        f = qp.eq_bs_price
        for i in range(m):
            f(Sl[i], Kl[i], Tl[i], rl[i], sl[i], 0.0, side[i])

    def batch():
        qp.eq_bs_price_batch(S, K, T, r, sig, q, is_call)

    def batch_greeks():
        qp.eq_bs_greeks_batch(S, K, T, r, sig, q, is_call)

    ones = [x[:1].copy() for x in (S, K, T, r, sig, q, is_call)]

    def one():
        qp.eq_bs_price_batch(*ones)

    print(f"module={a.module} n={n} python={platform.python_version()} "
          f"machine={platform.machine()} {platform.processor() or platform.system()}")
    rows = [
        ("python reference bs_price, loop", _median_ns(py_ref, a.repeat) / m),
        ("binding eq_bs_price, loop", _median_ns(scalar, a.repeat) / m),
        ("binding eq_bs_price_batch, one call", _median_ns(batch, a.repeat) / n),
        ("binding eq_bs_greeks_batch, one call", _median_ns(batch_greeks, a.repeat) / n),
        ("binding eq_bs_price_batch, length 1", _median_ns(one, 2001)),
    ]
    print("| call | ns per option |")
    print("|---|---:|")
    for name, ns in rows:
        print(f"| {name} | {ns:,.1f} |")


if __name__ == "__main__":
    main()
