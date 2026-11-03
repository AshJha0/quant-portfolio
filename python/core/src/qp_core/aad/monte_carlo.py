"""Monte Carlo pricer with pathwise adjoint Greeks.

Plain idea: simulate geometric Brownian motion, average the discounted
payoff, and differentiate each simulated path with respect to the model
inputs. Averaging those path derivatives gives the Greeks of the Monte Carlo
estimate itself (the pathwise method of Broadie and Glasserman 1996,
"Estimating security price derivatives using simulation", *Management
Science* 42(2)); adjoint mode gets all of them from one backward sweep per
path (Giles and Glasserman 2006, "Smoking adjoints", *Risk* 19(1)).

Formal statement: ``ln S_{i+1} = ln S_i + (r - q - sigma^2/2) dt + sigma sqrt(dt) Z_i``,
price ``e^{-rT} mean(payoff(S_1..S_n))``. The payoffs here (European and
arithmetic-average Asian calls) are Lipschitz, so the pathwise derivative is
unbiased; the kink at the strike is crossed with probability zero
(Glasserman 2004, *Monte Carlo Methods in Financial Engineering*, s. 7.2).
Each path is recorded, swept and dropped (``Tape.mark``/``rewind``), so the
tape never holds more than one path.

Worked example: with the same normals, the adjoint delta equals the
central-difference delta of the same estimator to about 1e-9 relative, and
the European figures sit within three standard errors of Black-Scholes
(``tests/test_aad_python.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from .tape import Real, Tape, Var, exp, log, sqrt

PAYOFFS = ("european", "asian")


def _path_payoff(log_s0: Real, drift: Real, vol: Real, K: Real, z: Sequence[float],
                 payoff: str) -> Real:
    if payoff == "european":
        # ln S_T depends on the inputs only through sum(z): one exp and four
        # tape nodes per path, whatever the number of steps.
        under = exp(log_s0 + drift * len(z) + vol * float(sum(z)))
    else:
        x = log_s0
        total: Real = 0.0
        for zi in z:
            x = x + drift + vol * float(zi)
            total = total + exp(x)
        under = total / len(z)
    return under - K if under > K else 0.0


def mc_price(S: Real, K: Real, T: Real, r: Real, q: Real, sigma: Real,
             normals: np.ndarray, payoff: str = "european") -> Real:
    """Generic Monte Carlo price on the given ``(paths, steps)`` normals."""
    if payoff not in PAYOFFS:
        raise ValueError(f"payoff must be one of {PAYOFFS}")
    n_paths, n_steps = int(normals.shape[0]), int(normals.shape[1])
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * sqrt(dt)
    log_s0 = log(S)
    acc: Real = 0.0
    for p in range(n_paths):
        acc = acc + _path_payoff(log_s0, drift, vol, K, normals[p], payoff)
    return exp(-r * T) * acc / n_paths


class McGreeks(NamedTuple):
    price: float
    delta: float
    vega: float
    rho: float
    rho_q: float
    dK: float
    dT: float
    std_error: float


def mc_greeks_aad(S: float, K: float, T: float, r: float, q: float, sigma: float,
                  normals: np.ndarray, payoff: str = "european") -> McGreeks:
    """Price and pathwise adjoint Greeks; one backward sweep per path."""
    if payoff not in PAYOFFS:
        raise ValueError(f"payoff must be one of {PAYOFFS}")
    n_paths, n_steps = int(normals.shape[0]), int(normals.shape[1])
    tape = Tape()
    xs = [tape.var(v) for v in (S, K, T, r, q, sigma)]
    vS, vK, vT, vr, vq, vs = xs
    dt = vT / n_steps
    drift = (vr - vq - 0.5 * vs * vs) * dt
    vol = vs * sqrt(dt)
    log_s0 = log(vS)
    disc = exp(-vr * vT)
    mark = tape.mark()
    tape.zero_adjoints()
    total = 0.0
    sq = 0.0
    disc_v = disc.value if isinstance(disc, Var) else float(disc)
    for p in range(n_paths):
        pay = _path_payoff(log_s0, drift, vol, vK, normals[p], payoff)
        pv = disc_v * (pay.value if isinstance(pay, Var) else float(pay))
        total += pv
        sq += pv * pv
        if isinstance(pay, Var):
            # d(disc * mean payoff): payoff part per path, discount part below.
            tape.backward(pay, disc_v / n_paths, stop=mark)
        tape.rewind(mark)
    mean_pay = total / disc_v / n_paths
    # Adjoint of the discount factor: d price / d disc = mean payoff.
    if isinstance(disc, Var):
        tape.backward(disc, mean_pay, stop=mark)
    tape.propagate(mark)
    g = [tape.adjoint(x) for x in xs]
    price = total / n_paths
    var = max(sq / n_paths - price * price, 0.0)
    return McGreeks(price, g[0], g[5], g[3], g[4], g[1], g[2], math.sqrt(var / n_paths))


def normals(n_paths: int, n_steps: int, seed: int = 20261103) -> np.ndarray:
    """Standard normals from NumPy's PCG64 generator with a fixed seed."""
    return np.random.default_rng(seed).standard_normal((n_paths, n_steps))
