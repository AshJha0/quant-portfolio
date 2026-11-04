"""Order-sign memory, aggregate impact and the stylised-fact checks.

Plain idea
----------
Trade signs are persistent: a buy is usually followed by more buys, for
hours. The standard explanation (Lillo, Mike and Farmer 2005) is that large
orders are split into many small ones. Price impact of aggregate signed
volume is concave: doubling the imbalance less than doubles the move.

Formal statement
----------------
* Sign autocorrelation ``C(l) = E[e_t e_{t+l}]`` decays as ``l^-gamma`` with
  ``0 < gamma < 1`` (Lillo and Farmer 2004; Bouchaud et al. 2018 ch. 10).
  If metaorder lengths have a Pareto tail ``P(L > x) ~ x^-a`` with
  ``1 < a < 2``, then ``gamma = a - 1`` (Lillo, Mike and Farmer 2005).
* Aggregate impact: over a window, ``E[R | Q] ~ sign(Q) |Q|^delta`` for the
  signed volume ``Q`` and mid return ``R`` (Patzelt and Bouchaud 2018 for the
  shape). ``delta`` is estimated by log-log OLS on binned conditional means,
  with a bootstrap CI over windows, and compared with 0.5. This is
  aggregate order-flow impact, not metaorder impact; the two are different
  objects (Bouchaud et al. 2018 ch. 12-13).

Crypto caveat: the free data here is Binance aggTrades (sign from the
``is_buyer_maker`` flag) and Coinbase. Crypto venues are fragmented, run all
day, and have different participants; exponents fitted there are not
evidence about equities and are labelled as such.

Worked example
--------------
>>> s = lmf_signs(200_000, tail=1.5, n_active=20, seed=4)
>>> g, lo, hi = acf_exponent(s, lags=(10, 1000))
>>> bool(0.3 <= g <= 0.8)
True
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from qp_core.model import model

FloatArray = NDArray[np.float64]

# Named bands of the stylised-fact checks (plan W25 acceptance).
SIGN_ACF_EXPONENT_BAND = (0.3, 0.8)  # Lillo and Farmer (2004) report 0.2-0.7 on LSE stocks


def lmf_signs(n: int, tail: float, n_active: int, seed: int) -> NDArray[np.int64]:
    """Lillo-Mike-Farmer order signs: ``n_active`` concurrent metaorders.

    Each metaorder has a random sign and a Pareto(``tail``) length; at each
    step one active metaorder (chosen uniformly) emits a child; a finished
    metaorder is replaced by a new one.
    """
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1, 1]), size=n_active)
    left = np.ceil(rng.pareto(tail, size=n_active) + 1).astype(np.int64)
    out = np.empty(n, dtype=np.int64)
    picks = rng.integers(0, n_active, size=n)
    for t in range(n):
        k = picks[t]
        out[t] = signs[k]
        left[k] -= 1
        if left[k] == 0:
            signs[k] = 1 if rng.random() < 0.5 else -1
            left[k] = int(np.ceil(rng.pareto(tail) + 1))
    return out


def sign_acf(signs: NDArray[np.int64], max_lag: int) -> FloatArray:
    """``C(l)`` for ``l = 1..max_lag`` by FFT (biased estimator, mean removed)."""
    x = signs.astype(float) - signs.mean()
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: max_lag + 1] / n
    out: FloatArray = ac[1:] / ac[0]
    return out


def acf_exponent(signs: NDArray[np.int64], lags: tuple[int, int] = (10, 1000),
                 n_boot: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """``gamma`` in ``C(l) ~ l^-gamma`` by log-log OLS on log-spaced lags.

    Returns the estimate and a 95% block-bootstrap interval (blocks of
    ``10 * lags[1]`` signs keep the memory inside a block).
    """

    def fit(s: NDArray[np.int64]) -> float:
        c = sign_acf(s, lags[1])
        ls = np.unique(np.geomspace(lags[0], lags[1], 30).astype(int))
        y = c[ls - 1]
        ok = y > 0
        if ok.sum() < 3:
            return float("nan")
        slope = np.polyfit(np.log(ls[ok]), np.log(y[ok]), 1)[0]
        return float(-slope)

    g = fit(signs)
    rng = np.random.default_rng(seed)
    blk = 10 * lags[1]
    nb = signs.size // blk
    if nb < 4 or n_boot <= 0:
        return g, float("nan"), float("nan")
    blocks = signs[: nb * blk].reshape(nb, blk)
    bs = [fit(blocks[rng.integers(0, nb, nb)].ravel()) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return g, float(lo), float(hi)


@dataclass(frozen=True)
class ImpactFit:
    delta: float
    ci: tuple[float, float]
    n_windows: int


def window_imbalance(times: FloatArray, signed_qty: FloatArray, mid_before: FloatArray,
                     window: float) -> tuple[FloatArray, FloatArray]:
    """Signed volume ``Q`` and mid change ``R`` per fixed window of trade time.

    ``mid_before[i]`` is the mid (or trade price) just before trade ``i``;
    the window's return is from its first to the first of the next window.
    """
    w = np.floor((times - times[0]) / window).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, np.diff(w) != 0])
    q = np.add.reduceat(signed_qty, starts)
    p0 = mid_before[starts]
    r = np.r_[p0[1:] - p0[:-1], np.nan]
    return q[:-1], r[:-1]


@model(id="EXEC-IMPACT-FIT")
def fit_impact_exponent(q: FloatArray, r: FloatArray, n_bins: int = 15, n_boot: int = 300,
                        seed: int = 0) -> ImpactFit:
    """``delta`` in ``E[R | Q] ~ sign(Q)|Q|^delta`` with a bootstrap 95% CI.

    Sign symmetry is used (``R * sign(Q)`` against ``|Q|``); bins are
    quantiles of ``|Q|``; bins with a non-positive mean are dropped.
    """
    q = np.asarray(q, float)
    r = np.asarray(r, float)
    ok = np.isfinite(q) & np.isfinite(r) & (q != 0)
    q, r = q[ok], r[ok]

    def fit(qq: FloatArray, rr: FloatArray) -> float:
        a = np.abs(qq)
        y = rr * np.sign(qq)
        edges = np.quantile(a, np.linspace(0, 1, n_bins + 1))
        idx = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() >= 5:
                my = y[m].mean()
                if my > 0:
                    xs.append(np.log(a[m].mean()))
                    ys.append(np.log(my))
        if len(xs) < 3:
            return float("nan")
        return float(np.polyfit(xs, ys, 1)[0])

    d = fit(q, r)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, q.size, q.size)
        bs.append(fit(q[i], r[i]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return ImpactFit(d, (float(lo), float(hi)), int(q.size))


def load_binance_aggtrades(path: Path) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Read a Binance ``aggTrades`` CSV (data.binance.vision, no account).

    Columns: agg id, price, qty, first id, last id, time (ms or us),
    is_buyer_maker, [is_best_match]. The taker sign is ``-1`` when the buyer
    was the maker. Returns ``(time_s, signed_qty, price)``.
    """
    t, sq, px = [], [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].isdigit():
                continue  # header
            p, qty, ts = float(row[1]), float(row[2]), float(row[5])
            buyer_maker = row[6].strip().lower() == "true"
            t.append(ts)
            sq.append(-qty if buyer_maker else qty)
            px.append(p)
    tt = np.asarray(t)
    scale = 1e-6 if tt.size and tt[0] > 1e14 else 1e-3
    return tt * scale, np.asarray(sq), np.asarray(px)


def signature_ratio(mid: FloatArray, short: int = 1, long: int = 50) -> float:
    """Realised variance per step at ``short`` over that at ``long`` sampling.

    Above 1 when prices mean-revert at short lags (bid-ask bounce, tick
    discreteness); Andersen et al. (2000) "signature plot".
    """
    m = np.asarray(mid, float)
    m = m[np.isfinite(m)]

    def rv(k: int) -> float:
        d = m[::k]
        return float(np.mean(np.diff(d) ** 2) / k)

    return rv(short) / rv(long)
