"""Can a predator tell that our parent order is working? Detector and AUC.

Plain idea
----------
A predator sees the public lit tape and the results of its own dark pings.
A regular router leaves a pattern: the same child size at the same phase of
a fixed interval, and resting dark orders that a 1-lot ping can hit. We
build the predator's detector, train it on episodes with and without our
parent, and measure how well it separates them on held-out episodes (AUC).
Randomisation should push the AUC towards 0.5 (a coin flip).

Formal statement
----------------
Episode features, from the predator's view only:

* ``ping_hits``: dark pings of ours that filled, signed by the inferred side.
* ``phase_conc``: for candidate periods ``P`` in ``PERIODS_US``, fold the
  buy-aggressor lit prints by ``ts mod P`` into 20 bins, take the largest bin
  share of volume, and keep the maximum over ``P`` (a periodogram in
  miniature).
* ``size_mode``: the largest share of buy-aggressor prints that have one
  identical size.

The detector is a logistic regression on these features (fitted by
maximum likelihood with a small ridge), trained on a random half of the
episodes. The AUC on the other half is the Mann-Whitney statistic; its 95 %
interval is a percentile bootstrap over held-out episodes (Efron and
Tibshirani 1993, ch. 13).

What randomisation cannot hide: the parent's net demand. A detector given
the signed lit volume (``imbalance``) separates parent from no-parent
episodes whatever the router does, because buying 100 lots moves the
imbalance. ``imbalance_auc`` is reported alongside so nobody mistakes
"pattern hidden" for "intent hidden".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from qp_exec.lob import Side

from .dark import Predator
from .market import Market, MarketParams
from .router import SmartRouter

PERIODS_US = (100_000, 200_000, 500_000)
DARK = ("DARK_MID", "DARK_FBA")


def _features(m: Market, pred: Predator, t0: int, t1: int) -> tuple[list[float], float]:
    buys = [(ts, q) for v in m.lit.values() for ts, s, q in v.tape if s > 0 and t0 <= ts < t1]
    sells = sum(q for v in m.lit.values() for ts, s, q in v.tape if s < 0 and t0 <= ts < t1)
    hits = sum(1 if s == Side.BUY else -1 for ts, s, _ in pred.detections if t0 <= ts < t1)
    vol = sum(q for _, q in buys) or 1
    conc = 0.0
    for p in PERIODS_US:
        bins = np.zeros(20)
        for ts, q in buys:
            bins[int((ts % p) * 20 // p)] += q
        conc = max(conc, float(bins.max() / vol))
    sizes = np.bincount([q for _, q in buys]) if buys else np.zeros(1)
    mode = float(sizes.max() / max(1, len(buys)))
    return [float(hits), conc, mode], float(vol - sells)


def episode(seed: int, parent: bool, randomise: bool, qty: int = 100,
            horizon_us: int = 2_000_000) -> tuple[list[float], float]:
    m = Market(seed, MarketParams(informed_share=0.0))
    pred = Predator(m)
    m.predators.append(pred)
    m.run_until(100_000)
    t0 = m.t
    if parent:
        SmartRouter(m, Side.BUY, qty, horizon_us, dark=DARK, randomise=randomise,
                    seed=seed, lam_u=1.0).run()
    end = t0 + horizon_us + 200_000
    m.run_until(max(m.t, end))
    return _features(m, pred, t0, end)


def auc(score_pos: np.ndarray, score_neg: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as one half."""
    p = np.asarray(score_pos)[:, None]
    n = np.asarray(score_neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def _logit_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    mu, sd = x.mean(0), x.std(0) + 1e-12
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])

    def nll(w: np.ndarray) -> tuple[float, np.ndarray]:
        s = z @ w
        ll = np.logaddexp(0, s) - y * s
        g = z.T @ (1 / (1 + np.exp(-s)) - y)
        return float(ll.sum() + ridge * w @ w), g + 2 * ridge * w

    w = minimize(nll, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B").x
    return np.concatenate([w, mu, sd])


def _logit_score(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    k = x.shape[1]
    w, mu, sd = params[:k + 1], params[k + 1:2 * k + 1], params[2 * k + 1:]
    return np.column_stack([np.ones(len(x)), (x - mu) / sd]) @ w


@dataclass(frozen=True)
class GamingResult:
    randomise: bool
    n_per_class: int
    auc: float
    ci: tuple[float, float]
    imbalance_auc: float


def detector_auc(randomise: bool, n: int = 60, seed: int = 0, boot: int = 1000) -> GamingResult:
    """Train on half the episodes, AUC on the other half, bootstrap CI."""
    xs, ys, imb = [], [], []
    for i in range(n):
        for parent in (True, False):
            f, im = episode(seed * 100_000 + 2 * i + int(parent), parent, randomise)
            xs.append(f)
            ys.append(int(parent))
            imb.append(im)
    x, y, ib = np.asarray(xs), np.asarray(ys, float), np.asarray(imb)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    tr, te = idx[: len(y) // 2], idx[len(y) // 2:]
    params = _logit_fit(x[tr], y[tr])
    s = _logit_score(params, x[te])
    yt = y[te]
    a = auc(s[yt == 1], s[yt == 0])
    bs = []
    for _ in range(boot):
        j = rng.integers(0, len(te), len(te))
        sj, yj = s[j], yt[j]
        if yj.min() == yj.max():
            continue
        bs.append(auc(sj[yj == 1], sj[yj == 0]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ia = auc(ib[te][yt == 1], ib[te][yt == 0])
    return GamingResult(randomise, n, a, (float(lo), float(hi)), ia)
