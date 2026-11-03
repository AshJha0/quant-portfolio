"""Counterparty CVA and DVA with collateral and wrong-way risk.

The existing ``exposure.cva`` integrates a real-world PD term structure
against an uncollateralised profile. This module is the desk version:

1. **Risk-neutral hazard.** Default probabilities come from a hazard
   lambda implied from CDS (credit triangle lambda = s / (1 - R), or a
   bootstrapped curve passed as survival values). Reference: Gregory (2020),
   "The xVA Challenge", 4th ed., section 12.2.

2. **Unilateral and bilateral CVA.** On a grid 0 = t_0 < ... < t_n:

       CVA = (1 - R_c) sum_i D(t_i) EPE(t_i) Q_b(t_i) [Q_c(t_{i-1}) - Q_c(t_i)]
       DVA = (1 - R_b) sum_i D(t_i) ENE(t_i) Q_c(t_i) [Q_b(t_{i-1}) - Q_b(t_i)]

   with ENE >= 0 the expected negative exposure, Q_c and Q_b the survival
   curves of the counterparty and the bank. Setting Q_b = 1 gives the
   unilateral CVA. The Q_b(t_i) factor is the first-to-default weighting of
   Gregory (2020) eq. (14.3), assuming independent defaults.

3. **CSA with margin period of risk.** Under a two-way CSA with threshold H
   and minimum transfer amount MTA, the collateral held at t is the value
   called at t - MPoR (the last successful call before the close-out):

       C(t) = sign(V) * max(|V(t - MPoR)| - H, 0),  set to 0 when below MTA,
       E(t) = max(V(t) - C(t), 0).

   Reference: BCBS 189 (Basel III, 2011) para 41 (MPoR); Gregory (2020)
   section 8.5. Closed-form check: with zero rates, H = 0 and a forward on
   a GBM spot, EPE(t) = N S_0 (2 Phi(sigma sqrt(MPoR) / 2) - 1).

4. **Wrong-way risk.** Hull and White (2012), "CVA and wrong-way risk",
   Financial Analysts Journal 68(5), eq. (3): the counterparty hazard on each
   path depends on the trade value, lambda(t) = exp(a(t) + b V(t)). a(t) is
   calibrated step by step so the average survival over paths matches the
   CDS-implied curve; b > 0 is wrong-way, b < 0 right-way, and b = 0 gives
   back the independent CVA on the same paths exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qp_core.model import model
from scipy.optimize import brentq
from scipy.stats import norm


def flat_survival(times: np.ndarray, hazard: float) -> np.ndarray:
    """Q(t) = exp(-lambda t)."""
    if hazard < 0.0:
        raise ValueError("hazard must be non-negative")
    return np.exp(-hazard * np.asarray(times, dtype=float))


def hazard_from_spread(spread: float, recovery: float) -> float:
    """Credit triangle: lambda = s / (1 - R) (exact for continuous premium)."""
    if spread < 0.0 or not 0.0 <= recovery < 1.0:
        raise ValueError("need spread >= 0 and recovery in [0, 1)")
    return spread / (1.0 - recovery)


def _check_grid(times: np.ndarray, *arrays: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 2 or t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be a 1-D increasing grid starting at 0")
    for a in arrays:
        if np.asarray(a).shape != t.shape:
            raise ValueError("every profile must have the same shape as times")
    return t


@dataclass(frozen=True)
class XvaResult:
    cva: float
    dva: float

    @property
    def bcva(self) -> float:
        """Bilateral adjustment to the risk-free value: -CVA + DVA."""
        return self.dva - self.cva


@model(id="FX-CR-CVA")
def bilateral_cva(
    times: np.ndarray,
    epe: np.ndarray,
    ene: np.ndarray,
    surv_cpty: np.ndarray,
    surv_own: np.ndarray,
    recovery_cpty: float = 0.4,
    recovery_own: float = 0.4,
    rate: float = 0.0,
) -> XvaResult:
    """CVA and DVA on a time grid (see module docstring, item 2).

    ``ene`` is the expected negative exposure as a non-negative number.
    """
    t = _check_grid(times, epe, ene, surv_cpty, surv_own)
    epe_a, ene_a = np.asarray(epe, float), np.asarray(ene, float)
    qc, qb = np.asarray(surv_cpty, float), np.asarray(surv_own, float)
    if np.any(epe_a < 0.0) or np.any(ene_a < 0.0):
        raise ValueError("epe and ene must be non-negative")
    if qc[0] != 1.0 or qb[0] != 1.0 or np.any(np.diff(qc) > 0) or np.any(np.diff(qb) > 0):
        raise ValueError("survival curves must start at 1 and be non-increasing")
    d = np.exp(-rate * t[1:])
    cva = (1.0 - recovery_cpty) * float(np.sum(d * epe_a[1:] * qb[1:] * -np.diff(qc)))
    dva = (1.0 - recovery_own) * float(np.sum(d * ene_a[1:] * qc[1:] * -np.diff(qb)))
    return XvaResult(cva, dva)


def simulate_forward_values(
    spot: float,
    strike: float,
    notional: float,
    maturity: float,
    vol: float,
    rd: float,
    rf: float,
    n_steps: int,
    n_paths: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark-to-market paths of a long FX forward (buy ``notional`` foreign at ``strike``).

    GBM spot under the domestic risk-neutral measure (Garman-Kohlhagen):
    V(t) = N (S_t exp(-rf (T - t)) - K exp(-rd (T - t))). Antithetic normals.
    Returns (times, values) with values of shape (n_paths, n_steps + 1).
    """
    if min(spot, strike, maturity, vol) <= 0.0 or n_steps < 1 or n_paths < 2 or n_paths % 2:
        raise ValueError("need positive spot/strike/maturity/vol, n_steps >= 1, even n_paths >= 2")
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    z = rng.standard_normal((n_paths // 2, n_steps))
    z = np.vstack([z, -z])
    inc = (rd - rf - 0.5 * vol * vol) * dt + vol * np.sqrt(dt) * z
    log_s = np.log(spot) + np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)
    t = np.linspace(0.0, maturity, n_steps + 1)
    tau = maturity - t
    v = notional * (np.exp(log_s) * np.exp(-rf * tau) - strike * np.exp(-rd * tau))
    return t, v


def collateralised_exposure(
    values: np.ndarray,
    mpor_steps: int,
    threshold: float = 0.0,
    mta: float = 0.0,
) -> np.ndarray:
    """Positive exposure per path under a two-way CSA (module docstring, item 3).

    ``mpor_steps`` is the margin period of risk in grid steps; before the
    first call (t < MPoR) no collateral is held.
    """
    v = np.asarray(values, dtype=float)
    if mpor_steps < 0 or threshold < 0.0 or mta < 0.0:
        raise ValueError("mpor_steps, threshold and mta must be non-negative")
    lagged = np.zeros_like(v)
    if mpor_steps == 0:
        lagged = v.copy()
    else:
        lagged[:, mpor_steps:] = v[:, :-mpor_steps]
    coll = np.sign(lagged) * np.maximum(np.abs(lagged) - threshold, 0.0)
    coll[np.abs(coll) < mta] = 0.0
    return np.maximum(v - coll, 0.0)


def epe_mpor_closed_form(spot: float, notional: float, vol: float, mpor: float) -> float:
    """EPE of a fully collateralised (H = 0) forward with zero rates, for t >= MPoR."""
    return notional * spot * (2.0 * norm.cdf(0.5 * vol * np.sqrt(mpor)) - 1.0)


def cva_wrong_way(
    times: np.ndarray,
    values: np.ndarray,
    surv_target: np.ndarray,
    b: float,
    recovery: float = 0.4,
    rate: float = 0.0,
) -> float:
    """Hull-White (2012) wrong-way CVA on simulated value paths.

    Hazard on path p over (t_{i-1}, t_i] is exp(a_i + b V_p(t_i)); a_i is
    solved so mean_p Q_p(t_i) equals ``surv_target[i]``.
    """
    t = _check_grid(times, surv_target)
    v = np.asarray(values, dtype=float)
    q_tgt = np.asarray(surv_target, dtype=float)
    if v.ndim != 2 or v.shape[1] != t.size:
        raise ValueError("values must have shape (n_paths, len(times))")
    if q_tgt[0] != 1.0 or np.any(np.diff(q_tgt) >= 0.0):
        raise ValueError("surv_target must start at 1 and strictly decrease")
    log_q = np.zeros(v.shape[0])
    cva = 0.0
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        bv = b * v[:, i]
        shift = float(np.max(bv))

        def gap(
            a: float,
            bv: np.ndarray = bv,
            dt: float = dt,
            shift: float = shift,
            log_q: np.ndarray = log_q,
            target: float = float(q_tgt[i]),
        ) -> float:
            lam = np.exp(a + bv - shift)
            return float(np.mean(np.exp(log_q - lam * dt))) - target

        a = brentq(gap, -60.0, 60.0, xtol=1e-14, rtol=1e-14, maxiter=500)
        new_log_q = log_q - np.exp(a + bv - shift) * dt
        default_prob = np.exp(log_q) - np.exp(new_log_q)
        cva += np.exp(-rate * t[i]) * float(np.mean(np.maximum(v[:, i], 0.0) * default_prob))
        log_q = new_log_q
    return (1.0 - recovery) * cva
