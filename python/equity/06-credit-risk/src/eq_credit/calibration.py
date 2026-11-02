"""PD calibration to a central tendency, with a margin of conservatism.

Central tendency
----------------
A ranking model gives PDs whose average need not match the long-run average
default rate (LRA). Calibration shifts every score by the same constant in
log-odds space, which keeps the ranking (and so the AUC) unchanged, until
the portfolio-average PD equals the target:

    PD_i(c) = 1 / (1 + exp(-(logit(PD_i) + c))),   mean_i PD_i(c) = CT.

The mean is strictly increasing in c, so the root is unique (found with
Brent's method). Reference: EBA/GL/2017/16 section 5.3.5 (calibration to the
long-run average default rate) and para 92.

Long-run average default rate
-----------------------------
LRA = the arithmetic mean of yearly one-year default rates (EBA/GL/2017/16
para 80), or a weighted mean when the yearly counts differ a lot.

Margin of conservatism
----------------------
EBA/GL/2017/16 section 4.4.3 asks for an MoC in three categories: A (data
and method deficiencies), B (changes in standards) and C (general
estimation error). This module quantifies category C as the one-sided
confidence half-width of the LRA:

    MoC_C = t_{q, n-1} * s / sqrt(n)

with s the sample standard deviation of the n yearly default rates, and
adds caller-supplied A and B add-ons. The final PD is

    PD_final = max(PD_calibrated + MoC_A + MoC_B + MoC_C, floor),

with the 0.03% floor of CRR article 160(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import t as student_t

PD_FLOOR_CRR = 0.0003


def long_run_average(default_rates: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Arithmetic (or weighted) mean of yearly default rates."""
    dr = np.asarray(default_rates, dtype=float)
    if dr.ndim != 1 or dr.size == 0:
        raise ValueError("default_rates must be a non-empty 1-D array")
    if np.any(dr < 0.0) or np.any(dr > 1.0):
        raise ValueError("default rates must lie in [0, 1]")
    if weights is None:
        return float(np.mean(dr))
    w = np.asarray(weights, dtype=float)
    if w.shape != dr.shape or np.any(w < 0.0) or w.sum() <= 0.0:
        raise ValueError("weights must be non-negative, same shape, positive sum")
    return float(np.sum(w * dr) / w.sum())


def calibrate_to_central_tendency(pd_raw: np.ndarray, target: float) -> tuple[np.ndarray, float]:
    """Shift PDs in log-odds space so their mean equals ``target``.

    Returns (calibrated PDs, shift c).
    """
    p = np.asarray(pd_raw, dtype=float)
    if p.size == 0 or np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("raw PDs must lie strictly inside (0, 1)")
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly inside (0, 1)")
    z = logit(p)

    def gap(c: float) -> float:
        return float(np.mean(expit(z + c))) - target

    lo, hi = -50.0, 50.0
    c = brentq(gap, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    return expit(z + c), float(c)


def moc_estimation_error(default_rates: np.ndarray, confidence: float = 0.90) -> float:
    """Category C MoC: one-sided t half-width of the LRA (see module docstring)."""
    dr = np.asarray(default_rates, dtype=float)
    n = dr.size
    if n < 2:
        raise ValueError("need at least two yearly default rates")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")
    s = float(np.std(dr, ddof=1))
    return float(student_t.ppf(confidence, n - 1) * s / np.sqrt(n))


@dataclass(frozen=True)
class CalibrationResult:
    pd_calibrated: np.ndarray
    pd_final: np.ndarray
    central_tendency: float
    shift: float
    moc_a: float
    moc_b: float
    moc_c: float


def calibrate_with_moc(
    pd_raw: np.ndarray,
    default_rates: np.ndarray,
    confidence: float = 0.90,
    moc_a: float = 0.0,
    moc_b: float = 0.0,
    floor: float = PD_FLOOR_CRR,
) -> CalibrationResult:
    """Calibrate to the LRA, then add the A, B and C margins and apply the floor."""
    if moc_a < 0.0 or moc_b < 0.0:
        raise ValueError("MoC add-ons must be non-negative")
    ct = long_run_average(default_rates)
    pd_cal, c = calibrate_to_central_tendency(pd_raw, ct)
    moc_c = moc_estimation_error(default_rates, confidence)
    final = np.clip(pd_cal + moc_a + moc_b + moc_c, floor, 1.0)
    return CalibrationResult(pd_cal, final, ct, c, moc_a, moc_b, moc_c)
