"""Seeded synthetic corporate loan book with a KNOWN ground-truth PD model.

The generator produces an origination-cohort loan book whose one-year default
indicator is drawn from a fully specified "true" PD model.  Because the true
log-odds are known in closed form, every downstream stage (WOE binning, IRLS
logistic regression, validation, capital) can be tested against ground truth.

True log-odds specification (all effects on the 1-year default log-odds)
------------------------------------------------------------------------
    eta_i = b0 + f_lev(leverage) + f_ic(interest_coverage) + f_cr(current_ratio)
               + f_roa(roa) + f_size(log_assets) + f_beh(behavioral_score)
               + sector_effect[sector]                     (+ calibration_shift)

with deliberately *nonlinear* components so that WOE binning has something
real to find that a linear logit on raw features misses:

* ``f_lev``  — piecewise-linear, **capped**: risk increases in leverage up to
  1.0 (debt = assets) and is flat beyond (already distressed).
* ``f_ic``   — **saturating**: extra interest coverage beyond 6x buys nothing.
* ``f_cr``   — **U-shaped** in current ratio: both very low (illiquidity) and
  very high (idle working capital / dying sales) are risky; minimum risk near
  a current ratio of 1.5.
* ``noise_1``, ``noise_2`` — generated but given ZERO true effect.

``b0`` is calibrated by bisection so the population mean PD hits
``target_default_rate`` (default 3%).

Data pathologies (all optional, all seeded):

* MCAR missingness on ``current_ratio``.
* **Informative** missingness on ``behavioral_score``: riskier obligors are
  more likely to have no payment history ("thin file"), so missingness itself
  carries signal — the WOE missing bin should pick this up.
* Measurement outliers on ``leverage`` and ``current_ratio`` (injected after
  the true PD is computed, i.e. pure noise).
* Optional exact duplicate rows (data-entry duplicates).
* Post-outcome fields (``writeoff_flag``, ``recovery_amount``,
  ``days_past_due_max``) that are functions of the default outcome — planted
  leakage for the cleaning/WOE guards to catch.

Units and conventions: leverage = total debt / total assets (ratio);
interest_coverage = EBITDA / interest expense (x); roa = net income / assets
(decimal p.a.); log_assets = ln(total assets in USD); behavioral_score in
[0, 100] (higher = better payment history); EAD in USD; LGD as a decimal
fraction of EAD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "SECTORS",
    "SECTOR_EFFECTS",
    "TRUE_COEFFS",
    "true_log_odds",
    "generate_loan_book",
    "generate_oot_sample",
]

SECTORS: tuple[str, ...] = (
    "manufacturing",
    "retail",
    "construction",
    "services",
    "energy",
    "healthcare",
)

#: Additive sector effects on the true log-odds (construction/retail riskier).
SECTOR_EFFECTS: dict[str, float] = {
    "manufacturing": 0.00,
    "retail": 0.35,
    "construction": 0.55,
    "services": -0.10,
    "energy": 0.20,
    "healthcare": -0.30,
}

#: Slopes of the (transformed) true log-odds components.  Noise features have
#: zero true effect by construction.
TRUE_COEFFS: dict[str, float] = {
    "leverage_capped": 2.1,      # per unit of min(leverage, 1.0)
    "ic_saturating": -0.22,      # per unit of clip(interest_coverage, -2, 6)
    "cr_ushape": 0.95,           # per unit of min(ln(cr / 1.5)^2, 4)
    "roa": -5.0,                 # per unit of ROA
    "log_assets": -0.22,         # per unit of (log_assets - 17)
    "behavioral_score": -0.024,  # per point of (behavioral_score - 60)
    "noise_1": 0.0,
    "noise_2": 0.0,
}


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))


def true_log_odds(
    leverage: np.ndarray,
    interest_coverage: np.ndarray,
    current_ratio: np.ndarray,
    roa: np.ndarray,
    log_assets: np.ndarray,
    behavioral_score: np.ndarray,
    sector: np.ndarray,
    intercept: float = 0.0,
) -> np.ndarray:
    """Ground-truth log-odds of 1-year default, excluding noise features.

    Parameters
    ----------
    leverage, interest_coverage, current_ratio, roa, log_assets,
    behavioral_score : np.ndarray
        Raw (clean, pre-outlier, non-missing) feature arrays.
    sector : np.ndarray
        Array of sector labels drawn from :data:`SECTORS`.
    intercept : float
        Additive intercept ``b0``.

    Returns
    -------
    np.ndarray
        True log-odds ``eta`` per obligor.
    """
    c = TRUE_COEFFS
    f_lev = c["leverage_capped"] * np.minimum(leverage, 1.0)
    f_ic = c["ic_saturating"] * np.clip(interest_coverage, -2.0, 6.0)
    with np.errstate(divide="ignore"):
        log_ratio = np.log(np.maximum(current_ratio, 1e-6) / 1.5)
    f_cr = c["cr_ushape"] * np.minimum(log_ratio**2, 4.0)
    f_roa = c["roa"] * roa
    f_size = c["log_assets"] * (log_assets - 17.0)
    f_beh = c["behavioral_score"] * (behavioral_score - 60.0)
    f_sec = np.array([SECTOR_EFFECTS[s] for s in sector])
    return intercept + f_lev + f_ic + f_cr + f_roa + f_size + f_beh + f_sec


def _calibrate_intercept(eta_no_intercept: np.ndarray, target: float) -> float:
    """Bisect ``b0`` so that ``mean(sigmoid(b0 + eta))`` equals ``target``."""
    lo, hi = -20.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _sigmoid(mid + eta_no_intercept).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_loan_book(
    n_loans: int = 20_000,
    seed: int = 42,
    *,
    target_default_rate: float = 0.03,
    drift: float = 0.0,
    calibration_shift: float = 0.0,
    start: str = "2019-01-01",
    end: str = "2021-12-31",
    missing: bool = True,
    outliers: bool = True,
    n_duplicates: int = 0,
    include_post_outcome: bool = True,
) -> pd.DataFrame:
    """Generate a synthetic corporate loan book with known true PDs.

    Parameters
    ----------
    n_loans : int
        Number of loans before duplicate injection.
    seed : int
        Seed for the ``numpy.random.Generator``; identical inputs give
        identical output.
    target_default_rate : float
        Population mean of the true PD; the intercept is bisected to hit it
        (before ``calibration_shift`` is applied).
    drift : float
        Feature-distribution drift intensity in [0, ~1]: shifts leverage up,
        scales interest coverage down and shifts behavioral scores down.
        Use for out-of-time samples.
    calibration_shift : float
        Additive shift on the true log-odds AFTER intercept calibration —
        shifts the realised base rate away from ``target_default_rate``
        (portfolio-level miscalibration, e.g. a downturn year).
    start, end : str
        Origination date range (dates uniform over the range).
    missing : bool
        Inject MCAR missingness on ``current_ratio`` (~4%) and informative
        missingness on ``behavioral_score`` (riskier => more likely missing).
    outliers : bool
        Inject measurement outliers (0.5% of rows) on ``leverage`` and
        ``current_ratio`` after the true PD is computed.
    n_duplicates : int
        Number of exact duplicate rows appended (same ``loan_id``).
    include_post_outcome : bool
        Include post-outcome (forbidden, leaky) fields ``writeoff_flag``,
        ``recovery_amount`` and ``days_past_due_max``.

    Returns
    -------
    pd.DataFrame
        Columns: ``loan_id, origination_date, sector, leverage,
        interest_coverage, current_ratio, roa, log_assets, behavioral_score,
        noise_1, noise_2, ead, lgd, default, true_pd`` plus, if requested,
        the post-outcome fields.
    """
    if n_loans < 1:
        raise ValueError("n_loans must be >= 1")
    rng = np.random.default_rng(seed)
    n = n_loans

    log_assets = rng.normal(17.0, 1.5, n)
    leverage = 1.6 * rng.beta(2.0, 4.0, n) + 0.12 * drift
    interest_coverage = (np.exp(rng.normal(1.2, 0.8, n)) - 0.5) * (1.0 - 0.25 * drift)
    current_ratio = np.exp(rng.normal(np.log(1.5), 0.5, n))
    roa = rng.normal(0.05, 0.06, n) - 0.015 * drift
    sector = rng.choice(SECTORS, size=n, p=[0.25, 0.18, 0.12, 0.22, 0.10, 0.13])

    # Behavioral score correlated with fundamentals (so it is predictive) plus
    # its own noise; drift pushes scores down (e.g. payment-holiday distortions).
    behavioral_score = np.clip(
        60.0
        - 14.0 * leverage
        + 4.0 * np.log1p(np.maximum(interest_coverage, 0.0))
        + 60.0 * roa
        + rng.normal(0.0, 11.0, n)
        - 4.0 * drift,
        0.0,
        100.0,
    )

    noise_1 = rng.normal(0.0, 1.0, n)
    noise_2 = rng.uniform(-1.0, 1.0, n)

    eta_raw = true_log_odds(
        leverage, interest_coverage, current_ratio, roa, log_assets,
        behavioral_score, sector,
    )
    b0 = _calibrate_intercept(eta_raw, target_default_rate)
    eta = b0 + eta_raw + calibration_shift
    true_pd = _sigmoid(eta)
    default = (rng.uniform(size=n) < true_pd).astype(np.int64)

    # Exposure and loss-given-default (independent of PD by construction —
    # see the assumptions register in docs/METHODOLOGY.md).
    ead = np.exp(rng.normal(np.log(1_000_000.0), 0.8, n))
    lgd = 0.1 + 0.8 * rng.beta(2.0, 2.0, n)

    dates = pd.to_datetime(start) + pd.to_timedelta(
        rng.integers(0, max((pd.to_datetime(end) - pd.to_datetime(start)).days, 1), n),
        unit="D",
    )

    df = pd.DataFrame(
        {
            "loan_id": [f"L{seed:04d}-{i:06d}" for i in range(n)],
            "origination_date": dates,
            "sector": sector,
            "leverage": leverage,
            "interest_coverage": interest_coverage,
            "current_ratio": current_ratio,
            "roa": roa,
            "log_assets": log_assets,
            "behavioral_score": behavioral_score,
            "noise_1": noise_1,
            "noise_2": noise_2,
            "ead": ead,
            "lgd": lgd,
            "default": default,
            "true_pd": true_pd,
        }
    )

    if include_post_outcome:
        # Pure post-outcome fields: knowable only AFTER the default outcome.
        flip = rng.uniform(size=n) < 0.03  # 3% label noise so IV is huge, not inf
        df["writeoff_flag"] = np.where(flip, 1 - default, default).astype(np.int64)
        df["recovery_amount"] = np.where(
            default == 1, ead * (1.0 - lgd) * rng.uniform(0.8, 1.2, n), 0.0
        )
        df["days_past_due_max"] = np.where(
            default == 1,
            rng.integers(90, 720, n),
            rng.integers(0, 30, n),
        ).astype(np.int64)

    if outliers:
        k = max(int(0.005 * n), 1) if n >= 200 else 0
        if k:
            idx = rng.choice(n, size=k, replace=False)
            df.loc[idx[: k // 2], "leverage"] *= 6.0
            df.loc[idx[k // 2 :], "current_ratio"] *= 12.0

    if missing:
        mcar = rng.uniform(size=n) < 0.04
        df.loc[mcar, "current_ratio"] = np.nan
        # Informative missingness: thin-file obligors are riskier.
        p_miss = np.clip(0.02 + 2.5 * true_pd, 0.0, 0.40)
        beh_missing = rng.uniform(size=n) < p_miss
        df.loc[beh_missing, "behavioral_score"] = np.nan

    if n_duplicates > 0:
        dup_idx = rng.choice(n, size=min(n_duplicates, n), replace=False)
        df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    return df


def generate_oot_sample(
    n_loans: int = 8_000,
    seed: int = 123,
    *,
    drift: float = 0.5,
    calibration_shift: float = 0.25,
    target_default_rate: float = 0.03,
    start: str = "2022-01-01",
    end: str = "2022-12-31",
    **kwargs: object,
) -> pd.DataFrame:
    """Out-of-time sample: later origination dates, drifted features and a
    shifted calibration (higher realised base rate).

    Parameters mirror :func:`generate_loan_book`; ``drift`` and
    ``calibration_shift`` default to a mild-but-visible regime change.
    """
    return generate_loan_book(
        n_loans,
        seed,
        target_default_rate=target_default_rate,
        drift=drift,
        calibration_shift=calibration_shift,
        start=start,
        end=end,
        **kwargs,  # type: ignore[arg-type]
    )
