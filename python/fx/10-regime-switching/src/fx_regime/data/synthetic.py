"""Synthetic risk-on / risk-off (RORO) FX panel generator with known parameters.

Everything in this module is deterministic given a seed and fully offline.
The generator is the ground truth against which the whole pipeline is
validated: it produces currency-vs-USD log-return panels driven by a hidden
Markov chain with a KNOWN transition matrix and KNOWN per-state, per-block
means / vols / correlations, plus a persistent deposit-rate panel so that
carry accrual can be computed exactly.

Conventions
-----------
* Returns are DAILY LOG RETURNS of each currency measured against USD:
  a positive return means the currency APPRECIATED against USD.  (For a
  BASE/QUOTE pair like AUDUSD this is the log return of the quoted price;
  for USDJPY it is minus the log return of the quoted price.)
* Deposit rates are ANNUALISED, continuously-compounded-style simple
  decimals (0.045 = 4.5%).  Daily carry accrual for a long position in
  currency c against USD is (r_c - r_USD) / 252 (ACT/252 simplification,
  documented in docs/METHODOLOGY.md).
* Vols and drifts below are annualised; the generator converts with
  1/252 and 1/sqrt(252).

States
------
0 ``risk_on``     : carry / EM currencies drift up vs USD, havens drift
                    down, vols low, correlations moderate.
1 ``risk_off``    : violent carry unwind — carry / EM fall hard, JPY/CHF
                    rally, vols spike, risk-block correlations spike,
                    haven-vs-carry correlation turns strongly negative.
2 ``usd_squeeze`` : (3-state variant only) everything falls against USD,
                    havens included, with very high common correlation —
                    a 2008 / March-2020 dollar-funding squeeze.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Currency universe and blocks
# --------------------------------------------------------------------------

G10_CARRY: tuple[str, ...] = ("AUD", "NZD", "NOK", "CAD")
G10_NEUTRAL: tuple[str, ...] = ("EUR", "GBP", "SEK")
HAVENS: tuple[str, ...] = ("JPY", "CHF")
EM: tuple[str, ...] = ("MXN", "ZAR", "BRL")

CURRENCIES: tuple[str, ...] = G10_CARRY + G10_NEUTRAL + HAVENS + EM
G10: tuple[str, ...] = G10_CARRY + G10_NEUTRAL + HAVENS

STATE_NAMES_2 = ("risk_on", "risk_off")
STATE_NAMES_3 = ("risk_on", "risk_off", "usd_squeeze")

#: Long-run mean deposit rates (annualised decimals).  Persistent carry
#: differentials: antipodeans and EM high, JPY/CHF at/below zero.
MEAN_DEPOSIT_RATES: dict[str, float] = {
    "AUD": 0.045,
    "NZD": 0.055,
    "NOK": 0.040,
    "CAD": 0.035,
    "EUR": 0.015,
    "GBP": 0.025,
    "SEK": 0.010,
    "JPY": 0.001,
    "CHF": -0.005,
    "MXN": 0.090,
    "ZAR": 0.075,
    "BRL": 0.100,
    "USD": 0.020,
}

#: Half-spread transaction costs per currency-vs-USD leg, in PIPS on a
#: quote near 1.0 (1 pip = 1e-4 in price ≈ 1 bp of notional).  EM wider.
SPREAD_PIPS: dict[str, float] = {
    "AUD": 1.0,
    "NZD": 1.5,
    "NOK": 2.5,
    "CAD": 1.0,
    "EUR": 0.5,
    "GBP": 0.8,
    "SEK": 2.5,
    "JPY": 0.8,
    "CHF": 1.0,
    "MXN": 6.0,
    "ZAR": 10.0,
    "BRL": 12.0,
}
PIP_FRACTION = 1e-4  # 1 pip expressed as a fraction of notional

# --------------------------------------------------------------------------
# Per-state annualised drift / vol tables (currency vs USD)
# --------------------------------------------------------------------------

_DRIFT = {
    "risk_on": {"carry": 0.020, "neutral": 0.000, "haven": -0.020, "em": 0.010},
    "risk_off": {"carry": -0.250, "neutral": -0.080, "haven": 0.130, "em": -0.400},
    "usd_squeeze": {"carry": -0.240, "neutral": -0.170, "haven": -0.160, "em": -0.350},
}
_VOL = {
    "risk_on": {"carry": 0.090, "neutral": 0.080, "haven": 0.080, "em": 0.120},
    "risk_off": {"carry": 0.200, "neutral": 0.120, "haven": 0.130, "em": 0.300},
    "usd_squeeze": {"carry": 0.190, "neutral": 0.150, "haven": 0.120, "em": 0.280},
}
# (within_risk, within_haven, risk_haven_cross, neutral_loading)
_CORR = {
    "risk_on": {"rr": 0.45, "hh": 0.50, "hr": 0.10, "nr": 0.30, "hn": 0.15, "nn": 0.40},
    "risk_off": {"rr": 0.85, "hh": 0.70, "hr": -0.50, "nr": 0.65, "hn": -0.30, "nn": 0.60},
    "usd_squeeze": {"rr": 0.85, "hh": 0.75, "hr": 0.60, "nr": 0.80, "hn": 0.55, "nn": 0.80},
}

TRANSITION_2 = np.array([[0.990, 0.010], [0.050, 0.950]])
TRANSITION_3 = np.array(
    [
        [0.985, 0.011, 0.004],
        [0.030, 0.950, 0.020],
        [0.015, 0.045, 0.940],
    ]
)


def _block_of(ccy: str) -> str:
    if ccy in G10_CARRY:
        return "carry"
    if ccy in G10_NEUTRAL:
        return "neutral"
    if ccy in HAVENS:
        return "haven"
    if ccy in EM:
        return "em"
    raise ValueError(f"unknown currency {ccy!r}")


def state_correlation(state: str, currencies: tuple[str, ...] = CURRENCIES) -> np.ndarray:
    """Build the block correlation matrix for one state.

    EM currencies share the risk block's correlation entries.  The matrix
    is symmetrised and, if needed, shifted to be positive definite (min
    eigenvalue >= 1e-6) — the shift is tiny and re-normalised to unit
    diagonal.

    Parameters
    ----------
    state : {"risk_on", "risk_off", "usd_squeeze"}
    currencies : tuple of str

    Returns
    -------
    (p, p) ndarray, unit diagonal, positive definite.
    """
    c = _CORR[state]
    p = len(currencies)
    corr = np.eye(p)
    kind = {"carry": "r", "em": "r", "neutral": "n", "haven": "h"}
    for i in range(p):
        for j in range(i + 1, p):
            a, b = kind[_block_of(currencies[i])], kind[_block_of(currencies[j])]
            key = "".join(sorted(a + b))
            corr[i, j] = corr[j, i] = c[key]
    w, _ = np.linalg.eigh(corr)
    if w.min() < 1e-6:
        corr = corr + (1e-6 - w.min()) * np.eye(p)
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    return corr


def state_moments(
    state: str, currencies: tuple[str, ...] = CURRENCIES
) -> tuple[np.ndarray, np.ndarray]:
    """Annualised drift vector and covariance matrix for one state.

    Returns
    -------
    mu : (p,) annualised drifts of currency-vs-USD log returns.
    cov : (p, p) annualised covariance.
    """
    mu = np.array([_DRIFT[state][_block_of(c)] for c in currencies])
    vol = np.array([_VOL[state][_block_of(c)] for c in currencies])
    corr = state_correlation(state, currencies)
    cov = corr * np.outer(vol, vol)
    return mu, cov


@dataclass
class SyntheticPanel:
    """Container for a generated RORO panel with full ground truth.

    Attributes
    ----------
    returns : DataFrame (T x p)
        Daily log returns of each currency vs USD.
    deposit_rates : DataFrame (T x (p+1))
        Annualised deposit rates, includes a ``USD`` column.
    states : ndarray of int (T,)
        True hidden state at each date (0=risk_on, 1=risk_off,
        2=usd_squeeze in the 3-state variant).
    state_names : tuple of str
    transition : ndarray (k, k)
        True transition matrix.
    means : ndarray (k, p)
        True DAILY mean log return per state.
    covs : ndarray (k, p, p)
        True DAILY covariance per state.
    blocks : dict
        Currency-block membership.
    """

    returns: pd.DataFrame
    deposit_rates: pd.DataFrame
    states: np.ndarray
    state_names: tuple[str, ...]
    transition: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    blocks: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "carry": G10_CARRY,
            "neutral": G10_NEUTRAL,
            "haven": HAVENS,
            "em": EM,
        }
    )

    @property
    def n_states(self) -> int:
        return len(self.state_names)


def simulate_markov_chain(
    transition: np.ndarray,
    n_periods: int,
    rng: np.random.Generator,
    initial_state: int = 0,
) -> np.ndarray:
    """Simulate a first-order Markov chain path.

    Parameters
    ----------
    transition : (k, k) row-stochastic matrix.
    n_periods : int
    rng : numpy Generator
    initial_state : int

    Returns
    -------
    (n_periods,) int array of states.
    """
    transition = np.asarray(transition, dtype=float)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be square")
    if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("transition rows must sum to 1")
    k = transition.shape[0]
    if not 0 <= initial_state < k:
        raise ValueError("initial_state out of range")
    cdf = np.cumsum(transition, axis=1)
    states = np.empty(n_periods, dtype=int)
    s = initial_state
    u = rng.random(n_periods)
    for t in range(n_periods):
        states[t] = s
        s = int(np.searchsorted(cdf[s], u[t], side="right"))
        s = min(s, k - 1)
    return states


def _simulate_rates(
    n_periods: int, rng: np.random.Generator, phi: float = 0.998, sigma: float = 0.0015
) -> pd.DataFrame:
    """AR(1) deposit-rate panel around persistent long-run means."""
    cols = list(CURRENCIES) + ["USD"]
    mu = np.array([MEAN_DEPOSIT_RATES[c] for c in cols])
    r = np.empty((n_periods, len(cols)))
    r[0] = mu
    eps = rng.standard_normal((n_periods, len(cols))) * sigma / np.sqrt(252.0)
    for t in range(1, n_periods):
        r[t] = mu + phi * (r[t - 1] - mu) + eps[t]
    return pd.DataFrame(r, columns=cols)


def generate_roro_panel(
    n_periods: int = 2000,
    n_states: int = 3,
    seed: int = 0,
    transition: np.ndarray | None = None,
    plant_flip_at: int | None = None,
    plant_flip_len: int = 45,
    plant_flip_state: int = 1,
    start: str = "2015-01-01",
) -> SyntheticPanel:
    """Generate a RORO currency-vs-USD return panel with known ground truth.

    Parameters
    ----------
    n_periods : int
        Number of business days.  Must be >= 30.
    n_states : {2, 3}
        2-state = risk_on/risk_off; 3-state adds a USD-squeeze state.
    seed : int
        Seed for the numpy Generator (fully deterministic output).
    transition : optional (k, k) row-stochastic matrix
        Overrides the default known transition matrix.
    plant_flip_at : optional int
        If given, force the hidden state to ``plant_flip_state`` for
        ``plant_flip_len`` days starting at this index — a planted
        2008-style flip at a known date (the chain resumes from that
        state afterwards).
    plant_flip_len, plant_flip_state : int
        Length and state of the planted episode.
    start : str
        First business date of the index.

    Returns
    -------
    SyntheticPanel
    """
    if n_periods < 30:
        raise ValueError("n_periods must be >= 30")
    if n_states not in (2, 3):
        raise ValueError("n_states must be 2 or 3")
    rng = np.random.default_rng(seed)
    state_names = STATE_NAMES_2 if n_states == 2 else STATE_NAMES_3
    P = TRANSITION_2 if n_states == 2 else TRANSITION_3
    if transition is not None:
        P = np.asarray(transition, dtype=float)
        if P.shape != (n_states, n_states):
            raise ValueError("transition has wrong shape")

    states = simulate_markov_chain(P, n_periods, rng, initial_state=0)
    if plant_flip_at is not None:
        if not 0 <= plant_flip_at < n_periods:
            raise ValueError("plant_flip_at out of range")
        if not 0 <= plant_flip_state < n_states:
            raise ValueError("plant_flip_state out of range")
        end = min(plant_flip_at + plant_flip_len, n_periods)
        states[plant_flip_at:end] = plant_flip_state
        # resume the chain from the planted state so there is no
        # impossible jump at the seam
        if end < n_periods:
            tail = simulate_markov_chain(
                P, n_periods - end, np.random.default_rng(seed + 1),
                initial_state=plant_flip_state,
            )
            states[end:] = tail

    p = len(CURRENCIES)
    means = np.empty((n_states, p))
    covs = np.empty((n_states, p, p))
    chols = []
    for s, name in enumerate(state_names):
        mu_a, cov_a = state_moments(name)
        means[s] = mu_a / 252.0
        covs[s] = cov_a / 252.0
        chols.append(np.linalg.cholesky(covs[s]))

    z = rng.standard_normal((n_periods, p))
    rets = np.empty((n_periods, p))
    for t in range(n_periods):
        s = states[t]
        rets[t] = means[s] + chols[s] @ z[t]

    dates = pd.bdate_range(start, periods=n_periods)
    returns = pd.DataFrame(rets, index=dates, columns=list(CURRENCIES))
    rates = _simulate_rates(n_periods, rng)
    rates.index = dates

    return SyntheticPanel(
        returns=returns,
        deposit_rates=rates,
        states=states,
        state_names=state_names,
        transition=P,
        means=means,
        covs=covs,
    )


def generate_null_gbm_panel(
    n_periods: int = 2000, seed: int = 0, start: str = "2015-01-01"
) -> SyntheticPanel:
    """Single-regime Gaussian ('null GBM') panel — no regime structure.

    Uses the risk_on moments for every day with zero drift.  Any regime
    filter run on this panel should find nothing exploitable; BIC should
    prefer one state.  Used as the null-hypothesis guard in tests.

    Returns
    -------
    SyntheticPanel with ``states`` all zero and a 1x1 transition matrix.
    """
    if n_periods < 30:
        raise ValueError("n_periods must be >= 30")
    rng = np.random.default_rng(seed)
    p = len(CURRENCIES)
    _, cov_a = state_moments("risk_on")
    cov = cov_a / 252.0
    chol = np.linalg.cholesky(cov)
    rets = rng.standard_normal((n_periods, p)) @ chol.T
    dates = pd.bdate_range(start, periods=n_periods)
    returns = pd.DataFrame(rets, index=dates, columns=list(CURRENCIES))
    rates = _simulate_rates(n_periods, rng)
    rates.index = dates
    return SyntheticPanel(
        returns=returns,
        deposit_rates=rates,
        states=np.zeros(n_periods, dtype=int),
        state_names=("null",),
        transition=np.array([[1.0]]),
        means=np.zeros((1, p)),
        covs=cov[None, :, :],
    )
