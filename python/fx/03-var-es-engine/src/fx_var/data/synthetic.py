"""Seeded synthetic FX market generators: G10/EM blocks, regimes, GARCH, pegs.

Everything here is deterministic given a seed and runs offline - it is the
only data source the test suite touches.

Design
------
* **Blocks**: G10 majors and EM currencies form two correlation blocks with
  weaker cross-block links; JPY behaves as a safe haven (its correlation to
  risk currencies flips negative in the stress regime).
* **Regimes**: 'calm' and 'stress' correlation matrices; the two-state
  Markov switcher raises vols (x2) and correlations together, reproducing
  the empirical "correlations go to one in a crisis".
* **GARCH(1,1)** per factor (alpha=0.09, beta=0.89) generates the
  volatility clustering that makes unconditional VaR fail Christoffersen
  backtests - and FHS pass them.
* **Pegs**: managed currencies (HKD-like band, SAR-like hard peg) realise
  near-zero daily vol with a rare Poisson jump - the pattern that blinds
  HS/parametric VaR and motivates the peg-break stress add-on.

All vols are annualised; daily = annual / sqrt(252).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..book import Book, Cash, Forward, Market, Option, Spot
from ..common import TRADING_DAYS_PER_YEAR, fx_factor, ir_factor, split_pair, vol_factor

__all__ = [
    "G10",
    "EM",
    "PEGGED",
    "ANNUAL_VOLS",
    "default_correlation",
    "simulate_fx_returns",
    "simulate_history",
    "demo_market",
    "demo_book",
    "demo_em_book",
]

G10: list[str] = ["EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK"]
EM: list[str] = ["MXN", "BRL", "TRY", "ZAR"]
PEGGED: list[str] = ["HKD", "SAR"]

#: Annualised log-return vols vs USD (stylised long-run levels).
ANNUAL_VOLS: dict[str, float] = {
    "EUR": 0.080, "JPY": 0.100, "GBP": 0.090, "CHF": 0.070, "AUD": 0.110,
    "NZD": 0.120, "CAD": 0.070, "SEK": 0.100, "NOK": 0.110,
    "MXN": 0.130, "BRL": 0.160, "TRY": 0.200, "ZAR": 0.160,
    "HKD": 0.004, "SAR": 0.002,
}

_RISK_CCYS = {"AUD", "NZD", "CAD", "SEK", "NOK"} | set(EM)


def _nearest_psd(corr: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Clip eigenvalues to make a tweaked correlation matrix PSD again."""
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, floor, None)
    m = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(m))
    m = m / np.outer(d, d)
    np.fill_diagonal(m, 1.0)
    return m


def default_correlation(ccys: list[str] | None = None, regime: str = "calm") -> pd.DataFrame:
    """Block correlation matrix across currencies (vs USD) for a regime.

    calm  : G10 intra 0.55, EM intra 0.45, cross 0.25, pegs ~0.
    stress: G10 intra 0.75, EM intra 0.75, cross 0.60, and JPY's correlation
            to risk currencies flips to -0.30 (safe-haven flight).
    """
    if regime not in ("calm", "stress"):
        raise ValueError(f"regime must be 'calm' or 'stress', got {regime!r}")
    if ccys is None:
        ccys = G10 + EM + PEGGED
    n = len(ccys)
    if regime == "calm":
        g10, em, cross, peg = 0.55, 0.45, 0.25, 0.03
    else:
        g10, em, cross, peg = 0.75, 0.75, 0.60, 0.03
    c = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ccys[i], ccys[j]
            if a in PEGGED or b in PEGGED:
                rho = peg
            elif a in G10 and b in G10:
                rho = g10
            elif a in EM and b in EM:
                rho = em
            else:
                rho = cross
            if regime == "stress" and "JPY" in (a, b):
                other = b if a == "JPY" else a
                if other in _RISK_CCYS:
                    rho = -0.30
            c[i, j] = c[j, i] = rho
    c = _nearest_psd(c)
    return pd.DataFrame(c, index=ccys, columns=ccys)


def simulate_fx_returns(
    ccys: list[str] | None = None,
    n_days: int = 1000,
    seed: int | np.random.Generator = 0,
    garch: bool = False,
    regime_switching: bool = False,
    peg_jump_prob: float = 0.0,
    peg_jump_mean: float = -0.08,
    peg_jump_std: float = 0.03,
    return_state: bool = False,
):
    """Simulate daily FX log returns vs USD with block/regime structure.

    Parameters
    ----------
    ccys : list of str, optional
        Currencies (default: all G10 + EM + pegged).
    n_days : int
        Sample length.
    seed : int or Generator
        Deterministic seed (convention: always explicit).
    garch : bool
        Per-factor GARCH(1,1) vols (alpha=0.09, beta=0.89) instead of
        constant vols - produces volatility clustering.
    regime_switching : bool
        Two-state Markov calm/stress regime (P(c->s)=0.02, P(s->c)=0.10);
        stress doubles vols and switches to the stress correlation matrix.
    peg_jump_prob : float
        Daily probability that a pegged ccy in ``ccys`` jumps (devalues) by
        ``exp(N(peg_jump_mean, peg_jump_std))-1``; 0 disables jumps.
    return_state : bool
        Also return a DataFrame with the true conditional vols (columns as
        factors) and the regime indicator - used by calibration tests.

    Returns
    -------
    pandas.DataFrame (and optionally the state DataFrame)
        Columns are factor names ``FX:CCY``.
    """
    if ccys is None:
        ccys = G10 + EM + PEGGED
    unknown = [c for c in ccys if c not in ANNUAL_VOLS]
    if unknown:
        raise ValueError(f"no vol calibration for currencies {unknown}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    k = len(ccys)
    daily_vol = np.array([ANNUAL_VOLS[c] for c in ccys]) / np.sqrt(TRADING_DAYS_PER_YEAR)

    chol = {
        r: np.linalg.cholesky(default_correlation(ccys, r).to_numpy() + 1e-10 * np.eye(k))
        for r in ("calm", "stress")
    }
    # regime path
    state = np.zeros(n_days, dtype=int)  # 0 calm, 1 stress
    if regime_switching:
        u = rng.random(n_days)
        s = 0
        for t in range(n_days):
            s = (1 if u[t] < 0.02 else 0) if s == 0 else (0 if u[t] < 0.10 else 1)
            state[t] = s

    # GARCH(1,1) recursion targeting each ccy's unconditional daily variance
    a_g, b_g = 0.09, 0.89
    r = np.empty((n_days, k))
    sig = np.empty((n_days, k))
    var_t = daily_vol**2
    z_all = rng.standard_normal((n_days, k))
    for t in range(n_days):
        regime = "stress" if state[t] else "calm"
        mult = 2.0 if state[t] else 1.0
        sig_t = np.sqrt(var_t) * mult
        sig[t] = sig_t
        z = chol[regime] @ z_all[t]
        r[t] = sig_t * z
        if garch:
            uncond = daily_vol**2
            var_t = uncond * (1.0 - a_g - b_g) + a_g * (r[t] / mult) ** 2 + b_g * var_t
        # else var_t stays at unconditional

    cols = [fx_factor(c) for c in ccys]
    # pegged currencies: overwrite with near-zero band noise + rare jumps
    for j, c in enumerate(ccys):
        if c in PEGGED:
            band = ANNUAL_VOLS[c] / np.sqrt(TRADING_DAYS_PER_YEAR)
            r[:, j] = rng.standard_normal(n_days) * band
            if peg_jump_prob > 0:
                hits = rng.random(n_days) < peg_jump_prob
                jumps = peg_jump_mean + peg_jump_std * rng.standard_normal(n_days)
                r[:, j] += hits * jumps
    out = pd.DataFrame(r, columns=cols)
    if return_state:
        st = pd.DataFrame(sig, columns=cols)
        st["regime"] = state
        return out, st
    return out


def simulate_history(
    book: Book,
    market: Market,
    n_days: int = 1000,
    seed: int | np.random.Generator = 0,
    garch: bool = False,
    regime_switching: bool = False,
    peg_jump_prob: float = 0.0,
    ir_daily_vol: float = 8e-5,
    vol_daily_vol: float = 0.0025,
) -> pd.DataFrame:
    """Simulate a full factor history covering everything ``book`` needs.

    FX factors come from :func:`simulate_fx_returns`; IR factors are i.i.d.
    normal daily rate changes (default 0.8bp/day ~ 1.3%/yr in rate points);
    VOL factors are i.i.d. normal daily implied-vol changes (default 25bp of
    vol per day).  IR/VOL are independent of FX - adequate for a VaR factor
    history where FX dominates (documented assumption A7).
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    factors = book.factors(market)
    fx_ccys = [f.split(":")[1] for f in factors if f.startswith("FX:")]
    fx = simulate_fx_returns(fx_ccys, n_days, rng, garch=garch,
                             regime_switching=regime_switching,
                             peg_jump_prob=peg_jump_prob)
    data = {c: fx[c] for c in fx.columns}
    for f in factors:
        if f.startswith("IR:"):
            data[f] = pd.Series(rng.standard_normal(n_days) * ir_daily_vol)
        elif f.startswith("VOL:"):
            data[f] = pd.Series(rng.standard_normal(n_days) * vol_daily_vol)
    return pd.DataFrame(data)[factors]


def demo_market() -> Market:
    """Canned demo market snapshot (stylised mid-2020s levels).

    Spots are USD per 1 unit (EURUSD 1.08, USDJPY ~149, USDMXN ~18.5 ...);
    rates are cc zero rates ACT/365; vols are ATM annualised.
    """
    spot_usd = {
        "USD": 1.0, "EUR": 1.08, "JPY": 1.0 / 149.0, "GBP": 1.27,
        "CHF": 1.12, "AUD": 0.66, "NZD": 0.61, "CAD": 0.74, "SEK": 0.095,
        "NOK": 0.094, "MXN": 1.0 / 18.5, "BRL": 0.185, "TRY": 0.031,
        "ZAR": 0.055, "HKD": 1.0 / 7.8, "SAR": 1.0 / 3.75,
    }
    rates = {
        "USD": 0.053, "EUR": 0.039, "JPY": 0.001, "GBP": 0.052, "CHF": 0.017,
        "AUD": 0.043, "NZD": 0.055, "CAD": 0.050, "SEK": 0.040, "NOK": 0.045,
        "MXN": 0.110, "BRL": 0.105, "TRY": 0.450, "ZAR": 0.0825,
        "HKD": 0.050, "SAR": 0.055,
    }
    vols = {
        "EURUSD": 0.075, "USDJPY": 0.100, "GBPUSD": 0.085, "EURJPY": 0.095,
        "USDCHF": 0.070, "AUDUSD": 0.110, "USDMXN": 0.130, "USDBRL": 0.160,
        "USDTRY": 0.250, "USDHKD": 0.015,
    }
    return Market(spot_usd, rates, vols)


def demo_book(base: str = "USD") -> Book:
    """Canned demo book: G10 spots, one forward, one option, EM and peg legs.

    Long EUR and USD/JPY carry-style G10 spot risk, a 6m EURUSD forward
    (rate-leg risk), a 3m EURUSD call (delta/vega/gamma), a long-MXN EM
    position and a long-HKD peg position that makes the peg-blindness
    machinery observable end to end.
    """
    positions = [
        Spot("EURUSD", 25_000_000),          # long 25m EUR vs USD
        Spot("USDJPY", 15_000_000),          # long 15m USD vs JPY
        Spot("GBPUSD", -10_000_000),         # short 10m GBP vs USD
        Forward("EURUSD", 20_000_000, 0.5),  # 6m outright, ATM forward strike
        Option("EURUSD", 30_000_000, 1.10, 0.25, "call"),
        Spot("USDMXN", -20_000_000),         # short USD / long MXN (EM carry)
        Spot("USDHKD", -50_000_000),         # long HKD against USD (peg)
        Cash(base, 5_000_000),               # base-ccy cash: riskless
    ]
    return Book(positions, base=base)


def demo_em_book(base: str = "USD") -> Book:
    """EM-heavy long-local-currency book used for fat-tail demonstrations."""
    positions = [
        Spot("USDMXN", -25_000_000),
        Spot("USDBRL", -20_000_000),
        Spot("USDTRY", -10_000_000),
        Spot("USDZAR", -15_000_000),
    ]
    return Book(positions, base=base)
