"""Seeded synthetic FX market generator with risk-on/off factor structure.

Conventions (see repo CONVENTIONS.md):

* All pairs are quoted BASE/QUOTE with the currency as base and USD as quote,
  i.e. the panel column ``AUD`` holds the AUDUSD spot = USD per 1 AUD.  A rise
  in the series means the currency APPRECIATES against USD.
* Deposit rates are continuously compounded, annualised, ACT/252 accrual
  (one business day = 1/252 of a year).
* Every generator takes an explicit integer ``seed`` and is fully
  deterministic given that seed.

Structural features (all first-class, all unit-tested):

1. **Risk-on/off single factor** — every currency's spot return loads on one
   global risk factor ``f_t``.  AUD/NZD and the EM block load positively
   (risk-on currencies), JPY/CHF load negatively (safe havens).
2. **Persistent rate differentials** — deposit rates follow slow
   Ornstein-Uhlenbeck processes around currency-specific long-run means
   (JPY/CHF near zero, EM high), so the carry cross-section is persistent.
3. **Carry-crash mechanism** — on rare crash days (correlated with a large
   negative risk-factor draw) high-yield currencies suffer an *additional*
   depreciation proportional to their rate differential.  This builds the
   classic negative skew of carry into the data generating process while
   leaving carry profitable on average (empirical UIP failure).
4. **PPP anchor** — a slow-moving purchasing-power-parity fair-value series
   per currency; spots weakly mean-revert towards it (multi-year half-life),
   giving the value signal genuine but slow content.
5. **Persistent trends** — a slow AR(1) drift per currency gives the 12-1
   momentum style genuine content without affecting the factor structure.
6. Optional **pegged currency** (``HKD``-style: near-zero spot vol, fixed
   rate) to exercise zero-vol edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252

#: Universe: G10 ex-USD plus a liquid EM block.  Values are
#: (long-run deposit rate, risk-factor loading, idio daily vol, initial spot).
_CCY_PARAMS: dict[str, tuple[float, float, float, float]] = {
    "EUR": (0.015, 0.30, 0.0040, 1.10),
    "JPY": (0.001, -0.80, 0.0045, 0.0091),
    "GBP": (0.020, 0.40, 0.0045, 1.30),
    "AUD": (0.035, 1.20, 0.0050, 0.72),
    "NZD": (0.035, 1.10, 0.0055, 0.66),
    "CHF": (0.003, -0.60, 0.0045, 1.05),
    "CAD": (0.022, 0.70, 0.0040, 0.76),
    "SEK": (0.012, 0.60, 0.0050, 0.105),
    "NOK": (0.028, 0.80, 0.0055, 0.115),
    "MXN": (0.080, 1.30, 0.0065, 0.055),
    "BRL": (0.100, 1.40, 0.0080, 0.20),
    "TRY": (0.150, 1.50, 0.0090, 0.12),
}

USD_RATE_MEAN: float = 0.025

G10: list[str] = ["EUR", "JPY", "GBP", "AUD", "NZD", "CHF", "CAD", "SEK", "NOK"]
EM: list[str] = ["MXN", "BRL", "TRY"]


@dataclass
class FXPanel:
    """Container for a synthetic FX market panel.

    Attributes
    ----------
    spots : pd.DataFrame
        Spot levels, columns = currencies, quoted CCYUSD (USD per 1 CCY).
    rates : pd.DataFrame
        Continuously compounded annualised deposit rates; includes a ``USD``
        column in addition to every currency column of ``spots``.
    ppp : pd.DataFrame
        PPP fair-value anchor levels, same shape/quoting as ``spots``.
    crash_days : pd.Series
        Boolean indicator of carry-crash days (diagnostic; not used by the
        library code paths, only by tests and docs).
    risk_factor : pd.Series
        The realised global risk factor (diagnostic).
    """

    spots: pd.DataFrame
    rates: pd.DataFrame
    ppp: pd.DataFrame
    crash_days: pd.Series = field(repr=False, default=None)  # type: ignore[assignment]
    risk_factor: pd.Series = field(repr=False, default=None)  # type: ignore[assignment]


def make_panel(
    seed: int = 0,
    n_days: int = 2520,
    currencies: list[str] | None = None,
    include_peg: bool = False,
    crash_prob: float = 1.0 / 188.0,
    crash_carry_kappa: float = 0.4,
    start: str = "2015-01-01",
) -> FXPanel:
    """Generate a seeded synthetic FX panel with factor and crash structure.

    Parameters
    ----------
    seed : int
        Seed for ``numpy.random.default_rng``; identical seeds give identical
        panels.
    n_days : int
        Number of business days to simulate.
    currencies : list of str, optional
        Subset of the built-in universe (default: G10 ex-USD + MXN/BRL/TRY).
    include_peg : bool
        If True, append a pegged currency ``PEG`` with near-zero spot vol and
        a deposit rate pinned to USD (zero differential) — exercises zero-vol
        edge cases downstream.
    crash_prob : float
        Daily probability of a carry-crash event (default ~1.3 per year).
    crash_carry_kappa : float
        On a crash day each currency loses an extra
        ``kappa * max(rate - usd_rate, 0) * severity`` in log-spot terms,
        where severity ~ |N(1, 0.3)|.  This ties crash losses to the carry of
        the currency — the mechanism behind carry's negative skew.
    start : str
        First business date of the index.

    Returns
    -------
    FXPanel
        Spots, rates (incl. USD), PPP anchors and diagnostics.

    Raises
    ------
    ValueError
        If ``n_days < 2`` or an unknown currency is requested.
    """
    if n_days < 2:
        raise ValueError(f"n_days must be >= 2, got {n_days}")
    ccys = list(_CCY_PARAMS) if currencies is None else list(currencies)
    unknown = [c for c in ccys if c not in _CCY_PARAMS]
    if unknown:
        raise ValueError(f"unknown currencies: {unknown}")

    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    n = len(ccys)

    rate_mean = np.array([_CCY_PARAMS[c][0] for c in ccys])
    beta = np.array([_CCY_PARAMS[c][1] for c in ccys])
    idio_vol = np.array([_CCY_PARAMS[c][2] for c in ccys])
    s0 = np.array([_CCY_PARAMS[c][3] for c in ccys])

    # --- deposit rates: slow OU around persistent means, floored at 0 -----
    theta, rate_vol = 0.01, 0.0002
    rates = np.empty((n_days, n + 1))
    r_now = np.concatenate([rate_mean, [USD_RATE_MEAN]])
    mu_r = r_now.copy()
    for t in range(n_days):
        rates[t] = r_now
        r_now = np.maximum(
            r_now + theta * (mu_r - r_now) + rate_vol * rng.standard_normal(n + 1), 0.0
        )

    # --- global risk factor with crash mixture ---------------------------
    f_vol = 0.004
    f = f_vol * rng.standard_normal(n_days)
    crash = rng.random(n_days) < crash_prob
    crash[0] = False  # keep the first return day clean
    f[crash] -= np.abs(rng.normal(0.02, 0.006, crash.sum()))

    # --- PPP anchor: slow random walk from initial level ------------------
    ppp_vol = 0.005 / np.sqrt(TRADING_DAYS)
    log_ppp = np.log(s0)[None, :] + np.cumsum(
        ppp_vol * rng.standard_normal((n_days, n)), axis=0
    )

    # --- persistent per-currency trends (fuel for the momentum style) -----
    phi, trend_sd = 0.997, 0.0005
    trend = np.zeros((n_days, n))
    eps_m = trend_sd * np.sqrt(1 - phi**2) * rng.standard_normal((n_days, n))
    for t in range(1, n_days):
        trend[t] = phi * trend[t - 1] + eps_m[t]

    # --- spots: factor + trend + idio + weak PPP reversion + carry crash --
    kappa_ppp = np.log(2.0) / (2.0 * TRADING_DAYS)  # 2-year half-life
    log_s = np.empty((n_days, n))
    log_s[0] = np.log(s0)
    severity = np.abs(rng.normal(1.0, 0.3, n_days))
    idio = idio_vol[None, :] * rng.standard_normal((n_days, n))
    for t in range(1, n_days):
        diff_prev = np.maximum(rates[t - 1, :n] - rates[t - 1, n], 0.0)
        crash_hit = (
            -crash_carry_kappa * diff_prev * severity[t] if crash[t] else 0.0
        )
        r_t = (
            beta * f[t]
            + trend[t]
            + idio[t]
            - kappa_ppp * (log_s[t - 1] - log_ppp[t - 1])
            + crash_hit
        )
        log_s[t] = log_s[t - 1] + r_t

    spots = pd.DataFrame(np.exp(log_s), index=idx, columns=ccys)
    ppp = pd.DataFrame(np.exp(log_ppp), index=idx, columns=ccys)
    rates_df = pd.DataFrame(rates, index=idx, columns=ccys + ["USD"])

    if include_peg:
        spots["PEG"] = 0.1282 * np.exp(
            1e-6 * np.cumsum(rng.standard_normal(n_days))
        )
        ppp["PEG"] = 0.1282
        rates_df.insert(len(ccys), "PEG", rates_df["USD"].to_numpy())

    return FXPanel(
        spots=spots,
        rates=rates_df,
        ppp=ppp,
        crash_days=pd.Series(crash, index=idx, name="crash"),
        risk_factor=pd.Series(f, index=idx, name="risk_factor"),
    )


@dataclass
class EquityFXMarket:
    """Synthetic international equity portfolio + FX returns for hedging demos.

    Attributes
    ----------
    unhedged_returns : pd.Series
        Daily log returns of the international equity portfolio measured in
        the base currency (USD), currency exposure UNHEDGED.
    local_returns : pd.DataFrame
        Local-currency equity log returns per market.
    fx_returns : pd.DataFrame
        Daily log returns of each exposure currency vs USD (CCYUSD).
    exposures : pd.Series
        Portfolio weight held in each non-USD currency (sums to <= 1; the
        remainder is the USD home market).
    """

    unhedged_returns: pd.Series
    local_returns: pd.DataFrame
    fx_returns: pd.DataFrame
    exposures: pd.Series


def make_equity_portfolio(seed: int = 0, n_days: int = 2520) -> EquityFXMarket:
    """Generate a synthetic international equity portfolio for the hedging demo.

    Markets: US (home), EUR, JPY, GBP, AUD, CHF.  Local equity returns load on
    a global equity factor; FX returns load on the same factor with the
    risk-on/off sign pattern (AUD positive, JPY/CHF negative), so safe-haven
    currencies naturally hedge equity risk and the optimal hedge ratio for
    them is below 1.

    Returns
    -------
    EquityFXMarket
        See class docstring.  All series share one business-day index.
    """
    if n_days < 2:
        raise ValueError(f"n_days must be >= 2, got {n_days}")
    rng = np.random.default_rng(seed + 777)
    idx = pd.bdate_range("2015-01-01", periods=n_days)

    markets = ["US", "EUR", "JPY", "GBP", "AUD", "CHF"]
    weights = pd.Series(
        [0.40, 0.20, 0.12, 0.10, 0.08, 0.10], index=markets, name="weight"
    )
    eq_beta = np.array([1.0, 0.95, 0.85, 0.90, 1.05, 0.80])
    fx_ccys = ["EUR", "JPY", "GBP", "AUD", "CHF"]
    fx_beta = pd.Series([0.01, -0.06, 0.02, 0.05, -0.04], index=fx_ccys)

    g = 0.009 * rng.standard_normal(n_days) + 0.0002  # global equity factor
    local = pd.DataFrame(
        eq_beta[None, :] * g[:, None]
        + 0.006 * rng.standard_normal((n_days, len(markets))),
        index=idx,
        columns=markets,
    )
    fx = pd.DataFrame(
        fx_beta.to_numpy()[None, :] * g[:, None]
        + 0.0045 * rng.standard_normal((n_days, len(fx_ccys))),
        index=idx,
        columns=fx_ccys,
    )

    exposures = weights[fx_ccys].rename("exposure")
    unhedged = (local * weights).sum(axis=1) + (fx * exposures).sum(axis=1)
    unhedged.name = "unhedged"
    return EquityFXMarket(
        unhedged_returns=unhedged,
        local_returns=local,
        fx_returns=fx,
        exposures=exposures,
    )
