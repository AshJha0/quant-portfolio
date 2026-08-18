"""FX-native regime features with point-in-time (PIT) discipline.

Six features, each an FX desk staple for reading risk-on/risk-off:

``avg_vol``    Average rolling realised vol across G10 currencies vs USD,
               annualised — a synthetic VXY analog.
``carry_ret``  Rolling return of the rank-carry basket (long high-yielders,
               short low-yielders, dollar-neutral), spot + carry accrual.
``haven_rs``   Safe-haven relative strength: rolling return of (JPY+CHF)/2
               minus (AUD+NZD)/2.  Spikes up in risk-off.
``usd_corr``   Average pairwise rolling correlation of the risk-block
               (G10 carry + EM) currency-vs-USD returns — the
               'one-trade market' gauge that spikes in unwinds.
``em_g10``     EM-vs-G10 spread: rolling EM basket return minus rolling
               G10 basket return.
``usd_str``    Broad USD strength: minus the rolling average return of
               ALL currencies vs USD (the dollar factor).  Strongly
               positive in a 2008/2020-style USD funding squeeze;
               separates 'everything falls vs USD' from ordinary
               risk-off (havens rally) and from correlated rallies.
``fwd_ts``     Term-structure proxy of forward points: average annualised
               forward discount of the carry basket vs USD implied by
               covered interest parity (r_ccy − r_USD).

All features are standardised with an EXPANDING window: the z-score at
time t uses only data up to and including t.  This is enforced by a
mutation test (perturbing the future must not change the past).

Conventions: returns are daily log returns of currency vs USD; deposit
rates annualised decimals; vol annualised with sqrt(252).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.synthetic import EM, G10, G10_CARRY, HAVENS

TRADING_DAYS = 252

FEATURE_COLUMNS = (
    "avg_vol", "carry_ret", "haven_rs", "usd_corr", "em_g10", "usd_str",
    "fwd_ts",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Windows and basket sizes for the feature block.

    Windows are deliberately SHORT (1-2 weeks): FX risk regimes last
    days-to-weeks, and longer windows smear regime transitions into a
    spurious 'transitional' state that an HMM will happily invent (see
    docs/VALIDATION.md).

    Attributes
    ----------
    vol_window : int
        Rolling window (days) for realised vol.
    ret_window : int
        Rolling window (days) for basket returns / relative strength.
    corr_window : int
        Rolling window (days) for pairwise correlations.
    n_carry_long, n_carry_short : int
        Basket sizes for the rank-carry basket used in ``carry_ret``.
    std_min_periods : int
        Minimum observations before an expanding z-score is emitted.
    """

    vol_window: int = 8
    ret_window: int = 5
    corr_window: int = 12
    n_carry_long: int = 3
    n_carry_short: int = 3
    std_min_periods: int = 60


def carry_basket_weights(
    rates_row: pd.Series,
    currencies: list[str],
    n_long: int = 3,
    n_short: int = 3,
) -> pd.Series:
    """Rank-carry basket weights: long top-n yielders, short bottom-n.

    Weights are equal-sized, dollar-neutral (sum to zero): +1/n_long on
    each long leg, -1/n_short on each short leg.  Ranking is by deposit
    rate differential vs USD (equivalently by deposit rate).

    Parameters
    ----------
    rates_row : Series
        Annualised deposit rates indexed by currency (may include USD;
        USD is ignored for ranking).
    currencies : list of str
        Universe to rank (currency-vs-USD legs).
    n_long, n_short : int

    Returns
    -------
    Series of weights indexed by ``currencies``, summing to zero.
    """
    if n_long <= 0 or n_short <= 0:
        raise ValueError("basket sizes must be positive")
    if n_long + n_short > len(currencies):
        raise ValueError("basket sizes exceed universe")
    r = rates_row.reindex(currencies)
    order = r.sort_values(kind="stable").index
    w = pd.Series(0.0, index=currencies)
    w[order[-n_long:]] = 1.0 / n_long
    w[order[:n_short]] = -1.0 / n_short
    return w


def realised_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling annualised realised vol per column (population ddof=0)."""
    if window < 2:
        raise ValueError("window must be >= 2")
    return returns.rolling(window).std(ddof=0) * np.sqrt(TRADING_DAYS)


def avg_pairwise_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    """Average pairwise rolling correlation across columns.

    NaN pair-correlations (e.g. a pegged, zero-vol currency) are ignored;
    if every pair is NaN in a window the value is 0.0.

    Returns
    -------
    Series aligned to ``returns.index`` (first ``window-1`` entries NaN).
    """
    if window < 3:
        raise ValueError("window must be >= 3")
    cols = list(returns.columns)
    p = len(cols)
    if p < 2:
        raise ValueError("need at least 2 columns")
    pair_corrs = []
    for i in range(p):
        for j in range(i + 1, p):
            pair_corrs.append(
                returns[cols[i]].rolling(window).corr(returns[cols[j]])
            )
    mat = pd.concat(pair_corrs, axis=1)
    # inf can appear from zero-variance windows in pandas; treat as NaN
    mat = mat.replace([np.inf, -np.inf], np.nan)
    out = mat.mean(axis=1, skipna=True)
    all_nan = mat.isna().all(axis=1)
    # windows fully inside the sample but with no valid pair -> 0.0
    valid = np.arange(len(out)) >= window - 1
    out[all_nan & valid] = 0.0
    return out


def expanding_standardize(
    df: pd.DataFrame, min_periods: int = 60
) -> pd.DataFrame:
    """Expanding-window z-score: PIT-safe standardisation.

    z_t = (x_t - mean(x_{1..t})) / std(x_{1..t}, ddof=0).  Uses only data
    up to and including t; rows with fewer than ``min_periods``
    observations are NaN.  Zero-variance columns give z = 0.

    This function is the subject of a mutation test: perturbing x_{t+1:}
    must leave z_{1..t} unchanged.
    """
    if min_periods < 2:
        raise ValueError("min_periods must be >= 2")
    mean = df.expanding(min_periods=min_periods).mean()
    std = df.expanding(min_periods=min_periods).std(ddof=0)
    z = (df - mean) / std.replace(0.0, np.nan)
    z = z.where(~(std == 0.0), 0.0)
    z[mean.isna()] = np.nan
    return z


def build_features(
    returns: pd.DataFrame,
    deposit_rates: pd.DataFrame,
    config: FeatureConfig | None = None,
    standardize: bool = True,
) -> pd.DataFrame:
    """Build the six-feature FX regime block (PIT-safe).

    Parameters
    ----------
    returns : DataFrame (T x p)
        Daily log returns of currency vs USD.  Columns must include the
        haven pair (JPY, CHF) and at least two risk-block currencies.
    deposit_rates : DataFrame
        Annualised deposit rates; must include a ``USD`` column and the
        currencies in ``returns``.
    config : FeatureConfig, optional
    standardize : bool
        If True (default), expanding z-scores are returned and warm-up
        rows dropped; if False, raw features (with rolling warm-up NaNs
        dropped) are returned.

    Returns
    -------
    DataFrame with columns ``FEATURE_COLUMNS``, warm-up rows dropped.

    Raises
    ------
    ValueError
        If the series is too short for the configured windows.
    """
    cfg = config or FeatureConfig()
    warmup = max(cfg.vol_window, cfg.ret_window, cfg.corr_window)
    min_len = warmup + (cfg.std_min_periods if standardize else 1) + 1
    if len(returns) < min_len:
        raise ValueError(
            f"series too short: {len(returns)} rows < required {min_len}"
        )
    if "USD" not in deposit_rates.columns:
        raise ValueError("deposit_rates must include a USD column")
    for c in ("JPY", "CHF"):
        if c not in returns.columns:
            raise ValueError(f"returns must include haven currency {c}")

    cols = list(returns.columns)
    g10_cols = [c for c in cols if c in G10]
    risk_cols = [c for c in cols if c in G10_CARRY or c in EM]
    em_cols = [c for c in cols if c in EM]
    if len(risk_cols) < 2:
        raise ValueError("need at least 2 risk-block currencies")

    # 1. synthetic VXY analog: average G10 realised vol
    avg_vol = realised_vol(returns[g10_cols], cfg.vol_window).mean(axis=1)

    # 2. carry basket rolling return (spot + accrual), PIT weights from
    #    the PREVIOUS day's rates
    rates = deposit_rates.reindex(returns.index).ffill()
    diffs = rates[cols].sub(rates["USD"], axis=0)
    rank = diffs.rank(axis=1, method="first")
    n_l, n_s = cfg.n_carry_long, cfg.n_carry_short
    if n_l + n_s > len(cols):
        raise ValueError("carry basket sizes exceed universe")
    w_long = (rank > len(cols) - n_l).astype(float) / n_l
    w_short = (rank <= n_s).astype(float) / n_s
    w = (w_long - w_short).shift(1)  # weights known at t-1 applied to day t
    carry_daily = (w * (returns + diffs.shift(1) / TRADING_DAYS)).sum(
        axis=1, skipna=False
    )
    carry_ret = carry_daily.rolling(cfg.ret_window).sum()

    # 3. safe-haven relative strength
    haven_leg = returns[list(HAVENS)].mean(axis=1)
    risk_proxy = [c for c in ("AUD", "NZD") if c in cols] or risk_cols[:2]
    haven_rs = (haven_leg - returns[risk_proxy].mean(axis=1)).rolling(
        cfg.ret_window
    ).sum()

    # 4. cross-sectional correlation of risk-block USD pairs
    usd_corr = avg_pairwise_correlation(returns[risk_cols], cfg.corr_window)

    # 5. EM vs G10 spread
    if em_cols:
        em_g10 = (
            returns[em_cols].mean(axis=1) - returns[g10_cols].mean(axis=1)
        ).rolling(cfg.ret_window).sum()
    else:
        em_g10 = pd.Series(0.0, index=returns.index)

    # 6. broad USD strength (dollar factor): -average currency return
    usd_str = (-returns.mean(axis=1)).rolling(cfg.ret_window).sum()

    # 7. forward-point term-structure proxy: average annualised forward
    #    discount of the current carry-basket longs (CIP: fwd points ≈
    #    spot * (r_USD - r_ccy) * tau; we report r_ccy - r_USD)
    long_mask = w_long.shift(1)
    fwd_ts = (long_mask * diffs.shift(1)).sum(axis=1) / max(n_l, 1)

    feats = pd.DataFrame(
        {
            "avg_vol": avg_vol,
            "carry_ret": carry_ret,
            "haven_rs": haven_rs,
            "usd_corr": usd_corr,
            "em_g10": em_g10,
            "usd_str": usd_str,
            "fwd_ts": fwd_ts,
        }
    )
    feats = feats.iloc[warmup:]
    if standardize:
        feats = expanding_standardize(feats, cfg.std_min_periods).dropna()
    else:
        feats = feats.dropna()
    return feats
