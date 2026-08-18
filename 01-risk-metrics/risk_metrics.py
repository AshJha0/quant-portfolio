"""
Risk metrics library.

Every function takes a pandas Series of *simple daily returns* unless
stated otherwise. Log vs simple returns matters: log returns aggregate
additively over time, simple returns aggregate across a portfolio.
For single-asset daily risk measurement the difference is small, but
the choice is documented here and in the README.

Conventions used throughout:
- 252 trading days per year.
- VaR is reported as a POSITIVE number representing a loss
  (i.e. VaR 95% = 2.1% means "on the worst 5% of days you lose
  at least 2.1%").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Return construction
# ----------------------------------------------------------------------
def simple_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


# ----------------------------------------------------------------------
# Volatility
# ----------------------------------------------------------------------
def annualised_volatility(returns: pd.Series) -> float:
    """Unconditional volatility, scaled by sqrt(252).

    Assumption: returns are i.i.d., so variance scales linearly with
    time. Real returns show volatility clustering, which makes this
    an approximation -- see rolling_volatility and ewma_volatility.
    """
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Rolling annualised volatility. window=21 ~ one trading month."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics-style EWMA volatility (lambda = 0.94 for daily data).

    Reacts faster to regime changes than an equal-weighted window
    because recent observations get exponentially more weight.
    """
    var = returns.ewm(alpha=1 - lam, adjust=False).var(bias=True)
    return np.sqrt(var * TRADING_DAYS)


# ----------------------------------------------------------------------
# Value at Risk
# ----------------------------------------------------------------------
def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (empirical) VaR.

    No distributional assumption -- just the empirical quantile.
    Weakness: assumes the past sample window is representative of the
    future; a calm sample will understate risk.
    """
    return -np.percentile(returns, 100 * (1 - confidence))


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian (variance-covariance) VaR.

    Assumes returns ~ Normal(mu, sigma). Daily equity returns have
    excess kurtosis, so this systematically understates tail risk at
    high confidence levels. Included precisely to demonstrate that gap.
    """
    mu, sigma = returns.mean(), returns.std(ddof=1)
    return -(mu + sigma * stats.norm.ppf(1 - confidence))


def var_cornish_fisher(returns: pd.Series, confidence: float = 0.95) -> float:
    """Modified VaR using a Cornish-Fisher expansion.

    Adjusts the Gaussian quantile for observed skewness and kurtosis.
    A middle ground: parametric, but acknowledges non-normality.
    """
    mu, sigma = returns.mean(), returns.std(ddof=1)
    s = stats.skew(returns)
    k = stats.kurtosis(returns)  # excess kurtosis
    z = stats.norm.ppf(1 - confidence)
    z_cf = (z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36)
    return -(mu + sigma * z_cf)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Expected Shortfall (CVaR).

    Average loss GIVEN that the loss exceeds VaR. Unlike VaR it is a
    coherent risk measure (sub-additive) and describes how bad the
    tail is, not just where it starts.
    """
    var = var_historical(returns, confidence)
    tail = returns[returns <= -var]
    return -tail.mean()


# ----------------------------------------------------------------------
# Other risk statistics
# ----------------------------------------------------------------------
def max_drawdown(prices: pd.Series) -> dict:
    """Maximum peak-to-trough decline, with dates."""
    running_max = prices.cummax()
    drawdown = prices / running_max - 1
    trough_date = drawdown.idxmin()
    peak_date = prices.loc[:trough_date].idxmax()
    return {
        "max_drawdown": drawdown.min(),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "drawdown_series": drawdown,
    }


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.03) -> float:
    """Annualised Sharpe ratio.

    Assumptions: constant risk-free rate; sqrt(252) scaling (valid only
    if returns are i.i.d. -- autocorrelation biases this upward).
    """
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    return excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.03) -> float:
    """Like Sharpe, but penalises only downside deviation."""
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    downside = excess[excess < 0]
    return excess.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS)


def normality_report(returns: pd.Series) -> dict:
    """Skewness, kurtosis and Jarque-Bera test.

    If JB rejects normality (it almost always does for daily returns),
    Gaussian VaR should be treated with suspicion.
    """
    jb_stat, jb_p = stats.jarque_bera(returns)
    return {
        "skewness": stats.skew(returns),
        "excess_kurtosis": stats.kurtosis(returns),
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_p,
        "normality_rejected_at_5pct": jb_p < 0.05,
    }
