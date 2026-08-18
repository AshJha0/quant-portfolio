"""Currency total returns, FX style signals and style-portfolio construction.

Conventions
-----------
* Spots are quoted CCYUSD (USD per 1 unit of foreign currency); a positive
  spot log return means the foreign currency appreciated vs USD.
* Deposit rates are continuously compounded, annualised; daily accrual uses
  ACT/252 (``dt = 1/252``).
* **Total return** of holding currency *i* financed in the base currency over
  one day is *exactly* decomposed as

  ``total_t = spot_t + carry_t``,
  ``spot_t = log(S_t / S_{t-1})``,
  ``carry_t = (i_{t-1} - i_base,{t-1}) * dt``

  — the carry leg accrues at the rates OBSERVED AT THE PREVIOUS CLOSE, so no
  same-day rate information leaks into the return (no lookahead).
* Style signals are formed from information available at *t*; style weights
  are applied to returns from *t+1* onward (the one-day implementation lag is
  enforced in :func:`style_returns`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252
DT: float = 1.0 / TRADING_DAYS


@dataclass
class ReturnDecomposition:
    """Exact additive decomposition ``total = spot + carry`` (daily log returns)."""

    total: pd.DataFrame
    spot: pd.DataFrame
    carry: pd.DataFrame


def spot_log_returns(spots: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns of spot levels.

    Parameters
    ----------
    spots : pd.DataFrame
        Positive spot levels, CCYUSD quoting.

    Returns
    -------
    pd.DataFrame
        ``log(S_t/S_{t-1})``; first row dropped.

    Raises
    ------
    ValueError
        If any spot level is non-positive.
    """
    if (spots <= 0).any().any():
        raise ValueError("spot levels must be strictly positive")
    return np.log(spots).diff().iloc[1:]


def carry_log_returns(
    rates: pd.DataFrame,
    currencies: list[str],
    base: str = "USD",
    dt: float = DT,
) -> pd.DataFrame:
    """Daily carry accrual ``(i_ccy - i_base) * dt`` using PREVIOUS-day rates.

    Parameters
    ----------
    rates : pd.DataFrame
        Annualised continuously compounded deposit rates; must contain the
        ``base`` column and every currency in ``currencies``.
    currencies : list of str
        Currencies to accrue carry for.
    base : str
        Funding currency (column of ``rates``).
    dt : float
        Year fraction per row (default 1/252).

    Returns
    -------
    pd.DataFrame
        Carry accrual aligned so that row *t* uses rates observed at *t-1*;
        the first row (undefined) is dropped.
    """
    if base not in rates.columns:
        raise ValueError(f"rates has no base column {base!r}")
    missing = [c for c in currencies if c not in rates.columns]
    if missing:
        raise ValueError(f"rates missing currencies: {missing}")
    diff = rates[currencies].sub(rates[base], axis=0)
    return (diff * dt).shift(1).iloc[1:]


def total_log_returns(
    spots: pd.DataFrame,
    rates: pd.DataFrame,
    base: str = "USD",
    dt: float = DT,
) -> ReturnDecomposition:
    """Build currency total-return series: spot return + lagged carry accrual.

    Returns
    -------
    ReturnDecomposition
        ``total``, ``spot`` and ``carry`` frames sharing one index; the
        identity ``total == spot + carry`` holds exactly (to float precision).
    """
    spot = spot_log_returns(spots)
    carry = carry_log_returns(rates, list(spots.columns), base=base, dt=dt)
    carry = carry.reindex(spot.index)
    if carry.isna().any().any():
        raise ValueError("rates index must cover the spots index")
    return ReturnDecomposition(total=spot + carry, spot=spot, carry=carry)


# ---------------------------------------------------------------------------
# Style signals
# ---------------------------------------------------------------------------


def carry_signal(
    rates: pd.DataFrame, currencies: list[str], base: str = "USD"
) -> pd.DataFrame:
    """CARRY signal: current deposit-rate differential ``i_ccy - i_base``.

    Higher differential = more attractive long.  Uses same-day rates: the
    signal at *t* is known at the close of *t* and is applied from *t+1* by
    :func:`style_returns`.
    """
    if base not in rates.columns:
        raise ValueError(f"rates has no base column {base!r}")
    return rates[currencies].sub(rates[base], axis=0)


def momentum_signal(
    spots: pd.DataFrame, lookback: int = 252, skip: int = 21
) -> pd.DataFrame:
    """MOMENTUM signal, 12-1 style: ``log(S_{t-skip} / S_{t-lookback})``.

    Skipping the most recent ``skip`` days avoids short-horizon reversal.
    Uses only data up to ``t - skip`` — strictly backward-looking.

    Raises
    ------
    ValueError
        If ``lookback <= skip`` or either is negative.
    """
    if skip < 0 or lookback <= skip:
        raise ValueError(f"need lookback > skip >= 0, got {lookback}, {skip}")
    return np.log(spots.shift(skip) / spots.shift(lookback))


def value_signal(spots: pd.DataFrame, ppp: pd.DataFrame) -> pd.DataFrame:
    """VALUE signal: PPP deviation ``log(PPP_t / S_t)``.

    Positive when the currency trades BELOW its PPP anchor (undervalued →
    expected to appreciate → attractive long); negative when overvalued.
    """
    if list(spots.columns) != list(ppp.columns):
        raise ValueError("spots and ppp must share identical columns")
    return np.log(ppp / spots)


# ---------------------------------------------------------------------------
# Dollar-neutral rank weights and style portfolios
# ---------------------------------------------------------------------------


def rank_weights(signal: pd.Series, gross: float = 2.0) -> pd.Series:
    """Cross-sectional dollar-neutral rank weights from one signal snapshot.

    Currencies are ranked by signal; ranks are demeaned (so weights sum to
    ZERO — dollar-neutral long-short) and scaled so gross leverage
    ``sum |w| = gross``.  Degenerate cross-sections (fewer than two finite
    signals, or all signals equal) return all-zero weights.

    Parameters
    ----------
    signal : pd.Series
        Signal per currency; NaNs are excluded (weight 0).
    gross : float
        Gross-leverage budget, must be >= 0.  ``gross=0`` returns zeros.

    Returns
    -------
    pd.Series
        Weights indexed like ``signal``; ``sum(w) == 0`` and
        ``sum(|w|) == gross`` in non-degenerate cases.
    """
    if gross < 0:
        raise ValueError(f"gross must be >= 0, got {gross}")
    w = pd.Series(0.0, index=signal.index)
    valid = signal.dropna()
    if gross == 0 or len(valid) < 2 or valid.nunique() == 1:
        return w
    ranks = valid.rank(method="average")
    centred = ranks - ranks.mean()
    w[valid.index] = centred * (gross / centred.abs().sum())
    return w


def signal_weights(signal_panel: pd.DataFrame, gross: float = 2.0) -> pd.DataFrame:
    """Apply :func:`rank_weights` to every row of a signal panel."""
    return signal_panel.apply(lambda row: rank_weights(row, gross=gross), axis=1)


def style_returns(
    total_returns: pd.DataFrame,
    signal_panel: pd.DataFrame,
    gross: float = 2.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Daily returns of a dollar-neutral style portfolio built from a signal.

    Weights are formed from the signal at *t* and applied to total returns at
    *t+1* (one-day implementation lag — no lookahead).  Days on which the
    signal is entirely missing (e.g. inside the momentum lookback window)
    carry zero weights.

    Parameters
    ----------
    total_returns : pd.DataFrame
        Currency total log returns (spot + carry).
    signal_panel : pd.DataFrame
        Signal values per day and currency, same columns as ``total_returns``.
    gross : float
        Gross-leverage budget passed to :func:`rank_weights`.

    Returns
    -------
    (pd.Series, pd.DataFrame)
        Style return series and the (lagged, i.e. as-applied) weight panel,
        both indexed like ``total_returns``.
    """
    if list(total_returns.columns) != list(signal_panel.columns):
        raise ValueError("total_returns and signal_panel must share columns")
    weights = signal_weights(signal_panel, gross=gross)
    applied = weights.shift(1).reindex(total_returns.index).fillna(0.0)
    ret = (applied * total_returns).sum(axis=1)
    return ret, applied


def shrunk_means(
    returns: pd.DataFrame, intensity: float | None = None
) -> tuple[pd.Series, float]:
    """Shrink sample mean returns towards the cross-sectional grand mean.

    James-Stein-flavoured estimator: with K assets and T observations,

    ``mu_shrunk = (1 - lam) * mu_sample + lam * grand_mean``,
    ``lam = min(1, ((K - 3) * avg_var / T) / sum((mu_i - grand_mean)^2))``.

    With K <= 3 the data-driven intensity is 0 (no shrinkage) — the JS
    dominance result needs K > 3.  Shrinking means is the single most
    effective defence against estimation error in MVO (means are ~an order of
    magnitude harder to estimate than covariances at these horizons).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily return panel.
    intensity : float, optional
        Override in [0, 1]; if None, use the data-driven formula above.

    Returns
    -------
    (pd.Series, float)
        Shrunk mean vector (per period) and the intensity used.
    """
    mu = returns.mean()
    grand = float(mu.mean())
    if intensity is None:
        k, t = returns.shape[1], len(returns)
        dispersion = float(((mu - grand) ** 2).sum())
        if k <= 3 or dispersion <= 0 or t < 2:
            lam = 0.0
        else:
            avg_var = float(returns.var(ddof=1).mean())
            lam = min(1.0, (k - 3) * avg_var / t / dispersion)
    else:
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {intensity}")
        lam = float(intensity)
    return (1 - lam) * mu + lam * grand, lam
