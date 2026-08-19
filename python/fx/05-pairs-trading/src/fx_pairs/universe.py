"""FX universe construction: USD legs, cross rates, candidate pairs, correlation screen.

Conventions (see CONVENTIONS.md):

* Pairs are quoted BASE/QUOTE, e.g. ``EURUSD`` = USD per 1 EUR.
* Internally every currency is carried as a **USD leg**: the value of one unit
  of the currency in USD (i.e. the ``CCYUSD`` rate).  ``USD`` itself has a
  constant leg of 1.  Market-convention pairs are then either the leg itself
  (``EURUSD``) or its reciprocal (``USDJPY = 1 / JPYUSD``).
* Cross rates are **exact ratios** of USD legs:
  ``BASEQUOTE = (BASEUSD) / (QUOTEUSD)``.  No-arbitrage (triangular
  consistency) therefore holds identically in this representation — a fact the
  cointegration machinery must recognise as a *degenerate* spread, not a
  tradable one (see :mod:`fx_pairs.cointegration`).
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from ._validation import require_finite

__all__ = [
    "G10_CURRENCIES",
    "EM_CURRENCIES",
    "USD_BASE_CURRENCIES",
    "DEFAULT_PIP_SPREADS",
    "market_pair",
    "pip_size",
    "pip_spread",
    "make_cross",
    "market_price_from_legs",
    "triangular_spread",
    "enumerate_candidate_pairs",
    "correlation_screen",
]

#: G10 currencies (USD included as numeraire).
G10_CURRENCIES: tuple[str, ...] = (
    "USD", "EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK",
)

#: Emerging-market subset used in this project.
EM_CURRENCIES: tuple[str, ...] = ("MXN", "ZAR", "TRY")

#: Currencies conventionally quoted as BASE against USD (``EURUSD`` style).
#: Everything else is quoted ``USDXXX``.
USD_BASE_CURRENCIES: frozenset[str] = frozenset({"EUR", "GBP", "AUD", "NZD"})

#: Indicative full bid-ask spreads in pips (majors tight, EM wide).
#: Units: pips of the quoted pair (pip = 0.0001, or 0.01 for JPY-quoted pairs).
DEFAULT_PIP_SPREADS: dict[str, float] = {
    "EURUSD": 0.5, "GBPUSD": 0.8, "AUDUSD": 0.7, "NZDUSD": 1.0,
    "USDJPY": 0.5, "USDCHF": 1.0, "USDCAD": 1.2, "USDSEK": 15.0,
    "USDNOK": 15.0, "USDMXN": 30.0, "USDZAR": 60.0, "USDTRY": 150.0,
    # common crosses
    "EURGBP": 0.8, "EURCHF": 1.2, "EURJPY": 0.8, "AUDNZD": 2.0,
    "NOKSEK": 8.0, "AUDCAD": 1.8, "GBPJPY": 1.5, "CADJPY": 1.5,
}


def market_pair(ccy: str) -> str:
    """Market-convention USD pair name for a currency.

    Parameters
    ----------
    ccy : str
        ISO currency code, e.g. ``"EUR"`` or ``"JPY"``.

    Returns
    -------
    str
        ``"EURUSD"``-style name if the currency is conventionally the base
        (EUR, GBP, AUD, NZD), else ``"USDJPY"``-style.
    """
    ccy = ccy.upper()
    if ccy == "USD":
        raise ValueError("USD has no USD pair against itself")
    return f"{ccy}USD" if ccy in USD_BASE_CURRENCIES else f"USD{ccy}"


def pip_size(pair: str) -> float:
    """Pip size of a pair: 0.01 for JPY-quoted pairs, else 0.0001."""
    if len(pair) != 6:
        raise ValueError(f"pair must be a 6-letter code, got {pair!r}")
    return 0.01 if pair.upper().endswith("JPY") else 1e-4


def pip_spread(pair: str, overrides: dict[str, float] | None = None) -> float:
    """Full bid-ask spread in pips for a pair (falls back to a wide default).

    Parameters
    ----------
    pair : str
        6-letter pair code.
    overrides : dict, optional
        Pair -> pips mapping taking precedence over :data:`DEFAULT_PIP_SPREADS`.
    """
    table = dict(DEFAULT_PIP_SPREADS)
    if overrides:
        table.update({k.upper(): v for k, v in overrides.items()})
    return table.get(pair.upper(), 5.0)


def _leg(legs: pd.DataFrame, ccy: str) -> pd.Series:
    """USD leg (CCYUSD) for ``ccy``; USD is a constant 1 if not a column."""
    ccy = ccy.upper()
    if ccy in legs.columns:
        return legs[ccy]
    if ccy == "USD":
        return pd.Series(1.0, index=legs.index, name="USD")
    raise KeyError(f"currency {ccy!r} not in legs panel {list(legs.columns)}")


def make_cross(legs: pd.DataFrame, base: str, quote: str) -> pd.Series:
    """Cross rate BASE/QUOTE from USD legs — an exact no-arbitrage ratio.

    ``BASEQUOTE = BASEUSD / QUOTEUSD`` (units: QUOTE per 1 BASE).

    Parameters
    ----------
    legs : pandas.DataFrame
        Columns are currency codes; values are USD per 1 unit (CCYUSD).
    base, quote : str
        ISO currency codes.

    Returns
    -------
    pandas.Series
        Cross-rate series named ``f"{base}{quote}"``.
    """
    base, quote = base.upper(), quote.upper()
    if base == quote:
        raise ValueError("base and quote must differ")
    out = _leg(legs, base) / _leg(legs, quote)
    out.name = f"{base}{quote}"
    return out


def market_price_from_legs(legs: pd.DataFrame, pair: str) -> pd.Series:
    """Price series for a 6-letter pair code built from USD legs."""
    if len(pair) != 6:
        raise ValueError(f"pair must be a 6-letter code, got {pair!r}")
    return make_cross(legs, pair[:3], pair[3:])


def triangular_spread(
    legs: pd.DataFrame, base: str, mid: str, quote: str
) -> pd.Series:
    """Log triangular spread ``log(BASE/MID) + log(MID/QUOTE) - log(BASE/QUOTE)``.

    Under no-arbitrage this is identically zero (crosses are exact ratios of
    USD legs).  Used as the null case for degenerate-spread detection: the
    'spread' has (numerically) zero variance and must not be reported as a
    tradable cointegration.
    """
    a = np.log(make_cross(legs, base, mid))
    b = np.log(make_cross(legs, mid, quote))
    c = np.log(make_cross(legs, base, quote))
    out = a + b - c
    out.name = f"tri({base},{mid},{quote})"
    return out


def enumerate_candidate_pairs(names: list[str] | tuple[str, ...]) -> list[tuple[str, str]]:
    """All unordered pairs of distinct instrument names, preserving input order."""
    uniq = list(dict.fromkeys(names))
    return list(itertools.combinations(uniq, 2))


def correlation_screen(
    prices: pd.DataFrame,
    min_abs_corr: float = 0.5,
    vol_tol: float = 1e-10,
) -> pd.DataFrame:
    """Screen candidate pairs of currency pairs on log-return correlation.

    Pegged / zero-volatility instruments (e.g. a hard peg) are dropped with a
    ``UserWarning`` — a z-score is undefined on a zero-variance spread and a
    peg offers no mean reversion to trade, only event risk.

    Parameters
    ----------
    prices : pandas.DataFrame
        Columns are pair names, values are spot rates.
    min_abs_corr : float
        Keep candidate pairs with ``|corr| >= min_abs_corr`` (daily log
        returns).
    vol_tol : float
        Columns with log-return standard deviation below this are treated as
        pegged and excluded.

    Returns
    -------
    pandas.DataFrame
        Columns ``["pair_1", "pair_2", "corr"]`` sorted by ``|corr|``
        descending.
    """
    if prices.shape[1] < 2:
        raise ValueError("need at least two instruments to screen")
    # A NaN threshold makes every `abs(rho) >= min_abs_corr` comparison False
    # and returns an empty screen -- indistinguishable from "no candidates".
    require_finite(min_abs_corr=min_abs_corr, vol_tol=vol_tol)
    rets = np.log(prices).diff().dropna(how="all")
    stds = rets.std()
    pegged = [c for c in prices.columns if not np.isfinite(stds[c]) or stds[c] < vol_tol]
    if pegged:
        warnings.warn(
            f"dropping pegged/zero-volatility instruments from screen: {pegged}",
            UserWarning,
            stacklevel=2,
        )
    keep = [c for c in prices.columns if c not in pegged]
    corr = rets[keep].corr()
    rows = []
    for a, b in enumerate_candidate_pairs(keep):
        rho = float(corr.loc[a, b])
        if abs(rho) >= min_abs_corr:
            rows.append((a, b, rho))
    out = pd.DataFrame(rows, columns=["pair_1", "pair_2", "corr"])
    if len(out):
        out = out.reindex(out["corr"].abs().sort_values(ascending=False).index)
        out = out.reset_index(drop=True)
    return out
