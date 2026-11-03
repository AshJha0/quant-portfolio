"""Tiered FX transaction costs, a stress multiplier and breakeven (W6).

Plain idea: a flat 1 bp cost for every currency is too low for most of them.
Costs rise as a currency trades less and as markets get stressed.

Formal statement. The one-way cost of currency c on day t, in bp of notional,
is ``tier_bp(c) * (1 + stress_slope * pct_t)``. ``pct_t`` is the percentile
(0 to 1) of the currency's trailing 21-day realised volatility within its own
history up to t, so there is no look-ahead. Results are reported at cost
multipliers 0x, 1x and 2x, and as the breakeven multiplier
``mean(gross) / mean(cost at 1x)``.

Evidence class: ASSUMPTION. The tier levels are set by the maintainer within
the range of quoted half-spreads discussed by Menkhoff, Sarno, Schmeling and
Schrimpf (2012, JF, section I.B) and Gilmore and Hayashi (2011, AEJ: Macro),
with the order of the tiers following turnover shares in the BIS Triennial
Central Bank Survey (2022). They are not values copied from those sources.
Every study reports 0x, 1x, 2x and breakeven so a reader can apply their own.

Worked example: 1.0 of USD/MXN turnover on a calm day (pct 0) costs 5 bp;
on the most volatile day so far (pct 1, slope 1) it costs 10 bp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIERS_BP: dict[str, float] = {
    # G3 against USD
    "EUR": 1.0, "JPY": 1.0, "GBP": 1.0,
    # rest of G10
    "CHF": 2.0, "AUD": 2.0, "CAD": 2.0, "NZD": 2.5, "SEK": 2.5, "NOK": 2.5, "DKK": 2.5,
    # financial-centre pegs and liquid EM
    "HKD": 3.0, "SGD": 3.0, "MXN": 5.0, "PLN": 5.0, "CZK": 5.0, "HUF": 5.0, "ZAR": 5.0,
    "KRW": 5.0, "CNY": 5.0, "ILS": 5.0,
    # other EM
    "TRY": 7.5, "BRL": 7.5, "INR": 7.5, "THB": 7.5, "MYR": 7.5, "RON": 7.5, "BGN": 7.5,
    "HRK": 7.5, "IDR": 10.0, "PHP": 10.0, "RUB": 10.0,
}
DEFAULT_BP = 15.0  # anything not listed: legacy pegs and thin markets
MULTIPLIERS = (0.0, 1.0, 2.0)


def tier_bp(ccy: str) -> float:
    """One-way cost tier of ``ccy`` in bp."""
    return TIERS_BP.get(ccy.upper(), DEFAULT_BP)


def vol_percentile(log_returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Expanding percentile of trailing realised vol, using data up to t only."""
    vol = log_returns.rolling(window, min_periods=window).std()
    return vol.expanding(min_periods=window).rank(pct=True)


def daily_cost(turnover: pd.DataFrame, log_returns: pd.DataFrame | None = None,
               stress_slope: float = 1.0, multiplier: float = 1.0) -> pd.Series:
    """Cost, in return units, of per-currency one-way ``turnover`` on each date."""
    bp = np.array([tier_bp(c) for c in turnover.columns])
    unit = pd.DataFrame(np.broadcast_to(bp, turnover.shape).copy(),
                        index=turnover.index, columns=turnover.columns)
    if log_returns is not None:
        pct = vol_percentile(log_returns.reindex_like(turnover)).fillna(0.0)
        unit = unit * (1.0 + stress_slope * pct)
    return (turnover.abs() * unit * 1e-4 * multiplier).sum(axis=1)


def cost_grid(gross: pd.Series, cost_1x: pd.Series) -> dict[str, float | None]:
    """Mean net return at 0x, 1x and 2x costs, and the breakeven multiplier."""
    out: dict[str, float | None] = {f"mean_net_{m:g}x": float((gross - m * cost_1x).mean())
                                    for m in MULTIPLIERS}
    mc = float(cost_1x.mean())
    out["breakeven_multiplier"] = float(gross.mean()) / mc if mc > 0 else None
    return out
