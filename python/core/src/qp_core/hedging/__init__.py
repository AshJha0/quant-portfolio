"""Hedging policies with transaction costs (W26).

Modules
-------
policies   Time, Whalley-Wilmott band and Leland rules; Leland volatility
           and the band half-width.
simulate   Hedge a short option along paths; P&L = hedge error - cost;
           Ederington effectiveness with a standard error.
greeks     Gamma-vega neutralisation with two options, then delta.
fx         FX exposure hedging (none, full, threshold) with cost.
backtest   Out-of-sample window test on a daily index with a vol proxy.

References: Leland (1985), J. Finance 40(5); Whalley and Wilmott (1997),
Math. Finance 7(3); Ederington (1979), J. Finance 34(1).
"""

from __future__ import annotations

from qp_core.hedging.backtest import RuleStats, oos_hedge_test
from qp_core.hedging.fx import FxHedgePolicy, FxHedgeReport, run_fx_hedge
from qp_core.hedging.greeks import GreekHedge, OptionGreeks, gamma_vega_hedge, hedge_cost
from qp_core.hedging.policies import (
    HedgeRule,
    HedgeRuleSpec,
    leland_number,
    leland_sigma,
    whalley_wilmott_halfwidth,
)
from qp_core.hedging.simulate import Effectiveness, HedgeResult, effectiveness, gbm_paths, hedge_short_option

__all__ = [
    "Effectiveness",
    "FxHedgePolicy",
    "FxHedgeReport",
    "GreekHedge",
    "HedgeResult",
    "HedgeRule",
    "HedgeRuleSpec",
    "OptionGreeks",
    "RuleStats",
    "effectiveness",
    "gamma_vega_hedge",
    "gbm_paths",
    "hedge_cost",
    "hedge_short_option",
    "leland_number",
    "leland_sigma",
    "oos_hedge_test",
    "run_fx_hedge",
    "whalley_wilmott_halfwidth",
]
