"""fx_rates — FX-linked fixed income: multi-currency curves, CIP forward
curves, cross-currency basis, FX forward/swap and cross-currency swap
pricing and risk.

Pipeline: yield curve construction (two currencies) -> bootstrapping ->
FX forward curve via CIP -> cross-currency basis -> FX swap / forward
pricing & risk -> DV01/KRD per currency -> scenarios & CIP arbitrage.

Conventions: pairs BASE/QUOTE (EURUSD = USD per EUR); domestic = quote ccy,
foreign = base ccy; zero rates continuously compounded annualised; all PVs
in the quote currency.  See CONVENTIONS.md and docs/METHODOLOGY.md.
"""

from .arbitrage import (
    CIPArbitrageResult,
    CIPQuotes,
    detect_cip_arbitrage,
    no_arb_bounds,
)
from .bootstrap import (
    basis_adjusted_curve,
    bootstrap_curve,
    curve_from_fx_forwards,
    deposit_rate_from_df,
    df_from_deposit,
    implied_basis_from_forwards,
    par_swap_rate,
    reprice_deposits,
    reprice_swaps,
)
from .curve import DiscountCurve
from .daycount import (
    VALID_CONVENTIONS,
    add_calendar_days,
    forward_settlement_date,
    spot_date,
    tenor_to_years,
    year_fraction,
)
from .fxforward import (
    FXForward,
    FXSwap,
    MarketState,
    cip_forward,
    forward_points,
    forward_points_table,
    market_forward,
)
from .risk import (
    basis_dv01,
    book_risk_report,
    book_value,
    dv01,
    fx_delta,
    key_rate_dv01,
    position_risk,
)
from .scenarios import (
    Scenario,
    apply_scenario,
    carry_table,
    forward_carry,
    historical_scenarios,
    scenario_table,
)
from .xccy import (
    CrossCurrencySwap,
    solve_par_basis,
    solve_par_rate_base,
    solve_par_rate_quote,
)

__version__ = "1.0.0"

__all__ = [
    # daycount
    "VALID_CONVENTIONS", "year_fraction", "add_calendar_days", "spot_date",
    "forward_settlement_date", "tenor_to_years",
    # curve
    "DiscountCurve",
    # bootstrap
    "df_from_deposit", "deposit_rate_from_df", "par_swap_rate",
    "bootstrap_curve", "reprice_deposits", "reprice_swaps",
    "basis_adjusted_curve", "implied_basis_from_forwards",
    "curve_from_fx_forwards",
    # fx forwards
    "MarketState", "cip_forward", "market_forward", "forward_points",
    "forward_points_table", "FXForward", "FXSwap",
    # xccy
    "CrossCurrencySwap", "solve_par_rate_quote", "solve_par_rate_base",
    "solve_par_basis",
    # risk
    "fx_delta", "dv01", "key_rate_dv01", "basis_dv01", "position_risk",
    "book_value", "book_risk_report",
    # scenarios
    "Scenario", "apply_scenario", "scenario_table", "historical_scenarios",
    "forward_carry", "carry_table",
    # arbitrage
    "CIPQuotes", "CIPArbitrageResult", "no_arb_bounds", "detect_cip_arbitrage",
]
