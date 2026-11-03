"""fx_rates: FX-linked fixed income: multi-currency curves, CIP forward
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
    basis_discount_curve,
    basis_swap_par_spread,
    bootstrap_curve,
    curve_from_fx_forwards,
    deposit_rate_from_df,
    df_from_deposit,
    implied_basis_from_forwards,
    par_swap_rate,
    reprice_deposits,
    reprice_swaps,
)
from .calendars import BusinessCalendar, HolidayCalendar, get_calendar, joint_calendar
from .curve import DiscountCurve
from .daycount import (
    MONEY_MARKET_DAYCOUNT,
    VALID_CONVENTIONS,
    add_calendar_days,
    forward_settlement_date,
    money_market_year_fraction,
    pair_calendar,
    spot_date,
    tenor_dates,
    tenor_to_years,
    year_fraction,
)
from .fxforward import (
    FXForward,
    FXSwap,
    MarketState,
    cip_forward,
    curve_from_forward_points,
    forward_for_date,
    forward_points,
    forward_points_for_date,
    forward_points_table,
    market_forward,
)
from .ndf import NDF, ndf_fixing_date
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
from .swaps import (
    DatedFXSwap,
    fair_swap_points,
    price_fx_swap,
    roll_position,
    tom_next_swap,
)
from .xccy import (
    BasisSwap,
    CrossCurrencySwap,
    solve_par_basis,
    solve_par_rate_base,
    solve_par_rate_quote,
)

__version__ = "3.0.0.dev0"

__all__ = [
    "MONEY_MARKET_DAYCOUNT",
    "NDF",
    # daycount
    "VALID_CONVENTIONS",
    # xccy
    "BasisSwap",
    # calendars
    "BusinessCalendar",
    "CIPArbitrageResult",
    # arbitrage
    "CIPQuotes",
    "CrossCurrencySwap",
    # dated FX swaps
    "DatedFXSwap",
    # curve
    "DiscountCurve",
    "FXForward",
    "FXSwap",
    "HolidayCalendar",
    # fx forwards
    "MarketState",
    # scenarios
    "Scenario",
    "add_calendar_days",
    "apply_scenario",
    "basis_adjusted_curve",
    "basis_discount_curve",
    "basis_dv01",
    "basis_swap_par_spread",
    "book_risk_report",
    "book_value",
    "bootstrap_curve",
    "carry_table",
    "cip_forward",
    "curve_from_forward_points",
    "curve_from_fx_forwards",
    "deposit_rate_from_df",
    "detect_cip_arbitrage",
    # bootstrap
    "df_from_deposit",
    "dv01",
    "fair_swap_points",
    "forward_carry",
    "forward_for_date",
    "forward_points",
    "forward_points_for_date",
    "forward_points_table",
    "forward_settlement_date",
    # risk
    "fx_delta",
    "get_calendar",
    "historical_scenarios",
    "implied_basis_from_forwards",
    "joint_calendar",
    "key_rate_dv01",
    "market_forward",
    "money_market_year_fraction",
    "ndf_fixing_date",
    "no_arb_bounds",
    "pair_calendar",
    "par_swap_rate",
    "position_risk",
    "price_fx_swap",
    "reprice_deposits",
    "reprice_swaps",
    "roll_position",
    "scenario_table",
    "solve_par_basis",
    "solve_par_rate_base",
    "solve_par_rate_quote",
    "spot_date",
    "tenor_dates",
    "tenor_to_years",
    "tom_next_swap",
    "year_fraction",
]
