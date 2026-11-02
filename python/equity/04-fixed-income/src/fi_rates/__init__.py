"""fi_rates: Fixed income pricing & risk analytics.

Pipeline: yield curve construction -> bootstrapping -> bond pricing ->
duration -> convexity -> key-rate duration -> scenario analysis.
"""

from .bond import (
    FixedRateBond,
    ZeroCouponBond,
    accrued_interest,
    annuity_pv,
    bond_cashflows,
    clean_price_from_curve,
    curve_time,
    dirty_price_from_curve,
    frn_price_from_curve,
    price_from_ytm,
    ytm_from_price,
    z_spread_from_price,
    zcb_price_from_curve,
)
from .bootstrap import (
    FRA,
    BondQuote,
    Deposit,
    ParSwap,
    bootstrap_bond_curve,
    bootstrap_curve,
    reprice_instruments,
)
from .curve import INTERPOLATIONS, DiscountCurve, ExtrapolationWarning
from .daycount import (
    SUPPORTED_CONVENTIONS,
    SUPPORTED_FREQUENCIES,
    add_months,
    adjust_modified_following,
    generate_schedule,
    year_fraction,
)
from .keyrates import (
    DEFAULT_KEY_TENORS,
    key_rate_convexities,
    key_rate_durations,
    key_rate_dv01s,
    krd_report,
    triangle_weights,
)
from .risk import (
    Position,
    ZeroNetValueWarning,
    convexity,
    convexity_curve,
    dv01,
    dv01_curve,
    macaulay_duration,
    modified_duration,
    numerical_convexity,
    numerical_modified_duration,
    pnl_approximation_table,
    portfolio_risk,
    portfolio_value,
)
from .scenarios import (
    HISTORICAL_SCENARIOS,
    Scenario,
    apply_scenario,
    butterfly_scenario,
    carry_rolldown,
    parallel_scenario,
    scenario_pnl_table,
    steepener_scenario,
)

__version__ = "1.0.0"

__all__ = [
    # keyrates
    "DEFAULT_KEY_TENORS",
    "FRA",
    "HISTORICAL_SCENARIOS",
    "INTERPOLATIONS",
    # daycount
    "SUPPORTED_CONVENTIONS",
    "SUPPORTED_FREQUENCIES",
    "BondQuote",
    # bootstrap
    "Deposit",
    # curve
    "DiscountCurve",
    "ExtrapolationWarning",
    # bond
    "FixedRateBond",
    "ParSwap",
    "Position",
    # scenarios
    "Scenario",
    "ZeroCouponBond",
    "ZeroNetValueWarning",
    "accrued_interest",
    "add_months",
    "adjust_modified_following",
    "annuity_pv",
    "apply_scenario",
    "bond_cashflows",
    "bootstrap_bond_curve",
    "bootstrap_curve",
    "butterfly_scenario",
    "carry_rolldown",
    "clean_price_from_curve",
    "convexity",
    "convexity_curve",
    "curve_time",
    "dirty_price_from_curve",
    "dv01",
    "dv01_curve",
    "frn_price_from_curve",
    "generate_schedule",
    "key_rate_convexities",
    "key_rate_durations",
    "key_rate_dv01s",
    "krd_report",
    # risk
    "macaulay_duration",
    "modified_duration",
    "numerical_convexity",
    "numerical_modified_duration",
    "parallel_scenario",
    "pnl_approximation_table",
    "portfolio_risk",
    "portfolio_value",
    "price_from_ytm",
    "reprice_instruments",
    "scenario_pnl_table",
    "steepener_scenario",
    "triangle_weights",
    "year_fraction",
    "ytm_from_price",
    "z_spread_from_price",
    "zcb_price_from_curve",
]
