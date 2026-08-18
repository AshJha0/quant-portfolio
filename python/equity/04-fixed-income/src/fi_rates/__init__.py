"""fi_rates — Fixed income pricing & risk analytics.

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
    key_rate_dv01s,
    key_rate_durations,
    krd_report,
    triangle_weights,
)
from .risk import (
    Position,
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
    # daycount
    "SUPPORTED_CONVENTIONS",
    "SUPPORTED_FREQUENCIES",
    "year_fraction",
    "add_months",
    "adjust_modified_following",
    "generate_schedule",
    # curve
    "DiscountCurve",
    "ExtrapolationWarning",
    "INTERPOLATIONS",
    # bootstrap
    "Deposit",
    "FRA",
    "ParSwap",
    "BondQuote",
    "bootstrap_curve",
    "bootstrap_bond_curve",
    "reprice_instruments",
    # bond
    "FixedRateBond",
    "ZeroCouponBond",
    "bond_cashflows",
    "accrued_interest",
    "clean_price_from_curve",
    "dirty_price_from_curve",
    "price_from_ytm",
    "ytm_from_price",
    "z_spread_from_price",
    "zcb_price_from_curve",
    "frn_price_from_curve",
    "annuity_pv",
    "curve_time",
    # risk
    "macaulay_duration",
    "modified_duration",
    "convexity",
    "dv01",
    "dv01_curve",
    "convexity_curve",
    "numerical_modified_duration",
    "numerical_convexity",
    "pnl_approximation_table",
    "Position",
    "portfolio_value",
    "portfolio_risk",
    # keyrates
    "DEFAULT_KEY_TENORS",
    "triangle_weights",
    "key_rate_dv01s",
    "key_rate_durations",
    "key_rate_convexities",
    "krd_report",
    # scenarios
    "Scenario",
    "parallel_scenario",
    "steepener_scenario",
    "butterfly_scenario",
    "HISTORICAL_SCENARIOS",
    "apply_scenario",
    "scenario_pnl_table",
    "carry_rolldown",
]
