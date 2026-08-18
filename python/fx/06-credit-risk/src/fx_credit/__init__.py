"""fx_credit: FX / cross-border credit risk.

Three blocks:

1. Sovereign PD scorecard: synthetic country-year panel -> panel-aware
   cleaning & leakage guards -> WOE/IV binning -> from-scratch IRLS logistic
   -> PDO scorecard & rating bands -> validation (AUC/KS/HL/PSI).
2. FX settlement (Herstatt) risk: time-zone payment windows, gross vs
   CLS/PvP principal exposure.
3. Counterparty pre-settlement risk on FX forwards: GBM EE/PFE profiles,
   netting sets, CVA off the block-1 PD term structure; EL and Basel /
   Vasicek capital.
"""

from . import data
from .cleaning import (
    LEAKY_FIELDS,
    clean_panel,
    drop_leaky_fields,
    assert_no_leaky_fields,
    time_split,
    country_holdout_split,
)
from .woe import (
    WOEBin,
    WOETable,
    woe_table,
    monotone_merge,
    woe_transform,
    iv_report,
    flag_leaky_iv,
)
from .model import (
    LogisticFit,
    fit_logistic_irls,
    predict_pd,
    score_from_pd,
    pd_from_score,
    RATING_BANDS,
    RATING_ORDER,
    assign_rating,
    rating_midpoint_pd,
)
from .validation import (
    auc,
    gini,
    ks_statistic,
    hosmer_lemeshow,
    HosmerLemeshowResult,
    psi,
    bootstrap_auc_ci,
    within_country_autocorrelation,
)
from .settlement import (
    PAYMENT_SYSTEM_HOURS_UTC,
    FXTrade,
    SettlementExposure,
    at_risk_window_hours,
    time_zone_gap_matrix,
    settlement_exposure,
    gross_settlement_exposure,
    net_settlement_exposure,
    book_settlement_report,
)
from .exposure import (
    FXForward,
    ExposureProfile,
    simulate_fx_paths,
    forward_mtm,
    exposure_profile,
    netting_set_profile,
    hazard_from_pd1y,
    pd_term_structure,
    cva,
    cva_for_forward,
)
from .capital import (
    SOVEREIGN_RHO,
    STANDARDIZED_SOVEREIGN_RW,
    expected_loss,
    basel_corporate_correlation,
    vasicek_conditional_pd,
    vasicek_capital,
    standardized_rw,
    capital_table,
)

__all__ = [
    "data",
    # cleaning
    "LEAKY_FIELDS",
    "clean_panel",
    "drop_leaky_fields",
    "assert_no_leaky_fields",
    "time_split",
    "country_holdout_split",
    # woe
    "WOEBin",
    "WOETable",
    "woe_table",
    "monotone_merge",
    "woe_transform",
    "iv_report",
    "flag_leaky_iv",
    # model
    "LogisticFit",
    "fit_logistic_irls",
    "predict_pd",
    "score_from_pd",
    "pd_from_score",
    "RATING_BANDS",
    "RATING_ORDER",
    "assign_rating",
    "rating_midpoint_pd",
    # validation
    "auc",
    "gini",
    "ks_statistic",
    "hosmer_lemeshow",
    "HosmerLemeshowResult",
    "psi",
    "bootstrap_auc_ci",
    "within_country_autocorrelation",
    # settlement
    "PAYMENT_SYSTEM_HOURS_UTC",
    "FXTrade",
    "SettlementExposure",
    "at_risk_window_hours",
    "time_zone_gap_matrix",
    "settlement_exposure",
    "gross_settlement_exposure",
    "net_settlement_exposure",
    "book_settlement_report",
    # exposure
    "FXForward",
    "ExposureProfile",
    "simulate_fx_paths",
    "forward_mtm",
    "exposure_profile",
    "netting_set_profile",
    "hazard_from_pd1y",
    "pd_term_structure",
    "cva",
    "cva_for_forward",
    # capital
    "SOVEREIGN_RHO",
    "STANDARDIZED_SOVEREIGN_RW",
    "expected_loss",
    "basel_corporate_correlation",
    "vasicek_conditional_pd",
    "vasicek_capital",
    "standardized_rw",
    "capital_table",
]

__version__ = "1.0.0"
