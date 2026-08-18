"""eq_credit — corporate credit risk / PD modelling (bank-style scorecard).

Pipeline: data cleaning -> WOE/IV binning -> IRLS logistic scorecard -> PD ->
validation (AUC/KS, Hosmer-Lemeshow, PSI) -> expected loss -> Basel IRB and
Vasicek economic capital.
"""

from eq_credit.cleaning import (
    FORBIDDEN_POST_OUTCOME_FIELDS,
    LeakageError,
    MedianImputer,
    WinsorBounds,
    apply_winsor,
    check_leakage,
    drop_duplicate_loans,
    find_duplicates,
    fit_winsor_bounds,
    train_oot_split,
)
from eq_credit.model import (
    LogisticFit,
    ScorecardScaling,
    SeparationWarning,
    crosscheck_sklearn,
    fit_logistic,
    scorecard_points_table,
    stepwise_select,
)
from eq_credit.portfolio_risk import (
    asset_correlation,
    basel_k,
    basel_report,
    economic_capital,
    el_by_bucket,
    expected_loss,
    maturity_adjustment_b,
    risk_weighted_assets,
    simulate_portfolio_losses,
    vasicek_cdf,
    vasicek_quantile,
)
from eq_credit.validation import (
    bootstrap_auc_ci,
    brier_score,
    calibration_table,
    decile_table,
    gini,
    hosmer_lemeshow,
    is_monotone,
    ks_statistic,
    ks_table,
    psi,
    psi_from_proportions,
    psi_report,
    psi_status,
    roc_auc,
    roc_curve_points,
)
from eq_credit.woe import (
    FeatureBinning,
    SuspiciousIVWarning,
    WOETransformer,
    fit_categorical_binning,
    fit_numeric_binning,
    iv_strength,
    woe_iv_from_counts,
)

__all__ = [
    # cleaning
    "FORBIDDEN_POST_OUTCOME_FIELDS",
    "LeakageError",
    "check_leakage",
    "find_duplicates",
    "drop_duplicate_loans",
    "WinsorBounds",
    "fit_winsor_bounds",
    "apply_winsor",
    "MedianImputer",
    "train_oot_split",
    # woe
    "SuspiciousIVWarning",
    "iv_strength",
    "woe_iv_from_counts",
    "FeatureBinning",
    "fit_numeric_binning",
    "fit_categorical_binning",
    "WOETransformer",
    # model
    "SeparationWarning",
    "LogisticFit",
    "fit_logistic",
    "crosscheck_sklearn",
    "stepwise_select",
    "ScorecardScaling",
    "scorecard_points_table",
    # validation
    "roc_curve_points",
    "roc_auc",
    "gini",
    "ks_statistic",
    "ks_table",
    "bootstrap_auc_ci",
    "brier_score",
    "hosmer_lemeshow",
    "calibration_table",
    "psi_from_proportions",
    "psi",
    "psi_report",
    "psi_status",
    "decile_table",
    "is_monotone",
    # portfolio risk
    "expected_loss",
    "el_by_bucket",
    "asset_correlation",
    "maturity_adjustment_b",
    "basel_k",
    "risk_weighted_assets",
    "basel_report",
    "vasicek_cdf",
    "vasicek_quantile",
    "simulate_portfolio_losses",
    "economic_capital",
]

__version__ = "1.0.0"
