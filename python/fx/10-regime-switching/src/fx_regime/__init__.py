"""fx_regime: FX regime-switching quant strategy (risk-on / risk-off).

Pipeline: FX-native feature engineering -> PCA -> GMM/HMM (from
scratch) -> point-in-time filtered regime detection -> regime-
conditional carry / safe-haven / long-USD baskets -> walk-forward
backtesting -> oracle-vs-filtered risk analysis.
"""

from .data.synthetic import (
    CURRENCIES,
    EM,
    G10,
    G10_CARRY,
    G10_NEUTRAL,
    HAVENS,
    MEAN_DEPOSIT_RATES,
    PIP_FRACTION,
    SPREAD_PIPS,
    STATE_NAMES_2,
    STATE_NAMES_3,
    TRANSITION_2,
    TRANSITION_3,
    SyntheticPanel,
    generate_null_gbm_panel,
    generate_roro_panel,
    simulate_markov_chain,
    state_correlation,
    state_moments,
)
from .features import (
    FEATURE_COLUMNS,
    FeatureConfig,
    avg_pairwise_correlation,
    build_features,
    carry_basket_weights,
    expanding_standardize,
    realised_vol,
)
from .pca import PCAResult, fit_pca, roro_axis_check
from .gmm import GMMResult, fit_gmm, gaussian_logpdf, select_k_bic
from .hmm import (
    HMMModel,
    expected_durations,
    filtered_probabilities,
    fit_hmm,
    hmm_bic,
    log_backward,
    log_forward,
    match_states,
    smoothed_probabilities,
    stationary_distribution,
    viterbi,
)
from .detection import (
    DetectionConfig,
    DetectionResult,
    apply_hysteresis,
    label_states,
    run_detection,
)
from .strategy import (
    StrategyConfig,
    base_weights,
    carry_accrual,
    regime_weights,
    transaction_cost,
    vol_target_scale,
)
from .backtest import (
    BacktestResult,
    oracle_regimes,
    run_backtest,
    static_carry_regimes,
)
from .risk import (
    carry_drawdown_decomposition,
    comparison_table,
    detection_lag,
    detection_lag_report,
    oracle_gap_decomposition,
    per_regime_stats,
    perf_stats,
    regime_spells,
    transition_attribution,
)

__all__ = [
    # data
    "CURRENCIES", "EM", "G10", "G10_CARRY", "G10_NEUTRAL", "HAVENS",
    "MEAN_DEPOSIT_RATES", "PIP_FRACTION", "SPREAD_PIPS",
    "STATE_NAMES_2", "STATE_NAMES_3", "TRANSITION_2", "TRANSITION_3",
    "SyntheticPanel", "generate_null_gbm_panel", "generate_roro_panel",
    "simulate_markov_chain", "state_correlation", "state_moments",
    # features
    "FEATURE_COLUMNS", "FeatureConfig", "avg_pairwise_correlation",
    "build_features", "carry_basket_weights", "expanding_standardize",
    "realised_vol",
    # pca
    "PCAResult", "fit_pca", "roro_axis_check",
    # gmm
    "GMMResult", "fit_gmm", "gaussian_logpdf", "select_k_bic",
    # hmm
    "HMMModel", "expected_durations", "filtered_probabilities", "fit_hmm",
    "hmm_bic", "log_backward", "log_forward", "match_states",
    "smoothed_probabilities", "stationary_distribution", "viterbi",
    # detection
    "DetectionConfig", "DetectionResult", "apply_hysteresis",
    "label_states", "run_detection",
    # strategy
    "StrategyConfig", "base_weights", "carry_accrual", "regime_weights",
    "transaction_cost", "vol_target_scale",
    # backtest
    "BacktestResult", "oracle_regimes", "run_backtest",
    "static_carry_regimes",
    # risk
    "carry_drawdown_decomposition", "comparison_table", "detection_lag",
    "detection_lag_report", "oracle_gap_decomposition", "per_regime_stats",
    "perf_stats", "regime_spells", "transition_attribution",
]
