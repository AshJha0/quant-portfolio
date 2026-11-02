"""fx_regime: FX regime-switching quant strategy (risk-on / risk-off).

Pipeline: FX-native feature engineering -> PCA -> GMM/HMM (from
scratch) -> point-in-time filtered regime detection -> regime-
conditional carry / safe-haven / long-USD baskets -> walk-forward
backtesting -> oracle-vs-filtered risk analysis.
"""

from .backtest import (
    BacktestResult,
    ExecutionConfig,
    oracle_regimes,
    run_backtest,
    static_carry_regimes,
)
from .data.synthetic import (
    CURRENCIES,
    EM,
    G10,
    G10_CARRY,
    G10_NEUTRAL,
    HAVENS,
    MEAN_DEPOSIT_RATES,
    PIP_FRACTION,
    PIP_SIZE,
    QUOTES,
    SPREAD_PIPS,
    STATE_NAMES_2,
    STATE_NAMES_3,
    TRANSITION_2,
    TRANSITION_3,
    SyntheticPanel,
    generate_null_gbm_panel,
    generate_roro_panel,
    half_spread_fraction,
    quote_levels,
    simulate_markov_chain,
    state_correlation,
    state_moments,
)
from .detection import (
    DetectionConfig,
    DetectionResult,
    apply_hysteresis,
    label_states,
    run_detection,
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
from .pca import PCAResult, fit_pca, roro_axis_check
from .risk import (
    carry_drawdown_decomposition,
    comparison_table,
    detection_lag,
    detection_lag_report,
    oracle_gap_decomposition,
    per_regime_stats,
    perf_stats,
    random_timing_regimes,
    regime_spells,
    transition_attribution,
)
from .strategy import (
    StrategyConfig,
    base_weights,
    carry_accrual,
    regime_weights,
    transaction_cost,
    vol_target_scale,
)

__all__ = [
    # data
    "CURRENCIES",
    "EM",
    # features
    "FEATURE_COLUMNS",
    "G10",
    "G10_CARRY",
    "G10_NEUTRAL",
    "HAVENS",
    "MEAN_DEPOSIT_RATES",
    "PIP_FRACTION",
    "PIP_SIZE",
    "QUOTES",
    "SPREAD_PIPS",
    "STATE_NAMES_2",
    "STATE_NAMES_3",
    "TRANSITION_2",
    "TRANSITION_3",
    # backtest
    "BacktestResult",
    # detection
    "DetectionConfig",
    "DetectionResult",
    "ExecutionConfig",
    "FeatureConfig",
    # gmm
    "GMMResult",
    # hmm
    "HMMModel",
    # pca
    "PCAResult",
    # strategy
    "StrategyConfig",
    "SyntheticPanel",
    "apply_hysteresis",
    "avg_pairwise_correlation",
    "base_weights",
    "build_features",
    "carry_accrual",
    "carry_basket_weights",
    # risk
    "carry_drawdown_decomposition",
    "comparison_table",
    "detection_lag",
    "detection_lag_report",
    "expanding_standardize",
    "expected_durations",
    "filtered_probabilities",
    "fit_gmm",
    "fit_hmm",
    "fit_pca",
    "gaussian_logpdf",
    "generate_null_gbm_panel",
    "generate_roro_panel",
    "half_spread_fraction",
    "hmm_bic",
    "label_states",
    "log_backward",
    "log_forward",
    "match_states",
    "oracle_gap_decomposition",
    "oracle_regimes",
    "per_regime_stats",
    "perf_stats",
    "quote_levels",
    "random_timing_regimes",
    "realised_vol",
    "regime_spells",
    "regime_weights",
    "roro_axis_check",
    "run_backtest",
    "run_detection",
    "select_k_bic",
    "simulate_markov_chain",
    "smoothed_probabilities",
    "state_correlation",
    "state_moments",
    "static_carry_regimes",
    "stationary_distribution",
    "transaction_cost",
    "transition_attribution",
    "viterbi",
    "vol_target_scale",
]
