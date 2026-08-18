"""eq_port: equity portfolio optimization & risk allocation.

Pipeline: return estimation -> covariance modeling -> mean-variance
optimization -> efficient frontier -> risk parity -> Sharpe/metrics ->
walk-forward backtesting. See docs/METHODOLOGY.md for the estimation-error
story that motivates the whole stack.
"""

from .backtest import (
    BacktestResult,
    make_erc_strategy,
    make_min_variance_strategy,
    make_static_strategy,
    make_tangency_strategy,
    run_backtest,
    run_race,
    strategy_equal_weight,
)
from .covariance import (
    LedoitWolfResult,
    condition_number,
    ewma_cov,
    is_psd,
    ledoit_wolf_cc,
    psd_repair,
    sample_cov,
    single_factor_cov,
)
from .metrics import (
    LoSharpeResult,
    annualized_return,
    annualized_vol,
    calmar_ratio,
    diversification_ratio,
    effective_n,
    max_drawdown,
    realized_risk_contributions,
    sharpe_lo,
    sharpe_ratio,
    sortino_ratio,
    summary_table,
)
from .mvo import (
    FrontierResult,
    efficient_frontier,
    max_sharpe_constrained,
    min_variance_constrained,
    min_variance_weights,
    portfolio_return,
    portfolio_vol,
    tangency_weights,
    target_return_portfolio,
    target_risk_portfolio,
)
from .returns_est import (
    BlackLittermanResult,
    JamesSteinResult,
    black_litterman,
    ewma_mean,
    implied_equilibrium_returns,
    james_stein_mean,
    sample_mean,
)
from .risk_parity import (
    erc_weights,
    inverse_vol_weights,
    risk_contributions,
    vol_target_overlay,
)

__version__ = "1.0.0"

__all__ = [
    # returns_est
    "sample_mean", "ewma_mean", "james_stein_mean", "JamesSteinResult",
    "implied_equilibrium_returns", "black_litterman", "BlackLittermanResult",
    # covariance
    "sample_cov", "ewma_cov", "ledoit_wolf_cc", "LedoitWolfResult",
    "single_factor_cov", "psd_repair", "condition_number", "is_psd",
    # mvo
    "min_variance_weights", "tangency_weights", "min_variance_constrained",
    "max_sharpe_constrained", "target_return_portfolio", "target_risk_portfolio",
    "efficient_frontier", "FrontierResult", "portfolio_vol", "portfolio_return",
    # risk parity
    "risk_contributions", "erc_weights", "inverse_vol_weights", "vol_target_overlay",
    # backtest
    "BacktestResult", "run_backtest", "run_race", "strategy_equal_weight",
    "make_min_variance_strategy", "make_tangency_strategy", "make_erc_strategy",
    "make_static_strategy",
    # metrics
    "annualized_return", "annualized_vol", "sharpe_ratio", "sharpe_lo",
    "LoSharpeResult", "sortino_ratio", "max_drawdown", "calmar_ratio",
    "diversification_ratio", "effective_n", "realized_risk_contributions",
    "summary_table",
]
