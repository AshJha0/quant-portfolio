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
    SharpeDiffResult,
    annualized_return,
    annualized_vol,
    calmar_ratio,
    diversification_ratio,
    effective_n,
    max_drawdown,
    realized_risk_contributions,
    sharpe_difference_test,
    sharpe_lo,
    sharpe_ratio,
    sortino_ratio,
    summary_table,
)
from .mvo import (
    FrontierResult,
    MaxSharpeResult,
    efficient_frontier,
    max_sharpe_constrained,
    max_sharpe_qp,
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

__version__ = "3.0.0"

__all__ = [
    # backtest
    "BacktestResult",
    "BlackLittermanResult",
    "FrontierResult",
    "JamesSteinResult",
    "LedoitWolfResult",
    "LoSharpeResult",
    "MaxSharpeResult",
    "SharpeDiffResult",
    # metrics
    "annualized_return",
    "annualized_vol",
    "black_litterman",
    "calmar_ratio",
    "condition_number",
    "diversification_ratio",
    "effective_n",
    "efficient_frontier",
    "erc_weights",
    "ewma_cov",
    "ewma_mean",
    "implied_equilibrium_returns",
    "inverse_vol_weights",
    "is_psd",
    "james_stein_mean",
    "ledoit_wolf_cc",
    "make_erc_strategy",
    "make_min_variance_strategy",
    "make_static_strategy",
    "make_tangency_strategy",
    "max_drawdown",
    "max_sharpe_constrained",
    "max_sharpe_qp",
    "min_variance_constrained",
    # mvo
    "min_variance_weights",
    "portfolio_return",
    "portfolio_vol",
    "psd_repair",
    "realized_risk_contributions",
    # risk parity
    "risk_contributions",
    "run_backtest",
    "run_race",
    # covariance
    "sample_cov",
    # returns_est
    "sample_mean",
    "sharpe_difference_test",
    "sharpe_lo",
    "sharpe_ratio",
    "single_factor_cov",
    "sortino_ratio",
    "strategy_equal_weight",
    "summary_table",
    "tangency_weights",
    "target_return_portfolio",
    "target_risk_portfolio",
    "vol_target_overlay",
]
