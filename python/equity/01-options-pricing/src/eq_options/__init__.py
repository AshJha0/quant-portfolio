"""eq_options: equity options pricing and Greeks engine.

Pipeline: Black-Scholes -> CRR binomial -> Black-76 -> Monte Carlo ->
Greeks -> delta hedging -> model comparison.

Conventions (portfolio-wide): rates and dividend yields continuously
compounded, annualised, ACT/365F; ``T`` in years; vols quoted on
log-returns, annualised.
"""

from .binomial import crr_price, early_exercise_premium
from .black76 import Black76Greeks, b76_d1_d2, black76_greeks, black76_price
from .black_scholes import (
    bs_price,
    d1_d2,
    forward_price,
    implied_vol,
    intrinsic_value,
    validate_inputs,
)
from .comparison import compare_models, mc_convergence_table, tree_convergence_table
from .greeks import BSGreeks, bs_greeks, compare_greeks, fd_greeks
from .hedging import HedgeResult, pnl_std_vs_frequency, simulate_delta_hedge
from .monte_carlo import (
    MCResult,
    mc_delta_lr,
    mc_delta_pathwise,
    mc_greek_fd,
    mc_price,
    mc_vega_lr,
    mc_vega_pathwise,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # black_scholes
    "bs_price",
    "d1_d2",
    "implied_vol",
    "intrinsic_value",
    "forward_price",
    "validate_inputs",
    # binomial
    "crr_price",
    "early_exercise_premium",
    # black76
    "black76_price",
    "black76_greeks",
    "Black76Greeks",
    "b76_d1_d2",
    # monte_carlo
    "MCResult",
    "mc_price",
    "mc_delta_pathwise",
    "mc_vega_pathwise",
    "mc_delta_lr",
    "mc_vega_lr",
    "mc_greek_fd",
    # greeks
    "BSGreeks",
    "bs_greeks",
    "fd_greeks",
    "compare_greeks",
    # hedging
    "HedgeResult",
    "simulate_delta_hedge",
    "pnl_std_vs_frequency",
    # comparison
    "compare_models",
    "tree_convergence_table",
    "mc_convergence_table",
]
