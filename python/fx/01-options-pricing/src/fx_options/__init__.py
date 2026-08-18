"""fx_options: FX options pricing & Greeks engine.

Garman-Kohlhagen, CRR binomial, Black-76 on the forward, Monte Carlo,
the four FX delta conventions, analytic Greeks (both rhos, vanna/volga),
and a delta-hedging simulator with foreign-interest accounting.

Conventions: pairs quoted BASE/QUOTE (EURUSD = USD per EUR); r_d = quote
ccy rate, r_f = base ccy rate; rates continuously compounded, ACT/365F.
"""

from .binomial import binomial_convergence, binomial_price
from .black76 import black76_from_spot, black76_price
from .comparison import (binomial_convergence_table, compare_models,
                         mc_convergence_table)
from .deltas import (CONVENTIONS, atm_dns_strike, atm_forward_strike, delta,
                     forward_to_spot_delta, premium_adjust_spot_delta,
                     spot_to_forward_delta, strike_from_delta)
from .forwards import (cip_forward, forward_points,
                       synthetic_forward_from_options)
from .garman_kohlhagen import (d1, d2, gk_call, gk_price, gk_put,
                               implied_vol)
from .greeks import (GreeksResult, analytic_greeks,
                     finite_difference_greeks, gamma, vanna, vega, volga)
from .hedging import HedgeResult, hedge_frequency_study, simulate_delta_hedge
from .monte_carlo import MCResult, digital_price, mc_digital_price, mc_price

__version__ = "1.0.0"

__all__ = [
    "d1", "d2", "gk_price", "gk_call", "gk_put", "implied_vol",
    "cip_forward", "forward_points", "synthetic_forward_from_options",
    "CONVENTIONS", "delta", "spot_to_forward_delta", "forward_to_spot_delta",
    "premium_adjust_spot_delta", "strike_from_delta", "atm_forward_strike",
    "atm_dns_strike",
    "binomial_price", "binomial_convergence",
    "black76_price", "black76_from_spot",
    "MCResult", "mc_price", "mc_digital_price", "digital_price",
    "GreeksResult", "analytic_greeks", "finite_difference_greeks",
    "gamma", "vega", "vanna", "volga",
    "HedgeResult", "simulate_delta_hedge", "hedge_frequency_study",
    "compare_models", "binomial_convergence_table", "mc_convergence_table",
    "__version__",
]
