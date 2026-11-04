"""fx_options: FX options pricing & Greeks engine.

Garman-Kohlhagen, CRR binomial, Black-76 on the forward, Monte Carlo,
the four FX delta conventions, analytic Greeks (both rhos, vanna/volga),
and a delta-hedging simulator with foreign-interest accounting.

Conventions: pairs quoted BASE/QUOTE (EURUSD = USD per EUR); r_d = quote
ccy rate, r_f = base ccy rate; rates continuously compounded, ACT/365F.
"""

from .arrays import gk_greeks_array, gk_price_array
from .binomial import binomial_convergence, binomial_price
from .black76 import black76_from_spot, black76_price
from .comparison import binomial_convergence_table, compare_models, mc_convergence_table
from .deltas import (
                         CONVENTIONS,
                         atm_dns_strike,
                         atm_forward_strike,
                         delta,
                         forward_to_spot_delta,
                         premium_adjust_spot_delta,
                         spot_to_forward_delta,
                         strike_from_delta,
)
from .forwards import (
    cip_forward,
    forward_from_dates,
    forward_points,
    synthetic_forward_from_options,
)
from .garman_kohlhagen import d1, d2, gk_call, gk_price, gk_put, implied_vol
from .greeks import GreeksResult, analytic_greeks, finite_difference_greeks, gamma, vanna, vega, volga
from .hedging import HedgeResult, hedge_frequency_study, simulate_delta_hedge
from .monte_carlo import MCResult, digital_price, mc_digital_price, mc_price

__version__ = "3.0.0"

__all__ = [
                         "CONVENTIONS",
                         "GreeksResult",
                         "HedgeResult",
                         "MCResult",
                         "__version__",
                         "analytic_greeks",
                         "atm_dns_strike",
                         "atm_forward_strike",
                         "binomial_convergence",
                         "binomial_convergence_table",
                         "binomial_price",
                         "black76_from_spot",
                         "black76_price",
                         "cip_forward",
                         "compare_models",
                         "d1",
                         "d2",
                         "delta",
                         "digital_price",
                         "finite_difference_greeks",
                         "forward_from_dates",
                         "forward_points",
                         "forward_to_spot_delta",
                         "gamma",
                         "gk_call",
                         "gk_greeks_array",
                         "gk_price",
                         "gk_price_array",
                         "gk_put",
                         "hedge_frequency_study",
                         "implied_vol",
                         "mc_convergence_table",
                         "mc_digital_price",
                         "mc_price",
                         "premium_adjust_spot_delta",
                         "simulate_delta_hedge",
                         "spot_to_forward_delta",
                         "strike_from_delta",
                         "synthetic_forward_from_options",
                         "vanna",
                         "vega",
                         "volga",
]
