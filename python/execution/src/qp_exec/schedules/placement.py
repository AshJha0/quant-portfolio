"""The passive-or-cross decision for one child order.

Plain idea
----------
Posting at the near touch earns the half spread but may not fill, and when it
does fill the price has often just moved against you. Crossing pays the half
spread and a fee but is certain. Post when the expected cost of posting is
lower. Ideas follow the owner's electronic-trading placement module; the code
and the model below are rewritten.

Formal statement
----------------
Costs per unit, in ticks relative to the mid at decision, positive = cost.
``s`` half spread, ``f_take`` taker fee, ``r_make`` maker rebate, ``a``
adverse-selection loss on a passive fill (minus the mean markout of passive
fills, measured with ``qp_exec.lob.fills.markouts``), ``mu`` expected mid
drift against us per unit time (alpha / urgency), ``T`` the horizon we wait
before crossing anyway::

    cost_cross = s + f_take
    cost_post  = p * (-s - r_make + a) + (1 - p) * (s + f_take + mu * T)

``p`` is the fill probability within ``T``: the queue ahead ``Q`` plus our
``q`` must trade through. With market orders of mean size ``m`` arriving at
rate ``lam`` (Poisson; cancels ahead of us ignored, so ``p`` is conservative,
Cont and de Larrard 2013 add them)::

    p = P(N(T) >= ceil((Q + q) / m)),  N(T) ~ Poisson(lam * T)

Worked example
--------------
>>> d = decide(queue_ahead=0, qty=1, half_spread=0.5, take_fee=0.1, make_rebate=0.05,
...            adverse=0.2, drift=0.0, horizon=10.0, market_rate=1.0, mean_size=1.0)
>>> d.action, round(d.p_fill, 4)
('post', 1.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class Placement:
    action: str  # "post" or "cross"
    p_fill: float
    cost_post: float
    cost_cross: float


def fill_probability(queue_ahead: float, qty: float, market_rate: float, mean_size: float,
                     horizon: float) -> float:
    if mean_size <= 0 or market_rate < 0 or horizon < 0:
        raise ValueError("mean_size > 0, market_rate >= 0, horizon >= 0")
    k = math.ceil((queue_ahead + qty) / mean_size)
    return float(stats.poisson.sf(k - 1, market_rate * horizon))


def decide(queue_ahead: float, qty: float, half_spread: float, take_fee: float,
           make_rebate: float, adverse: float, drift: float, horizon: float,
           market_rate: float, mean_size: float) -> Placement:
    p = fill_probability(queue_ahead, qty, market_rate, mean_size, horizon)
    cross = half_spread + take_fee
    post = p * (-half_spread - make_rebate + adverse) + (1 - p) * (cross + drift * horizon)
    return Placement("post" if post < cross else "cross", p, post, cross)
