"""Hedge a short European option along price paths and report error and cost.

Plain idea
----------
Sell one option at the model price, hold shares according to a rule, pay
the proportional cost on every trade, and at expiry pay the option off. What
is left is the hedging P&L. Its spread across paths is the risk the rule
leaves; its mean cost is what the rule spends. The effectiveness metric
compares the spread with that of the same short option left unhedged.

Formal statement
----------------
For a path ``S_0..S_N`` on a grid of step ``dt`` and a rule giving holdings
``h_i``, with cash accruing at ``r``:

    cash_0 = premium - h_0 S_0 - k |h_0| S_0
    cash_i = cash_{i-1} e^{r dt} - (h_i - h_{i-1}) S_i - k |h_i - h_{i-1}| S_i
    PnL    = cash_N e^{r dt} + h_{N-1} S_N - payoff(S_N)

(the final share position is valued at ``S_N``; liquidation cost is not
charged, matching Leland 1985). The P&L splits exactly into
``PnL = hedge_error - cost`` where ``cost`` is the compounded sum of the
``k |dh| S`` terms; ``test_hedging.py`` checks the identity to 1e-9.

Effectiveness is ``1 - Var(PnL_hedged) / Var(PnL_unhedged)`` (Ederington
1979), reported with a delta-method standard error from the path sample.

Worked example
--------------
>>> import numpy as np
>>> from qp_core.hedging.simulate import gbm_paths, hedge_short_option
>>> from qp_core.hedging.policies import HedgeRule, HedgeRuleSpec
>>> paths = gbm_paths(100.0, 0.0, 0.2, 0.25, 63, 2000, seed=1)
>>> res = hedge_short_option(paths, 100.0, 0.25, 0.0, 0.2, True,
...                          HedgeRuleSpec(HedgeRule.TIME, cost=0.0))
>>> bool(abs(res.pnl.mean()) < 0.05 and res.pnl.std() < 0.6)
True
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qp_core.black_scholes import carry_greeks
from qp_core.errors import InputError
from qp_core.hedging.policies import HedgeRule, HedgeRuleSpec, leland_sigma, whalley_wilmott_halfwidth

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "Effectiveness",
    "HedgeResult",
    "effectiveness",
    "gbm_paths",
    "hedge_short_option",
]


@dataclass(frozen=True)
class HedgeResult:
    """Per-path outcome; ``pnl == hedge_error - cost`` element by element."""

    pnl: FloatArray
    hedge_error: FloatArray
    cost: FloatArray
    trades: npt.NDArray[np.int64]
    unhedged_pnl: FloatArray
    premium: float


@dataclass(frozen=True)
class Effectiveness:
    """``1 - Var(hedged)/Var(unhedged)`` with a delta-method standard error."""

    value: float
    se: float
    mean_cost: float
    mean_cost_se: float
    pnl_std: float


def gbm_paths(
    s0: float, mu: float, sigma: float, horizon: float, steps: int, n_paths: int, seed: int = 0
) -> FloatArray:
    """Geometric Brownian motion, shape ``(n_paths, steps + 1)``, exact log-Euler."""
    if not (s0 > 0.0 and sigma >= 0.0 and horizon > 0.0 and steps >= 1 and n_paths >= 1):
        raise InputError("gbm_paths needs s0 > 0, sigma >= 0, horizon > 0, steps and n_paths >= 1")
    dt = horizon / steps
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, steps))
    incr = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
    logs = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    out: FloatArray = s0 * np.exp(logs)
    return out


def _delta_gamma(
    s: float, k: float, tau: float, r: float, sigma: float, is_call: bool
) -> tuple[float, float]:
    if tau <= 0.0:
        itm = s > k if is_call else s < k
        return (1.0 if is_call else -1.0) if itm else 0.0, 0.0
    g = carry_greeks(s, k, tau, r, 0.0, sigma, is_call)
    return g.delta, g.gamma


def hedge_short_option(
    paths: FloatArray,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
    spec: HedgeRuleSpec,
    hedge_sigma: FloatArray | None = None,
) -> HedgeResult:
    """Run ``spec`` on every row of ``paths`` (grid spans ``[0, maturity]``).

    ``sigma`` prices the premium. ``hedge_sigma`` (shape ``(n_paths, steps)``)
    optionally supplies the volatility used for deltas at each step, as in
    the out-of-sample test where an implied-volatility proxy is observed
    daily; by default ``sigma`` is used throughout.
    """
    p = np.asarray(paths, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2 or not np.all(np.isfinite(p)) or np.any(p <= 0.0):
        raise InputError("paths must be a finite positive 2-D array with at least two columns")
    if not (strike > 0.0 and maturity > 0.0 and sigma > 0.0):
        raise InputError("strike, maturity and sigma must be > 0")
    n_paths, n_cols = p.shape
    steps = n_cols - 1
    dt = maturity / steps
    if hedge_sigma is not None:
        hs = np.asarray(hedge_sigma, dtype=np.float64)
        if hs.shape != (n_paths, steps) or np.any(~np.isfinite(hs)) or np.any(hs <= 0.0):
            raise InputError("hedge_sigma must be finite, > 0, shape (n_paths, steps)")
    else:
        hs = np.full((n_paths, steps), sigma)
    growth = math.exp(rate * dt)
    k = spec.cost
    phi = 1.0 if is_call else -1.0

    pnl = np.empty(n_paths)
    cost_out = np.empty(n_paths)
    trades = np.zeros(n_paths, dtype=np.int64)
    unhedged = np.empty(n_paths)
    for j in range(n_paths):
        premium = carry_greeks(float(p[j, 0]), strike, maturity, rate, 0.0, sigma, is_call).price
        cash = premium
        cost_fv = 0.0
        h = 0.0
        for i in range(steps):
            s = float(p[j, i])
            tau = maturity - i * dt
            vol = float(hs[j, i])
            if spec.rule is HedgeRule.LELAND and k > 0.0:
                vol = leland_sigma(vol, k, dt * spec.every)
            delta, gamma = _delta_gamma(s, strike, tau, rate, vol, is_call)
            if spec.rule is HedgeRule.BAND:
                half = whalley_wilmott_halfwidth(s, gamma, k, spec.risk_aversion, rate, tau)
                target = h if abs(h - delta) <= half else (delta + half if h > delta else delta - half)
                if i == 0:
                    target = delta
            else:
                target = delta if i % spec.every == 0 else h
            dh = target - h
            if dh != 0.0:
                trades[j] += 1
            c = k * abs(dh) * s
            if i > 0:
                cash *= growth
                cost_fv *= growth
            cash -= dh * s + c
            cost_fv += c
            h = target
        s_n = float(p[j, -1])
        payoff = max(phi * (s_n - strike), 0.0)
        cash *= growth
        cost_fv *= growth
        pnl[j] = cash + h * s_n - payoff
        cost_out[j] = cost_fv
        unhedged[j] = premium * growth**steps - payoff
    return HedgeResult(
        pnl=pnl,
        hedge_error=pnl + cost_out,
        cost=cost_out,
        trades=trades,
        unhedged_pnl=unhedged,
        premium=float(carry_greeks(float(p[0, 0]), strike, maturity, rate, 0.0, sigma, is_call).price),
    )


def effectiveness(result: HedgeResult) -> Effectiveness:
    """Ederington effectiveness with a delta-method SE (needs >= 10 paths).

    For ``E = 1 - a/b`` with ``a = Var(hedged)``, ``b = Var(unhedged)``, the
    SE uses the sample covariance of the squared deviations, so it accounts
    for the two variances being estimated on the same paths.
    """
    x = result.pnl
    y = result.unhedged_pnl
    n = x.size
    if n < 10:
        raise InputError(f"effectiveness needs at least 10 paths, got {n}")
    dx = (x - x.mean()) ** 2
    dy = (y - y.mean()) ** 2
    a, b = float(dx.mean()), float(dy.mean())
    if b <= 0.0:
        raise InputError("unhedged P&L has zero variance")
    cov = np.cov(np.vstack([dx, dy]), ddof=1) / n
    grad = np.array([-1.0 / b, a / (b * b)])
    se = math.sqrt(max(float(grad @ cov @ grad), 0.0))
    c = result.cost
    return Effectiveness(
        value=1.0 - a / b,
        se=se,
        mean_cost=float(c.mean()),
        mean_cost_se=float(c.std(ddof=1) / math.sqrt(n)),
        pnl_std=float(x.std(ddof=1)),
    )
