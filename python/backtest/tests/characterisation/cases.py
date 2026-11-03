"""Characterisation cases: the same public calls before and after migration.

Each case returns ``{key: array}``. Inputs are seeded synthetic panels so the
corpus is reproducible without data files.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from qp_core.execution import ExecutionConfig
from qp_core.stats import Financing

NEXT = ExecutionConfig()
NEXT1 = ExecutionConfig("next", fill_fraction=1.0)
AUCTION = ExecutionConfig("close_auction", auction_cost_bp=10.0)
EXECS = {"next": NEXT, "next1": NEXT1, "auction": AUCTION}


def _prices(seed: int, t_n: int, n: int, vol: float = 0.015) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=t_n)
    p = 40.0 * np.exp(np.cumsum(rng.normal(0.0002, vol, (t_n, n)), axis=0))
    return pd.DataFrame(p, index=idx, columns=[f"N{i:02d}" for i in range(n)])


# -- eq_signal_backtest -------------------------------------------------------

def signal_cases() -> dict[str, np.ndarray]:
    from eq_signal_backtest.engine import strategy_returns

    p = _prices(1, 300, 1)["N00"]
    rng = np.random.default_rng(2)
    sig = pd.Series((rng.random(300) < 0.5).astype(float), index=p.index)
    sig = sig.rolling(5).max().fillna(0.0)            # runs, not daily flips
    out = {}
    for ek, ex in EXECS.items():
        for c in (0.0, 5.0):
            r, pos, tr = strategy_returns(p, sig, c, ex)
            out[f"{ek}|{c}|ret"] = r.to_numpy()
            out[f"{ek}|{c}|pos"] = pos.to_numpy()
            out[f"{ek}|{c}|trades"] = tr.to_numpy()
    return out


# -- eq_port ------------------------------------------------------------------

def port_cases() -> dict[str, np.ndarray]:
    from eq_port.backtest import make_static_strategy, run_backtest, strategy_equal_weight

    ret = _prices(3, 200, 4).pct_change().iloc[1:]
    strategies = {"ew": strategy_equal_weight,
                  "lev": make_static_strategy(np.array([0.5, 0.4, 0.3, 0.3])),
                  "part": make_static_strategy(np.array([0.2, 0.2, 0.2, 0.1]))}
    out = {}
    for sk, fn in strategies.items():
        for k in (1, 5):
            for c in (0.0, 10.0):
                for fk, fin in (("none", None), ("fin", Financing(0.04, borrow_spread_bp=50.0))):
                    res = run_backtest(ret, fn, window=20, rebalance_every=k, cost_bps=c,
                                       financing=fin)
                    key = f"{sk}|{k}|{c}|{fk}"
                    out[f"{key}|net"] = res.net_returns.to_numpy()
                    out[f"{key}|gross"] = res.gross_returns.to_numpy()
                    out[f"{key}|turnover"] = res.turnover.to_numpy()
                    out[f"{key}|costs"] = res.costs.to_numpy()
    return out


# -- eq_algo ------------------------------------------------------------------

def algo_cases() -> dict[str, np.ndarray]:
    from eq_algo.backtest import BacktestConfig, run_backtest

    prices = _prices(4, 160, 40)
    rng = np.random.default_rng(5)
    signal = pd.DataFrame(rng.normal(size=prices.shape), index=prices.index,
                          columns=prices.columns).rolling(5, min_periods=1).mean()
    volumes = pd.DataFrame(rng.uniform(2e5, 2e6, prices.shape), index=prices.index,
                           columns=prices.columns)
    cfgs = {"free": BacktestConfig(n_quantiles=5, max_weight=0.2, linear_cost_bps=0.0),
            "lin": BacktestConfig(n_quantiles=5, max_weight=0.2, borrow_bps_annual=50.0),
            "aum": BacktestConfig(n_quantiles=5, max_weight=0.2, aum=5e7,
                                  borrow_bps_annual=50.0)}
    out = {}
    for ck, cfg in cfgs.items():
        for ek, ex in EXECS.items():
            for fk, fin in (("none", None), ("fin", Financing(0.03, borrow_spread_bp=25.0))):
                res = run_backtest(prices, signal, cfg, volumes=volumes, execution=ex,
                                   financing=fin)
                for col in res.ledger.columns:
                    out[f"{ck}|{ek}|{fk}|{col}"] = res.ledger[col].to_numpy(dtype=float)
    return out


# -- eq_regime ----------------------------------------------------------------

def regime_cases() -> dict[str, np.ndarray]:
    from eq_regime.backtest import run_ledger

    p = _prices(6, 260, 1)["N00"]
    r = p.pct_change().fillna(0.0)
    rng = np.random.default_rng(7)
    w_levels = pd.Series(rng.choice([0.0, 0.5, 1.0, 1.4], size=260), index=p.index)
    w = w_levels.rolling(10).max().fillna(0.0).iloc[:-1]
    w01 = (w > 0.6).astype(float)
    out = {}
    for wk, ws in (("levels", w), ("01", w01)):
        for ek, ex in EXECS.items():
            for c, cash in ((0.0, 0.0), (5.0, 0.0), (5.0, 0.04)):
                led = run_ledger(ws, r, cost_bps=c, execution=ex, cash_rate=cash)
                for col in led.columns:
                    out[f"{wk}|{ek}|{c}|{cash}|{col}"] = led[col].to_numpy(dtype=float)
    return out


# -- eq_pairs -----------------------------------------------------------------

def pairs_cases() -> dict[str, np.ndarray]:
    from eq_pairs.backtest import CostModel, backtest_pair

    prices = _prices(8, 180, 2, vol=0.012)
    y, x = prices["N00"], prices["N01"]
    rng = np.random.default_rng(9)
    raw = rng.choice([-1, 0, 0, 1], size=180)
    tgt = pd.Series(raw, index=prices.index).rolling(6).median().fillna(0).astype(int)
    costs = {"default": CostModel(),
             "full": CostModel(impact_coef=0.7, financing_bps=80.0, short_rebate_bps=30.0,
                               adv_dollars=2e7)}
    out = {}
    for ck, cm in costs.items():
        for ek, ex in EXECS.items():
            for sizing in ("dollar", "beta"):
                for cae in (True, False):
                    res = backtest_pair(y, x, tgt, beta=1.3, costs=cm, sizing=sizing,
                                        execution=ex, close_at_end=cae)
                    key = f"{ck}|{ek}|{sizing}|{cae}"
                    for col in res.daily.columns:
                        out[f"{key}|daily|{col}"] = res.daily[col].to_numpy(dtype=float)
                    for col in ("shares", "price", "notional", "commission", "slippage",
                                "impact", "auction"):
                        out[f"{key}|ledger|{col}"] = res.ledger[col].to_numpy(dtype=float)
                    out[f"{key}|trades|pnl"] = res.trades["pnl"].to_numpy(dtype=float)
                    out[f"{key}|trades|bars"] = res.trades["bars_held"].to_numpy(dtype=float)
                    out[f"{key}|trades|dir"] = res.trades["direction"].to_numpy(dtype=float)
                    out[f"{key}|trades|eos"] = (res.trades["exit_reason"]
                                                == "end_of_sample").to_numpy(dtype=float)
    return out


ALL: dict[str, Callable[[], dict[str, np.ndarray]]] = {
    "signal": signal_cases,
    "port": port_cases,
    "algo": algo_cases,
    "regime": regime_cases,
    "pairs": pairs_cases,
}
