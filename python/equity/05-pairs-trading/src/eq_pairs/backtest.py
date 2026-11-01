"""Event-driven daily backtester for a portfolio of pairs, plus walk-forward.

Execution timing
----------------
Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost. ``ExecutionConfig`` selects
the mode. The synthetic data has closes only, so the t+1 fill is modelled
as a point on the path from close t to close t+1: f = 1 is the next close,
small f approximates a next-open fill. The engine never fills a decision
taken at t at the close of t except in the close-auction mode, which charges
``auction_cost_bp`` on every traded dollar for doing so.

Accounting conventions (all dollars), for day t with fill price p_fill:

* Gross P&L: holdings carried into day t earn q_old (p_fill - p_{t-1});
  the holdings after the fill earn q_new (p_t - p_fill). With no trade
  this is q (p_t - p_{t-1}).
* Commission: ``cost_bps`` per leg on traded notional |shares| x p_fill.
* Slippage: ``slippage_bps`` per leg on traded notional (fixed part).
* Impact: square-root law, ``impact_coef * sigma_leg * sqrt(notional /
  adv_dollars)`` as a fraction of traded notional, per leg. ``sigma_leg`` is
  the leg's daily return volatility known before the fill.
* Borrow: short legs carried overnight into day t accrue ``borrow_bps``
  (annualised, ACT/252) on their market value at p_fill.
* Financing: long legs carried overnight pay ``financing_bps`` (annual);
  short proceeds earn ``short_rebate_bps`` (annual).
* Auction: in close-auction mode, ``auction_cost_bp`` per leg on traded
  notional.
* The identity net = gross - commission - slippage - impact - borrow -
  financing - auction holds to the penny and is asserted in tests.
* Dividends: the synthetic prices are total-return prices, so a dividend on
  the short leg is already inside the price P&L.
* Positions are sized once at entry (gross dollar target) and not rebalanced
  while the trade is on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from qp_core.execution import ExecutionConfig

from .cointegration import engle_granger
from .signals import SignalRules, generate_signals, size_positions, time_stop_bars
from .spread import OUFit, compute_spread, fit_ou_ols

__all__ = [
    "CostModel",
    "ExecutionConfig",
    "PairResult",
    "PortfolioResult",
    "align_pair",
    "backtest_pair",
    "backtest_portfolio",
    "WalkForwardWindow",
    "walk_forward_windows",
    "walk_forward_pair",
    "walk_forward_portfolio",
    "COST_COLUMNS",
]

COST_COLUMNS = ["commission", "slippage", "impact", "borrow", "financing", "auction"]



DEFAULT_EXECUTION = ExecutionConfig()


@dataclass(frozen=True)
class CostModel:
    """Transaction, impact, borrow and financing assumptions.

    Attributes
    ----------
    cost_bps : float
        Commission + fees per leg, bp of traded notional.
    slippage_bps : float
        Fixed slippage per leg, bp of traded notional.
    borrow_bps : float
        Annualised stock-borrow fee on short market value, bp, ACT/252.
    days_per_year : int
        Accrual basis for borrow and financing (default 252).
    impact_coef : float
        Square-root impact coefficient Y (cost fraction = Y sigma
        sqrt(Q/ADV)). 0 switches impact off. Published estimates are of
        order 0.5 to 1.
    adv_dollars : float
        Average daily dollar volume assumed for every name.
    financing_bps : float
        Annual financing charged on long market value carried overnight.
    short_rebate_bps : float
        Annual rebate earned on short proceeds carried overnight.
    """

    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    borrow_bps: float = 50.0
    days_per_year: int = 252
    impact_coef: float = 0.0
    adv_dollars: float = 50_000_000.0
    financing_bps: float = 0.0
    short_rebate_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("cost_bps", "slippage_bps", "borrow_bps", "impact_coef",
                     "financing_bps", "short_rebate_bps"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {v}")
        if not math.isfinite(self.adv_dollars) or self.adv_dollars <= 0:
            raise ValueError(f"adv_dollars must be finite and positive, got {self.adv_dollars}")
        if self.days_per_year <= 0:
            raise ValueError(f"days_per_year must be positive, got {self.days_per_year}")

    @property
    def daily_borrow_rate(self) -> float:
        return self.borrow_bps / 1e4 / self.days_per_year

    @property
    def daily_financing_rate(self) -> float:
        return self.financing_bps / 1e4 / self.days_per_year

    @property
    def daily_rebate_rate(self) -> float:
        return self.short_rebate_bps / 1e4 / self.days_per_year


ZERO_COSTS = CostModel(cost_bps=0.0, slippage_bps=0.0, borrow_bps=0.0)


@dataclass
class PairResult:
    """Backtest output for one pair.

    Attributes
    ----------
    name : str
        Pair label, e.g. "CO0_Y/CO0_X".
    daily : DataFrame
        Index = dates. Columns: gross_pnl, commission, slippage, borrow,
        net_pnl, position (executed direction, end of day), q_y, q_x,
        gross_exposure, net_exposure.
    ledger : DataFrame
        One row per leg execution: date, leg, shares, price, notional,
        commission, slippage.
    trades : DataFrame
        One row per round trip: entry_date, exit_date, direction, bars_held,
        pnl (net), exit_reason.
    """

    name: str
    daily: pd.DataFrame
    ledger: pd.DataFrame
    trades: pd.DataFrame

    @property
    def net_pnl(self) -> float:
        return float(self.daily["net_pnl"].sum())

    @property
    def gross_pnl(self) -> float:
        return float(self.daily["gross_pnl"].sum())

    @property
    def total_costs(self) -> float:
        return float(
            self.daily[COST_COLUMNS].to_numpy().sum()
        )


_LEDGER_COLS = ["date", "leg", "shares", "price", "notional", "commission", "slippage",
                "impact", "auction"]
_TRADE_COLS = ["entry_date", "exit_date", "direction", "bars_held", "pnl", "exit_reason"]


def align_pair(
    y: pd.Series, x: pd.Series, policy: str = "ffill", limit: int = 5
) -> tuple[pd.Series, pd.Series]:
    """Align two price series and apply the missing-data policy.

    Policy "ffill": stale-forward-fill gaps up to ``limit`` consecutive
    days; if any NaN survives (long halt, mismatched listing period), raise
   : silent extrapolation over long gaps is how backtests go wrong.
    Policy "drop": keep only dates where both legs print.

    Returns
    -------
    (y_aligned, x_aligned) with identical DatetimeIndex.
    """
    if policy not in ("ffill", "drop"):
        raise ValueError(f"policy must be 'ffill' or 'drop', got {policy!r}")
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1)
    # trim leading/trailing rows where either leg has never/stopped trading
    df = df.loc[df.notna().all(axis=1).idxmax() :]
    last_valid = df.notna().all(axis=1)[::-1].idxmax()
    df = df.loc[:last_valid]
    if policy == "drop":
        df = df.dropna()
    else:
        df = df.ffill(limit=limit)
        if df.isna().any().any():
            n_bad = int(df.isna().any(axis=1).sum())
            raise ValueError(
                f"{n_bad} rows still missing after ffill(limit={limit}); "
                "gap too long: inspect the data or use policy='drop'"
            )
    if len(df) < 3:
        raise ValueError("fewer than 3 overlapping observations after alignment")
    return df["y"], df["x"]


def _trailing_daily_vol(p: np.ndarray, window: int = 60, min_obs: int = 20,
                        fallback: float = 0.02) -> np.ndarray:
    """sigma[t] = sd of simple returns ending at t-1 (known before day t's fill)."""
    r = pd.Series(p).pct_change()
    sig = r.rolling(window, min_periods=min_obs).std().shift(1).to_numpy()
    return np.where(np.isfinite(sig), sig, fallback)


def backtest_pair(
    y: pd.Series,
    x: pd.Series,
    target: pd.Series,
    beta: float,
    name: str = "pair",
    costs: CostModel = CostModel(),
    gross: float = 1_000_000.0,
    sizing: str = "dollar",
    close_at_end: bool = True,
    execution: ExecutionConfig = DEFAULT_EXECUTION,
    leg_vol: Optional[tuple[float, float]] = None,
) -> PairResult:
    """Backtest one pair under the decide-at-t, fill-at-modelled-t+1 rule.

    Parameters
    ----------
    y, x : pandas.Series
        Close prices (dollars), identical index (use :func:`align_pair`).
    target : pandas.Series
        Desired spread position in {-1, 0, +1} decided at each date t using
        information through t's close. In "next" mode the engine fills
        ``target_t`` on day t+1 at ``p_t + f (p_{t+1} - p_t)``. In
        "close_auction" mode it fills ``target_t`` at ``p_t`` and pays the
        auction cost. NaNs are treated as 0 (flat).
    beta : float
        Hedge ratio for beta-neutral sizing (ignored in dollar mode).
    costs : CostModel
    gross : float
        Gross dollar exposure per position at entry.
    sizing : {"dollar", "beta"}
        See :func:`eq_pairs.signals.size_positions`.
    close_at_end : bool
        Force-close any open position on the final bar (default True).
    execution : ExecutionConfig
        Fill timing and price (default: next mode, f = 0.5).
    leg_vol : (sigma_y, sigma_x), optional
        Daily return vol of each leg for the impact model. Default: trailing
        60-day estimate known before each fill (2% until 20 returns exist).
    """
    if not isinstance(execution, ExecutionConfig):
        raise TypeError("execution must be an ExecutionConfig")
    if not y.index.equals(x.index):
        raise ValueError("y and x indices differ; use align_pair first")
    if not y.index.equals(target.index):
        raise ValueError("target index must match the price index")
    n = len(y)
    if n < 2:
        raise ValueError("need at least 2 bars to backtest")
    if not math.isfinite(gross) or gross <= 0:
        raise ValueError(f"gross must be finite and positive, got {gross}")

    tgt = target.fillna(0).astype(int).to_numpy()
    bad = set(np.unique(tgt)) - {-1, 0, 1}
    if bad:
        raise ValueError(f"target contains invalid values {sorted(bad)}")
    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    if np.any(~np.isfinite(yv)) or np.any(~np.isfinite(xv)):
        raise ValueError("prices contain NaN/inf; use align_pair first")

    if leg_vol is None:
        sig_y = _trailing_daily_vol(yv)
        sig_x = _trailing_daily_vol(xv)
    else:
        if len(leg_vol) != 2 or not all(math.isfinite(v) and v >= 0 for v in leg_vol):
            raise ValueError(f"leg_vol must be two finite non-negative floats, got {leg_vol}")
        sig_y = np.full(n, float(leg_vol[0]))
        sig_x = np.full(n, float(leg_vol[1]))

    auction = execution.mode == "close_auction"
    f = 0.0 if auction else execution.fill_fraction
    auction_rate = execution.auction_cost_bp / 1e4 if auction else 0.0

    idx = y.index
    daily = {
        k: np.zeros(n)
        for k in ("gross_pnl", *COST_COLUMNS, "net_pnl", "q_y", "q_x",
                  "gross_exposure", "net_exposure")
    }
    position = np.zeros(n, dtype=int)
    ledger_rows: list[dict] = []
    trade_rows: list[dict] = []

    q_y = q_x = 0.0
    direction = 0
    open_trade: Optional[dict] = None

    def execute_leg(t: int, leg: str, d_shares: float, price: float, sigma: float) -> dict:
        notional = abs(d_shares) * price
        c = {
            "commission": notional * costs.cost_bps / 1e4,
            "slippage": notional * costs.slippage_bps / 1e4,
            "impact": notional * costs.impact_coef * sigma
            * math.sqrt(notional / costs.adv_dollars),
            "auction": notional * auction_rate,
        }
        if d_shares != 0.0:
            ledger_rows.append({"date": idx[t], "leg": leg, "shares": d_shares,
                                "price": price, "notional": notional, **c})
        return c

    for t in range(n):
        # 1) fill price for today's trade and the decision being filled
        if auction:
            pf_y, pf_x = yv[t], xv[t]
            desired = int(tgt[t])
        elif t == 0:
            pf_y, pf_x = yv[0], xv[0]
            desired = 0
        else:
            pf_y = yv[t - 1] + f * (yv[t] - yv[t - 1])
            pf_x = xv[t - 1] + f * (xv[t] - xv[t - 1])
            desired = int(tgt[t - 1])
        if close_at_end and t == n - 1:
            desired = 0

        # 2) carried holdings: price move up to the fill, overnight carry costs
        carried_pnl = carry_cost = 0.0
        if t > 0 and direction != 0:
            carried_pnl = q_y * (pf_y - yv[t - 1]) + q_x * (pf_x - xv[t - 1])
            short_mv = (-q_y * pf_y if q_y < 0 else 0.0) + (-q_x * pf_x if q_x < 0 else 0.0)
            long_mv = (q_y * pf_y if q_y > 0 else 0.0) + (q_x * pf_x if q_x > 0 else 0.0)
            daily["borrow"][t] = short_mv * costs.daily_borrow_rate
            daily["financing"][t] = (long_mv * costs.daily_financing_rate
                                     - short_mv * costs.daily_rebate_rate)
            carry_cost = daily["borrow"][t] + daily["financing"][t]

        # 3) trade at the fill price
        exit_cost = entry_cost = 0.0
        traded = desired != direction
        if traded:
            if direction != 0:
                for leg, q, p, sg in (("y", -q_y, pf_y, sig_y[t]), ("x", -q_x, pf_x, sig_x[t])):
                    c = execute_leg(t, leg, q, p, sg)
                    for k, v in c.items():
                        daily[k][t] += v
                    exit_cost += sum(c.values())
                q_y = q_x = 0.0
            if desired != 0:
                q_y, q_x = size_positions(pf_y, pf_x, desired, beta, gross=gross, mode=sizing)
                for leg, q, p, sg in (("y", q_y, pf_y, sig_y[t]), ("x", q_x, pf_x, sig_x[t])):
                    c = execute_leg(t, leg, q, p, sg)
                    for k, v in c.items():
                        daily[k][t] += v
                    entry_cost += sum(c.values())

        # 4) holdings after the fill earn the rest of the day
        new_pnl = q_y * (yv[t] - pf_y) + q_x * (xv[t] - pf_x)
        daily["gross_pnl"][t] = carried_pnl + new_pnl

        # 5) trade bookkeeping
        if traded:
            if open_trade is not None:
                open_trade["pnl"] += carried_pnl - carry_cost - exit_cost
                open_trade["exit_date"] = idx[t]
                open_trade["bars_held"] += 1
                last_signal = tgt[t] if auction else tgt[t - 1]
                open_trade["exit_reason"] = (
                    "end_of_sample" if (close_at_end and t == n - 1 and last_signal != 0)
                    else "signal"
                )
                trade_rows.append(open_trade)
                open_trade = None
            if desired != 0:
                open_trade = {"entry_date": idx[t], "exit_date": pd.NaT,
                              "direction": desired, "bars_held": 0,
                              "pnl": new_pnl - entry_cost, "exit_reason": ""}
            direction = desired
        elif open_trade is not None:
            open_trade["pnl"] += carried_pnl + new_pnl - carry_cost
            if t > 0:
                open_trade["bars_held"] += 1

        # 6) end-of-day state
        position[t] = direction
        daily["q_y"][t] = q_y
        daily["q_x"][t] = q_x
        daily["gross_exposure"][t] = abs(q_y) * yv[t] + abs(q_x) * xv[t]
        daily["net_exposure"][t] = q_y * yv[t] + q_x * xv[t]
        daily["net_pnl"][t] = daily["gross_pnl"][t] - sum(daily[k][t] for k in COST_COLUMNS)

    df = pd.DataFrame(daily, index=idx)
    df["position"] = position
    ledger = pd.DataFrame(ledger_rows, columns=_LEDGER_COLS)
    trades = pd.DataFrame(trade_rows, columns=_TRADE_COLS)
    return PairResult(name=name, daily=df, ledger=ledger, trades=trades)


@dataclass
class PortfolioResult:
    """Aggregate of several PairResults on a common calendar.

    Attributes
    ----------
    pairs : list of PairResult
    daily : DataFrame
        Summed daily columns across pairs (gross_pnl, commission, slippage,
        borrow, net_pnl, gross_exposure, net_exposure, n_positions).
    """

    pairs: list[PairResult]
    daily: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("PortfolioResult needs at least one PairResult")
        cols = ["gross_pnl", *COST_COLUMNS, "net_pnl", "gross_exposure", "net_exposure"]
        frames = [p.daily[cols] for p in self.pairs]
        idx = frames[0].index
        for f in frames[1:]:
            idx = idx.union(f.index)
        agg = sum(f.reindex(idx).fillna(0.0) for f in frames)
        agg["n_positions"] = sum(
            (p.daily["position"] != 0).astype(int).reindex(idx).fillna(0)
            for p in self.pairs
        )
        self.daily = agg

    @property
    def net_pnl(self) -> float:
        return float(self.daily["net_pnl"].sum())

    def attribution(self) -> pd.DataFrame:
        """Per-pair P&L attribution (gross, costs breakdown, net, trades)."""
        rows = []
        for p in self.pairs:
            rows.append(
                {
                    "pair": p.name,
                    "gross_pnl": p.gross_pnl,
                    "commission": float(p.daily["commission"].sum()),
                    "slippage": float(p.daily["slippage"].sum()),
                    "impact": float(p.daily["impact"].sum()),
                    "borrow": float(p.daily["borrow"].sum()),
                    "financing": float(p.daily["financing"].sum()),
                    "auction": float(p.daily["auction"].sum()),
                    "net_pnl": p.net_pnl,
                    "n_trades": len(p.trades),
                }
            )
        return pd.DataFrame(rows).set_index("pair").sort_values("net_pnl", ascending=False)


def backtest_portfolio(
    price_panel: pd.DataFrame,
    pair_targets: dict[tuple[str, str], pd.Series],
    betas: dict[tuple[str, str], float],
    costs: CostModel = CostModel(),
    gross_per_pair: float = 1_000_000.0,
    sizing: str = "dollar",
    execution: ExecutionConfig = DEFAULT_EXECUTION,
) -> PortfolioResult:
    """Backtest a set of pairs and aggregate.

    Parameters
    ----------
    price_panel : DataFrame
        Close prices, columns = tickers.
    pair_targets : dict (a, b) -> target Series
        Desired positions per pair (decided-at-t convention).
    betas : dict (a, b) -> hedge ratio.
    """
    if not pair_targets:
        raise ValueError("pair_targets is empty")
    results = []
    for (a, b), target in pair_targets.items():
        yv, xv = align_pair(price_panel[a], price_panel[b])
        common = yv.index.intersection(target.index)
        res = backtest_pair(
            yv.loc[common],
            xv.loc[common],
            target.loc[common],
            beta=betas[(a, b)],
            name=f"{a}/{b}",
            costs=costs,
            gross=gross_per_pair,
            sizing=sizing,
            execution=execution,
        )
        results.append(res)
    return PortfolioResult(pairs=results)


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WalkForwardWindow:
    """One formation -> trading split (all bounds inclusive, positional)."""

    formation_start: int
    formation_end: int
    trading_start: int
    trading_end: int

    def __post_init__(self) -> None:
        if not (
            self.formation_start
            <= self.formation_end
            < self.trading_start
            <= self.trading_end
        ):
            raise ValueError(f"malformed walk-forward window: {self}")


def walk_forward_windows(
    n_obs: int,
    formation: int,
    trading: int,
    step: Optional[int] = None,
    allow_partial: bool = False,
) -> list[WalkForwardWindow]:
    """Rolling formation/trading splits over ``n_obs`` observations.

    formation window = [i, i+formation), trading window =
    [i+formation, i+formation+trading), stepping by ``step`` (default =
    ``trading``, i.e. contiguous non-overlapping trading windows). With
    ``allow_partial`` (and the default step) the trailing observations that
    do not fill a whole trading window form one shorter last window instead
    of being dropped. The
    formation and trading windows of each split NEVER overlap: asserted in
    the constructor and unit-tested.
    """
    if formation < 30:
        raise ValueError(f"formation must be >= 30 obs, got {formation}")
    if trading < 2:
        raise ValueError(f"trading must be >= 2 obs, got {trading}")
    step = trading if step is None else step
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    windows = []
    i = 0
    while i + formation + trading <= n_obs:
        windows.append(
            WalkForwardWindow(
                formation_start=i,
                formation_end=i + formation - 1,
                trading_start=i + formation,
                trading_end=i + formation + trading - 1,
            )
        )
        i += step
    if allow_partial and step == trading:
        start = (windows[-1].trading_end + 1) if windows else formation
        if n_obs - start >= 2 and start - formation >= 0:
            windows.append(
                WalkForwardWindow(
                    formation_start=start - formation,
                    formation_end=start - 1,
                    trading_start=start,
                    trading_end=n_obs - 1,
                )
            )
    return windows


def _fit_formation(
    y_f: pd.Series,
    x_f: pd.Series,
    eg_level: str = "5%",
    min_half_life: float = 1.0,
    max_half_life: float = 126.0,
) -> Optional[dict]:
    """Fit EG + OU on a formation window; None if the pair fails any gate.

    Gates: EG rejects no-cointegration at ``eg_level``; the OU fit is
    mean-reverting with half-life in [min_half_life, max_half_life] days.
    """
    try:
        eg = engle_granger(y_f.to_numpy(), x_f.to_numpy())
    except ValueError:
        return None
    if not eg.cointegrated(eg_level):
        return None
    ou = fit_ou_ols(eg.resid)
    if not ou.mean_reverting or not (
        min_half_life <= ou.half_life <= max_half_life
    ):
        return None
    return {
        "beta": eg.beta,
        "alpha": eg.alpha,
        "mu": ou.mu,
        "stat_std": ou.stationary_std,
        "half_life": ou.half_life,
        "kappa": ou.kappa,
        "adf_stat": eg.stat,
    }


def _formation_vol(y_f: pd.Series, x_f: pd.Series) -> tuple[float, float]:
    """Daily return vol of each leg over the formation window (for impact)."""
    return (float(y_f.pct_change().std()), float(x_f.pct_change().std()))


def _frozen_z(y: pd.Series, x: pd.Series, params: dict) -> pd.Series:
    """Z-score of the spread using FROZEN formation parameters only."""
    s = compute_spread(y, x, params["beta"], params["alpha"])
    return (s - params["mu"]) / params["stat_std"]


def walk_forward_pair(
    y: pd.Series,
    x: pd.Series,
    formation: int = 252,
    trading: int = 63,
    rules: Optional[SignalRules] = None,
    costs: CostModel = CostModel(),
    gross: float = 1_000_000.0,
    sizing: str = "dollar",
    time_stop_k: float = 3.0,
    eg_level: str = "5%",
    name: str = "pair",
    execution: ExecutionConfig = DEFAULT_EXECUTION,
    allow_partial: bool = False,
) -> tuple[Optional[PairResult], pd.DataFrame]:
    """Walk-forward backtest of a single pair.

    Each split: fit hedge ratio + OU on the formation window; if the pair
    passes the cointegration and half-life gates, trade the next ``trading``
    bars with ALL parameters frozen at their formation values (beta, alpha,
    mu, stationary std, time stop). Positions are force-closed at each
    trading-window end.

    Returns
    -------
    (result, windows) : PairResult or None if no window traded, plus a
    DataFrame of per-window records (bounds, frozen params, traded flag)
    that tests use to verify the freeze and non-overlap properties.
    """
    y, x = align_pair(y, x)
    wins = walk_forward_windows(len(y), formation, trading, allow_partial=allow_partial)
    if not wins:
        raise ValueError(
            f"sample of {len(y)} obs too short for formation={formation}, "
            f"trading={trading}"
        )
    daily_frames = []
    ledger_frames = []
    trade_frames = []
    records = []
    for w in wins:
        y_f = y.iloc[w.formation_start : w.formation_end + 1]
        x_f = x.iloc[w.formation_start : w.formation_end + 1]
        y_t = y.iloc[w.trading_start : w.trading_end + 1]
        x_t = x.iloc[w.trading_start : w.trading_end + 1]
        params = _fit_formation(y_f, x_f, eg_level=eg_level)
        rec = {
            "formation_start": y.index[w.formation_start],
            "formation_end": y.index[w.formation_end],
            "trading_start": y.index[w.trading_start],
            "trading_end": y.index[w.trading_end],
            "traded": params is not None,
        }
        if params is not None:
            rec.update(params)
            wr = rules or SignalRules(
                max_holding=time_stop_bars(params["half_life"], k=time_stop_k)
            )
            z = _frozen_z(y_t, x_t, params)
            target = generate_signals(z, wr)["position"]
            res = backtest_pair(
                y_t,
                x_t,
                target,
                beta=params["beta"],
                name=name,
                costs=costs,
                gross=gross,
                sizing=sizing,
                close_at_end=True,
                execution=execution,
                leg_vol=_formation_vol(y_f, x_f),
            )
            daily_frames.append(res.daily)
            if len(res.ledger):
                ledger_frames.append(res.ledger)
            if len(res.trades):
                trade_frames.append(res.trades)
        records.append(rec)
    windows_df = pd.DataFrame(records)
    if not daily_frames:
        return None, windows_df
    daily = pd.concat(daily_frames)
    daily = daily[~daily.index.duplicated(keep="first")].sort_index()
    ledger = (
        pd.concat(ledger_frames, ignore_index=True)
        if ledger_frames
        else pd.DataFrame(columns=_LEDGER_COLS)
    )
    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(columns=_TRADE_COLS)
    )
    return PairResult(name=name, daily=daily, ledger=ledger, trades=trades), windows_df


def walk_forward_portfolio(
    prices: pd.DataFrame,
    candidate_pairs: Sequence[tuple[str, str]],
    formation: int = 252,
    trading: int = 63,
    max_pairs: int = 10,
    min_corr: float = 0.60,
    rules: Optional[SignalRules] = None,
    costs: CostModel = CostModel(),
    gross_per_pair: float = 1_000_000.0,
    sizing: str = "dollar",
    eg_level: str = "5%",
    execution: ExecutionConfig = DEFAULT_EXECUTION,
    allow_partial: bool = False,
    hedge: str = "static",
    hedge_window: int = 60,
    screen: str = "corr",
    ssd_top_n: int = 20,
    rule_overrides: Optional[dict] = None,
) -> tuple[Optional[PortfolioResult], pd.DataFrame]:
    """Walk-forward portfolio: re-select and re-fit pairs each formation window.

    Per split: screen candidates on the formation window, run EG + OU gates,
    rank survivors by ADF statistic (most negative first), trade up to
    ``max_pairs`` in the trading window with frozen gate parameters.

    screen : {"corr", "ssd"}
        "corr" keeps pairs with formation return correlation >= ``min_corr``;
        "ssd" keeps the ``ssd_top_n`` pairs with the smallest sum of squared
        differences of normalised formation prices (Gatev et al. distance).
    hedge : {"static", "rolling", "rls"}
        "static": spread and z-score from the frozen formation OLS + OU fit.
        "rolling": hedge ratio from rolling OLS over ``hedge_window`` days
        and a rolling z-score over the same window, both using data up to t
        only. "rls": recursive least squares, forgetting factor
        ``1 - 1/hedge_window``, no intercept, with the same rolling z-score.
        The EG and half-life gates are the same in all three.
    rule_overrides : dict, optional
        Extra ``SignalRules`` fields (for example ``entry_z``); the time stop
        stays 3 x the formation half-life unless overridden.

    Returns
    -------
    (PortfolioResult or None, per-window selection DataFrame).
    """
    from .signals import zscore_rolling
    from .spread import rls_hedge_ratio, rolling_ols_hedge_ratio
    from .universe import correlation_screen, ssd_screen  # local import to avoid cycle

    if hedge not in ("static", "rolling", "rls"):
        raise ValueError(f"hedge must be 'static', 'rolling' or 'rls', got {hedge!r}")
    if screen not in ("corr", "ssd"):
        raise ValueError(f"screen must be 'corr' or 'ssd', got {screen!r}")
    if hedge_window < 10 or hedge_window >= formation:
        raise ValueError(f"hedge_window must be in [10, formation), got {hedge_window}")
    if rules is not None and rule_overrides:
        raise ValueError("pass either rules or rule_overrides, not both")

    wins = walk_forward_windows(len(prices), formation, trading, allow_partial=allow_partial)
    if not wins:
        raise ValueError("sample too short for the requested windows")
    per_pair_frames: dict[str, list[PairResult]] = {}
    records = []
    for w in wins:
        pf = prices.iloc[w.formation_start : w.formation_end + 1]
        pt = prices.iloc[w.trading_start : w.trading_end + 1]
        if screen == "corr":
            surv = correlation_screen(pf, list(candidate_pairs), min_corr=min_corr)
        else:
            surv = ssd_screen(pf, list(candidate_pairs), top_n=ssd_top_n)
        fitted = []
        for a, b in surv.index:
            params = _fit_formation(pf[a], pf[b], eg_level=eg_level)
            if params is not None:
                fitted.append(((a, b), params))
        fitted.sort(key=lambda kv: kv[1]["adf_stat"])
        fitted = fitted[:max_pairs]
        records.append(
            {
                "trading_start": prices.index[w.trading_start],
                "trading_end": prices.index[w.trading_end],
                "n_candidates": len(candidate_pairs),
                "n_corr_survivors": len(surv),
                "n_traded": len(fitted),
            }
        )
        for (a, b), params in fitted:
            wr = rules or SignalRules(
                **{"max_holding": time_stop_bars(params["half_life"], k=3.0),
                   **(rule_overrides or {})}
            )
            if hedge == "static":
                z = _frozen_z(pt[a], pt[b], params)
            else:
                both = prices.iloc[w.formation_start : w.trading_end + 1]
                if hedge == "rolling":
                    hb = rolling_ols_hedge_ratio(both[a], both[b], window=hedge_window)
                    spread = both[a] - hb["alpha"] - hb["beta"] * both[b]
                else:
                    hb = rls_hedge_ratio(both[a], both[b], lam=1.0 - 1.0 / hedge_window,
                                         intercept=False)
                    spread = both[a] - hb["beta"] * both[b]
                z = zscore_rolling(spread, window=hedge_window).loc[pt.index]
            target = generate_signals(z, wr)["position"]
            res = backtest_pair(
                pt[a],
                pt[b],
                target,
                beta=params["beta"],
                name=f"{a}/{b}",
                costs=costs,
                gross=gross_per_pair,
                sizing=sizing,
                execution=execution,
                leg_vol=_formation_vol(pf[a], pf[b]),
            )
            per_pair_frames.setdefault(f"{a}/{b}", []).append(res)
    windows_df = pd.DataFrame(records)
    if not per_pair_frames:
        return None, windows_df
    merged: list[PairResult] = []
    for name, parts in per_pair_frames.items():
        daily = pd.concat([p.daily for p in parts])
        daily = daily[~daily.index.duplicated(keep="first")].sort_index()
        ledger = pd.concat(
            [p.ledger for p in parts if len(p.ledger)], ignore_index=True
        ) if any(len(p.ledger) for p in parts) else pd.DataFrame(columns=_LEDGER_COLS)
        trades = pd.concat(
            [p.trades for p in parts if len(p.trades)], ignore_index=True
        ) if any(len(p.trades) for p in parts) else pd.DataFrame(columns=_TRADE_COLS)
        merged.append(PairResult(name=name, daily=daily, ledger=ledger, trades=trades))
    return PortfolioResult(pairs=merged), windows_df
