"""Event-driven daily backtester for a portfolio of pairs, plus walk-forward.

Execution timing — THE no-lookahead rule
----------------------------------------
A signal value at date t is a statistic of data through t's close. The
engine therefore executes the position decided at t at the NEXT bar's close
(t+1): ``executed_t = target_{t-1}``. We chose "t-1 signal, t close" over
"t close signal, t+1 open" because the synthetic data (and most daily
research data) has closes only; the choice is documented here, enforced
structurally (the engine physically cannot see ``target_t`` when trading at
t), and proven by a lookahead-detector test: a spread engineered so that
same-day execution is profitable and lagged execution loses — the engine
must produce the losing (honest) number (tests/test_backtest.py).

Accounting conventions (all dollars):

* Mark-to-market: holdings carried into day t earn
  q . (P_t - P_{t-1}) (gross P&L), BEFORE any rebalancing at t's close.
* Rebalancing happens at t's close at the close price; commissions and
  slippage are charged as explicit cash so the identity
  net = gross - commission - slippage - borrow holds to the penny and is
  asserted in tests.
* Commission: ``cost_bps`` per leg on traded notional |shares| x close.
* Slippage: ``slippage_bps`` per leg on traded notional (a fixed-impact
  model; see docs/METHODOLOGY.md for its limits).
* Borrow: short legs accrue ``borrow_bps`` (annualised, ACT/252) on the
  short market value at today's close, for positions held overnight into
  today. No borrow is charged on the entry day.
* Positions are sized once at entry (gross dollar target) and NOT rebalanced
  daily while the trade is on — turnover stays interpretable and the ledger
  maps 1:1 to entries/exits/flips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .cointegration import engle_granger
from .signals import SignalRules, generate_signals, size_positions, time_stop_bars
from .spread import OUFit, compute_spread, fit_ou_ols

__all__ = [
    "CostModel",
    "PairResult",
    "PortfolioResult",
    "align_pair",
    "backtest_pair",
    "backtest_portfolio",
    "WalkForwardWindow",
    "walk_forward_windows",
    "walk_forward_pair",
    "walk_forward_portfolio",
]


@dataclass(frozen=True)
class CostModel:
    """Transaction-cost assumptions.

    Attributes
    ----------
    cost_bps : float
        Commission + fees per leg, in basis points of traded notional.
    slippage_bps : float
        Price impact per leg, bps of traded notional (fixed-impact model).
    borrow_bps : float
        Annualised stock-borrow fee on short market value, bps, ACT/252.
    days_per_year : int
        Accrual basis for borrow (default 252).
    """

    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    borrow_bps: float = 50.0
    days_per_year: int = 252

    def __post_init__(self) -> None:
        for name in ("cost_bps", "slippage_bps", "borrow_bps"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")
        if self.days_per_year <= 0:
            raise ValueError(f"days_per_year must be positive, got {self.days_per_year}")

    @property
    def daily_borrow_rate(self) -> float:
        return self.borrow_bps / 1e4 / self.days_per_year


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
            self.daily[["commission", "slippage", "borrow"]].to_numpy().sum()
        )


_LEDGER_COLS = ["date", "leg", "shares", "price", "notional", "commission", "slippage"]
_TRADE_COLS = ["entry_date", "exit_date", "direction", "bars_held", "pnl", "exit_reason"]


def align_pair(
    y: pd.Series, x: pd.Series, policy: str = "ffill", limit: int = 5
) -> tuple[pd.Series, pd.Series]:
    """Align two price series and apply the missing-data policy.

    Policy "ffill": stale-forward-fill gaps up to ``limit`` consecutive
    days; if any NaN survives (long halt, mismatched listing period), raise
    — silent extrapolation over long gaps is how backtests go wrong.
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
                "gap too long — inspect the data or use policy='drop'"
            )
    if len(df) < 3:
        raise ValueError("fewer than 3 overlapping observations after alignment")
    return df["y"], df["x"]


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
) -> PairResult:
    """Backtest one pair under the strict t-1 signal / t close execution rule.

    Parameters
    ----------
    y, x : pandas.Series
        Close prices (dollars), identical index (use :func:`align_pair`).
    target : pandas.Series
        Desired spread position in {-1, 0, +1} DECIDED at each date t using
        information through t (e.g. ``generate_signals(...)["position"]``).
        The engine executes ``target_{t-1}`` at t's close — it never reads
        ``target_t`` on day t. NaNs are treated as 0 (flat).
    beta : float
        Hedge ratio for beta-neutral sizing (ignored in dollar mode).
    costs : CostModel
    gross : float
        Gross dollar exposure per position at entry.
    sizing : {"dollar", "beta"}
        See :func:`eq_pairs.signals.size_positions`.
    close_at_end : bool
        Force-close any open position at the final bar (default True).

    Returns
    -------
    PairResult
    """
    if not y.index.equals(x.index):
        raise ValueError("y and x indices differ; use align_pair first")
    if not y.index.equals(target.index):
        raise ValueError("target index must match the price index")
    n = len(y)
    if n < 2:
        raise ValueError("need at least 2 bars to backtest")

    tgt = target.fillna(0).astype(int).to_numpy()
    bad = set(np.unique(tgt)) - {-1, 0, 1}
    if bad:
        raise ValueError(f"target contains invalid values {sorted(bad)}")
    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    if np.any(~np.isfinite(yv)) or np.any(~np.isfinite(xv)):
        raise ValueError("prices contain NaN/inf; use align_pair first")

    idx = y.index
    daily = {
        k: np.zeros(n)
        for k in (
            "gross_pnl",
            "commission",
            "slippage",
            "borrow",
            "net_pnl",
            "q_y",
            "q_x",
            "gross_exposure",
            "net_exposure",
        )
    }
    position = np.zeros(n, dtype=int)
    ledger_rows: list[dict] = []
    trade_rows: list[dict] = []

    q_y = q_x = 0.0
    direction = 0
    open_trade: Optional[dict] = None

    def execute_leg(t: int, leg: str, d_shares: float, price: float) -> tuple[float, float]:
        if d_shares == 0.0:
            return 0.0, 0.0
        notional = abs(d_shares) * price
        commission = notional * costs.cost_bps / 1e4
        slippage = notional * costs.slippage_bps / 1e4
        ledger_rows.append(
            {
                "date": idx[t],
                "leg": leg,
                "shares": d_shares,
                "price": price,
                "notional": notional,
                "commission": commission,
                "slippage": slippage,
            }
        )
        return commission, slippage

    for t in range(n):
        # 1) mark-to-market holdings carried into today
        if t > 0 and direction != 0:
            daily["gross_pnl"][t] = q_y * (yv[t] - yv[t - 1]) + q_x * (
                xv[t] - xv[t - 1]
            )
            short_mv = 0.0
            if q_y < 0:
                short_mv += -q_y * yv[t]
            if q_x < 0:
                short_mv += -q_x * xv[t]
            daily["borrow"][t] = short_mv * costs.daily_borrow_rate

        # 2) desired position: strictly yesterday's decision
        desired = int(tgt[t - 1]) if t > 0 else 0
        if close_at_end and t == n - 1:
            desired = 0

        # 3) rebalance at today's close
        if desired != direction:
            comm = slip = 0.0
            if direction != 0:  # close existing
                c1, s1 = execute_leg(t, "y", -q_y, yv[t])
                c2, s2 = execute_leg(t, "x", -q_x, xv[t])
                comm += c1 + c2
                slip += s1 + s2
                q_y = q_x = 0.0
            entry_comm_slip = 0.0
            if desired != 0:  # open new
                q_y, q_x = size_positions(
                    yv[t], xv[t], desired, beta, gross=gross, mode=sizing
                )
                c1, s1 = execute_leg(t, "y", q_y, yv[t])
                c2, s2 = execute_leg(t, "x", q_x, xv[t])
                comm += c1 + c2
                slip += s1 + s2
                entry_comm_slip = c1 + s1 + c2 + s2
            daily["commission"][t] += comm
            daily["slippage"][t] += slip
            # trade bookkeeping: today's MTM and exit costs belong to the
            # closing trade; entry costs to the new one
            if open_trade is not None:
                open_trade["pnl"] += (
                    daily["gross_pnl"][t]
                    - daily["borrow"][t]
                    + entry_comm_slip  # exclude the new trade's entry costs
                    - (comm + slip)
                )
                open_trade["exit_date"] = idx[t]
                open_trade["bars_held"] += 1
                open_trade["exit_reason"] = (
                    "end_of_sample" if (close_at_end and t == n - 1 and tgt[t - 1] != 0) else "signal"
                )
                trade_rows.append(open_trade)
                open_trade = None
            if desired != 0:
                open_trade = {
                    "entry_date": idx[t],
                    "exit_date": pd.NaT,
                    "direction": desired,
                    "bars_held": 0,
                    "pnl": -entry_comm_slip,
                    "exit_reason": "",
                }
            direction = desired
        else:
            if open_trade is not None:
                open_trade["pnl"] += daily["gross_pnl"][t] - daily["borrow"][t]
                if t > 0:
                    open_trade["bars_held"] += 1

        # 4) end-of-day state
        position[t] = direction
        daily["q_y"][t] = q_y
        daily["q_x"][t] = q_x
        daily["gross_exposure"][t] = abs(q_y) * yv[t] + abs(q_x) * xv[t]
        daily["net_exposure"][t] = q_y * yv[t] + q_x * xv[t]
        daily["net_pnl"][t] = (
            daily["gross_pnl"][t]
            - daily["commission"][t]
            - daily["slippage"][t]
            - daily["borrow"][t]
        )

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
        cols = ["gross_pnl", "commission", "slippage", "borrow", "net_pnl",
                "gross_exposure", "net_exposure"]
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
                    "borrow": float(p.daily["borrow"].sum()),
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
    n_obs: int, formation: int, trading: int, step: Optional[int] = None
) -> list[WalkForwardWindow]:
    """Rolling formation/trading splits over ``n_obs`` observations.

    formation window = [i, i+formation), trading window =
    [i+formation, i+formation+trading), stepping by ``step`` (default =
    ``trading``, i.e. contiguous non-overlapping trading windows). The
    formation and trading windows of each split NEVER overlap — asserted in
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
    wins = walk_forward_windows(len(y), formation, trading)
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
) -> tuple[Optional[PortfolioResult], pd.DataFrame]:
    """Walk-forward portfolio: re-select and re-fit pairs each formation window.

    Per split: correlation-screen candidates on formation returns, run EG +
    OU gates, rank survivors by ADF statistic (most negative first), trade
    up to ``max_pairs`` in the trading window with frozen parameters.

    Returns
    -------
    (PortfolioResult or None, per-window selection DataFrame).
    """
    from .universe import correlation_screen  # local import to avoid cycle

    wins = walk_forward_windows(len(prices), formation, trading)
    if not wins:
        raise ValueError("sample too short for the requested windows")
    per_pair_frames: dict[str, list[PairResult]] = {}
    records = []
    for w in wins:
        pf = prices.iloc[w.formation_start : w.formation_end + 1]
        pt = prices.iloc[w.trading_start : w.trading_end + 1]
        surv = correlation_screen(pf, list(candidate_pairs), min_corr=min_corr)
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
                max_holding=time_stop_bars(params["half_life"], k=3.0)
            )
            z = _frozen_z(pt[a], pt[b], params)
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
