"""End-to-end pipeline: synthetic data, in-sample parameter selection,
out-of-sample backtest, fill and cost variants, deflated Sharpe and PBO,
walk-forward validation, parameter sensitivity, and a multi-seed
distribution of every headline number.

Runs offline on the bundled synthetic generator. Every number it prints is
a property of that generator, not of any market.

Usage
-----
    python examples/run_pipeline.py                  # bundled synthetic data
    python examples/run_pipeline.py --seeds 20       # fewer seeds, faster
    python examples/run_pipeline.py --out DIR        # where to write files
    python examples/run_pipeline.py --ticker SPY     # real data (needs
                                                     # `pip install
                                                     # eq-signal-backtest[live]`
                                                     # and network access)

Outputs: ``report.txt`` and ``figures/*.png`` under ``--out`` (default: a
new directory under the system temp dir). Nothing is written into the
repository.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from eq_signal_backtest import (
    TRADING_DAYS,
    ExecutionConfig,
    run_backtest,
    strategy_returns,
)
from eq_signal_backtest.data.synthetic import _DEFAULT_SEED, generate
from eq_signal_backtest.research_stats import (
    TrialLog,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    lo_sharpe_se,
    pbo_cscv,
    seed_distribution,
    sharpe_ratio,
)
from eq_signal_backtest.signals import ma_crossover_signal
from eq_signal_backtest.split import (
    select_best_params,
    train_test_split,
    walk_forward_backtest,
)

FAST_RANGE = range(10, 71, 10)
SLOW_RANGE = range(100, 251, 25)
COST_BPS = 5.0
N_SEEDS = 100
FORMATION, TRADING = 756, 252  # about 3y formation, 1y trading


def default_out_dir() -> Path:
    """A fresh directory under the system temp dir, never inside the repo."""
    return Path(tempfile.mkdtemp(prefix="eq_signal_backtest_"))


def source_label(ticker: str | None) -> str:
    return ticker if ticker else f"bundled synthetic (seed={_DEFAULT_SEED})"


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def fmt(stats: dict) -> str:
    return (
        f"CAGR {stats['cagr']:+.2%} | vol {stats['volatility']:.2%} | "
        f"Sharpe {stats['sharpe']:.2f} | maxDD {stats['max_drawdown']:.2%}"
    )


def load_prices(source: str | None) -> pd.Series:
    if source is None:
        df = generate()
        return df.set_index("Date")["Adj Close"].astype(float)
    from eq_signal_backtest.data.live import load_prices as _load_live

    return _load_live(source)


def one_seed(seed: int) -> dict:
    """Selection, out-of-sample and walk-forward Sharpe on one synthetic path."""
    px = generate(seed=seed).set_index("Date")["Adj Close"].astype(float)
    sp = train_test_split(px, train_frac=0.7)
    f, s, _ = select_best_params(sp.train, FAST_RANGE, SLOW_RANGE, cost_bps=COST_BPS)
    res_is = run_backtest(sp.train, ma_crossover_signal(sp.train, f, s), COST_BPS)
    res_oos = run_backtest(sp.test, ma_crossover_signal(sp.test, f, s), COST_BPS)
    wf = walk_forward_backtest(px, FAST_RANGE, SLOW_RANGE, FORMATION, TRADING, COST_BPS)
    return {
        "seed": seed,
        "is": res_is.stats["sharpe"],
        "oos": res_oos.stats["sharpe"],
        "oos_bh": res_oos.stats["benchmark"]["sharpe"],
        "wf": wf.stats["sharpe"],
        "wf_bh": wf.stats["benchmark"]["sharpe"],
    }


def main(ticker: str | None = None, out_dir: Path | None = None,
         n_seeds: int = N_SEEDS) -> None:
    t0 = time.time()
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    add = lines.append

    # ---------------------------------------------------------------- 1
    hr("1) DATA")
    prices = load_prices(ticker)
    label = source_label(ticker)
    print(f"source: {label}")
    print(f"{len(prices)} rows, {prices.index[0].date()} .. {prices.index[-1].date()}")
    add("=" * 72)
    add("TRADING SIGNAL BACKTEST: MA CROSSOVER (long/flat)")
    add(f"Data: {label} | {prices.index[0].date()} to {prices.index[-1].date()}")
    add("Fill: decision at close t, fill at p_t + 0.5 (p_{t+1} - p_t), 5 bp per trade")
    add("=" * 72)

    # ---------------------------------------------------------------- 2
    hr("2) TRAIN/TEST SPLIT (70/30) AND IN-SAMPLE PARAMETER SELECTION")
    split = train_test_split(prices, train_frac=0.7)
    best_fast, best_slow, grid = select_best_params(
        split.train, FAST_RANGE, SLOW_RANGE, cost_bps=COST_BPS
    )
    add(f"\nTrain/test split at {split.split_date.date()} (70/30)")
    add(f"Parameters selected on training data: fast={best_fast}, slow={best_slow}")
    add("(Grid-search argmax on the training window. The search itself is a")
    add("source of selection bias; the deflated Sharpe below accounts for it.)")
    print("\n".join(lines[-3:]))

    # ---------------------------------------------------------------- 3
    hr("3) IN-SAMPLE / OUT-OF-SAMPLE / FULL-PERIOD / FILL AND COST VARIANTS")

    def evaluate(px: pd.Series, cost_bps: float = COST_BPS, execution=None):
        sig = ma_crossover_signal(px, best_fast, best_slow)
        return run_backtest(px, sig, cost_bps=cost_bps, execution=execution)

    res_train = evaluate(split.train)
    res_test = evaluate(split.test)
    res_full = evaluate(prices)
    res_free = evaluate(prices, cost_bps=0.0)
    res_next1 = evaluate(prices, execution=ExecutionConfig("next", 1.0))
    res_auc = evaluate(prices, execution=ExecutionConfig("close_auction"))

    add("\n--- Performance (one synthetic path) ---")
    add(f"In-sample   strategy : {fmt(res_train.stats)}")
    add(f"In-sample   buy&hold : {fmt(res_train.stats['benchmark'])}")
    add(f"Out-sample  strategy : {fmt(res_test.stats)}")
    add(f"Out-sample  buy&hold : {fmt(res_test.stats['benchmark'])}")
    add(f"Full period strategy : {fmt(res_full.stats)}  ({res_full.n_trades} trades)")
    add(f"Full, zero costs     : {fmt(res_free.stats)}")
    add(f"Full, next-close fill: {fmt(res_next1.stats)}")
    add(f"Full, close auction  : {fmt(res_auc.stats)}  (10 bp auction cost)")
    cost_drag = res_free.stats["cagr"] - res_full.stats["cagr"]
    add(f"Transaction cost drag: {cost_drag:.2%} CAGR at {COST_BPS:.0f} bps per trade")
    print("\n".join(lines[-10:]))

    is_sh = res_train.stats["sharpe"]
    oos_sh = res_test.stats["sharpe"]
    oos_r, _, _ = strategy_returns(
        split.test, ma_crossover_signal(split.test, best_fast, best_slow), COST_BPS
    )
    se = lo_sharpe_se(oos_r.to_numpy(), TRADING_DAYS)
    _, b_lo, b_hi = block_bootstrap_ci(
        oos_r.to_numpy(), lambda x: sharpe_ratio(x, TRADING_DAYS), n_boot=1000, seed=1
    )
    add("\n--- One synthetic path: in-sample vs out-of-sample Sharpe ---")
    add(f"Sharpe in-sample -> out-of-sample: {is_sh:.2f} -> {oos_sh:.2f}")
    add(f"Out-of-sample Sharpe: Lo/Newey-West SE {se:.2f}; "
        f"block-bootstrap 95% CI [{b_lo:.2f}, {b_hi:.2f}]")

    # Every grid cell is a trial: log it, then deflate the selected Sharpe.
    log = TrialLog()
    grid_rets: dict[tuple[int, int], np.ndarray] = {}
    for fa in FAST_RANGE:
        for sl in SLOW_RANGE:
            if fa >= sl:
                continue
            r_, _, _ = strategy_returns(
                split.train, ma_crossover_signal(split.train, fa, sl), COST_BPS
            )
            x = r_.to_numpy()
            sd = x.std(ddof=1)
            log.log(f"ma_{fa}_{sl}", float(x.mean() / sd) if sd > 0 else None,
                    fast=fa, slow=sl)
            grid_rets[(fa, sl)] = x
    dsr = deflated_sharpe_ratio(grid_rets[(best_fast, best_slow)], log.n_trials,
                                log.sharpes())
    full_mat = np.column_stack([
        strategy_returns(prices, ma_crossover_signal(prices, fa, sl), COST_BPS)[0].to_numpy()
        for (fa, sl) in grid_rets
    ])
    pbo = pbo_cscv(full_mat[max(SLOW_RANGE):], n_splits=16)
    add(f"Trials logged in the grid search: {log.n_trials}")
    add(f"Deflated Sharpe of the selected in-sample result: {dsr:.3f}")
    add(f"PBO (CSCV, 16 blocks, full sample, {log.n_trials} configurations): {pbo:.2f}")
    print("\n".join(lines[-6:]))

    # ---------------------------------------------------------------- 4
    hr("4) WALK-FORWARD VALIDATION (re-select every window, trade frozen params)")
    wf = walk_forward_backtest(
        prices, FAST_RANGE, SLOW_RANGE, formation=FORMATION, trading=TRADING,
        cost_bps=COST_BPS,
    )
    print(wf.windows.to_string(index=False))
    add(f"\n--- Walk-forward validation (formation {FORMATION}d -> trading {TRADING}d) ---")
    add(wf.windows.to_string(index=False))
    add(f"Walk-forward (stitched OOS) strategy: {fmt(wf.stats)}  ({wf.n_trades} trades)")
    add(f"Walk-forward buy&hold               : {fmt(wf.stats['benchmark'])}")
    print("\n".join(lines[-2:]))

    # ---------------------------------------------------------------- 5
    hr("5) PARAMETER SENSITIVITY (in-sample grid)")
    finite = grid.to_numpy()[np.isfinite(grid.to_numpy())]
    near_best = np.abs(finite - finite.max()) < 0.25 * max(abs(finite.max()), 1e-9)
    plateau_share = near_best.mean()
    add("\n--- Parameter sensitivity ---")
    add(f"In-sample grid Sharpe range: [{finite.min():.2f}, {finite.max():.2f}]; "
        f"{plateau_share:.0%} of cells within 25% of the best.")
    add("The grid is in-sample only. Its shape says nothing about out-of-sample")
    add("performance; read it with the deflated Sharpe, PBO and the multi-seed section.")
    print("\n".join(lines[-3:]))

    # ---------------------------------------------------------------- 6
    hr(f"6) MULTI-SEED DISTRIBUTION ({n_seeds} synthetic seeds, 1..{n_seeds})")
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(one_seed, range(1, n_seeds + 1)))
    df = pd.DataFrame(rows)
    # A strategy that never trades in a window has no return variance and an
    # undefined Sharpe; it earned nothing, so it is scored 0 and counted.
    n_flat = int(df[["is", "oos", "wf"]].isna().any(axis=1).sum())
    df[["is", "oos", "wf"]] = df[["is", "oos", "wf"]].fillna(0.0)
    df["oos_minus_bh"] = df["oos"] - df["oos_bh"]
    df["wf_minus_bh"] = df["wf"] - df["wf_bh"]
    add(f"\n--- Multi-seed (synthetic, {n_seeds} seeds): mean [95% CI] (sd) ---")
    names = {
        "is": "In-sample Sharpe (selected)",
        "oos": "Out-of-sample Sharpe",
        "oos_bh": "Out-of-sample buy&hold Sharpe",
        "oos_minus_bh": "Out-of-sample strategy minus B&H",
        "wf": "Walk-forward Sharpe",
        "wf_bh": "Walk-forward buy&hold Sharpe",
        "wf_minus_bh": "Walk-forward strategy minus B&H",
    }
    for k, name in names.items():
        add(f"{name:34s}: {seed_distribution(df[k].to_numpy()).fmt(2)}")
    add(f"Seeds where OOS strategy Sharpe beats B&H: {(df['oos_minus_bh'] > 0).mean():.0%}")
    add(f"Seeds where WF strategy Sharpe beats B&H : {(df['wf_minus_bh'] > 0).mean():.0%}")
    add(f"Seeds with a flat (never invested) window, Sharpe scored 0: {n_flat}")
    print("\n".join(lines[-11:]))

    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {out_dir / 'report.txt'}")

    # ---------------------------------------------------------------- 7
    hr("7) FIGURES")
    _save_figures(prices, split, res_full, res_test, grid, wf, fig_dir)
    print(f"Figures written to {fig_dir}")

    hr("DONE")
    print(f"total runtime: {time.time() - t0:.1f}s")


def _save_figures(prices, split, res_full, res_test, grid, wf, fig_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed (pip install eq-signal-backtest[plots]); "
              "skipping figures.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(res_full.equity, lw=1.0, label="strategy (5 bps costs)")
    ax.plot(res_full.benchmark, lw=1.0, label="buy & hold")
    ax.axvline(split.split_date, color="k", ls=":", lw=1, label="train/test split")
    ax.legend()
    ax.set_title("Equity curves, full period (synthetic)")

    ax = axes[0, 1]
    ax.plot(res_test.equity, lw=1.0, label="strategy")
    ax.plot(res_test.benchmark, lw=1.0, label="buy & hold")
    ax.legend()
    ax.set_title("Out-of-sample only (synthetic)")

    ax = axes[1, 0]
    dd = res_full.equity / res_full.equity.cummax() - 1
    dd_b = res_full.benchmark / res_full.benchmark.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, alpha=0.6, label="strategy")
    ax.plot(dd_b, lw=0.8, color="k", label="buy & hold")
    ax.legend()
    ax.set_title("Drawdowns (full period)")

    ax = axes[1, 1]
    vals = grid.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel("slow window")
    ax.set_ylabel("fast window")
    fig.colorbar(im, ax=ax, label="in-sample Sharpe")
    ax.set_title("Parameter sensitivity (train set)")

    fig.tight_layout()
    fig.savefig(fig_dir / "backtest_overview.png", dpi=130)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(wf.equity, lw=1.0, label="walk-forward strategy (stitched OOS)")
    ax2.plot(wf.benchmark, lw=1.0, label="buy & hold (same OOS dates)")
    for _, row in wf.windows.iterrows():
        ax2.axvline(row["trading_start"], color="k", ls=":", lw=0.5, alpha=0.5)
    ax2.legend()
    ax2.set_title("Walk-forward: stitched out-of-sample equity (synthetic)")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "walk_forward.png", dpi=130)
    plt.close(fig2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticker", default=None, help="Yahoo Finance ticker for live data (optional)"
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: a new temp dir)")
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="number of synthetic seeds for the multi-seed section")
    args = parser.parse_args()
    main(args.ticker, args.out, args.seeds)
