"""End-to-end pipeline: synthetic data -> in-sample parameter selection ->
out-of-sample backtest -> cost-free comparison -> walk-forward validation
-> parameter-sensitivity heatmap -> report.

Reproduces (and extends) the legacy ``analysis.py`` script's four key
numbers, using the packaged ``eq_signal_backtest`` API instead of inline
script logic. Runs entirely offline on the bundled synthetic generator.

Usage
-----
    python examples/run_pipeline.py                  # bundled synthetic data
    python examples/run_pipeline.py --ticker SPY      # real data (needs
                                                       # `pip install
                                                       # eq-signal-backtest[live]`
                                                       # and network access)

Outputs: ``output/report.txt`` and ``output/figures/backtest_overview.png``
(paths relative to this project's root, not the current working directory).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from eq_signal_backtest import (
    TRADING_DAYS,
    performance_stats,
    run_backtest,
)
from eq_signal_backtest.data.synthetic import generate
from eq_signal_backtest.signals import ma_crossover_signal
from eq_signal_backtest.split import (
    select_best_params,
    train_test_split,
    walk_forward_backtest,
)

FAST_RANGE = range(10, 71, 10)
SLOW_RANGE = range(100, 251, 25)
COST_BPS = 5.0


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


def main(ticker: str | None = None) -> None:
    t0 = time.time()
    out_dir = ROOT / "output"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    add = lines.append

    # ---------------------------------------------------------------- 1
    hr("1) DATA")
    prices = load_prices(ticker)
    source_label = ticker if ticker else "bundled synthetic (seed=2)"
    print(f"source: {source_label}")
    print(f"{len(prices)} rows, {prices.index[0].date()} .. {prices.index[-1].date()}")
    add("=" * 72)
    add("TRADING SIGNAL BACKTEST -- MA CROSSOVER (long/flat)")
    add(f"Data: {source_label} | {prices.index[0].date()} to {prices.index[-1].date()}")
    add("=" * 72)

    # ---------------------------------------------------------------- 2
    hr("2) TRAIN/TEST SPLIT (70/30) AND IN-SAMPLE PARAMETER SELECTION")
    split = train_test_split(prices, train_frac=0.7)
    best_fast, best_slow, grid = select_best_params(
        split.train, FAST_RANGE, SLOW_RANGE, cost_bps=COST_BPS
    )
    print(f"train/test split at {split.split_date.date()}")
    print(f"parameters selected on training data: fast={best_fast}, slow={best_slow}")
    print(
        "(argmax of an in-sample Sharpe grid is itself a mild form of "
        "selection bias -- see docs/METHODOLOGY.md. The number that "
        "matters is the OUT-OF-SAMPLE row below.)"
    )
    add(f"\nTrain/test split at {split.split_date.date()} (70/30)")
    add(f"Parameters selected on training data: fast={best_fast}, slow={best_slow}")
    add("(Selected by grid-search Sharpe on the training window. The grid")
    add("search itself is a source of selection bias, so the number that")
    add("matters is the OUT-OF-SAMPLE row below.)")

    # ---------------------------------------------------------------- 3
    hr("3) IN-SAMPLE / OUT-OF-SAMPLE / FULL-PERIOD / COST-FREE BACKTESTS")

    def evaluate(px: pd.Series, cost_bps: float = COST_BPS):
        sig = ma_crossover_signal(px, best_fast, best_slow)
        return run_backtest(px, sig, cost_bps=cost_bps)

    res_train = evaluate(split.train)
    res_test = evaluate(split.test)
    res_full = evaluate(prices)
    res_free = evaluate(prices, cost_bps=0.0)

    add("\n--- Performance ---")
    add(f"In-sample   strategy : {fmt(res_train.stats)}")
    add(f"In-sample   buy&hold : {fmt(res_train.stats['benchmark'])}")
    add(f"Out-sample  strategy : {fmt(res_test.stats)}")
    add(f"Out-sample  buy&hold : {fmt(res_test.stats['benchmark'])}")
    add(f"Full period strategy : {fmt(res_full.stats)}  ({res_full.n_trades} trades)")
    add(f"Full, zero costs     : {fmt(res_free.stats)}")
    cost_drag = res_free.stats["cagr"] - res_full.stats["cagr"]
    add(f"Transaction cost drag: {cost_drag:.2%} CAGR at {COST_BPS:.0f} bps per trade")
    for line in lines[-8:]:
        print(line)

    is_sh = res_train.stats["sharpe"]
    oos_sh = res_test.stats["sharpe"]
    add("\n--- Honest read (Key Number 1: in-sample vs out-of-sample Sharpe) ---")
    add(f"Sharpe decay in-sample -> out-of-sample: {is_sh:.2f} -> {oos_sh:.2f}")
    if oos_sh < is_sh * 0.5:
        add("Substantial decay: much of the in-sample edge was likely")
        add("fitted noise rather than a persistent effect.")
    else:
        add("Decay is moderate, but one out-of-sample window is weak")
        add("evidence -- walk-forward analysis (section 4) is more convincing.")
    print(f"\nKey number 1 -- in-sample Sharpe {is_sh:.2f} -> out-of-sample Sharpe {oos_sh:.2f}")
    print(
        f"Key number 2 -- strategy vs buy&hold (full period): "
        f"maxDD {res_full.stats['max_drawdown']:.2%} vs "
        f"{res_full.stats['benchmark']['max_drawdown']:.2%}"
    )
    print(f"Key number 3 -- transaction cost drag: {cost_drag:.2%} CAGR")

    # ---------------------------------------------------------------- 4
    hr("4) WALK-FORWARD VALIDATION (re-select every window, trade frozen params)")
    formation, trading = 756, 252  # ~3y formation, 1y trading
    wf = walk_forward_backtest(
        prices, FAST_RANGE, SLOW_RANGE, formation=formation, trading=trading,
        cost_bps=COST_BPS,
    )
    print(wf.windows.to_string(index=False))
    print(f"\nwalk-forward (stitched OOS): {fmt(wf.stats)}  ({wf.n_trades} trades)")
    print(f"walk-forward buy&hold      : {fmt(wf.stats['benchmark'])}")
    add("\n--- Walk-forward validation (formation "
        f"{formation}d -> trading {trading}d) ---")
    add(wf.windows.to_string(index=False))
    add(f"Walk-forward (stitched OOS) strategy: {fmt(wf.stats)}")
    add(f"Walk-forward buy&hold               : {fmt(wf.stats['benchmark'])}")

    # ---------------------------------------------------------------- 5
    hr("5) PARAMETER SENSITIVITY (Key Number 4: plateau vs spike)")
    grid_range = grid.to_numpy()
    finite = grid_range[np.isfinite(grid_range)]
    spread = finite.max() - finite.min() if finite.size else np.nan
    near_best = np.abs(finite - finite.max()) < 0.25 * max(abs(finite.max()), 1e-9)
    plateau_share = near_best.mean() if finite.size else np.nan
    print(f"grid Sharpe range (in-sample): [{finite.min():.2f}, {finite.max():.2f}]")
    print(f"share of cells within 25% of the best Sharpe: {plateau_share:.0%}")
    if plateau_share > 0.3:
        character = "a broad plateau -- consistent with a robust effect"
    else:
        character = "a narrow spike -- consistent with curve-fitting"
    print(f"character: {character}")
    # A plateau is evidence about the SHAPE of the in-sample surface, not
    # about out-of-sample performance. Saying so explicitly matters: the
    # two can, and here do, point in different directions.
    reconciliation = (
        "Note: the grid is measured in-sample only. A plateau means the "
        f"in-sample result is not one lucky cell -- it does NOT mean the "
        f"effect survives out-of-sample. Here the out-of-sample Sharpe is "
        f"{oos_sh:.2f} against an in-sample {is_sh:.2f}, so the plateau is "
        "a plateau of equally over-fitted parameters, not evidence of a "
        "tradeable edge."
        if plateau_share > 0.3 and oos_sh < is_sh * 0.5
        else "Note: the grid is measured in-sample only; read it alongside "
        "the out-of-sample and walk-forward rows above, never instead of them."
    )
    print(reconciliation)
    add("\n--- Parameter sensitivity ---")
    add(f"In-sample grid Sharpe range: [{finite.min():.2f}, {finite.max():.2f}]; "
        f"{plateau_share:.0%} of cells within 25% of the best -> {character}.")
    add(reconciliation)
    add("Where this class of strategy works: sustained trends (it stays")
    add("invested) and prolonged bear markets (it steps aside).")
    add("Where it fails: choppy, range-bound markets, where every")
    add("crossover is a whipsaw that pays costs and captures no move.")

    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report + "\n")
    print(f"\nReport written to {out_dir / 'report.txt'}")

    # ---------------------------------------------------------------- 6
    hr("6) FIGURES")
    _save_figures(prices, split, res_full, res_test, grid, wf, fig_dir)
    print(f"Figures written to {fig_dir}/")

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
    ax.set_title("Equity curves, full period")

    ax = axes[0, 1]
    ax.plot(res_test.equity, lw=1.0, label="strategy")
    ax.plot(res_test.benchmark, lw=1.0, label="buy & hold")
    ax.legend()
    ax.set_title("Out-of-sample only")

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
    ax2.set_title("Walk-forward: stitched out-of-sample equity")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "walk_forward.png", dpi=130)
    plt.close(fig2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticker", default=None, help="Yahoo Finance ticker for live data (optional)"
    )
    args = parser.parse_args()
    main(args.ticker)
