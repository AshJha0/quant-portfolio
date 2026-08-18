"""
Backtest a 50/200 moving-average crossover with:

  - a 70/30 in-sample / out-of-sample split (parameters are chosen on
    the first 70% of history and judged on the final 30%),
  - transaction costs,
  - a buy-and-hold benchmark,
  - a parameter-sensitivity heatmap to expose overfitting.

Outputs: output/report.txt, output/figures/*.png

Usage:
    python analysis.py                 # bundled sample data
    python analysis.py data/SPY.csv    # real data (see download_data.py)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as bt


def load_prices(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df.set_index("Date")[price_col].astype(float)


def fmt(stats: dict) -> str:
    return (f"CAGR {stats['cagr']:+.2%} | vol {stats['volatility']:.2%} | "
            f"Sharpe {stats['sharpe']:.2f} | maxDD {stats['max_drawdown']:.2%}")


def main(path: str = "data/sample_prices.csv") -> None:
    os.makedirs("output/figures", exist_ok=True)
    prices = load_prices(path)

    split = int(len(prices) * 0.7)
    train, test = prices.iloc[:split], prices.iloc[split:]

    # ------------------------------------------------------------------
    # 1. Choose parameters ON THE TRAINING SET ONLY
    # ------------------------------------------------------------------
    grid = bt.parameter_grid(train,
                             fast_range=range(10, 71, 10),
                             slow_range=range(100, 251, 25))
    best_fast, best_slow = grid.stack().idxmax()
    # Note: picking the argmax of a grid IS a mild form of overfitting.
    # That is exactly why the strategy must then be judged out-of-sample.

    # ------------------------------------------------------------------
    # 2. Evaluate in-sample and out-of-sample
    # ------------------------------------------------------------------
    def evaluate(px: pd.Series) -> bt.BacktestResult:
        sig = bt.ma_crossover_signal(px, best_fast, best_slow)
        return bt.run_backtest(px, sig, cost_bps=5.0)

    res_train = evaluate(train)
    res_test = evaluate(test)
    res_full = evaluate(prices)
    res_free = bt.run_backtest(  # cost-free variant, for comparison
        prices, bt.ma_crossover_signal(prices, best_fast, best_slow),
        cost_bps=0.0)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    lines = []
    add = lines.append
    add("=" * 72)
    add("TRADING SIGNAL BACKTEST -- MA CROSSOVER (long/flat)")
    add(f"Data: {path} | {prices.index[0].date()} to {prices.index[-1].date()}")
    add(f"Train/test split at {train.index[-1].date()} (70/30)")
    add("=" * 72)
    add(f"\nParameters selected on training data: fast={best_fast}, "
        f"slow={best_slow}")
    add("(Selected by grid-search Sharpe on the training window. The grid")
    add("search itself is a source of selection bias, so the number that")
    add("matters is the OUT-OF-SAMPLE row below.)")

    add("\n--- Performance ---")
    add(f"In-sample   strategy : {fmt(res_train.stats)}")
    add(f"In-sample   buy&hold : {fmt(res_train.stats['benchmark'])}")
    add(f"Out-sample  strategy : {fmt(res_test.stats)}")
    add(f"Out-sample  buy&hold : {fmt(res_test.stats['benchmark'])}")
    add(f"Full period strategy : {fmt(res_full.stats)}  "
        f"({res_full.n_trades} trades)")
    add(f"Full, zero costs     : {fmt(res_free.stats)}")
    cost_drag = (res_free.stats['cagr'] - res_full.stats['cagr'])
    add(f"Transaction cost drag: {cost_drag:.2%} CAGR at 5 bps per trade")

    is_sh = res_train.stats["sharpe"]
    oos_sh = res_test.stats["sharpe"]
    add("\n--- Honest read ---")
    add(f"Sharpe decay in-sample -> out-of-sample: {is_sh:.2f} -> {oos_sh:.2f}")
    if oos_sh < is_sh * 0.5:
        add("Substantial decay: much of the in-sample edge was likely")
        add("fitted noise rather than a persistent effect.")
    else:
        add("Decay is moderate, but one out-of-sample window is weak")
        add("evidence -- walk-forward analysis would be more convincing.")
    add("Where this class of strategy works: sustained trends (it stays")
    add("invested) and prolonged bear markets (it steps aside).")
    add("Where it fails: choppy, range-bound markets, where every")
    add("crossover is a whipsaw that pays costs and captures no move.")

    report = "\n".join(lines)
    with open("output/report.txt", "w") as f:
        f.write(report + "\n")
    print(report)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(res_full.equity, lw=1.0, label="strategy (5 bps costs)")
    ax.plot(res_full.benchmark, lw=1.0, label="buy & hold")
    ax.axvline(train.index[-1], color="k", ls=":", lw=1,
               label="train/test split")
    ax.legend()
    ax.set_title(f"Equity curves, MA({best_fast},{best_slow})")

    ax = axes[0, 1]
    eq, bench = res_test.equity, res_test.benchmark
    ax.plot(eq, lw=1.0, label="strategy")
    ax.plot(bench, lw=1.0, label="buy & hold")
    ax.legend()
    ax.set_title("Out-of-sample only")

    ax = axes[1, 0]
    dd = res_full.equity / res_full.equity.cummax() - 1
    dd_b = res_full.benchmark / res_full.benchmark.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, alpha=0.6, label="strategy")
    ax.plot(dd_b, lw=0.8, color="k", label="buy & hold")
    ax.legend()
    ax.set_title("Drawdowns")

    ax = axes[1, 1]
    im = ax.imshow(grid.values, aspect="auto", cmap="RdYlGn",
                   vmin=-abs(grid.values).max(), vmax=abs(grid.values).max())
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel("slow window")
    ax.set_ylabel("fast window")
    fig.colorbar(im, ax=ax, label="in-sample Sharpe")
    ax.set_title("Parameter sensitivity (train set)")

    fig.tight_layout()
    fig.savefig("output/figures/backtest_overview.png", dpi=130)
    print("\nFigures written to output/figures/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample_prices.csv")
