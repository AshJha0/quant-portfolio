"""
Run the full risk analysis on a price series and produce:

  output/report.txt   -- all metrics with interpretation
  output/figures/*.png -- price, drawdown, rolling vol, return distribution

Usage:
    python analysis.py                # uses data/sample_prices.csv
    python analysis.py data/SPY.csv   # or any CSV with Date, Adj Close
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import risk_metrics as rm


def load_prices(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df.set_index("Date")[price_col].astype(float)


def main(path: str = "data/sample_prices.csv") -> None:
    os.makedirs("output/figures", exist_ok=True)
    prices = load_prices(path)
    rets = rm.simple_returns(prices)

    # ------------------------------------------------------------------
    # Compute everything
    # ------------------------------------------------------------------
    vol = rm.annualised_volatility(rets)
    roll_vol = rm.rolling_volatility(rets)
    ewma_vol = rm.ewma_volatility(rets)

    conf_levels = [0.95, 0.99]
    var_table = {
        c: {
            "historical": rm.var_historical(rets, c),
            "parametric": rm.var_parametric(rets, c),
            "cornish_fisher": rm.var_cornish_fisher(rets, c),
            "expected_shortfall": rm.expected_shortfall(rets, c),
        }
        for c in conf_levels
    }

    dd = rm.max_drawdown(prices)
    sharpe = rm.sharpe_ratio(rets)
    sortino = rm.sortino_ratio(rets)
    norm = rm.normality_report(rets)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    lines = []
    add = lines.append
    add("=" * 70)
    add("RISK METRICS REPORT")
    add(f"Data file : {path}")
    add(f"Period    : {prices.index[0].date()} to {prices.index[-1].date()}"
        f"  ({len(rets)} daily returns)")
    add("=" * 70)

    add("\n--- Volatility ---")
    add(f"Annualised volatility (full sample) : {vol:.2%}")
    add(f"Latest 21-day rolling volatility    : {roll_vol.iloc[-1]:.2%}")
    add(f"Latest EWMA (lambda=0.94) volatility: {ewma_vol.iloc[-1]:.2%}")
    add("Interpretation: if EWMA/rolling vol differs materially from the")
    add("full-sample figure, the market is in a different regime than the")
    add("long-run average and the unconditional number is misleading.")

    add("\n--- Value at Risk (1-day, % of portfolio value) ---")
    for c in conf_levels:
        v = var_table[c]
        add(f"  {c:.0%} confidence:")
        add(f"    Historical VaR      : {v['historical']:.2%}")
        add(f"    Gaussian VaR        : {v['parametric']:.2%}")
        add(f"    Cornish-Fisher VaR  : {v['cornish_fisher']:.2%}")
        add(f"    Expected Shortfall  : {v['expected_shortfall']:.2%}")
    gap99 = var_table[0.99]["historical"] - var_table[0.99]["parametric"]
    add(f"\nGaussian vs historical gap at 99%: {gap99:+.2%}")
    add("A positive gap means the normal distribution UNDERSTATES tail")
    add("risk, which is the typical finding for daily equity returns.")

    add("\n--- Drawdown ---")
    add(f"Maximum drawdown : {dd['max_drawdown']:.2%}")
    add(f"Peak date        : {dd['peak_date'].date()}")
    add(f"Trough date      : {dd['trough_date'].date()}")

    add("\n--- Risk-adjusted performance (rf = 3% annual) ---")
    add(f"Sharpe ratio  : {sharpe:.2f}")
    add(f"Sortino ratio : {sortino:.2f}")

    add("\n--- Distribution diagnostics ---")
    add(f"Skewness         : {norm['skewness']:+.3f}")
    add(f"Excess kurtosis  : {norm['excess_kurtosis']:+.3f}")
    add(f"Jarque-Bera stat : {norm['jarque_bera_stat']:.1f} "
        f"(p = {norm['jarque_bera_pvalue']:.2e})")
    add(f"Normality rejected at 5%: {norm['normality_rejected_at_5pct']}")
    add("Excess kurtosis > 0 means fat tails: extreme days happen more")
    add("often than a normal distribution predicts, which is exactly why")
    add("historical VaR and Gaussian VaR disagree above.")

    report = "\n".join(lines)
    with open("output/report.txt", "w") as f:
        f.write(report + "\n")
    print(report)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(prices.index, prices.values, lw=0.8)
    axes[0, 0].set_title("Price")

    axes[0, 1].fill_between(dd["drawdown_series"].index,
                            dd["drawdown_series"].values, 0,
                            color="firebrick", alpha=0.6)
    axes[0, 1].set_title(f"Drawdown (max {dd['max_drawdown']:.1%})")

    axes[1, 0].plot(roll_vol.index, roll_vol.values, lw=0.8,
                    label="21d rolling")
    axes[1, 0].plot(ewma_vol.index, ewma_vol.values, lw=0.8,
                    label="EWMA λ=0.94")
    axes[1, 0].axhline(vol, color="k", ls="--", lw=0.8, label="full sample")
    axes[1, 0].legend()
    axes[1, 0].set_title("Annualised volatility")

    ax = axes[1, 1]
    ax.hist(rets, bins=80, density=True, alpha=0.6, label="empirical")
    x = np.linspace(rets.min(), rets.max(), 400)
    ax.plot(x, stats.norm.pdf(x, rets.mean(), rets.std()),
            "r--", lw=1.2, label="normal fit")
    ax.axvline(-rm.var_historical(rets, 0.99), color="k", lw=1,
               label="99% hist. VaR")
    ax.legend()
    ax.set_title("Daily return distribution")

    fig.tight_layout()
    fig.savefig("output/figures/risk_overview.png", dpi=130)
    print("\nFigures written to output/figures/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample_prices.csv")
