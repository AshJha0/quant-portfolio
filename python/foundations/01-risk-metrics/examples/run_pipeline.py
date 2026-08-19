"""End-to-end risk metrics pipeline.

Generates (or loads) a daily price series, computes the full metrics
suite -- volatility (3 estimators), VaR (3 methods), Expected
Shortfall, drawdown, Sharpe/Sortino, and distribution diagnostics --
prints a report, and saves figures.

Usage
-----
    python examples/run_pipeline.py
        # synthetic data, deterministic (seed=2), fully offline

    python examples/run_pipeline.py --csv data/SPY.csv
        # any CSV with Date, Adj Close (or Close) columns

    python examples/run_pipeline.py --seed 7 --n-days 1500
        # a different synthetic sample

Outputs go to ``output/report.txt`` and ``output/figures/`` (relative
to the current working directory).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import eq_risk_metrics as rm  # noqa: E402
from eq_risk_metrics.data import generate  # noqa: E402

CONF_LEVELS = (0.95, 0.99)


def load_prices_from_csv(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df.set_index("Date")[price_col].astype(float)


def build_report(prices: pd.Series, source: str) -> tuple[str, dict]:
    rets = rm.simple_returns(prices)

    vol = rm.annualised_volatility(rets)
    roll_vol = rm.rolling_volatility(rets)
    ewma_vol = rm.ewma_volatility(rets)

    var_table = {
        c: {
            "historical": rm.var_historical(rets, c),
            "parametric": rm.var_parametric(rets, c),
            "cornish_fisher": rm.var_cornish_fisher(rets, c),
            "expected_shortfall": rm.expected_shortfall(rets, c),
        }
        for c in CONF_LEVELS
    }

    backtest = {
        (c, method): rm.kupiec_pof_test(rets, var_table[c][method], c)
        for c in CONF_LEVELS
        for method in ("historical", "parametric")
    }

    dd = rm.max_drawdown(prices)
    sharpe = rm.sharpe_ratio(rets)
    sortino = rm.sortino_ratio(rets)
    norm = rm.normality_report(rets)

    lines: list[str] = []
    add = lines.append
    add("=" * 70)
    add("RISK METRICS REPORT")
    add(f"Data source : {source}")
    add(
        f"Period      : {prices.index[0].date()} to {prices.index[-1].date()}"
        f"  ({len(rets)} daily returns)"
    )
    add("=" * 70)

    add("\n--- Volatility ---")
    add(f"Annualised volatility (full sample) : {vol:.2%}")
    add(f"Latest 21-day rolling volatility    : {roll_vol.iloc[-1]:.2%}")
    add(f"Latest EWMA (lambda=0.94) volatility: {ewma_vol.iloc[-1]:.2%}")
    ewma_gap = ewma_vol.iloc[-1] - vol
    add(
        f"EWMA - full-sample gap: {ewma_gap:+.2%} -- "
        + (
            "material divergence: the market is in a different regime than "
            "the long-run average and the unconditional figure is stale."
            if abs(ewma_gap) > 0.05
            else "close to the unconditional figure: no strong regime signal."
        )
    )

    add("\n--- Value at Risk (1-day, % of position value) ---")
    for c in CONF_LEVELS:
        v = var_table[c]
        add(f"  {c:.0%} confidence:")
        add(f"    Historical VaR      : {v['historical']:.2%}")
        add(f"    Gaussian VaR        : {v['parametric']:.2%}")
        add(f"    Cornish-Fisher VaR  : {v['cornish_fisher']:.2%}")
        add(f"    Expected Shortfall  : {v['expected_shortfall']:.2%}")
    gap99 = var_table[0.99]["historical"] - var_table[0.99]["parametric"]
    add(f"\nGaussian vs historical gap at 99%: {gap99:+.2%}")
    add(
        "A positive gap means the normal distribution UNDERSTATES tail "
        "risk, which is the typical finding for daily equity returns."
    )
    es_gap99 = var_table[0.99]["expected_shortfall"] - var_table[0.99]["historical"]
    add(f"ES - historical VaR gap at 99%: {es_gap99:+.2%} (ES >= VaR always)")

    add("\n--- VaR backtest (Kupiec proportion-of-failures, in-sample) ---")
    for c in CONF_LEVELS:
        for method, label in (("historical", "Historical"), ("parametric", "Gaussian  ")):
            b = backtest[(c, method)]
            add(
                f"  {c:.0%} {label}: {b['n_exceptions']:>4} exceptions in "
                f"{b['n_observations']} days (expected {b['expected_exceptions']:.1f}), "
                f"LR = {b['lr_statistic']:.2f}, p = {b['p_value']:.3f} -> "
                + ("REJECT" if b["reject_at_5pct"] else "not rejected")
            )
    add(
        "The Gaussian estimator failing at 99% while the historical one "
        "passes is this project's central claim, made falsifiable: the "
        "normal assumption does not merely look wrong on a QQ plot, it "
        "breaks the coverage the confidence level promises."
    )
    add(
        "Caveat: this is an IN-SAMPLE backtest -- both estimators are fitted "
        "on the same window they are scored on, so passing only confirms the "
        "quantile arithmetic, not forecasting power. A real control runs "
        "out-of-sample on a rolling forecast and adds an independence "
        "(clustering) test -- see docs/DESK_GUIDE.md."
    )

    add("\n--- Drawdown ---")
    add(f"Maximum drawdown : {dd['max_drawdown']:.2%}")
    add(f"Peak date        : {pd.Timestamp(dd['peak_date']).date()}")
    add(f"Trough date      : {pd.Timestamp(dd['trough_date']).date()}")

    add("\n--- Risk-adjusted performance (rf = 3% annual) ---")
    add(f"Sharpe ratio  : {sharpe:.2f}")
    add(f"Sortino ratio : {sortino:.2f}")

    add("\n--- Distribution diagnostics ---")
    add(f"Skewness         : {norm['skewness']:+.3f}")
    add(f"Excess kurtosis  : {norm['excess_kurtosis']:+.3f}")
    add(
        f"Jarque-Bera stat : {norm['jarque_bera_stat']:.1f} "
        f"(p = {norm['jarque_bera_pvalue']:.2e})"
    )
    add(f"Normality rejected at 5%: {norm['normality_rejected_at_5pct']}")
    add(
        "Excess kurtosis > 0 means fat tails: extreme days happen more "
        "often than a normal distribution predicts, which is exactly why "
        "historical VaR and Gaussian VaR disagree above."
    )

    context = {
        "prices": prices,
        "rets": rets,
        "roll_vol": roll_vol,
        "ewma_vol": ewma_vol,
        "vol": vol,
        "dd": dd,
        "var_table": var_table,
        "backtest": backtest,
    }
    return "\n".join(lines), context


def save_figures(context: dict, out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prices, rets = context["prices"], context["rets"]
    dd, roll_vol, ewma_vol, vol = (
        context["dd"],
        context["roll_vol"],
        context["ewma_vol"],
        context["vol"],
    )

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(prices.index, prices.values, lw=0.8)
    axes[0, 0].set_title("Price")

    axes[0, 1].fill_between(
        dd["drawdown_series"].index,
        dd["drawdown_series"].values,
        0,
        color="firebrick",
        alpha=0.6,
    )
    axes[0, 1].set_title(f"Drawdown (max {dd['max_drawdown']:.1%})")

    axes[1, 0].plot(roll_vol.index, roll_vol.values, lw=0.8, label="21d rolling")
    axes[1, 0].plot(ewma_vol.index, ewma_vol.values, lw=0.8, label="EWMA λ=0.94")
    axes[1, 0].axhline(vol, color="k", ls="--", lw=0.8, label="full sample")
    axes[1, 0].legend()
    axes[1, 0].set_title("Annualised volatility")

    ax = axes[1, 1]
    ax.hist(rets, bins=80, density=True, alpha=0.6, label="empirical")
    x = np.linspace(rets.min(), rets.max(), 400)
    ax.plot(x, stats.norm.pdf(x, rets.mean(), rets.std()), "r--", lw=1.2, label="normal fit")
    ax.axvline(
        -rm.var_historical(rets, 0.99), color="k", lw=1, label="99% hist. VaR"
    )
    ax.legend()
    ax.set_title("Daily return distribution")

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "risk_overview.png")
    fig.savefig(fig_path, dpi=130)
    print(f"\nFigures written to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None, help="Path to a price CSV")
    parser.add_argument("--seed", type=int, default=2, help="Synthetic data seed")
    parser.add_argument("--n-days", type=int, default=2520, help="Synthetic sample length")
    parser.add_argument(
        "--out-dir", type=str, default="output", help="Output directory for report/figures"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip matplotlib figure generation"
    )
    args = parser.parse_args()

    if args.csv:
        prices = load_prices_from_csv(args.csv)
        source = args.csv
    else:
        df = generate(n_days=args.n_days, seed=args.seed)
        prices = df.set_index("Date")["Adj Close"]
        source = f"synthetic (seed={args.seed}, n_days={args.n_days})"

    report, context = build_report(prices, source)
    print(report)

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {report_path}")

    if not args.no_plots:
        try:
            save_figures(context, os.path.join(args.out_dir, "figures"))
        except ImportError:
            print("\nmatplotlib not installed -- skipping figures (pip install '.[plots]')")


if __name__ == "__main__":
    main()
