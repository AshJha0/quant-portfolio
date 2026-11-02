"""Print the cost-sensitivity and choppy-regime tables in docs/VALIDATION.md.

    python examples/validation_tables.py

Synthetic data only (bundled generator, seed 32, and a seeded
mean-reverting path, seed 99). Writes nothing to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from eq_signal_backtest import run_backtest
from eq_signal_backtest.data.synthetic import generate
from eq_signal_backtest.signals import ma_crossover_signal


def range_bound_prices(n=750, seed=99, kappa=0.05, sigma=0.012) -> pd.Series:
    """Same construction as tests/test_edge_cases.py::_range_bound_prices."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = x[t - 1] + kappa * (0.0 - x[t - 1]) + sigma * rng.standard_normal()
    return pd.Series(100 * np.exp(x), index=pd.bdate_range("2020-01-01", periods=n))


def main() -> None:
    prices = generate().set_index("Date")["Adj Close"].astype(float)
    print("Cost sensitivity (synthetic, seed 32, full period)")
    print("| (fast, slow) | trades | Sharpe 0bp | Sharpe 5bp | Sharpe 20bp "
          "| CAGR 0bp | CAGR 5bp | CAGR 20bp |")
    for fa, sl in [(10, 50), (10, 100), (40, 100), (60, 250)]:
        sig = ma_crossover_signal(prices, fa, sl)
        rs = [run_backtest(prices, sig, c) for c in (0.0, 5.0, 20.0)]
        print(f"| ({fa}, {sl}) | {rs[1].n_trades} | "
              + " | ".join(f"{r.stats['sharpe']:.3f}" for r in rs) + " | "
              + " | ".join(f"{r.stats['cagr']:.2%}" for r in rs) + " |")

    px = range_bound_prices()
    sig = ma_crossover_signal(px, 20, 100)
    costed = run_backtest(px, sig, 5.0)
    free = run_backtest(px, sig, 0.0)
    print("\nChoppy regime (synthetic mean-reverting path, seed 99, MA 20/100)")
    print(f"trades {costed.n_trades}")
    print(f"strategy 5bp: CAGR {costed.stats['cagr']:.2%} Sharpe {costed.stats['sharpe']:.2f} "
          f"maxDD {costed.stats['max_drawdown']:.2%}")
    print(f"strategy 0bp: CAGR {free.stats['cagr']:.2%}")
    b = costed.stats["benchmark"]
    print(f"buy & hold  : CAGR {b['cagr']:.2%} Sharpe {b['sharpe']:.2f} "
          f"maxDD {b['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
