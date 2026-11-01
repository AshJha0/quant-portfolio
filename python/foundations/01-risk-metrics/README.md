# Single-Asset Equity Risk Metrics (`eq_risk_metrics`)

Compute and interpret core risk metrics for a single asset: volatility
(three estimators), Value at Risk (three methods), Expected Shortfall,
maximum drawdown, Sharpe/Sortino ratios, distribution diagnostics, and a
Kupiec coverage backtest of the VaR numbers themselves.

The point of this project is not the arithmetic, every metric here is a
few lines of NumPy/SciPy. The point is that **different, individually
reasonable methods give different answers on the same data**, and this
project makes that gap visible and measured rather than asserted:
historical VaR beats Gaussian VaR at 99% confidence because daily equity
returns have fat tails; Expected Shortfall sits above VaR because the
tail is a slope, not a cliff edge; Jarque-Bera decisively rejects
normality; and volatility is time-varying enough that a single
unconditional number can describe a market that no longer exists. The
Kupiec backtest closes the loop: on the bundled data the Gaussian 99% VaR
is **rejected** for undercounting exceptions, while the historical one is
not, the disagreement between methods has a right answer, and it is
measurable.

```
data (synthetic two-regime series, or real prices via yfinance)
              │
              ▼
returns (simple / log) ──► volatility (full-sample / rolling / EWMA)
              │
              ▼
Value at Risk (historical / Gaussian / Cornish-Fisher) ──► Expected Shortfall
              │
              ▼
drawdown ──► Sharpe / Sortino ──► normality diagnostics (Jarque-Bera)
              │
              ▼
VaR backtest (Kupiec proportion-of-failures: does 99% VaR really fail 1% of days?)
```

**Conventions:** 252 trading days a year. Volatility is annualised by
`sqrt(252)`. VaR and ES are **positive loss fractions** at a stated
confidence and a 1-day horizon. With `alpha = 1 - confidence` and
`k = n * alpha`, historical VaR is the `ceil(k)`-th largest loss (no
interpolation) and ES is the Acerbi-Tasche average of the worst `k`
losses. EWMA volatility is the zero-mean RiskMetrics recursion seeded
over the first 30 returns. Cornish-Fisher VaR raises when the sample
skew and kurtosis are outside its domain. Every public function raises
`ValueError` on NaN, infinite, empty or too-short input.

This is a **single-asset** project,
portfolio-level risk (correlation, diversification) is out of scope; see
`python/equity/03-var-es-engine` in this portfolio for the multi-asset
extension.

## Quickstart

```bash
cd python/foundations/01-risk-metrics
pip install -e ".[dev,plots]"        # numpy, pandas, scipy + pytest, hypothesis, matplotlib
pytest -q                            # 217 tests, offline, seeded, ~45 s
python examples/run_pipeline.py      # full report + figures on bundled synthetic data
```

```python
from eq_risk_metrics import simple_returns, var_historical, var_parametric, expected_shortfall
from eq_risk_metrics.data import generate

prices = generate(seed=2).set_index("Date")["Adj Close"]
rets = simple_returns(prices)
var_historical(rets, 0.99)     # 0.0318 (3.18%)
var_parametric(rets, 0.99)     # 0.0223 (2.23%) -- understates the tail
expected_shortfall(rets, 0.99) # 0.0434 (4.34%) -- worse than either VaR

# ...and the understatement is not a matter of taste: it is testable.
from eq_risk_metrics import kupiec_pof_test
kupiec_pof_test(rets, var_parametric(rets, 0.99), 0.99)["reject_at_5pct"]  # True
kupiec_pof_test(rets, var_historical(rets, 0.99), 0.99)["reject_at_5pct"]  # False
```

To use real market data (recommended before presenting results to
anyone):

```bash
pip install -e ".[live]"
python -m eq_risk_metrics.data.live SPY 2016-01-01   # writes data/SPY.csv
python examples/run_pipeline.py --csv data/SPY.csv
```

## Results summary (bundled synthetic data, seed=2, 2,519 daily returns)

Produced by `python examples/run_pipeline.py --no-plots` on synthetic data.

| Metric | Value |
|---|---:|
| Annualised volatility (full sample) | 15.64% |
| Latest EWMA (λ=0.94) volatility | 11.61% |
| Historical VaR, 99% | 3.18% |
| Gaussian VaR, 99% | 2.23% |
| Cornish-Fisher VaR, 99% | not reported: skew -0.51, kurtosis +9.28 fail the CF domain check |
| Expected Shortfall, 99% (Acerbi-Tasche) | 4.34% |
| Maximum drawdown | -28.27% |
| Sharpe ratio (rf = 3%) | 0.85 |
| Sortino ratio (rf = 3%) | 1.16 |
| Skewness | -0.51 |
| Excess kurtosis | +9.28 |
| Jarque-Bera p-value | ≈ 0 (normality rejected) |
| Kupiec POF, 99% historical VaR | 25 exceptions vs 25.2 expected, p = 0.97 → **not rejected** |
| Kupiec POF, 99% Gaussian VaR | 60 exceptions vs 25.2 expected, p ≈ 0 → **rejected** |

The five observations these numbers are meant to demonstrate, historical
99% VaR > Gaussian 99% VaR, ES noticeably above VaR, JB rejects normality
decisively, volatility clusters (21-day rolling vol ranges 5.3%–45.7%
over the sample), and the Gaussian estimator *fails its coverage backtest*
at 99% while the historical one passes, are all confirmed. The last one
is the point of the whole project: the fat-tail critique of Gaussian VaR
is not an aesthetic preference about distributions, it is a measurable
failure of the promise the confidence level makes. Full numbers,
reproduction instructions, and known failure modes: `docs/VALIDATION.md`.

## Data

The package ships a **synthetic** generator
(`eq_risk_metrics.data.synthetic.generate`), a two-regime (calm/stressed)
model with Student-t shocks, seeded and deterministic. It exists so the
project runs fully offline and reproduces the stylised facts real daily
equity returns show, fat tails, volatility clustering, occasional
drawdowns, so the analysis is meaningful without a network connection.
Any conclusions presented to a reader should be drawn from real data via
`eq_risk_metrics.data.live` (yfinance, import-guarded, importing the
package never requires yfinance or network access).

When using real data, adjusted close prices are used, so dividends and
splits are already folded into the return series.

## Layout

```
src/eq_risk_metrics/     volatility, var_es, performance, diagnostics,
                         backtest, data/{synthetic,live}
tests/                   217 offline seeded pytest tests
examples/run_pipeline.py end-to-end report + figures reproducing the
                         numbers above
docs/                    METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```

## Documentation contract

- **Why these methods, vs alternatives** (incl. Kupiec vs Christoffersen vs Basel traffic-light, and why ES is hard to backtest at all), [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §1–7
- **Assumptions register (A1–A9, each with "what breaks")**, [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §8
- **Validation evidence** (analytic checks, bundled-data results), [docs/VALIDATION.md](docs/VALIDATION.md) §1–2
- **Failure modes** (short samples, constant/degenerate returns, regime
  divergence, numerical limits), [docs/VALIDATION.md](docs/VALIDATION.md) §3
- **Desk usage** (who consumes VaR/ES, reporting, governance, Kupiec
  backtesting), [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md) §1–3
- **Real-life scenarios & edge cases** (limit breach, regime shift, a PM
  arguing with Gaussian VaR, new-listing short samples; every edge case
  unit-tested), [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md) §4, `tests/test_edge_cases.py`
