# Single-Asset Equity Risk Metrics (`eq_risk_metrics`)

Compute and interpret core risk metrics for a single asset: volatility
(three estimators), Value at Risk (three methods), Expected Shortfall,
maximum drawdown, Sharpe/Sortino ratios, and distribution diagnostics.

The point of this project is not the arithmetic — every metric here is a
few lines of NumPy/SciPy. The point is that **different, individually
reasonable methods give different answers on the same data**, and this
project makes that gap visible and measured rather than asserted:
historical VaR beats Gaussian VaR at 99% confidence because daily equity
returns have fat tails; Expected Shortfall sits above VaR because the
tail is a slope, not a cliff edge; Jarque-Bera decisively rejects
normality; and volatility is time-varying enough that a single
unconditional number can describe a market that no longer exists.

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
```

**Conventions:** 252 trading days/year; volatility annualised by
`sqrt(252)`; VaR/ES reported as **positive loss fractions** at a stated
confidence and 1-day horizon. This is a **single-asset** project —
portfolio-level risk (correlation, diversification) is out of scope; see
`python/equity/03-var-es-engine` in this portfolio for the multi-asset
extension.

## Quickstart

```bash
cd python/foundations/01-risk-metrics
pip install -e ".[dev,plots]"        # numpy, pandas, scipy + pytest, matplotlib
pytest -q                            # 82 tests, offline, seeded, ~1.5 s
python examples/run_pipeline.py      # full report + figures on bundled synthetic data
```

```python
from eq_risk_metrics import simple_returns, var_historical, var_parametric, expected_shortfall
from eq_risk_metrics.data import generate

prices = generate(seed=2).set_index("Date")["Adj Close"]
rets = simple_returns(prices)
var_historical(rets, 0.99)     # 0.0238 (2.38%)
var_parametric(rets, 0.99)     # 0.0201 (2.01%) -- understates the tail
expected_shortfall(rets, 0.99) # 0.0327 (3.27%) -- worse than either VaR
```

To use real market data (recommended before presenting results to
anyone):

```bash
pip install -e ".[live]"
python -m eq_risk_metrics.data.live SPY 2016-01-01   # writes data/SPY.csv
python examples/run_pipeline.py --csv data/SPY.csv
```

## Results summary (bundled synthetic data, seed=2, 2,519 daily returns)

| Metric | Value |
|---|---:|
| Annualised volatility (full sample) | 13.90% |
| Latest EWMA (λ=0.94) volatility | 12.71% |
| Historical VaR, 99% | 2.38% |
| Gaussian VaR, 99% | 2.01% |
| Cornish-Fisher VaR, 99% | 3.12% |
| Expected Shortfall, 99% | 3.27% |
| Maximum drawdown | −27.11% |
| Sharpe ratio (rf = 3%) | 0.33 |
| Sortino ratio (rf = 3%) | 0.47 |
| Excess kurtosis | +5.02 |
| Jarque-Bera p-value | ≈ 0 (normality rejected) |

The four observations these numbers are meant to demonstrate — historical
99% VaR > Gaussian 99% VaR, ES noticeably above VaR, JB rejects normality
decisively, and volatility clusters (21-day rolling vol ranges 5.2%–35.3%
over the sample) — are all confirmed. Full numbers, reproduction
instructions, and known failure modes: `docs/VALIDATION.md`.

## Data

The package ships a **synthetic** generator
(`eq_risk_metrics.data.synthetic.generate`), a two-regime (calm/stressed)
model with Student-t shocks, seeded and deterministic. It exists so the
project runs fully offline and reproduces the stylised facts real daily
equity returns show — fat tails, volatility clustering, occasional
drawdowns — so the analysis is meaningful without a network connection.
Any conclusions presented to a reader should be drawn from real data via
`eq_risk_metrics.data.live` (yfinance, import-guarded — importing the
package never requires yfinance or network access).

When using real data, adjusted close prices are used, so dividends and
splits are already folded into the return series.

## Layout

```
src/eq_risk_metrics/     volatility, var_es, performance, diagnostics,
                         data/{synthetic,live}
tests/                   82 offline seeded pytest tests
examples/run_pipeline.py end-to-end report + figures reproducing the
                         numbers above
docs/                    METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```

## Documentation contract

- **Why these methods, vs alternatives** — [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §1–6
- **Assumptions register (A1–A8, each with "what breaks")** — [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §7
- **Validation evidence** (analytic checks, bundled-data results) — [docs/VALIDATION.md](docs/VALIDATION.md) §1–2
- **Failure modes** (short samples, constant/degenerate returns, regime
  divergence, numerical limits) — [docs/VALIDATION.md](docs/VALIDATION.md) §3
- **Desk usage** (who consumes VaR/ES, reporting, governance, backtesting) — [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md) §1–3
- **Real-life scenarios & edge cases** (limit breach, regime shift, a PM
  arguing with Gaussian VaR, new-listing short samples; every edge case
  unit-tested) — [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md) §4, `tests/test_edge_cases.py`
