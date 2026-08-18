"""
Generate a realistic synthetic daily price series for demo purposes.

This exists so the project runs out-of-the-box without an internet
connection. It is NOT real market data -- run download_data.py to
replace it with real prices before presenting results.

The generator uses a two-regime model (calm / stressed) with
Student-t shocks to reproduce the stylised facts real returns show:
fat tails, volatility clustering, and occasional drawdowns.
"""
import numpy as np
import pandas as pd


def generate(n_days: int = 2520, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Regime 0: calm (most of the time), Regime 1: stressed
    mu = {0: 0.10 / 252, 1: -0.15 / 252}            # daily drift
    sigma = {0: 0.12 / np.sqrt(252), 1: 0.28 / np.sqrt(252)}
    p_stay = {0: 0.995, 1: 0.95}                    # regime persistence

    state = 0
    rets = np.empty(n_days)
    for t in range(n_days):
        if rng.random() > p_stay[state]:
            state = 1 - state
        # Student-t shocks (df=5) rescaled to unit variance -> fat tails
        shock = rng.standard_t(df=5) / np.sqrt(5 / 3)
        rets[t] = mu[state] + sigma[state] * shock

    prices = 100 * np.exp(np.cumsum(rets))
    dates = pd.bdate_range(end="2026-08-14", periods=n_days)
    return pd.DataFrame({"Date": dates, "Adj Close": prices.round(4)})


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/sample_prices.csv", index=False)
    print(f"Wrote {len(df)} rows to data/sample_prices.csv")
