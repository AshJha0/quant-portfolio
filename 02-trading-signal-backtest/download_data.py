"""
Download real historical prices with yfinance (run this on your own
machine with internet access):

    pip install yfinance
    python download_data.py SPY 2016-01-01

Then rerun the analysis on real data:

    python analysis.py data/SPY.csv
"""
import sys


def main(ticker: str = "SPY", start: str = "2016-01-01") -> None:
    import yfinance as yf  # imported here so the rest of the project
    #                        works without yfinance installed
    df = yf.download(ticker, start=start, auto_adjust=True)
    out = df[["Close"]].rename(columns={"Close": "Adj Close"})
    out = out.reset_index()
    # yfinance sometimes returns MultiIndex columns; flatten them
    out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    path = f"data/{ticker}.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} rows to {path}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    start = sys.argv[2] if len(sys.argv) > 2 else "2016-01-01"
    main(ticker, start)
