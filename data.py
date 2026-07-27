import yfinance as yf
import pandas as pd
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
NASDAQ_100_TICKERS = [
    "ADBE", "AMD", "ABNB", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN",
    "ADI", "ANSS", "AAPL", "AMAT", "ARM", "ASML", "AZN", "TEAM",
    "ADSK", "ADP", "AXON", "BKR", "BIIB", "BKNG", "AVGO", "CDNS",
    "CDW", "CHTR", "CTAS", "CSCO", "CCEP", "CTSH", "CMCSA", "CEG",
    "CPRT", "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM", "FANG",
    "DASH", "EA", "EXC", "FAST", "FTNT", "GEHC", "GILD", "GFS",
    "HON", "IDXX", "INTC", "INTU", "ISRG", "KDP", "KLAC", "KHC",
    "LRCX", "LIN", "LULU", "MAR", "MRVL", "MELI", "META", "MCHP",
    "MU", "MSFT", "MSTR", "MDLZ", "MDB", "MNST", "NFLX", "NVDA",
    "NXPI", "ORLY", "ODFL", "ON", "PCAR", "PANW", "PAYX", "PYPL",
    "PDD", "PEP", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI",
    "SNPS", "TTWO", "TMUS", "TSLA", "TXN", "TTD", "VRSK", "VRTX",
    "WBD", "WDAY", "XEL", "ZS"
]

START_DATE = "2010-01-01"
END_DATE = "2026-08-30"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUT_FILE = DATA_DIR / "NASDAQ_100_TICKERS.csv"

# ----------------------------
# DOWNLOAD
# ----------------------------
print("Downloading data from Yahoo Finance...")

raw = yf.download(
    tickers=TOP_100_TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    group_by="ticker",
    threads=False,          # safer for large batches
    progress=True
)

# ----------------------------
# RESHAPE TO CLEAN LONG FORMAT
# ----------------------------
df = (
    raw.stack(level=0)
       .reset_index()
       .rename(columns={"level_1": "ticker"})
)

# normalize column names
df.columns = (
    df.columns
      .str.lower()
      .str.replace(" ", "_")
)

# ----------------------------
# CLEAN FAILURES
# ----------------------------
# drop rows where Yahoo returned junk / missing data
df = df.dropna(subset=["adj_close"])

# enforce types
df["date"] = pd.to_datetime(df["date"])
df["ticker"] = df["ticker"].astype(str)

# sort for sanity
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

# ----------------------------
# SAVE
# ----------------------------
df.to_csv(OUT_FILE, index=False)

print("\n✅ DONE")
print(f"Saved file: {OUT_FILE.resolve()}")
print(f"Rows: {len(df):,}")
print(f"Tickers: {df['ticker'].nunique()}")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
