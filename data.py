import yfinance as yf
import pandas as pd
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
TOP_100_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","BRK-B","UNH",
    "XOM","JNJ","JPM","V","PG","MA","HD","AVGO","CVX","LLY",
    "MRK","COST","ABBV","PEP","KO","ADBE","WMT","CRM","BAC","NFLX",
    "ACN","TMO","ORCL","MCD","DHR","LIN","CSCO","AMD","INTU","TXN",
    "QCOM","IBM","PM","AMAT","GE","CAT","NOW","GS","SPGI","ISRG",
    "BLK","INTC","AXP","LOW","BKNG","MS","PLD","ELV","ADP","VRTX",
    "LMT","MDLZ","SCHW","C","GILD","ADI","MMC","MO","CI","SYK",
    "CB","ZTS","TGT","REGN","PNC","DUK","SO","AON","BDX","NSC",
    "ITW","EMR","USB","CL","EQIX","APD","SHW","EOG","HUM","ICE"
]

START_DATE = "2010-01-01"
END_DATE = "2024-06-30"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUT_FILE = DATA_DIR / "top_100_stocks_daily.csv"

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
