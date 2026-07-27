# FEATURE ENGINEERING + FAMA-FRENCH MODEL

import pandas as pd
import numpy as np
import ta
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/nasdaq_100_daily.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

df["ticker"] = df["ticker"].astype(str)
df = df.set_index(["date", "ticker"]).sort_index()

print("Data loaded:", df.shape)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# RSI - short-term momentum
df["rsi_14"] = df.groupby(level="ticker")["adj_close"].transform(lambda x: ta.momentum.RSIIndicator(x, window=14).rsi())

# ATR - absolute volatility
df["atr_14"] = df.groupby(level="ticker", group_keys=False).apply(lambda x: ta.volatility.AverageTrueRange(high=x["high"], low=x["low"], close=x["close"], window=14).average_true_range())

# ATR % - volatility relative to stock price
df["atr_pct"] = df["atr_14"] / df["adj_close"]

# 3-month momentum
df["momentum_3m"] = df.groupby(level="ticker")["adj_close"].pct_change(63)

# MACD - trend momentum
df["macd"] = df.groupby(level="ticker")["adj_close"].transform(lambda x: ta.trend.MACD(close=x).macd())

# Dollar volume - liquidity
df["dollar_volume"] = (df["adj_close"] * df["volume"]) / 1_000_000

print("\nFEATURES:")
print(df[["rsi_14", "atr_pct", "momentum_3m", "macd", "dollar_volume"]].tail())


# ============================================================
# 3. CREATE WEEKLY STOCK DATA
# ============================================================

weekly_prices = df["adj_close"].unstack("ticker").resample("W-FRI").last()
weekly_returns = weekly_prices.pct_change()

print("\nWEEKLY RETURNS:")
print(weekly_returns.tail())


# ============================================================
# 4. LOAD FAMA-FRENCH FACTORS
# ============================================================

ff = pd.read_csv("data/F-F_Research_Data_Factors_weekly 2.csv", skiprows=4)
ff = ff[ff.iloc[:, 0].astype(str).str.match(r"^\d{8}$")].copy()
ff.rename(columns={ff.columns[0]: "date"}, inplace=True)
ff["date"] = pd.to_datetime(ff["date"], format="%Y%m%d")
ff = ff.set_index("date").astype(float) / 100
ff.columns = ["Mkt-RF", "SMB", "HML", "RF"]

print("\nFAMA-FRENCH:")
print(ff.tail())


# ============================================================
# 5. ALIGN STOCK RETURNS WITH FAMA-FRENCH
# ============================================================

weekly_long = weekly_returns.stack().rename("weekly_return").to_frame()
weekly_long.index.names = ["date", "ticker"]

weekly_long = weekly_long.join(ff, on="date")

# Actual excess return = stock return - risk-free rate
weekly_long["excess_return"] = weekly_long["weekly_return"] - weekly_long["RF"]

weekly_long = weekly_long.dropna(subset=["excess_return", "Mkt-RF", "SMB", "HML"])

print("\nWEEKLY FACTOR DATA:")
print(weekly_long.tail())


# ============================================================
# 6. ROLLING FAMA-FRENCH REGRESSION
# ============================================================

WINDOW = 52
MIN_OBS = 26

def run_rolling_regression(stock):
    stock = stock.sort_index()
    X = sm.add_constant(stock[["Mkt-RF", "SMB", "HML"]])
    y = stock["excess_return"]
    return RollingOLS(y, X, window=WINDOW, min_nobs=MIN_OBS).fit(params_only=True).params

factor_results = weekly_long.groupby(level="ticker", group_keys=False).apply(run_rolling_regression)

factor_results = factor_results.rename(columns={"const": "alpha", "Mkt-RF": "beta_mkt", "SMB": "beta_smb", "HML": "beta_hml"})

print("\nFACTOR RESULTS:")
print(factor_results.tail(20))


# // MACHINE LEARNING MODEL //

# CLUSTERING, PREDICTION, PORTFOLIO OPTIMIZATION
#- predict which stocks to be included in portfolio
#- which stock to be long or short
#- predict the magnitude of position in each stock
#- which stocks to use in the portfolio based on grouping or clustering algorithms

# k-mean clustering (4 groups is optimal for month)
# create month column
# df['month'] = df.index.get_level_values('date').to_period('M')



# ============================================================
# 7. CREATE MONTHLY FEATURE DATA
# ============================================================

# Take the last available feature observation of each month
monthly_features = df[["rsi_14", "atr_pct", "momentum_3m", "macd", "dollar_volume"]].groupby([pd.Grouper(level="date", freq="ME"), "ticker"]).last()

# Convert weekly Fama-French results to monthly
monthly_factors = factor_results.groupby([pd.Grouper(level="date", freq="ME"), "ticker"]).last()

# Combine technical + factor features
monthly_data = monthly_features.join(monthly_factors, how="left")

print("\nMONTHLY DATA:")
print(monthly_data.tail())


# ============================================================
# 8. KMEANS CLUSTERING
# ============================================================

# Features used by KMeans
CLUSTER_FEATURES = ["rsi_14", "atr_pct", "momentum_3m", "alpha", "beta_mkt", "beta_smb", "beta_hml"]

def get_clusters(group):
    group = group.copy()
    clean = group.dropna(subset=CLUSTER_FEATURES)
    if len(clean) < 4:
        group["cluster"] = np.nan
        return group
    scaler = StandardScaler()
    X = scaler.fit_transform(clean[CLUSTER_FEATURES])
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    group.loc[clean.index, "cluster"] = model.fit_predict(X)
    return group

monthly_data = monthly_data.groupby(level="date", group_keys=False).apply(get_clusters)

print("\nCLUSTER COUNTS:")
print(monthly_data["cluster"].value_counts())


# ============================================================
# 9. IDENTIFY MOMENTUM CLUSTER
# ============================================================

# KMeans cluster numbers have no fixed meaning, so find the cluster
# with the highest average 3-month momentum each month
def select_momentum_cluster(group):
    if group["cluster"].dropna().empty:
        group["selected"] = False
        return group
    cluster_momentum = group.groupby("cluster")["momentum_3m"].mean()
    best_cluster = cluster_momentum.idxmax()
    group["selected"] = group["cluster"] == best_cluster
    return group

monthly_data = monthly_data.groupby(level="date", group_keys=False).apply(select_momentum_cluster)

momentum_df = monthly_data[monthly_data["selected"] == True]

print("\nSELECTED MOMENTUM STOCKS:")
print(momentum_df.tail(20))


# ============================================================
# 10. CLUSTER VISUALIZATION
# ============================================================

latest_month = monthly_data.index.get_level_values("date").max()
latest_data = monthly_data.xs(latest_month, level="date").dropna(subset=["cluster"])

plt.figure(figsize=(8, 6))
plt.scatter(latest_data["atr_pct"], latest_data["momentum_3m"], c=latest_data["cluster"], alpha=0.8)
plt.xlabel("ATR %")
plt.ylabel("3-Month Momentum")
plt.title(f"NASDAQ-100 Momentum Clusters | {latest_month.date()}")
plt.tight_layout()
plt.savefig("clusters_latest.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 11. DAILY RETURNS
# ============================================================

prices = df["adj_close"].unstack("ticker").sort_index()
daily_returns = prices.pct_change()

print("\nDAILY RETURNS:")
print(daily_returns.tail())


# ============================================================
# 12. MAXIMUM SHARPE OPTIMIZER
# ============================================================

def optimize_weights(returns):
    expected_returns = returns.mean() * 252
    covariance = returns.cov() * 252
    n = len(expected_returns)
    initial_weights = np.ones(n) / n
    bounds = [(0, 0.25)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    def negative_sharpe(w):
        portfolio_return = w @ expected_returns
        portfolio_volatility = np.sqrt(w @ covariance @ w)
        return -(portfolio_return / portfolio_volatility)
    result = minimize(negative_sharpe, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        return pd.Series(result.x, index=expected_returns.index)
    return pd.Series(initial_weights, index=expected_returns.index)


# ============================================================
# 13. MONTHLY WALK-FORWARD BACKTEST
# ============================================================

portfolio_results = []

rebalance_dates = momentum_df.index.get_level_values("date").unique().sort_values()

for rebalance_date in rebalance_dates:

    selected_stocks = momentum_df.xs(rebalance_date, level="date").index.tolist()

    if len(selected_stocks) < 4:
        continue

    training_start = rebalance_date - pd.DateOffset(months=12)

    training_returns = daily_returns.loc[(daily_returns.index >= training_start) & (daily_returns.index < rebalance_date), selected_stocks].dropna(axis=1, how="any")

    if training_returns.shape[0] < 126 or training_returns.shape[1] < 4:
        continue

    weights = optimize_weights(training_returns)

    holding_start = rebalance_date + pd.Timedelta(days=1)
    holding_end = rebalance_date + pd.offsets.MonthEnd(1)

    forward_returns = daily_returns.loc[holding_start:holding_end, weights.index]

    if forward_returns.empty:
        continue

    daily_portfolio_returns = forward_returns.mul(weights, axis=1).sum(axis=1)

    monthly_return = (1 + daily_portfolio_returns).prod() - 1

    portfolio_results.append({"date": holding_end, "strategy_return": monthly_return, "stocks": len(weights)})


# ============================================================
# 14. CREATE PORTFOLIO DATAFRAME
# ============================================================

portfolio_df = pd.DataFrame(portfolio_results).set_index("date").sort_index()

print("\nPORTFOLIO RETURNS:")
print(portfolio_df.tail())


# ============================================================
# 15. NASDAQ-100 BENCHMARK
# ============================================================

import yfinance as yf

benchmark = yf.download("^NDX", start=portfolio_df.index.min(), end=portfolio_df.index.max() + pd.Timedelta(days=5), auto_adjust=True, progress=False)

benchmark_returns = benchmark["Close"].squeeze().pct_change()
benchmark_monthly = benchmark_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
benchmark_monthly.name = "NASDAQ-100"

portfolio_df = portfolio_df.join(benchmark_monthly)


# ============================================================
# 16. CUMULATIVE RETURNS
# ============================================================

cumulative_returns = (1 + portfolio_df[["strategy_return", "NASDAQ-100"]]).cumprod()

plt.figure(figsize=(14, 6))
plt.plot(cumulative_returns.index, cumulative_returns["strategy_return"], label="Strategy")
plt.plot(cumulative_returns.index, cumulative_returns["NASDAQ-100"], label="NASDAQ-100")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.title("Strategy vs NASDAQ-100")
plt.legend()
plt.tight_layout()
plt.savefig("cumulative_returns.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nCUMULATIVE RETURNS:")
print(cumulative_returns.tail())


# ============================================================
# 17. PERFORMANCE STATISTICS
# ============================================================

strategy = portfolio_df["strategy_return"].dropna()
benchmark = portfolio_df["NASDAQ-100"].dropna()

# CAGR
years = len(strategy) / 12
cagr = (1 + strategy).prod() ** (1 / years) - 1

# Annualized volatility
volatility = strategy.std() * np.sqrt(12)

# Sharpe ratio
sharpe = (strategy.mean() / strategy.std()) * np.sqrt(12)

# Downside volatility
downside_returns = strategy[strategy < 0]
downside_volatility = downside_returns.std() * np.sqrt(12)

# Sortino ratio
sortino = (strategy.mean() * 12) / downside_volatility

# Maximum drawdown
wealth = (1 + strategy).cumprod()
running_max = wealth.cummax()
drawdown = wealth / running_max - 1
max_drawdown = drawdown.min()

# Calmar ratio
calmar = cagr / abs(max_drawdown)

# Win rate
win_rate = (strategy > 0).mean()

# Benchmark CAGR
benchmark_years = len(benchmark) / 12
benchmark_cagr = (1 + benchmark).prod() ** (1 / benchmark_years) - 1

# Benchmark volatility
benchmark_volatility = benchmark.std() * np.sqrt(12)

# Benchmark Sharpe
benchmark_sharpe = (benchmark.mean() / benchmark.std()) * np.sqrt(12)

# Strategy beta vs NASDAQ-100
aligned = portfolio_df[["strategy_return", "NASDAQ-100"]].dropna()
beta = aligned["strategy_return"].cov(aligned["NASDAQ-100"]) / aligned["NASDAQ-100"].var()

# Annualized alpha
alpha = (aligned["strategy_return"].mean() - beta * aligned["NASDAQ-100"].mean()) * 12

# Print results
print("\n" + "=" * 50)
print("PORTFOLIO PERFORMANCE")
print("=" * 50)
print(f"CAGR:                 {cagr:.2%}")
print(f"Annual Volatility:    {volatility:.2%}")
print(f"Sharpe Ratio:         {sharpe:.2f}")
print(f"Sortino Ratio:        {sortino:.2f}")
print(f"Maximum Drawdown:     {max_drawdown:.2%}")
print(f"Calmar Ratio:         {calmar:.2f}")
print(f"Positive Months:      {win_rate:.2%}")
print("-" * 50)
print("NASDAQ-100 BENCHMARK")
print("-" * 50)
print(f"Benchmark CAGR:       {benchmark_cagr:.2%}")
print(f"Benchmark Volatility: {benchmark_volatility:.2%}")
print(f"Benchmark Sharpe:     {benchmark_sharpe:.2f}")
print("-" * 50)
print("RELATIVE PERFORMANCE")
print("-" * 50)
print(f"Alpha:                {alpha:.2%}")
print(f"Beta:                 {beta:.2f}")
print("=" * 50)

# ============================================================
# 18. FINAL VISUALIZATIONS
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# 1. CUMULATIVE WEALTH
cumulative = (1 + portfolio_df[["strategy_return", "NASDAQ-100"]]).cumprod()

plt.figure(figsize=(14, 6))
plt.plot(cumulative.index, cumulative["strategy_return"], label="Strategy", linewidth=2)
plt.plot(cumulative.index, cumulative["NASDAQ-100"], label="NASDAQ-100", linewidth=2)
plt.title("Systematic Portfolio vs NASDAQ-100")
plt.xlabel("Year")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/01_cumulative_returns.png", dpi=300, bbox_inches="tight")
plt.close()


# 2. DRAWDOWN
strategy_wealth = (1 + strategy).cumprod()
strategy_drawdown = strategy_wealth / strategy_wealth.cummax() - 1

benchmark_wealth = (1 + benchmark).cumprod()
benchmark_drawdown = benchmark_wealth / benchmark_wealth.cummax() - 1

plt.figure(figsize=(14, 5))
plt.plot(strategy_drawdown.index, strategy_drawdown, label="Strategy")
plt.plot(benchmark_drawdown.index, benchmark_drawdown, label="NASDAQ-100")
plt.axhline(0, linewidth=1)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.title("Portfolio Drawdown")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/02_drawdown.png", dpi=300, bbox_inches="tight")
plt.close()


# 3. ROLLING 12-MONTH SHARPE
rolling_sharpe = strategy.rolling(12).mean() / strategy.rolling(12).std() * np.sqrt(12)

plt.figure(figsize=(14, 5))
plt.plot(rolling_sharpe.index, rolling_sharpe)
plt.axhline(0, linewidth=1)
plt.axhline(1, linestyle="--", linewidth=1)
plt.title("Rolling 12-Month Sharpe Ratio")
plt.ylabel("Sharpe Ratio")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/03_rolling_sharpe.png", dpi=300, bbox_inches="tight")
plt.close()


# 4. MONTHLY RETURNS
plt.figure(figsize=(14, 5))
plt.bar(strategy.index, strategy.values, width=20)
plt.axhline(0, linewidth=1)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.title("Strategy Monthly Returns")
plt.ylabel("Monthly Return")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("results/04_monthly_returns.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 10. CLUSTER VISUALIZATION
# ============================================================

valid_cluster_data = monthly_data.dropna(subset=["cluster"])

latest_cluster_month = valid_cluster_data.index.get_level_values("date").max()

latest_data = valid_cluster_data.xs(latest_cluster_month, level="date")

print(f"\nLatest valid cluster month: {latest_cluster_month.date()}")
print(f"Stocks available for plot: {len(latest_data)}")
print("\nCluster counts:")
print(latest_data["cluster"].value_counts().sort_index())

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    latest_data["atr_pct"],
    latest_data["momentum_3m"],
    c=latest_data["cluster"],
    cmap="viridis",
    s=70,
    alpha=0.8
)

plt.xlabel("ATR %")
plt.ylabel("3-Month Momentum")
plt.title(f"NASDAQ-100 KMeans Clusters | {latest_cluster_month.strftime('%B %Y')}")
plt.colorbar(scatter, label="Cluster")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/clusters_latest.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nPlots saved:")
print("results/01_cumulative_returns.png")
print("results/02_drawdown.png")
print("03_rolling_sharpe.png")
print("04_monthly_returns.png")
print("clusters_latest.png")
