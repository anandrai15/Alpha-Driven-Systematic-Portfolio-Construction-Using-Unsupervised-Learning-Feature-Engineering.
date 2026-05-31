#feature model1

import pandas as pd
import numpy as np
import matplotlib
from sklearn import cluster
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime as dt
import sklearn
from sklearn.cluster import KMeans
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import ta
import requests
import pandas as pd
import pathlib as path
import yfinance as yf

#// Importing data and calculating technical indicators //


DATA_PATH = "data/top_100_stocks_daily.csv"
df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"]
)
# enforce types (safety)
df["ticker"] = df["ticker"].astype(str)
# set MultiIndex ONLY if you need it
df = df.set_index(["date", "ticker"]).sort_index()
import warnings
warnings.filterwarnings('ignore')

# RSI
df["rsi_14"] = (
    df.groupby(level="ticker")["adj_close"]
      .transform(lambda x: ta.momentum.RSIIndicator(x, window=14).rsi()))
#print(df.head())

#ATR
df["atr_14"] = (
    df.groupby(level="ticker", group_keys=False)
      .apply(lambda x: ta.volatility.AverageTrueRange(
          high=x["high"],
          low=x["low"],
          close=x["close"],
          window=14
      ).average_true_range())
)
#print(df.tail())
#bollinger bands
df['bb_low'] = df.groupby(level="ticker")["adj_close"].transform(lambda x: ta.volatility.BollingerBands(close=np.log1p(x), window=20).bollinger_lband())
df['bb_mid'] = df.groupby(level="ticker")["adj_close"].transform(lambda x: ta.volatility.BollingerBands(close=np.log1p(x), window=20).bollinger_mavg())
df['bb_high'] = df.groupby(level="ticker")["adj_close"].transform(lambda x: ta.volatility.BollingerBands(close=np.log1p(x), window=20).bollinger_hband())
#print(df.tail())
#MACD
df['macd'] = df.groupby(level="ticker")["adj_close"].transform(lambda x: ta.trend.MACD(close=np.log1p(x)).macd())
df['dollar_volume'] = (df['adj_close'] * df['volume'])/1e6
#print(df.tail())
#Moving average exponential(ema)
df['ema_20'] = df.groupby(level="ticker")['adj_close'].transform(lambda x: ta.trend.EMAIndicator(close=np.log1p(x),window=20).ema_indicator())


print(df.tail())

# // Mental model for feature engineering:
#- technical indicators (RSI, ATR, Bollinger Bands, MACD, EMA)
# Long format → modeling, ML, joins
# Wide format → resampling, rolling, math
# unstack() and stack() are tools to move between them
# You are using them exactly right.
# monthly average (long format)

# // monthly average volume //

monthly_avg_volume = (
    df.groupby([pd.Grouper(level="date", freq="ME"), "ticker"])["volume"]
      .mean()
      .rename("monthly_avg_volume")
)
print(monthly_avg_volume.head())


# // monthly returns/weekly returns //


#resampling to monthly/weekly returns
#one week return
weekly_returns = (
    df["adj_close"]
      .groupby(level="ticker")
      .resample("W", level="date")
      .last()
      .pct_change(1)
      .rename("weekly_return")
      .swaplevel()
      .sort_index()
)
print(weekly_returns.head())
weekly_daily = (
    weekly_returns
      .unstack("ticker")
      .reindex(
          df.index.get_level_values("date").unique(),
          method="ffill"
      )
      .stack()
      .to_frame("weekly_return")
)

df = df.merge(
    weekly_daily,
    left_index=True,
    right_index=True,
    how="left"
)
print(df.tail())


# // Fama-French Factor //

ff = pd.read_csv( "data/F-F_Research_Data_Factors_weekly 2.csv",skiprows=4)
ff = ff[ff.iloc[:, 0].astype(str).str.match(r"^\d{8}$")]


# rename date column
ff.rename(columns={ff.columns[0]: "date"}, inplace=True)
# parse date
ff["date"] = pd.to_datetime(ff["date"], format="%Y%m%d")
# set index
ff.set_index("date", inplace=True)
# convert percent → decimal
ff = ff.astype(float) / 100
ff.columns = ["Mkt-RF", "SMB", "HML", "RF"]


print(ff.head())
#print(ff.tail())
df["excess_return"] = df["weekly_return"] 


# forward-fill FF to all trading days
ff_daily = (
    ff
    .reindex(df.index.get_level_values("date").unique(), method="ffill")
)
#print(ff_daily.head())
df = df.join(ff_daily, on="date")
#print(df.tail())

Y_COL = "excess_return"
X_COLS = ["Mkt-RF", "SMB", "HML"]



# // Rolling Regression //

WINDOW = 52      # 1 year rolling
MIN_OBS = 26     # minimum weeks required

rolling_results = (
    df.dropna(subset=[Y_COL] + X_COLS)
      .groupby(level="ticker", group_keys=False)
      .apply(
          lambda x: RollingOLS(
              endog=x[Y_COL],
              exog=sm.add_constant(x[X_COLS]),
              window=min(WINDOW, len(x)),
              min_nobs=MIN_OBS
          )
          .fit(params_only=True)
          .params
      )
)
#print(rolling_results.tail())
#Separate alphas and betas cleanly

alphas = rolling_results["const"].rename("alpha")

betas = (
    rolling_results
    .drop(columns="const")
    .rename(columns={
        "Mkt-RF": "beta_mkt",
        "SMB": "beta_smb",
        "HML": "beta_hml"
    })
)
print(alphas.tail())
print(betas.tail())

#merge aplhas and betas back to main df 

df = df.join(alphas, how="left")
df = df.join(betas, how="left")

print(df.tail())
#print(ff.columns)



# // MACHINE LEARNING MODEL //

# CLUSTERING, PREDICTION, PORTFOLIO OPTIMIZATION
#- predict which stocks to be included in portfolio
#- which stock to be long or short
#- predict the magnitude of position in each stock
#- which stocks to use in the portfolio based on grouping or clustering algorithms

# k-mean clustering (4 groups is optimal for month)
# create month column
# df['month'] = df.index.get_level_values('date').to_period('M')



from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def get_clusters(g):

    g = g.copy()
    features = ['rsi_14', 'atr_14']
    g = g.dropna(subset=features)

    if len(g) < 4:
        g['cluster'] = np.nan
        return g

    X = g[features].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Centroid Control (RSI Anchored) 
    target_rsi = np.array([30, 45, 55, 70])

    # Convert target RSI into scaled space
    rsi_scaled = (target_rsi - scaler.mean_[0]) / scaler.scale_[0]

    # For ATR, use 0 (mean in scaled space)
    atr_scaled = np.zeros_like(rsi_scaled)

    initial_centroids = np.column_stack((rsi_scaled, atr_scaled))

    kmeans = KMeans(
        n_clusters=4,
        init=initial_centroids,
        n_init=1,   # important when supplying centroids
        random_state=0
    )

    g['cluster'] = kmeans.fit_predict(X_scaled)

    return g

df = df.groupby(level='date', group_keys=False).apply(get_clusters)
print(df.groupby('cluster')['rsi_14'].mean())

latest_month = df.index.get_level_values('date').max()
g = df.xs(latest_month, level='date')

print("Cluster value counts in latest month:")
print(g['cluster'].value_counts(dropna=False))



def plot_month(df, month):

    g = df.xs(month, level='date')
    g = g.dropna(subset=['cluster', 'rsi_14', 'atr_14'])

    color_map = g['cluster'].map({
        0: 'blue',
        1: 'green',
        2: 'orange',
        3: 'black'
    })

    plt.figure(figsize=(8,6))

    plt.scatter(
        g['atr_14'],
        g['rsi_14'],
        c=color_map,
        alpha=0.8
    )

    plt.xlabel("ATR 14")
    plt.ylabel("RSI 14")
    plt.title(f"RSI-Anchored Clusters | {month}")
    plt.savefig(f"clusters_{month}.png", dpi=300, bbox_inches="tight")
    plt.close()

latest_month = df.index.get_level_values('date').max()
plot_month(df, latest_month)




#portfolio optimization using efficient frontier (maximum sharpe)
# rsi around 70 is good momentum, the idea is that it should keep performing

momentum_df= df[df['cluster']==3]
print(momentum_df.tail())

# assume you have returns dataframe (T x N)
# rows = dates, columns = tickers
prices= df['close'].unstack(level='ticker')
monthly_returns= prices.pct_change().dropna()
expected_returns = monthly_returns.mean() * 12   # annualized if monthly

# RISK-MODEL: covariance matrix of returns

cov= monthly_returns.cov()*12
#efficient - frontier portfolio( minimum variance model)
from scipy.optimize import minimize

def portfolio_variance(w, cov):
    return w.T @ cov @ w

def portfolio_return(w, mu):
    return w @ mu


# // Efficient Frontier Setup //

n = len(expected_returns)
w0 = np.ones(n) / n
bounds = [(0, 1) for _ in range(n)]

target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 30)

risks = []
rets = []

for tr in target_returns:

    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, tr=tr: w @ expected_returns- tr}
    )

    result = minimize(
        portfolio_variance,
        w0,
        args=(cov,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        risks.append(np.sqrt(result.fun))
        rets.append(tr)
        w_sharpe = result.x 
print("Optimization success:", result.success)
print("Sharpe weights sum:", w_sharpe.sum())
print("Max Sharpe Return:", portfolio_return(w_sharpe, expected_returns))
print("Max Sharpe Vol:", np.sqrt(portfolio_variance(w_sharpe, cov)))


# Efficient Frontier Plot


plt.figure(figsize=(8, 5))

plt.plot(risks, rets, lw=2, label="Efficient Frontier")

plt.scatter(
    np.sqrt(portfolio_variance(w_sharpe, cov)),
    portfolio_return(w_sharpe, expected_returns),
    s=80,
    marker="*",
    label="Max Sharpe"
)

plt.xlabel("Volatility")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")
plt.legend()

plt.tight_layout()

plt.savefig(
    "efficient_frontier.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

    #calculate daily return for each stck which could land up in our portfolio
    #then loop over each month start , select the stocks for the month and calculate their weights for the next month
    #if the maximum sharpe ratio fails for a given month apply equally weighted weighs
    #calculated each day portfolio return
    # Placeholder to avoid indentation error

df['log_return'] = (
    np.log(df['adj_close'])
      .groupby(level='ticker')
      .diff()
)
print(df[['adj_close', 'log_return']].tail(15))

# 1. Sort data
df = df.sort_index()

# 2. Create price matrix
prices = df['adj_close'].unstack(level='ticker')

#3. Compute log returns
log_returns = np.log(prices).diff().dropna()

portfolio_df = pd.DataFrame()
fixed_dates = (
    momentum_df
        .index
        .get_level_values('date')
        .to_series()
        .resample('ME')
        .last()
        .dropna()
        .values
)

portfolio_df = pd.DataFrame()
for start_date in fixed_dates:

    optimization_start_date = start_date - pd.DateOffset(months=12)
    optimization_end_date = start_date
    end_date = start_date + pd.offsets.MonthEnd(1)

    month_slice = momentum_df.xs(start_date, level='date')
    momentum_stocks = month_slice.index.tolist()

    if len(momentum_stocks) < 2:
        continue

    train_data = log_returns.loc[
        (log_returns.index >= optimization_start_date) &
        (log_returns.index < optimization_end_date),
        momentum_stocks
    ].dropna(how='any')

    if train_data.shape[0] < 30:
        continue

    expected_returns = train_data.mean() * 12
    cov = train_data.cov() * 12

    n = len(expected_returns)
    w0 = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]

    def negative_sharpe(w):
        port_return = w @ expected_returns
        port_vol = np.sqrt(w @ cov @ w)
        return -(port_return / port_vol)

    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })

    result = minimize(
        negative_sharpe,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        optimized_weights = pd.Series(result.x, index=expected_returns.index)
    else:
        optimized_weights = pd.Series(np.ones(n)/n, index=expected_returns.index)

    optimized_weights = optimized_weights.to_frame(name='weight')

    print("\nOptimized Weights:")
    print(optimized_weights.sort_values(by='weight', ascending=False))
    print("Sum:", optimized_weights['weight'].sum())

    # ✅ FORWARD RETURNS (NOW INSIDE LOOP)
    temp_df = log_returns.loc[start_date:end_date, momentum_stocks]

    temp_df = (
        temp_df
            .stack()
            .to_frame('return')
            .reset_index(level=0)
            .merge(optimized_weights, left_on='ticker', right_index=True)
    )

    temp_df['weighted_return'] = temp_df['return'] * temp_df['weight']

    monthly_return = temp_df['weighted_return'].sum()

    portfolio_df = pd.concat([
        portfolio_df,
        pd.DataFrame({
            'date': [end_date],
            'strategy_return': [monthly_return]
        })
    ], axis=0)

print(portfolio_df)

#cleaning dataframe
portfolio_df = portfolio_df.sort_values('date')
portfolio_df = portfolio_df.set_index('date')

#plot monhtly strategy return
#plt.figure()
#portfolio_df['strategy_return'].plot()
#plt.title("Monthly Strategy Return")
#plt.xlabel("Date")
#plt.ylabel("Return")
#plt.show()


#n = len(expected_returns)
equal_weights = pd.DataFrame(
    1/n,
    index=expected_returns.index,
    columns=['weight']
)
print(equal_weights)
print("Sum:", equal_weights['weight'].sum())

# // Visualize portfolio returns and compare to SP500 returns //

spy = yf.download('SPY', start='2015-01-01')

spy_ret = np.log(spy['Close']).diff().dropna()

spy_ret = spy_ret.squeeze()   # ensures it's a Series
spy_ret.name = 'SPY Buy&Hold'
portfolio_df = (
    portfolio_df
        .groupby('date')['strategy_return']
        .sum()
        .to_frame()
)

spy_ret = spy_ret.resample('ME').sum()
portfolio_df = portfolio_df.join(spy_ret)
print(portfolio_df)

# // calculate cumulative return for startegy_return and spy buy_n_hold //

import matplotlib.ticker as mtick

plt.style.use('ggplot')
portfolio_cumulative_return = (1 + portfolio_df).cumprod()
portfolio_cumulative_return.loc['2015-01-01':].plot(figsize=(16,6))
plt.title('strategy Returns Over Time')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.ylabel('Return')
plt.savefig("cumulative_returns.png", dpi=300, bbox_inches="tight")
plt.show()
print(portfolio_cumulative_return)


# // Performance Statistics //


strategy = portfolio_df["strategy_return"].dropna()

cagr = (
    portfolio_cumulative_return["strategy_return"].iloc[-1]
) ** (12 / len(strategy)) - 1

volatility = strategy.std() * np.sqrt(12)

sharpe = strategy.mean() / strategy.std() * np.sqrt(12)

max_drawdown = (
    portfolio_cumulative_return["strategy_return"] /
    portfolio_cumulative_return["strategy_return"].cummax() - 1
).min()

print(f"CAGR: {cagr:.2%}")
print(f"Volatility: {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
