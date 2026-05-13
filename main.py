import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

data = yf.download(tickers, start="2018-01-01")["Close"]

returns = data.pct_change().dropna()

weights = [1/len(tickers)] * len(tickers)

portfolio_returns = returns.dot(weights)

cumulative = (1 + portfolio_returns).cumprod()

plt.plot(cumulative)
plt.title("Equal Weight Portfolio")
plt.show()
