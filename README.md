# Alpha Driven Unsupervised Momentum Portfolio with Factor Analysis and Rolling Sharpe Optimization

A quantitative research project that combines **technical feature engineering**, **Fama-French factor modeling**, **unsupervised clustering**, and **rolling portfolio optimization** to build a monthly rotating equity strategy. The model selects a momentum-oriented stock cluster, estimates risk and return from a rolling lookback window, and allocates capital using a **long-only maximum Sharpe ratio** framework.

---

## Overview

This project asks a simple quant question:

> Can portfolio selection be improved by combining market features, factor exposure, clustering, and optimization?

Instead of manually choosing stocks, the pipeline:

1. builds technical indicators for each stock,
2. estimates rolling alpha and factor betas,
3. groups stocks into regimes using clustering,
4. selects the strongest momentum cluster,
5. optimizes portfolio weights each month,
6. and evaluates performance out of sample.

The result is a full research workflow that is interpretable, repeatable, and easy to extend.

---

## Why This Strategy Was Selected

This strategy was chosen because it combines several ideas that matter in real quant research:

### Momentum
Momentum is one of the most studied market effects, but it is unstable across time. Clustering helps isolate the subset of stocks currently exhibiting strong trend behavior.

### Technical Features
Indicators such as **RSI**, **ATR**, **MACD**, **EMA**, and **Bollinger Bands** help capture trend strength, volatility, and price extension.

### Factor Awareness
Rolling Fama-French regression adds interpretability by measuring how each stock behaves relative to market, size, and value factors.

### Optimization
A rolling **maximum Sharpe** optimizer dynamically adapts portfolio weights based on recent return and risk estimates.

### Benchmarks
The strategy is compared against **equal-weight** and **SPY buy-and-hold** so the results are meaningful, not just visually attractive.

---

## What the Project Covers

- Technical indicators for feature engineering
- Weekly and monthly return construction
- Fama-French 3-factor modeling
- Rolling regression with `RollingOLS`
- Monthly cross-sectional clustering with `KMeans`
- RSI-anchored cluster initialization
- Momentum regime selection
- Mean-variance portfolio optimization
- Maximum Sharpe ratio allocation
- Out-of-sample monthly backtesting
- Equal-weight benchmarking
- SPY benchmark comparison
- Efficient frontier visualization
- Cumulative return analysis
- Saved performance plots

---

## Data Sources

The project uses:

- **Daily OHLCV stock data** for the stock universe
- **Weekly Fama-French factor data**
- **SPY** as the market benchmark

The data is transformed into a MultiIndex structure to support grouped feature engineering, rolling regression, and monthly rebalancing.

---

## Methodology

### 1. Feature Engineering
Each stock is enriched with indicators that capture different aspects of market behavior:

- **RSI (14)** — momentum and overbought/oversold state
- **ATR (14)** — volatility regime
- **Bollinger Bands** — deviation from recent mean
- **MACD** — trend momentum
- **EMA (20)** — short-term trend
- **Dollar Volume** — liquidity proxy

These features create a compact representation of stock behavior for clustering.

### 2. Weekly Returns and Factor Alignment
Weekly returns are computed and aligned with Fama-French factors. This enables dynamic estimation of:

- alpha
- market beta
- size beta
- value beta

using a rolling window.

### 3. Rolling Factor Modeling
A rolling regression is run for each stock using a one-year lookback. This produces time-varying estimates of alpha and betas, making the model more interpretable and more realistic than a static factor fit.

### 4. Cross-Sectional Clustering
Stocks are clustered each month using **RSI** and **ATR**. The centroids are initialized around RSI levels so the clusters remain interpretable.

The clusters broadly separate stocks into:

- weak momentum,
- neutral behavior,
- moderate trend,
- strong momentum.

The highest momentum cluster is used as the investable universe.

### 5. Rolling Portfolio Optimization
For each rebalance date:

- the previous 12 months of returns are used as the training window,
- expected returns and covariance are estimated,
- a long-only maximum Sharpe portfolio is solved,
- and the weights are applied to the next month.

This gives a true out-of-sample monthly backtest.

### 6. Benchmarking
The strategy is evaluated against:

- **Equal-weight portfolio**
- **SPY buy-and-hold**

This is important because a strategy only matters if it outperforms simple baselines.

---

## Results

The backtest produced **90 monthly observations**.

### Strategy Performance
- **CAGR:** 25.10%
- **Volatility:** 21.69%
- **Sharpe Ratio:** 1.15
- **Max Drawdown:** -23.35%

### Interpretation
The strategy delivered:
- strong long-run growth,
- positive risk-adjusted performance,
- and a drawdown profile that is reasonable for an equity strategy, though still meaningful.

A Sharpe ratio above 1.0 is a good sign for a systematic strategy, especially one built from a relatively simple monthly rotation framework. The drawdown shows that the model is not overly smooth, which is realistic for a momentum-driven portfolio.

---

## Conclusion

This strategy suggests that a **momentum + clustering + rolling optimization** framework can produce a meaningful equity portfolio with solid risk-adjusted performance.

The main takeaway is not just the return itself, but the fact that the pipeline is:
- systematic,
- explainable,
- adaptive through time,
- and benchmarked against simpler alternatives.

At the same time, the drawdown reminds us that maximum Sharpe optimization can still produce unstable allocations if market conditions change. That makes the project a strong foundation for further research such as:
- transaction cost modeling,
- turnover reduction,
- covariance shrinkage,
- sector-neutral constraints,
- or walk-forward parameter tuning.

---
