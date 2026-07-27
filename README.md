# Alpha-Driven Systematic Momentum Portfolio Construction & Management Using Unsupervised Learning, Factor Analysis and Rolling Sharpe Optimization

A quantitative research project combining **technical feature engineering, Fama–French factor modeling, unsupervised learning, and rolling portfolio optimization** to construct a systematic momentum portfolio from the **NASDAQ-100 universe**.

The model identifies momentum-oriented groups of stocks using K-Means clustering, estimates dynamic factor exposures, and allocates capital using a constrained **long-only maximum Sharpe ratio** framework.

---

## Overview

This project investigates a simple quantitative research question:

 ''Can portfolio selection be improved by combining momentum, volatility, factor exposures, unsupervised clustering, and portfolio optimization?''

Rather than manually selecting securities, the framework:

1. constructs market and momentum features for each stock,
2. estimates rolling Fama–French alpha and factor betas,
3. creates monthly cross-sectional observations,
4. groups stocks using K-Means clustering,
5. dynamically identifies the strongest momentum cluster,
6. optimizes portfolio weights using historical risk and return,
7. holds the resulting portfolio during the following month,
8. and evaluates performance against the NASDAQ-100.

The result is an interpretable and reproducible systematic portfolio research framework.

---

## Why This Strategy Was Selected

### Momentum

Momentum is one of the most widely documented effects in financial markets, but its strength varies across securities and market regimes.

Rather than applying a fixed momentum threshold, clustering allows the model to identify groups of stocks currently exhibiting similar characteristics.

### Technical & Market Features

The framework uses:

* RSI (14) — short-term momentum condition
* ATR % — volatility normalized by stock price
* 3-Month Momentum** — medium-term price strength
* MACD — trend and momentum structure
* Dollar Volume** — liquidity proxy

ATR is normalized by price to make volatility comparable across securities with different nominal prices.

### Factor Awareness

A rolling **Fama–French three-factor regression** estimates each stock's:

* Alpha
* Market Beta
* SMB Beta
* HML Beta

This allows clustering to consider both **price behavior and systematic factor exposure**.

### Unsupervised Learning

K-Means is used to identify cross-sectional groups of stocks without assigning predefined labels.

Because K-Means cluster numbers have no persistent economic meaning, the model dynamically identifies the cluster with the **highest average 3-month momentum** at every rebalance.

### Portfolio Optimization

Stocks belonging to the selected cluster are allocated using a rolling **maximum Sharpe ratio optimizer** based on the previous 12 months of daily returns.

A maximum 25% position constraint is imposed to reduce excessive concentration.

---

## Strategy Architecture

NASDAQ-100 Universe
        ↓
Feature Engineering
        ↓
Fama–French 3-Factor Model
        ↓
Rolling Alpha & Factor Betas
        ↓
Monthly Cross-Section
        ↓
Feature Standardization
        ↓
K-Means Clustering
        ↓
Momentum Cluster Selection
        ↓
Maximum-Sharpe Optimization
        ↓
Next-Month Portfolio
        ↓
Walk-Forward Backtest
        ↓
Performance & Risk Analysis

---

## Data

The research uses:

* Daily OHLCV equity data
* Weekly Fama–French factor data
* NASDAQ-100 index data as the primary benchmark

The equity sample begins in **2010**, providing observations across multiple market and volatility regimes.

Adjusted prices are used for return calculations.

The raw equity dataset can be generated using data.py, while the Fama–French factor file is stored inside the data/ directory.

---

## Methodology

### 1. Feature Engineering

Each security is transformed into a set of market characteristics:
RSI
ATR %
3-Month Momentum
MACD
Dollar Volume

These variables capture momentum, volatility, trend behavior and liquidity.

---

### 2. Fama–French Factor Modeling
Weekly stock returns are aligned with weekly Fama–French observations.

Stock excess return is calculated as:
[Ri- Rf]

A rolling three-factor regression is then estimated:

[Ri- Rf =
\alpha_i +
\beta_M(MKT-RF) +
\beta_S SMB +
\beta_H HML +
\epsilon_i]

using a **52-week rolling window**.

This generates time-varying estimates of:

Alpha
Market Beta
SMB Beta
HML Beta

---

### 3. Monthly Cross-Section

Technical features and rolling factor estimates are converted into monthly observations.

The clustering model uses:

RSI
ATR %
3-Month Momentum
Alpha
Market Beta
SMB Beta
HML Beta

Features are standardized before clustering to prevent variables with larger numerical scales from dominating the model.

---

### 4. K-Means Clustering

The NASDAQ-100 universe is divided into **four cross-sectional clusters each month**.

Instead of assuming that a particular cluster number always represents momentum, the model calculates the average 3-month momentum of each cluster.

The cluster exhibiting the **highest average momentum** becomes the investable universe for the next portfolio.

![K-Means Clusters](results/clusters_latest.png)

---

### 5. Rolling Portfolio Optimization

For every monthly rebalance:

* stocks are selected from the momentum cluster,
* the previous 12 months of daily returns form the estimation window,
* expected returns and covariance are annualized using 252 trading days,
* maximum-Sharpe weights are calculated,
* individual stock weights are capped at 25%,
* and the resulting portfolio is held during the following month.

If optimization fails, the framework falls back to equal weighting.

This creates a walk-forward portfolio construction process** in which portfolio formation precedes the period used to evaluate its return.

---

## Results

The current backtest produced the following performance:

| Metric                    |   Strategy | NASDAQ-100 |
| ------------------------- | ---------: | ---------: |
| **CAGR**                  | **27.28%** |     16.42% |
| **Annualized Volatility** |     24.44% |     17.60% |
| **Sharpe Ratio**          |   **1.12** |       0.96 |
| **Sortino Ratio**         |       1.87 |          — |
| **Maximum Drawdown**      |    -47.53% |          — |
| **Calmar Ratio**          |       0.57 |          — |
| **Positive Months**       |     63.24% |          — |
| **Beta vs NASDAQ-100**    |       0.89 |          — |

### Cumulative Performance

![Strategy vs NASDAQ-100](results/01_cumulative_returns.png)

The strategy historically generated higher annualized returns and a higher Sharpe ratio than the NASDAQ-100 benchmark.

However, the additional return came with meaningful tail risk, with the strategy experiencing a maximum drawdown of approximately **47.5%**.

---

## Risk Analysis

### Drawdown

![Portfolio Drawdown](results/02_drawdown.png)

The drawdown profile demonstrates that strong long-term momentum performance does not eliminate significant losses during adverse market regimes.

### Rolling Sharpe Ratio

![Rolling Sharpe](results/03_rolling_sharpe.png)

The rolling Sharpe ratio highlights how the strategy's risk-adjusted performance changes across different market environments.

### Monthly Returns

![Monthly Returns](results/04_monthly_returns.png)

Monthly returns provide a more granular view of the strategy's performance distribution and periods of unusually strong gains or losses.

---

## Key Findings

The research suggests several useful conclusions:

* Combining **market features and factor exposures** creates a richer cross-sectional representation than momentum alone.
* K-Means allows the investable universe to adapt as stock characteristics change.
* Dynamically identifying the momentum cluster avoids assigning permanent meaning to arbitrary K-Means labels.
* Rolling optimization provides adaptive portfolio weights but can introduce concentration and estimation risk.
* The strategy historically outperformed the NASDAQ-100 on both **CAGR and Sharpe ratio**, but experienced substantially larger drawdowns.

The results therefore highlight both the potential and the limitations of combining unsupervised learning with momentum-based portfolio construction.

---

## Limitations

The current framework has several important limitations:

* A fixed NASDAQ-100 universe is used rather than historical point-in-time constituents.
* Transaction costs, slippage and market impact are not fully incorporated.
* Maximum-Sharpe optimization relies on noisy historical estimates of expected returns and covariance.
* K-Means uses a fixed four-cluster specification.
* Historical backtest performance does not imply future performance.

These limitations are important when interpreting the headline results.

---

## Future Research

Potential extensions include:

* Transaction-cost and turnover modeling
* Equal-weight vs optimized portfolio comparison
* Point-in-time NASDAQ-100 constituents
* Covariance shrinkage
* Volatility targeting
* Alternative clustering methods
* Alternative momentum horizons
* Sector and exposure constraints
* Out-of-sample parameter robustness testing

---

## Tech Stack

 Python · Pandas · NumPy · Scikit-learn · Statsmodels · SciPy · yfinance · Matplotlib

---

## Conclusion

This project demonstrates an end-to-end quantitative research workflow combining:

**feature engineering → factor modeling → unsupervised learning → portfolio optimization → walk-forward evaluation.**

Rather than using machine learning to directly predict stock prices, the framework uses unsupervised learning to identify **cross-sectional structure within the NASDAQ-100** and integrates those signals into a systematic portfolio construction process.

The current results show promising historical risk-adjusted performance while also highlighting an important challenge: **higher returns do not necessarily imply better downside protection.**

This provides a foundation for further research into portfolio robustness, transaction costs, risk controls and regime-dependent allocation.

---

## Disclaimer

This repository is an independent **quantitative research and educational project. Backtested results are hypothetical and subject to data limitations, modeling assumptions, survivorship bias, transaction costs and estimation error.

It does not constitute investment advice.
