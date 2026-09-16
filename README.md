# Quantitative Portfolio Analysis & Optimization

A Python-based quantitative finance project that analyzes historical A-share market data and applies portfolio risk measurement, constrained portfolio optimization, efficient frontier analysis, out-of-sample backtesting, and robustness testing.

## Project Overview

This project was developed to explore how quantitative methods can be applied to portfolio construction and risk management.

Using historical price data for three A-share stocks, the project:

- Processes and cleans historical market data
- Calculates daily and annualized returns
- Measures portfolio volatility and covariance
- Analyzes correlations between assets
- Evaluates portfolio performance using Sharpe ratio and CAGR
- Measures downside risk using Value at Risk (VaR) and maximum drawdown
- Implements minimum-volatility and maximum-Sharpe portfolio optimization
- Applies portfolio weight constraints to control concentration risk
- Simulates portfolios to construct an efficient frontier
- Performs out-of-sample backtesting
- Tests the sensitivity of portfolio optimization to weight constraints

The project focuses on the relationship between expected return, portfolio risk, diversification, and model robustness.

---

## Project Structure

```text
quantitative-portfolio-analysis/
│
├── main.py
├── prepare_a_share_data.py
│
├── data/
│   └── stock_data_a_share.csv
│
├── results/
│   ├── constrained_portfolio_results.csv
│   ├── constrained_efficient_frontier_data.csv
│   ├── constrained_out_of_sample_results.csv
│   ├── constrained_training_weights.csv
│   └── robustness_analysis.csv
│
├── figures/
│   ├── constrained_efficient_frontier.png
│   ├── constrained_out_of_sample.png
│   └── robustness_sharpe.png
│
├── README.md
└── .gitignore

Data

The project uses historical daily price data for three A-share stocks:

Ticker	Stock
000938	紫光股份 (ZGC)
603986	兆易创新 (GDS)
002185	华天科技 (HTKJ)

The analysis period covers:

2023-01-03 to 2026-08-31

The raw historical data were cleaned and merged into a single time-series dataset before analysis.

For portfolio analysis, closing prices are used to calculate daily returns.

Methodology
1. Daily Returns

Daily simple returns are calculated as:

$$ R_t = \frac{P_t}{P_{t-1}} - 1 $$

where:

$P_t$ = closing price at time $t$
$P_{t-1}$ = previous trading day's closing price
2. Annualized Return

The annualized arithmetic mean return is calculated as:

$$ E(R_{annual}) = \bar{R}_{daily} \times 252 $$

where 252 represents the approximate number of trading days in a year.

3. Annualized Volatility

Daily volatility is annualized using:

$$ \sigma_{annual} = \sigma_{daily} \times \sqrt{252} $$
4. Covariance and Correlation

The project calculates the covariance matrix and correlation matrix of daily returns to capture the relationships between assets.

Portfolio variance is calculated as:

$$ \sigma_p^2 = w^T \Sigma w $$

where:

$w$ = portfolio weight vector
$\Sigma$ = covariance matrix
Portfolio Construction

Four portfolio strategies are evaluated.

Current Portfolio

A fixed portfolio with:

ZGC: 40%
GDS: 30%
HTKJ: 30%
Equal Weight Portfolio

Each stock receives:

ZGC: 33.33%
GDS: 33.33%
HTKJ: 33.33%
Minimum Volatility Portfolio

The portfolio weights are optimized to minimize annualized portfolio volatility.

Maximum Sharpe Portfolio

The portfolio weights are optimized to maximize:

$$ Sharpe = \frac{E(R_p)-R_f}{\sigma_p} $$

where:

$E(R_p)$ = expected annual portfolio return
$R_f$ = risk-free rate
$\sigma_p$ = annualized portfolio volatility

For the constrained optimization, each individual asset is subject to a maximum portfolio weight of 60%.

Efficient Frontier

A Monte Carlo simulation of feasible portfolios is used to visualize the relationship between expected annual return and annualized volatility.

The efficient frontier analysis uses a consistent return definition:

Y-axis: Expected Annual Return
X-axis: Annualized Volatility
Color scale: Sharpe Ratio

The portfolio weight constraint is incorporated into the simulation to reduce excessive concentration.

Out-of-Sample Backtesting

To evaluate whether optimization results generalize beyond the historical sample, the dataset is divided into training and testing periods.

Training Period

2023-01-03 to 2025-12-31

Testing Period

2026-01-05 to 2026-08-31

Portfolio optimization is performed using only the training data.

The resulting portfolio weights are then applied to the testing period without further optimization.

This provides an out-of-sample evaluation of:

Total Return
Annualized Return
Annualized Volatility
Sharpe Ratio
Maximum Drawdown
Robustness Analysis

The project evaluates the sensitivity of the Maximum Sharpe portfolio to different maximum asset-weight constraints.

The tested constraints are:

40%
50%
60%
70%
100%

For each constraint, portfolio weights are optimized using the training period and subsequently evaluated on the testing period.

The analysis examines how changes in concentration constraints affect:

Portfolio allocation
Out-of-sample return
Out-of-sample volatility
Sharpe ratio
Maximum drawdown
Key Findings
1. Historical optimization favored GDS

The full-sample annualized mean returns were approximately:

Asset	Annualized Mean Return	Annualized Volatility
ZGC	31.60%	51.05%
GDS	55.38%	56.79%
HTKJ	30.43%	45.37%

The higher historical return estimate for GDS caused the Maximum Sharpe optimization to assign a relatively large weight to GDS.

2. Weight constraints reduced concentration

Under the 60% maximum-weight constraint, the full-sample Maximum Sharpe portfolio allocated:

ZGC: 27.40%
GDS: 60.00%
HTKJ: 12.60%

Without a meaningful concentration constraint, the optimization became highly concentrated in GDS.

3. In-sample optimization did not necessarily produce the strongest out-of-sample risk-adjusted performance

During the 2026 testing period:

Strategy	Total Return	Volatility	Sharpe Ratio	Maximum Drawdown
Current 40/30/30	77.11%	55.46%	1.902	-31.74%
Equal Weight	77.84%	56.92%	1.879	-33.63%
Minimum Volatility	69.44%	59.28%	1.699	-36.01%
Maximum Sharpe	83.74%	66.65%	1.771	-45.25%

The Maximum Sharpe portfolio achieved the highest total return in the testing period, but also experienced higher volatility and a deeper maximum drawdown than the Current 40/30/30 portfolio.

4. Concentration increased as the maximum weight constraint was relaxed

The robustness analysis showed that relaxing the maximum weight constraint led to progressively higher GDS allocation.

At the same time:

Out-of-sample return increased
Out-of-sample volatility increased
Maximum drawdown became deeper
Out-of-sample Sharpe ratio decreased

This illustrates the trade-off between return potential and concentration risk.

Risk Analysis

The project incorporates several commonly used portfolio risk measures.

Sharpe Ratio

Measures risk-adjusted return relative to portfolio volatility.

Value at Risk

Historical 95% one-day VaR is used to estimate the loss threshold corresponding to the lower tail of historical daily portfolio returns.

VaR should not be interpreted as a maximum possible loss.

Maximum Drawdown

Measures the largest peak-to-trough decline in portfolio value during the analysis period.

Rolling Volatility

A 60-trading-day rolling volatility measure is used to examine changes in portfolio risk over time.

Technology Stack
Python
Pandas
NumPy
SciPy
Matplotlib

Key Python libraries:

pandas
numpy
scipy
matplotlib
Limitations

This project is intended for educational and portfolio-analysis purposes rather than investment advice.

Several limitations should be considered:

Historical returns may not represent future expected returns.
Portfolio optimization can be sensitive to expected-return estimates.
The analysis does not incorporate transaction costs, taxes, slippage, or liquidity constraints.
The current backtest assumes daily portfolio rebalancing to the target weights.
The 60% maximum-weight constraint is a modeling assumption rather than an industry-standard threshold.
The dataset contains only three stocks and therefore does not represent a diversified market portfolio.
The out-of-sample testing period is relatively short compared with the training period.
The analysis does not account for corporate actions beyond those reflected in the provided price data.
Future Improvements

Potential extensions include:

Rolling-window portfolio optimization
Transaction-cost modeling
Turnover analysis
Alternative risk measures such as Sortino Ratio and CVaR
Larger and more diversified asset universes
Factor-based portfolio analysis
Different estimation methods for expected returns and covariance
Longer out-of-sample testing periods
Conclusion

This project demonstrates how quantitative methods can be used to connect historical market data, portfolio construction, risk measurement, optimization, and model validation.

A key observation from the analysis is that portfolio optimization results can be sensitive to estimation assumptions and concentration constraints. Out-of-sample backtesting and robustness analysis therefore provide important complements to in-sample optimization.