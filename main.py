import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_FILE = "data/stock_data_a_share.csv"

STOCKS = ["ZGC", "GDS", "HTKJ"]

CURRENT_WEIGHTS = np.array([0.40, 0.30, 0.30])

MIN_WEIGHT = 0.00
MAX_WEIGHT = 0.60

RISK_FREE_RATE = 0.00
TRADING_DAYS = 252

TRAIN_END = "2025-12-31"
TEST_START = "2026-01-01"


# ============================================================
# 1.5. CREATE OUTPUT DIRECTORIES
# ============================================================

from pathlib import Path

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

data = pd.read_csv(DATA_FILE)

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values("Date")

data = data.set_index("Date")

prices = data[STOCKS].copy()

returns = prices.pct_change().dropna()


# ============================================================
# 3. BASIC STATISTICS
# ============================================================

annual_mean_returns = returns.mean() * TRADING_DAYS

annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)

cov_matrix = returns.cov() * TRADING_DAYS

correlation_matrix = returns.corr()


print("=" * 70)
print("CONSTRAINED PORTFOLIO OPTIMIZATION")
print("=" * 70)

print("\nData Period:")
print(f"{prices.index.min().date()} to {prices.index.max().date()}")

print("\nNumber of Trading Days:")
print(len(prices))

print("\nNumber of Return Observations:")
print(len(returns))

print("\nAnnualized Mean Return:")
print(annual_mean_returns)

print("\nAnnualized Volatility:")
print(annual_volatility)

print("\nCovariance Matrix:")
print(cov_matrix)


# ============================================================
# 4. PORTFOLIO FUNCTIONS
# ============================================================

def portfolio_return(weights, mean_returns):
    return np.dot(weights, mean_returns)


def portfolio_volatility(weights, covariance):
    return np.sqrt(
        np.dot(weights.T, np.dot(covariance, weights))
    )


def portfolio_sharpe(weights, mean_returns, covariance):
    ret = portfolio_return(weights, mean_returns)

    vol = portfolio_volatility(weights, covariance)

    return (ret - RISK_FREE_RATE) / vol


def optimize_min_vol(mean_returns, covariance):

    n = len(mean_returns)

    x0 = np.ones(n) / n

    bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    result = minimize(
        lambda weights: portfolio_volatility(
            weights,
            covariance
        ),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result.x


def optimize_max_sharpe(mean_returns, covariance):

    n = len(mean_returns)

    x0 = np.ones(n) / n

    bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    result = minimize(
        lambda weights: -portfolio_sharpe(
            weights,
            mean_returns,
            covariance
        ),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result.x


# ============================================================
# 5. HISTORICAL PORTFOLIO PERFORMANCE
# ============================================================

def historical_statistics(weights, portfolio_returns):

    cumulative = (1 + portfolio_returns).cumprod()

    total_return = cumulative.iloc[-1] - 1

    years = (
        portfolio_returns.index[-1]
        - portfolio_returns.index[0]
    ).days / 365.25

    cagr = (1 + total_return) ** (1 / years) - 1

    annualized_vol = (
        portfolio_returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    sharpe = (
        portfolio_returns.mean()
        * TRADING_DAYS
        - RISK_FREE_RATE
    ) / annualized_vol

    running_max = cumulative.cummax()

    drawdown = cumulative / running_max - 1

    max_drawdown = drawdown.min()

    return {
        "CAGR": cagr,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_drawdown
    }


# ============================================================
# 6. OPTIMIZATION
# ============================================================

min_vol_weights = optimize_min_vol(
    annual_mean_returns,
    cov_matrix
)

max_sharpe_weights = optimize_max_sharpe(
    annual_mean_returns,
    cov_matrix
)

equal_weights = np.ones(len(STOCKS)) / len(STOCKS)


print("\n" + "=" * 70)
print("CONSTRAINED OPTIMIZED WEIGHTS")
print("=" * 70)


def print_weights(name, weights):

    print(f"\n{name}")

    for stock, weight in zip(STOCKS, weights):

        print(
            f"{stock}: {weight:.2%}"
        )


print_weights(
    "Current Portfolio:",
    CURRENT_WEIGHTS
)

print_weights(
    "Equal Weight Portfolio:",
    equal_weights
)

print_weights(
    "Minimum Volatility Portfolio (Max Weight = 60%):",
    min_vol_weights
)

print_weights(
    "Maximum Sharpe Portfolio (Max Weight = 60%):",
    max_sharpe_weights
)


# ============================================================
# 7. FULL SAMPLE COMPARISON
# ============================================================

portfolio_weights = {

    "Current 40/30/30":
        CURRENT_WEIGHTS,

    "Equal Weight":
        equal_weights,

    "Minimum Volatility":
        min_vol_weights,

    "Maximum Sharpe":
        max_sharpe_weights
}


full_sample_results = {}

for name, weights in portfolio_weights.items():

    portfolio_returns = returns.dot(weights)

    full_sample_results[name] = \
        historical_statistics(
            weights,
            portfolio_returns
        )


full_sample_table = pd.DataFrame(
    full_sample_results
).T


print("\n" + "=" * 70)
print("FULL-SAMPLE PORTFOLIO COMPARISON")
print("=" * 70)

print(
    full_sample_table.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


full_sample_table.to_csv(
    "results/constrained_portfolio_results.csv"
)


# ============================================================
# 8. CORRECTED EFFICIENT FRONTIER
# ============================================================
#
# IMPORTANT:
#
# Efficient Frontier uses:
#
# Expected Annual Return
# + Annualized Volatility
#
# Both are based on the same annualized arithmetic-return
# definition.
#
# CAGR is NOT used here.
#
# ============================================================

np.random.seed(42)

NUM_PORTFOLIOS = 20000

frontier_results = []

for _ in range(NUM_PORTFOLIOS):

    weights = np.random.uniform(
        MIN_WEIGHT,
        MAX_WEIGHT,
        len(STOCKS)
    )

    weights = weights / weights.sum()

    if np.any(weights > MAX_WEIGHT):

        continue

    expected_return = portfolio_return(
        weights,
        annual_mean_returns
    )

    volatility = portfolio_volatility(
        weights,
        cov_matrix
    )

    sharpe = (
        expected_return
        - RISK_FREE_RATE
    ) / volatility

    frontier_results.append(
        [
            expected_return,
            volatility,
            sharpe,
            *weights
        ]
    )


frontier_columns = [
    "Expected Annual Return",
    "Annualized Volatility",
    "Sharpe Ratio"
] + STOCKS


frontier_df = pd.DataFrame(
    frontier_results,
    columns=frontier_columns
)


print("\n" + "=" * 70)
print("CORRECTED EFFICIENT FRONTIER")
print("=" * 70)

print("\nNumber of Valid Simulated Portfolios:")

print(len(frontier_df))


# Save frontier data

frontier_df.to_csv(
    "results/constrained_efficient_frontier_data.csv",
    index=False
)


# ============================================================
# 9. CALCULATE FRONTIER POINTS USING SAME RETURN DEFINITION
# ============================================================

def optimization_point(weights):

    expected_return = portfolio_return(
        weights,
        annual_mean_returns
    )

    volatility = portfolio_volatility(
        weights,
        cov_matrix
    )

    sharpe = (
        expected_return
        - RISK_FREE_RATE
    ) / volatility

    return (
        expected_return,
        volatility,
        sharpe
    )


current_point = optimization_point(
    CURRENT_WEIGHTS
)

equal_point = optimization_point(
    equal_weights
)

min_vol_point = optimization_point(
    min_vol_weights
)

max_sharpe_point = optimization_point(
    max_sharpe_weights
)


# ============================================================
# 10. EFFICIENT FRONTIER PLOT
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    frontier_df["Annualized Volatility"],
    frontier_df["Expected Annual Return"],
    c=frontier_df["Sharpe Ratio"],
    cmap="viridis",
    s=8,
    alpha=0.5
)

plt.scatter(
    current_point[1],
    current_point[0],
    marker="*",
    s=250,
    label="Current 40/30/30"
)

plt.scatter(
    equal_point[1],
    equal_point[0],
    marker="o",
    s=120,
    label="Equal Weight"
)

plt.scatter(
    min_vol_point[1],
    min_vol_point[0],
    marker="D",
    s=120,
    label="Minimum Volatility"
)

plt.scatter(
    max_sharpe_point[1],
    max_sharpe_point[0],
    marker="X",
    s=150,
    label="Maximum Sharpe"
)

plt.xlabel(
    "Annualized Volatility"
)

plt.ylabel(
    "Expected Annual Return"
)

plt.title(
    "Constrained Efficient Frontier"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "figures/constrained_efficient_frontier.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. OUT-OF-SAMPLE BACKTEST
# ============================================================

print("\n" + "=" * 70)
print("OUT-OF-SAMPLE BACKTESTING")
print("=" * 70)


train_returns = returns[
    returns.index <= TRAIN_END
]

test_returns = returns[
    returns.index >= TEST_START
]


print("\nTraining Period:")

print(
    f"{train_returns.index.min().date()} "
    f"to "
    f"{train_returns.index.max().date()}"
)


print("\nTesting Period:")

print(
    f"{test_returns.index.min().date()} "
    f"to "
    f"{test_returns.index.max().date()}"
)


# Training statistics

train_mean_returns = (
    train_returns.mean()
    * TRADING_DAYS
)

train_covariance = (
    train_returns.cov()
    * TRADING_DAYS
)


train_min_vol_weights = optimize_min_vol(
    train_mean_returns,
    train_covariance
)

train_max_sharpe_weights = optimize_max_sharpe(
    train_mean_returns,
    train_covariance
)


training_weights = {

    "Current 40/30/30":
        CURRENT_WEIGHTS,

    "Equal Weight":
        equal_weights,

    "Minimum Volatility":
        train_min_vol_weights,

    "Maximum Sharpe":
        train_max_sharpe_weights
}


print("\nTraining-Based Weights:")


for name, weights in training_weights.items():

    print(f"\n{name}")

    for stock, weight in zip(
        STOCKS,
        weights
    ):

        print(
            f"{stock}: {weight:.2%}"
        )


# Save training weights

training_weights_df = pd.DataFrame(
    training_weights,
    index=STOCKS
).T

training_weights_df.to_csv(
    "results/constrained_training_weights.csv"
)


# ============================================================
# 12. OOS PERFORMANCE
# ============================================================

oos_results = {}

oos_portfolios = {}

for name, weights in training_weights.items():

    portfolio_returns = (
        test_returns.dot(weights)
    )

    oos_portfolios[name] = portfolio_returns

    cumulative = (
        1 + portfolio_returns
    ).cumprod()

    total_return = (
        cumulative.iloc[-1] - 1
    )

    years = (
        portfolio_returns.index[-1]
        - portfolio_returns.index[0]
    ).days / 365.25

    annualized_return = (
        (1 + total_return)
        ** (1 / years)
        - 1
    )

    annualized_volatility = (
        portfolio_returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    sharpe = (
        portfolio_returns.mean()
        * TRADING_DAYS
        - RISK_FREE_RATE
    ) / annualized_volatility

    running_max = (
        cumulative.cummax()
    )

    drawdown = (
        cumulative / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    oos_results[name] = {

        "Total Return":
            total_return,

        "Annualized Return":
            annualized_return,

        "Annualized Volatility":
            annualized_volatility,

        "Sharpe Ratio":
            sharpe,

        "Maximum Drawdown":
            max_drawdown
    }


oos_table = pd.DataFrame(
    oos_results
).T


print("\n" + "=" * 70)
print("OUT-OF-SAMPLE RESULTS")
print("=" * 70)

print(
    oos_table.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


oos_table.to_csv(
    "results/constrained_out_of_sample_results.csv"
)


# ============================================================
# 13. OOS CUMULATIVE PERFORMANCE
# ============================================================

plt.figure(figsize=(10, 7))

for name, portfolio_returns in oos_portfolios.items():

    cumulative = (
        1 + portfolio_returns
    ).cumprod()

    plt.plot(
        cumulative,
        label=name
    )

plt.title(
    "Out-of-Sample Portfolio Performance"
)

plt.xlabel("Date")

plt.ylabel(
    "Growth of $1"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "figures/constrained_out_of_sample.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. ROBUSTNESS ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ROBUSTNESS ANALYSIS")
print("=" * 70)


MAX_WEIGHT_LEVELS = [
    0.40,
    0.50,
    0.60,
    0.70,
    1.00
]


robustness_results = []


for max_weight in MAX_WEIGHT_LEVELS:

    def optimize_max_sharpe_custom(
        mean_returns,
        covariance
    ):

        n = len(mean_returns)

        x0 = np.ones(n) / n

        bounds = [
            (0.00, max_weight)
        ] * n

        constraints = {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1
        }

        result = minimize(

            lambda weights:
                -portfolio_sharpe(
                    weights,
                    mean_returns,
                    covariance
                ),

            x0,

            method="SLSQP",

            bounds=bounds,

            constraints=constraints
        )

        return result.x


    robustness_weights = (
        optimize_max_sharpe_custom(
            train_mean_returns,
            train_covariance
        )
    )


    robustness_oos_returns = (
        test_returns.dot(
            robustness_weights
        )
    )


    cumulative = (
        1 + robustness_oos_returns
    ).cumprod()


    total_return = (
        cumulative.iloc[-1] - 1
    )


    years = (
        robustness_oos_returns.index[-1]
        - robustness_oos_returns.index[0]
    ).days / 365.25


    annualized_return = (
        (1 + total_return)
        ** (1 / years)
        - 1
    )


    volatility = (
        robustness_oos_returns.std()
        * np.sqrt(TRADING_DAYS)
    )


    sharpe = (
        robustness_oos_returns.mean()
        * TRADING_DAYS
        - RISK_FREE_RATE
    ) / volatility


    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max
        - 1
    )

    max_drawdown = drawdown.min()


    robustness_results.append({

        "Max Weight Constraint":
            max_weight,

        "ZGC Weight":
            robustness_weights[0],

        "GDS Weight":
            robustness_weights[1],

        "HTKJ Weight":
            robustness_weights[2],

        "OOS Total Return":
            total_return,

        "OOS Annualized Return":
            annualized_return,

        "OOS Volatility":
            volatility,

        "OOS Sharpe":
            sharpe,

        "OOS Maximum Drawdown":
            max_drawdown
    })


robustness_df = pd.DataFrame(
    robustness_results
)


print(
    robustness_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


robustness_df.to_csv(
    "results/robustness_analysis.csv",
    index=False
)


# ============================================================
# 15. ROBUSTNESS PLOT
# ============================================================

plt.figure(figsize=(10, 7))

plt.plot(
    robustness_df["Max Weight Constraint"],
    robustness_df["OOS Sharpe"],
    marker="o"
)

plt.xlabel(
    "Maximum Weight Constraint"
)

plt.ylabel(
    "Out-of-Sample Sharpe Ratio"
)

plt.title(
    "Robustness Analysis: Weight Constraint vs OOS Sharpe"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "figures/robustness_sharpe.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ROBUSTNESS ANALYSIS COMPLETED")
print("=" * 70)


print("\nGenerated Files:")

print(
    "results/constrained_portfolio_results.csv"
)

print(
    "figures/constrained_efficient_frontier.png"
)

print(
    "results/constrained_efficient_frontier_data.csv"
)

print(
    "results/constrained_out_of_sample_results.csv"
)

print(
    "figures/constrained_out_of_sample.png"
)

print(
    "results/constrained_training_weights.csv"
)

print(
    "results/robustness_analysis.csv"
)

print(
    "figures/robustness_sharpe.png"
)

print(
    "\nAll calculations completed successfully."
)