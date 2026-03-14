---
name: skfolio
description: Expert knowledge and step-by-step runbook for the skfolio Python library for portfolio optimization. Use when user asks to "optimize a portfolio", "backtest a strategy", "build a Mean-Variance model", "implement HRP", "use skfolio", "minimize CVaR", "maximize Sharpe ratio", "compute portfolio risk measures", "run walk-forward cross-validation", "use Black-Litterman", "run Entropy Pooling", "implement NCO", or any task involving portfolio construction with skfolio.
---

# skfolio — Portfolio Optimization Runbook

`skfolio` extends `scikit-learn` for portfolio optimization. It follows the same `fit(X)` / `predict(X)` API. `X` is always asset **returns** (not prices) of shape `(n_observations, n_assets)`.

## Step 1 — Prepare data

Always convert prices to returns first. Never shuffle time-series data.

```python
from sklearn.model_selection import train_test_split
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()          # also: load_factors_dataset(), load_sp500_index()
X = prices_to_returns(prices)          # simple returns, shape (n_obs, n_assets)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)  # NEVER shuffle

# If using factors or a benchmark index:
X, y = prices_to_returns(prices, factor_prices)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)
```

## Step 2 — Choose an optimization estimator

Pick based on user's goal:

| Goal | Estimator | Notes |
| :--- | :--- | :--- |
| Maximize Sharpe / minimize risk | `MeanRisk` | Most flexible; 15+ risk measures |
| Equal risk contribution | `RiskBudgeting` | Long-only, convex |
| Maximize diversification | `MaximumDiversification` | Ratio of weighted vols / total vol |
| Distributionally robust CVaR | `DistributionallyRobustCVaR` | Wasserstein ball; needs Mosek for large N |
| Hierarchical Risk Parity (HRP) | `HierarchicalRiskParity` | No covariance inversion; stable |
| Equal Risk Contribution (HERC) | `HierarchicalEqualRiskContribution` | Uses dendrogram shape |
| Cluster then optimize | `NestedClustersOptimization` (NCO) | Inner + outer estimators |
| Ensemble stacking | `StackingOptimization` | Combines multiple models |
| Naive benchmarks | `EqualWeighted`, `InverseVolatility`, `Random` | Use as baselines |

## Step 3 — Configure the prior estimator (optional)

Every convex estimator accepts a `prior_estimator` that computes `(mu, covariance, returns)`.
Default is `EmpiricalPrior()`. Override to improve estimation quality.

```python
from skfolio.prior import EmpiricalPrior, BlackLitterman, FactorModel, EntropyPooling, SyntheticData
from skfolio.moments import LedoitWolf, ShrunkMu, DenoiseCovariance, EWCovariance

# Empirical with improved estimators (recommended for large N)
prior = EmpiricalPrior(
    mu_estimator=ShrunkMu(),              # James-Stein shrinkage
    covariance_estimator=LedoitWolf(),    # shrunk covariance
)

# Black-Litterman: incorporate analyst views
views = ["AAPL - BBY == 0.0003", "CVX - KO == 0.0004", "MSFT == 0.0006"]
prior = BlackLitterman(views=views)

# Factor model (X = asset returns, y = factor returns)
prior = FactorModel()   # fit(X_train, y_train)

# Entropy Pooling: non-parametric views on any moment
prior = EntropyPooling(
    mean_views=["JPM == -0.002", "BAC >= prior(BAC) * 1.2"],
    variance_views=["BAC == prior(BAC) * 4"],
    correlation_views=["(BAC,JPM) == 0.80"],
    cvar_views=["GE == 0.08"],
    groups={"BAC": ["Financials"], "JPM": ["Financials"]},
)

# Synthetic data via Vine Copula (for tail risk / stress tests)
from skfolio.distribution import VineCopula
prior = SyntheticData(distribution_estimator=VineCopula(log_transform=True, n_jobs=-1), n_samples=2000)

# Composing priors: Black-Litterman on factors fed into a Factor Model
prior = FactorModel(factor_prior_estimator=BlackLitterman(views=factor_views))
```

## Step 4 — Build and fit the model

```python
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio import RiskMeasure

# Maximum Sharpe Ratio
model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.VARIANCE,    # or CVAR, SEMI_DEVIATION, MAX_DRAWDOWN, ...
    prior_estimator=prior,                # optional; defaults to EmpiricalPrior()
)

# Minimum CVaR
model = MeanRisk(risk_measure=RiskMeasure.CVAR)

# CVaR Risk Parity
from skfolio.optimization import RiskBudgeting
model = RiskBudgeting(risk_measure=RiskMeasure.CVAR)

# HRP with mutual information distance
from skfolio.optimization import HierarchicalRiskParity
from skfolio.distance import MutualInformation
model = HierarchicalRiskParity(risk_measure=RiskMeasure.SEMI_DEVIATION, distance_estimator=MutualInformation())

# Fit
model.fit(X_train)
print(model.weights_)

# Predict → returns a Portfolio object
portfolio = model.predict(X_test)
print(portfolio.annualized_sharpe_ratio)
print(portfolio.cvar)
```

## Step 5 — Add constraints

All constraints go on `MeanRisk` (and `RiskBudgeting` supports a subset):

```python
model = MeanRisk(
    # Weight bounds
    min_weights=0.0,          # scalar, dict, or array; 0.0 = long-only
    max_weights=0.10,         # max 10% per asset
    budget=1.0,               # sum of weights = 1 (fully invested)
    max_short=0.3,            # max total short exposure
    max_long=1.3,             # max total long exposure

    # Group / linear constraints
    groups={"AAPL": ["Tech", "US"], "TLT": ["Bond", "US"]},
    linear_constraints=["Tech >= 0.20", "Bond <= 0.40", "US == 0.70"],

    # Cardinality (requires MIP solver: solver="SCIP" or "MOSEK")
    cardinality=10,
    group_cardinalities={"Healthcare": 2, "Tech": 3},

    # Costs and turnover
    transaction_costs=0.001,   # 10bps; must match return periodicity
    management_fees=0.0002,
    previous_weights=prev_w,
    max_turnover=0.20,

    # Tracking error
    max_tracking_error=0.003,  # 30bps vs benchmark in y

    # Regularization
    l1_coef=1e-4,
    l2_coef=1e-4,

    # Risk measure constraints (multi-objective)
    max_cvar=0.05,
    max_max_drawdown=0.15,
    min_return=0.0001,

    # Risk measure parameters
    cvar_beta=0.95,
    evar_beta=0.95,
    cdar_beta=0.95,
)
```

## Step 6 — Evaluate the portfolio

```python
portfolio = model.predict(X_test)

# Key measures
portfolio.sharpe_ratio
portfolio.annualized_sharpe_ratio
portfolio.cvar                        # CVaR at cvar_beta
portfolio.max_drawdown
portfolio.calmar_ratio
portfolio.sortino_ratio
portfolio.annualized_mean
portfolio.diversification             # only on Portfolio, not MultiPeriodPortfolio

# Summary of all 40+ measures
print(portfolio.summary())

# Contributions
portfolio.contribution(measure=RiskMeasure.CVAR)

# Plots (return Plotly figures)
portfolio.plot_cumulative_returns()
portfolio.plot_drawdowns()
portfolio.plot_rolling_measure(measure=RatioMeasure.SHARPE_RATIO, window=63)
portfolio.plot_composition()
```

## Step 7 — Cross-validate with walk-forward (recommended for research)

```python
from skfolio.model_selection import WalkForward, CombinatorialPurgedCV, MultipleRandomizedCV, cross_val_predict
from skfolio import RatioMeasure

# Simple walk-forward → returns MultiPeriodPortfolio (one path)
cv = WalkForward(train_size=252, test_size=21)           # calendar days; or use freq="WOM-3FRI"
pred = cross_val_predict(model, X, cv=cv)
print(pred.annualized_sharpe_ratio)

# Combinatorial Purged CV → returns Population of MultiPeriodPortfolio (many paths)
from skfolio.model_selection import optimal_folds_number
n_folds, n_test_folds = optimal_folds_number(
    n_observations=len(X), target_n_test_paths=50, target_train_size=252
)
cv = CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_test_folds)
pred = cross_val_predict(model, X, cv=cv, n_jobs=-1)
print(pred.summary())
print(pred.measures_mean(RatioMeasure.ANNUALIZED_SHARPE_RATIO))

# Multiple Randomized CV (Monte Carlo: random asset subsets + time windows)
cv = MultipleRandomizedCV(
    walk_forward=WalkForward(test_size=3, train_size=6, freq="WOM-3FRI"),
    n_subsamples=100, asset_subset_size=15, window_size=2*252,
)
pred = cross_val_predict(model, X, cv=cv, n_jobs=-1)
```

## Step 8 — Handle failures (production and research)

```python
from skfolio.optimization import EqualWeighted

# Fallback chain: try progressively relaxed models
model = MeanRisk(
    min_weights=0.05,
    fallback=[
        MeanRisk(min_weights=0.02),   # relaxed fallback
        EqualWeighted(),               # safe fallback
        "previous_weights",            # terminal safety net
    ],
)
model.fit(X_train)
print(model.fallback_chain_)          # audit trail

# Research mode: don't crash on infeasible windows
model = MeanRisk(raise_on_failure=False)
pred = cross_val_predict(model, X, cv=WalkForward(252, 21))
print(pred.n_failed_portfolios)
print(pred.n_fallback_portfolios)
```

## Step 9 — Hyperparameter tuning

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=model,
    cv=WalkForward(train_size=252, test_size=60),
    n_jobs=-1,
    param_grid={
        "risk_measure": [RiskMeasure.VARIANCE, RiskMeasure.CVAR],
        "prior_estimator__covariance_estimator": [EmpiricalCovariance(), LedoitWolf()],
    },
)
grid_search.fit(X_train)
best_model = grid_search.best_estimator_
```

## Step 10 — Analyze a Population of portfolios

```python
from skfolio import Population, RiskMeasure, RatioMeasure, PerfMeasure

# Efficient frontier
model = MeanRisk(risk_measure=RiskMeasure.CVAR, efficient_frontier_size=30)
model.fit(X_train)
population = model.predict(X_test)    # Population object

population.plot_measures(
    x=RiskMeasure.CVAR,
    y=PerfMeasure.ANNUALIZED_MEAN,
    color_scale=RatioMeasure.CVAR_RATIO,
)
population.summary()
population.plot_cumulative_returns()

# Combine populations from different models
pop_all = pop_model1 + pop_model2 + pop_benchmark
```

---

## Key Classes Reference

### Optimization
- `MeanRisk` — unified mean-risk: MINIMIZE_RISK / MAXIMIZE_RETURN / MAXIMIZE_UTILITY / MAXIMIZE_RATIO
- `RiskBudgeting` — risk parity; `risk_budget` sets target contributions
- `MaximumDiversification` — maximizes weighted-vol / total-vol ratio
- `DistributionallyRobustCVaR(wasserstein_ball_radius=0.01)` — Wasserstein robust CVaR
- `BenchmarkTracker` — minimizes tracking error vs a benchmark in `y`
- `HierarchicalRiskParity`, `HierarchicalEqualRiskContribution`, `NestedClustersOptimization`
- `StackingOptimization(estimators=[...], final_estimator=..., cv=KFold())`
- `EqualWeighted`, `InverseVolatility`, `Random`

### Risk Measures (used in optimization and portfolio)
`RiskMeasure`: VARIANCE, SEMI_VARIANCE, STANDARD_DEVIATION, SEMI_DEVIATION, MEAN_ABSOLUTE_DEVIATION, FIRST_LOWER_PARTIAL_MOMENT, CVAR, EVAR, WORST_REALIZATION, CDAR, MAX_DRAWDOWN, AVERAGE_DRAWDOWN, EDAR, ULCER_INDEX, GINI_MEAN_DIFFERENCE

`ExtraRiskMeasure` (portfolio only, not in convex optimization): VALUE_AT_RISK, DRAWDOWN_AT_RISK, SKEW, KURTOSIS, FOURTH_CENTRAL_MOMENT

`RatioMeasure`: SHARPE_RATIO, ANNUALIZED_SHARPE_RATIO, SORTINO_RATIO, CALMAR_RATIO, CVAR_RATIO, CDAR_RATIO, ...

### Moments
- `EmpiricalCovariance`, `LedoitWolf`, `OAS`, `ShrunkCovariance`
- `EWCovariance(span=60)` — exponentially weighted
- `GerberCovariance` — robust to outliers
- `DenoiseCovariance` — random matrix theory denoising
- `DetoneCovariance` — removes market mode
- `GraphicalLassoCV` — sparse precision matrix
- `ImpliedCovariance` — uses implied volatility surface
- `EmpiricalMu`, `EWMu(span=60)`, `ShrunkMu`, `EquilibriumMu(risk_aversion=2.0)`

### Distance (for HRP/HERC/NCO)
`PearsonDistance`, `SpearmanDistance`, `KendallDistance`, `CovarianceDistance`, `DistanceCorrelation`, `MutualInformation`

### Pre-selection (use in sklearn Pipeline)
`DropCorrelated(threshold=0.95)`, `DropZeroVariance`, `SelectComplete`, `SelectKExtremes`, `SelectNonDominated`, `SelectNonExpiring`

### Uncertainty Sets (robust optimization)
`EmpiricalMuUncertaintySet`, `BootstrapMuUncertaintySet(confidence_level=0.9)`
`EmpiricalCovarianceUncertaintySet`, `BootstrapCovarianceUncertaintySet`

---

## Common Gotchas

- **Never shuffle** time-series data. Always `shuffle=False` in sklearn splitters.
- **Transaction costs periodicity**: daily returns → daily costs. Annual costs on daily data → divide by 252.
- **`MAXIMIZE_RATIO` with `VARIANCE`** silently maximizes Sharpe (variance is not 1-homogeneous; skfolio handles conversion internally and warns).
- **High-dimensional data** (`N > T`): always use `LedoitWolf` or `DenoiseCovariance` instead of `EmpiricalCovariance`.
- **Cardinality constraints** require a MIP solver: install with `pip install cvxpy[SCIP]` and pass `solver="SCIP"`.
- **Factor model** uses `fit(X, y)` where `y` is factor returns (2D). `y` is lowercase even for multiple factors, following sklearn convention.
- **`log_wealth`** is the Kelly Criterion / CAGR surrogate. Available as `PerfMeasure.LOG_WEALTH` on `Portfolio`.
- **`efficient_frontier_size`** only works with `MINIMIZE_RISK` objective.
- **`previous_weights`** must be set for `transaction_costs` and `max_turnover` to have effect.
