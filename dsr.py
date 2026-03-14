"""
Deflated Sharpe Ratio helpers for portfolio research.

The core functions in this module work on raw return vectors so they can be used
inside sklearn-compatible pipelines, custom skfolio validation harnesses, or an
outer hyperparameter search controller that tracks Sharpe ratios across trials.

Integration
-----------
For a single evaluation run (e.g. ``train.py``), leave ``search_sr_trials`` empty
so each case is scored with PSR versus zero. For hyperparameter search, the
controller should pass the Sharpe ratio of every tested configuration into
``make_dsr_scorer(sr_trials)`` or set ``ValidationConfig.search_sr_trials``;
N is the number of strategy configurations, not the number of CV folds.

Example
-------
```python
from sklearn.model_selection import GridSearchCV

from dsr import make_dsr_scorer

# When re-scoring or ranking after a search, use all trial Sharpes for deflation
trial_sharpes = [...]  # e.g. from all GridSearchCV / RandomizedSearchCV candidates
search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring=make_dsr_scorer(trial_sharpes),
    n_jobs=-1,
)
```
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike

EULER_MASCHERONI = 0.5772156649015329
_STANDARD_NORMAL = NormalDist()
_EPS = np.finfo(float).eps
_TRIAL_ATTRS = (
    "sr_trials_",
    "dsr_sr_trials_",
    "search_sr_trials_",
    "sr_trials",
    "dsr_sr_trials",
    "search_sr_trials",
)


def _to_finite_1d(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _sample_sharpe_ratio(returns: np.ndarray) -> float:
    if returns.size < 2:
        return float("nan")

    volatility = returns.std(ddof=1)
    if not np.isfinite(volatility) or volatility <= _EPS:
        mean = float(returns.mean())
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return 0.0
    return float(returns.mean() / volatility)


def _sample_skewness_and_kurtosis(returns: np.ndarray) -> tuple[float, float]:
    centered = returns - returns.mean()
    second_moment = float(np.mean(centered**2))
    if second_moment <= _EPS:
        return 0.0, 3.0

    third_moment = float(np.mean(centered**3))
    fourth_moment = float(np.mean(centered**4))
    skewness = third_moment / second_moment**1.5
    kurtosis = fourth_moment / second_moment**2
    return float(skewness), float(kurtosis)


def probabilistic_sharpe_ratio(returns: ArrayLike, sr_star: float = 0.0) -> float:
    """
    Compute the Probabilistic Sharpe Ratio (PSR) from a 1D return vector.

    NaNs and infinities are ignored. Returns `nan` when fewer than two finite
    observations remain.
    """

    clean_returns = _to_finite_1d(returns)
    sample_size = clean_returns.size
    if sample_size < 2:
        return float("nan")

    sharpe_ratio = _sample_sharpe_ratio(clean_returns)
    if np.isposinf(sharpe_ratio):
        return 1.0
    if np.isneginf(sharpe_ratio):
        return 0.0

    skewness, kurtosis = _sample_skewness_and_kurtosis(clean_returns)
    denominator_sq = 1.0 - skewness * sharpe_ratio
    denominator_sq += ((kurtosis - 1.0) / 4.0) * sharpe_ratio**2
    denominator_sq = max(float(denominator_sq), _EPS)

    numerator = (sharpe_ratio - float(sr_star)) * np.sqrt(sample_size - 1.0)
    z_score = numerator / np.sqrt(denominator_sq)
    return float(_STANDARD_NORMAL.cdf(z_score))


def expected_max_sharpe(sr_estimates: ArrayLike) -> float:
    """
    Estimate the Sharpe ratio expected from luck across multiple trials.

    The input must contain Sharpe ratios across tested strategy configurations,
    not across validation folds.
    """

    sharpe_estimates = _to_finite_1d(sr_estimates)
    n_trials = sharpe_estimates.size
    if n_trials == 0:
        return 0.0

    mean_sr = float(sharpe_estimates.mean())
    if n_trials == 1:
        return mean_sr

    std_sr = float(sharpe_estimates.std(ddof=0))
    if std_sr <= _EPS:
        return mean_sr

    first_tail = np.clip(1.0 - 1.0 / n_trials, _EPS, 1.0 - _EPS)
    second_tail = np.clip(
        1.0 - 1.0 / (n_trials * np.e),
        _EPS,
        1.0 - _EPS,
    )
    blended_quantile = (1.0 - EULER_MASCHERONI) * _STANDARD_NORMAL.inv_cdf(first_tail)
    blended_quantile += EULER_MASCHERONI * _STANDARD_NORMAL.inv_cdf(second_tail)
    return float(mean_sr + std_sr * blended_quantile)


def deflated_sharpe_ratio(returns: ArrayLike, sr_trials: ArrayLike) -> float:
    """
    Compute the Deflated Sharpe Ratio score for one validation return vector.
    """

    sr_star = expected_max_sharpe(sr_trials)
    return probabilistic_sharpe_ratio(returns, sr_star=sr_star)


def median_path_deflated_sharpe_ratio(
    path_returns: list[ArrayLike] | tuple[ArrayLike, ...],
    sr_trials: ArrayLike,
) -> float:
    """
    Aggregate multiple validation paths with the median path-level DSR.
    """

    scores = np.asarray(
        [deflated_sharpe_ratio(returns, sr_trials) for returns in path_returns],
        dtype=float,
    )
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return float("nan")
    return float(np.median(finite_scores))


def extract_path_returns(prediction: object) -> tuple[np.ndarray, ...]:
    """
    Extract one 1D return vector per validation path from a skfolio prediction.
    """

    if hasattr(prediction, "returns"):
        return (np.asarray(prediction.returns, dtype=float).reshape(-1),)

    try:
        return tuple(
            np.asarray(portfolio.returns, dtype=float).reshape(-1)
            for portfolio in prediction
        )
    except TypeError as exc:
        raise TypeError(
            "Prediction object does not expose portfolio returns per validation path."
        ) from exc


def _resolve_sr_trials(estimator: object) -> np.ndarray:
    candidates = [estimator]
    if hasattr(estimator, "steps") and estimator.steps:
        candidates.append(estimator.steps[-1][1])

    for candidate in candidates:
        for attr_name in _TRIAL_ATTRS:
            if hasattr(candidate, attr_name):
                return _to_finite_1d(getattr(candidate, attr_name))
        if hasattr(candidate, "get_search_sr_trials"):
            return _to_finite_1d(candidate.get_search_sr_trials())
    return np.asarray([], dtype=float)


def dsr_scorer(estimator, X, y=None) -> float:
    """
    sklearn-compatible scorer for skfolio estimators and pipelines.

    The scorer looks for trial-level Sharpe estimates on the estimator via one of:
    `sr_trials_`, `dsr_sr_trials_`, or `search_sr_trials_`. When no trial history
    is available it falls back to PSR against a zero benchmark.
    """

    del y
    prediction = estimator.predict(X)
    path_returns = extract_path_returns(prediction)
    sr_trials = _resolve_sr_trials(estimator)

    if len(path_returns) == 1:
        return float(deflated_sharpe_ratio(path_returns[0], sr_trials))
    return float(median_path_deflated_sharpe_ratio(path_returns, sr_trials))


def make_dsr_scorer(sr_trials: ArrayLike):
    """
    Build a scorer closure for sklearn APIs that accept scorer callables.
    """

    clean_sr_trials = _to_finite_1d(sr_trials)

    def scorer(estimator, X, y=None) -> float:
        del y
        prediction = estimator.predict(X)
        path_returns = extract_path_returns(prediction)
        if len(path_returns) == 1:
            return float(deflated_sharpe_ratio(path_returns[0], clean_sr_trials))
        return float(median_path_deflated_sharpe_ratio(path_returns, clean_sr_trials))

    return scorer


__all__ = [
    "EULER_MASCHERONI",
    "deflated_sharpe_ratio",
    "dsr_scorer",
    "expected_max_sharpe",
    "extract_path_returns",
    "make_dsr_scorer",
    "median_path_deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
]
