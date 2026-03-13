"""
Single-file autonomous portfolio research surface.

`train.py` owns two things:
1. the experiment family the agent is searching over
2. the robust validation suite used to collapse results to one scalar metric
"""

from dataclasses import asdict, dataclass
import random
import time

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from skfolio import RiskMeasure
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
    cross_val_predict,
)
from skfolio.moments import DenoiseCovariance, EWMu, GerberCovariance, ShrunkMu
from skfolio.optimization import (
    HierarchicalRiskParity,
    MeanRisk,
    NestedClustersOptimization,
    ObjectiveFunction,
    RiskBudgeting,
)
from skfolio.pre_selection import SelectKExtremes
from skfolio.prior import EmpiricalPrior, FactorModel

from prepare import DatasetCase, TIME_BUDGET, extract_path_sharpes, get_all_datasets


@dataclass(frozen=True)
class ExperimentConfig:
    # This dataclass is the primary research surface: agents can mutate these
    # fields to explore new estimators, priors, constraints, and portfolio
    # construction choices without rewriting the evaluation loop.
    model_family: str = "mean_risk"
    objective: str = "maximize_ratio"
    risk_measure: str = "variance"
    prior_kind: str = "empirical"
    mu_estimator: str = "shrunk"
    covariance_estimator: str = "denoise"
    use_preselection: bool = False
    preselection_k: int = 10
    allow_short: bool = False
    max_long: float = 0.20
    max_short: float = 0.20
    l2_coef: float = 0.01
    # Prefer deterministic execution over maximum throughput.
    n_jobs: int = 1


@dataclass(frozen=True)
class ValidationConfig:
    # Keep validation separate from model design so strategy changes can be
    # compared under a stable, reproducible protocol.
    walk_forward_train_size: int = 252
    walk_forward_test_size: int = 63
    purged_n_folds: int = 8
    purged_n_test_folds: int = 2
    purged_size: int = 5
    embargo_size: int = 5
    randomized_subsamples: int = 6
    randomized_window_size: int = 756
    randomized_min_assets: int = 8
    seed: int = 0
    fail_case_score: float = -5.0


@dataclass(frozen=True)
class EvaluationSummary:
    val_sharpe: float
    details: pd.DataFrame


# These defaults define the current research baseline. Future agent iterations
# should update them deliberately so comparisons stay easy to interpret.
EXPERIMENT = ExperimentConfig()
VALIDATION = ValidationConfig()


def set_global_seed(seed: int) -> None:
    # Seed the common random generators used directly in this script and by the
    # randomized validation helpers.
    random.seed(seed)
    np.random.seed(seed)


def build_mu_estimator(name: str):
    # Keep the estimator factories small and explicit: adding a new option should
    # mean adding one new branch and a clear name in the config.
    if name == "none":
        return None
    if name == "shrunk":
        return ShrunkMu()
    if name == "ewm":
        return EWMu(alpha=0.10)
    raise ValueError(f"Unknown mu_estimator: {name}")


def build_covariance_estimator(name: str):
    if name == "none":
        return None
    if name == "denoise":
        return DenoiseCovariance()
    if name == "gerber":
        return GerberCovariance()
    raise ValueError(f"Unknown covariance_estimator: {name}")


def build_empirical_prior(config: ExperimentConfig):
    # Centralize prior wiring so changes to the return/covariance stack only need
    # to happen in one place.
    return EmpiricalPrior(
        mu_estimator=build_mu_estimator(config.mu_estimator),
        covariance_estimator=build_covariance_estimator(config.covariance_estimator),
    )


def build_prior(config: ExperimentConfig, dataset: DatasetCase):
    # Dataset-aware fallback is the key simplification: one research config can
    # still exploit factor-aware priors where targets exist without breaking
    # asset-only datasets.
    if config.prior_kind == "factor" and dataset.y is not None:
        return FactorModel()
    if config.prior_kind in {"empirical", "factor"}:
        return build_empirical_prior(config)
    raise ValueError(f"Unknown prior_kind: {config.prior_kind}")


def build_mean_risk(config: ExperimentConfig, dataset: DatasetCase):
    # This is the current baseline optimizer and the simplest place to test new
    # objective/risk/constraint ideas.
    min_weights = -config.max_short if config.allow_short else 0.0
    return MeanRisk(
        objective_function=ObjectiveFunction[config.objective.upper()],
        risk_measure=RiskMeasure[config.risk_measure.upper()],
        prior_estimator=build_prior(config, dataset),
        min_weights=min_weights,
        max_weights=config.max_long,
        l2_coef=config.l2_coef,
        raise_on_failure=False,
    )


def build_risk_budgeting(config: ExperimentConfig, dataset: DatasetCase):
    # Alternative construction step: same prior plumbing, different portfolio
    # assembly logic.
    return RiskBudgeting(
        risk_measure=RiskMeasure[config.risk_measure.upper()],
        prior_estimator=build_prior(config, dataset),
        raise_on_failure=False,
    )


def build_hierarchical_risk_parity(config: ExperimentConfig, dataset: DatasetCase):
    # Hierarchical methods let the agent explore clustering-based diversification
    # without changing the broader evaluation protocol.
    return HierarchicalRiskParity(
        risk_measure=RiskMeasure[config.risk_measure.upper()],
        prior_estimator=build_prior(config, dataset),
        raise_on_failure=False,
    )


def build_nested_clusters(config: ExperimentConfig, dataset: DatasetCase):
    # This is already a multi-step strategy: an inner optimizer builds cluster
    # sleeves and an outer optimizer allocates across them. It is a useful
    # template for richer composed research pipelines.
    return NestedClustersOptimization(
        inner_estimator=MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
            risk_measure=RiskMeasure[config.risk_measure.upper()],
            prior_estimator=build_prior(config, dataset),
            min_weights=0.0,
            max_weights=0.50,
            l2_coef=config.l2_coef,
            raise_on_failure=False,
        ),
        outer_estimator=RiskBudgeting(risk_measure=RiskMeasure.VARIANCE, raise_on_failure=False),
        cv=KFold(n_splits=3, shuffle=False),
        n_jobs=config.n_jobs,
    )


def build_estimator(config: ExperimentConfig, dataset: DatasetCase):
    # Model-family dispatch is intentionally explicit so the searchable strategy
    # space stays inspectable and diff-friendly.
    builders = {
        "mean_risk": build_mean_risk,
        "risk_budgeting": build_risk_budgeting,
        "hierarchical_risk_parity": build_hierarchical_risk_parity,
        "nested_clusters": build_nested_clusters,
    }
    try:
        return builders[config.model_family](config, dataset)
    except KeyError as exc:
        raise ValueError(f"Unknown model_family: {config.model_family}") from exc


def build_model(config: ExperimentConfig, dataset: DatasetCase):
    # The full model may be a plain estimator or a multi-step pipeline. This is
    # the natural extension point for richer research workflows.
    estimator = build_estimator(config, dataset)
    if not config.use_preselection:
        return estimator

    set_config(transform_output="pandas")
    return Pipeline(
        [
            # Keep pre-selection explicit in the pipeline so it is evaluated
            # inside CV rather than leaking information across folds.
            ("pre_selection", SelectKExtremes(k=config.preselection_k, highest=True)),
            ("optimization", estimator),
        ]
    )


def get_walk_forward_cv(validation: ValidationConfig) -> WalkForward:
    # Walk-forward approximates the standard research workflow: fit on the past,
    # evaluate on the next block, then roll forward.
    return WalkForward(
        train_size=validation.walk_forward_train_size,
        test_size=validation.walk_forward_test_size,
    )


def get_combinatorial_purged_cv(validation: ValidationConfig) -> CombinatorialPurgedCV:
    # Purged CV is the more conservative protocol for time series because it
    # reduces leakage from overlapping samples and nearby observations.
    return CombinatorialPurgedCV(
        n_folds=validation.purged_n_folds,
        n_test_folds=validation.purged_n_test_folds,
        purged_size=validation.purged_size,
        embargo_size=validation.embargo_size,
    )


def get_multiple_randomized_cv(dataset: DatasetCase, validation: ValidationConfig):
    # This stress test adds controlled randomness over windows and asset subsets.
    # With a fixed seed it remains reproducible while still probing robustness.
    if dataset.y is not None or dataset.X.shape[1] < validation.randomized_min_assets:
        return None

    asset_subset_size = min(max(4, dataset.X.shape[1] // 2), dataset.X.shape[1])
    window_size = min(validation.randomized_window_size, len(dataset.X))
    return MultipleRandomizedCV(
        walk_forward=get_walk_forward_cv(validation),
        n_subsamples=validation.randomized_subsamples,
        asset_subset_size=asset_subset_size,
        window_size=window_size,
        random_state=validation.seed,
    )


def iter_validation_cases(validation: ValidationConfig):
    # Every dataset is evaluated under the same family of validation schemes so a
    # single strategy is not overfit to one market regime or one test protocol.
    for dataset in get_all_datasets():
        yield dataset, "walk_forward", get_walk_forward_cv(validation)
        yield dataset, "combinatorial_purged", get_combinatorial_purged_cv(validation)
        randomized_cv = get_multiple_randomized_cv(dataset, validation)
        if randomized_cv is not None:
            yield dataset, "multiple_randomized", randomized_cv


def summarize_case_scores(path_scores: np.ndarray, fail_case_score: float) -> dict[str, float | int]:
    # Collapse each validation case to a small set of comparable diagnostics plus
    # one scalar score for the outer research loop.
    finite_scores = path_scores[np.isfinite(path_scores)]
    if finite_scores.size == 0:
        return {
            "n_paths": int(path_scores.size),
            "n_finite_paths": 0,
            "path_mean": float("nan"),
            "path_std": float("nan"),
            "case_score": fail_case_score,
        }

    return {
        "n_paths": int(path_scores.size),
        "n_finite_paths": int(finite_scores.size),
        "path_mean": float(finite_scores.mean()),
        "path_std": float(finite_scores.std()),
        # One simple metric per case: the median OOS Sharpe across paths.
        "case_score": float(np.median(finite_scores)),
    }


def evaluate_experiment(
    config: ExperimentConfig = EXPERIMENT,
    validation: ValidationConfig = VALIDATION,
    timeout_seconds: float | None = TIME_BUDGET,
) -> EvaluationSummary:
    # This is the core benchmark harness: build one strategy, run it across the
    # full validation matrix, and return both the aggregate score and per-case
    # diagnostics for later inspection.
    set_global_seed(validation.seed)
    rows: list[dict[str, str | float | int | bool]] = []
    start = time.time()

    for dataset, cv_name, cv in iter_validation_cases(validation):
        if timeout_seconds is not None and (time.time() - start) >= timeout_seconds:
            break

        error = ""
        try:
            model = build_model(config, dataset)
            # `cross_val_predict` is the key abstraction here: any estimator or
            # pipeline that follows the expected API can be dropped into the same
            # evaluation harness.
            portfolios = cross_val_predict(model, dataset.X, y=dataset.y, cv=cv, n_jobs=config.n_jobs)
            path_scores = extract_path_sharpes(portfolios)
        except Exception as exc:
            # Research runs should fail soft: record the issue, score the case as
            # failed, and continue so broad searches do not stop on one bad idea.
            path_scores = np.asarray([], dtype=float)
            error = f"{type(exc).__name__}: {exc}"

        score_stats = summarize_case_scores(path_scores, validation.fail_case_score)
        rows.append(
            {
                "dataset": dataset.name,
                "cv": cv_name,
                "n_obs": dataset.X.shape[0],
                "n_assets": dataset.X.shape[1],
                "has_targets": dataset.y is not None,
                "error": error,
                **score_stats,
            }
        )

    details = pd.DataFrame(rows)
    if details.empty:
        return EvaluationSummary(val_sharpe=float("-inf"), details=details)

    return EvaluationSummary(val_sharpe=float(details["case_score"].mean()), details=details)


def format_details_table(summary: EvaluationSummary) -> str:
    # Keep the console output stable and human-readable because agents and humans
    # will both use it as a lightweight audit trail.
    details = summary.details.copy()
    if details.empty:
        return "(no completed evaluation cases)"

    display = details.copy()
    for col in ["path_mean", "path_std", "case_score"]:
        display[col] = display[col].map(lambda value: f"{value:.6f}" if pd.notna(value) else "nan")
    return display.to_string(index=False)


def main():
    # `main` makes the file runnable as a standalone benchmark script, which is
    # useful for quick baselines before folding changes into larger agent loops.
    t_start = time.time()
    summary = evaluate_experiment(EXPERIMENT, VALIDATION, timeout_seconds=TIME_BUDGET)
    total_seconds = time.time() - t_start
    failed_cases = int((summary.details["n_finite_paths"] == 0).sum()) if not summary.details.empty else 0

    print("Experiment config:")
    for key, value in asdict(EXPERIMENT).items():
        print(f"  {key}: {value}")
    print()
    print("Validation config:")
    for key, value in asdict(VALIDATION).items():
        print(f"  {key}: {value}")
    print()
    print("Per-case results:")
    print(format_details_table(summary))
    print("---")
    print(f"val_sharpe:       {summary.val_sharpe:.6f}")
    print(f"evaluated_cases:  {len(summary.details)}")
    print(f"failed_cases:     {failed_cases}")
    print(f"total_seconds:    {total_seconds:.1f}")


if __name__ == "__main__":
    main()
