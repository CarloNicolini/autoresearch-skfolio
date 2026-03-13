"""
Single-file autonomous portfolio research surface.

`train.py` owns three things:
1. the experiment surface the agent is allowed to search over
2. the robust validation suite used to score a strategy
3. the reporting layer that explains why a strategy is or is not valuable
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
    EqualWeighted,
    HierarchicalRiskParity,
    InverseVolatility,
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
    # These metadata fields are the research ledger: every experiment should say
    # what changed, why it might help, and which baseline it aims to beat.
    experiment_name: str = "mean_risk_baseline"
    changed_axis: str = "baseline"
    hypothesis: str = "A simple mean-risk allocator should beat naive diversification."
    expected_benefit: str = "Use estimated moments to improve risk-adjusted allocation."
    expected_risk: str = "Moment estimation noise may hurt robustness on some datasets."
    baseline_reference: str = "inverse_volatility_baseline"
    # These are explicit strategy-composition slots. Future agents should prefer
    # changing one slot at a time so ablations stay interpretable.
    preprocessor_kind: str = "none"
    pre_selector_kind: str = "none"
    optimizer_kind: str = "mean_risk"
    post_processor_kind: str = "none"
    objective: str = "maximize_ratio"
    risk_measure: str = "variance"
    prior_kind: str = "empirical"
    mu_estimator: str = "shrunk"
    covariance_estimator: str = "denoise"
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
    include_baseline_ladder: bool = True
    baseline_timeout_seconds: float | None = 120.0


@dataclass(frozen=True)
class RobustnessConfig:
    # Hard gates keep the research loop honest. The scalar metric is primary, but
    # a high score with obvious fragility should not count as a real improvement.
    max_failed_cases: int = 6
    max_original_vs_reversed_gap: float = 0.75
    max_price_vs_relatives_gap: float = 0.75
    max_family_gain_share: float = 0.60


@dataclass(frozen=True)
class ReportingConfig:
    top_case_count: int = 3


@dataclass(frozen=True)
class EvaluationSummary:
    experiment_name: str
    val_sharpe: float
    details: pd.DataFrame
    family_summary: pd.DataFrame
    direction_summary: pd.DataFrame
    diagnostics: dict[str, float | int | bool | str]
    family_gain_attribution: pd.DataFrame


# These defaults define the current research baseline. Future agent iterations
# should update them deliberately so comparisons stay easy to interpret.
EXPERIMENT = ExperimentConfig()
VALIDATION = ValidationConfig()
ROBUSTNESS = RobustnessConfig()
REPORTING = ReportingConfig()


def get_baseline_ladder() -> list[ExperimentConfig]:
    # The ladder gives the agent a concrete value proposition target: every new
    # method should explain why it is better than one or more simple baselines.
    return [
        ExperimentConfig(
            experiment_name="equal_weight_baseline",
            changed_axis="baseline_ladder",
            hypothesis="Equal weighting sets the naive diversification floor.",
            expected_benefit="Very robust simple benchmark.",
            expected_risk="Ignores relative risk and return structure.",
            baseline_reference="none",
            optimizer_kind="equal_weight",
            prior_kind="empirical",
            mu_estimator="none",
            covariance_estimator="none",
        ),
        ExperimentConfig(
            experiment_name="inverse_volatility_baseline",
            changed_axis="baseline_ladder",
            hypothesis="Inverse-volatility should improve on equal weight by scaling down risky names.",
            expected_benefit="Simple risk-aware benchmark.",
            expected_risk="Still ignores cross-asset correlation.",
            baseline_reference="equal_weight_baseline",
            optimizer_kind="inverse_volatility",
            prior_kind="empirical",
            mu_estimator="none",
        ),
        ExperimentConfig(
            experiment_name="mean_risk_baseline",
            changed_axis="baseline_ladder",
            hypothesis="Mean-risk should improve on inverse-volatility when moment estimates are useful.",
            expected_benefit="Uses expected return and covariance information.",
            expected_risk="Sensitive to estimation noise.",
            baseline_reference="inverse_volatility_baseline",
            optimizer_kind="mean_risk",
        ),
        ExperimentConfig(
            experiment_name="risk_budgeting_baseline",
            changed_axis="baseline_ladder",
            hypothesis="Risk budgeting may trade some efficiency for robustness.",
            expected_benefit="More stable diversification profile.",
            expected_risk="Can underuse return information.",
            baseline_reference="mean_risk_baseline",
            optimizer_kind="risk_budgeting",
        ),
        ExperimentConfig(
            experiment_name="hierarchical_risk_parity_baseline",
            changed_axis="baseline_ladder",
            hypothesis="Hierarchical structure may improve diversification under unstable covariances.",
            expected_benefit="Clustering-aware allocation.",
            expected_risk="Can be overly driven by the clustering structure.",
            baseline_reference="risk_budgeting_baseline",
            optimizer_kind="hierarchical_risk_parity",
        ),
    ]


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


def build_equal_weight(config: ExperimentConfig, dataset: DatasetCase):
    return EqualWeighted(raise_on_failure=False)


def build_inverse_volatility(config: ExperimentConfig, dataset: DatasetCase):
    return InverseVolatility(
        prior_estimator=build_prior(config, dataset),
        raise_on_failure=False,
    )


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


def build_optimizer(config: ExperimentConfig, dataset: DatasetCase):
    # Optimizer dispatch is intentionally explicit so the searchable strategy
    # space stays inspectable and diff-friendly.
    builders = {
        "equal_weight": build_equal_weight,
        "inverse_volatility": build_inverse_volatility,
        "mean_risk": build_mean_risk,
        "risk_budgeting": build_risk_budgeting,
        "hierarchical_risk_parity": build_hierarchical_risk_parity,
        "nested_clusters": build_nested_clusters,
    }
    try:
        return builders[config.optimizer_kind](config, dataset)
    except KeyError as exc:
        raise ValueError(f"Unknown optimizer_kind: {config.optimizer_kind}") from exc


def build_preprocessor_steps(config: ExperimentConfig, dataset: DatasetCase) -> list[tuple[str, object]]:
    # This slot exists so future agents can add deterministic feature transforms
    # without rewriting the rest of the pipeline builder.
    if config.preprocessor_kind == "none":
        return []
    raise ValueError(f"Unknown preprocessor_kind: {config.preprocessor_kind}")


def build_pre_selector_steps(config: ExperimentConfig, dataset: DatasetCase) -> list[tuple[str, object]]:
    # Keep pre-selection explicit in the pipeline so it is evaluated inside CV
    # rather than leaking information across folds.
    if config.pre_selector_kind == "none":
        return []
    if config.pre_selector_kind == "k_extremes":
        set_config(transform_output="pandas")
        return [("pre_selection", SelectKExtremes(k=config.preselection_k, highest=True))]
    raise ValueError(f"Unknown pre_selector_kind: {config.pre_selector_kind}")


def build_post_processor_steps(config: ExperimentConfig, dataset: DatasetCase) -> list[tuple[str, object]]:
    # This slot is reserved for future portfolio post-processing stages once a
    # clean sklearn-compatible abstraction is needed.
    if config.post_processor_kind == "none":
        return []
    raise ValueError(f"Unknown post_processor_kind: {config.post_processor_kind}")


def build_model(config: ExperimentConfig, dataset: DatasetCase):
    # The full model may be a plain estimator or a multi-step pipeline. This is
    # the natural extension point for richer research workflows.
    steps = []
    steps.extend(build_preprocessor_steps(config, dataset))
    steps.extend(build_pre_selector_steps(config, dataset))
    steps.append(("optimization", build_optimizer(config, dataset)))
    steps.extend(build_post_processor_steps(config, dataset))
    if len(steps) == 1:
        return steps[0][1]
    return Pipeline(steps)


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


def iter_validation_cases(datasets: list[DatasetCase], validation: ValidationConfig):
    # Every dataset is evaluated under the same family of validation schemes so a
    # single strategy is not overfit to one market regime or one test protocol.
    for dataset in datasets:
        yield dataset, "walk_forward", get_walk_forward_cv(validation)
        yield dataset, "combinatorial_purged", get_combinatorial_purged_cv(validation)
        randomized_cv = get_multiple_randomized_cv(dataset, validation)
        if randomized_cv is not None:
            yield dataset, "multiple_randomized", randomized_cv


def get_dataset_base_name(name: str) -> str:
    return name.removesuffix("_reversed")


def get_dataset_direction(name: str) -> str:
    return "reversed" if name.endswith("_reversed") else "original"


def get_dataset_family(dataset: DatasetCase) -> str:
    base_name = get_dataset_base_name(dataset.name)
    if dataset.y is not None:
        return "factor_aware"
    if "relatives" in base_name:
        return "relatives"
    return "price"


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


def summarize_group_scores(details: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame(
            columns=[
                group_col,
                "n_cases",
                "n_failed_cases",
                "mean_case_score",
                "median_case_score",
                "worst_case_score",
            ]
        )
    summary = (
        details.groupby(group_col, dropna=False)
        .agg(
            n_cases=("case_score", "size"),
            n_failed_cases=("n_finite_paths", lambda values: int((values == 0).sum())),
            mean_case_score=("case_score", "mean"),
            median_case_score=("case_score", "median"),
            worst_case_score=("case_score", "min"),
        )
        .reset_index()
    )
    return summary.sort_values(group_col).reset_index(drop=True)


def build_family_gain_attribution(
    family_summary: pd.DataFrame,
    reference_family_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "dataset_family",
        "reference_mean_case_score",
        "mean_case_score",
        "delta_vs_reference",
        "gain_share",
    ]
    if reference_family_summary is None or family_summary.empty:
        return pd.DataFrame(columns=columns)

    merged = family_summary.merge(
        reference_family_summary[["dataset_family", "mean_case_score"]],
        on="dataset_family",
        how="left",
        suffixes=("", "_reference"),
    )
    merged.rename(columns={"mean_case_score_reference": "reference_mean_case_score"}, inplace=True)
    merged["delta_vs_reference"] = merged["mean_case_score"] - merged["reference_mean_case_score"]
    total_gain = float(merged["delta_vs_reference"].abs().sum())
    if total_gain > 0:
        merged["gain_share"] = merged["delta_vs_reference"].abs() / total_gain
    else:
        merged["gain_share"] = 0.0
    return merged[columns].sort_values("dataset_family").reset_index(drop=True)


def compute_diagnostics(
    details: pd.DataFrame,
    family_summary: pd.DataFrame,
    direction_summary: pd.DataFrame,
    family_gain_attribution: pd.DataFrame,
    robustness: RobustnessConfig,
) -> dict[str, float | int | bool | str]:
    if details.empty:
        return {
            "failed_cases": 0,
            "worst_case_score": float("-inf"),
            "median_case_score": float("-inf"),
            "score_std_across_cases": float("nan"),
            "original_vs_reversed_gap": float("nan"),
            "price_vs_relatives_gap": float("nan"),
            "max_family_gain_share": float("nan"),
            "pass_failed_cases_gate": False,
            "pass_original_vs_reversed_gate": False,
            "pass_price_vs_relatives_gate": False,
            "pass_family_gain_share_gate": False,
            "robust_pass": False,
        }

    failed_cases = int((details["n_finite_paths"] == 0).sum())
    original_mean = float(
        details.loc[details["direction"] == "original", "case_score"].mean()
    )
    reversed_mean = float(
        details.loc[details["direction"] == "reversed", "case_score"].mean()
    )
    price_mean = float(details.loc[details["dataset_family"] == "price", "case_score"].mean())
    relatives_mean = float(
        details.loc[details["dataset_family"] == "relatives", "case_score"].mean()
    )
    original_vs_reversed_gap = abs(original_mean - reversed_mean)
    price_vs_relatives_gap = abs(price_mean - relatives_mean)
    max_family_gain_share = (
        float(family_gain_attribution["gain_share"].max())
        if not family_gain_attribution.empty
        else 0.0
    )
    pass_failed_cases_gate = failed_cases <= robustness.max_failed_cases
    pass_original_vs_reversed_gate = original_vs_reversed_gap <= robustness.max_original_vs_reversed_gap
    pass_price_vs_relatives_gate = (
        np.isnan(price_vs_relatives_gap)
        or price_vs_relatives_gap <= robustness.max_price_vs_relatives_gap
    )
    pass_family_gain_share_gate = max_family_gain_share <= robustness.max_family_gain_share
    robust_pass = all(
        [
            pass_failed_cases_gate,
            pass_original_vs_reversed_gate,
            pass_price_vs_relatives_gate,
            pass_family_gain_share_gate,
        ]
    )
    return {
        "failed_cases": failed_cases,
        "worst_case_score": float(details["case_score"].min()),
        "median_case_score": float(details["case_score"].median()),
        "score_std_across_cases": float(details["case_score"].to_numpy(dtype=float).std()),
        "original_vs_reversed_gap": float(original_vs_reversed_gap),
        "price_vs_relatives_gap": float(price_vs_relatives_gap),
        "max_family_gain_share": float(max_family_gain_share),
        "pass_failed_cases_gate": pass_failed_cases_gate,
        "pass_original_vs_reversed_gate": pass_original_vs_reversed_gate,
        "pass_price_vs_relatives_gate": pass_price_vs_relatives_gate,
        "pass_family_gain_share_gate": pass_family_gain_share_gate,
        "robust_pass": robust_pass,
    }


def evaluate_experiment(
    config: ExperimentConfig,
    validation: ValidationConfig,
    robustness: RobustnessConfig,
    datasets: list[DatasetCase],
    timeout_seconds: float | None = TIME_BUDGET,
    reference_summary: EvaluationSummary | None = None,
) -> EvaluationSummary:
    # This is the core benchmark harness: build one strategy, run it across the
    # full validation matrix, and return both the aggregate score and per-case
    # diagnostics for later inspection.
    set_global_seed(validation.seed)
    rows: list[dict[str, str | float | int | bool]] = []
    start = time.time()

    for dataset, cv_name, cv in iter_validation_cases(datasets, validation):
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
                "dataset_base": get_dataset_base_name(dataset.name),
                "dataset_family": get_dataset_family(dataset),
                "direction": get_dataset_direction(dataset.name),
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
        family_summary = summarize_group_scores(details, "dataset_family")
        direction_summary = summarize_group_scores(details, "direction")
        family_gain_attribution = build_family_gain_attribution(family_summary, None)
        diagnostics = compute_diagnostics(
            details,
            family_summary,
            direction_summary,
            family_gain_attribution,
            robustness,
        )
        return EvaluationSummary(
            experiment_name=config.experiment_name,
            val_sharpe=float("-inf"),
            details=details,
            family_summary=family_summary,
            direction_summary=direction_summary,
            diagnostics=diagnostics,
            family_gain_attribution=family_gain_attribution,
        )

    family_summary = summarize_group_scores(details, "dataset_family")
    direction_summary = summarize_group_scores(details, "direction")
    reference_family_summary = None if reference_summary is None else reference_summary.family_summary
    family_gain_attribution = build_family_gain_attribution(family_summary, reference_family_summary)
    diagnostics = compute_diagnostics(
        details,
        family_summary,
        direction_summary,
        family_gain_attribution,
        robustness,
    )
    return EvaluationSummary(
        experiment_name=config.experiment_name,
        val_sharpe=float(details["case_score"].mean()),
        details=details,
        family_summary=family_summary,
        direction_summary=direction_summary,
        diagnostics=diagnostics,
        family_gain_attribution=family_gain_attribution,
    )


def evaluate_baseline_ladder(
    validation: ValidationConfig,
    robustness: RobustnessConfig,
    datasets: list[DatasetCase],
) -> dict[str, EvaluationSummary]:
    summaries: dict[str, EvaluationSummary] = {}
    if not validation.include_baseline_ladder:
        return summaries

    for config in get_baseline_ladder():
        reference_summary = summaries.get(config.baseline_reference)
        summaries[config.experiment_name] = evaluate_experiment(
            config=config,
            validation=validation,
            robustness=robustness,
            datasets=datasets,
            timeout_seconds=validation.baseline_timeout_seconds,
            reference_summary=reference_summary,
        )
    return summaries


def format_float(value: float) -> str:
    return f"{value:.6f}" if pd.notna(value) else "nan"


def format_details_table(summary: EvaluationSummary) -> str:
    # Keep the console output stable and human-readable because agents and humans
    # will both use it as a lightweight audit trail.
    details = summary.details.copy()
    if details.empty:
        return "(no completed evaluation cases)"

    display = details.copy()
    for col in ["path_mean", "path_std", "case_score"]:
        display[col] = display[col].map(format_float)
    return display.to_string(index=False)


def format_group_summary(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "(no grouped results)"
    display = summary.copy()
    for col in ["mean_case_score", "median_case_score", "worst_case_score"]:
        display[col] = display[col].map(format_float)
    return display.to_string(index=False)


def format_gain_attribution(summary: EvaluationSummary) -> str:
    if summary.family_gain_attribution.empty:
        return "(no reference attribution)"
    display = summary.family_gain_attribution.copy()
    for col in ["reference_mean_case_score", "mean_case_score", "delta_vs_reference", "gain_share"]:
        display[col] = display[col].map(format_float)
    return display.to_string(index=False)


def format_baseline_ladder(baseline_summaries: dict[str, EvaluationSummary]) -> str:
    if not baseline_summaries:
        return "(baseline ladder disabled)"
    rows = []
    ladder_configs = {config.experiment_name: config for config in get_baseline_ladder()}
    for name, summary in baseline_summaries.items():
        config = ladder_configs[name]
        rows.append(
            {
                "experiment_name": name,
                "optimizer_kind": config.optimizer_kind,
                "baseline_reference": config.baseline_reference,
                "val_sharpe": summary.val_sharpe,
                "worst_case_score": summary.diagnostics["worst_case_score"],
                "failed_cases": summary.diagnostics["failed_cases"],
                "robust_pass": summary.diagnostics["robust_pass"],
            }
        )
    display = pd.DataFrame(rows)
    for col in ["val_sharpe", "worst_case_score"]:
        display[col] = display[col].map(format_float)
    return display.to_string(index=False)


def format_extreme_cases(summary: EvaluationSummary, top_case_count: int) -> str:
    if summary.details.empty:
        return "(no completed evaluation cases)"
    columns = ["dataset", "cv", "dataset_family", "direction", "case_score", "error"]
    worst = summary.details.nsmallest(top_case_count, "case_score")[columns].copy()
    worst.insert(0, "segment", "worst")
    best = summary.details.nlargest(top_case_count, "case_score")[columns].copy()
    best.insert(0, "segment", "best")
    display = pd.concat([worst, best], ignore_index=True)
    display["case_score"] = display["case_score"].map(format_float)
    return display.to_string(index=False)


def format_diagnostics(summary: EvaluationSummary) -> str:
    diagnostics = summary.diagnostics
    rows = [
        {"metric": "failed_cases", "value": diagnostics["failed_cases"]},
        {"metric": "worst_case_score", "value": format_float(float(diagnostics["worst_case_score"]))},
        {"metric": "median_case_score", "value": format_float(float(diagnostics["median_case_score"]))},
        {
            "metric": "score_std_across_cases",
            "value": format_float(float(diagnostics["score_std_across_cases"])),
        },
        {
            "metric": "original_vs_reversed_gap",
            "value": format_float(float(diagnostics["original_vs_reversed_gap"])),
        },
        {
            "metric": "price_vs_relatives_gap",
            "value": format_float(float(diagnostics["price_vs_relatives_gap"])),
        },
        {
            "metric": "max_family_gain_share",
            "value": format_float(float(diagnostics["max_family_gain_share"])),
        },
        {"metric": "pass_failed_cases_gate", "value": diagnostics["pass_failed_cases_gate"]},
        {
            "metric": "pass_original_vs_reversed_gate",
            "value": diagnostics["pass_original_vs_reversed_gate"],
        },
        {
            "metric": "pass_price_vs_relatives_gate",
            "value": diagnostics["pass_price_vs_relatives_gate"],
        },
        {
            "metric": "pass_family_gain_share_gate",
            "value": diagnostics["pass_family_gain_share_gate"],
        },
        {"metric": "robust_pass", "value": diagnostics["robust_pass"]},
    ]
    return pd.DataFrame(rows).to_string(index=False)


def build_results_tsv_row(config: ExperimentConfig, summary: EvaluationSummary, status: str) -> str:
    # This row is printed so the experiment ledger can be updated consistently.
    return "\t".join(
        [
            "<commit>",
            f"{summary.val_sharpe:.6f}",
            status,
            config.changed_axis,
            config.baseline_reference,
            config.hypothesis,
            config.experiment_name,
        ]
    )


def main():
    # `main` makes the file runnable as a standalone benchmark script, which is
    # useful for quick baselines before folding changes into larger agent loops.
    t_start = time.time()
    datasets = get_all_datasets()
    baseline_summaries = evaluate_baseline_ladder(VALIDATION, ROBUSTNESS, datasets)
    reference_summary = baseline_summaries.get(EXPERIMENT.baseline_reference)
    summary = evaluate_experiment(
        config=EXPERIMENT,
        validation=VALIDATION,
        robustness=ROBUSTNESS,
        datasets=datasets,
        timeout_seconds=TIME_BUDGET,
        reference_summary=reference_summary,
    )
    total_seconds = time.time() - t_start

    print("Experiment metadata:")
    for key in [
        "experiment_name",
        "changed_axis",
        "hypothesis",
        "expected_benefit",
        "expected_risk",
        "baseline_reference",
    ]:
        print(f"  {key}: {getattr(EXPERIMENT, key)}")
    print()
    print("Experiment config:")
    for key, value in asdict(EXPERIMENT).items():
        print(f"  {key}: {value}")
    print()
    print("Validation config:")
    for key, value in asdict(VALIDATION).items():
        print(f"  {key}: {value}")
    print()
    print("Robustness gates:")
    for key, value in asdict(ROBUSTNESS).items():
        print(f"  {key}: {value}")
    print()
    print("Baseline ladder:")
    print(format_baseline_ladder(baseline_summaries))
    print()
    print("Family summary:")
    print(format_group_summary(summary.family_summary))
    print()
    print("Direction summary:")
    print(format_group_summary(summary.direction_summary))
    print()
    print("Family gain attribution vs reference:")
    print(format_gain_attribution(summary))
    print()
    print("Extreme cases:")
    print(format_extreme_cases(summary, REPORTING.top_case_count))
    print()
    print("Per-case results:")
    print(format_details_table(summary))
    print()
    print("Robustness diagnostics:")
    print(format_diagnostics(summary))
    print("---")
    print(f"val_sharpe:       {summary.val_sharpe:.6f}")
    print(f"evaluated_cases:  {len(summary.details)}")
    print(f"failed_cases:     {summary.diagnostics['failed_cases']}")
    print(f"robust_pass:      {summary.diagnostics['robust_pass']}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print("results_tsv_row:")
    status = "keep" if summary.diagnostics["robust_pass"] else "discard"
    print(build_results_tsv_row(EXPERIMENT, summary, status))


if __name__ == "__main__":
    main()
