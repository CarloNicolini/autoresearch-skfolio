# autoresearch (quantitative finance)

Autonomous portfolio research: an agent iterates on a single training script to improve one robust out-of-sample metric across multiple datasets and validation regimes, using [skfolio](https://skfolio.org/).

The spirit should match a strong single-file pretraining script:

- one main script (`train.py`)
- directly editable hyperparameter blocks, no CLI flag maze
- fast fail on obviously broken ideas
- explicit metrics and explicit guardrails
- easy for a future autonomous agent to read, modify, and rerun honestly

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar13`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main/master.
3. **Read the in-scope files**: The repo is small. Read these for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed dataset and scoring helpers. It loads the dataset suite, builds reversed counterparts, and exposes helper functions used by `train.py`. **Prefer not to modify.**
   - `train.py` — the main research surface. It contains both model construction and the robust validation suite.
4. **Verify data**: Run `uv run prepare.py` to confirm datasets load (asset-only, factor-aware, and reversed cases).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs the full evaluation once: all datasets, reversed counterparts, and the full validation suite in `train.py`, under one time budget. Launch with: `uv run train.py`.

**What you CAN do:**

- Modify `train.py` only in most experiments.
- Change the experiment surface in `EXPERIMENT` and the helper builders: model family, `prior_estimator`, covariance/mu estimators, `risk_measure`, `objective_function`, pipelines, constraints, long/short limits, L1/L2, etc.
- Add new skfolio feature branches when they fit the current architecture cleanly: additional priors, uncertainty sets, pre-selection steps, clustering/ensemble methods, or nested search wrappers.
- Add honest missing-data handling branches. Prefer fold-local logic such as `SelectComplete`, pairwise estimators, and controlled imputation over global cleaning.
- Add explicit friction controls such as `transaction_costs`, `management_fees`, `max_turnover`, and `l1_coef` when supported by the optimizer.
- Adjust the validation suite in `train.py` only when you are making it more robust or more representative, not easier.

**What you CANNOT do:**

- Lightly weaken validation to inflate the score.
- Change the metric definition casually. There must always be one scalar metric and it must remain hard to game.
- Install new packages. Use only dependencies already in `pyproject.toml` (skfolio, sklearn, pandas, numpy).

**Goal: get the highest val_sharpe.**

`val_sharpe` must remain one scalar metric with this meaning:

- For each dataset/CV case, compute the annualized out-of-sample Sharpe ratio for every reconstructed path.
- Collapse that case to one number using the **median** path Sharpe.
- Aggregate all case scores with an **equal-weight mean**.
- If a case errors or produces no finite paths, assign it a hard failure floor.

This keeps the metric simple while forcing robustness into the validation design rather than into a complicated objective formula.

**Simplicity criterion**: Prefer simpler changes. A small gain that adds heavy complexity may not be worth it. Removing complexity while keeping or improving results is a win.

**First run**: Run the script as-is to establish the baseline.

## Baseline Ladder

Every experiment should be interpretable relative to a fixed benchmark ladder, not
only relative to the previous commit.

The built-in ladder in `train.py` is:

1. `equal_weight_baseline`
2. `inverse_volatility_baseline`
3. `mean_risk_baseline`
4. `risk_budgeting_baseline`
5. `hierarchical_risk_parity_baseline`

Guiding rule:

- A new method should state clearly which baseline it is trying to beat and why.
- If it cannot beat at least the simpler baselines robustly, it is not a strong value proposition.
- Improvements should be explained in terms of where they come from, not only by quoting one scalar score.

## Validation Philosophy

The validation suite in `train.py` is the real moat. It should be stricter than the model.

Use three kinds of outer validation:

1. `WalkForward` for standard sequential rebalancing.
2. `CombinatorialPurgedCV` for leakage-resistant path recombination and more adverse train/test slicing.
3. `MultipleRandomizedCV` for random contiguous windows and random asset subsets on eligible asset-only datasets.

Guiding rules:

- Original and reversed datasets must both be included.
- Factor-aware cases should exist so the agent can use `FactorModel` and related ideas.
- Asset-only cases should also exist so the agent can use randomized sub-universe backtests.
- A feature that only works on some datasets should degrade gracefully or fall back cleanly on the others.
- A model that only shines on one split is not interesting.
- Missing-data handling must stay inside the fold. Training-time asset selection, moment estimation, and any imputation must be fit on the training slice only.
- `CombinatorialPurgedCV` is a first-class guardrail, not an optional extra. If a parameterization is invalid for a dataset, the script should shrink to a valid conservative configuration or skip that case cleanly rather than crash.
- `MultipleRandomizedCV` is a stress test against accidental tuning to one universe or one date range. If it breaks, fix the split logic or data-handling logic instead of disabling it casually.

## Missing Data Policy

Incomplete data is normal in portfolio research: assets start late, expire, default, delist, or simply have bad vendor coverage.

Preferred handling order:

1. Use fold-local selectors such as `SelectComplete` to remove assets that are not investable over the relevant training span.
2. Prefer native missing-data support when it is real and local to the estimator, for example pairwise moment estimators defined inside `train.py`.
3. Use fold-local imputation only when it is required to keep out-of-sample paths well-defined, especially to capture default events without crashing on subsequent NaNs.

Honesty rule:

- Do not claim "native NaN support" unless the estimator truly accepts NaNs during `fit`.
- Pipeline imputation is acceptable only when it is trained inside CV and documented as a pragmatic path-construction choice, not as a free lunch.

The skfolio incomplete-data example is the reference pattern for this mindset: [`SelectComplete` plus fold-local imputation](https://skfolio.org/auto_examples/pre_selection/plot_4_incomplete_dataset.html).

## Backtest Overfitting Sources

Future agents should explicitly guard against these failure modes:

1. Survivorship bias: excluding dead, expired, or late-inception assets because they are inconvenient.
2. Look-ahead bias: fitting selectors, imputers, or hyperparameters on data that includes the future.
3. Leakage from overlapping labels and nearby observations: standard IID cross-validation is not enough for market data.
4. Multiple testing and selection bias: trying many variants and only remembering the winner.
5. Estimation error maximization: classic mean-variance optimizers can overreact to noisy moments.
6. Universe-selection bias: a method may only work on one asset subset or one market family.
7. Regime concentration: a result may come mostly from one time direction, one era, or one stress pocket.
8. Overtrading bias: ignoring turnover, transaction costs, and fees.
9. Storytelling bias: reporting the scalar metric without showing where gains actually came from.
10. Silent infrastructure bias: crashes, invalid split configurations, or hidden fallback behavior can selectively remove hard cases.

In this template, the corresponding defenses should be:

- fold-local preprocessing and pre-selection
- `WalkForward`, `CombinatorialPurgedCV`, and `MultipleRandomizedCV`
- reversed datasets and family attribution tables
- baseline ladders and hard failure floors
- friction-aware optimizers where available
- fast fail for obviously broken ideas, but with honest failure scoring rather than silent omission

## Robustness Gates

`val_sharpe` remains the primary objective, but a run should also clear hard
robustness gates. These gates are defined in `train.py` and should remain hard
to game.

Current gate families:

1. Limit the number of failed dataset/CV cases.
2. Limit the gap between original and reversed datasets.
3. Limit the gap between price datasets and relatives datasets.
4. Limit how much of the gain vs the reference baseline comes from one dataset family.

Interpretation rule:

- A run with a higher `val_sharpe` but failing robustness gates is not a strong baseline candidate.
- A run with modestly lower `val_sharpe` but much stronger robustness may still be a better research baseline.

## Dataset Family Attribution

The reporting in `train.py` groups cases into broad families so gains can be
explained, not just measured.

Main families:

1. `price`
2. `relatives`
3. `factor_aware`
4. `original` versus `reversed` direction splits

Expected behavior:

- Good strategies should not depend on one family only.
- If a gain is concentrated in one family, the experiment description should say so explicitly.
- The family attribution table should be used to explain the value proposition of a new idea.

## Search Space

The point is not to hardcode a single family. The point is to let agents traverse skfolio's feature space methodically.

Primary axes:

1. **Optimizer family**
   - `MeanRisk`
   - `RiskBudgeting`
   - `HierarchicalRiskParity`
   - `NestedClustersOptimization`
   - later: stacking/ensemble methods
2. **Missing-data policy**
   - `nan_handling="pipeline"`: `SelectComplete` plus fold-local zero imputation for robust path construction
   - `nan_handling="native"`: fold-local asset selection plus native/pairwise moment estimators that genuinely accept NaNs in training
   - `nan_handling="none"` only for already-clean datasets or deliberately strict ablations
3. **Prior and moment estimation**
   - empirical priors
   - `EmpiricalMu`, `ShrunkMu`, `EWMu`
   - pairwise empirical moments for native NaN support
   - `EmpiricalCovariance`, `LedoitWolf`, `EWCovariance`, `DenoiseCovariance`, `GerberCovariance`
   - factor-aware priors when targets exist
4. **Risk / objective pair**
   - variance / semi-variance / CVaR / CDaR / etc.
   - minimize risk vs maximize ratio vs utility
5. **Constraints and frictions**
   - long-only vs long-short
   - max weight / budget / turnover / transaction costs / fees
   - regularization (`l1_coef`, `l2_coef`)
6. **Pre-selection**
   - k-extremes
   - correlation dropping
   - completeness filters
   - non-expiring asset rules where metadata exists
7. **Model-selection wrappers**
   - nested `GridSearchCV`
   - nested `RandomizedSearchCV`
8. **Advanced priors and stress tools**
   - `FactorModel`
   - uncertainty sets
   - `SyntheticData`
   - `EntropyPooling`
   - `OpinionPooling`
   - later only if they can be parameterized mechanically

## Strategy Composition Slots

Treat `train.py` as a small research DSL with explicit strategy slots.

Current slots:

1. `nan_handling`
2. `preprocessor_kind`
3. `pre_selector_kind`
4. `optimizer_kind`
5. `prior_kind`
6. `post_processor_kind`

Research rule:

- Prefer changing one slot at a time.
- If you add a new method, add it as a clean branch inside the relevant slot builder.
- Avoid mixing unrelated slot changes in one experiment unless the earlier ablations already justified them.

## Research Discipline

Follow these rules when exploring:

1. Change one main axis at a time. Do not mix ten ideas into one diff.
2. Start from the simplest family that could plausibly work. Only escalate complexity after a plateau.
3. Treat `train.py` as a small research DSL. Add clean branches and helpers instead of piling ad hoc logic everywhere.
4. If using factor-aware priors, make them dataset-aware. Do not let them crash asset-only cases.
5. Missing-data handling must be explicit. Prefer native support when real, otherwise use fold-local pipeline logic and say so.
6. If using `GridSearchCV` or `RandomizedSearchCV`, use them only inside the estimator built for each outer fold. Outer validation must stay out-of-sample.
7. Do not invent discretionary Black-Litterman or entropy-pooling views unless they are deterministic and reproducible from the data or explicitly supplied by the human.
8. Use ensemble or stacking methods only after you already have a few strong, diverse base estimators.
9. Prefer robust baselines over exotic constructions that only improve one corner case.
10. Every experiment should declare a hypothesis, expected benefit, expected risk, and reference baseline.
11. Use the grouped summaries and gain attribution output to explain why an improvement is believable.
12. Keep the file hackable: edit direct config blocks and small builder branches instead of introducing a CLI or a framework.

## Output format

The script prints a summary like:

```text
---
val_sharpe:       0.452123
robust_pass:      True
total_seconds:   47.3
```

Extract the main metric with:

```bash
grep "^val_sharpe:" run.log
```

## Logging results

When an experiment finishes, append one row to `results.tsv` (tab-separated). Header and columns:

```text
commit [tab] val_sharpe [tab] status [tab] axis [tab] baseline_ref [tab] hypothesis [tab] description
```

1. Git commit hash (short, 7 chars)
2. val_sharpe (e.g. 0.452123) — use -999.0 or similar for crashes
3. status: `keep`, `discard`, or `crash`
4. The main changed axis
5. The baseline reference used for attribution
6. The experiment hypothesis
7. Short description of what this experiment tried

Example:

```text
commit [tab] val_sharpe [tab] status [tab] axis [tab] baseline_ref [tab] hypothesis [tab] description
a1b2c3d [tab] 0.452123 [tab] keep [tab] baseline [tab] inverse_volatility_baseline [tab] Mean-risk should beat inverse volatility [tab] baseline MeanRisk
b2c3d4e [tab] 0.481200 [tab] keep [tab] covariance [tab] mean_risk_baseline [tab] Gerber covariance may be more robust [tab] RiskBudgeting + GerberCovariance
c3d4e5f [tab] 0.440000 [tab] discard [tab] optimizer [tab] mean_risk_baseline [tab] HRP may diversify better [tab] HierarchicalRiskParity (worse)
d4e5f6g [tab] -999.0 [tab] crash [tab] prior [tab] mean_risk_baseline [tab] factor prior may help [tab] invalid prior_estimator
```

The script prints a `results_tsv_row:` line to make this logging easier and more consistent.

## The experiment loop

Run on a dedicated branch (e.g. `autoresearch/mar13`).

LOOP:

1. Inspect current branch/commit.
2. Edit `train.py` with one experimental idea.
   - Usually this means the model family, prior, risk measure, constraints, or a new clean skfolio feature branch.
   - More rarely it means strengthening the validation suite.
   - Update the experiment metadata so the hypothesis and changed axis are explicit.
3. `git commit`
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_sharpe:\|^failed_cases:\|^robust_pass:\|^total_seconds:" run.log`
6. If the run crashed (no val_sharpe), run `tail -n 50 run.log` to debug. Fix trivial bugs and re-run; if the idea is broken, log as `crash` and move on.
7. Read the family summary, direction summary, and gain attribution tables. Make sure the improvement is broad enough to be believable.
8. Append the row to `results.tsv`.
9. If val_sharpe improved and the robustness gates still pass, keep the commit and advance.
10. If val_sharpe stayed the same or decreased, or robustness got materially worse, discard and move on.

You are an autonomous researcher: keep improvements, discard regressions, advance the branch. Rewind only sparingly if stuck.

**Timeout**: If a run exceeds the time budget (see `prepare.py`), it may return a partial metric; treat long stalls as failure if needed.

**Crashes**: Log as `crash` and skip, or fix and re-run if it’s a small bug.

**NEVER STOP**: Once the loop has started, do not pause to ask the human whether to continue. Run until the user stops you.
