# autoresearch (quantitative finance)

Autonomous portfolio research: an agent edits a single training script to improve out-of-sample portfolio performance. The setup uses [skfolio](https://skfolio.org/) for portfolio optimization and runs fixed, broad validation over multiple datasets, their reversed-return counterparts, and more than one cross-validation regime.

## Idea

Give an AI agent a minimal but real portfolio-research harness: fixed data helpers in `prepare.py`, and one file to change: `train.py`. The agent tries different models (`MeanRisk`, `RiskBudgeting`, `HierarchicalRiskParity`, `NestedClustersOptimization`), priors, covariance estimators, risk measures, pre-selection steps, and hyperparameters. Each run performs a robust outer-validation suite on all datasets; improvements are kept, regressions discarded. You get a reproducible research loop and a `results.tsv` log.

## How it works

- **`prepare.py`** — Fixed. Loads skfolio datasets, adds reversed-return variants, exposes factor-aware cases, and provides score-extraction helpers. Keep it boring.
- **`train.py`** — The main research surface. It builds the portfolio model and defines the outer-validation suite (`WalkForward`, `CombinatorialPurgedCV`, `MultipleRandomizedCV` when eligible).
- **`program.md`** — Instructions for the agent: setup, experiment loop, how to log results and when to keep/discard.

By design, each run does one full evaluation over every dataset/CV case. The metric is **`val_sharpe`**: the equal-weight mean of per-case median annualized out-of-sample Sharpe ratios.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (skfolio, sklearn, pandas, numpy)
uv sync

# 3. Verify datasets load
uv run prepare.py

# 4. Run one experiment (build model in train.py, run fixed evaluation, print scores)
uv run train.py
```

To run in autonomous mode, point your agent at `program.md` and let it iterate on `train.py`, committing and logging to `results.tsv`.

## Project structure

```text
prepare.py      — constants, data helpers, reversed datasets, score extraction
train.py        — model definition, validation suite, one-metric evaluation
program.md      — agent instructions
pyproject.toml  — dependencies (skfolio, scikit-learn, pandas, numpy)
results.tsv     — experiment log (commit, val_sharpe, status, description)
```

## Design choices

- **Single file to edit.** Only `train.py` is changed; scope stays small and diffs clear.
- **Single-file research surface.** `train.py` owns both model construction and the robust validation suite.
- **Fixed helper layer.** `prepare.py` stays simple and stable so experiments stay comparable.
- **Multiple datasets and reversed.** Evaluation runs on several return series and their negated versions to stress-test directionality.
- **One metric, hard validation.** The score stays simple because the outer CV suite is intentionally strict.
- **No GPU.** Pure sklearn/skfolio; runs on any machine.

## License

MIT
