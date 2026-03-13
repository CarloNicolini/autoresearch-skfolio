"""
Dataset and scoring helpers for autonomous portfolio research.

`prepare.py` stays intentionally simple:
- load a fixed suite of skfolio datasets
- build reversed counterparts
- expose helpers used by `train.py`
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.measures import RatioMeasure
from skfolio.preprocessing import prices_to_returns

TIME_BUDGET = 300


@dataclass(frozen=True)
class DatasetCase:
    name: str
    X: pd.DataFrame
    y: pd.DataFrame | None = None


def _negate_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    return pd.DataFrame(-frame.to_numpy(), index=frame.index, columns=frame.columns)


def _load_sp500_and_factors() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()
    X = prices_to_returns(prices)
    factors = prices_to_returns(factor_prices)
    common_index = X.index.intersection(factors.index)
    return X.loc[common_index], factors.loc[common_index]


def get_all_datasets() -> list[DatasetCase]:
    """
    Load the fixed dataset suite.

    We keep both an asset-only S&P 500 case and a factor-aware S&P 500 case so
    the agent can explore both standard priors and `FactorModel`-style priors.
    """
    sp500_returns, sp500_factors = _load_sp500_and_factors()
    factor_returns = prices_to_returns(load_factors_dataset())

    return [
        DatasetCase("sp500", sp500_returns),
        DatasetCase("sp500_reversed", _negate_frame(sp500_returns)),
        DatasetCase("sp500_factor", sp500_returns, sp500_factors),
        DatasetCase("sp500_factor_reversed", _negate_frame(sp500_returns), _negate_frame(sp500_factors)),
        DatasetCase("factors", factor_returns),
        DatasetCase("factors_reversed", _negate_frame(factor_returns)),
    ]


def extract_path_sharpes(portfolios: object) -> np.ndarray:
    """
    Return one annualized Sharpe per out-of-sample path.

    `cross_val_predict` returns:
    - `MultiPeriodPortfolio` for single-path CV
    - `Population` for multi-path CV
    """
    if hasattr(portfolios, "annualized_sharpe_ratio"):
        return np.asarray([float(portfolios.annualized_sharpe_ratio)], dtype=float)
    return np.asarray(portfolios.measures(RatioMeasure.ANNUALIZED_SHARPE_RATIO), dtype=float)


def describe_datasets() -> pd.DataFrame:
    rows = []
    for dataset in get_all_datasets():
        rows.append(
            {
                "dataset": dataset.name,
                "n_obs": dataset.X.shape[0],
                "n_assets": dataset.X.shape[1],
                "has_targets": dataset.y is not None,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Verifying datasets...")
    print(describe_datasets().to_string(index=False))
    print()
    print("Done. Ready to run train.py.")
