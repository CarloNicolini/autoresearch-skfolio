"""
Dataset and scoring helpers for autonomous portfolio research.

`prepare.py` keeps one small job:
- download/cache the dataset `.csv.gz` assets we rely on
- convert price datasets to net returns
- expose both regular and time-reversed cases for `train.py`
"""

from dataclasses import dataclass
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

from skfolio.measures import RatioMeasure
from skfolio.preprocessing import prices_to_returns

TIME_BUDGET = 300
UNIVERSAL_PORTFOLIO_DATA_URL = (
    "https://raw.githubusercontent.com/CarloNicolini/skfolio/"
    "universal_portfolio/src/skfolio/datasets/data"
)
SKFOLIO_DATASETS_URL = (
    "https://raw.githubusercontent.com/skfolio/skfolio-datasets/main/datasets"
)


@dataclass(frozen=True)
class RemoteDatasetSpec:
    name: str
    filename: str
    url: str
    kind: str
    has_datetime_index: bool = True
    synthetic_start: str | None = None


@dataclass(frozen=True)
class DatasetCase:
    name: str
    X: pd.DataFrame
    y: pd.DataFrame | None = None


PRICE_DATASETS: tuple[RemoteDatasetSpec, ...] = (
    RemoteDatasetSpec(
        name="sp500",
        filename="sp500_dataset.csv.gz",
        url=f"{UNIVERSAL_PORTFOLIO_DATA_URL}/sp500_dataset.csv.gz",
        kind="prices",
    ),
    RemoteDatasetSpec(
        name="ftse100",
        filename="ftse100_dataset.csv.gz",
        url=f"{SKFOLIO_DATASETS_URL}/ftse100_dataset.csv.gz",
        kind="prices",
    ),
)

RELATIVE_DATASETS: tuple[RemoteDatasetSpec, ...] = (
    RemoteDatasetSpec(
        name="djia_relatives",
        filename="djia_relatives.csv.gz",
        url=f"{UNIVERSAL_PORTFOLIO_DATA_URL}/djia_relatives.csv.gz",
        kind="relatives",
        has_datetime_index=False,
        synthetic_start="2001-01-14",
    ),
    RemoteDatasetSpec(
        name="msci_relatives",
        filename="msci_relatives.csv.gz",
        url=f"{UNIVERSAL_PORTFOLIO_DATA_URL}/msci_relatives.csv.gz",
        kind="relatives",
        has_datetime_index=False,
        synthetic_start="2006-04-01",
    ),
    RemoteDatasetSpec(
        name="nyse_o_relatives",
        filename="nyse_o_relatives.csv.gz",
        url=f"{UNIVERSAL_PORTFOLIO_DATA_URL}/nyse_o_relatives.csv.gz",
        kind="relatives",
        has_datetime_index=False,
    ),
    RemoteDatasetSpec(
        name="sp500_relatives",
        filename="sp500_relatives.csv.gz",
        url=f"{UNIVERSAL_PORTFOLIO_DATA_URL}/sp500_relatives.csv.gz",
        kind="relatives",
        has_datetime_index=False,
    ),
    RemoteDatasetSpec(
        name="tse_relatives",
        filename="tse_relatives.csv.gz",
        url=f"{UNIVERSAL_PORTFOLIO_DATA_URL}/tse_relatives.csv.gz",
        kind="relatives",
        has_datetime_index=False,
    ),
)

PRICE_DATASET_BY_NAME = {spec.name: spec for spec in PRICE_DATASETS}


def get_data_home(data_home: str | Path | None = None) -> Path:
    if data_home is None:
        data_home = os.environ.get("SKFOLIO_DATA", os.path.join("~", "skfolio_data"))
    path = Path(data_home).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_dataset_file(
    spec: RemoteDatasetSpec,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> Path:
    target_path = get_data_home(data_home) / spec.filename
    if target_path.exists():
        return target_path
    if not download_if_missing:
        raise OSError(f"Dataset not found locally: {target_path}")
    urllib.request.urlretrieve(spec.url, target_path)
    return target_path


def _read_dataset_frame(
    spec: RemoteDatasetSpec,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    path = _download_dataset_file(
        spec,
        data_home=data_home,
        download_if_missing=download_if_missing,
    )
    frame = pd.read_csv(path, index_col=0, compression="gzip")
    if spec.has_datetime_index:
        frame.index = pd.to_datetime(frame.index)
    if spec.synthetic_start is not None:
        frame["Date"] = pd.date_range(
            start=spec.synthetic_start,
            periods=len(frame),
            freq="B",
            name="Date",
        )
        frame = frame.set_index("Date")
    return frame


def _reverse_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[::-1].copy()


def _to_linear_returns(frame: pd.DataFrame, source_kind: str) -> pd.DataFrame:
    if source_kind == "prices":
        returns = prices_to_returns(frame)
    elif source_kind == "relatives":
        returns = frame - 1.0
    else:
        raise ValueError(f"Unknown dataset kind: {source_kind}")
    return returns


def _load_price_dataset(
    name: str,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    spec = PRICE_DATASET_BY_NAME[name]
    prices = _read_dataset_frame(
        spec,
        data_home=data_home,
        download_if_missing=download_if_missing,
    )
    return _to_linear_returns(prices, spec.kind)


def load_sp500_dataset(
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    return _load_price_dataset("sp500", data_home, download_if_missing)


def load_ftse100_dataset(
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    return _load_price_dataset("ftse100", data_home, download_if_missing)


def _load_relatives_dataset(
    spec: RemoteDatasetSpec,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    relatives = _read_dataset_frame(
        spec,
        data_home=data_home,
        download_if_missing=download_if_missing,
    )
    return _to_linear_returns(relatives, spec.kind)


def get_all_datasets(include_reversed: bool = False) -> list[DatasetCase]:
    """
    Load the dataset suite used by the research loop.

    The suite mixes:
    - price-based universes converted to net returns
    - price-relative universes converted to net returns
    - a reversed-time version of every case
    """
    cases = []

    for spec in PRICE_DATASETS:
        returns = _load_price_dataset(spec.name)
        cases.append(DatasetCase(spec.name, returns))
        if include_reversed:
            cases.append(DatasetCase(f"{spec.name}_reversed", _reverse_frame(returns)))

    for spec in RELATIVE_DATASETS:
        returns = _load_relatives_dataset(spec)
        cases.append(DatasetCase(spec.name, returns))
        if include_reversed:
            cases.append(DatasetCase(f"{spec.name}_reversed", _reverse_frame(returns)))

    return cases


def extract_path_sharpes(portfolios: object) -> np.ndarray:
    """
    Return one annualized Sharpe per out-of-sample path.

    `cross_val_predict` returns:
    - `MultiPeriodPortfolio` for single-path CV
    - `Population` for multi-path CV
    """
    if hasattr(portfolios, "annualized_sharpe_ratio"):
        return np.asarray([float(portfolios.annualized_sharpe_ratio)], dtype=float)
    return np.asarray(
        portfolios.measures(RatioMeasure.ANNUALIZED_SHARPE_RATIO), dtype=float
    )


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
