from pathlib import Path

import pandas as pd
import pytest

from src.data_utils import (
    DataFileNotFoundError,
    DataValidationError,
    load_csv,
    validate_required_columns,
)


def test_load_csv_returns_dataframe(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "sample.csv"

    pd.DataFrame(
        {"value": [1, 2]}
    ).to_csv(file_path, index=False)

    result = load_csv(file_path)

    assert result.shape == (2, 1)


def test_load_csv_raises_for_missing_required_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(DataFileNotFoundError):
        load_csv(
            missing_path,
            required=True,
        )


def test_optional_missing_file_returns_empty_dataframe(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.csv"

    result = load_csv(
        missing_path,
        required=False,
    )

    assert result.empty


def test_validate_required_columns_detects_missing_column(
) -> None:
    dataframe = pd.DataFrame(
        {"record_id": ["OBS_1"]}
    )

    with pytest.raises(DataValidationError):
        validate_required_columns(
            dataframe,
            ["record_id", "indicator_code"],
        )