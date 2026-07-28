import pandas as pd
import pytest

from src.data_utils import DataValidationError
from src.forecast_utils import (
    calculate_target_gap,
    get_forecast_years,
    validate_forecast_bounds,
)


def test_target_gap_is_calculated_correctly(
) -> None:
    result = calculate_target_gap(
        projection=53.2,
        target_rate=60.0,
    )

    assert result == pytest.approx(6.8)


def test_target_gap_cannot_be_negative(
) -> None:
    result = calculate_target_gap(
        projection=65.0,
        target_rate=60.0,
    )

    assert result == 0.0


def test_forecast_years_are_2025_to_2027(
) -> None:
    assert get_forecast_years() == [
        2025,
        2026,
        2027,
    ]


def test_valid_forecast_passes_validation(
) -> None:
    dataframe = pd.DataFrame({
        "year": [2025, 2026, 2027],
        "pessimistic": [49.5, 50.0, 50.5],
        "base": [50.3, 51.7, 53.2],
        "optimistic": [50.5, 52.0, 53.5],
    })

    validate_forecast_bounds(dataframe)


def test_forecast_above_100_raises_error(
) -> None:
    dataframe = pd.DataFrame({
        "year": [2025],
        "pessimistic": [49.5],
        "base": [101.0],
        "optimistic": [50.5],
    })

    with pytest.raises(DataValidationError):
        validate_forecast_bounds(dataframe)