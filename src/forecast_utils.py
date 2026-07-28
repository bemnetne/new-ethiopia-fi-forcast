import pandas as pd

from src.config import CONFIG
from src.data_utils import (
    DataValidationError,
    validate_required_columns,
)

SCENARIO_COLUMNS = [
    "pessimistic",
    "base",
    "optimistic",
]


def validate_forecast_bounds(
    forecast_dataframe: pd.DataFrame,
    scenario_columns: list[str] | None = None,
) -> None:
    """Validate that forecast rates remain from 0 to 100."""

    columns = scenario_columns if scenario_columns is not None else SCENARIO_COLUMNS

    validate_required_columns(
        dataframe=forecast_dataframe,
        required_columns=["year", *columns],
    )

    for column in columns:
        values = pd.to_numeric(
            forecast_dataframe[column],
            errors="coerce",
        )

        if values.isna().any():
            raise DataValidationError(f"{column} contains non-numeric values.")

        invalid_values = ~values.between(
            CONFIG.forecast.minimum_rate,
            CONFIG.forecast.maximum_rate,
        )

        if invalid_values.any():
            raise DataValidationError(
                f"{column} contains values outside the valid percentage range."
            )


def calculate_target_gap(
    projection: float,
    target_rate: float | None = None,
) -> float:
    """Calculate the remaining gap to the inclusion target."""

    target = CONFIG.forecast.target_rate if target_rate is None else target_rate

    return max(target - projection, 0.0)


def get_forecast_years() -> list[int]:
    """Return the configured forecast period."""

    return list(
        range(
            CONFIG.forecast.start_year,
            CONFIG.forecast.end_year + 1,
        )
    )
