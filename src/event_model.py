import pandas as pd

from src.config import (
    DIRECTION_SCORES,
    MAGNITUDE_SCORES,
)
from src.data_utils import (
    DataValidationError,
    validate_required_columns,
)

REQUIRED_EVENT_COLUMNS = [
    "impact_direction",
    "impact_magnitude",
]


def calculate_effect_score(
    direction: str,
    magnitude: str,
) -> int:
    """Convert an event direction and magnitude to a score."""

    normalized_direction = (
        str(direction).strip().lower()
    )

    normalized_magnitude = (
        str(magnitude).strip().lower()
    )

    if normalized_direction not in DIRECTION_SCORES:
        raise DataValidationError(
            f"Unsupported impact direction: {direction}"
        )

    if normalized_magnitude not in MAGNITUDE_SCORES:
        raise DataValidationError(
            f"Unsupported impact magnitude: {magnitude}"
        )

    return (
        DIRECTION_SCORES[normalized_direction]
        * MAGNITUDE_SCORES[normalized_magnitude]
    )


def add_effect_scores(
    event_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add direction, magnitude and signed effect scores.

    The original DataFrame is not modified.
    """

    validate_required_columns(
        dataframe=event_dataframe,
        required_columns=REQUIRED_EVENT_COLUMNS,
    )

    result = event_dataframe.copy()

    result["direction_score"] = (
        result["impact_direction"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(DIRECTION_SCORES)
    )

    result["magnitude_score"] = (
        result["impact_magnitude"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(MAGNITUDE_SCORES)
    )

    if result[
        ["direction_score", "magnitude_score"]
    ].isna().any().any():
        raise DataValidationError(
            "Some event directions or magnitudes "
            "could not be converted to scores."
        )

    result["signed_effect_score"] = (
        result["direction_score"]
        * result["magnitude_score"]
    )

    return result