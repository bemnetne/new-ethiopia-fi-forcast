import pandas as pd
import pytest

from src.data_utils import DataValidationError
from src.event_model import (
    add_effect_scores,
    calculate_effect_score,
)


def test_high_increase_score_is_positive_three() -> None:
    result = calculate_effect_score(
        direction="increase",
        magnitude="high",
    )

    assert result == 3


def test_medium_decrease_score_is_negative_two() -> None:
    result = calculate_effect_score(
        direction="decrease",
        magnitude="medium",
    )

    assert result == -2


def test_invalid_direction_raises_error() -> None:
    with pytest.raises(DataValidationError):
        calculate_effect_score(
            direction="unknown",
            magnitude="high",
        )


def test_add_effect_scores_preserves_original_dataframe() -> None:
    source = pd.DataFrame(
        {
            "impact_direction": ["increase"],
            "impact_magnitude": ["medium"],
        }
    )

    result = add_effect_scores(source)

    assert "signed_effect_score" not in source.columns
    assert result["signed_effect_score"].iloc[0] == 2
