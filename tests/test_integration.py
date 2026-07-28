from pathlib import Path

import pandas as pd

from src.data_loader import load_and_clean_csv
from src.event_model import add_effect_scores
from src.forecast_utils import validate_forecast_bounds


def test_event_and_forecast_workflow(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.csv"

    event_data = pd.DataFrame({
        "Impact Direction": [
            "increase",
            "decrease",
        ],
        "Impact Magnitude": [
            "high",
            "low",
        ],
    })

    event_data.to_csv(
        event_path,
        index=False,
    )

    loaded_events = load_and_clean_csv(
        event_path
    )

    scored_events = add_effect_scores(
        loaded_events
    )

    assert scored_events[
        "signed_effect_score"
    ].tolist() == [3, -1]

    forecast_data = pd.DataFrame({
        "year": [2025, 2026, 2027],
        "pessimistic": [49.5, 50.0, 50.5],
        "base": [50.3, 51.7, 53.2],
        "optimistic": [50.5, 52.0, 53.5],
    })

    validate_forecast_bounds(
        forecast_data
    )