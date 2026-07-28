import pandas as pd
import pytest

from src.explainability import (
    ExplainabilityError,
    calculate_global_importance,
    fit_and_explain_model,
    validate_model_data,
)


@pytest.fixture
def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trend_years": [
                0.0,
                3.0,
                6.0,
                10.0,
                13.0,
            ],
            "event_effect_score": [
                0.0,
                0.0,
                0.0,
                2.0,
                6.0,
            ],
            "mobile_money_rate": [
                0.0,
                0.0,
                0.0,
                4.7,
                9.45,
            ],
        }
    )


@pytest.fixture
def sample_target() -> pd.Series:
    return pd.Series(
        [
            14.0,
            22.0,
            35.0,
            46.0,
            49.0,
        ]
    )


def test_validate_model_data_accepts_valid_data(
    sample_features,
    sample_target,
):
    validate_model_data(
        sample_features,
        sample_target,
    )


def test_validate_model_data_rejects_missing_values(
    sample_features,
    sample_target,
):
    invalid_features = sample_features.copy()

    invalid_features.loc[
        0,
        "trend_years",
    ] = None

    with pytest.raises(ExplainabilityError):
        validate_model_data(
            invalid_features,
            sample_target,
        )


def test_fit_and_explain_returns_predictions(
    sample_features,
    sample_target,
):
    result = fit_and_explain_model(
        feature_dataframe=sample_features,
        target=sample_target,
    )

    assert len(result.predictions) == 5
    assert result.shap_values.values.shape == (
        5,
        3,
    )


def test_global_importance_contains_all_features(
    sample_features,
    sample_target,
):
    result = fit_and_explain_model(
        feature_dataframe=sample_features,
        target=sample_target,
    )

    importance = calculate_global_importance(result.shap_values)

    assert set(importance["feature"]) == set(sample_features.columns)


def test_local_explanations_created_for_each_row(
    sample_features,
    sample_target,
):
    result = fit_and_explain_model(
        feature_dataframe=sample_features,
        target=sample_target,
    )

    expected_rows = len(sample_features) * len(sample_features.columns)

    assert len(result.local_explanations) == expected_rows
