"""SHAP explainability utilities for financial inclusion forecasts."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ExplainabilityError(ValueError):
    """Raised when model explanation inputs are invalid."""


@dataclass(frozen=True)
class ExplainabilityResult:
    """Artifacts generated from a fitted explainable model."""

    model: Pipeline
    features: pd.DataFrame
    predictions: np.ndarray
    shap_values: shap.Explanation
    global_importance: pd.DataFrame
    local_explanations: pd.DataFrame
    diagnostics: pd.DataFrame


def validate_model_data(
    feature_dataframe: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Validate the feature matrix and target."""

    if feature_dataframe.empty:
        raise ExplainabilityError("The feature dataframe is empty.")

    if target.empty:
        raise ExplainabilityError("The target series is empty.")

    if len(feature_dataframe) != len(target):
        raise ExplainabilityError(
            "Features and target must contain the same number of rows."
        )

    if feature_dataframe.columns.duplicated().any():
        raise ExplainabilityError("Feature names must be unique.")

    if feature_dataframe.isna().any().any():
        raise ExplainabilityError("Feature values must not contain missing values.")

    if target.isna().any():
        raise ExplainabilityError("Target values must not contain missing values.")

    non_numeric_columns = [
        column
        for column in feature_dataframe.columns
        if not pd.api.types.is_numeric_dtype(feature_dataframe[column])
    ]

    if non_numeric_columns:
        raise ExplainabilityError(
            "All model features must be numeric. "
            f"Invalid columns: {non_numeric_columns}"
        )


def build_explainable_model(
    alpha: float = 1.0,
) -> Pipeline:
    """
    Build a regularized linear model.

    Ridge regression is used instead of unrestricted linear
    regression because the dataset is small and the features
    may be correlated.
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "regressor",
                Ridge(alpha=alpha),
            ),
        ]
    )


def calculate_global_importance(
    shap_values: shap.Explanation,
) -> pd.DataFrame:
    """Calculate mean absolute SHAP importance."""

    values = np.asarray(shap_values.values)

    mean_absolute_shap = np.abs(values).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": shap_values.feature_names,
            "mean_absolute_shap": mean_absolute_shap,
        }
    )

    return importance.sort_values(
        "mean_absolute_shap",
        ascending=False,
    ).reset_index(drop=True)


def create_local_explanations(
    shap_values: shap.Explanation,
    feature_dataframe: pd.DataFrame,
    predictions: np.ndarray,
    observation_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """Create row-level SHAP contribution records."""

    records: list[dict[str, object]] = []

    for row_position in range(len(feature_dataframe)):
        observation_id = (
            observation_ids.iloc[row_position]
            if observation_ids is not None
            else row_position
        )

        base_value = float(
            np.asarray(shap_values.base_values[row_position]).reshape(-1)[0]
        )

        for feature_position, feature_name in enumerate(feature_dataframe.columns):
            records.append(
                {
                    "observation_id": observation_id,
                    "prediction": float(predictions[row_position]),
                    "base_value": base_value,
                    "feature": feature_name,
                    "feature_value": float(
                        feature_dataframe.iloc[
                            row_position,
                            feature_position,
                        ]
                    ),
                    "shap_value": float(
                        shap_values.values[
                            row_position,
                            feature_position,
                        ]
                    ),
                }
            )

    return pd.DataFrame(records)


def detect_concerning_patterns(
    feature_dataframe: pd.DataFrame,
    shap_values: shap.Explanation,
) -> pd.DataFrame:
    """
    Detect simple warning patterns in model explanations.

    These checks are indicators for review, not proof of bias
    or model failure.
    """

    global_importance = calculate_global_importance(shap_values)

    total_importance = float(global_importance["mean_absolute_shap"].sum())

    if total_importance == 0:
        global_importance["importance_share"] = 0.0
    else:
        global_importance["importance_share"] = (
            global_importance["mean_absolute_shap"] / total_importance
        )

    rows: list[dict[str, object]] = []

    top_feature = global_importance.iloc[0]

    if float(top_feature["importance_share"]) > 0.70:
        rows.append(
            {
                "pattern": "Feature concentration",
                "severity": "warning",
                "feature": top_feature["feature"],
                "detail": (
                    "One feature contributes more than 70% "
                    "of total mean absolute SHAP importance. "
                    "The forecast may depend too strongly on "
                    "this feature."
                ),
            }
        )

    correlation_matrix = feature_dataframe.corr().abs()

    for first_position, first_feature in enumerate(correlation_matrix.columns):
        for second_position in range(
            first_position + 1,
            len(correlation_matrix.columns),
        ):
            second_feature = correlation_matrix.columns[second_position]

            correlation = float(
                correlation_matrix.loc[
                    first_feature,
                    second_feature,
                ]
            )

            if correlation >= 0.90:
                rows.append(
                    {
                        "pattern": ("Highly correlated features"),
                        "severity": "warning",
                        "feature": (f"{first_feature} / {second_feature}"),
                        "detail": (
                            "The features have an absolute "
                            f"correlation of {correlation:.2f}. "
                            "Their SHAP contributions may be "
                            "unstable or shared."
                        ),
                    }
                )

    if not rows:
        rows.append(
            {
                "pattern": ("No automatic warning triggered"),
                "severity": "information",
                "feature": "All features",
                "detail": (
                    "The automatic concentration and "
                    "correlation checks did not identify "
                    "a major warning. Manual review is "
                    "still required."
                ),
            }
        )

    return pd.DataFrame(rows)


def fit_and_explain_model(
    feature_dataframe: pd.DataFrame,
    target: pd.Series,
    observation_ids: pd.Series | None = None,
    alpha: float = 1.0,
) -> ExplainabilityResult:
    """Fit a Ridge model and generate SHAP explanations."""

    validate_model_data(
        feature_dataframe=feature_dataframe,
        target=target,
    )

    model = build_explainable_model(alpha=alpha)

    model.fit(
        feature_dataframe,
        target,
    )

    predictions = model.predict(feature_dataframe)

    transformed_features = model.named_steps["scaler"].transform(feature_dataframe)

    transformed_dataframe = pd.DataFrame(
        transformed_features,
        columns=feature_dataframe.columns,
        index=feature_dataframe.index,
    )

    regression_model = model.named_steps["regressor"]

    explainer = shap.LinearExplainer(
        regression_model,
        transformed_dataframe,
    )

    shap_values = explainer(transformed_dataframe)

    shap_values.feature_names = list(feature_dataframe.columns)

    shap_values.data = feature_dataframe.to_numpy()

    global_importance = calculate_global_importance(shap_values)

    local_explanations = create_local_explanations(
        shap_values=shap_values,
        feature_dataframe=feature_dataframe,
        predictions=predictions,
        observation_ids=observation_ids,
    )

    diagnostics = detect_concerning_patterns(
        feature_dataframe=feature_dataframe,
        shap_values=shap_values,
    )

    return ExplainabilityResult(
        model=model,
        features=feature_dataframe,
        predictions=predictions,
        shap_values=shap_values,
        global_importance=global_importance,
        local_explanations=local_explanations,
        diagnostics=diagnostics,
    )


def save_explainability_artifacts(
    result: ExplainabilityResult,
    output_directory: Path,
    model_filename: str = ("access_explainable_model.joblib"),
) -> None:
    """Save model and explanation tables."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        result.model,
        output_directory / model_filename,
    )

    result.global_importance.to_csv(
        output_directory / "access_shap_global.csv",
        index=False,
    )

    result.local_explanations.to_csv(
        output_directory / "access_shap_local.csv",
        index=False,
    )

    result.diagnostics.to_csv(
        output_directory / "access_shap_diagnostics.csv",
        index=False,
    )
