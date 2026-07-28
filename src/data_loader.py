from pathlib import Path

import pandas as pd

from src.config import (
    ACCESS_FORECAST_PATH,
    ACCESS_SHAP_DIAGNOSTICS_PATH,
    ACCESS_SHAP_GLOBAL_PATH,
    ACCESS_SHAP_LOCAL_PATH,
    ENRICHED_DATA_PATH,
    EVENT_ASSOCIATION_SUMMARY_PATH,
    FINAL_FORECAST_PATH,
    FORECAST_UNCERTAINTY_PATH,
    REFERENCE_CSV_PATH,
    REFERENCE_EXCEL_PATH,
    REFINED_EVENT_IMPACTS_PATH,
    UNIFIED_CSV_PATH,
    UNIFIED_EXCEL_PATH,
    USAGE_FORECAST_PATH,
)
from src.data_utils import (
    DataFileNotFoundError,
    DataValidationError,
    load_csv,
)


def clean_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy with standardized column names."""

    result = dataframe.copy()

    result.columns = (
        result.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return result


def load_and_clean_csv(
    file_path: Path,
    required: bool = True,
) -> pd.DataFrame:
    """Load a CSV and standardize its column names."""

    dataframe = load_csv(
        file_path=file_path,
        required=required,
    )

    if dataframe.empty:
        return dataframe

    return clean_column_names(dataframe)


def convert_excel_files_to_csv() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the project starter Excel files to CSV."""

    if not UNIFIED_EXCEL_PATH.exists():
        raise DataFileNotFoundError(f"File not found: {UNIFIED_EXCEL_PATH}")

    if not REFERENCE_EXCEL_PATH.exists():
        raise DataFileNotFoundError(f"File not found: {REFERENCE_EXCEL_PATH}")

    workbook: dict[str, pd.DataFrame] = pd.read_excel(
        UNIFIED_EXCEL_PATH,
        sheet_name=None,
    )

    dataframes: list[pd.DataFrame] = []

    for sheet_name, dataframe in workbook.items():
        cleaned_dataframe = clean_column_names(dataframe)

        if "record_type" in cleaned_dataframe.columns:
            dataframes.append(cleaned_dataframe)

        print(f"{sheet_name}: {cleaned_dataframe.shape}")

    if not dataframes:
        raise DataValidationError("No workbook sheet contains record_type.")

    unified_dataframe = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False,
    ).dropna(how="all")

    reference_dataframe = clean_column_names(pd.read_excel(REFERENCE_EXCEL_PATH))

    UNIFIED_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unified_dataframe.to_csv(
        UNIFIED_CSV_PATH,
        index=False,
    )

    reference_dataframe.to_csv(
        REFERENCE_CSV_PATH,
        index=False,
    )

    return unified_dataframe, reference_dataframe


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the unified and reference datasets."""

    unified_dataframe = load_and_clean_csv(
        UNIFIED_CSV_PATH,
        required=True,
    )

    reference_dataframe = load_and_clean_csv(
        REFERENCE_CSV_PATH,
        required=True,
    )

    return unified_dataframe, reference_dataframe


def load_enriched_data(
    required: bool = True,
) -> pd.DataFrame:
    """Load the enriched financial inclusion dataset."""

    return load_and_clean_csv(
        ENRICHED_DATA_PATH,
        required=required,
    )


def load_event_impacts(
    required: bool = False,
) -> pd.DataFrame:
    """Load the refined event-impact results."""

    return load_and_clean_csv(
        REFINED_EVENT_IMPACTS_PATH,
        required=required,
    )


def load_event_association_summary(
    required: bool = False,
) -> pd.DataFrame:
    """Load the event-indicator association summary."""

    return load_and_clean_csv(
        EVENT_ASSOCIATION_SUMMARY_PATH,
        required=required,
    )


def load_access_forecasts(
    required: bool = False,
) -> pd.DataFrame:
    """Load Account Ownership forecast scenarios."""

    return load_and_clean_csv(
        ACCESS_FORECAST_PATH,
        required=required,
    )


def load_usage_forecasts(
    required: bool = False,
) -> pd.DataFrame:
    """Load Digital Payment Usage forecast scenarios."""

    return load_and_clean_csv(
        USAGE_FORECAST_PATH,
        required=required,
    )


def load_final_forecasts(
    required: bool = False,
) -> pd.DataFrame:
    """Load the combined final forecast results."""

    return load_and_clean_csv(
        FINAL_FORECAST_PATH,
        required=required,
    )


def load_forecast_uncertainty(
    required: bool = False,
) -> pd.DataFrame:
    """Load the forecast uncertainty summary."""

    return load_and_clean_csv(
        FORECAST_UNCERTAINTY_PATH,
        required=required,
    )


def load_dashboard_data() -> dict[str, pd.DataFrame]:
    """Load every dataset used by the dashboard."""

    return {
        "access_shap_global": load_csv(ACCESS_SHAP_GLOBAL_PATH),
        "access_shap_local": load_csv(ACCESS_SHAP_LOCAL_PATH),
        "access_shap_diagnostics": load_csv(ACCESS_SHAP_DIAGNOSTICS_PATH),
        "enriched": load_enriched_data(required=False),
        "event_impacts": load_event_impacts(),
        "association_summary": (load_event_association_summary()),
        "access_forecasts": (load_access_forecasts()),
        "usage_forecasts": (load_usage_forecasts()),
        "final_forecasts": (load_final_forecasts()),
        "uncertainty": (load_forecast_uncertainty()),
    }
