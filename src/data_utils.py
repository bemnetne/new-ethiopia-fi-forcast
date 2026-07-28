from pathlib import Path

import pandas as pd


class DataFileNotFoundError(FileNotFoundError):
    """Raised when a required project data file is missing."""


class DataValidationError(ValueError):
    """Raised when a dataset fails validation."""


def load_csv(
    file_path: Path,
    required: bool = True,
) -> pd.DataFrame:
    """
    Load a CSV file.

    Parameters
    ----------
    file_path:
        Full path to the CSV file.
    required:
        When True, raise an error if the file is missing.
        When False, return an empty DataFrame.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    if not file_path.exists():
        if required:
            raise DataFileNotFoundError(
                f"Required file was not found: {file_path}"
            )

        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)

    except pd.errors.EmptyDataError as error:
        raise DataValidationError(
            f"The file is empty: {file_path}"
        ) from error

    except pd.errors.ParserError as error:
        raise DataValidationError(
            f"The file could not be parsed: {file_path}"
        ) from error


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """
    Confirm that a DataFrame contains required columns.

    Raises
    ------
    DataValidationError
        When one or more required columns are missing.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise DataValidationError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )


def validate_not_empty(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Confirm that a DataFrame contains at least one row."""

    if dataframe.empty:
        raise DataValidationError(
            f"{dataset_name} contains no records."
        )


def convert_to_datetime(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Convert a column to datetime without modifying the original
    DataFrame.
    """

    validate_required_columns(
        dataframe=dataframe,
        required_columns=[column],
    )

    result = dataframe.copy()

    result[column] = pd.to_datetime(
        result[column],
        errors="coerce",
    )

    return result


def convert_to_numeric(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Convert a column to numeric without modifying the original
    DataFrame.
    """

    validate_required_columns(
        dataframe=dataframe,
        required_columns=[column],
    )

    result = dataframe.copy()

    result[column] = pd.to_numeric(
        result[column],
        errors="coerce",
    )

    return result