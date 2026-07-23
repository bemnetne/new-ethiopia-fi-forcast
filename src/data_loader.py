import pandas as pd

from src.config import (
    UNIFIED_EXCEL_PATH,
    REFERENCE_EXCEL_PATH,
    UNIFIED_CSV_PATH,
    REFERENCE_CSV_PATH,
    ENRICHED_DATA_PATH,
)


def clean_column_names(df):
    """
    Clean dataframe column names.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


def convert_excel_files_to_csv():
    """
    Read the Excel starter files and save them as CSV files.
    """

    # Check that the Excel files exist
    if not UNIFIED_EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {UNIFIED_EXCEL_PATH}"
        )

    if not REFERENCE_EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {REFERENCE_EXCEL_PATH}"
        )

    # Read all sheets from the unified workbook
    workbook = pd.read_excel(
        UNIFIED_EXCEL_PATH,
        sheet_name=None,
    )

    print("Sheets found in unified workbook:")

    dataframes = []

    for sheet_name, df in workbook.items():
        print(f"- {sheet_name}: {df.shape}")

        df = clean_column_names(df)

        # Include only sheets that contain record_type
        if "record_type" in df.columns:
            dataframes.append(df)

    # Combine the data and impact-link sheets
    unified_df = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False,
    )

    # Remove completely empty rows
    unified_df = unified_df.dropna(
        how="all"
    )

    # Read the reference-code workbook
    reference_df = pd.read_excel(
        REFERENCE_EXCEL_PATH
    )

    reference_df = clean_column_names(
        reference_df
    )

    # Save both files as CSV
    unified_df.to_csv(
        UNIFIED_CSV_PATH,
        index=False,
    )

    reference_df.to_csv(
        REFERENCE_CSV_PATH,
        index=False,
    )

    print("\nFiles saved successfully:")
    print(UNIFIED_CSV_PATH)
    print(REFERENCE_CSV_PATH)

    return unified_df, reference_df


def load_data():
    """
    Load the CSV files used for Task 1.
    """

    if not UNIFIED_CSV_PATH.exists():
        raise FileNotFoundError(
            "Unified CSV file is missing. "
            "Run convert_excel_files_to_csv() first."
        )

    if not REFERENCE_CSV_PATH.exists():
        raise FileNotFoundError(
            "Reference-code CSV file is missing. "
            "Run convert_excel_files_to_csv() first."
        )

    unified_df = pd.read_csv(
        UNIFIED_CSV_PATH
    )

    reference_df = pd.read_csv(
        REFERENCE_CSV_PATH
    )

    unified_df = clean_column_names(
        unified_df
    )

    reference_df = clean_column_names(
        reference_df
    )

    return unified_df, reference_df

def load_enriched_data():
    """
    Load the processed dataset created during
    data preparation and enrichment.
    """

    if not ENRICHED_DATA_PATH.exists():
        raise FileNotFoundError(
            "The enriched dataset was not found. "
            "Complete the data-enrichment stage first."
        )

    enriched_df = pd.read_csv(
        ENRICHED_DATA_PATH
    )

    enriched_df = clean_column_names(
        enriched_df
    )

    return enriched_df