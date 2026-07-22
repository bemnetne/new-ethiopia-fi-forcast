from pathlib import Path


# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main data folders
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Excel input files
UNIFIED_EXCEL_PATH = RAW_DATA_DIR / "ethiopia_fi_unified_data.xlsx"
REFERENCE_EXCEL_PATH = RAW_DATA_DIR / "reference_codes.xlsx"

# CSV output files
UNIFIED_CSV_PATH = RAW_DATA_DIR / "ethiopia_fi_unified_data.csv"
REFERENCE_CSV_PATH = RAW_DATA_DIR / "reference_codes.csv"

# Manually researched enrichment records
ENRICHMENT_CSV_PATH = (
    RAW_DATA_DIR / "enrichment_records.csv"
)

# Final Task 1 analysis-ready dataset
ENRICHED_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "ethiopia_fi_enriched.csv"
)