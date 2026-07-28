from dataclasses import dataclass
from pathlib import Path

# =========================================================
# PROJECT DIRECTORIES
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


# =========================================================
# RAW DATA FILES
# =========================================================

UNIFIED_EXCEL_PATH = (
    RAW_DATA_DIR / "ethiopia_fi_unified_data.xlsx"
)

REFERENCE_EXCEL_PATH = (
    RAW_DATA_DIR / "reference_codes.xlsx"
)

UNIFIED_CSV_PATH = (
    RAW_DATA_DIR / "ethiopia_fi_unified_data.csv"
)

REFERENCE_CSV_PATH = (
    RAW_DATA_DIR / "reference_codes.csv"
)

ENRICHMENT_CSV_PATH = (
    RAW_DATA_DIR / "enrichment_records.csv"
)


# =========================================================
# PROCESSED DATA FILES
# =========================================================

ENRICHED_DATA_PATH = (
    PROCESSED_DATA_DIR / "ethiopia_fi_enriched.csv"
)

REFINED_EVENT_IMPACTS_PATH = (
    PROCESSED_DATA_DIR / "refined_event_impacts.csv"
)

EVENT_ASSOCIATION_SUMMARY_PATH = (
    PROCESSED_DATA_DIR
    / "event_indicator_association_summary.csv"
)

ACCESS_FORECAST_PATH = (
    PROCESSED_DATA_DIR
    / "access_forecast_scenarios.csv"
)

USAGE_FORECAST_PATH = (
    PROCESSED_DATA_DIR
    / "usage_forecast_scenarios.csv"
)

FINAL_FORECAST_PATH = (
    PROCESSED_DATA_DIR
    / "final_financial_inclusion_forecasts.csv"
)

FORECAST_UNCERTAINTY_PATH = (
    PROCESSED_DATA_DIR
    / "forecast_uncertainty_summary.csv"
)


# =========================================================
# BUSINESS CONSTANTS
# =========================================================

FORECAST_START_YEAR = 2025
FORECAST_END_YEAR = 2027

FINANCIAL_INCLUSION_TARGET = 60.0
MIN_PERCENTAGE = 0.0
MAX_PERCENTAGE = 100.0

ACCESS_INDICATOR_CODE = "ACC_OWNERSHIP"
USAGE_INDICATOR_CODE = "USG_DIGITAL_PAYMENT_RATE"

EVENT_BUILDUP_MONTHS = 12


# =========================================================
# EVENT-SCORING CONSTANTS
# =========================================================

DIRECTION_SCORES: dict[str, int] = {
    "increase": 1,
    "decrease": -1,
    "no_change": 0,
}

MAGNITUDE_SCORES: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


# =========================================================
# CONFIGURATION DATACLASSES
# =========================================================

@dataclass(frozen=True)
class ForecastConfig:
    """Configuration used by the forecasting workflow."""

    start_year: int = FORECAST_START_YEAR
    end_year: int = FORECAST_END_YEAR
    target_rate: float = FINANCIAL_INCLUSION_TARGET
    minimum_rate: float = MIN_PERCENTAGE
    maximum_rate: float = MAX_PERCENTAGE
    access_indicator: str = ACCESS_INDICATOR_CODE
    usage_indicator: str = USAGE_INDICATOR_CODE


@dataclass(frozen=True)
class EventModelConfig:
    """Configuration used by the event-impact model."""

    buildup_months: int = EVENT_BUILDUP_MONTHS


@dataclass(frozen=True)
class ProjectConfig:
    """Central configuration for the project."""

    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    reports_dir: Path = REPORTS_DIR
    figures_dir: Path = FIGURES_DIR
    dashboard_dir: Path = DASHBOARD_DIR

    forecast: ForecastConfig = ForecastConfig()
    event_model: EventModelConfig = EventModelConfig()


CONFIG = ProjectConfig()