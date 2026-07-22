import pandas as pd


def count_records(df, column):
    """
    Count records for a selected categorical column.

    Missing or blank values are shown as 'Not assigned'.
    """

    values = df[column].copy()

    values = values.fillna("Not assigned")
    values = values.astype(str).str.strip()
    values = values.replace("", "Not assigned")

    counts = (
        values
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        column,
        "record_count",
    ]

    return counts


def parse_dates(date_column):
    """
    Convert a date column into pandas datetime format.

    The function supports:
    - normal date strings
    - Excel serial date values
    """

    # Try identifying numeric Excel dates
    numeric_dates = pd.to_numeric(
        date_column,
        errors="coerce",
    )

    # Parse non-numeric values as normal dates
    normal_dates = pd.to_datetime(
        date_column.where(numeric_dates.isna()),
        errors="coerce",
    )

    # Convert numeric Excel dates
    excel_dates = pd.to_datetime(
        numeric_dates,
        unit="D",
        origin="1899-12-30",
        errors="coerce",
    )

    # Use normal dates first, then Excel dates
    final_dates = normal_dates.fillna(
        excel_dates
    )

    return final_dates


def get_observation_date_range(df):
    """
    Summarize the temporal coverage of observation records.
    """

    observations = df[
        df["record_type"] == "observation"
    ].copy()

    observations["observation_date"] = parse_dates(
        observations["observation_date"]
    )

    date_summary = pd.DataFrame({
        "first_observation": [
            observations["observation_date"].min()
        ],
        "last_observation": [
            observations["observation_date"].max()
        ],
        "observations_with_dates": [
            observations["observation_date"].notna().sum()
        ],
        "observations_without_dates": [
            observations["observation_date"].isna().sum()
        ],
    })

    return date_summary


def get_indicator_coverage(df):
    """
    Show the historical coverage of every observation indicator.
    """

    observations = df[
        df["record_type"] == "observation"
    ].copy()

    observations["observation_date"] = parse_dates(
        observations["observation_date"]
    )

    observations["year"] = (
        observations["observation_date"]
        .dt.year
    )

    coverage = (
        observations
        .groupby(
            [
                "indicator_code",
                "indicator",
                "pillar",
            ],
            dropna=False,
        )
        .agg(
            record_count=(
                "record_id",
                "count",
            ),
            first_date=(
                "observation_date",
                "min",
            ),
            last_date=(
                "observation_date",
                "max",
            ),
            years_covered=(
                "year",
                lambda values: ", ".join(
                    str(year)
                    for year in sorted(
                        values.dropna()
                        .astype(int)
                        .unique()
                    )
                ),
            ),
            source_count=(
                "source_name",
                "nunique",
            ),
            source_types=(
                "source_type",
                lambda values: ", ".join(
                    sorted(
                        values.dropna()
                        .astype(str)
                        .unique()
                    )
                ),
            ),
        )
        .reset_index()
    )

    coverage = coverage.sort_values(
        [
            "pillar",
            "indicator_code",
        ]
    ).reset_index(drop=True)

    return coverage


def get_event_catalog(df):
    """
    Return all catalogued events and their dates.
    """

    events = df[
        df["record_type"] == "event"
    ].copy()

    events["observation_date"] = parse_dates(
        events["observation_date"]
    )

    selected_columns = [
        "record_id",
        "observation_date",
        "category",
        "indicator",
        "value_text",
        "source_type",
        "confidence",
    ]

    events = events[
        selected_columns
    ].sort_values(
        "observation_date"
    ).reset_index(drop=True)

    return events


def get_impact_link_details(df):
    """
    Connect impact-link records to their parent events.
    """

    impact_links = df[
        df["record_type"] == "impact_link"
    ].copy()

    events = df[
        df["record_type"] == "event"
    ][
        [
            "record_id",
            "indicator",
            "observation_date",
            "category",
        ]
    ].copy()

    events["observation_date"] = parse_dates(
        events["observation_date"]
    )

    events = events.rename(
        columns={
            "record_id": "event_id",
            "indicator": "event_name",
            "observation_date": "event_date",
            "category": "event_category",
        }
    )

    impact_details = impact_links.merge(
        events,
        left_on="parent_id",
        right_on="event_id",
        how="left",
    )

    selected_columns = [
        "record_id",
        "parent_id",
        "event_name",
        "event_date",
        "event_category",
        "pillar",
        "related_indicator",
        "relationship_type",
        "impact_direction",
        "impact_magnitude",
        "impact_estimate",
        "lag_months",
        "evidence_basis",
        "comparable_country",
        "confidence",
    ]

    impact_details = impact_details[
        selected_columns
    ].sort_values(
        [
            "event_date",
            "record_id",
        ]
    ).reset_index(drop=True)

    return impact_details


def get_event_impact_summary(df):
    """
    Count the number of impact links connected to every event.

    Events without impact links are retained with a count of zero.
    """

    events = df[
        df["record_type"] == "event"
    ][
        [
            "record_id",
            "indicator",
            "category",
            "observation_date",
        ]
    ].copy()

    events["observation_date"] = parse_dates(
        events["observation_date"]
    )

    impact_links = df[
        df["record_type"] == "impact_link"
    ].copy()

    impact_counts = (
        impact_links
        .groupby("parent_id")
        .agg(
            impact_link_count=(
                "record_id",
                "count",
            ),
            affected_indicators=(
                "related_indicator",
                lambda values: ", ".join(
                    sorted(
                        values.dropna()
                        .astype(str)
                        .unique()
                    )
                ),
            ),
        )
        .reset_index()
    )

    summary = events.merge(
        impact_counts,
        left_on="record_id",
        right_on="parent_id",
        how="left",
    )

    summary["impact_link_count"] = (
        summary["impact_link_count"]
        .fillna(0)
        .astype(int)
    )

    summary["affected_indicators"] = (
        summary["affected_indicators"]
        .fillna("None")
    )

    summary = summary[
        [
            "record_id",
            "observation_date",
            "category",
            "indicator",
            "impact_link_count",
            "affected_indicators",
        ]
    ].sort_values(
        "observation_date"
    ).reset_index(drop=True)

    return summary