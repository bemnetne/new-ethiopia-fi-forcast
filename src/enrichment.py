import pandas as pd

from src.config import (
    ENRICHMENT_CSV_PATH,
    ENRICHED_DATA_PATH,
)
REQUIRED_FIELDS = {
    "observation": [
        "record_id",
        "record_type",
        "pillar",
        "indicator",
        "indicator_code",
        "value_numeric",
        "observation_date",
        "source_name",
        "source_url",
        "confidence",
    ],

    "event": [
        "record_id",
        "record_type",
        "category",
        "indicator",
        "observation_date",
    ],

    "impact_link": [
        "record_id",
        "record_type",
        "parent_id",
        "pillar",
        "related_indicator",
        "impact_direction",
        "impact_magnitude",
        "lag_months",
        "evidence_basis",
    ],
}

def build_record(
    columns,
    record_id,
    record_type,
    **values,
):
    """
    Create one record using the unified dataset schema.

    Columns that are not relevant to the record are left empty.
    """

    record = {
        column: None
        for column in columns
    }

    record["record_id"] = record_id
    record["record_type"] = record_type

    record.update(values)

    return record


def save_enrichment_records(records):
    """
    Save manually researched enrichment records as CSV.
    """

    enrichment_df = pd.DataFrame(records)

    ENRICHMENT_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enrichment_df.to_csv(
        ENRICHMENT_CSV_PATH,
        index=False,
    )

    return enrichment_df


def validate_enrichment(
    existing_df,
    enrichment_df,
):
    """
    Perform basic validation before combining the datasets.
    """

    checks = []

    def add_check(check_name, issue_count):
        checks.append({
            "check": check_name,
            "issues_found": int(issue_count),
            "status": (
                "Passed"
                if issue_count == 0
                else "Review required"
            ),
        })

    # Check 1: duplicated IDs inside the new data
    duplicate_new_ids = (
        enrichment_df["record_id"]
        .duplicated()
        .sum()
    )

    add_check(
        "Duplicate IDs inside enrichment data",
        duplicate_new_ids,
    )

    # Check 2: new IDs must not already exist
    overlapping_ids = (
        enrichment_df["record_id"]
        .isin(existing_df["record_id"])
        .sum()
    )

    add_check(
        "Enrichment IDs already in starter data",
        overlapping_ids,
    )

    # Check 3: only supported record types
    valid_record_types = {
        "observation",
        "event",
        "impact_link",
        "target",
    }

    invalid_record_types = (
        ~enrichment_df["record_type"]
        .isin(valid_record_types)
    ).sum()

    add_check(
        "Invalid record types",
        invalid_record_types,
    )

    # Check 4: events should not have pillars
    event_pillars = (
        enrichment_df.loc[
            enrichment_df["record_type"]
            == "event",
            "pillar",
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    events_with_pillars = (
        event_pillars != ""
    ).sum()

    add_check(
        "Events incorrectly assigned to pillars",
        events_with_pillars,
    )

    # Combine temporarily for relationship checks
    combined_df = pd.concat(
        [
            existing_df,
            enrichment_df,
        ],
        ignore_index=True,
        sort=False,
    )

    new_impact_links = enrichment_df[
        enrichment_df["record_type"]
        == "impact_link"
    ].copy()

    # Check 5: parent_id must point to an event
    event_ids = set(
        combined_df.loc[
            combined_df["record_type"]
            == "event",
            "record_id",
        ].dropna()
    )

    unmatched_parent_ids = (
        ~new_impact_links["parent_id"]
        .isin(event_ids)
    ).sum()

    add_check(
        "Impact links without matching events",
        unmatched_parent_ids,
    )

    # Check 6: related indicators must exist
    indicator_codes = set(
        combined_df.loc[
            combined_df["record_type"].isin(
                [
                    "observation",
                    "target",
                ]
            ),
            "indicator_code",
        ]
        .dropna()
        .astype(str)
    )

    unmatched_indicators = (
        ~new_impact_links[
            "related_indicator"
        ]
        .isin(indicator_codes)
    ).sum()

    add_check(
        "Impact links with unknown indicators",
        unmatched_indicators,
    )

    return pd.DataFrame(checks)


def append_enrichment(
    existing_df,
    enrichment_df,
):
    """
    Append the new records and save the processed dataset.
    """

    # Keep exactly the same schema and column order
    enrichment_df = enrichment_df.reindex(
        columns=existing_df.columns
    )

    enriched_df = pd.concat(
        [
            existing_df,
            enrichment_df,
        ],
        ignore_index=True,
    )

    ENRICHED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched_df.to_csv(
        ENRICHED_DATA_PATH,
        index=False,
    )

    return enriched_df

def is_missing(series):
    """
    Identify values that are empty or missing.
    """

    return (
        series.isna()
        | series.astype(str).str.strip().eq("")
    )

def validate_required_fields(enrichment_df):
    """
    Check that required fields are populated
    for each enrichment record type.
    """

    results = []

    for record_type, required_fields in REQUIRED_FIELDS.items():

        records = enrichment_df[
            enrichment_df["record_type"]
            == record_type
        ]

        for field in required_fields:

            if field not in enrichment_df.columns:
                missing_count = len(records)

            else:
                missing_count = is_missing(
                    records[field]
                ).sum()

            status = (
                "Passed"
                if missing_count == 0
                else "Review required"
            )

            results.append({
                "record_type": record_type,
                "field": field,
                "missing_count": missing_count,
                "status": status,
            })

    return pd.DataFrame(results)

def validate_record_type_rules(enrichment_df):
    """
    Check the special schema rules for each record type.
    """

    results = []

    # Observations should not have a category
    observations = enrichment_df[
        enrichment_df["record_type"]
        == "observation"
    ]

    observations_with_category = (
        ~is_missing(observations["category"])
    ).sum()

    results.append({
        "check": "Observations with category assigned",
        "issues_found": observations_with_category,
        "status": (
            "Passed"
            if observations_with_category == 0
            else "Review required"
        ),
    })

    # Events should not have a pillar
    events = enrichment_df[
        enrichment_df["record_type"]
        == "event"
    ]

    events_with_pillar = (
        ~is_missing(events["pillar"])
    ).sum()

    results.append({
        "check": "Events with pillar assigned",
        "issues_found": events_with_pillar,
        "status": (
            "Passed"
            if events_with_pillar == 0
            else "Review required"
        ),
    })

    # Impact links should not have a category
    impact_links = enrichment_df[
        enrichment_df["record_type"]
        == "impact_link"
    ]

    impact_links_with_category = (
        ~is_missing(impact_links["category"])
    ).sum()

    results.append({
        "check": "Impact links with category assigned",
        "issues_found": impact_links_with_category,
        "status": (
            "Passed"
            if impact_links_with_category == 0
            else "Review required"
        ),
    })

    return pd.DataFrame(results)

def validate_impact_links(
    existing_df,
    enrichment_df,
):
    """
    Validate event and indicator references
    used by new impact links.
    """

    combined_df = pd.concat(
        [
            existing_df,
            enrichment_df,
        ],
        ignore_index=True,
        sort=False,
    )

    impact_links = enrichment_df[
        enrichment_df["record_type"]
        == "impact_link"
    ].copy()

    results = []

    # Valid parent IDs are IDs of event records
    valid_event_ids = set(
        combined_df.loc[
            combined_df["record_type"]
            == "event",
            "record_id",
        ].dropna()
    )

    invalid_parent_ids = (
        ~impact_links["parent_id"]
        .isin(valid_event_ids)
    ).sum()

    results.append({
        "check": "Impact links with invalid parent_id",
        "issues_found": invalid_parent_ids,
        "status": (
            "Passed"
            if invalid_parent_ids == 0
            else "Review required"
        ),
    })

    # Valid related indicators come from
    # observations or targets
    valid_indicator_codes = set(
        combined_df.loc[
            combined_df["record_type"].isin(
                [
                    "observation",
                    "target",
                ]
            ),
            "indicator_code",
        ].dropna()
    )

    invalid_related_indicators = (
        ~impact_links["related_indicator"]
        .isin(valid_indicator_codes)
    ).sum()

    results.append({
        "check": "Impact links with invalid related_indicator",
        "issues_found": invalid_related_indicators,
        "status": (
            "Passed"
            if invalid_related_indicators == 0
            else "Review required"
        ),
    })

    return pd.DataFrame(results)

def validate_impact_pillars(
    existing_df,
    enrichment_df,
):
    """
    Check that each impact-link pillar matches
    the pillar of its related indicator.
    """

    combined_df = pd.concat(
        [
            existing_df,
            enrichment_df,
        ],
        ignore_index=True,
        sort=False,
    )

    # Build indicator-to-pillar lookup
    indicator_lookup = (
        combined_df[
            combined_df["record_type"].isin(
                [
                    "observation",
                    "target",
                ]
            )
        ][
            [
                "indicator_code",
                "pillar",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset="indicator_code"
        )
    )

    indicator_lookup = dict(
        zip(
            indicator_lookup["indicator_code"],
            indicator_lookup["pillar"],
        )
    )

    impact_links = enrichment_df[
        enrichment_df["record_type"]
        == "impact_link"
    ].copy()

    impact_links["expected_pillar"] = (
        impact_links["related_indicator"]
        .map(indicator_lookup)
    )

    impact_links["pillar_matches"] = (
        impact_links["pillar"]
        == impact_links["expected_pillar"]
    )

    return impact_links[
        [
            "record_id",
            "parent_id",
            "related_indicator",
            "pillar",
            "expected_pillar",
            "pillar_matches",
        ]
    ]