import pandas as pd

from src.data_explorer import parse_dates


def prepare_usage_data(df):
    """
    Select observation records and create a year column.
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

    observations["value_numeric"] = pd.to_numeric(
        observations["value_numeric"],
        errors="coerce",
    )

    return observations


def get_indicator_series(df, indicator_code):
    """
    Return annual values for one indicator.
    """

    observations = prepare_usage_data(df)

    data = observations[
        observations["indicator_code"]
        == indicator_code
    ].copy()

    columns = [
        "year",
        "indicator_code",
        "indicator",
        "value_numeric",
        "unit",
        "confidence",
    ]

    return (
        data[columns]
        .sort_values("year")
        .reset_index(drop=True)
    )


def get_registered_active_gap(df):
    """
    Compare registered and active M-Pesa users.
    """

    registered = get_indicator_series(
        df,
        "USG_MPESA_USERS",
    )[
        ["year", "value_numeric"]
    ].rename(
        columns={
            "value_numeric": "registered_users",
        }
    )

    active = get_indicator_series(
        df,
        "USG_MPESA_ACTIVE",
    )[
        ["year", "value_numeric"]
    ].rename(
        columns={
            "value_numeric": "active_users",
        }
    )

    gap = registered.merge(
        active,
        on="year",
        how="inner",
    )

    gap["inactive_users"] = (
        gap["registered_users"]
        - gap["active_users"]
    )

    gap["calculated_active_rate"] = (
        gap["active_users"]
        / gap["registered_users"]
        * 100
    ).round(1)

    return gap


def get_transaction_counts(df):
    """
    Return comparable transaction-count indicators.
    """

    observations = prepare_usage_data(df)

    indicator_names = {
        "USG_P2P_COUNT": "P2P",
        "USG_ATM_COUNT": "ATM",
        "USG_POS_COUNT": "Merchant/POS",
    }

    data = observations[
        observations["indicator_code"]
        .isin(indicator_names)
    ].copy()

    data["payment_channel"] = (
        data["indicator_code"]
        .map(indicator_names)
    )

    data["transactions_millions"] = (
        data["value_numeric"]
        / 1_000_000
    )

    return data[
        [
            "year",
            "payment_channel",
            "value_numeric",
            "transactions_millions",
            "confidence",
        ]
    ].sort_values(
        [
            "year",
            "payment_channel",
        ]
    ).reset_index(drop=True)