import pandas as pd
import matplotlib.pyplot as plt

from src.data_explorer import parse_dates


def prepare_access_observations(df):
    """
    Select observation records and prepare
    their dates and years.
    """

    observations = df[
        df["record_type"] == "observation"
    ].copy()

    observations["observation_date"] = (
        parse_dates(
            observations["observation_date"]
        )
    )

    observations["year"] = (
        observations["observation_date"]
        .dt.year
    )

    return observations


def get_account_ownership_trajectory(df):
    """
    Return national account ownership observations.

    Gender-disaggregated rows are excluded from
    the national trajectory.
    """

    observations = prepare_access_observations(
        df
    )

    gender = (
        observations["gender"]
        .fillna("all")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    location = (
        observations["location"]
        .fillna("national")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    ownership = observations[
        (
            observations["indicator_code"]
            == "ACC_OWNERSHIP"
        )
        & (gender == "all")
        & (location == "national")
    ].copy()

    ownership = ownership[
        [
            "record_id",
            "year",
            "value_numeric",
            "unit",
            "source_name",
            "confidence",
        ]
    ]

    ownership = ownership.rename(
        columns={
            "value_numeric": "ownership_rate",
        }
    )

    ownership = ownership.sort_values(
        "year"
    ).reset_index(drop=True)

    return ownership


def calculate_ownership_growth(
    ownership_df,
):
    """
    Calculate changes between survey years.
    """

    growth = ownership_df.copy()

    growth["previous_year"] = (
        growth["year"].shift(1)
    )

    growth["previous_rate"] = (
        growth["ownership_rate"].shift(1)
    )

    growth["years_between"] = (
        growth["year"]
        - growth["previous_year"]
    )

    growth["percentage_point_change"] = (
        growth["ownership_rate"]
        - growth["previous_rate"]
    )

    growth["annual_pp_change"] = (
        growth["percentage_point_change"]
        / growth["years_between"]
    )

    growth["relative_growth_percent"] = (
        (
            growth["ownership_rate"]
            / growth["previous_rate"]
            - 1
        )
        * 100
    )

    growth["period"] = (
        growth["previous_year"]
        .fillna(0)
        .astype(int)
        .astype(str)
        + "-"
        + growth["year"]
        .astype(int)
        .astype(str)
    )

    growth = growth.dropna(
        subset=["previous_rate"]
    ).reset_index(drop=True)

    growth[
        [
            "percentage_point_change",
            "annual_pp_change",
            "relative_growth_percent",
        ]
    ] = growth[
        [
            "percentage_point_change",
            "annual_pp_change",
            "relative_growth_percent",
        ]
    ].round(2)

    return growth


def plot_account_ownership(
    ownership_df,
    output_path=None,
):
    """
    Plot Ethiopia's national account ownership rate.
    """

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        ownership_df["year"],
        ownership_df["ownership_rate"],
        marker="o",
    )

    for _, row in ownership_df.iterrows():
        ax.text(
            row["year"],
            row["ownership_rate"] + 1,
            f'{row["ownership_rate"]:.0f}%',
            ha="center",
        )

    ax.set_title(
        "Ethiopia Account Ownership Rate, 2011–2024"
    )

    ax.set_xlabel("Survey year")
    ax.set_ylabel("Account ownership rate (%)")

    ax.set_xticks(
        ownership_df["year"]
    )

    ax.set_ylim(
        0,
        ownership_df["ownership_rate"].max()
        + 10,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig


def plot_ownership_growth(
    growth_df,
    output_path=None,
):
    """
    Plot percentage-point growth between surveys.
    """

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        growth_df["period"],
        growth_df["percentage_point_change"],
    )

    for bar, value in zip(
        bars,
        growth_df["percentage_point_change"],
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{value:+.0f} pp",
            ha="center",
        )

    ax.set_title(
        "Change in Account Ownership Between Survey Years"
    )

    ax.set_xlabel("Survey period")
    ax.set_ylabel("Percentage-point change")

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig


def get_gender_ownership(df):
    """
    Return male and female account ownership
    observations when available.
    """

    observations = prepare_access_observations(
        df
    )

    observations["gender_clean"] = (
        observations["gender"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    gender_data = observations[
        (
            observations["indicator_code"]
            == "ACC_OWNERSHIP"
        )
        & (
            observations["gender_clean"]
            .isin(["male", "female"])
        )
    ].copy()

    gender_data = gender_data[
        [
            "year",
            "gender_clean",
            "value_numeric",
            "unit",
            "confidence",
        ]
    ]

    gender_data = gender_data.rename(
        columns={
            "gender_clean": "gender",
            "value_numeric": "ownership_rate",
        }
    )

    return gender_data.sort_values(
        [
            "year",
            "gender",
        ]
    ).reset_index(drop=True)


def plot_gender_ownership(
    gender_df,
    output_path=None,
):
    """
    Plot male and female account ownership.
    """

    pivot = gender_df.pivot_table(
        index="year",
        columns="gender",
        values="ownership_rate",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(8, 5),
    )

    ax.set_title(
        "Account Ownership by Gender"
    )

    ax.set_xlabel("Survey year")
    ax.set_ylabel("Account ownership rate (%)")

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    ax.legend(
        title="Gender"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    fig = ax.get_figure()

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig


def get_gender_gap_trend(df):
    """
    Return the reported account ownership
    gender-gap indicator.
    """

    observations = prepare_access_observations(
        df
    )

    gender_gap = observations[
        observations["indicator_code"]
        == "GEN_GAP_ACC"
    ].copy()

    gender_gap = gender_gap[
        [
            "year",
            "value_numeric",
            "unit",
            "source_name",
            "confidence",
        ]
    ]

    gender_gap = gender_gap.rename(
        columns={
            "value_numeric": "gender_gap",
        }
    )

    return gender_gap.sort_values(
        "year"
    ).reset_index(drop=True)


def plot_gender_gap(
    gender_gap_df,
    output_path=None,
):
    """
    Plot the account ownership gender gap.
    """

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        gender_gap_df["year"],
        gender_gap_df["gender_gap"],
        marker="o",
    )

    for _, row in gender_gap_df.iterrows():
        ax.text(
            row["year"],
            row["gender_gap"] + 0.5,
            f'{row["gender_gap"]:.1f} pp',
            ha="center",
        )

    ax.set_title(
        "Account Ownership Gender Gap"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Gender gap, percentage points")

    ax.set_xticks(
        gender_gap_df["year"]
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig


def get_urban_rural_ownership(df):
    """
    Return urban and rural account ownership
    observations when available.
    """

    observations = prepare_access_observations(
        df
    )

    observations["location_clean"] = (
        observations["location"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    location_data = observations[
        (
            observations["indicator_code"]
            == "ACC_OWNERSHIP"
        )
        & (
            observations["location_clean"]
            .isin(["urban", "rural"])
        )
    ].copy()

    location_data = location_data[
        [
            "year",
            "location_clean",
            "value_numeric",
            "unit",
            "confidence",
        ]
    ]

    location_data = location_data.rename(
        columns={
            "location_clean": "location",
            "value_numeric": "ownership_rate",
        }
    )

    return location_data.sort_values(
        [
            "year",
            "location",
        ]
    ).reset_index(drop=True)


def plot_urban_rural_ownership(
    location_df,
    output_path=None,
):
    """
    Plot urban and rural account ownership.
    """

    pivot = location_df.pivot_table(
        index="year",
        columns="location",
        values="ownership_rate",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(8, 5),
    )

    ax.set_title(
        "Account Ownership by Location"
    )

    ax.set_xlabel("Survey year")
    ax.set_ylabel("Account ownership rate (%)")

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    ax.legend(
        title="Location"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    fig = ax.get_figure()

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig


def get_slowdown_context(df):
    """
    Return recent indicators that may help explain
    the 2021–2024 account ownership slowdown.
    """

    observations = prepare_access_observations(
        df
    )

    indicator_roles = {
        "ACC_MM_ACCOUNT":
            "Direct mobile money access",

        "USG_TELEBIRR_USERS":
            "Registered wallet scale",

        "USG_MPESA_USERS":
            "Registered wallet scale",

        "USG_ACTIVE_RATE":
            "Registered-to-active conversion",

        "USG_SECTOR_ACTIVE_ACCOUNT_RATE":
            "Registered-to-active conversion",

        "ACC_4G_COV":
            "Digital infrastructure",

        "ACC_FAYDA":
            "Identification and KYC",

        "GEN_GAP_ACC":
            "Unequal access",

        "AFF_DATA_INCOME":
            "Affordability constraint",

        "ACC_TELEBIRR_AGENTS":
            "Physical service reach",

        "USG_TELEBIRR_MERCHANTS":
            "Merchant acceptance",

        "USG_DIGITAL_PAYMENT_RATE":
            "Actual digital usage",
    }

    selected_codes = list(
        indicator_roles.keys()
    )

    context = observations[
        observations["indicator_code"]
        .isin(selected_codes)
    ].copy()

    context = context.sort_values(
        [
            "indicator_code",
            "observation_date",
        ]
    )

    latest_context = (
        context
        .groupby(
            "indicator_code",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    latest_context["possible_role"] = (
        latest_context["indicator_code"]
        .map(indicator_roles)
    )

    latest_context = latest_context[
        [
            "indicator_code",
            "indicator",
            "possible_role",
            "year",
            "value_numeric",
            "unit",
            "confidence",
        ]
    ]

    return latest_context.sort_values(
        [
            "possible_role",
            "indicator_code",
        ]
    ).reset_index(drop=True)


def get_access_event_timeline(
    df,
    start_year=2021,
    end_year=2024,
):
    """
    Return events that occurred during the
    slowdown period.
    """

    events = df[
        df["record_type"] == "event"
    ].copy()

    events["observation_date"] = (
        parse_dates(
            events["observation_date"]
        )
    )

    events["year"] = (
        events["observation_date"]
        .dt.year
    )

    events = events[
        events["year"].between(
            start_year,
            end_year,
        )
    ]

    return events[
        [
            "record_id",
            "year",
            "observation_date",
            "category",
            "indicator",
            "confidence",
        ]
    ].sort_values(
        "observation_date"
    ).reset_index(drop=True)