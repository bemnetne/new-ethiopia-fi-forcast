import pandas as pd
import matplotlib.pyplot as plt

from src.data_explorer import parse_dates


def summarize_category(df, column):
    """
    Count records and calculate percentages
    for a categorical column.
    """

    values = (
        df[column]
        .fillna("Not assigned")
        .astype(str)
        .str.strip()
        .replace("", "Not assigned")
    )

    summary = (
        values
        .value_counts()
        .reset_index()
    )

    summary.columns = [
        column,
        "record_count",
    ]

    summary["percentage"] = (
        summary["record_count"]
        / len(df)
        * 100
    ).round(1)

    return summary


def prepare_observations(df):
    """
    Select observation records and create
    a usable year column.
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


def create_temporal_coverage_table(
    observations,
):
    """
    Create an indicator-by-year coverage table.

    The cell value is the number of observations
    available for an indicator in a given year.
    """

    coverage = pd.crosstab(
        observations["indicator_code"],
        observations["year"],
    )

    first_year = int(
        observations["year"].min()
    )

    last_year = int(
        observations["year"].max()
    )

    all_years = list(
        range(
            first_year,
            last_year + 1,
        )
    )

    coverage = coverage.reindex(
        columns=all_years,
        fill_value=0,
    )

    return coverage


def plot_temporal_coverage(
    coverage_table,
    output_path=None,
):
    """
    Plot which indicators have observations
    in each year.
    """

    presence_table = (
        coverage_table > 0
    ).astype(int)

    figure_height = max(
        7,
        len(presence_table) * 0.35,
    )

    fig, ax = plt.subplots(
        figsize=(14, figure_height)
    )

    image = ax.imshow(
        presence_table.values,
        aspect="auto",
    )

    ax.set_xticks(
        range(
            len(presence_table.columns)
        )
    )

    ax.set_xticklabels(
        presence_table.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(
            len(presence_table.index)
        )
    )

    ax.set_yticklabels(
        presence_table.index
    )

    ax.set_title(
        "Temporal Coverage of Financial Inclusion Indicators"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Indicator code")

    # Show the number of records in available cells
    for row_number in range(
        len(coverage_table.index)
    ):
        for column_number in range(
            len(coverage_table.columns)
        ):
            value = coverage_table.iloc[
                row_number,
                column_number,
            ]

            if value > 0:
                ax.text(
                    column_number,
                    row_number,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    fig.colorbar(
        image,
        ax=ax,
        label="Data availability",
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


def create_indicator_coverage_summary(
    observations,
):
    """
    Summarize the number of records and years
    available for each observation indicator.
    """

    coverage_summary = (
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
            unique_years=(
                "year",
                "nunique",
            ),
            first_year=(
                "year",
                "min",
            ),
            last_year=(
                "year",
                "max",
            ),
            years_covered=(
                "year",
                lambda values: ", ".join(
                    str(year)
                    for year in sorted(
                        values
                        .dropna()
                        .astype(int)
                        .unique()
                    )
                ),
            ),
        )
        .reset_index()
    )

    def classify_coverage(
        year_count,
    ):
        if year_count == 1:
            return "Very sparse"

        if year_count == 2:
            return "Sparse"

        if year_count == 3:
            return "Limited"

        return "Relatively stronger"

    coverage_summary[
        "coverage_status"
    ] = (
        coverage_summary["unique_years"]
        .apply(classify_coverage)
    )

    coverage_summary = (
        coverage_summary
        .sort_values(
            [
                "unique_years",
                "record_count",
                "indicator_code",
            ]
        )
        .reset_index(drop=True)
    )

    return coverage_summary


def plot_confidence_distribution(
    confidence_summary,
    output_path=None,
):
    """
    Plot the number of records under each
    confidence level.
    """

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        confidence_summary["confidence"],
        confidence_summary["record_count"],
    )

    ax.set_title(
        "Distribution of Confidence Levels"
    )

    ax.set_xlabel("Confidence level")
    ax.set_ylabel("Number of records")

    for index, row in (
        confidence_summary
        .reset_index(drop=True)
        .iterrows()
    ):
        ax.text(
            index,
            row["record_count"],
            str(row["record_count"]),
            ha="center",
            va="bottom",
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