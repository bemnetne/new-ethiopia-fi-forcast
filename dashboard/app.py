import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# PROJECT PATHS
# =========================================================

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import CONFIG
from src.data_loader import load_dashboard_data
from src.forecast_utils import calculate_target_gap

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Ethiopia Financial Inclusion Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DATA LOADING
# =========================================================


@st.cache_data
def get_dashboard_data() -> dict[str, pd.DataFrame]:
    """Load and cache dashboard data."""

    return load_dashboard_data()


data = get_dashboard_data()

# =========================================================
# DATA PREPARATION HELPERS
# =========================================================


def prepare_observations(
    enriched_df: pd.DataFrame,
) -> pd.DataFrame:
    if enriched_df.empty:
        return pd.DataFrame()

    observations = enriched_df.copy()

    if "record_type" in observations.columns:
        observations = observations[
            observations["record_type"].astype(str).str.lower().eq("observation")
        ].copy()

    if "observation_date" in observations.columns:
        observations["observation_date"] = pd.to_datetime(
            observations["observation_date"],
            errors="coerce",
        )

        observations["year"] = observations["observation_date"].dt.year

    elif "year" in observations.columns:
        observations["year"] = pd.to_numeric(
            observations["year"],
            errors="coerce",
        )

    else:
        observations["year"] = np.nan

    if "value_numeric" in observations.columns:
        observations["value_numeric"] = pd.to_numeric(
            observations["value_numeric"],
            errors="coerce",
        )

    return observations


observations_df = prepare_observations(data["enriched"])


def get_latest_indicator_value(
    indicator_code: str,
) -> tuple[float | None, int | None]:
    """
    Return the latest available value and year for an indicator.
    """

    indicator_data = get_indicator_series(indicator_code)

    if indicator_data.empty:
        return None, None

    latest_row = indicator_data.sort_values("year").iloc[-1]

    return (
        float(latest_row["value_numeric"]),
        int(latest_row["year"]),
    )


def format_percentage(
    value: float | None,
    decimal_places: int = 1,
) -> str:
    """Format a percentage or return Not available."""

    if value is None or pd.isna(value):
        return "Not available"

    return f"{value:.{decimal_places}f}%"


def calculate_growth(
    indicator_code: str,
    start_year: int,
    end_year: int,
) -> float | None:
    """
    Calculate percentage-point growth between two years.
    """

    indicator_data = get_indicator_series(indicator_code)

    if indicator_data.empty:
        return None

    start_values = indicator_data.loc[
        indicator_data["year"].eq(start_year),
        "value_numeric",
    ]

    end_values = indicator_data.loc[
        indicator_data["year"].eq(end_year),
        "value_numeric",
    ]

    if start_values.empty or end_values.empty:
        return None

    return float(end_values.iloc[0] - start_values.iloc[0])


def format_growth(
    value: float | None,
) -> str:
    """Format percentage-point growth."""

    if value is None or pd.isna(value):
        return "Not available"

    sign = "+" if value >= 0 else ""

    return f"{sign}{value:.1f} pp"


def get_final_year_projection(
    forecast_dataframe: pd.DataFrame,
    scenario: str,
    forecast_year: int,
) -> float | None:
    """
    Return the forecast value for one scenario and year.
    """

    required_columns = {
        "year",
        scenario,
    }

    if forecast_dataframe.empty or not required_columns.issubset(
        forecast_dataframe.columns
    ):
        return None

    forecast_rows = forecast_dataframe.loc[
        forecast_dataframe["year"].eq(forecast_year),
        scenario,
    ]

    if forecast_rows.empty:
        return None

    return float(forecast_rows.iloc[0])


def get_indicator_series(
    indicator_code: str,
) -> pd.DataFrame:
    if observations_df.empty:
        return pd.DataFrame()

    if "indicator_code" not in observations_df.columns:
        return pd.DataFrame()

    result = observations_df[
        observations_df["indicator_code"].astype(str).str.strip().eq(indicator_code)
    ].copy()

    if "gender" in result.columns:
        gender_value = result["gender"].fillna("").astype(str).str.lower().str.strip()

        result = result[
            gender_value.isin(
                [
                    "",
                    "all",
                    "overall",
                    "national",
                    "total",
                ]
            )
        ]

    if "location" in result.columns:
        location_value = (
            result["location"].fillna("").astype(str).str.lower().str.strip()
        )

        result = result[
            location_value.isin(
                [
                    "",
                    "all",
                    "overall",
                    "national",
                    "ethiopia",
                    "total",
                ]
            )
        ]

    result = result.dropna(subset=["year", "value_numeric"])

    if result.empty:
        return result

    return (
        result.groupby(
            "year",
            as_index=False,
        )["value_numeric"]
        .mean()
        .sort_values("year")
    )


access_history = get_indicator_series("ACC_OWNERSHIP")

usage_history = get_indicator_series("USG_DIGITAL_PAYMENT_RATE")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Dashboard Sections")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Overview",
        "Trends",
        "Event Impacts",
        "Forecasts",
        "Model Explainability",
        "Inclusion Projections",
        "Data Explorer",
    ],
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Forecast period:** 2025–2027

    **Main measures**

    - Account Ownership
    - Digital Payment Usage
    """
)


# =========================================================
# COMMON HEADER
# =========================================================

st.title("Ethiopia Financial Inclusion Dashboard")

st.caption(
    "Explore historical progress, payment-channel trends, "
    "major events and financial inclusion forecasts."
)


# =========================================================
# OVERVIEW PAGE
# =========================================================


def show_overview() -> None:
    """Display current financial inclusion metrics."""

    st.header("Overview")

    account_value, account_year = get_latest_indicator_value("ACC_OWNERSHIP")

    usage_value, usage_year = get_latest_indicator_value("USG_DIGITAL_PAYMENT_RATE")

    mobile_money_value, mobile_money_year = get_latest_indicator_value("ACC_MM_ACCOUNT")

    account_long_term_growth = calculate_growth(
        indicator_code="ACC_OWNERSHIP",
        start_year=2011,
        end_year=2024,
    )

    account_recent_growth = calculate_growth(
        indicator_code="ACC_OWNERSHIP",
        start_year=2021,
        end_year=2024,
    )

    mobile_money_growth = calculate_growth(
        indicator_code="ACC_MM_ACCOUNT",
        start_year=2021,
        end_year=2024,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        (
            f"Account Ownership, {account_year}"
            if account_year is not None
            else "Account Ownership"
        ),
        format_percentage(
            account_value,
            decimal_places=1,
        ),
        (
            format_growth(account_recent_growth)
            if account_recent_growth is not None
            else None
        ),
    )

    col2.metric(
        (
            f"Digital Payment Usage, {usage_year}"
            if usage_year is not None
            else "Digital Payment Usage"
        ),
        format_percentage(
            usage_value,
            decimal_places=1,
        ),
    )

    col3.metric(
        (
            f"Mobile Money Account Rate, {mobile_money_year}"
            if mobile_money_year is not None
            else "Mobile Money Account Rate"
        ),
        format_percentage(
            mobile_money_value,
            decimal_places=2,
        ),
        (
            format_growth(mobile_money_growth)
            if mobile_money_growth is not None
            else None
        ),
    )

    col4.metric(
        "P2P / ATM Crossover Ratio",
        "Not available",
    )

    st.caption(
        "The P2P/ATM ratio is not displayed because the "
        "processed dataset does not currently contain enough "
        "comparable ATM and P2P observations to calculate it "
        "reliably."
    )

    st.subheader("Growth highlights")

    growth_1, growth_2, growth_3 = st.columns(3)

    growth_1.metric(
        "Account ownership growth",
        format_growth(account_long_term_growth),
        "2011 to 2024",
    )

    growth_2.metric(
        "Recent Access growth",
        format_growth(account_recent_growth),
        "2021 to 2024",
    )

    growth_3.metric(
        "Mobile money growth",
        format_growth(mobile_money_growth),
        "2021 to 2024",
    )

    st.info(
        """
        Account ownership has increased strongly over the
        long term, but recent growth has slowed. Digital
        payment usage is expected to expand faster than new
        account ownership, meaning future progress may depend
        more on active use than on opening additional accounts.
        """
    )

    if not access_history.empty:
        figure = px.line(
            access_history,
            x="year",
            y="value_numeric",
            markers=True,
            title="Account Ownership Trend",
            labels={
                "year": "Year",
                "value_numeric": ("Account ownership (%)"),
            },
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    overview_data = pd.DataFrame(
        {
            "metric": [
                "Account Ownership",
                "Digital Payment Usage",
                "Mobile Money Account Rate",
                "P2P / ATM Crossover Ratio",
            ],
            "year": [
                account_year,
                usage_year,
                mobile_money_year,
                None,
            ],
            "value": [
                account_value,
                usage_value,
                mobile_money_value,
                None,
            ],
            "unit": [
                "percent",
                "percent",
                "percent",
                "ratio",
            ],
            "status": [
                "available",
                "available",
                "available",
                "not available",
            ],
        }
    )

    st.download_button(
        label="Download overview data",
        data=overview_data.to_csv(index=False),
        file_name="overview_metrics.csv",
        mime="text/csv",
    )


# =========================================================
# TRENDS PAGE
# =========================================================


def show_trends() -> None:
    st.header("Trends")

    if observations_df.empty:
        st.warning("Historical observations are not available.")
        return

    available_years = observations_df["year"].dropna().astype(int)

    if available_years.empty:
        st.warning("No valid observation years found.")
        return

    min_year = int(available_years.min())
    max_year = int(available_years.max())

    selected_range = st.slider(
        "Select date range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )

    filtered_observations = observations_df[
        observations_df["year"].between(
            selected_range[0],
            selected_range[1],
        )
    ].copy()

    st.subheader("Financial inclusion time series")

    selected_indicator = st.selectbox(
        "Choose an indicator",
        options=[
            "ACC_OWNERSHIP",
            "ACC_MM_ACCOUNT",
            "USG_DIGITAL_PAYMENT_RATE",
            "USG_P2P_COUNT",
        ],
        format_func=lambda value: {
            "ACC_OWNERSHIP": "Account Ownership",
            "ACC_MM_ACCOUNT": "Mobile Money Account Rate",
            "USG_DIGITAL_PAYMENT_RATE": ("Digital Payment Usage"),
            "USG_P2P_COUNT": "P2P Transaction Count",
        }.get(value, value),
    )

    indicator_data = filtered_observations[
        filtered_observations["indicator_code"].astype(str).eq(selected_indicator)
    ].copy()

    if indicator_data.empty:
        st.info(
            "No observations are available for this indicator "
            "within the selected date range."
        )

    else:
        indicator_data = indicator_data.groupby(
            "year",
            as_index=False,
        )["value_numeric"].mean()

        fig = px.line(
            indicator_data,
            x="year",
            y="value_numeric",
            markers=True,
            title="Selected Indicator Trend",
            labels={
                "year": "Year",
                "value_numeric": "Value",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader("Channel comparison")

    channel_codes = [
        "USG_P2P_COUNT",
        "USG_ATM_COUNT",
        "USG_POS_COUNT",
        "USG_DIGITAL_TXN_VALUE",
    ]

    channel_data = filtered_observations[
        filtered_observations["indicator_code"].astype(str).isin(channel_codes)
    ].copy()

    if channel_data.empty:
        st.info("Limited channel-comparison data is available.")

    else:
        channel_data = channel_data.groupby(
            [
                "year",
                "indicator_code",
            ],
            as_index=False,
        )["value_numeric"].mean()

        fig = px.line(
            channel_data,
            x="year",
            y="value_numeric",
            color="indicator_code",
            markers=True,
            title="Payment Channel Comparison",
            labels={
                "year": "Year",
                "value_numeric": "Value",
                "indicator_code": "Channel",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.download_button(
        "Download filtered trend data",
        data=filtered_observations.to_csv(index=False),
        file_name="filtered_trend_data.csv",
        mime="text/csv",
    )


# =========================================================
# EVENT IMPACTS PAGE
# =========================================================


def show_event_impacts() -> None:
    st.header("Event Impacts")

    event_df = data["event_impacts"]

    if event_df.empty:
        st.warning("The refined event-impact file is not available.")

        st.code(str(PROCESSED_DATA_DIR / "refined_event_impacts.csv"))

        return

    indicator_options = sorted(
        event_df["related_indicator"].dropna().astype(str).unique()
    )

    selected_indicators = st.multiselect(
        "Filter by affected indicator",
        options=indicator_options,
        default=indicator_options,
    )

    filtered_events = event_df[
        event_df["related_indicator"].astype(str).isin(selected_indicators)
    ].copy()

    if "refined_effect_score" in filtered_events.columns:
        chart_data = filtered_events.groupby(
            "event_name",
            as_index=False,
        )["refined_effect_score"].sum()

        fig = px.bar(
            chart_data,
            x="refined_effect_score",
            y="event_name",
            orientation="h",
            title="Expected Event Influence",
            labels={
                "refined_effect_score": ("Relative effect score"),
                "event_name": "Event",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.caption(
        "Effect scores show expected direction and relative "
        "strength. They are not percentage-point estimates."
    )

    display_columns = [
        column
        for column in [
            "event_name",
            "related_indicator",
            "impact_direction",
            "refined_magnitude",
            "refined_lag_months",
            "estimate_confidence",
            "adjustment_reason",
        ]
        if column in filtered_events.columns
    ]

    st.dataframe(
        filtered_events[display_columns],
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download event-impact data",
        data=filtered_events.to_csv(index=False),
        file_name="event_impacts.csv",
        mime="text/csv",
    )


# =========================================================
# FORECASTS PAGE
# =========================================================


def show_forecasts() -> None:
    """Display model forecasts and uncertainty."""

    st.header("Forecasts")

    access_df = data["access_forecasts"]
    usage_df = data["usage_forecasts"]
    uncertainty_df = data["uncertainty"]

    if access_df.empty or usage_df.empty:
        st.warning("Forecast files are missing from data/processed.")
        return

    target_option = st.radio(
        "Select target",
        [
            "Account Ownership",
            "Digital Payment Usage",
        ],
        horizontal=True,
    )

    view_option = st.selectbox(
        "Select forecast view",
        [
            "Base forecast",
            "Scenario comparison",
        ],
    )

    selected_df = (
        access_df.copy() if target_option == "Account Ownership" else usage_df.copy()
    )

    required_columns = {
        "year",
        "pessimistic",
        "base",
        "optimistic",
    }

    missing_columns = required_columns - set(selected_df.columns)

    if missing_columns:
        st.error(
            f"The selected forecast file is missing columns: {sorted(missing_columns)}"
        )
        return

    selected_df["year"] = pd.to_numeric(
        selected_df["year"],
        errors="coerce",
    )

    for scenario_column in [
        "pessimistic",
        "base",
        "optimistic",
    ]:
        selected_df[scenario_column] = pd.to_numeric(
            selected_df[scenario_column],
            errors="coerce",
        )

    selected_df = selected_df.dropna(
        subset=[
            "year",
            "pessimistic",
            "base",
            "optimistic",
        ]
    )

    if view_option == "Base forecast":
        figure = px.line(
            selected_df,
            x="year",
            y="base",
            markers=True,
            title=(f"{target_option}: Base Forecast"),
            labels={
                "year": "Year",
                "base": "Forecast rate (%)",
            },
        )

    else:
        long_dataframe = selected_df.melt(
            id_vars="year",
            value_vars=[
                "pessimistic",
                "base",
                "optimistic",
            ],
            var_name="scenario",
            value_name="forecast_rate",
        )

        figure = px.line(
            long_dataframe,
            x="year",
            y="forecast_rate",
            color="scenario",
            markers=True,
            title=(f"{target_option}: Scenario Comparison"),
            labels={
                "year": "Year",
                "forecast_rate": ("Forecast rate (%)"),
                "scenario": "Scenario",
            },
        )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    forecast_end_year = CONFIG.forecast.end_year

    pessimistic_value = get_final_year_projection(
        forecast_dataframe=selected_df,
        scenario="pessimistic",
        forecast_year=forecast_end_year,
    )

    base_value = get_final_year_projection(
        forecast_dataframe=selected_df,
        scenario="base",
        forecast_year=forecast_end_year,
    )

    optimistic_value = get_final_year_projection(
        forecast_dataframe=selected_df,
        scenario="optimistic",
        forecast_year=forecast_end_year,
    )

    st.subheader(f"{forecast_end_year} scenario outcomes")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Pessimistic",
        format_percentage(pessimistic_value),
    )

    col2.metric(
        "Base",
        format_percentage(base_value),
    )

    col3.metric(
        "Optimistic",
        format_percentage(optimistic_value),
    )

    st.subheader("Forecast uncertainty")

    target_name = (
        "Account Ownership Rate"
        if target_option == "Account Ownership"
        else "Digital Payment Usage"
    )

    if uncertainty_df.empty:
        st.info("The forecast uncertainty file is not available.")

    elif "target" not in uncertainty_df.columns:
        st.info("The uncertainty dataset does not contain a target column.")

    else:
        target_uncertainty = uncertainty_df[
            uncertainty_df["target"].astype(str).eq(target_name)
        ].copy()

        required_uncertainty_columns = {
            "year",
            "lower_80",
            "upper_80",
            "base_forecast",
        }

        if target_uncertainty.empty or not required_uncertainty_columns.issubset(
            target_uncertainty.columns
        ):
            st.info("Uncertainty values are not available for the selected target.")

        else:
            uncertainty_chart = go.Figure()

            uncertainty_chart.add_trace(
                go.Scatter(
                    x=target_uncertainty["year"],
                    y=target_uncertainty["upper_80"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                )
            )

            uncertainty_chart.add_trace(
                go.Scatter(
                    x=target_uncertainty["year"],
                    y=target_uncertainty["lower_80"],
                    mode="lines",
                    fill="tonexty",
                    name="80% uncertainty range",
                )
            )

            uncertainty_chart.add_trace(
                go.Scatter(
                    x=target_uncertainty["year"],
                    y=target_uncertainty["base_forecast"],
                    mode="lines+markers",
                    name="Base forecast",
                )
            )

            uncertainty_chart.update_layout(
                title=(f"{target_option} Forecast with Uncertainty"),
                xaxis_title="Year",
                yaxis_title="Rate (%)",
            )

            st.plotly_chart(
                uncertainty_chart,
                use_container_width=True,
            )

    with st.expander("How reliable are these forecasts?"):
        if target_option == "Account Ownership":
            st.write(
                """
                Account Ownership forecasts are based on a
                small number of national historical observations.
                The trend is useful for planning, but the scenario
                range should be considered more important than a
                single exact value.
                """
            )

        else:
            st.write(
                """
                Digital Payment Usage has very limited direct
                national historical data. These results should be
                treated as planning scenarios rather than precise
                predictions, and the wider uncertainty reflects
                this additional risk.
                """
            )

    st.subheader("Business interpretation")

    if target_option == "Account Ownership":
        if base_value is not None:
            access_gap = calculate_target_gap(
                projection=base_value,
                target_rate=(CONFIG.forecast.target_rate),
            )

            st.info(
                f"Under the base scenario, Account "
                f"Ownership reaches approximately "
                f"{base_value:.1f}% by "
                f"{forecast_end_year}. This leaves a "
                f"{access_gap:.1f} percentage-point gap "
                f"to the "
                f"{CONFIG.forecast.target_rate:.0f}% "
                "target."
            )

    elif base_value is not None:
        st.info(
            f"Under the base scenario, Digital Payment "
            f"Usage reaches approximately "
            f"{base_value:.1f}% by "
            f"{forecast_end_year}. This suggests that "
            "active digital usage may grow faster than "
            "overall account ownership."
        )

    st.download_button(
        "Download selected forecast",
        data=selected_df.to_csv(index=False),
        file_name="selected_forecast.csv",
        mime="text/csv",
    )


def show_model_explainability() -> None:
    """Display global and local SHAP explanations."""

    st.header("Model Explainability")

    st.caption(
        "These explanations describe the behavior of "
        "the Account Ownership regression model. "
        "They do not prove causal relationships."
    )

    global_df = data.get(
        "access_shap_global",
        pd.DataFrame(),
    )

    local_df = data.get(
        "access_shap_local",
        pd.DataFrame(),
    )

    diagnostics_df = data.get(
        "access_shap_diagnostics",
        pd.DataFrame(),
    )

    if global_df.empty or local_df.empty:
        st.warning(
            "SHAP explanation files are not available. "
            "Run the forecasting notebook to generate them."
        )
        return

    st.subheader("Which features matter most globally?")

    required_global_columns = {
        "feature",
        "mean_absolute_shap",
    }

    if required_global_columns.issubset(global_df.columns):
        global_chart = px.bar(
            global_df.sort_values(
                "mean_absolute_shap",
                ascending=True,
            ),
            x="mean_absolute_shap",
            y="feature",
            orientation="h",
            title=("Global SHAP Feature Importance"),
            labels={
                "mean_absolute_shap": ("Mean absolute SHAP value"),
                "feature": "Feature",
            },
        )

        st.plotly_chart(
            global_chart,
            use_container_width=True,
        )

        importance_total = global_df["mean_absolute_shap"].sum()

        global_df["importance_share"] = (
            global_df["mean_absolute_shap"] / importance_total
        )

        top_feature = global_df.sort_values(
            "importance_share",
            ascending=False,
        ).iloc[0]

        top_share = float(top_feature["importance_share"]) * 100

        if top_share > 70:
            st.warning(
                f"The model assigns {top_share:.1f}% of total "
                f"importance to {top_feature['feature']}. "
                "The model may depend too heavily on one feature."
            )

        elif top_share > 40:
            st.info(
                f"The largest driver is "
                f"{top_feature['feature']} at "
                f"{top_share:.1f}% of total importance. "
                "The model remains moderately trend-driven, "
                "but demographic features also contribute "
                "meaningfully."
            )

        else:
            st.success(
                "Feature importance is distributed across "
                "multiple predictors, with no single feature "
                "dominating the model."
            )

    st.subheader("Why did the model make a specific prediction?")

    observation_options = sorted(local_df["observation_id"].dropna().unique())

    selected_observation = st.selectbox(
        "Select a historical year or prediction",
        observation_options,
    )

    selected_local = local_df[
        local_df["observation_id"].eq(selected_observation)
    ].copy()

    if not selected_local.empty:
        prediction = float(selected_local["prediction"].iloc[0])

        base_value = float(selected_local["base_value"].iloc[0])

        metric_1, metric_2 = st.columns(2)

        metric_1.metric(
            "Model prediction",
            f"{prediction:.2f}%",
        )

        metric_2.metric(
            "Average model output",
            f"{base_value:.2f}%",
        )

        selected_local["direction"] = np.where(
            selected_local["shap_value"] >= 0,
            "Increased prediction",
            "Reduced prediction",
        )

        local_chart = px.bar(
            selected_local.sort_values("shap_value"),
            x="shap_value",
            y="feature",
            color="direction",
            orientation="h",
            title=(f"Feature Contributions for {selected_observation}"),
            labels={
                "shap_value": ("Contribution to prediction"),
                "feature": "Feature",
                "direction": "Effect",
            },
        )

        st.plotly_chart(
            local_chart,
            use_container_width=True,
        )

        st.dataframe(
            selected_local[
                [
                    "feature",
                    "feature_value",
                    "shap_value",
                    "direction",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Are there any concerning patterns?")

    if diagnostics_df.empty:
        st.info("No automatic diagnostic results are available.")

    else:
        for _, diagnostic in diagnostics_df.iterrows():
            message = (
                f"**{diagnostic['pattern']} — "
                f"{diagnostic['feature']}**\n\n"
                f"{diagnostic['detail']}"
            )

            if str(diagnostic["severity"]).lower() == "warning":
                st.warning(message)
            else:
                st.info(message)

    with st.expander("Important interpretation limitations"):
        st.write(
            """
            SHAP describes how the fitted model uses its
            inputs. It does not establish that a feature
            caused a change in financial inclusion.

            The Account Ownership model has only a small
            number of historical survey observations.
            Correlated variables may divide or exchange
            importance, and the resulting explanations may
            change when new observations are added.

            Digital Payment Usage is not explained with SHAP
            because there is not enough direct historical
            target data to fit a defensible multivariable
            model.
            """
        )

    st.download_button(
        "Download global SHAP results",
        data=global_df.to_csv(index=False),
        file_name=("access_shap_global.csv"),
        mime="text/csv",
    )


# =========================================================
# INCLUSION PROJECTIONS PAGE
# =========================================================


def show_inclusion_projections() -> None:
    """
    Display scenario projections and business impact.
    """

    st.header("Inclusion Projections")

    access_df = data["access_forecasts"]

    if access_df.empty:
        st.warning("Access forecast data is not available.")
        return

    required_columns = {
        "year",
        "pessimistic",
        "base",
        "optimistic",
    }

    missing_columns = required_columns - set(access_df.columns)

    if missing_columns:
        st.error(
            f"The Access forecast dataset is missing columns: {sorted(missing_columns)}"
        )
        return

    access_df = access_df.copy()

    access_df["year"] = pd.to_numeric(
        access_df["year"],
        errors="coerce",
    )

    for scenario_column in [
        "pessimistic",
        "base",
        "optimistic",
    ]:
        access_df[scenario_column] = pd.to_numeric(
            access_df[scenario_column],
            errors="coerce",
        )

    access_df = access_df.dropna(
        subset=[
            "year",
            "pessimistic",
            "base",
            "optimistic",
        ]
    )

    scenario = st.selectbox(
        "Select scenario",
        [
            "pessimistic",
            "base",
            "optimistic",
        ],
        format_func=str.capitalize,
    )

    selected_projection = (
        access_df[["year", scenario]]
        .copy()
        .rename(
            columns={
                scenario: "projection",
            }
        )
    )

    target_rate = CONFIG.forecast.target_rate

    forecast_end_year = CONFIG.forecast.end_year

    selected_projection["remaining_to_target"] = selected_projection[
        "projection"
    ].apply(
        lambda projection: calculate_target_gap(
            projection=float(projection),
            target_rate=target_rate,
        )
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=selected_projection["year"],
            y=selected_projection["projection"],
            mode="lines+markers",
            name=scenario.capitalize(),
        )
    )

    figure.add_hline(
        y=target_rate,
        line_dash="dash",
        annotation_text=(f"{target_rate:.0f}% target"),
    )

    figure.update_layout(
        title=(f"Progress Toward {target_rate:.0f}% Account Ownership"),
        xaxis_title="Year",
        yaxis_title=("Account ownership (%)"),
        yaxis_range=[
            45,
            target_rate + 2,
        ],
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    latest_projection = get_final_year_projection(
        forecast_dataframe=access_df,
        scenario=scenario,
        forecast_year=forecast_end_year,
    )

    if latest_projection is None:
        st.warning(f"No projection is available for {forecast_end_year}.")
        return

    remaining_gap = calculate_target_gap(
        projection=latest_projection,
        target_rate=target_rate,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        f"{forecast_end_year} projection",
        f"{latest_projection:.1f}%",
    )

    col2.metric(
        "Target",
        f"{target_rate:.1f}%",
    )

    col3.metric(
        "Remaining gap",
        f"{remaining_gap:.1f} pp",
    )

    if remaining_gap == 0:
        st.success(
            f"Under the {scenario} scenario, Ethiopia "
            f"reaches the {target_rate:.0f}% Account "
            f"Ownership target by "
            f"{forecast_end_year}."
        )

    elif remaining_gap <= 5:
        st.warning(
            f"The {scenario} scenario comes close to "
            f"the target, but still leaves a "
            f"{remaining_gap:.1f} percentage-point "
            "gap."
        )

    else:
        st.error(
            f"The {scenario} scenario leaves a "
            f"{remaining_gap:.1f} percentage-point "
            "gap. Stronger rural, gender and "
            "mobile-money inclusion measures would "
            "be required."
        )

    st.subheader(f"{forecast_end_year} scenario comparison")

    final_year_rows = access_df.loc[
        access_df["year"].eq(forecast_end_year),
        [
            "pessimistic",
            "base",
            "optimistic",
        ],
    ]

    if final_year_rows.empty:
        st.info("Final-year scenario values are not available.")

    else:
        scenario_summary = final_year_rows.iloc[0].rename("projection").reset_index()

        scenario_summary.columns = [
            "scenario",
            "projection",
        ]

        scenario_summary["gap_to_target"] = scenario_summary["projection"].apply(
            lambda projection: calculate_target_gap(
                projection=float(projection),
                target_rate=target_rate,
            )
        )

        scenario_summary["scenario"] = scenario_summary["scenario"].str.capitalize()

        scenario_summary.columns = [
            "Scenario",
            (f"{forecast_end_year} projection (%)"),
            "Gap to target (pp)",
        ]

        st.dataframe(
            scenario_summary,
            hide_index=True,
            use_container_width=True,
        )

        comparison_figure = px.bar(
            scenario_summary,
            x="Scenario",
            y=(f"{forecast_end_year} projection (%)"),
            title=(f"{forecast_end_year} Account Ownership Scenarios"),
            labels={
                (f"{forecast_end_year} projection (%)"): "Projection (%)",
            },
        )

        comparison_figure.add_hline(
            y=target_rate,
            line_dash="dash",
            annotation_text=(f"{target_rate:.0f}% target"),
        )

        st.plotly_chart(
            comparison_figure,
            use_container_width=True,
        )

    st.subheader("Answers to the consortium's key questions")

    with st.expander(
        f"Will Ethiopia reach the Account Ownership target by {forecast_end_year}?"
    ):
        if remaining_gap == 0:
            st.write(
                f"Yes. Under the selected "
                f"{scenario} scenario, the target "
                f"is reached by "
                f"{forecast_end_year}."
            )

        else:
            st.write(
                f"No. Under the selected "
                f"{scenario} scenario, Account "
                f"Ownership reaches "
                f"{latest_projection:.1f}%, leaving "
                f"a {remaining_gap:.1f} "
                "percentage-point gap."
            )

    with st.expander("Will Usage grow faster than Access?"):
        st.write(
            """
            The base forecasts suggest that Digital
            Payment Usage may grow faster than Account
            Ownership. This means institutions may gain
            more impact by encouraging active use of
            existing accounts, rather than focusing only
            on opening new accounts.
            """
        )

    with st.expander("Which developments have the greatest potential impact?"):
        st.write(
            """
            Mobile money expansion, interoperability,
            agent and merchant growth, digital identity,
            affordable transaction services and stronger
            rural connectivity are expected to have the
            greatest influence.
            """
        )

    with st.expander("What could prevent the forecasts from being achieved?"):
        st.write(
            """
            High fees, inflation, weak rural connectivity,
            gender inequality, limited trust, low financial
            literacy, inactive accounts and insufficient
            merchant acceptance could slow progress.
            """
        )

    with st.expander("How should these projections be used?"):
        st.write(
            """
            The projections should be used as planning
            scenarios rather than exact predictions.
            Decision-makers should compare the pessimistic,
            base and optimistic outcomes and update the
            forecasts when new national data becomes
            available.
            """
        )

    st.download_button(
        "Download inclusion projections",
        data=selected_projection.to_csv(index=False),
        file_name=(f"inclusion_projection_{scenario}.csv"),
        mime="text/csv",
    )


# =========================================================
# DATA EXPLORER PAGE
# =========================================================


def show_data_explorer() -> None:
    st.header("Data Explorer")

    enriched_df = data["enriched"]

    if enriched_df.empty:
        st.warning("The enriched dataset is not available.")
        return

    record_types = sorted(enriched_df["record_type"].dropna().astype(str).unique())

    selected_types = st.multiselect(
        "Filter by record type",
        options=record_types,
        default=record_types,
    )

    filtered_df = enriched_df[
        enriched_df["record_type"].astype(str).isin(selected_types)
    ]

    st.write(f"Records displayed: {len(filtered_df)}")

    st.dataframe(
        filtered_df,
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download filtered dataset",
        data=filtered_df.to_csv(index=False),
        file_name="financial_inclusion_data.csv",
        mime="text/csv",
    )


# =========================================================
# PAGE ROUTING
# =========================================================

if page == "Overview":
    show_overview()

elif page == "Trends":
    show_trends()

elif page == "Event Impacts":
    show_event_impacts()

elif page == "Forecasts":
    show_forecasts()

elif page == "Model Explainability":
    show_model_explainability()

elif page == "Inclusion Projections":
    show_inclusion_projections()

elif page == "Data Explorer":
    show_data_explorer()
