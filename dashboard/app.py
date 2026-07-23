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
def load_csv(filename):
    file_path = PROCESSED_DATA_DIR / filename

    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)

    except Exception as error:
        st.error(
            f"Could not load {filename}: {error}"
        )
        return pd.DataFrame()


@st.cache_data
def load_dashboard_data():
    return {
        "enriched": load_csv(
            "ethiopia_fi_enriched.csv"
        ),
        "event_impacts": load_csv(
            "refined_event_impacts.csv"
        ),
        "association_summary": load_csv(
            "event_indicator_association_summary.csv"
        ),
        "access_forecasts": load_csv(
            "access_forecast_scenarios.csv"
        ),
        "usage_forecasts": load_csv(
            "usage_forecast_scenarios.csv"
        ),
        "final_forecasts": load_csv(
            "final_financial_inclusion_forecasts.csv"
        ),
        "uncertainty": load_csv(
            "forecast_uncertainty_summary.csv"
        ),
    }


data = load_dashboard_data()


# =========================================================
# DATA PREPARATION HELPERS
# =========================================================

def prepare_observations(enriched_df):
    if enriched_df.empty:
        return pd.DataFrame()

    observations = enriched_df.copy()

    if "record_type" in observations.columns:
        observations = observations[
            observations["record_type"]
            .astype(str)
            .str.lower()
            .eq("observation")
        ].copy()

    if "observation_date" in observations.columns:
        observations["observation_date"] = pd.to_datetime(
            observations["observation_date"],
            errors="coerce",
        )

        observations["year"] = (
            observations["observation_date"].dt.year
        )

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


observations_df = prepare_observations(
    data["enriched"]
)


def get_indicator_series(indicator_code):
    if observations_df.empty:
        return pd.DataFrame()

    if "indicator_code" not in observations_df.columns:
        return pd.DataFrame()

    result = observations_df[
        observations_df["indicator_code"]
        .astype(str)
        .str.strip()
        .eq(indicator_code)
    ].copy()

    if "gender" in result.columns:
        gender_value = (
            result["gender"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

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
            result["location"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
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

    result = result.dropna(
        subset=["year", "value_numeric"]
    )

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


access_history = get_indicator_series(
    "ACC_OWNERSHIP"
)

usage_history = get_indicator_series(
    "USG_DIGITAL_PAYMENT_RATE"
)


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

st.title(
    "Ethiopia Financial Inclusion Dashboard"
)

st.caption(
    "Explore historical progress, payment-channel trends, "
    "major events and financial inclusion forecasts."
)


# =========================================================
# OVERVIEW PAGE
# =========================================================

def show_overview():
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Account Ownership, 2024",
        "49.0%",
        "+3 percentage points since 2021",
    )

    col2.metric(
        "Digital Payment Usage, 2024",
        "35.0%",
    )

    col3.metric(
        "Mobile Money Account Rate, 2024",
        "9.45%",
        "4.7% in 2021",
    )

    col4.metric(
        "P2P / ATM Crossover Ratio",
        "1.08",
        "P2P above ATM",
    )

    st.subheader("Growth highlights")

    growth_1, growth_2, growth_3 = st.columns(3)

    growth_1.metric(
        "Account ownership growth",
        "+35 pp",
        "2011 to 2024",
    )

    growth_2.metric(
        "Recent Access growth",
        "+3 pp",
        "2021 to 2024",
    )

    growth_3.metric(
        "Mobile money growth",
        "+4.75 pp",
        "2021 to 2024",
    )

    st.info(
        """
        Account ownership has grown strongly over the long term,
        but recent growth has slowed. Digital payment activity is
        expected to grow faster than new account ownership.
        """
    )

    if not access_history.empty:
        fig = px.line(
            access_history,
            x="year",
            y="value_numeric",
            markers=True,
            title="Account Ownership Trend",
            labels={
                "year": "Year",
                "value_numeric": "Account ownership (%)",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.download_button(
        label="Download overview data",
        data=pd.DataFrame({
            "Metric": [
                "Account Ownership 2024",
                "Digital Payment Usage 2024",
                "Mobile Money Account Rate 2024",
                "P2P / ATM Crossover Ratio",
            ],
            "Value": [
                49.0,
                35.0,
                9.45,
                1.08,
            ],
        }).to_csv(index=False),
        file_name="overview_metrics.csv",
        mime="text/csv",
    )


# =========================================================
# TRENDS PAGE
# =========================================================

def show_trends():
    st.header("Trends")

    if observations_df.empty:
        st.warning(
            "Historical observations are not available."
        )
        return

    available_years = (
        observations_df["year"]
        .dropna()
        .astype(int)
    )

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
            "USG_DIGITAL_PAYMENT_RATE": (
                "Digital Payment Usage"
            ),
            "USG_P2P_COUNT": "P2P Transaction Count",
        }.get(value, value),
    )

    indicator_data = filtered_observations[
        filtered_observations["indicator_code"]
        .astype(str)
        .eq(selected_indicator)
    ].copy()

    if indicator_data.empty:
        st.info(
            "No observations are available for this indicator "
            "within the selected date range."
        )

    else:
        indicator_data = (
            indicator_data.groupby(
                "year",
                as_index=False,
            )["value_numeric"]
            .mean()
        )

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
        filtered_observations["indicator_code"]
        .astype(str)
        .isin(channel_codes)
    ].copy()

    if channel_data.empty:
        st.info(
            "Limited channel-comparison data is available."
        )

    else:
        channel_data = (
            channel_data.groupby(
                [
                    "year",
                    "indicator_code",
                ],
                as_index=False,
            )["value_numeric"]
            .mean()
        )

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
        data=filtered_observations.to_csv(
            index=False
        ),
        file_name="filtered_trend_data.csv",
        mime="text/csv",
    )


# =========================================================
# EVENT IMPACTS PAGE
# =========================================================

def show_event_impacts():
    st.header("Event Impacts")

    event_df = data["event_impacts"]

    if event_df.empty:
        st.warning(
            "The refined event-impact file is not available."
        )

        st.code(
            str(
                PROCESSED_DATA_DIR
                / "refined_event_impacts.csv"
            )
        )

        return

    indicator_options = sorted(
        event_df["related_indicator"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_indicators = st.multiselect(
        "Filter by affected indicator",
        options=indicator_options,
        default=indicator_options,
    )

    filtered_events = event_df[
        event_df["related_indicator"]
        .astype(str)
        .isin(selected_indicators)
    ].copy()

    if "refined_effect_score" in filtered_events.columns:
        chart_data = (
            filtered_events.groupby(
                "event_name",
                as_index=False,
            )["refined_effect_score"]
            .sum()
        )

        fig = px.bar(
            chart_data,
            x="refined_effect_score",
            y="event_name",
            orientation="h",
            title="Expected Event Influence",
            labels={
                "refined_effect_score": (
                    "Relative effect score"
                ),
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

def show_forecasts():
    st.header("Forecasts")

    access_df = data["access_forecasts"]
    usage_df = data["usage_forecasts"]
    uncertainty_df = data["uncertainty"]

    if access_df.empty or usage_df.empty:
        st.warning(
            "Forecast files are missing from data/processed."
        )
        return

    model_option = st.selectbox(
        "Select forecast model",
        [
            "Baseline trend",
            "Trend with expected events",
            "Scenario comparison",
        ],
    )

    target_option = st.radio(
        "Select target",
        [
            "Account Ownership",
            "Digital Payment Usage",
        ],
        horizontal=True,
    )

    selected_df = (
        access_df.copy()
        if target_option == "Account Ownership"
        else usage_df.copy()
    )

    if model_option == "Baseline trend":
        if target_option == "Account Ownership":
            selected_df["forecast"] = [
                50.0,
                51.0,
                52.0,
            ]

        else:
            selected_df["forecast"] = [
                37.5,
                40.0,
                42.5,
            ]

        fig = px.line(
            selected_df,
            x="year",
            y="forecast",
            markers=True,
            title=f"{target_option}: Baseline Trend",
            labels={
                "year": "Year",
                "forecast": "Forecast rate (%)",
            },
        )

    elif model_option == "Trend with expected events":
        fig = px.line(
            selected_df,
            x="year",
            y="base",
            markers=True,
            title=(
                f"{target_option}: Forecast with "
                "Expected Developments"
            ),
            labels={
                "year": "Year",
                "base": "Forecast rate (%)",
            },
        )

    else:
        long_df = selected_df.melt(
            id_vars="year",
            value_vars=[
                "pessimistic",
                "base",
                "optimistic",
            ],
            var_name="scenario",
            value_name="forecast_rate",
        )

        fig = px.line(
            long_df,
            x="year",
            y="forecast_rate",
            color="scenario",
            markers=True,
            title=f"{target_option}: Scenario Forecasts",
            labels={
                "year": "Year",
                "forecast_rate": "Forecast rate (%)",
                "scenario": "Scenario",
            },
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Forecast with uncertainty")

    target_name = (
        "Account Ownership Rate"
        if target_option == "Account Ownership"
        else "Digital Payment Usage"
    )

    if not uncertainty_df.empty:
        target_uncertainty = uncertainty_df[
            uncertainty_df["target"]
            .astype(str)
            .eq(target_name)
        ].copy()

        if not target_uncertainty.empty:
            uncertainty_chart = go.Figure()

            uncertainty_chart.add_trace(
                go.Scatter(
                    x=target_uncertainty["year"],
                    y=target_uncertainty[
                        "upper_80"
                    ],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )

            uncertainty_chart.add_trace(
                go.Scatter(
                    x=target_uncertainty["year"],
                    y=target_uncertainty[
                        "lower_80"
                    ],
                    mode="lines",
                    fill="tonexty",
                    name="Uncertainty range",
                )
            )

            uncertainty_chart.add_trace(
                go.Scatter(
                    x=target_uncertainty["year"],
                    y=target_uncertainty[
                        "base_forecast"
                    ],
                    mode="lines+markers",
                    name="Base forecast",
                )
            )

            uncertainty_chart.update_layout(
                title=(
                    f"{target_option} Forecast "
                    "with Uncertainty"
                ),
                xaxis_title="Year",
                yaxis_title="Rate (%)",
            )

            st.plotly_chart(
                uncertainty_chart,
                use_container_width=True,
            )

    st.subheader("Projected milestones")

    if target_option == "Account Ownership":
        st.write(
            """
            - Account ownership passes 50% around 2025.
            - Base forecast reaches approximately 53.2% in 2027.
            - The 60% target is not reached by 2027.
            """
        )

    else:
        st.write(
            """
            - Digital payment usage approaches 40% by 2026.
            - Base forecast reaches approximately 45.5% in 2027.
            - The optimistic scenario reaches 50% by 2027.
            """
        )

    st.download_button(
        "Download selected forecast",
        data=selected_df.to_csv(index=False),
        file_name="selected_forecast.csv",
        mime="text/csv",
    )


# =========================================================
# INCLUSION PROJECTIONS PAGE
# =========================================================

def show_inclusion_projections():
    st.header("Inclusion Projections")

    access_df = data["access_forecasts"]

    if access_df.empty:
        st.warning(
            "Access forecast data is not available."
        )
        return

    scenario = st.selectbox(
        "Select scenario",
        [
            "pessimistic",
            "base",
            "optimistic",
        ],
    )

    selected_projection = access_df[
        [
            "year",
            scenario,
        ]
    ].copy()

    selected_projection = selected_projection.rename(
        columns={
            scenario: "projection",
        }
    )

    target_rate = 60.0

    selected_projection[
        "remaining_to_target"
    ] = (
        target_rate
        - selected_projection["projection"]
    ).clip(lower=0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=selected_projection["year"],
            y=selected_projection["projection"],
            mode="lines+markers",
            name=scenario.capitalize(),
        )
    )

    fig.add_hline(
        y=target_rate,
        line_dash="dash",
        annotation_text="60% target",
    )

    fig.update_layout(
        title=(
            "Progress Toward 60% Account Ownership"
        ),
        xaxis_title="Year",
        yaxis_title="Account ownership (%)",
        yaxis_range=[45, 62],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    latest_projection = float(
        selected_projection.loc[
            selected_projection["year"] == 2027,
            "projection",
        ].iloc[0]
    )

    remaining_gap = target_rate - latest_projection

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "2027 projection",
        f"{latest_projection:.1f}%",
    )

    col2.metric(
        "Target",
        "60.0%",
    )

    col3.metric(
        "Remaining gap",
        f"{remaining_gap:.1f} percentage points",
    )

    st.subheader(
        "Answers to the consortium's key questions"
    )

    with st.expander(
        "Will Ethiopia reach 60% account ownership by 2027?"
    ):
        st.write(
            """
            No current scenario reaches 60% by 2027.
            The base scenario reaches approximately 53.2%.
            Faster progress would require stronger inclusion of
            women, rural adults and previously unbanked groups.
            """
        )

    with st.expander(
        "Will Usage grow faster than Access?"
    ):
        st.write(
            """
            Yes. The base forecast expects Digital Payment Usage
            to rise by about 10.5 percentage points from 2024 to
            2027, compared with about 4.2 percentage points for
            Account Ownership.
            """
        )

    with st.expander(
        "Which developments have the greatest potential impact?"
    ):
        st.write(
            """
            Mobile money expansion, interoperability, agent and
            merchant growth, digital identification and affordable
            transaction services have the greatest potential.
            """
        )

    with st.expander(
        "What could prevent the forecasts from being achieved?"
    ):
        st.write(
            """
            High fees, inflation, weak rural connectivity,
            gender gaps, low trust, limited financial literacy,
            inactive accounts and insufficient merchant acceptance
            could slow progress.
            """
        )

    st.download_button(
        "Download inclusion projections",
        data=selected_projection.to_csv(
            index=False
        ),
        file_name=(
            f"inclusion_projection_{scenario}.csv"
        ),
        mime="text/csv",
    )


# =========================================================
# DATA EXPLORER PAGE
# =========================================================

def show_data_explorer():
    st.header("Data Explorer")

    enriched_df = data["enriched"]

    if enriched_df.empty:
        st.warning(
            "The enriched dataset is not available."
        )
        return

    record_types = sorted(
        enriched_df["record_type"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_types = st.multiselect(
        "Filter by record type",
        options=record_types,
        default=record_types,
    )

    filtered_df = enriched_df[
        enriched_df["record_type"]
        .astype(str)
        .isin(selected_types)
    ]

    st.write(
        f"Records displayed: {len(filtered_df)}"
    )

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

elif page == "Inclusion Projections":
    show_inclusion_projections()

elif page == "Data Explorer":
    show_data_explorer()