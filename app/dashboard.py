
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from scipy.stats import linregress

# -----------------------------------
# PROJECT CONFIGURATION
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

st.set_page_config(
    page_title="Bengaluru Rainfall Analysis",
    page_icon="🌧️",
    layout="wide"
)

# -----------------------------------
# LOAD DATA
# -----------------------------------

@st.cache_data
def load_data():

    daily = pd.read_csv(
        DATA_DIR / "bengaluru_rainfall_cleaned.csv"
    )

    annual = pd.read_csv(
        REPORT_DIR / "annual_rainfall.csv"
    )

    monthly = pd.read_csv(
        REPORT_DIR / "monthly_rainfall.csv"
    )

    monthly_avg = pd.read_csv(
        REPORT_DIR / "monthly_average_rainfall.csv"
    )

    seasonal = pd.read_csv(
        REPORT_DIR / "seasonal_rainfall.csv"
    )

    seasonal_avg = pd.read_csv(
        REPORT_DIR / "seasonal_average_rainfall.csv"
    )

    daily["date"] = pd.to_datetime(daily["date"])

    return daily, annual, monthly, monthly_avg, seasonal, seasonal_avg


try:
    daily, annual, monthly, monthly_avg, seasonal, seasonal_avg = load_data()

except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


# -----------------------------------
# HEADER
# -----------------------------------

st.title("🌧️ Bengaluru Rainfall Analysis")
st.subheader("Historical Rainfall Pattern Analysis (2015–2025)")

st.markdown(
    """
    This application analyzes historical rainfall data of Bengaluru
    using statistical methods, mathematical evaluation and
    graphical visualization.
    """
)

st.divider()


# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("Analysis Controls")

years = sorted(daily["year"].unique())

selected_years = st.sidebar.multiselect(
    "Select Years",
    options=years,
    default=years
)

if not selected_years:
    st.warning("Please select at least one year.")
    st.stop()

filtered_daily = daily[
    daily["year"].isin(selected_years)
].copy()

filtered_annual = annual[
    annual["year"].isin(selected_years)
].copy()

filtered_monthly = monthly[
    monthly["year"].isin(selected_years)
].copy()

filtered_seasonal = seasonal[
    seasonal["year"].isin(selected_years)
].copy()


# -----------------------------------
# KEY STATISTICS
# -----------------------------------

total_rainfall = filtered_daily["rainfall_mm"].sum()
average_rainfall = filtered_daily["rainfall_mm"].mean()

maximum_rainfall = filtered_daily["rainfall_mm"].max()

rainy_days = (
    filtered_daily["rainfall_mm"] > 0
).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Rainfall",
    f"{total_rainfall:,.1f} mm"
)

col2.metric(
    "Average Daily Rainfall",
    f"{average_rainfall:.2f} mm"
)

col3.metric(
    "Maximum Daily Rainfall",
    f"{maximum_rainfall:.1f} mm"
)

col4.metric(
    "Rainy Days",
    f"{rainy_days:,}"
)

st.divider()


# -----------------------------------
# ANNUAL ANALYSIS
# -----------------------------------

st.header("1. Annual Rainfall Analysis")

fig_annual = px.bar(
    filtered_annual,
    x="year",
    y="annual_rainfall_mm",
    title="Year-wise Total Rainfall",
    labels={
        "year": "Year",
        "annual_rainfall_mm": "Rainfall (mm)"
    },
    text_auto=".1f"
)

st.plotly_chart(fig_annual, use_container_width=True)

fig_trend = px.line(
    filtered_annual,
    x="year",
    y="annual_rainfall_mm",
    markers=True,
    title="Annual Rainfall Trend",
    labels={
        "year": "Year",
        "annual_rainfall_mm": "Rainfall (mm)"
    }
)

st.plotly_chart(fig_trend, use_container_width=True)


# -----------------------------------
# MONTHLY ANALYSIS
# -----------------------------------

st.header("2. Monthly Rainfall Analysis")

monthly_selected_avg = (
    filtered_monthly.groupby(["month", "month_name"])
    ["monthly_rainfall_mm"]
    .mean()
    .reset_index()
    .sort_values("month")
)

fig_monthly = px.bar(
    monthly_selected_avg,
    x="month_name",
    y="monthly_rainfall_mm",
    title="Average Monthly Rainfall",
    labels={
        "month_name": "Month",
        "monthly_rainfall_mm": "Average Rainfall (mm)"
    },
    text_auto=".1f"
)

st.plotly_chart(fig_monthly, use_container_width=True)


# -----------------------------------
# SEASONAL ANALYSIS
# -----------------------------------

st.header("3. Seasonal Rainfall Analysis")

seasonal_selected_avg = (
    filtered_seasonal.groupby("season")
    ["seasonal_rainfall_mm"]
    .mean()
    .reset_index()
)

fig_season = px.bar(
    seasonal_selected_avg,
    x="season",
    y="seasonal_rainfall_mm",
    title="Average Seasonal Rainfall",
    labels={
        "season": "Season",
        "seasonal_rainfall_mm": "Average Rainfall (mm)"
    },
    text_auto=".1f"
)

st.plotly_chart(fig_season, use_container_width=True)


# -----------------------------------
# HEATMAP
# -----------------------------------

st.header("4. Monthly Rainfall Heatmap")

heatmap_data = filtered_monthly.pivot(
    index="year",
    columns="month",
    values="monthly_rainfall_mm"
)

fig_heatmap = px.imshow(
    heatmap_data,
    aspect="auto",
    color_continuous_scale="Blues",
    labels={
        "x": "Month",
        "y": "Year",
        "color": "Rainfall (mm)"
    },
    title="Monthly Rainfall Distribution by Year"
)

st.plotly_chart(fig_heatmap, use_container_width=True)


# -----------------------------------
# DAILY DISTRIBUTION
# -----------------------------------

st.header("5. Daily Rainfall Distribution")

fig_hist = px.histogram(
    filtered_daily,
    x="rainfall_mm",
    nbins=50,
    title="Distribution of Daily Rainfall",
    labels={
        "rainfall_mm": "Daily Rainfall (mm)",
        "count": "Number of Days"
    }
)

st.plotly_chart(fig_hist, use_container_width=True)


# -----------------------------------
# SEASONAL BOXPLOT
# -----------------------------------

st.header("6. Seasonal Rainfall Distribution")

season_order = [
    "Winter",
    "Summer",
    "Southwest Monsoon",
    "Northeast Monsoon"
]

fig_box = px.box(
    filtered_daily,
    x="season",
    y="rainfall_mm",
    category_orders={"season": season_order},
    title="Daily Rainfall Distribution by Season",
    labels={
        "season": "Season",
        "rainfall_mm": "Daily Rainfall (mm)"
    }
)

st.plotly_chart(fig_box, use_container_width=True)


# -----------------------------------
# AUTOMATED FINDINGS
# -----------------------------------

st.header("7. Automated Analytical Findings")

if not filtered_annual.empty:

    wettest_year = filtered_annual.loc[
        filtered_annual["annual_rainfall_mm"].idxmax()
    ]

    driest_year = filtered_annual.loc[
        filtered_annual["annual_rainfall_mm"].idxmin()
    ]

    wettest_month = monthly_selected_avg.loc[
        monthly_selected_avg["monthly_rainfall_mm"].idxmax()
    ]

    st.success(
        f"Wettest year: {int(wettest_year['year'])} "
        f"with {wettest_year['annual_rainfall_mm']:.1f} mm."
    )

    st.info(
        f"Lowest rainfall year: {int(driest_year['year'])} "
        f"with {driest_year['annual_rainfall_mm']:.1f} mm."
    )

    st.write(
        f"Highest average rainfall month: "
        f"{wettest_month['month_name']} "
        f"({wettest_month['monthly_rainfall_mm']:.2f} mm)."
    )

    if len(filtered_annual) >= 3:

        result = linregress(
            filtered_annual["year"],
            filtered_annual["annual_rainfall_mm"]
        )

        st.write(
            f"Regression slope: {result.slope:.2f} mm/year"
        )

        st.write(
            f"R-squared: {result.rvalue ** 2:.4f}"
        )

        st.write(
            f"P-value: {result.pvalue:.4f}"
        )

        if result.pvalue < 0.05:
            if result.slope > 0:
                st.success(
                    "The selected period shows a statistically "
                    "significant increasing linear trend."
                )
            elif result.slope < 0:
                st.warning(
                    "The selected period shows a statistically "
                    "significant decreasing linear trend."
                )
            else:
                st.info("The estimated linear slope is zero.")
        else:
            st.info(
                "No statistically significant linear trend was "
                "detected for the selected period at the 5% level."
            )


# -----------------------------------
# DATA TABLE
# -----------------------------------

st.header("8. Historical Rainfall Data")

st.dataframe(
    filtered_daily,
    use_container_width=True,
    hide_index=True
)

csv_data = filtered_daily.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Selected Rainfall Data",
    data=csv_data,
    file_name="bengaluru_rainfall_analysis.csv",
    mime="text/csv"
)


# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.caption(
    "Bengaluru Historical Rainfall Analysis | 2015–2025 | "
    "Statistical Analysis and Visualization"
)
