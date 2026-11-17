
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

FIGURE_DIR = REPORT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Graph styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11


def save_graph(filename):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300,
                bbox_inches="tight")
    plt.close()
    print("Generated:", filename)


def generate_visualizations():

    print("\n====================================")
    print(" BENGALURU RAINFALL VISUALIZATION")
    print("====================================")

    # Load datasets
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

    seasonal_avg = pd.read_csv(
        REPORT_DIR / "seasonal_average_rainfall.csv"
    )

    daily["date"] = pd.to_datetime(daily["date"])

    # 1. Annual rainfall trend
    plt.figure()

    sns.lineplot(
        data=annual,
        x="year",
        y="annual_rainfall_mm",
        marker="o",
        linewidth=2,
        label="Actual rainfall"
    )

    sns.lineplot(
        data=annual,
        x="year",
        y="trend_rainfall_mm",
        linestyle="--",
        label="Linear regression trend"
    )

    plt.title("Annual Rainfall Trend in Bengaluru (2015-2025)")
    plt.xlabel("Year")
    plt.ylabel("Rainfall (mm)")
    plt.xticks(annual["year"], rotation=45)
    plt.legend()

    save_graph("01_annual_rainfall_trend.png")

    # 2. Annual rainfall comparison
    plt.figure()

    sns.barplot(
        data=annual,
        x="year",
        y="annual_rainfall_mm"
    )

    plt.title("Year-wise Rainfall Comparison")
    plt.xlabel("Year")
    plt.ylabel("Total Rainfall (mm)")
    plt.xticks(rotation=45)

    save_graph("02_annual_rainfall_comparison.png")

    # 3. Monthly average rainfall
    plt.figure()

    sns.barplot(
        data=monthly_avg,
        x="month_name",
        y="average_rainfall_mm"
    )

    plt.title("Average Monthly Rainfall in Bengaluru")
    plt.xlabel("Month")
    plt.ylabel("Average Rainfall (mm)")
    plt.xticks(rotation=45)

    save_graph("03_monthly_average_rainfall.png")

    # 4. Seasonal rainfall comparison
    season_order = [
        "Winter",
        "Summer",
        "Southwest Monsoon",
        "Northeast Monsoon"
    ]

    plt.figure()

    sns.barplot(
        data=seasonal_avg,
        x="season",
        y="average_rainfall_mm",
        order=season_order
    )

    plt.title("Average Seasonal Rainfall")
    plt.xlabel("Season")
    plt.ylabel("Average Rainfall (mm)")
    plt.xticks(rotation=15)

    save_graph("04_seasonal_rainfall.png")

    # 5. Monthly rainfall heatmap
    monthly_pivot = monthly.pivot(
        index="year",
        columns="month",
        values="monthly_rainfall_mm"
    )

    plt.figure(figsize=(14, 7))

    sns.heatmap(
        monthly_pivot,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Rainfall (mm)"}
    )

    plt.title("Monthly Rainfall Heatmap (2015-2025)")
    plt.xlabel("Month")
    plt.ylabel("Year")

    save_graph("05_monthly_rainfall_heatmap.png")

    # 6. Daily rainfall distribution
    plt.figure()

    sns.histplot(
        data=daily,
        x="rainfall_mm",
        bins=50,
        kde=True
    )

    plt.title("Distribution of Daily Rainfall")
    plt.xlabel("Daily Rainfall (mm)")
    plt.ylabel("Number of Days")

    save_graph("06_daily_rainfall_distribution.png")

    # 7. Seasonal rainfall distribution
    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=daily,
        x="season",
        y="rainfall_mm",
        order=season_order
    )

    plt.title("Daily Rainfall Distribution by Season")
    plt.xlabel("Season")
    plt.ylabel("Daily Rainfall (mm)")
    plt.xticks(rotation=15)

    save_graph("07_seasonal_rainfall_boxplot.png")

    print("\nALL GRAPHS GENERATED SUCCESSFULLY!")
    print("Graph location:", FIGURE_DIR)


if __name__ == "__main__":
    generate_visualizations()
