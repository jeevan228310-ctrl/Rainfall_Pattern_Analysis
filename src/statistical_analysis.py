
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "bengaluru_rainfall_cleaned.csv"
)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def perform_statistical_analysis():

    print("\n====================================")
    print(" BENGALURU RAINFALL STATISTICAL ANALYSIS")
    print("====================================")

    if not INPUT_FILE.exists():
        print("ERROR: Cleaned dataset not found.")
        return None

    # Load cleaned dataset
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df["rainfall_mm"] = pd.to_numeric(
        df["rainfall_mm"], errors="coerce"
    )

    # Validate rainfall observations
    rainfall = df["rainfall_mm"].dropna()

    if rainfall.empty:
        print("ERROR: No valid rainfall observations.")
        return None

    # --------------------------------
    # 1. DESCRIPTIVE STATISTICS
    # --------------------------------

    statistics = {
        "Count": rainfall.count(),
        "Mean": rainfall.mean(),
        "Median": rainfall.median(),
        "Mode": rainfall.mode().iloc[0],
        "Minimum": rainfall.min(),
        "Maximum": rainfall.max(),
        "Range": rainfall.max() - rainfall.min(),
        "Variance": rainfall.var(),
        "Standard Deviation": rainfall.std(),
        "25th Percentile": rainfall.quantile(0.25),
        "50th Percentile": rainfall.quantile(0.50),
        "75th Percentile": rainfall.quantile(0.75),
        "Skewness": rainfall.skew(),
        "Kurtosis": rainfall.kurt()
    }

    stats_df = pd.DataFrame(
        list(statistics.items()),
        columns=["Measurement", "Value"]
    )

    # --------------------------------
    # 2. ANNUAL RAINFALL ANALYSIS
    # --------------------------------

    annual = (
        df.groupby("year")["rainfall_mm"]
        .sum()
        .reset_index()
    )

    annual.columns = ["year", "annual_rainfall_mm"]

    # --------------------------------
    # 3. MONTHLY RAINFALL ANALYSIS
    # --------------------------------

    monthly = (
        df.groupby(["year", "month", "month_name"])
        ["rainfall_mm"]
        .sum()
        .reset_index()
    )

    monthly.columns = [
        "year", "month", "month_name", "monthly_rainfall_mm"
    ]

    monthly_average = (
        monthly.groupby(["month", "month_name"])
        ["monthly_rainfall_mm"]
        .mean()
        .reset_index()
        .sort_values("month")
    )

    monthly_average.columns = [
        "month", "month_name", "average_rainfall_mm"
    ]

    # --------------------------------
    # 4. SEASONAL RAINFALL ANALYSIS
    # --------------------------------

    seasonal = (
        df.groupby(["year", "season"])["rainfall_mm"]
        .sum()
        .reset_index()
    )

    seasonal.columns = [
        "year", "season", "seasonal_rainfall_mm"
    ]

    seasonal_average = (
        seasonal.groupby("season")["seasonal_rainfall_mm"]
        .mean()
        .reset_index()
    )

    seasonal_average.columns = [
        "season", "average_rainfall_mm"
    ]

    # --------------------------------
    # 5. CORRELATION AND COVARIANCE
    # --------------------------------

    covariance = annual["year"].cov(
        annual["annual_rainfall_mm"]
    )

    correlation = annual["year"].corr(
        annual["annual_rainfall_mm"]
    )

    # --------------------------------
    # 6. LINEAR REGRESSION
    # --------------------------------

    regression = linregress(
        annual["year"],
        annual["annual_rainfall_mm"]
    )

    slope = regression.slope
    intercept = regression.intercept
    r_squared = regression.rvalue ** 2
    p_value = regression.pvalue

    annual["trend_rainfall_mm"] = (
        slope * annual["year"] + intercept
    )

    # --------------------------------
    # 7. SAVE RESULTS
    # --------------------------------

    stats_df.to_csv(
        REPORT_DIR / "descriptive_statistics.csv",
        index=False
    )

    annual.to_csv(
        REPORT_DIR / "annual_rainfall.csv",
        index=False
    )

    monthly.to_csv(
        REPORT_DIR / "monthly_rainfall.csv",
        index=False
    )

    monthly_average.to_csv(
        REPORT_DIR / "monthly_average_rainfall.csv",
        index=False
    )

    seasonal.to_csv(
        REPORT_DIR / "seasonal_rainfall.csv",
        index=False
    )

    seasonal_average.to_csv(
        REPORT_DIR / "seasonal_average_rainfall.csv",
        index=False
    )

    # --------------------------------
    # 8. DISPLAY RESULTS
    # --------------------------------

    print("\nDESCRIPTIVE STATISTICS")
    print("----------------------")

    for name, value in statistics.items():
        print(f"{name}: {value:.4f}"
              if isinstance(value, (float, np.floating))
              else f"{name}: {value}")

    print("\nANNUAL RAINFALL")
    print("----------------")
    print(annual.to_string(index=False))

    print("\nMONTHLY AVERAGE RAINFALL")
    print("------------------------")
    print(monthly_average.to_string(index=False))

    print("\nSEASONAL AVERAGE RAINFALL")
    print("-------------------------")
    print(seasonal_average.to_string(index=False))

    print("\nTREND ANALYSIS")
    print("--------------")
    print(f"Covariance: {covariance:.4f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"Regression Slope: {slope:.4f} mm/year")
    print(f"R-squared: {r_squared:.4f}")
    print(f"P-value: {p_value:.4f}")

    if slope > 0:
        print("Trend: Increasing annual rainfall")
    elif slope < 0:
        print("Trend: Decreasing annual rainfall")
    else:
        print("Trend: No linear change")

    print("\nSTATISTICAL ANALYSIS COMPLETED")
    print("Results saved in:", REPORT_DIR)

    return statistics


if __name__ == "__main__":
    perform_statistical_analysis()
