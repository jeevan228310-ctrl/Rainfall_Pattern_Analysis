
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "bengaluru_rainfall_2015_2025.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "bengaluru_rainfall_cleaned.csv"


def preprocess_data():

    print("\n====================================")
    print(" BENGALURU RAINFALL PREPROCESSING")
    print("====================================")

    # Check whether dataset exists
    if not RAW_FILE.exists():
        print("ERROR: Raw rainfall dataset not found.")
        return None

    # Load dataset
    df = pd.read_csv(RAW_FILE)

    print("\nOriginal Dataset Shape:", df.shape)

    # Check required columns
    required_columns = ["date", "rainfall_mm"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Convert columns into correct data types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rainfall_mm"] = pd.to_numeric(
        df["rainfall_mm"], errors="coerce"
    )

    # Sort records chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Check data quality
    missing_dates = df["date"].isna().sum()
    missing_rainfall = df["rainfall_mm"].isna().sum()

    duplicate_dates = df["date"].duplicated().sum()

    negative_rainfall = (
        df["rainfall_mm"].dropna() < 0
    ).sum()

    # Check calendar coverage
    expected_dates = pd.date_range(
        start="2015-01-01",
        end="2025-12-31",
        freq="D"
    )

    actual_dates = pd.DatetimeIndex(
        df["date"].dropna().unique()
    )

    missing_calendar_days = len(
        expected_dates.difference(actual_dates)
    )

    # Check invalid observations
    if missing_dates > 0:
        raise ValueError("Invalid dates found in dataset.")

    if duplicate_dates > 0:
        raise ValueError("Duplicate dates found in dataset.")

    if negative_rainfall > 0:
        raise ValueError("Negative rainfall values found.")

    # Add date-related features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()

    df["season"] = df["month"].map({
        1: "Winter",
        2: "Winter",
        3: "Summer",
        4: "Summer",
        5: "Summer",
        6: "Southwest Monsoon",
        7: "Southwest Monsoon",
        8: "Southwest Monsoon",
        9: "Southwest Monsoon",
        10: "Northeast Monsoon",
        11: "Northeast Monsoon",
        12: "Winter"
    })

    # Preserve missing rainfall observations.
    # Do not replace missing values with zero.
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDATA QUALITY REPORT")
    print("-------------------")

    print("Total Records:", len(df))
    print("Missing Dates:", missing_dates)
    print("Missing Rainfall:", missing_rainfall)
    print("Duplicate Dates:", duplicate_dates)
    print("Negative Rainfall:", negative_rainfall)
    print("Missing Calendar Days:", missing_calendar_days)

    print("\nPREPROCESSING COMPLETED")

    print("\nProcessed Dataset:")
    print(OUTPUT_FILE)

    print("\nFinal Dataset Preview:")
    print(df.head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    preprocess_data()
