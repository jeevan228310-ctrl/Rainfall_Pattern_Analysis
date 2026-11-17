
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# Bengaluru geographical coordinates
LATITUDE = 12.9716
LONGITUDE = 77.5946

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def collect_rainfall_data():

    print("\n====================================")
    print(" BENGALURU HISTORICAL RAINFALL DATA")
    print("====================================")

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata"
    }

    try:
        print("\nConnecting to Open-Meteo API...")

        response = requests.get(
            API_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        if "daily" not in data:
            raise ValueError("Rainfall data not available.")

        daily_data = data["daily"]

        df = pd.DataFrame({
            "date": daily_data["time"],
            "rainfall_mm": daily_data["precipitation_sum"]
        })

        # Data preprocessing
        df["date"] = pd.to_datetime(df["date"])

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.month_name()

        # Check duplicate dates
        duplicates = df["date"].duplicated().sum()

        # Sort records
        df = df.sort_values("date")

        # Save original observations, including missing values
        output_file = DATA_DIR / "bengaluru_rainfall_2015_2025.csv"

        df.to_csv(output_file, index=False)

        print("\nDATA COLLECTION SUCCESSFUL")

        print("\nDataset Information")
        print("-------------------")
        print("Location: Bengaluru")
        print("Latitude:", LATITUDE)
        print("Longitude:", LONGITUDE)
        print("Start Date:", START_DATE)
        print("End Date:", END_DATE)
        print("Total Records:", len(df))
        print("Duplicate Dates:", duplicates)
        print("Missing Rainfall Values:",
              df["rainfall_mm"].isna().sum())

        print("\nRainfall Summary")
        print("----------------")
        print(df["rainfall_mm"].describe())

        print("\nDataset saved at:")
        print(output_file)

        return df

    except requests.exceptions.RequestException as error:
        print("\nAPI CONNECTION ERROR:", error)

    except (ValueError, KeyError) as error:
        print("\nDATA ERROR:", error)

    except Exception as error:
        print("\nUNEXPECTED ERROR:", error)

    return None


if __name__ == "__main__":
    collect_rainfall_data()
