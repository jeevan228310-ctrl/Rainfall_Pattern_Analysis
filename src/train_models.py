
from pathlib import Path
import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------


# PROJECT PATH CONFIGURATION

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "bengaluru_rainfall_cleaned.csv"
)

MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "bengaluru_rainfall_cleaned.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------


def load_data():

    print("Loading existing cleaned dataset...")

    df = pd.read_csv(DATA_PATH)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert rainfall_mm to precipitation_sum
    if "rainfall_mm" in df.columns:
        df = df.rename(
            columns={"rainfall_mm": "precipitation_sum"}
        )

    # Validate columns AFTER renaming
    if "date" not in df.columns or "precipitation_sum" not in df.columns:
        raise ValueError(
            f"Dataset must contain date and precipitation_sum columns. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    df["precipitation_sum"] = pd.to_numeric(
        df["precipitation_sum"], errors="coerce"
    )

    df = df.dropna(
        subset=["date", "precipitation_sum"]
    )

    df = df.sort_values("date").drop_duplicates("date")

    if (df["precipitation_sum"] < 0).any():
        raise ValueError("Rainfall data contains negative values.")

    print("Dataset loaded successfully!")
    print("Total records:", len(df))

    return df.reset_index(drop=True)



# --------------------------------------------------
# DAILY FEATURE ENGINEERING
# --------------------------------------------------

def create_daily_features(df):
    data = df[["date", "precipitation_sum"]].copy()
    data = data.rename(columns={"precipitation_sum": "rainfall"})

    day_of_year = data["date"].dt.dayofyear

    data["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    data["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    data["month_sin"] = np.sin(
        2 * np.pi * data["date"].dt.month / 12
    )
    data["month_cos"] = np.cos(
        2 * np.pi * data["date"].dt.month / 12
    )
    data["year_index"] = data["date"].dt.year - data["date"].dt.year.min()

    # All rainfall input features use previous days only.
    for lag in [1, 2, 3, 7, 14, 30]:
        data[f"lag_{lag}"] = data["rainfall"].shift(lag)

    previous_rainfall = data["rainfall"].shift(1)

    data["rolling_mean_7"] = previous_rainfall.rolling(7).mean()
    data["rolling_mean_30"] = previous_rainfall.rolling(30).mean()

    data = data.dropna().reset_index(drop=True)

    features = [
        "day_sin", "day_cos",
        "month_sin", "month_cos",
        "year_index",
        "lag_1", "lag_2", "lag_3",
        "lag_7", "lag_14", "lag_30",
        "rolling_mean_7", "rolling_mean_30"
    ]

    return data, features


# --------------------------------------------------
# MONTHLY FEATURE ENGINEERING
# --------------------------------------------------

def create_monthly_features(df):
    daily = df[["date", "precipitation_sum"]].copy()
    daily = daily.set_index("date")

    monthly = daily["precipitation_sum"].resample("MS").sum()
    data = monthly.rename("rainfall").reset_index()

    month = data["date"].dt.month

    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)
    data["year_index"] = data["date"].dt.year - data["date"].dt.year.min()

    # Previous months only: no current-month rainfall is used as input.
    for lag in [1, 2, 3, 6, 12]:
        data[f"lag_{lag}"] = data["rainfall"].shift(lag)

    previous_rainfall = data["rainfall"].shift(1)

    data["rolling_mean_3"] = previous_rainfall.rolling(3).mean()
    data["rolling_mean_6"] = previous_rainfall.rolling(6).mean()

    data = data.dropna().reset_index(drop=True)

    features = [
        "month_sin", "month_cos", "year_index",
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_mean_6"
    ]

    return data, features


# --------------------------------------------------
# MODEL CANDIDATES
# --------------------------------------------------

def get_models():
    return {
        "Mean Baseline": DummyRegressor(strategy="mean"),

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=2,
            loss="huber",
            random_state=42
        )
    }


# --------------------------------------------------
# TRAIN, SELECT AND TEST
# --------------------------------------------------

def train_one_target(data, features, target_name, model_filename):
    X = data[features]
    y = data["rainfall"]

    total = len(data)

    # Chronological split: first 64% train, next 16% validation,
    # final 20% test. No random shuffling.
    train_end = int(total * 0.64)
    validation_end = int(total * 0.80)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_validation = X.iloc[train_end:validation_end]
    y_validation = y.iloc[train_end:validation_end]

    X_train_validation = X.iloc[:validation_end]
    y_train_validation = y.iloc[:validation_end]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    if min(len(X_train), len(X_validation), len(X_test)) < 10:
        raise ValueError(
            f"Not enough observations for {target_name} model training."
        )

    print(f"\n{'=' * 55}")
    print(f"TRAINING {target_name.upper()} RAINFALL MODELS")
    print(f"{'=' * 55}")
    print(f"Total usable observations: {total}")
    print(f"Training observations:     {len(X_train)}")
    print(f"Validation observations:  {len(X_validation)}")
    print(f"Testing observations:     {len(X_test)}")

    # Select the model using validation MAE.
    validation_results = []

    for model_name, model in get_models().items():
        model.fit(X_train, y_train)

        predictions = np.maximum(model.predict(X_validation), 0)

        mae = mean_absolute_error(y_validation, predictions)

        validation_results.append({
            "model": model_name,
            "validation_mae": float(mae)
        })

        print(f"{model_name}: validation MAE = {mae:.3f} mm")

    validation_results.sort(key=lambda item: item["validation_mae"])
    best_name = validation_results[0]["model"]

    print(f"\nSelected model: {best_name}")

    # Refit the selected model using training + validation data.
    final_model = get_models()[best_name]
    final_model.fit(X_train_validation, y_train_validation)

    # Evaluate once on the untouched chronological test section.
    test_predictions = np.maximum(final_model.predict(X_test), 0)

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    r2 = r2_score(y_test, test_predictions)

    metrics = {
        "target": target_name,
        "selected_model": best_name,
        "observations": int(total),
        "training_observations": int(len(X_train)),
        "validation_observations": int(len(X_validation)),
        "test_observations": int(len(X_test)),
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_r2": float(r2),
        "validation_results": validation_results
    }

    # Save the trained model and its required feature information.
    model_path = MODEL_DIR / model_filename

    joblib.dump({
        "model": final_model,
        "features": features,
        "selected_model": best_name,
        "target": target_name
    }, model_path)

    # Save actual and predicted test values for the dashboard.
    predictions_df = pd.DataFrame({
        "date": data.loc[X_test.index, "date"].values,
        "actual_rainfall": y_test.values,
        "predicted_rainfall": test_predictions
    })

    prediction_file = REPORT_DIR / f"{target_name}_test_predictions.csv"
    predictions_df.to_csv(prediction_file, index=False)

    print("\nTEST RESULTS")
    print(f"MAE:  {mae:.3f} mm")
    print(f"RMSE: {rmse:.3f} mm")
    print(f"R²:   {r2:.4f}")
    print(f"Model saved: {model_path}")
    print(f"Test predictions saved: {prediction_file}")

    return metrics


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():
    print("BENGALURU RAINFALL MACHINE LEARNING")
    print("Loading existing cleaned dataset...")

    df = load_data()

    print(f"Daily records loaded: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    daily_data, daily_features = create_daily_features(df)
    monthly_data, monthly_features = create_monthly_features(df)

    daily_metrics = train_one_target(
        daily_data,
        daily_features,
        "daily",
        "daily_rainfall_model.pkl"
    )

    monthly_metrics = train_one_target(
        monthly_data,
        monthly_features,
        "monthly",
        "monthly_rainfall_model.pkl"
    )

    all_metrics = {
        "daily": daily_metrics,
        "monthly": monthly_metrics
    }

    metrics_file = REPORT_DIR / "model_evaluation.json"

    with open(metrics_file, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=4)

    print("\n" + "=" * 55)
    print("BOTH MACHINE-LEARNING MODELS TRAINED SUCCESSFULLY")
    print("=" * 55)
    print(f"Daily model:   {daily_metrics['selected_model']}")
    print(f"Monthly model: {monthly_metrics['selected_model']}")
    print(f"Evaluation report: {metrics_file}")


if __name__ == "__main__":
    main()
