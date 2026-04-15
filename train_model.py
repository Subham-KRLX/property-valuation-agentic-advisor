from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import kagglehub
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

ASSETS_DIR = Path("assets")
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "Housing.csv"
MODEL_FILE = MODELS_DIR / "house_model.pkl"
METADATA_FILE = ASSETS_DIR / "model_metadata.json"
FEATURE_IMPORTANCE_PLOT = ASSETS_DIR / "feature_importance.png"
PREDICTED_VS_ACTUAL_PLOT = ASSETS_DIR / "predicted_vs_actual.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2

BINARY_COLUMNS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
]

FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
]

TARGET = "price"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sns.set_theme(style="whitegrid")


def setup_directories() -> None:
    """Ensure required directories exist."""
    for folder in [ASSETS_DIR, MODELS_DIR, DATA_DIR]:
        folder.mkdir(exist_ok=True)


def load_data(file_path: Path) -> pd.DataFrame:
    """Download the dataset from Kaggle if needed and return it as a DataFrame."""
    if not file_path.exists():
        logging.info("Downloading dataset from Kaggle...")
        try:
            download_path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
            downloaded_file = Path(download_path) / "Housing.csv"

            if not downloaded_file.exists():
                raise RuntimeError("Downloaded Kaggle dataset does not contain 'Housing.csv'.")

            shutil.copy(downloaded_file, file_path)
            logging.info("Dataset saved to %s", file_path)
        except Exception as exc:
            logging.error(
                "Failed to download dataset from Kaggle. \n"
                "To resolve this, you can either:\n"
                "1. Provide Kaggle credentials: Create an API token at kaggle.com and place 'kaggle.json' in '~/.kaggle/'\n"
                "2. Manual Download: Download 'Housing.csv' from https://www.kaggle.com/datasets/yasserh/housing-prices-dataset and place it in the 'data/' directory.\n"
                "Original error: %s",
                exc,
            )
            raise RuntimeError("Data collection failed. Model cannot be trained without Housing.csv.") from exc

    df = pd.read_csv(file_path)

    required_raw_cols = set(FEATURES) | {TARGET}
    missing_cols = required_raw_cols - set(df.columns)
    if missing_cols:
        logging.error("The dataset is missing required columns: %s", missing_cols)
        raise RuntimeError(f"Incomplete dataset. Missing: {missing_cols}")

    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean the dataset and encode binary features for model training."""
    raw_row_count = len(df)
    processed_df = df.dropna().copy()

    for col in BINARY_COLUMNS:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].str.lower().map({"yes": 1, "no": 0})

    summary = {
        "raw_row_count": raw_row_count,
        "processed_row_count": len(processed_df),
        "dropped_row_count": raw_row_count - len(processed_df),
    }
    return processed_df, summary


def split_dataset(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a deterministic train/test split."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def calculate_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics as plain floats."""
    mse = mean_squared_error(y_true, predictions)
    return {
        "r2": float(r2_score(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mse)),
    }


def log_metrics(label: str, metrics: dict[str, float]) -> None:
    """Log a compact metrics summary for a trained model."""
    logging.info("--- %s Evaluation ---", label)
    logging.info("  R-squared (R2):       %.4f", metrics["r2"])
    logging.info("  Mean Absolute Error:  %.4f", metrics["mae"])
    logging.info("  RMSE:                 %.4f", metrics["rmse"])


def train_candidate_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, object]:
    """Train the candidate regression models used in the project."""
    models: dict[str, object] = {
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
        "linear_regression": LinearRegression(),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def evaluate_candidates(
    models: dict[str, object], X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, dict[str, object]]:
    """Evaluate all trained candidates and return their metrics and predictions."""
    evaluations: dict[str, dict[str, object]] = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)
        evaluations[name] = {
            "metrics": metrics,
            "predictions": predictions,
        }
        log_metrics(name.replace("_", " ").title(), metrics)
    return evaluations


def save_model(model: RandomForestRegressor, output_path: Path) -> None:
    """Serialize the trained model to disk."""
    joblib.dump(model, output_path)
    logging.info("Model successfully saved to %s", output_path)


def save_feature_importance_plot(feature_importance: dict[str, float], output_path: Path) -> None:
    """Create a horizontal feature importance chart for the selected model."""
    importance_df = (
        pd.DataFrame({"Feature": list(feature_importance.keys()), "Importance": list(feature_importance.values())})
        .sort_values(by="Importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="#1f77b4")
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved feature importance plot to %s", output_path)


def save_predicted_vs_actual_plot(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    """Create a predicted-vs-actual scatter plot with a perfect-fit reference line."""
    lower_bound = min(float(np.min(y_true)), float(np.min(y_pred)))
    upper_bound = max(float(np.max(y_true)), float(np.max(y_pred)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.75, color="#2ca02c", edgecolors="white", linewidth=0.5)
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], linestyle="--", color="#d62728", linewidth=2)
    ax.set_title("Predicted vs Actual Prices")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved predicted-vs-actual plot to %s", output_path)


def build_metadata(
    model: RandomForestRegressor,
    selected_metrics: dict[str, float],
    evaluations: dict[str, dict[str, object]],
    data_summary: dict[str, int],
    train_rows: int,
    test_rows: int,
) -> dict[str, object]:
    """Build the structured metadata payload saved to assets/model_metadata.json."""
    feature_importance = {
        feature: float(importance)
        for feature, importance in zip(FEATURES, model.feature_importances_.tolist())
    }

    benchmark_summary = {}
    for key, evaluation in evaluations.items():
        benchmark_summary[key] = {
            "label": key.replace("_", " ").title(),
            "metrics": evaluation["metrics"],
        }

    return {
        "metrics": selected_metrics,
        "feature_importance": feature_importance,
        "benchmark_summary": {
            "selected_model": "RandomForestRegressor",
            "selection_reason": (
                "Random Forest remains the production model because it captures non-linear feature interactions "
                "and exposes feature importance for the app insights tab."
            ),
            "candidates": benchmark_summary,
        },
        "model": {
            "name": type(model).__name__,
            "artifact_path": str(MODEL_FILE),
            "selected_model_key": "random_forest",
            "hyperparameters": {
                "n_estimators": model.n_estimators,
                "random_state": model.random_state,
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
                "min_samples_leaf": model.min_samples_leaf,
            },
        },
        "training": {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(DATA_FILE),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_row_count": train_rows,
            "test_row_count": test_rows,
            **data_summary,
        },
        "schema": {
            "features": FEATURES,
            "target": TARGET,
            "binary_columns": BINARY_COLUMNS,
            "feature_count": len(FEATURES),
        },
        "artifacts": {
            "predicted_vs_actual": str(PREDICTED_VS_ACTUAL_PLOT),
            "feature_importance": str(FEATURE_IMPORTANCE_PLOT),
        },
    }


def save_metadata(metadata: dict[str, object], output_path: Path) -> None:
    """Write model metadata to disk as JSON."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
    logging.info("Model metadata saved to %s", output_path)


def main() -> None:
    """Run the end-to-end training pipeline."""
    setup_directories()
    logging.info("Starting model training pipeline...")

    try:
        raw_data = load_data(DATA_FILE)
    except RuntimeError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    processed_data, data_summary = preprocess_data(raw_data)

    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in processed_data.columns]
    if missing_columns:
        logging.error("Missing required columns: %s", missing_columns)
        sys.exit(1)

    X = processed_data[FEATURES]
    y = processed_data[TARGET]

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    models = train_candidate_models(X_train, y_train)
    evaluations = evaluate_candidates(models, X_test, y_test)

    selected_model = models["random_forest"]
    selected_metrics = evaluations["random_forest"]["metrics"]

    save_model(selected_model, MODEL_FILE)
    save_feature_importance_plot(
        {
            feature: float(importance)
            for feature, importance in zip(FEATURES, selected_model.feature_importances_.tolist())
        },
        FEATURE_IMPORTANCE_PLOT,
    )
    save_predicted_vs_actual_plot(
        y_test,
        evaluations["random_forest"]["predictions"],
        PREDICTED_VS_ACTUAL_PLOT,
    )

    metadata = build_metadata(
        model=selected_model,
        selected_metrics=selected_metrics,
        evaluations=evaluations,
        data_summary=data_summary,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )
    save_metadata(metadata, METADATA_FILE)


if __name__ == "__main__":
    main()
