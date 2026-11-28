from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_config(path: str | Path) -> dict:
    """Read a YAML configuration file and return its top-level mapping."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Expected a YAML mapping at the top level.")
    return config


def build_regressor(config: dict) -> RandomForestRegressor:
    """Construct an unfitted random forest using the configured settings."""
    model = config["model"]
    return RandomForestRegressor(
        random_state=model["random_state"],
        n_estimators=model["n_estimators"],
    )


def split_by_date(frame: pd.DataFrame, validation_start) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition by date; keep tied records in their original relative order."""
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    cutoff = pd.Timestamp(validation_start)
    if dates.isna().any() or pd.isna(cutoff):
        raise ValueError("Dates and validation_start must be present.")

    ordered = frame.assign(date=dates).sort_values("date", kind="stable")
    early = ordered["date"] < cutoff
    train, valid = ordered.loc[early], ordered.loc[~early]
    if train.empty or valid.empty:
        raise ValueError("The cutoff must leave rows in both partitions.")
    return train, valid


def fit_and_evaluate(
    train: pd.DataFrame, valid: pd.DataFrame, config: dict
) -> tuple[Pipeline, float]:
    """Fit on the earlier partition and score the later partition in percentage points."""
    numeric = config["features"]["numeric"]
    categorical = config["features"]["categorical"]
    predictors = numeric + categorical
    category_steps = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", category_steps, categorical),
    ])
    pipeline = make_pipeline(preprocess, build_regressor(config))
    pipeline.fit(train[predictors], train["spoilage_pct"])
    predictions = pipeline.predict(valid[predictors])
    return pipeline, float(mean_absolute_error(valid["spoilage_pct"], predictions))


def log_evaluation(config: dict, data_sha256: str, validation_mae: float) -> None:
    """Record provenance, settings and validation error in the caller's active run."""
    if mlflow.active_run() is None:
        raise RuntimeError("Open an MLflow run before logging evaluation.")
    model = config["model"]
    mlflow.log_params({
        "random_state": model["random_state"],
        "n_estimators": model["n_estimators"],
        "data_sha256": data_sha256,
    })
    mlflow.log_metric("validation_mae", validation_mae)
