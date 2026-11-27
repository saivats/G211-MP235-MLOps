"""Selection, registration, promotion and export for the MP235 pipeline."""
from copy import deepcopy
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_absolute_error

from mp235.cleaning import clean_data
from mp235.train import fit_and_evaluate, log_evaluation, split_by_date


def compare_trials(frame, cutoff, config, tree_counts, digest):
    early, valid = split_by_date(frame, cutoff)
    early = clean_data(early)
    valid = clean_data(valid, reference=early)
    trials = []
    for count in tree_counts:
        settings = deepcopy(config)
        settings["model"]["n_estimators"] = count
        with mlflow.start_run() as run:
            pipeline, mae = fit_and_evaluate(early, valid, settings)
            log_evaluation(settings, digest, mae)
            trials.append((mae, count, run.info.run_id, pipeline))
    return min(trials, key=lambda trial: trial[:2])


def register_candidate(client, name, pipeline, run_id):
    # Paths are relative to the repository root, including after a fresh clone.
    source = Path("models/registry") / run_id
    mlflow.sklearn.save_model(pipeline, source, serialization_format="cloudpickle",
                              pip_requirements=Path("requirements.txt").read_text().splitlines())
    return client.create_model_version(name, source.as_posix(), run_id=run_id)


def export_champion(client, name, version):
    model = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    destination = Path("models/model.joblib")
    pending = destination.with_suffix(".tmp")
    joblib.dump(model, pending)
    client.set_registered_model_alias(name, "champion", version)
    pending.replace(destination)


def chamber_mae(model, later, features):
    return {chamber: float(mean_absolute_error(rows.spoilage_pct, model.predict(rows[features])))
            for chamber, rows in later.groupby("chamber")}


def promotion(v1_mae, v2_mae):
    if not np.isfinite([v1_mae, v2_mae]).all() or min(v1_mae, v2_mae) < 0:
        raise ValueError("MAEs must be finite and nonnegative")
    return "promote" if v1_mae > 0 and v2_mae <= 0.92 * v1_mae else "keep"


def holdout_export(model, rows, features, destination):
    ids = rows["record_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Expected distinct, nonmissing holdout IDs")
    predictions = np.asarray(model.predict(rows[features]), dtype=float)
    if predictions.shape != (len(rows),) or not np.isfinite(predictions).all():
        raise ValueError("Expected one finite prediction per record")
    pd.DataFrame({"record_id": ids, "prediction": predictions}).to_csv(destination, index=False)
