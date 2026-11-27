"""Run from the root of a fresh review checkout with original files in data/raw."""
from hashlib import sha256
from pathlib import Path
import sqlite3

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

from mp235.cleaning import clean_data
from mp235.monitor import build_monitoring_report
from mp235.release import (compare_trials, register_candidate, export_champion, chamber_mae,
                           promotion, holdout_export)
from mp235.train import load_config, fit_and_evaluate, log_evaluation

def main():
    if any(Path(p).exists() for p in ("mlflow.db", "models/registry", "models/model.joblib",
                                      "predictions/holdout_predictions.csv")):
        raise FileExistsError("Use a fresh checkout to preserve existing runs and version history")
    raw = Path("data/raw")
    cfg = load_config("params.yaml")
    train_path = raw / "train.csv"
    digest = sha256(train_path.read_bytes()).hexdigest()
    if digest != "8d3101be66417af0cd357ecce71bd9cf37b10e35ec18cde06f6aa4026c1f1a0d":
        raise ValueError("Original train.csv hash mismatch")
    train = pd.read_csv(train_path)
    later = pd.read_csv(raw / "later.csv").sort_values("date", kind="stable")
    holdout = pd.read_csv(raw / "holdout_features.csv", dtype={"record_id": str})
    if len(later) != 300 or len(holdout) != 250:
        raise ValueError("Unexpected evaluation set size")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment = "G211-MP235"
    mlflow.set_experiment(experiment)
    client = mlflow.MlflowClient()
    name = "MP235-spoilage"
    client.create_registered_model(name)
    settings = cfg["experiments"]
    _, trees, best, v1 = compare_trials(train, settings["validation_start"], cfg,
                                        settings["tree_counts"], digest)
    features = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    clean = clean_data(train)
    v1.fit(clean[features], clean.spoilage_pct)
    first = register_candidate(client, name, v1, best)
    export_champion(client, name, first.version)
    group = chamber_mae(v1, later, features)
    reference = build_monitoring_report(clean, later)
    update, check = later.iloc[:210], later.iloc[210:]
    cfg["model"]["n_estimators"] = trees
    with mlflow.start_run(run_name="v2-retrain") as run:
        v2, new = fit_and_evaluate(pd.concat([clean, update]), check, cfg)
        log_evaluation(cfg, digest, new)
        second = register_candidate(client, name, v2, run.info.run_id)
    old = float(mean_absolute_error(check.spoilage_pct, v1.predict(check[features])))
    decision = promotion(old, new)
    winner = second if decision == "promote" else first
    export_champion(client, name, winner.version)
    assert client.get_model_version_by_alias(name, "champion").version == winner.version
    champion = mlflow.sklearn.load_model(f"models:/{name}@champion")
    dest = Path("predictions/holdout_predictions.csv")
    dest.parent.mkdir(exist_ok=True)
    holdout_export(champion, holdout, features, dest)
    path = Path("reports/findings.yaml")
    findings = yaml.safe_load(path.read_text())
    findings["tracking"].update(experiment_name=experiment,best_run_id=best,
                                registered_model_name=name,champion_version=int(winner.version))
    findings["monitoring"].update(psi=reference["psi"],most_drifted_feature=reference["most_drifted_feature"]["feature"],
                                  mae_by_group=group,worst_served_group=max(group,key=group.get),
                                  mae_v1_on_check=old,mae_v2_on_check=new,decision=decision)
    path.write_text(yaml.safe_dump(findings,sort_keys=False))
    with sqlite3.connect("mlflow.db") as db:
        assert db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0

if __name__ == "__main__":
    main()
