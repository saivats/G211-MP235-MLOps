from hashlib import sha256
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from mp235 import train as subject
from mp235.train import build_regressor, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_estimator_uses_project_settings():
    config = load_config(PROJECT_ROOT / "params.yaml")
    regressor = build_regressor(config)
    assert isinstance(regressor, RandomForestRegressor)
    assert regressor.get_params()["random_state"] == 42
    assert regressor.get_params()["n_estimators"] == 8


def test_estimator_uses_changed_yaml_settings(tmp_path):
    config = load_config(PROJECT_ROOT / "params.yaml")
    config["model"].update(random_state=17, n_estimators=11)
    alternate_path = tmp_path / "alternate.yaml"
    alternate_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    regressor = build_regressor(load_config(alternate_path))
    assert regressor.get_params()["random_state"] == 17
    assert regressor.get_params()["n_estimators"] == 11


def test_empty_config_has_a_clear_error(tmp_path):
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="top level"):
        load_config(empty_path)


@pytest.fixture
def config():
    return subject.load_config(PROJECT_ROOT / "params.yaml")


@pytest.fixture
def lots(config):
    frame = pd.DataFrame({
        name: [99., 1., 3., 101., np.nan, 5.]
        for name in config["features"]["numeric"]
    })
    frame["crop"] = ["apple", "potato", "onion", "apple", np.nan, "potato"]
    frame["chamber"] = ["C", "A", "A", "C", "B", "A"]
    frame["date"] = ["2026-01-03", "2026-01-01", "2026-01-02",
                     "2026-01-03", "2026-01-02", "2026-01-02"]
    frame["spoilage_pct"] = [9., 1., 3., 10., 2., 5.]
    frame["record_id"] = list("abcdef")
    frame["post_event"] = frame["spoilage_pct"]
    return frame


def test_split_keeps_boundary_dates_together(lots):
    early, late = subject.split_by_date(lots.iloc[[2, 3, 1, 4, 0, 5]], "2026-01-03")
    assert list(early.record_id) == list("bcef")
    assert early.date.is_monotonic_increasing
    assert late.date.is_monotonic_increasing
    assert list(late.record_id) == list("da")
    assert pd.to_datetime(early.date).max() < pd.to_datetime(late.date).min()


@pytest.mark.parametrize("bad_date", [None, "not-a-date"])
def test_reject_bad_dates(lots, bad_date):
    lots.loc[0, "date"] = bad_date
    with pytest.raises(ValueError):
        subject.split_by_date(lots, "2026-01-03")


@pytest.mark.parametrize("cutoff", ["2026-01-01", "2026-01-04"])
def test_reject_empty_partition(lots, cutoff):
    with pytest.raises(ValueError):
        subject.split_by_date(lots, cutoff)


def test_fit_learns_from_earlier_rows_only(lots, config, tmp_path):
    # Construct the partitions independently so a split defect cannot mask a fit defect.
    train = lots.iloc[[1, 2, 4, 5]].copy()
    valid = lots.iloc[[0, 3]].copy()
    pipeline, mae = subject.fit_and_evaluate(train, valid, config)
    cols = config["features"]["numeric"] + config["features"]["categorical"]
    assert set(pipeline.feature_names_in_) == set(cols)
    pred = pipeline.predict(valid[cols])
    assert np.isfinite(pred).all()
    assert mae == pytest.approx(mean_absolute_error(valid.spoilage_pct, pred))
    # Numeric transformer named "numeric" is a median SimpleImputer.
    prep = pipeline.steps[0][1]
    np.testing.assert_allclose(prep.named_transformers_["numeric"].statistics_, [3.] * 6)
    # Compare the encoded blocks for two categorical levels seen only in validation.
    changed = valid[cols].assign(crop="unseen-for-comparison", chamber="unknown")
    original = prep.transform(valid[cols])
    other = prep.transform(changed)
    if hasattr(original, "toarray"):
        original, other = original.toarray(), other.toarray()
    np.testing.assert_allclose(original, other)
    refit, _ = subject.fit_and_evaluate(train, valid, config)
    np.testing.assert_array_equal(pred, refit.predict(valid[cols]))
    path = tmp_path / "synthetic.joblib"
    joblib.dump(pipeline, path)
    np.testing.assert_array_equal(pred, joblib.load(path).predict(valid[cols]))


def test_tracking_readback(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous_uri = mlflow.get_tracking_uri()
    digest = sha256(b"synthetic teaching fixture").hexdigest()
    try:
        mlflow.set_tracking_uri("sqlite:///exercise.db")
        mlflow.set_experiment("exercise")
        with pytest.raises(RuntimeError, match="Open an MLflow run"):
            subject.log_evaluation(config, digest, 2.5)
        assert mlflow.active_run() is None
        with mlflow.start_run(tags={"data_kind": "synthetic"}) as run:
            subject.log_evaluation(config, digest, 2.5)
            assert mlflow.active_run().info.run_id == run.info.run_id
        stored = mlflow.MlflowClient().get_run(run.info.run_id)
        assert stored.info.status == "FINISHED"
        assert stored.data.params["data_sha256"] == digest
        assert stored.data.params["random_state"] == "42"
        assert stored.data.params["n_estimators"] == "8"
        assert stored.data.metrics["validation_mae"] == 2.5
    finally:
        mlflow.set_tracking_uri(previous_uri)
