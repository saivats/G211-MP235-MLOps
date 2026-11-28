"""Automated unit and integration tests for structured logging and PSI monitoring.

Uses synthetic data fixtures so tests run deterministically in CI without
requiring private training CSVs.
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from mp235.monitor import (
    NUM_BINS,
    NUMERIC_PREDICTORS,
    build_monitoring_report,
    calculate_dataset_psi,
    calculate_feature_psi,
    compute_bin_edges,
    find_most_drifted_feature,
    main,
)
from mp235.structured_logging import get_prediction_logger, log_prediction


# --- PSI Tests ---


def test_psi_identical_distributions_near_zero() -> None:
    """Identical input values have zero PSI under reference-derived bins."""
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(loc=50.0, scale=10.0, size=2000))
    cur = ref.copy()

    psi = calculate_feature_psi(ref, cur, num_bins=10)
    assert psi == 0.0


def test_psi_shifted_distribution_positive() -> None:
    """A strongly shifted distribution must yield a significantly positive PSI."""
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(loc=50.0, scale=10.0, size=2000))
    cur = pd.Series(rng.normal(loc=75.0, scale=10.0, size=2000))

    psi = calculate_feature_psi(ref, cur, num_bins=10)
    assert isinstance(psi, float)
    assert psi > 0.25, f"Expected significant drift (PSI > 0.25), got {psi}"


def test_psi_empty_bins_and_extreme_values() -> None:
    """Values far outside the reference range must be captured without zero-division or errors."""
    ref = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0] * 50)
    # Current has extreme outliers and values in only one extreme tail
    cur = pd.Series([1000.0, 2000.0, 5000.0] * 50)

    psi = calculate_feature_psi(ref, cur, num_bins=10)
    assert np.isfinite(psi)
    assert psi >= 0.0


def test_psi_constant_feature_handling() -> None:
    """Constant reference sensor reading must be handled safely."""
    ref = pd.Series([4.0] * 100)
    cur_same = pd.Series([4.0] * 100)
    cur_shifted = pd.Series([4.0] * 50 + [99.0] * 50)

    psi_same = calculate_feature_psi(ref, cur_same, num_bins=10)
    assert psi_same == 0.0

    psi_shifted = calculate_feature_psi(ref, cur_shifted, num_bins=10)
    assert np.isfinite(psi_shifted)
    assert psi_shifted >= 0.0


def test_reference_bins_are_ten_for_discrete_and_constant_data() -> None:
    discrete_edges = compute_bin_edges(pd.Series([0, 0, 1, 1, 2, 2]), NUM_BINS)
    constant_edges = compute_bin_edges(pd.Series([4.0] * 20), NUM_BINS)

    assert len(discrete_edges) == NUM_BINS + 1
    assert len(constant_edges) == NUM_BINS + 1
    assert np.all(np.diff(discrete_edges) > 0)
    assert np.all(np.diff(constant_edges) > 0)


def test_empty_or_invalid_distributions_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="reference must contain"):
        calculate_feature_psi(pd.Series([np.nan, np.inf]), pd.Series([1.0]))

    with pytest.raises(ValueError, match="current must contain"):
        calculate_feature_psi(pd.Series([1.0]), pd.Series([np.nan, -np.inf]))


def test_psi_result_is_reproducible() -> None:
    rng = np.random.default_rng(2714)
    reference = pd.Series(rng.normal(10, 2, size=600))
    current = pd.Series(rng.normal(12, 3, size=300))

    first = calculate_feature_psi(reference, current)
    second = calculate_feature_psi(reference, current)

    assert first == second


def test_most_drifted_feature_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        find_most_drifted_feature({"storage_days": float("nan")})


def test_all_six_numeric_features_processed() -> None:
    """All six configured numeric predictors must be evaluated and identified correctly."""
    rng = np.random.default_rng(123)
    n = 500

    ref_data = {
        "storage_days": rng.integers(10, 250, size=n),
        "temp_set_c": rng.normal(4.0, 1.5, size=n),
        "humidity_pct": rng.uniform(70.0, 95.0, size=n),
        "power_cuts_hours": rng.exponential(2.0, size=n),
        "bag_weight_kg": rng.uniform(25.0, 55.0, size=n),
        "door_openings_per_day": rng.poisson(8, size=n),
    }
    cur_data = {
        "storage_days": rng.integers(10, 250, size=n),
        "temp_set_c": rng.normal(4.0, 1.5, size=n),
        "humidity_pct": rng.uniform(70.0, 95.0, size=n),
        # Artificially heavily drift power_cuts_hours
        "power_cuts_hours": rng.exponential(2.0, size=n) + 25.0,
        "bag_weight_kg": rng.uniform(25.0, 55.0, size=n),
        "door_openings_per_day": rng.poisson(8, size=n),
    }

    ref_df = pd.DataFrame(ref_data)
    cur_df = pd.DataFrame(cur_data)

    results = calculate_dataset_psi(ref_df, cur_df, features=NUMERIC_PREDICTORS, num_bins=10)

    assert set(results.keys()) == set(NUMERIC_PREDICTORS)
    for feat, psi_val in results.items():
        assert np.isfinite(psi_val)
        assert psi_val >= 0.0

    most_drifted_feat, max_psi = find_most_drifted_feature(results)
    assert most_drifted_feat == "power_cuts_hours"
    assert max_psi == results["power_cuts_hours"]


def test_monitoring_report_contains_measured_summaries_and_maximum() -> None:
    rng = np.random.default_rng(77)
    reference = pd.DataFrame({
        feature: rng.normal(0.0, 1.0, size=500)
        for feature in NUMERIC_PREDICTORS
    })
    current = reference.copy()
    current["humidity_pct"] += 8.0
    reference.loc[0, "storage_days"] = np.nan

    report = build_monitoring_report(reference, current)

    assert report["num_bins"] == NUM_BINS
    assert set(report["psi"]) == set(NUMERIC_PREDICTORS)
    assert report["most_drifted_feature"]["feature"] == "humidity_pct"
    assert report["most_drifted_feature"]["psi"] == report["psi"]["humidity_pct"]
    assert report["distributions"]["storage_days"]["reference"]["count"] == 499
    assert (
        report["distributions"]["storage_days"]["reference"]
        ["missing_or_non_finite_count"] == 1
    )


def test_cli_cleans_reference_and_prints_json_report(tmp_path, monkeypatch, capsys) -> None:
    rng = np.random.default_rng(9)
    train = pd.DataFrame({
        feature: rng.normal(0.0, 1.0, size=100)
        for feature in NUMERIC_PREDICTORS
    })
    later = train.copy()
    later["power_cuts_hours"] += 10.0
    train_path = tmp_path / "train.csv"
    later_path = tmp_path / "later.csv"
    train.to_csv(train_path, index=False)
    later.to_csv(later_path, index=False)

    cleaning_module = types.ModuleType("mp235.cleaning")
    cleaned = []

    def clean_data(frame):
        cleaned.append(True)
        return frame.copy()

    cleaning_module.clean_data = clean_data
    monkeypatch.setitem(sys.modules, "mp235.cleaning", cleaning_module)

    result = main(["--train", str(train_path), "--later", str(later_path)])
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert cleaned == [True]
    assert output["num_bins"] == NUM_BINS
    assert output["most_drifted_feature"]["feature"] == "power_cuts_hours"


# --- Structured Logging Tests ---


def test_structured_log_valid_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Log record sent to stdout must parse as valid single-line JSON with required schema."""
    logger = get_prediction_logger("test.logger.valid")
    log_prediction(
        logger=logger,
        prediction=12.45,
        duration_ms=4.82,
        model_version="1.0",
        event_name="prediction_success",
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().split("\n") if line.strip()]
    assert len(lines) >= 1

    payload = json.loads(lines[-1])
    assert "timestamp" in payload
    assert payload["severity"] == "INFO"
    assert payload["event"] == "prediction_success"
    assert payload["prediction"] == 12.45
    assert payload["prediction_duration_ms"] == 4.82
    assert payload["model_version"] == "1.0"


def test_structured_log_strips_sensitive_fields(capsys: pytest.CaptureFixture[str]) -> None:
    """Sensitive fields must be excluded from prediction logs."""
    logger = get_prediction_logger("test.logger.privacy")
    log_prediction(
        logger=logger,
        prediction=8.1,
        duration_ms=2.1,
        model_version="1.0",
        extra_fields={
            "chamber": "Chamber-A",
            "crop": "potato",
            "password": "SUPER_SECRET_PASSWORD",
            "customer_name": "John Doe",
            "token": "bearer-xyz-123",
            "email": "user@example.com",
        },
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().split("\n") if line.strip()]
    payload = json.loads(lines[-1])

    # Safe fields should remain
    assert payload.get("chamber") == "Chamber-A"
    assert payload.get("crop") == "potato"

    # Sensitive fields must be completely stripped
    assert "password" not in payload
    assert "customer_name" not in payload
    assert "token" not in payload
    assert "email" not in payload


def test_no_duplicate_handlers_installed(capsys: pytest.CaptureFixture[str]) -> None:
    """Repeated calls to get_prediction_logger should not install duplicate handlers."""
    logger1 = get_prediction_logger("test.logger.singleton")
    logger2 = get_prediction_logger("test.logger.singleton")

    assert logger1 is logger2
    assert len(logger1.handlers) == 1

    log_prediction(logger1, prediction=5.0, duration_ms=1.2)
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().split("\n") if line.strip()]
    assert len(lines) == 1, "Exactly one log line should be emitted per prediction"
