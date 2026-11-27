"""Synthetic tests for MP235 data audit utilities."""

import pandas as pd

from mp235.data_audit import (
    count_duplicate_rows,
    profile_dataframe,
    check_leaking_column,
    check_unit_mixup,
    check_sentinel_value,
    check_duplicate_rows,
    check_stuck_sensor,
    check_target_typos,
    check_spelling_variants,
)


def test_duplicate_row_count():
    df = pd.DataFrame({
        "crop": ["potato", "onion", "potato"],
        "temperature": [5, 8, 5],
    })
    assert count_duplicate_rows(df) == 1


def test_profile_dataframe():
    df = pd.DataFrame({
        "crop": ["potato", "onion"],
        "temperature": [5.0, None],
    })
    result = profile_dataframe(df)
    assert result["rows"] == 2
    assert result["columns"] == 2
    assert result["missing_values"]["temperature"] == 1
    assert result["duplicate_rows"] == 0


def test_original_dataframe_is_not_modified():
    df = pd.DataFrame({"crop": ["potato", "onion"]})
    original = df.copy(deep=True)
    profile_dataframe(df)
    pd.testing.assert_frame_equal(df, original)


def test_check_leaking_column():
    train = pd.DataFrame({
        "temperature": [5, 6],
        "spoilage_pct": [3, 4],
        "post_event_score": [3, 4],
    })
    holdout = pd.DataFrame({"temperature": [5, 6]})
    result = check_leaking_column(train, holdout)
    assert result["present"] is True
    assert "post_event_score" in result["columns"]


def test_check_sentinel_value():
    df = pd.DataFrame({"temp_set_c": [5, -999, -999, 6]})
    result = check_sentinel_value(df)
    assert any(
        item["column"] == "temp_set_c"
        and item["value"] == -999
        and item["count"] == 2
        for item in result
    )


def test_check_duplicate_rows():
    df = pd.DataFrame({"value": [1, 2, 2]})
    result = check_duplicate_rows(df)
    assert result["present"] is True
    assert result["count"] == 1


def test_check_target_typos():
    df = pd.DataFrame({"spoilage_pct": [3.5, 4.2, 86.8]})
    result = check_target_typos(df)
    assert result["present"] is True
    assert result["count"] == 1


def test_check_spelling_variants():
    train = pd.DataFrame({"crop": ["apple", "Apple", "potato"]})
    later = pd.DataFrame({"crop": ["apple", "potato"]})
    result = check_spelling_variants(train, later)
    assert len(result) == 1
    assert "Apple" in result[0]["training_only_values"]


def test_check_stuck_sensor():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=8).strftime("%Y-%m-%d"),
        "chamber": ["Chamber 1"] * 8,
        "temp_set_c": [5.0] * 8,
    })
    result = check_stuck_sensor(df, min_days=7)
    assert len(result) == 1
    assert result[0]["days"] == 8


def test_check_unit_mixup():
    df = pd.DataFrame({
        "chamber": ["Chamber 1"] * 3 + ["Chamber 2"] * 3,
        "storage_days": [10, 20, 30, 100, 200, 300],
    })
    result = check_unit_mixup(df)
    assert any(
        item["column"] == "storage_days"
        and item["group_value"] == "Chamber 2"
        and item["candidate_factor"] == 10
        for item in result
    )
