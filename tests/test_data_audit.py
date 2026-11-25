"""Synthetic tests for MP235 data audit utilities."""

import pandas as pd

from mp235.data_audit import count_duplicate_rows, profile_dataframe


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
