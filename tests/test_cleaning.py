"""Synthetic tests for MP235 data cleaning."""

import pandas as pd
from mp235.cleaning import clean_data


def test_cleaning_removes_duplicates():
    df = pd.DataFrame({
        "chamber": ["Chamber 1", "Chamber 1"],
        "temp_set_c": [5.0, 5.0],
        "spoilage_pct": [3.0, 3.0],
    })

    cleaned = clean_data(df)

    assert len(cleaned) == 1


def test_cleaning_replaces_sentinel():
    df = pd.DataFrame({
        "chamber": ["Chamber 1", "Chamber 1", "Chamber 2"],
        "temp_set_c": [4.0, -999, 8.0],
        "spoilage_pct": [3.0, 4.0, 5.0],
    })

    cleaned = clean_data(df)

    assert -999 not in cleaned["temp_set_c"].values
    assert cleaned.loc[1, "temp_set_c"] == 4.0


def test_cleaning_corrects_target_typo():
    df = pd.DataFrame({
        "chamber": ["Chamber 1", "Chamber 1"],
        "temp_set_c": [4.0, 5.0],
        "spoilage_pct": [3.5, 86.8],
    })

    cleaned = clean_data(df)

    assert cleaned["spoilage_pct"].tolist() == [3.5, 8.68]


def test_cleaning_does_not_modify_original():
    df = pd.DataFrame({
        "chamber": ["Chamber 1", "Chamber 1"],
        "temp_set_c": [4.0, -999],
        "spoilage_pct": [3.0, 86.8],
    })

    original = df.copy(deep=True)
    clean_data(df)

    pd.testing.assert_frame_equal(df, original)


def test_cleaning_handles_stuck_humidity_sensor():
    import pandas as pd
    from mp235.cleaning import clean_data

    df = pd.DataFrame({
        "date": [
            "2026-04-25",
            "2026-04-26",
            "2026-05-18",
            "2026-05-19",
        ],
        "chamber": ["Chamber 1"] * 4,
        "humidity_pct": [80.0, 96.0, 96.0, 85.0],
        "temp_set_c": [5.0, 5.0, 5.0, 5.0],
    })

    cleaned = clean_data(df)

    # Legitimate readings outside the stuck period remain unchanged.
    assert cleaned.loc[0, "humidity_pct"] == 80.0
    assert cleaned.loc[3, "humidity_pct"] == 85.0

    # Stuck readings are replaced by the chamber median.
    assert cleaned.loc[1, "humidity_pct"] == 82.5
    assert cleaned.loc[2, "humidity_pct"] == 82.5
