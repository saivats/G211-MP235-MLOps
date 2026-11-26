"""Deterministic data cleaning utilities for MP235."""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean MP235 data without modifying the original DataFrame."""

    cleaned = df.copy(deep=True)

    # 1. Remove exact duplicate rows
    cleaned = cleaned.drop_duplicates().copy()

    # 2. Replace impossible temperature sentinel
    cleaned["temp_set_c"] = cleaned["temp_set_c"].replace(
        -999, float("nan")
    )

    # 3. Correct target typos (values approximately 10 times too large)
    if "spoilage_pct" in cleaned.columns:
        mask = cleaned["spoilage_pct"] > 20
        cleaned.loc[mask, "spoilage_pct"] = (
            cleaned.loc[mask, "spoilage_pct"] / 10
        )

    # 4. Handle confirmed stuck humidity sensor periods
    if {"date", "chamber", "humidity_pct"}.issubset(cleaned.columns):
        dates = pd.to_datetime(cleaned["date"], errors="coerce")

        stuck_periods = {
            "Chamber 1": ("2026-04-26", "2026-05-18"),
            "Chamber 2": ("2026-05-03", "2026-05-14"),
            "Chamber 3": ("2026-04-26", "2026-05-03"),
        }

        for chamber, (start, end) in stuck_periods.items():
            mask = (
                cleaned["chamber"].eq(chamber)
                & cleaned["humidity_pct"].eq(96.0)
                & dates.between(start, end, inclusive="both")
            )
            cleaned.loc[mask, "humidity_pct"] = float("nan")

    # 5. Impute missing temperature using chamber-wise median
    chamber_temp_medians = (
        cleaned.groupby("chamber")["temp_set_c"].transform("median")
    )
    cleaned["temp_set_c"] = cleaned["temp_set_c"].fillna(
        chamber_temp_medians
    )

    # 6. Impute missing humidity using chamber-wise median
    if "humidity_pct" in cleaned.columns:
        chamber_humidity_medians = (
            cleaned.groupby("chamber")["humidity_pct"].transform("median")
        )
        cleaned["humidity_pct"] = cleaned["humidity_pct"].fillna(
            chamber_humidity_medians
        )

    return cleaned
