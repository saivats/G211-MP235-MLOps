"""Deterministic data cleaning utilities for MP235."""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a copy of the MP235 dataset.

    Operations:
    1. Remove exact duplicate rows.
    2. Replace the -999 temperature sentinel with missing values.
    3. Correct target values that are 10 times too large.
    4. Impute missing temperatures using chamber-wise medians.
    """
    cleaned = df.copy(deep=True)

    # 1. Remove exact duplicate rows
    cleaned = cleaned.drop_duplicates().copy()

    # 2. Replace impossible temperature sentinel
    cleaned["temp_set_c"] = cleaned["temp_set_c"].replace(-999, float("nan"))

    # 3. Correct target typos (extra zero / 10 times too large)
    if "spoilage_pct" in cleaned.columns:
        mask = cleaned["spoilage_pct"] > 20
        cleaned.loc[mask, "spoilage_pct"] = (
            cleaned.loc[mask, "spoilage_pct"] / 10
        )

    # 4. Impute missing temperature using chamber-wise median
    chamber_medians = cleaned.groupby("chamber")["temp_set_c"].transform("median")
    cleaned["temp_set_c"] = cleaned["temp_set_c"].fillna(chamber_medians)

    return cleaned
