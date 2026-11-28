"""Deterministic data cleaning utilities for MP235."""

import pandas as pd


def clean_data(df: pd.DataFrame, *, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clean historical rows. For validation, pass the cleaned earlier rows as reference."""

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

        mask = dates.between("2026-04-26", "2026-05-18") & cleaned["humidity_pct"].eq(96)
        cleaned.loc[mask, "humidity_pct"] = float("nan")

    source = cleaned if reference is None else reference
    for col in ("temp_set_c", "humidity_pct"):
        if col in cleaned:
            medians = source.groupby("chamber")[col].median()
            cleaned[col] = cleaned[col].fillna(cleaned["chamber"].map(medians))
    return cleaned
