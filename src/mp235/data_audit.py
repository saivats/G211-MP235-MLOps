"""Initial data-quality audit utilities for MP235."""

import pandas as pd


def count_duplicate_rows(df: pd.DataFrame) -> int:
    """Return the number of extra fully duplicated rows."""
    return int(df.duplicated().sum())


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Generate a basic data-quality profile without modifying the data."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": {
            str(column): int(count)
            for column, count in df.isnull().sum().items()
        },
        "duplicate_rows": count_duplicate_rows(df),
        "column_names": df.columns.tolist(),
        "data_types": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }
