"""Data-quality anomaly detection utilities for MP235."""

import pandas as pd


SENTINEL_CANDIDATES = {-1, -999, 999, 9999}
UNIT_FACTORS = {10, 12, 60, 100, 1000}


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


def check_leaking_column(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target: str = "spoilage_pct",
) -> dict:
    """Identify training-only columns other than the expected target."""
    suspicious = [
        column for column in train.columns
        if column not in holdout.columns and column != target
    ]

    return {
        "present": bool(suspicious),
        "columns": suspicious,
    }


def check_unit_mixup(
    df: pd.DataFrame,
    group_column: str = "chamber",
) -> list:
    """Screen numeric features for chamber-level unit differences."""
    findings = []

    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        if column == "spoilage_pct":
            continue

        stats = df.groupby(group_column)[column].agg(
            ["min", "mean", "max"]
        )

        for group in stats.index:
            other_groups = stats.drop(index=group)

            for statistic in ["min", "mean", "max"]:
                value = stats.loc[group, statistic]
                comparison = other_groups[statistic].median()

                if pd.isna(value) or pd.isna(comparison):
                    continue

                if value == 0 or comparison == 0:
                    continue

                ratio = value / comparison

                for factor in UNIT_FACTORS:
                    if abs(ratio - factor) < 0.01 * factor:
                        findings.append({
                            "column": column,
                            "group_value": str(group),
                            "statistic": statistic,
                            "observed_ratio": float(ratio),
                            "candidate_factor": factor,
                        })

    return findings


def check_sentinel_value(df: pd.DataFrame) -> list:
    """Find repeated candidate sentinel values in numeric columns."""
    findings = []

    for column in df.select_dtypes(include="number").columns:
        counts = df[column].value_counts()

        for value, count in counts.items():
            if value in SENTINEL_CANDIDATES and count > 1:
                findings.append({
                    "column": column,
                    "value": float(value),
                    "count": int(count),
                })

    return findings


def check_duplicate_rows(df: pd.DataFrame) -> dict:
    """Return exact duplicate counts and affected rows."""
    return {
        "present": bool(df.duplicated().any()),
        "count": count_duplicate_rows(df),
        "rows_involved": int(df.duplicated(keep=False).sum()),
    }


def check_stuck_sensor(
    df: pd.DataFrame,
    date_column: str = "date",
    group_column: str = "chamber",
    min_days: int = 7,
) -> list:
    """Find identical numeric readings on consecutive calendar days.

    A day is considered only when all rows in that group have one
    identical, non-missing reading. Sentinel values are excluded.
    """
    findings = []

    numeric_columns = [
        column for column in df.select_dtypes(include="number").columns
        if column != "spoilage_pct"
    ]

    work = df.copy()
    work[date_column] = pd.to_datetime(
        work[date_column], errors="coerce"
    )

    for column in numeric_columns:
        for group, group_df in work.groupby(group_column):
            daily = group_df.groupby(date_column)[column].agg(
                lambda values: (
                    values.iloc[0]
                    if values.notna().all()
                    and values.nunique() == 1
                    else float("nan")
                )
            ).dropna()

            daily = daily[daily.isin(SENTINEL_CANDIDATES) == False]
            daily = daily.sort_index()

            if daily.empty:
                continue

            run_start = 0

            for index in range(1, len(daily) + 1):
                if index < len(daily):
                    same_value = daily.iloc[index] == daily.iloc[index - 1]
                    consecutive = (
                        daily.index[index] - daily.index[index - 1]
                    ).days == 1
                else:
                    same_value = False
                    consecutive = False

                if not (same_value and consecutive):
                    run = daily.iloc[run_start:index]

                    if len(run) >= min_days:
                        findings.append({
                            "column": column,
                            "group_value": str(group),
                            "value": float(run.iloc[0]),
                            "start_date": run.index[0].strftime("%Y-%m-%d"),
                            "end_date": run.index[-1].strftime("%Y-%m-%d"),
                            "days": len(run),
                        })

                    run_start = index

    return findings


def check_target_typos(
    df: pd.DataFrame,
    target: str = "spoilage_pct",
    threshold: float = 20,
) -> dict:
    """Identify target values above the expected target range."""
    if target not in df.columns:
        return {"present": False, "count": 0, "values": []}

    suspicious = df.loc[df[target] > threshold, target]

    return {
        "present": not suspicious.empty,
        "count": int(len(suspicious)),
        "values": suspicious.tolist(),
    }


def check_spelling_variants(
    train: pd.DataFrame,
    later: pd.DataFrame,
) -> list:
    """Find categorical spellings in training absent from later data."""
    findings = []

    categorical_columns = train.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        if column not in later.columns or column in {
            "record_id", "date"
        }:
            continue

        train_values = set(train[column].dropna().unique())
        later_values = set(later[column].dropna().unique())

        variants = train_values - later_values

        if variants:
            findings.append({
                "column": column,
                "training_only_values": sorted(map(str, variants)),
                "later_levels": len(later_values),
            })

    return findings
