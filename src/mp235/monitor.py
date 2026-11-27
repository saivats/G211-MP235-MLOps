"""Population Stability Index (PSI) drift monitoring for MP235."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

NUMERIC_PREDICTORS: list[str] = [
    "storage_days",
    "temp_set_c",
    "humidity_pct",
    "power_cuts_hours",
    "bag_weight_kg",
    "door_openings_per_day",
]
NUM_BINS = 10


def _finite_values(values: pd.Series, name: str) -> np.ndarray:
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be numeric.") from exc
    array = numeric.to_numpy(dtype=float, na_value=np.nan)
    return array[np.isfinite(array)]


def _validate_num_bins(num_bins: int) -> None:
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError("num_bins must be an integer greater than or equal to 2.")


def compute_bin_edges(reference: pd.Series, num_bins: int = NUM_BINS) -> np.ndarray:
    """Derive exactly ``num_bins`` stable bins from the reference distribution.

    Quantile edges are used when they are distinct. Discrete or constant
    reference data use equal-width reference-derived edges instead.
    """
    _validate_num_bins(num_bins)
    values = _finite_values(reference, "reference")
    if values.size == 0:
        raise ValueError("reference must contain at least one finite numeric value.")

    quantile_edges = np.quantile(values, np.linspace(0.0, 1.0, num_bins + 1))
    inner_edges = np.unique(quantile_edges[1:-1])
    if inner_edges.size != num_bins - 1:
        lower = float(values.min())
        upper = float(values.max())
        if lower == upper:
            radius = max(abs(lower) * 1e-6, 1e-6)
            lower -= radius
            upper += radius
        inner_edges = np.linspace(lower, upper, num_bins + 1)[1:-1]

    return np.concatenate(([-np.inf], inner_edges, [np.inf]))


def calculate_feature_psi(
    reference: pd.Series,
    current: pd.Series,
    num_bins: int = NUM_BINS,
    epsilon: float = 1e-4,
) -> float:
    """Calculate PSI using reference-derived bins and additive count smoothing.

    Missing and non-finite observations are excluded from PSI counts; callers
    can inspect them in the accompanying distribution summary.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number.")

    ref_values = _finite_values(reference, "reference")
    cur_values = _finite_values(current, "current")
    if ref_values.size == 0:
        raise ValueError("reference must contain at least one finite numeric value.")
    if cur_values.size == 0:
        raise ValueError("current must contain at least one finite numeric value.")

    edges = compute_bin_edges(pd.Series(ref_values), num_bins=num_bins)
    ref_counts, _ = np.histogram(ref_values, bins=edges)
    cur_counts, _ = np.histogram(cur_values, bins=edges)

    ref_props = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * num_bins)
    cur_props = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * num_bins)
    psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    return max(0.0, psi)


def calculate_dataset_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str] | None = None,
    num_bins: int = NUM_BINS,
) -> dict[str, float]:
    """Calculate PSI for each requested predictor in two dataframes."""
    target_features = features if features is not None else NUMERIC_PREDICTORS
    if not target_features:
        raise ValueError("At least one feature must be selected for PSI.")

    results: dict[str, float] = {}
    for feature in target_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            raise KeyError(
                f"Feature '{feature}' must be present in both reference and current data."
            )
        results[feature] = calculate_feature_psi(
            reference_df[feature],
            current_df[feature],
            num_bins=num_bins,
        )
    return results


def find_most_drifted_feature(psi_by_feature: dict[str, float]) -> tuple[str, float]:
    """Return the feature with the highest PSI, preserving input order for ties."""
    if not psi_by_feature:
        raise ValueError("Cannot find most drifted feature from an empty PSI dictionary.")
    if any(not np.isfinite(value) or value < 0 for value in psi_by_feature.values()):
        raise ValueError("PSI values must be finite and non-negative.")
    return max(psi_by_feature.items(), key=lambda item: item[1])


def _distribution_summary(values: pd.Series) -> dict[str, float | int]:
    finite = _finite_values(values, "summary")
    if finite.size == 0:
        raise ValueError("Each predictor must contain at least one finite value.")

    return {
        "count": int(finite.size),
        "missing_or_non_finite_count": int(len(values) - finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "min": float(np.min(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def build_monitoring_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict[str, object]:
    """Build measured PSI and distribution summaries for the six predictors."""
    psi_by_feature = calculate_dataset_psi(
        reference_df, current_df, num_bins=NUM_BINS
    )
    most_drifted, max_psi = find_most_drifted_feature(psi_by_feature)
    return {
        "num_bins": NUM_BINS,
        "psi": psi_by_feature,
        "most_drifted_feature": {"feature": most_drifted, "psi": max_psi},
        "distributions": {
            feature: {
                "reference": _distribution_summary(reference_df[feature]),
                "current": _distribution_summary(current_df[feature]),
            }
            for feature in NUMERIC_PREDICTORS
        },
    }


def _load_and_clean_reference(path: Path) -> pd.DataFrame:
    raw_reference = pd.read_csv(path)
    try:
        from mp235.cleaning import clean_data
    except ImportError as exc:
        raise ImportError(
            "The project cleaning module mp235.cleaning is unavailable; "
            "integrate Pradyumna's cleaning pipeline before running PSI on raw train data."
        ) from exc
    return clean_data(raw_reference)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare cleaned train predictors with later data using 10-bin PSI."
    )
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--later", type=Path, default=Path("data/raw/later.csv"))
    args = parser.parse_args(argv)

    try:
        reference = _load_and_clean_reference(args.train)
        current = pd.read_csv(args.later)
        report = build_monitoring_report(reference, current)
    except (FileNotFoundError, ImportError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
