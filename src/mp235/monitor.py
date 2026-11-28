"""Population Stability Index (PSI) drift monitoring foundation for MP235.

Computes 10-bin decile PSI between reference (train.csv) and current (later.csv)
distributions for the required six numeric predictors. Handles edge cases
including empty bins, duplicated quantiles, extreme values, and constant sensors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The six official numeric predictors for MP235
NUMERIC_PREDICTORS: list[str] = [
    "storage_days",
    "temp_set_c",
    "humidity_pct",
    "power_cuts_hours",
    "bag_weight_kg",
    "door_openings_per_day",
]


def compute_bin_edges(reference: pd.Series, num_bins: int = 10) -> np.ndarray:
    """Derives bin edges from reference distribution using quantiles.

    Guarantees:
    - Deduplicates identical quantiles (e.g. repeated values/zeros).
    - Extends outer bounds to [-inf, +inf] so extreme values are not dropped.
    - Gracefully handles constant features.
    """
    clean_ref = reference.dropna()
    if len(clean_ref) == 0:
        return np.array([-np.inf, np.inf])

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    raw_edges = np.percentile(clean_ref, quantiles * 100)
    unique_edges = np.unique(raw_edges)

    if len(unique_edges) <= 1:
        val = unique_edges[0] if len(unique_edges) == 1 else 0.0
        return np.array([-np.inf, val, np.inf])

    # Extend outer bounds to [-inf, +inf]
    unique_edges[0] = -np.inf
    unique_edges[-1] = np.inf
    return unique_edges


def calculate_feature_psi(
    reference: pd.Series,
    current: pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculates Population Stability Index (PSI) for a single numeric feature.

    Formula:
        PSI = sum((Actual_i - Expected_i) * ln(Actual_i / Expected_i))

    Guarantees:
    - 10-bin decile edges derived from reference.
    - Outer bounds [-inf, +inf] ensure no extreme/unseen values are omitted.
    - Epsilon Laplace smoothing prevents zero divisions and log(0) on empty bins.
    - Returns a finite, non-negative float.
    """
    ref_clean = reference.dropna()
    cur_clean = current.dropna()

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    # If both distributions are constant and identical
    if ref_clean.nunique() <= 1 and cur_clean.nunique() <= 1:
        if ref_clean.iloc[0] == cur_clean.iloc[0]:
            return 0.0

    edges = compute_bin_edges(ref_clean, num_bins=num_bins)

    ref_counts, _ = np.histogram(ref_clean, bins=edges)
    cur_counts, _ = np.histogram(cur_clean, bins=edges)

    ref_total = len(ref_clean)
    cur_total = len(cur_clean)

    # Convert to proportions with smoothing to safely handle empty bins
    ref_props = (ref_counts + epsilon) / (ref_total + epsilon * len(ref_counts))
    cur_props = (cur_counts + epsilon) / (cur_total + epsilon * len(cur_counts))

    psi_terms = (cur_props - ref_props) * np.log(cur_props / ref_props)
    total_psi = float(np.sum(psi_terms))

    if np.isnan(total_psi) or np.isneginf(total_psi):
        return 0.0
    return max(0.0, total_psi)


def calculate_dataset_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str] | None = None,
    num_bins: int = 10,
) -> dict[str, float]:
    """Calculates PSI for all specified numeric features across two DataFrames.

    Defaults to the six official MP235 numeric predictors.
    """
    target_features = features if features is not None else NUMERIC_PREDICTORS
    results: dict[str, float] = {}

    for feat in target_features:
        if feat in reference_df.columns and feat in current_df.columns:
            results[feat] = calculate_feature_psi(
                reference=reference_df[feat],
                current=current_df[feat],
                num_bins=num_bins,
            )
        else:
            raise KeyError(
                f"Feature '{feat}' must be present in both reference and current DataFrames."
            )

    return results


def find_most_drifted_feature(psi_by_feature: dict[str, float]) -> tuple[str, float]:
    """Returns the feature name and PSI value of the highest drifted feature."""
    if not psi_by_feature:
        raise ValueError("Cannot find most drifted feature from an empty PSI dictionary.")
    return max(psi_by_feature.items(), key=lambda item: item[1])
