# MP235 Data Quality Audit Report

**Group:** G211  
**Problem ID:** MP235  
**Role:** Data Engineering & DVC Lead

## 1. Dataset Overview

The project uses three datasets:

| Dataset | Original rows | Columns |
|---|---:|---:|
| train.csv | 1,027 | 11 |
| later.csv | 300 | 11 |
| holdout_features.csv | 250 | 10 |

The training dataset contains the target column `spoilage_pct`.
Later also contains spoilage_pct. Holdout contains only features and identifying metadata.

The original raw CSV files are preserved in `data/raw/`.

## 2. Data Quality Findings

Seven anomaly categories were investigated according to the MP235 Student Handbook.

### 2.1 Leaking Column

**Status:** Not observed.

No suspicious post-event feature was identified through the train and holdout column comparison. The target column `spoilage_pct` is expected to be present in training and later data.

### 2.2 Unit Mix-up

**Status:** Not observed.

Feature ranges were compared across Chamber 1, Chamber 2 and Chamber 3. No clear systematic unit-conversion pattern was identified.

### 2.3 Sentinel Value

**Status:** Confirmed.

- Column: `temp_set_c`
- Sentinel value: `-999`
- Occurrences: 36

The impossible temperature value appeared 12 times in each chamber.

### 2.4 Duplicate Rows

**Status:** Confirmed.

- Original training rows: 1,027
- Extra duplicate rows: 27
- Rows involved in duplicate groups: 54

Exact duplicates were identified using `DataFrame.duplicated()`.

### 2.5 Stuck Sensor

A stuck sensor was detected in the `humidity_pct` column.

Three continuous runs of identical humidity readings (96.0) were identified:

| Chamber | Value | Start Date | End Date | Duration |
|---|---:|---|---|---:|
| Chamber 1 | 96.0 | 2026-04-26 | 2026-05-18 | 23 days |
| Chamber 2 | 96.0 | 2026-05-03 | 2026-05-14 | 12 days |
| Chamber 3 | 96.0 | 2026-04-26 | 2026-05-03 | 8 days |

The longest run occurred in Chamber 1 and lasted 23 days.
This indicates a potentially malfunctioning humidity sensor.

### Stuck Humidity Sensor Treatment

Across all lots, every recorded humidity was 96 from 26 April through 18 May inclusive. Gaps in individual chamber observations split the chamber-specific runs above. The cleaner masks all affected readings in that interval.

Affected readings were replaced with missing values and imputed using the corresponding chamber's median humidity.

Legitimate humidity readings outside the identified date ranges were preserved.

A synthetic unit test verifies this behaviour.

### 2.6 Target Typos

**Status:** Confirmed.

- Suspicious target rows: 13
- Pattern: values approximately 10 times larger than the normal target range.

These values were corrected by dividing the affected targets by 10.

### 2.7 Spelling Variants

**Status:** Not observed.

The inspected categorical columns, `crop` and `chamber`, contained consistent category spellings across the datasets.

## 3. Cleaning Methodology

A reusable cleaning function was implemented in:

`src/mp235/cleaning.py`

The function performs the following operations:

1. Removes exact duplicate rows.
2. Replaces the `-999` temperature sentinel with missing values.
3. Corrects target values greater than 20 by dividing them by 10.
4. Masks affected humidity readings and imputes missing temperatures and humidity using the median temperature of the corresponding chamber.

For chronological validation, chamber medians come from the cleaned earlier partition.

The function operates on a copy of the input DataFrame and does not modify the original data.

## 4. Training Dataset Validation

| Validation | Before | After |
|---|---:|---:|
| Rows | 1,027 | 1,000 |
| Duplicate rows | 27 | 0 |
| Sentinel values | 36 | 0 |
| Missing temperatures | - | 0 |
| Target values above 20 | 13 | 0 |
| Maximum target | 105.1 | 14.27 |

Output file:

`data/processed/train_cleaned.csv`

## 5. Later Dataset Validation

| Validation | Before | After |
|---|---:|---:|
| Rows | 300 | 300 |
| Duplicate rows | 0 | 0 |
| Sentinel values | 0 | 0 |
| Missing temperatures | - | 0 |

Output file:

`data/processed/later_cleaned.csv`

Later data is supplied clean and its spoilage_pct labels are used directly for evaluation.

## 6. Raw Data Preservation

The original files were not modified during cleaning.

The following checks returned `True`:

- Original training CSV unchanged.
- Original later CSV unchanged.

Raw datasets remain under `data/raw/`, while processed datasets are stored separately under `data/processed/`.

## 7. Automated Testing

Synthetic tests were implemented in:

`tests/test_cleaning.py`

All four cleaning tests passed:

- Duplicate removal.
- Sentinel replacement and imputation.
- Target typo correction.
- Preservation of the original DataFrame.

Test command:

`PYTHONPATH=src python -m pytest tests/test_cleaning.py -v`

Result: **4 passed**.

## 8. Conclusion

The training and later datasets were processed using a reusable, deterministic cleaning function.

Four data-quality issues were confirmed in the original training dataset: sentinel values, duplicate rows, stuck humidity readings and target typos.

The original raw datasets remain unchanged, and the cleaned datasets are stored separately for subsequent model development and evaluation.

