## Dataset Description

The dataset is a time-series/tabular dataset with the following columns:

*   **id**
    A unique key constructed by concatenating code, sub_code, sub_category, horizon, and ts_index with a double underscore (`__`). This ensures each row is distinctly identifiable.
*   **code**
    A unique identifier for the entity.
*   **sub_code**
    A categorical attribute grouping entities into sub-families or segments.
*   **sub\_category**
    A categorical label describing the broad category to which the entity belongs.
*   **ts\_index**
    Integer timestamp of the observation: indicates when the features were recorded.
*   **horizon**
    A categorical forecast-horizon group. Typical codes are:
    1. 1 = short-term
    2. 3 = medium-term
    3. 10 = long-term
    4. 25 = extra long-term
    These codes do not represent the difference between ts\_index.
*   **weight**
    A numeric weight for each row, used to compute in the evaluation metric. **DO NOT USE AS A FEATURE**.These weights are used in the loss function formula (w).
*   **feature\_a, feature\_b, …, feature\_ch**
    A set of 86 anonymized features.

**Data shape**
Each row represents one forecast instance for a particular combination of (`code`, `sub_code`, `sub_category`, `ts_index`, `horizon`) along with its associated feature values. All features can be fed directly into typical regression models.

---

### Files
2 files

### Size
922.03 MB

### Type
parquet

### License
[Subject to Competition Rules](/path/to/license/link) *(Note: The actual link to the license rules is not present in the original HTML snippet, so a placeholder is used)*

---

### test.parquet (146.1 MB)

**Unable to show preview**
*(Image: error_light.svg)*
We don't have metadata for this file

---

### Data Explorer
922.03 MB
*   `test.parquet`
*   `train.parquet`

### Summary
**2 files**

[Download All files] *(This would typically be a button/link to download the files)*

---

**Download data**
```bash
kaggle competitions download -c ts-forecasting