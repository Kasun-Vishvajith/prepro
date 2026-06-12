# prepro

`prepro` is a Python library designed to streamline and automate data preprocessing workflows for machine learning and data analysis.

At the core of the library is the `workflow` system, which allows you to run a complete, multi-stage data cleaning and preparation pipeline either interactively or programmatically.

---

## The `workflow` Pipeline

The primary entry point is the **`workflow`** function defined in [workflow.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/workflow.py#L148). It coordinates 16 distinct preprocessing steps in a logical sequence.

### Quick Start

```python
import pandas as pd
from prepro import workflow

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# 1. Run in Interactive Mode (Wizard CLI)
cleaned_df = workflow(df, UI=True)

# 2. Run in Programmatic / Automated Mode (No Prompts)
train_df, test_df = workflow(
    df,
    target_col="target_variable",
    UI=False,
    run_split=True,
    run_scale=True,
    scale_params={"method": "minmax"},
    run_missing=True,
    missing_params={"strategy": "median"}
)
```

---

## 1. Interactive Mode (`UI=True`)

When `UI=True` (the default behavior), `workflow` starts an interactive command-line wizard. It:
1. Prints an initial data profile summary using [summary.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/summary.py).
2. Prompts you step-by-step to enable/disable specific modules.
3. Prompts you for sub-parameters (e.g., threshold values, methods, column selections) directly in the CLI using interactive safety fallbacks.
4. Executes the configured pipeline sequentially.

---

## 2. Programmatic Mode (`UI=False`)

When running in production or batch scripts, set `UI=False`. The pipeline is controlled entirely using boolean flags (`run_<step>`) and parameter dictionaries (`<step>_params`).

### Execution Order & Modules

Below is the list of preprocessing steps in their exact sequence of execution:

| Step | Flag Name | Parameter Dictionary | Description / Module Link |
| :--- | :--- | :--- | :--- |
| **1** | `run_target` | *(No config dict)* | Targets target variable encoding/transformations via [target_preprocess.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/target_preprocess.py). |
| **2** | `run_duplicates` | `duplicates_params` | Removes identical/redundant rows using [duplicates.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/duplicates.py). |
| **3** | `run_drop_useless` | `drop_useless_params` | Identifies and drops constant or high-ratio ID columns using [drop_useless.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/drop_useless.py). |
| **4** | `run_clean_strings` | `clean_strings_params` | Strips whitespace, converts case, and resolves fuzzy string typos via [clean_strings.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/clean_strings.py). |
| **5** | `run_cast` | `cast_params` | Infers and casts types, converting custom NA string placeholders via [cast.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/cast.py). |
| **6** | `run_extract_datetime` | `extract_datetime_params` | Automatically extracts calendar/time units or cyclical (sin/cos) features via [extract_datetime.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/extract_datetime.py). |
| **7** | `run_missing` | `missing_params` | Imputes missing data (`mean`, `median`, `mode`, or `knn`) and performs Little's MCAR test via [missing.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/missing.py). |
| **8** | `run_outliers` | `outliers_params` | Detects outliers (IQR/Z-score) and applies treatment (remove/winsorize) via [outliers.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/outliers.py). |
| **9** | `run_skewness` | `skewness_params` | Automatically transforms highly skewed columns (Box-Cox, Yeo-Johnson, Log) via [skewness.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/skewness.py). |
| **10** | `run_scale` | `scale_params` | Scales features (Standard, MinMax, Robust, MaxAbs) using [scale.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/scale.py). |
| **11** | `run_encode` | `encode_params` | Encodes categorical variables (One-Hot, Ordinal, Target, Frequency, Binary) via [encode.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/encode.py). |
| **12** | `run_collinearity` | `collinearity_params` | Drops or reports multicollinear variables (using VIF or correlation thresholds) via [collinearity.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/collinearity.py). |
| **13** | `run_variance_filter` | `variance_filter_params` | Filters low-variance or quasi-constant features using [variance_filter.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/variance_filter.py). |
| **14** | `run_balance` | `balance_params` | Addresses class imbalances (SMOTE, ADASYN, Undersampling, weights) via [balance.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/balance.py). |
| **15** | `run_polynomial` | `polynomial_params` | Generates polynomial or interaction features via [polynomial.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/polynomial.py). |
| **16** | `run_split` | `split_params` | Splits the processed DataFrame into train and test sets via [split_dataset.py](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/src/prepro/split_dataset.py). |

---

## Detailed Step Configurations

### Deduplication (`run_duplicates=True`)
* **`duplicates_params`**:
  * `subset` (list of str, optional): Columns to look at. Defaults to checking all columns.
  * `keep` (str, default `"first"`): Which duplicate row to keep (`"first"`, `"last"`, or `False` to drop all).
  * `report` (bool, default `True`): Prints step execution reports.

### Useless Column Drop (`run_drop_useless=True`)
* **`drop_useless_params`**:
  * `id_threshold` (float, default `0.9`): Ratio of unique values to total values above which a column is dropped as an ID column.
  * `const_threshold` (float, default `0.95`): Ratio of the dominant value above which a column is dropped as a constant column.
  * `drop` (bool, default `True`): Actually drop columns (if `False`, only reports them).

### String Cleaning (`run_clean_strings=True`)
* **`clean_strings_params`**:
  * `cols` (list of str, optional): Columns to clean. Checks all string/object columns if `None`.
  * `strip` (bool, default `True`): Strip leading/trailing whitespaces.
  * `lowercase` (bool, default `True`): Lowercase all strings.
  * `fix_typos` (bool, default `False`): Use fuzzy matches to fix minor spelling variations.
  * `typo_threshold` (float, default `0.8`): Similarity threshold for fixing typos.

### Data Type Casting (`run_cast=True`)
* **`cast_params`**:
  * `cast_map` (dict, optional): Manual mappings of `{col_name: target_type}`.
  * `na_strings` (list of str, optional): Custom string placeholders to replace with `NaN` (e.g., `["?", "N/A"]`).

### Datetime Extraction (`run_extract_datetime=True`)
* **`extract_datetime_params`**:
  * `cols` (list of str, optional): Specific datetime columns. Auto-detects if `None`.
  * `features` (list of str, optional): Time attributes to extract (e.g., `["year", "month", "day", "hour"]`).
  * `cyclical` (bool, default `False`): Encodes calendar values into sinusoidal signals (`sin`/`cos`).

### Imputation (`run_missing=True`)
* **`missing_params`**:
  * `strategy` (str, default `"mean"`): Method to impute numeric values (`"mean"`, `"median"`, `"mode"`, or `"knn"`).
  * `mcar_test` (bool, default `True`): Runs Little's MCAR statistical test to diagnose missingness mechanisms.
  * `add_indicator` (bool, default `True`): Adds binary indicator columns marking the location of missing values.
  * `threshold` (float, default `0.5`): Columns with a missingness ratio exceeding this threshold are dropped.

### Outliers (`run_outliers=True`)
* **`outliers_params`**:
  * `method` (str, default `"iqr"`): Outlier detection algorithm (`"iqr"` or `"zscore"`).
  * `treatment` (str, default `"winsorize"`): Outlier treatment strategy (`"winsorize"`, `"remove"`, or `None` to just detect).
  * `threshold` (float, default `1.5`): Multiplier threshold for detection.
  * `cols` (list of str, optional): Columns to process.

### Skewness Adjustment (`run_skewness=True`)
* **`skewness_params`**:
  * `method` (str, default `"yeojohnson"`): Skewness correction transform (`"yeojohnson"`, `"boxcox"`, or `"log"`).
  * `skew_threshold` (float, default `0.5`): Absolute skewness threshold above which columns will be transformed.
  * `cols` (list of str, optional): Numeric columns to evaluate.

### Feature Scaling (`run_scale=True`)
* **`scale_params`**:
  * `method` (str, default `"standard"`): Scaler to apply (`"standard"`, `"minmax"`, `"robust"`, or `"maxabs"`).
  * `cols` (list of str, optional): Numeric columns to scale.

### Categorical Encoding (`run_encode=True`)
* **`encode_params`**:
  * `method` (str, default `"auto"`): Encoding technique (`"auto"`, `"onehot"`, `"ordinal"`, `"target"`, `"frequency"`, `"binary"`).
  * `target_col` (str, optional): Target column name (required for supervised target encoding).
  * `drop_first` (bool, default `True`): Drops the first level in one-hot encoding.
  * `handle_unknown` (str, default `"ignore"`): Strategy for unseen categories during encoding.

### Multicollinearity Treatment (`run_collinearity=True`)
* **`collinearity_params`**:
  * `method` (str, default `"both"`): Collinearity detection strategy (`"vif"`, `"correlation"`, or `"both"`).
  * `vif_threshold` (float, default `5.0`): Maximum allowed Variance Inflation Factor.
  * `corr_threshold` (float, default `0.9`): Maximum pairwise correlation threshold.
  * `treatment` (str, default `"drop"`): Action to take (`"drop"` collinear features, or `"report_only"`).

### Variance Filtering (`run_variance_filter=True`)
* **`variance_filter_params`**:
  * `threshold` (float, default `0.01`): Minimum variance threshold for numerical columns.
  * `quasi_threshold` (float, default `0.95`): Dominant value ratio above which features are dropped.

### Class Balancing (`run_balance=False`)
> [!NOTE]
> Balancing is only executed if `target_col` is specified.
* **`balance_params`**:
  * `method` (str, default `"smote"`): Balancing strategy (`"smote"`, `"adasyn"`, `"undersample"`, or `"weights"`).
  * `ratio` (float, default `1.0`): Target ratio of minority to majority class size.
  * `seed` (int, optional): Random seed.

### Polynomial Generation (`run_polynomial=False`)
* **`polynomial_params`**:
  * `degree` (int, default `2`): Degree of polynomial combinations.
  * `interaction_only` (bool, default `False`): If `True`, creates only cross-interaction products (e.g., $x_1 \times x_2$) without individual powers ($x_1^2$).
  * `include_bias` (bool, default `False`): Includes intercept bias column.
  * `cols` (list of str, optional): Numeric columns to combine.

### Train-Test Splitting (`run_split=False`)
* **`split_params`**:
  * `train_proportion` (float, default `0.8`): Fraction of records allocated to the training split.
  * `seed` (int, default `42`): Random seed.

---

## Verification & Testing

Tests for the `prepro` library are available under the [tests](file:///c:/Users/kasun/Projects/BSc%20Hons%20in%20Data%20Science/Personal%20Projects/prepro/tests) directory. You can run unit tests using `pytest` to verify the execution of individual modules and full workflow integration.
