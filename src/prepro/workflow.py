from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from prepro.balance import balance
from prepro.cardinality import cardinality
from prepro.cast import cast
from prepro.clean_strings import clean_strings
from prepro.collinearity import collinearity
from prepro.detect_types import detect_types
from prepro.drop_useless import drop_useless
from prepro.duplicates import duplicates
from prepro.encode import encode
from prepro.extract_datetime import extract_datetime
from prepro.missing import missing
from prepro.outliers import outliers
from prepro.polynomial import polynomial
from prepro.scale import scale
from prepro.skewness import skewness
from prepro.split_dataset import split
from prepro.summary import summary
from prepro.target_preprocess import target
from prepro.variance_filter import variance_filter


def ask_bool(prompt: str, default: bool) -> bool:
    suffix = " (y/n) [default: y]: " if default else " (y/n) [default: n]: "
    while True:
        try:
            choice = input(prompt + suffix).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nAborting interactive wizard. Using default value.")
            return default
        if not choice:
            return default
        if choice in ["y", "yes", "true", "t", "1"]:
            return True
        if choice in ["n", "no", "false", "f", "0"]:
            return False
        print("Invalid input. Please enter 'y' or 'n'.")


def ask_choice(prompt: str, choices: List[str], default: str) -> str:
    choices_str = "/".join(choices)
    prompt_full = f"{prompt} ({choices_str}) [default: {default}]: "
    while True:
        try:
            choice = input(prompt_full).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nAborting interactive wizard. Using default value.")
            return default
        if not choice:
            return default
        matched = [c for c in choices if c.startswith(choice)]
        if len(matched) == 1:
            return matched[0]
        elif len(matched) > 1:
            if choice in choices:
                return choice
            print(f"Ambiguous choice. Possible matches: {matched}")
        else:
            print(f"Invalid input. Choices are: {choices}")


def ask_float(
    prompt: str,
    default: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    prompt_full = f"{prompt} [default: {default}]: "
    while True:
        try:
            val_str = input(prompt_full).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborting interactive wizard. Using default value.")
            return default
        if not val_str:
            return default
        try:
            val = float(val_str)
            if min_val is not None and val < min_val:
                print(f"Value must be >= {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"Value must be <= {max_val}.")
                continue
            return val
        except ValueError:
            print("Invalid input. Please enter a number.")


def ask_int(
    prompt: str,
    default: int,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
) -> int:
    prompt_full = f"{prompt} [default: {default}]: "
    while True:
        try:
            val_str = input(prompt_full).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborting interactive wizard. Using default value.")
            return default
        if not val_str:
            return default
        try:
            val = int(val_str)
            if min_val is not None and val < min_val:
                print(f"Value must be >= {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"Value must be <= {max_val}.")
                continue
            return val
        except ValueError:
            print("Invalid input. Please enter an integer.")


def ask_list(prompt: str) -> Optional[List[str]]:
    prompt_full = f"{prompt} (comma-separated, or press Enter for all/none): "
    try:
        val_str = input(prompt_full).strip()
    except (KeyboardInterrupt, EOFError):
        print("\nAborting interactive wizard. Using default (all/none).")
        return None
    if not val_str:
        return None
    return [item.strip() for item in val_str.split(",") if item.strip()]


def ask_cast_map(columns: List[str]) -> Optional[Dict[str, str]]:
    cast_map = {}
    print("\nDefine manual casting map for columns:")
    print("Supported types: int, float, str, bool, datetime, category")
    for col in columns:
        if ask_bool(f"  Cast column '{col}'?", False):
            target_type = ask_choice(
                f"  Select type for '{col}'",
                ["int", "float", "str", "bool", "datetime", "category"],
                "str",
            )
            cast_map[col] = target_type
    return cast_map if cast_map else None


def workflow(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    UI: bool = True,
    # Individual step flags (if UI=False, these control which steps run)
    run_cast: bool = True,
    run_duplicates: bool = True,
    run_drop_useless: bool = True,
    run_target: bool = True,
    run_split: bool = False,
    run_clean_strings: bool = True,
    run_cardinality: bool = True,
    run_detect_types: bool = True,
    run_extract_datetime: bool = True,
    run_missing: bool = True,
    run_outliers: bool = True,
    run_skewness: bool = True,
    run_scale: bool = True,
    run_encode: bool = True,
    run_balance: bool = False,
    run_variance_filter: bool = True,
    run_polynomial: bool = False,
    run_collinearity: bool = True,
    # Parameters for the steps
    cast_params: Optional[dict] = None,
    duplicates_params: Optional[dict] = None,
    drop_useless_params: Optional[dict] = None,
    split_params: Optional[dict] = None,
    clean_strings_params: Optional[dict] = None,
    cardinality_params: Optional[dict] = None,
    detect_types_params: Optional[dict] = None,
    extract_datetime_params: Optional[dict] = None,
    missing_params: Optional[dict] = None,
    outliers_params: Optional[dict] = None,
    skewness_params: Optional[dict] = None,
    scale_params: Optional[dict] = None,
    encode_params: Optional[dict] = None,
    balance_params: Optional[dict] = None,
    variance_filter_params: Optional[dict] = None,
    polynomial_params: Optional[dict] = None,
    collinearity_params: Optional[dict] = None,
    report: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Executes a comprehensive data preprocessing workflow in a logical 18-step sequence.
    Can be run interactively (UI=True) or automatically with custom params.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to preprocess.
    target_col : str, optional
        The name of the target column.
    UI : bool, default True
        If True, prompts the user interactively on the command line for settings.
    ...
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()

    # If interactive mode is enabled, prompt user for settings in the new 18-step order
    if UI:
        print("\n" + "=" * 80)
        print("                 PREPRO WORKFLOW INTERACTIVE WIZARD")
        print("=" * 80)

        # Dataset summary
        print("\nAnalyzing dataset. Please wait...")
        print(summary(df))
        print("\n" + "=" * 80)

        # 1. Type casting
        run_cast = ask_bool(
            "\nDo you want to detect and cast column data types?", True
        )
        if run_cast:
            manual_cast = ask_bool(
                "  Do you want to specify a manual casting map?", False
            )
            cast_map = None
            if manual_cast:
                cast_map = ask_cast_map(list(df.columns))
            na_strs = ask_list(
                "  String placeholders to replace with NA (e.g. ?,N/A)"
            )
            cast_params = {
                "cast_map": cast_map,
                "na_strings": na_strs,
                "report": report,
            }

        # 2. Deduplication
        run_duplicates = ask_bool(
            "\nDo you want to check for and remove duplicate rows?", True
        )
        if run_duplicates:
            dup_cols = ask_list("  Columns to consider for duplicates")
            dup_keep = ask_choice(
                "  Which duplicate row to keep?",
                ["first", "last", "false"],
                "first",
            )
            if dup_keep == "false":
                dup_keep = False
            duplicates_params = {
                "subset": dup_cols,
                "keep": dup_keep,
                "report": report,
            }

        # 3. Drop useless
        run_drop_useless = ask_bool(
            "\nDo you want to identify and drop useless columns (constant/ID columns)?",
            True,
        )
        if run_drop_useless:
            id_t = ask_float(
                "  ID column threshold (ratio of unique values to total values)",
                0.9,
                0.0,
                1.0,
            )
            const_t = ask_float(
                "  Constant column threshold (ratio of dominant value)",
                0.95,
                0.0,
                1.0,
            )
            drop_useless_params = {
                "id_threshold": id_t,
                "const_threshold": const_t,
                "drop": True,
                "report": report,
            }

        # 4. Target variable selection
        while True:
            t_col = input(
                "\nEnter the name of the target column (press Enter to skip target preprocessing): "
            ).strip()
            if not t_col:
                target_col = None
                run_target = False
                break
            elif t_col in df.columns:
                target_col = t_col
                run_target = True
                break
            else:
                print(
                    f"Column '{t_col}' not found. Available columns: {list(df.columns)}"
                )

        if run_target:
            print(f"Selected target column: '{target_col}'")

        # 5. Split Dataset
        run_split = ask_bool(
            "\nDo you want to split the dataset into train and test sets?", False
        )
        if run_split:
            s_prop = ask_float(
                "  Train split proportion (0.0 to 1.0)",
                0.8,
                0.0001,
                0.9999,
            )
            s_seed = ask_int("  Random seed for splitting", 42)
            split_params = {"train_proportion": s_prop, "seed": s_seed}

        # 6. Clean strings
        run_clean_strings = ask_bool(
            "\nDo you want to clean string/object columns (strip, lowercase, fix typos)?",
            True,
        )
        if run_clean_strings:
            cs_cols = ask_list("  Columns to clean")
            cs_strip = ask_bool("  Strip whitespaces?", True)
            cs_lower = ask_bool("  Convert to lowercase?", True)
            cs_typos = ask_bool("  Fix typos/fuzzy matches?", False)
            cs_t_threshold = 0.8
            if cs_typos:
                cs_t_threshold = ask_float(
                    "    Typo similarity threshold (0.0 to 1.0)",
                    0.8,
                    0.0,
                    1.0,
                )
            clean_strings_params = {
                "cols": cs_cols,
                "strip": cs_strip,
                "lowercase": cs_lower,
                "fix_typos": cs_typos,
                "typo_threshold": cs_t_threshold,
                "report": report,
            }

        # 7. Cardinality Analysis
        run_cardinality = ask_bool(
            "\nDo you want to run cardinality analysis?", True
        )
        if run_cardinality:
            card_cols = ask_list("  Specific columns to analyze for cardinality")
            card_thresh = ask_int("  High cardinality threshold", 50)
            cardinality_params = {
                "cols": card_cols,
                "high_threshold": card_thresh,
                "report": report,
            }

        # 8. Feature Type Detection
        run_detect_types = ask_bool(
            "\nDo you want to run feature type detection?", True
        )
        if run_detect_types:
            detect_types_params = {
                "report": report,
            }

        # 9. Datetime extraction
        run_extract_datetime = ask_bool(
            "\nDo you want to extract features from datetime columns?", True
        )
        if run_extract_datetime:
            dt_cols = ask_list("  Specific datetime columns to extract")
            dt_feats = ask_list(
                "  Datetime features to extract (e.g. year,month,day,hour)"
            )
            dt_cyc = ask_bool(
                "  Encode features as cyclical (sine/cosine)?", False
            )
            extract_datetime_params = {
                "cols": dt_cols,
                "features": dt_feats,
                "cyclical": dt_cyc,
                "report": report,
            }

        # 10. Missing Value Imputation
        run_missing = ask_bool(
            "\nDo you want to handle and impute missing values?", True
        )
        if run_missing:
            m_strat = ask_choice(
                "  Imputation strategy",
                ["mean", "median", "mode", "knn", "mice"],
                "mean",
            )
            m_mcar = ask_bool("  Run Little's MCAR test?", True)
            m_ind = ask_bool("  Add missing value indicator columns?", True)
            m_t = ask_float(
                "  Missingness threshold above which columns are dropped (0.0 to 1.0)",
                0.5,
                0.0,
                1.0,
            )
            missing_params = {
                "strategy": m_strat,
                "mcar_test": m_mcar,
                "add_indicator": m_ind,
                "threshold": m_t,
                "report": report,
            }

        # 11. Outliers
        run_outliers = ask_bool(
            "\nDo you want to detect and treat outliers?", True
        )
        if run_outliers:
            out_method = ask_choice(
                "  Outlier detection method", ["iqr", "zscore", "modified_zscore", "isolation_forest", "lof"], "iqr"
            )
            out_treat = ask_choice(
                "  Outlier treatment method",
                ["winsorize", "remove", "flag", "none"],
                "winsorize",
            )
            if out_treat == "none":
                out_treat = None
            out_t = ask_float("  Outlier detection threshold", 1.5, 0.0)
            out_cols = ask_list("  Specific numeric columns to check")
            outliers_params = {
                "method": out_method,
                "treatment": out_treat,
                "threshold": out_t,
                "cols": out_cols,
                "report": report,
            }

        # 12. Skewness
        run_skewness = ask_bool(
            "\nDo you want to transform skewed numeric columns?", True
        )
        if run_skewness:
            skew_method = ask_choice(
                "  Transformation method",
                ["yeojohnson", "boxcox", "log", "sqrt", "auto"],
                "yeojohnson",
            )
            skew_t = ask_float("  Skewness threshold for transformation", 0.5, 0.0)
            skew_cols = ask_list("  Specific numeric columns to check")
            skewness_params = {
                "method": skew_method,
                "skew_threshold": skew_t,
                "cols": skew_cols,
                "report": report,
            }

        # 13. Scale
        run_scale = ask_bool(
            "\nDo you want to scale numeric features?", True
        )
        if run_scale:
            scale_method = ask_choice(
                "  Scaling method",
                ["standard", "minmax", "robust", "maxabs"],
                "standard",
            )
            scale_cols = ask_list("  Specific columns to scale")
            scale_params = {
                "method": scale_method,
                "cols": scale_cols,
                "report": report,
            }

        # 14. Encode
        run_encode = ask_bool(
            "\nDo you want to encode categorical columns?", True
        )
        if run_encode:
            enc_method = ask_choice(
                "  Encoding method",
                ["auto", "onehot", "ordinal", "target", "frequency", "binary"],
                "auto",
            )
            enc_drop = ask_bool(
                "  Drop first category in onehot encoding?", True
            )
            encode_params = {
                "method": enc_method,
                "target_col": target_col,
                "drop_first": enc_drop,
                "handle_unknown": "ignore",
                "report": report,
            }

        # 15. Class Balance
        if target_col is not None:
            run_balance = ask_bool(
                "\nDo you want to handle class imbalance on the target?", False
            )
            if run_balance:
                bal_method = ask_choice(
                    "  Balancing method",
                    ["smote", "adasyn", "undersample", "weights"],
                    "smote",
                )
                bal_ratio = ask_float(
                    "  Desired ratio of minority to majority class", 1.0, 0.0
                )
                bal_seed = ask_int("  Random seed (0 for None)", 0)
                if bal_seed == 0:
                    bal_seed = None
                balance_params = {
                    "target": target_col,
                    "method": bal_method,
                    "ratio": bal_ratio,
                    "seed": bal_seed,
                    "report": report,
                }
        else:
            run_balance = False

        # 16. Variance Filter
        run_variance_filter = ask_bool(
            "\nDo you want to filter out low-variance features?", True
        )
        if run_variance_filter:
            vf_t = ask_float("  Variance threshold", 0.01, 0.0)
            vf_quasi = ask_float(
                "  Quasi-constant threshold (0.0 to 1.0)", 0.95, 0.0, 1.0
            )
            variance_filter_params = {
                "threshold": vf_t,
                "quasi_threshold": vf_quasi,
                "report": report,
            }

        # 17. Polynomial Features
        run_polynomial = ask_bool(
            "\nDo you want to generate polynomial features?", False
        )
        if run_polynomial:
            poly_deg = ask_int("  Degree of polynomial features", 2, 1)
            poly_int = ask_bool("  Generate interaction features only?", False)
            poly_bias = ask_bool("  Include bias column?", False)
            poly_cols = ask_list("  Specific columns for polynomial features")
            polynomial_params = {
                "degree": poly_deg,
                "interaction_only": poly_int,
                "include_bias": poly_bias,
                "cols": poly_cols,
                "report": report,
            }

        # 18. Collinearity
        run_collinearity = ask_bool(
            "\nDo you want to handle collinearity (highly correlated features)?", True
        )
        if run_collinearity:
            coll_method = ask_choice(
                "  Collinearity detection method",
                ["vif", "correlation", "both"],
                "both",
            )
            coll_vif = ask_float("  VIF threshold", 5.0, 1.0)
            coll_corr = ask_float(
                "  Correlation threshold (0.0 to 1.0)", 0.9, 0.0, 1.0
            )
            coll_treat = ask_choice(
                "  Collinearity treatment", ["drop", "warn", "report"], "drop"
            )
            collinearity_params = {
                "method": coll_method,
                "vif_threshold": coll_vif,
                "corr_threshold": coll_corr,
                "treatment": coll_treat,
                "report": report,
            }

        print("\n" + "=" * 80)
        print("                 STARTING PIPELINE EXECUTION")
        print("=" * 80 + "\n")

    # Initialize default param dicts if None
    cast_params = cast_params if cast_params is not None else {}
    duplicates_params = duplicates_params if duplicates_params is not None else {}
    drop_useless_params = drop_useless_params if drop_useless_params is not None else {}
    split_params = split_params if split_params is not None else {}
    clean_strings_params = clean_strings_params if clean_strings_params is not None else {}
    cardinality_params = cardinality_params if cardinality_params is not None else {}
    detect_types_params = detect_types_params if detect_types_params is not None else {}
    extract_datetime_params = extract_datetime_params if extract_datetime_params is not None else {}
    missing_params = missing_params if missing_params is not None else {}
    outliers_params = outliers_params if outliers_params is not None else {}
    skewness_params = skewness_params if skewness_params is not None else {}
    scale_params = scale_params if scale_params is not None else {}
    encode_params = encode_params if encode_params is not None else {}
    balance_params = balance_params if balance_params is not None else {}
    variance_filter_params = variance_filter_params if variance_filter_params is not None else {}
    polynomial_params = polynomial_params if polynomial_params is not None else {}
    collinearity_params = collinearity_params if collinearity_params is not None else {}

    # Sequential execution of preprocessing steps (1-18)
    
    # 1. Type casting
    if run_cast:
        if report:
            print(">>> Step 1: Casting column data types...")
        params = {"report": report}
        params.update(cast_params)
        df = cast(df, **params)

    # 2. Deduplication
    if run_duplicates:
        if report:
            print(">>> Step 2: Removing duplicate rows...")
        params = {"report": report}
        params.update(duplicates_params)
        df = duplicates(df, **params)

    # 3. Drop useless
    if run_drop_useless:
        if report:
            print(">>> Step 3: Dropping useless columns...")
        params = {"report": report}
        params.update(drop_useless_params)
        df = drop_useless(df, **params)

    # 4. Target preprocessing
    if run_target and target_col is not None:
        if report:
            print(">>> Step 4: Preprocessing target variable...")
        df = target(df, target_col)

    # 5. Train/test split + leakage guard Setup
    run_split_active = run_split
    if run_split:
        if report:
            print(">>> Step 5: Splitting dataset (Leakage Guard Active)...")
        params = {}
        params.update(split_params)
        if "train_proportion" not in params:
            params["train_proportion"] = 0.8
        train_df, test_df = split(df, **params)
        if report:
            print(
                f"    Split completed. Train shape: {train_df.shape}, Test shape: {test_df.shape}"
            )
    else:
        train_df = df
        test_df = None

    # 6. Clean strings
    if run_clean_strings:
        if report:
            print(">>> Step 6: Cleaning string columns...")
        params = {"report": report}
        params.update(clean_strings_params)
        if test_df is not None:
            if clean_strings_params.get("fix_typos", False):
                train_df, typo_maps = clean_strings(train_df, return_typo_maps=True, **params)
                test_params = params.copy()
                test_params["fix_typos"] = False
                test_params["report"] = False
                test_df = clean_strings(test_df, typo_maps=typo_maps, **test_params)
            else:
                train_df = clean_strings(train_df, **params)
                test_params = params.copy()
                test_params["report"] = False
                test_df = clean_strings(test_df, **test_params)
        else:
            train_df = clean_strings(train_df, **params)

    # 7. Cardinality analysis
    if run_cardinality:
        if report:
            print(">>> Step 7: Running cardinality analysis...")
        params = {"report": report}
        params.update(cardinality_params)
        if test_df is not None:
            cardinality(train_df, **params)
        else:
            cardinality(train_df, **params)

    # 8. Feature type detection
    if run_detect_types:
        if report:
            print(">>> Step 8: Running feature type detection...")
        params = {"report": report}
        params.update(detect_types_params)
        if test_df is not None:
            detect_types(train_df, **params)
        else:
            detect_types(train_df, **params)

    # 9. Datetime extraction
    if run_extract_datetime:
        if report:
            print(">>> Step 9: Extracting datetime features...")
        params = {"report": report}
        params.update(extract_datetime_params)
        if test_df is not None:
            if params.get("cols") is None:
                dtypes = detect_types(train_df)
                dt_cols = [col for col, t in dtypes.items() if t == "datetime"]
                params_with_cols = params.copy()
                params_with_cols["cols"] = dt_cols
            else:
                params_with_cols = params
            train_df = extract_datetime(train_df, **params_with_cols)
            test_params = params_with_cols.copy()
            test_params["report"] = False
            test_df = extract_datetime(test_df, **test_params)
        else:
            train_df = extract_datetime(train_df, **params)

    # 10. Missing value analysis & imputation
    if run_missing:
        if report:
            print(">>> Step 10: Imputing missing values...")
        params = {"report": report}
        params.update(missing_params)
        if test_df is not None:
            train_df, missing_state = missing(train_df, return_state=True, **params)
            test_params = params.copy()
            test_params["report"] = False
            test_df = missing(test_df, fitted_state=missing_state, **test_params)
        else:
            train_df = missing(train_df, **params)

    # 11. Outlier detection & treatment
    if run_outliers:
        if report:
            print(">>> Step 11: Handling outliers...")
        params = {"report": report}
        params.update(outliers_params)
        if test_df is not None:
            train_df, outlier_state = outliers(train_df, return_state=True, **params)
            test_params = params.copy()
            test_params["report"] = False
            test_df = outliers(test_df, fitted_state=outlier_state, **test_params)
        else:
            train_df = outliers(train_df, **params)

    # 12. Skewness & distribution correction
    if run_skewness:
        if report:
            print(">>> Step 12: Adjusting skewness...")
        params = {"report": report}
        params.update(skewness_params)
        if test_df is not None:
            train_df, skew_state = skewness(train_df, return_state=True, **params)
            test_params = params.copy()
            test_params["report"] = False
            test_df = skewness(test_df, fitted_state=skew_state, **test_params)
        else:
            train_df = skewness(train_df, **params)

    # 13. Feature scaling & normalization
    if run_scale:
        if report:
            print(">>> Step 13: Scaling numeric features...")
        params = {"report": report}
        params.update(scale_params)
        if test_df is not None:
            train_df, fitted_scaler = scale(train_df, return_scaler=True, **params)
            test_params = params.copy()
            test_params["report"] = False
            test_df = scale(test_df, fitted_scaler=fitted_scaler, **test_params)
        else:
            train_df = scale(train_df, **params)

    # 14. Encoding
    if run_encode:
        if report:
            print(">>> Step 14: Encoding categorical columns...")
        params = {"report": report}
        if target_col is not None and "target_col" not in encode_params:
            params["target_col"] = target_col
        params.update(encode_params)
        if test_df is not None:
            train_df, encode_state = encode(train_df, return_state=True, **params)
            test_params = params.copy()
            test_params["report"] = False
            test_df = encode(test_df, fitted_state=encode_state, **test_params)
        else:
            train_df = encode(train_df, **params)

    # 15. Class imbalance handling
    if run_balance and target_col is not None:
        if report:
            print(">>> Step 15: Handling class imbalance...")
        params = {"report": report}
        if "target" not in balance_params:
            params["target"] = target_col
        params.update(balance_params)
        if test_df is not None:
            method = params.get("method", "smote").lower().strip()
            if method == "weights":
                class_counts = train_df[target_col].value_counts()
                total = len(train_df)
                n_classes = len(class_counts)
                class_weights = {
                    cls: total / (n_classes * count)
                    for cls, count in class_counts.items()
                }
                train_df["sample_weight"] = train_df[target_col].map(class_weights)
                test_df["sample_weight"] = test_df[target_col].map(class_weights).fillna(1.0)
            else:
                train_df = balance(train_df, **params)
        else:
            train_df = balance(train_df, **params)

    # 16. Variance filtering
    if run_variance_filter:
        if report:
            print(">>> Step 16: Applying variance filter...")
        params = {"report": report}
        params.update(variance_filter_params)
        if test_df is not None:
            train_df, cols_to_drop = variance_filter(train_df, return_cols_to_drop=True, **params)
            test_params = params.copy()
            test_params["report"] = False
            test_df = variance_filter(test_df, cols_to_drop=cols_to_drop, **test_params)
        else:
            train_df = variance_filter(train_df, **params)

    # 17. Polynomial / interaction terms
    if run_polynomial:
        if report:
            print(">>> Step 17: Generating polynomial features...")
        params = {"report": report}
        params.update(polynomial_params)
        if test_df is not None:
            train_df, fill_values = polynomial(train_df, return_fill_values=True, **params)
            test_params = params.copy()
            test_params["report"] = False
            test_df = polynomial(test_df, fill_values=fill_values, **test_params)
        else:
            train_df = polynomial(train_df, **params)

    # 18. Multicollinearity detection
    if run_collinearity:
        if report:
            print(">>> Step 18: Handling collinear features...")
        params = {"report": report}
        params.update(collinearity_params)
        if test_df is not None:
            train_df, cols_to_drop = collinearity(train_df, return_cols_to_drop=True, **params)
            test_params = params.copy()
            test_params["report"] = False
            test_df = collinearity(test_df, cols_to_drop=cols_to_drop, **test_params)
        else:
            train_df = collinearity(train_df, **params)

    if run_split_active:
        if report:
            print(
                f"\nWorkflow complete. Data split into Train: {train_df.shape} and Test: {test_df.shape}"
            )
        return train_df, test_df

    if report:
        print(f"\nWorkflow complete. Cleaned DataFrame shape: {train_df.shape}")

    return train_df
