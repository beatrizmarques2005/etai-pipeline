"""
Cleaning, feature/target split, leak-safe preprocessor and dev/test split.

`clean_dataset` and `split_features_target` learn nothing from the data, so they are safe on
data to predict. Everything learned (imputation, encoding, scaling) is in `build_preprocessor`.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (makes IterativeImputer importable)
from sklearn.pipeline import Pipeline
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler,
)
from category_encoders import CountEncoder


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies {column: {"min": ..., "max": ...}} domain rules (either bound optional) and sets
    violations to NaN, in place. Returns the number of violations per column.
    """
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Placeholder tokens and rule violations become NaN, categories are canonicalized, redundant
    columns dropped. Row-preserving and target-agnostic.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))
    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), placeholder_tokens)

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    return out.drop(columns=columns_to_drop)


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """Training data only: drops duplicate rows and repeated ids (keeps the first)."""
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each listed column, before it is imputed."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """Returns (X, y, extras); `extras` = sensitive attribute and COMPAS score, for auditing. `y` is None without a target."""
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in always_drop]
    return df[feature_cols], y, extras


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
# sklearn's TargetEncoder cross-fits: a training row's encoding never uses its own label.
_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda seed: TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed)),
}


# "iterative" is MICE; "random_forest" is the missForest idea. Neither uses the target.
_NUMERIC_IMPUTERS = {
    "median": lambda cfg, seed: SimpleImputer(strategy="median"),
    "mean": lambda cfg, seed: SimpleImputer(strategy="mean"),
    "knn": lambda cfg, seed: KNNImputer(n_neighbors=cfg.get("n_neighbors", 5)),
    "iterative": lambda cfg, seed: IterativeImputer(max_iter=cfg.get("max_iter", 10), random_state=seed),
    "random_forest": lambda cfg, seed: IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=cfg.get("n_estimators", 50), min_samples_leaf=5,
                                        random_state=seed, n_jobs=-1),
        max_iter=cfg.get("max_iter", 10), random_state=seed),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Unfitted ColumnTransformer: numeric columns scaled then imputed (scaled first so distance-based
    imputers compare one scale), categoricals imputed and encoded, missingness flags passed through.
    """
    imputation = preprocessing_config.get("imputation", {})
    scaler_factory = _SCALERS[preprocessing_config["scaler"]]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[preprocessing_config["encoder"]](preprocessing_config.get("random_state"))
    numeric_strategy = imputation.get("numeric_strategy", "median")
    if numeric_strategy not in _NUMERIC_IMPUTERS:
        raise ValueError(f"Unknown numeric_strategy: {numeric_strategy}. Options: {list(_NUMERIC_IMPUTERS)}")

    numeric_pipeline = Pipeline([
        ("scale", scaler),
        ("impute", _NUMERIC_IMPUTERS[numeric_strategy](imputation, preprocessing_config.get("random_state"))),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])
    indicator_cols = [f"{c}_was_missing" for c in preprocessing_config.get("mnar_indicator_sources", [])]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, preprocessing_config["numeric_features"]),
        ("categorical", categorical_pipeline, preprocessing_config["categorical_features"]),
        ("indicators", "passthrough", indicator_cols),
    ])


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """Stratified split into a development set and a locked test set. Returns X, y, extras for each."""
    return train_test_split(X, y, extras, test_size=test_size, random_state=random_state, stratify=y)
