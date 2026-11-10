"""Data loading, cleaning, feature engineering and preprocessing pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RAW_DATA_FILE,
    TARGET,
    TEST_SIZE,
)

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

SUPPORT_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAMING_COLUMNS = ["StreamingTV", "StreamingMovies"]


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------
def load_raw_data(path: Path | str = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Run `python scripts/download_data.py` or place Telco-Customer-Churn.csv in data/raw/."
        )
    return pd.read_csv(path)


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].str.strip()
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw dataframe and return it with a data-quality report.

    Steps
    -----
    1. Normalise column names and strip stray whitespace.
    2. Remove exact duplicate rows.
    3. Convert ``TotalCharges`` to numeric (it contains blank strings for
       brand-new customers) and impute blanks from ``tenure * MonthlyCharges``.
    4. Encode the binary churn target.
    5. Validate numeric ranges and report outliers (kept intentionally).
    """
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    df = _strip_string_columns(df)

    report: dict = {
        "raw_rows": int(len(df)),
        "raw_columns": int(df.shape[1]),
        "duplicate_rows_removed": int(df.duplicated().sum()),
        "missing_values_before": int(df.isna().sum().sum()),
    }
    df = df.drop_duplicates().reset_index(drop=True)

    if "TotalCharges" in df.columns:
        total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
        missing_total = int(total_charges.isna().sum())
        estimated_total = df["MonthlyCharges"].astype(float) * df["tenure"].astype(float)
        # Brand-new customers have tenure == 0, so the estimate naturally becomes 0.
        df["TotalCharges"] = total_charges.fillna(estimated_total)
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
        report["missing_total_charges"] = missing_total
        report["total_charges_imputed"] = missing_total
        report["imputation_rule"] = "TotalCharges blank -> MonthlyCharges * tenure"

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
        missing_target = int(df[TARGET].isna().sum())
        if missing_target:
            df = df.dropna(subset=[TARGET]).reset_index(drop=True)
        df[TARGET] = df[TARGET].astype(int)
        report["missing_target_rows_removed"] = missing_target

    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Range validation (outliers are reported but deliberately kept: churn is
    # not a distance-based problem and extreme bills can be genuine signal).
    report["invalid_tenure_rows"] = int(
        ((df["tenure"] < 0) | (df["tenure"] > 72)).sum()
    ) if "tenure" in df.columns else 0
    report["negative_charge_rows"] = int(
        ((df.get("MonthlyCharges", 0) < 0) | (df.get("TotalCharges", 0) < 0)).sum()
    )

    report["clean_rows"] = int(len(df))
    report["clean_columns"] = int(df.shape[1])
    report["missing_values_after"] = int(df.isna().sum().sum())
    if TARGET in df.columns:
        report["churn_rate"] = float(df[TARGET].mean())
        report["churned_customers"] = int(df[TARGET].sum())
        report["retained_customers"] = int((1 - df[TARGET]).sum())

    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _count_yes(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(0, index=df.index)
    return (df[available] == "Yes").sum(axis=1).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable behavioural and billing features.

    All transformations are deterministic and use no target information, so
    they can safely be applied before the train/test split.
    """
    df = df.copy()

    if "tenure" in df.columns:
        df["TenureGroup"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
        ).astype(str)
        df["IsNewCustomer"] = np.where(df["tenure"] <= 6, "Yes", "No")

    if {"TotalCharges", "tenure", "MonthlyCharges"}.issubset(df.columns):
        # Average monthly spend over the customer's lifetime; guards tenure == 0.
        tenure_safe = df["tenure"].clip(lower=1)
        df["AvgMonthlyCharges"] = (df["TotalCharges"] / tenure_safe).round(2)
        df["MonthlyChargeDelta"] = (
            df["MonthlyCharges"] - df["AvgMonthlyCharges"]
        ).round(2)

    df["NumServices"] = _count_yes(df, SERVICE_COLUMNS)
    df["SupportServices"] = _count_yes(df, SUPPORT_COLUMNS)
    df["StreamingServices"] = _count_yes(df, STREAMING_COLUMNS)

    if {"Partner", "Dependents"}.issubset(df.columns):
        df["FamilySize"] = (df["Partner"] == "Yes").astype(int) + (
            df["Dependents"] == "Yes"
        ).astype(int)

    if "InternetService" in df.columns:
        df["HasInternet"] = np.where(df["InternetService"] == "No", "No", "Yes")

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier/target columns and return the model input frame."""
    drop_columns = [column for column in (ID_COLUMN, TARGET) if column in df.columns]
    X = df.drop(columns=drop_columns)
    missing = [
        column
        for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        if column not in X.columns
    ]
    if missing:
        raise ValueError(
            "Feature engineering did not produce the expected columns: "
            f"{missing}. Run engineer_features() before prepare_features()."
        )
    return X


# ---------------------------------------------------------------------------
# Model preprocessing and splitting
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Numeric imputation/scaling + one-hot encoding for categoricals."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split that preserves the churn ratio."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def prepare_customer_record(
    record: dict, input_features: list[str] | None = None
) -> pd.DataFrame:
    """Turn a single raw customer record into a model-ready dataframe."""
    df = pd.DataFrame([record])

    if input_features:
        missing = [column for column in input_features if column not in df.columns]
        if missing:
            raise ValueError(f"Customer record is missing required fields: {missing}")
        df = df[input_features].copy()

    if "TotalCharges" not in df.columns or pd.isna(df.loc[0, "TotalCharges"]):
        df["TotalCharges"] = float(df.loc[0, "MonthlyCharges"]) * float(
            df.loc[0, "tenure"]
        )

    df = _strip_string_columns(df)
    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["MonthlyCharges"].astype(float) * df["tenure"].astype(float)
        )

    engineered = engineer_features(df)
    return engineered
