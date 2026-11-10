"""Central configuration for the Telecom Customer Churn project."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_FILE = RAW_DIR / "Telco-Customer-Churn.csv"
CLEANED_DATA_FILE = PROCESSED_DIR / "telco_churn_clean.csv"
FEATURES_DATA_FILE = PROCESSED_DIR / "telco_churn_features.csv"
MODEL_FILE = MODELS_DIR / "churn_model.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"
METRICS_FILE = REPORTS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET = "Churn"
ID_COLUMN = "customerID"
# Recall target used to pick the operating threshold on out-of-fold predictions.
TARGET_RECALL = 0.80

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
# Fields a user supplies for a single prediction (before feature engineering).
RAW_INPUT_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Post feature-engineering groups used by the preprocessing column transformer.
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlyCharges",
    "MonthlyChargeDelta",
    "NumServices",
    "FamilySize",
    "SupportServices",
    "StreamingServices",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
    "HasInternet",
    "IsNewCustomer",
]

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#0F766E",
    "secondary": "#0E7490",
    "accent": "#E76F51",
    "retained": "#2A9D8F",
    "churned": "#E76F51",
    "neutral": "#64748B",
    "grid": "#E2E8F0",
}

CHURN_PALETTE = {"No": COLORS["retained"], "Yes": COLORS["churned"]}

RISK_BANDS = [
    (0.00, 0.30, "Low", "#2A9D8F"),
    (0.30, 0.60, "Moderate", "#F4A261"),
    (0.60, 1.01, "High", "#E76F51"),
]


def ensure_directories() -> None:
    """Create every directory the pipeline writes to."""
    for directory in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
