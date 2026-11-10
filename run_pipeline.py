"""End-to-end pipeline: clean -> engineer -> EDA -> train -> evaluate -> report."""

from __future__ import annotations

import time

from src.config import (
    CLEANED_DATA_FILE,
    FEATURES_DATA_FILE,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TARGET,
    ensure_directories,
)
from src.eda import run_eda
from src.evaluate import evaluate_and_save
from src.preprocessing import (
    clean_data,
    engineer_features,
    load_raw_data,
    prepare_features,
    split_data,
)
from src.report import generate_report
from src.train import train_models


def main() -> None:
    started = time.time()
    ensure_directories()

    print("=" * 72)
    print(" TELECOM CUSTOMER CHURN PREDICTION - END-TO-END PIPELINE")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load + clean
    # ------------------------------------------------------------------
    print("\n[1/6] Loading and cleaning data ...", flush=True)
    raw = load_raw_data()
    clean, clean_report = clean_data(raw)
    clean.to_csv(CLEANED_DATA_FILE, index=False)
    print(
        f"  raw: {clean_report['raw_rows']:,} rows | "
        f"clean: {clean_report['clean_rows']:,} rows | "
        f"churn rate: {clean_report['churn_rate']:.2%} | "
        f"TotalCharges imputed: {clean_report.get('missing_total_charges', 0)}"
    )

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    print("[2/6] Engineering features ...", flush=True)
    features = engineer_features(clean)
    features.to_csv(FEATURES_DATA_FILE, index=False)
    X = prepare_features(features)
    y = features[TARGET]
    print(f"  model input: {X.shape[0]:,} rows x {X.shape[1]} features")

    dataset_info = {
        "n_rows": int(len(features)),
        "raw_columns": int(raw.shape[1]),
        "n_features": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "churned_customers": int(y.sum()),
        "retained_customers": int((1 - y).sum()),
        "source": "IBM Telco Customer Churn",
    }

    # ------------------------------------------------------------------
    # 3. EDA
    # ------------------------------------------------------------------
    print("[3/6] Running EDA ...", flush=True)
    insights = run_eda(features)
    print(f"  7 figures -> {FIGURES_DIR}")

    # ------------------------------------------------------------------
    # 4. Train + tune
    # ------------------------------------------------------------------
    print("[4/6] Training and comparing models ...", flush=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    dataset_info.update(
        {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
        }
    )
    artifacts = train_models(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_info=dataset_info,
    )

    # ------------------------------------------------------------------
    # 5. Evaluate + save artefacts
    # ------------------------------------------------------------------
    print("\n[5/6] Evaluating and saving figures ...", flush=True)
    preprocessor = artifacts["best_pipeline"].named_steps["preprocessor"]
    dataset_info["encoded_features"] = int(
        len(preprocessor.get_feature_names_out())
    )
    metrics = evaluate_and_save(
        artifacts, y_test, clean_report, insights, dataset_info
    )
    report_path = generate_report(
        artifacts, metrics, insights, clean_report, dataset_info
    )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    tuned = artifacts["best_test_metrics_tuned"]
    print("\n" + "=" * 72)
    print(" RESULTS")
    print("=" * 72)
    print(artifacts["results"][
        ["Model", "CV ROC AUC", "Test Accuracy", "Test Recall", "Test F1", "Test ROC AUC"]
    ].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("-" * 72)
    print(f"Best model:        {artifacts['best_name']}")
    print(f"Threshold (tuned): {artifacts['threshold']:.2f}")
    print(f"Accuracy:          {tuned['accuracy']:.2%}")
    print(f"Recall:            {tuned['recall']:.2%}")
    print(f"Precision:         {tuned['precision']:.2%}")
    print(f"F1:                {tuned['f1']:.2%}")
    print(f"ROC-AUC:           {tuned['roc_auc']:.4f}")
    print(f"PR-AUC:            {tuned['pr_auc']:.4f}")
    print("-" * 72)
    print(f"Models:   {MODELS_DIR}")
    print(f"Reports:  {REPORTS_DIR}")
    print(f"Report:   {report_path}")
    print(f"Runtime:  {time.time() - started:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
