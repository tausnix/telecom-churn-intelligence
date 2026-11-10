"""Model training, hyper-parameter tuning and model selection."""

from __future__ import annotations

import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    METADATA_FILE,
    MODEL_FILE,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RAW_INPUT_FEATURES,
    TARGET_RECALL,
)
from src.metrics import classification_metrics, find_best_threshold
from src.preprocessing import build_preprocessor
from src.utils import save_json, slugify

# SVC(probability=True) is deprecated in scikit-learn 1.9 but remains the
# simplest calibrated-probability option for this project; ignore the notice.
warnings.filterwarnings(
    "ignore",
    message="The `probability` parameter was deprecated",
    category=FutureWarning,
)


def build_model_specs() -> dict:
    """Define the three candidate algorithms and their tuning grids."""
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "params": {"model__C": [0.01, 0.1, 1.0, 10.0]},
        },
        "Support Vector Machine": {
            "estimator": SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model__C": [0.5, 1.0, 5.0],
                "model__gamma": ["scale"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "params": {
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [1, 4],
            },
        },
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_info: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Tune, fit and compare all candidate models.

    Model selection uses a one-standard-error rule on cross-validated ROC AUC:
    models statistically tied with the leader are compared on CV F1, which
    better reflects the imbalanced precision/recall objective. The operating
    threshold of the winning model is then tuned on out-of-fold training
    predictions to hit a recall SLA of ``TARGET_RECALL``, leaving the test set
    untouched.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    specs = build_model_specs()

    results: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    test_metrics: dict[str, dict] = {}
    cv_metrics: dict[str, dict] = {}

    for name, spec in specs.items():
        if verbose:
            print(f"\n[{name}] tuning ...", flush=True)
        started = time.time()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", spec["estimator"]),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["params"],
            cv=cv,
            scoring={"roc_auc": "roc_auc", "f1": "f1", "recall": "recall"},
            refit="roc_auc",
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_train, y_train)

        best_index = search.best_index_
        scores = {
            metric: {
                "mean": float(search.cv_results_[f"mean_test_{metric}"][best_index]),
                "std": float(search.cv_results_[f"std_test_{metric}"][best_index]),
            }
            for metric in ("roc_auc", "f1", "recall")
        }
        cv_metrics[name] = scores

        best_model = search.best_estimator_
        proba = best_model.predict_proba(X_test)[:, 1]
        default_metrics = classification_metrics(y_test, proba, threshold=0.5)

        fitted_models[name] = best_model
        test_probabilities[name] = proba
        test_metrics[name] = default_metrics

        best_params = {
            key.replace("model__", ""): value
            for key, value in search.best_params_.items()
        }
        results.append(
            {
                "Model": name,
                "CV ROC AUC": scores["roc_auc"]["mean"],
                "CV F1": scores["f1"]["mean"],
                "CV Recall": scores["recall"]["mean"],
                "Test Accuracy": default_metrics["accuracy"],
                "Test Precision": default_metrics["precision"],
                "Test Recall": default_metrics["recall"],
                "Test F1": default_metrics["f1"],
                "Test ROC AUC": default_metrics["roc_auc"],
                "Test PR AUC": default_metrics["pr_auc"],
                "Best Params": str(best_params),
                "Fit Seconds": round(time.time() - started, 1),
            }
        )
        if verbose:
            print(
                f"  best params: {best_params}\n"
                f"  CV ROC-AUC: {scores['roc_auc']['mean']:.4f} ± "
                f"{scores['roc_auc']['std']:.4f} | "
                f"test recall@0.50: {default_metrics['recall']:.4f} | "
                f"test F1@0.50: {default_metrics['f1']:.4f}",
                flush=True,
            )

    results_df = (
        pd.DataFrame(results)
        .sort_values(["CV ROC AUC", "CV Recall"], ascending=False)
        .reset_index(drop=True)
    )

    # One-standard-error rule: models within 1 SE of the best CV ROC-AUC are
    # statistically tied, so prefer the one with the strongest CV F1.
    leader = results_df.iloc[0]
    one_se_floor = leader["CV ROC AUC"] - cv_metrics[leader["Model"]]["roc_auc"]["std"]
    tied = results_df[results_df["CV ROC AUC"] >= one_se_floor]
    best_name = str(
        tied.sort_values(["CV F1", "CV ROC AUC"], ascending=False).iloc[0]["Model"]
    )
    best_pipeline = fitted_models[best_name]

    if verbose:
        print(
            f"\nCV ROC-AUC leader: {leader['Model']} ({leader['CV ROC AUC']:.4f}); "
            f"one-SE floor: {one_se_floor:.4f}"
        )
        print(
            f"Selected model (highest CV F1 among tied models): {best_name}"
        )
        print("Tuning operating threshold on out-of-fold predictions ...", flush=True)

    oof_proba = cross_val_predict(
        best_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    # Recall-first operating point: hit the recall SLA on out-of-fold data,
    # then choose the most precise threshold that satisfies it.
    threshold, threshold_df = find_best_threshold(
        y_train,
        oof_proba,
        min_recall=TARGET_RECALL,
        objective="recall",
    )

    best_proba = test_probabilities[best_name]
    tuned_metrics = classification_metrics(y_test, best_proba, threshold=threshold)
    default_metrics_best = test_metrics[best_name]

    if verbose:
        print(
            f"  chosen threshold: {threshold:.3f}\n"
            f"  test accuracy: {tuned_metrics['accuracy']:.4f} | "
            f"recall: {tuned_metrics['recall']:.4f} | "
            f"precision: {tuned_metrics['precision']:.4f} | "
            f"F1: {tuned_metrics['f1']:.4f}",
            flush=True,
        )

    # Persist every model (the app lets users compare them) plus the winner.
    for name, pipeline in fitted_models.items():
        joblib.dump(pipeline, MODELS_DIR / f"{slugify(name)}.joblib")
    joblib.dump(best_pipeline, MODEL_FILE)

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": best_name,
        "threshold": threshold,
        "input_features": RAW_INPUT_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "dataset": dataset_info or {},
        "models": [
            {
                "name": row["Model"],
                "slug": slugify(row["Model"]),
                "path": f"models/{slugify(row['Model'])}.joblib",
                "threshold": threshold
                if row["Model"] == best_name
                else 0.5,
                "cv_metrics": cv_metrics[row["Model"]],
                "test_metrics": test_metrics[row["Model"]],
            }
            for _, row in results_df.iterrows()
        ],
    }
    save_json(metadata, METADATA_FILE)

    return {
        "results": results_df,
        "fitted_models": fitted_models,
        "best_name": best_name,
        "best_pipeline": best_pipeline,
        "threshold": threshold,
        "threshold_table": threshold_df,
        "test_probabilities": test_probabilities,
        "test_metrics_default": test_metrics,
        "best_test_metrics_default": default_metrics_best,
        "best_test_metrics_tuned": tuned_metrics,
        "cv_metrics": cv_metrics,
    }
