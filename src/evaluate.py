"""Model evaluation: comparison figures, curves, importances and metrics export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import (
    COLORS,
    REPORTS_DIR,
    TARGET_RECALL,
)
from src.utils import pct, save_figure, save_json, setup_plot_style

MODEL_COLORS = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]


def _rf_importance(pipeline) -> pd.Series:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    return pd.Series(model.feature_importances_, index=names)


def _lr_coefficients(pipeline) -> pd.Series:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    return pd.Series(model.coef_[0], index=names)


def plot_model_comparison(results: pd.DataFrame) -> None:
    metrics = ["Test Accuracy", "Test Recall", "Test F1", "Test ROC AUC"]
    plot_df = results.melt(
        id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score"
    )
    plot_df["Metric"] = plot_df["Metric"].str.replace("Test ", "", regex=False)

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=MODEL_COLORS,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8.5, color="#0F172A")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Model comparison on the held-out test set (threshold = 0.50)")
    ax.legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
    )
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, "08_model_comparison.png")


def plot_confusion_matrices(
    probabilities: dict[str, np.ndarray], y_test
) -> None:
    fig, axes = plt.subplots(1, len(probabilities), figsize=(14, 4.2))
    for ax, (name, proba) in zip(axes, probabilities.items()):
        predictions = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Retained", "Churned"]
        )
        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.grid(False)
    fig.suptitle(
        "Confusion matrices on the test set (default 0.50 threshold)", y=1.04
    )
    fig.tight_layout()
    save_figure(fig, "09_confusion_matrices.png")


def plot_roc_curves(probabilities: dict[str, np.ndarray], y_test) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6))
    for color, (name, proba) in zip(MODEL_COLORS, probabilities.items()):
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_value = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{name} (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — all candidate models")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    save_figure(fig, "10_roc_curves.png")


def plot_pr_curves(probabilities: dict[str, np.ndarray], y_test) -> None:
    baseline = float(np.mean(y_test))
    fig, ax = plt.subplots(figsize=(7.4, 6))
    for color, (name, proba) in zip(MODEL_COLORS, probabilities.items()):
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(recall, precision, color=color, linewidth=2.2, label=f"{name} (AP = {ap:.3f})")
    ax.axhline(
        baseline,
        linestyle="--",
        color="#94A3B8",
        linewidth=1,
        label=f"Churn baseline ({baseline:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curves — all candidate models")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    save_figure(fig, "11_precision_recall_curves.png")


def plot_threshold_curve(threshold_df: pd.DataFrame, chosen: float) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", color=COLORS["secondary"])
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", color=COLORS["primary"])
    ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1", color=COLORS["accent"], linewidth=2.4)
    ax.axvline(chosen, color="#0F172A", linestyle="--", linewidth=1.2)
    ax.annotate(
        f"Selected threshold = {chosen:.2f}",
        xy=(chosen, 0.05),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=9,
        color="#0F172A",
        rotation=90,
        va="bottom",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(
        f"Threshold tuning on out-of-fold predictions (recall target = {TARGET_RECALL:.0%})"
    )
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    save_figure(fig, "12_threshold_tuning.png")


def plot_feature_importance(
    rf_pipeline, lr_pipeline, top_n: int = 15
) -> tuple[pd.Series, pd.Series]:
    rf_importance = _rf_importance(rf_pipeline)
    coefficients = _lr_coefficients(lr_pipeline)

    rf_top = rf_importance.sort_values(ascending=False).head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6.4))
    colors = sns.color_palette("crest", n_colors=len(rf_top))
    bars = ax.barh(rf_top.index, rf_top.values, color=colors, height=0.72)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5, color="#0F172A")
    ax.set_title("Random Forest — top feature importances")
    ax.set_xlabel("Importance")
    ax.set_xlim(0, rf_top.max() * 1.18)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "13_feature_importance_rf.png")

    coef_top = pd.concat(
        [coefficients.sort_values().head(10), coefficients.sort_values().tail(10)]
    ).sort_values()
    fig, ax = plt.subplots(figsize=(9.6, 7))
    colors = [COLORS["retained"] if value < 0 else COLORS["churned"] for value in coef_top.values]
    bars = ax.barh(coef_top.index, coef_top.values, color=colors, height=0.72)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5, color="#0F172A")
    ax.axvline(0, color="#475569", linewidth=0.9)
    ax.set_title("Logistic Regression — most influential coefficients")
    ax.set_xlabel("Coefficient (standardised numerics; positive ⇒ higher churn risk)")
    ax.set_xlim(coef_top.min() * 1.25, coef_top.max() * 1.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "14_logistic_coefficients.png")

    rf_importance.sort_values(ascending=False).to_csv(
        REPORTS_DIR / "feature_importance.csv", header=["importance"], index_label="feature"
    )
    coefficients.sort_values(ascending=False).to_csv(
        REPORTS_DIR / "logistic_coefficients.csv",
        header=["coefficient"],
        index_label="feature",
    )
    return rf_importance, coefficients


def plot_probability_distribution(proba: np.ndarray, y_test, threshold: float) -> None:
    plot_df = pd.DataFrame(
        {"Probability": proba, "Churn": pd.Series(y_test).map({0: "No", 1: "Yes"}).values}
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.histplot(
        data=plot_df,
        x="Probability",
        hue="Churn",
        hue_order=["No", "Yes"],
        palette={"No": COLORS["retained"], "Yes": COLORS["churned"]},
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.3,
        legend=False,
        ax=ax,
    )
    legend_handles = [
        Patch(facecolor=COLORS["retained"], alpha=0.5, label="No"),
        Patch(facecolor=COLORS["churned"], alpha=0.5, label="Yes"),
    ]
    ax.legend(handles=legend_handles, title="Actual churn", frameon=False)
    ax.axvline(threshold, color="#0F172A", linestyle="--", linewidth=1.2)
    ax.annotate(
        f"threshold {threshold:.2f}",
        xy=(threshold, ax.get_ylim()[1] * 0.9),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_title("Predicted churn probability by actual outcome (test set)")
    ax.set_xlabel("Predicted probability of churn")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    save_figure(fig, "15_churn_probability_distribution.png")


def evaluate_and_save(
    artifacts: dict,
    y_test,
    clean_report: dict,
    eda_insights: dict,
    dataset_info: dict,
) -> dict:
    """Create every evaluation artefact and return the final metrics bundle."""
    setup_plot_style()
    results = artifacts["results"].copy()
    probabilities = artifacts["test_probabilities"]
    best_name = artifacts["best_name"]
    best_pipeline = artifacts["best_pipeline"]
    threshold = artifacts["threshold"]
    tuned = artifacts["best_test_metrics_tuned"]
    default_best = artifacts["best_test_metrics_default"]

    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    artifacts["threshold_table"].to_csv(
        REPORTS_DIR / "threshold_tuning.csv", index=False
    )

    plot_model_comparison(results)
    plot_confusion_matrices(probabilities, y_test)
    plot_roc_curves(probabilities, y_test)
    plot_pr_curves(probabilities, y_test)
    plot_threshold_curve(artifacts["threshold_table"], threshold)
    plot_probability_distribution(probabilities[best_name], y_test, threshold)
    plot_feature_importance(
        artifacts["fitted_models"]["Random Forest"],
        artifacts["fitted_models"]["Logistic Regression"],
    )

    metrics = {
        "dataset": dataset_info,
        "cleaning": clean_report,
        "eda": eda_insights,
        "model_comparison": results.to_dict(orient="records"),
        "best_model": best_name,
        "operating_threshold": threshold,
        "best_model_test_metrics_default_threshold": default_best,
        "best_model_test_metrics_tuned_threshold": tuned,
        "cv_metrics": artifacts["cv_metrics"],
        "runtime": {
            row["Model"]: row["Fit Seconds"] for _, row in results.iterrows()
        },
    }
    save_json(metrics, REPORTS_DIR / "metrics.json")
    return metrics
