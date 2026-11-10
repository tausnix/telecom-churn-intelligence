"""Exploratory data analysis: figures, segment insights and a markdown summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    CHURN_PALETTE,
    COLORS,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    TARGET,
)
from src.utils import pct, save_figure, setup_plot_style

CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
INTERNET_ORDER = ["DSL", "Fiber optic", "No"]
PAYMENT_ORDER = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
TENURE_ORDER = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]


def _churn_rate_table(df: pd.DataFrame, column: str, order: list[str] | None = None):
    table = (
        df.groupby(column, observed=True)[TARGET]
        .agg(customers="count", churn_rate="mean")
        .reset_index()
    )
    if order is not None:
        table[column] = pd.Categorical(table[column], categories=order, ordered=True)
        table = table.sort_values(column).dropna(subset=[column])
    table["churn_rate_pct"] = table["churn_rate"] * 100
    return table


def _annotate_bars(ax, bars, values, suffix="%"):
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}{suffix}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0F172A",
            xytext=(0, 4),
            textcoords="offset points",
        )


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------
def _plot_churn_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().sort_index()
    retained, churned = int(counts.get(0, 0)), int(counts.get(1, 0))
    total = retained + churned
    rate = churned / total

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    bars = axes[0].bar(
        ["Retained", "Churned"],
        [retained, churned],
        color=[COLORS["retained"], COLORS["churned"]],
        width=0.6,
    )
    axes[0].bar_label(
        bars,
        labels=[f"{retained:,}\n({retained / total:.1%})", f"{churned:,}\n({rate:.1%})"],
        padding=5,
        fontsize=10,
        fontweight="bold",
        color="#0F172A",
    )
    axes[0].set_title("Customer churn split")
    axes[0].set_ylabel("Customers")
    axes[0].set_ylim(0, retained * 1.22)
    axes[0].grid(axis="x", visible=False)

    axes[1].pie(
        [retained, churned],
        colors=[COLORS["retained"], COLORS["churned"]],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2},
    )
    axes[1].text(
        0,
        0.08,
        f"{rate:.1%}",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=COLORS["churned"],
    )
    axes[1].text(
        0,
        -0.16,
        "churn rate",
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["neutral"],
    )
    axes[1].legend(
        ["Retained", "Churned"], loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2
    )

    fig.suptitle("Target distribution — churn is imbalanced", fontsize=14, y=1.02)
    save_figure(fig, "01_churn_distribution.png")


def _plot_contract(df: pd.DataFrame) -> None:
    table = _churn_rate_table(df, "Contract", CONTRACT_ORDER)
    colors = [
        COLORS["churned"] if value == table["churn_rate_pct"].max() else COLORS["primary"]
        for value in table["churn_rate_pct"]
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    bars = ax.bar(
        [f"{row.Contract}\n(n={row.customers:,})" for row in table.itertuples()],
        table["churn_rate_pct"],
        color=colors,
        width=0.58,
    )
    _annotate_bars(ax, bars, table["churn_rate_pct"])
    ax.set_title("Churn rate by contract type")
    ax.set_ylabel("Churn rate (%)")
    ax.set_ylim(0, max(table["churn_rate_pct"]) * 1.28)
    ax.grid(axis="x", visible=False)
    save_figure(fig, "02_churn_by_contract.png")


def _plot_tenure(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    plot_df = df.assign(ChurnLabel=df[TARGET].map({0: "No", 1: "Yes"}))
    sns.histplot(
        data=plot_df,
        x="tenure",
        hue="ChurnLabel",
        hue_order=["No", "Yes"],
        palette=CHURN_PALETTE,
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.35,
        ax=axes[0],
    )
    axes[0].set_title("Tenure distribution by churn status")
    axes[0].set_xlabel("Tenure (months)")
    axes[0].set_ylabel("Density")
    legend = axes[0].get_legend()
    if legend is not None:
        legend.set_title("Churn")

    table = _churn_rate_table(df, "TenureGroup", TENURE_ORDER)
    bars = axes[1].bar(
        [f"{row.TenureGroup}\n(n={row.customers:,})" for row in table.itertuples()],
        table["churn_rate_pct"],
        color=[COLORS["churned"], "#F4A261", "#5EAAA8", COLORS["primary"]],
        width=0.6,
    )
    _annotate_bars(axes[1], bars, table["churn_rate_pct"])
    axes[1].set_title("Churn rate by tenure group")
    axes[1].set_ylabel("Churn rate (%)")
    axes[1].set_ylim(0, max(table["churn_rate_pct"]) * 1.3)
    axes[1].grid(axis="x", visible=False)

    fig.suptitle("New customers are dramatically more likely to churn", y=1.02)
    fig.tight_layout()
    save_figure(fig, "03_tenure_analysis.png")


def _plot_monthly_charges(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    plot_df = df.assign(ChurnLabel=df[TARGET].map({0: "No", 1: "Yes"}))

    sns.kdeplot(
        data=plot_df,
        x="MonthlyCharges",
        hue="ChurnLabel",
        hue_order=["No", "Yes"],
        palette=CHURN_PALETTE,
        fill=True,
        common_norm=False,
        alpha=0.35,
        linewidth=2,
        ax=axes[0],
    )
    axes[0].set_title("Monthly charges distribution")
    axes[0].set_xlabel("Monthly charges ($)")
    axes[0].set_ylabel("Density")
    legend = axes[0].get_legend()
    if legend is not None:
        legend.set_title("Churn")

    sns.boxplot(
        data=plot_df,
        x="ChurnLabel",
        y="MonthlyCharges",
        hue="ChurnLabel",
        order=["No", "Yes"],
        hue_order=["No", "Yes"],
        palette=CHURN_PALETTE,
        width=0.45,
        linewidth=1.2,
        fliersize=2,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Monthly charges by churn status")
    axes[1].set_xlabel("Churned")
    axes[1].set_ylabel("Monthly charges ($)")
    axes[1].set_xticks([0, 1], ["No", "Yes"])

    fig.suptitle("Churners tend to pay higher monthly bills", y=1.02)
    fig.tight_layout()
    save_figure(fig, "04_monthly_charges_by_churn.png")


def _plot_service_and_payment(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    for ax, column, order, title in [
        (axes[0], "InternetService", INTERNET_ORDER, "Churn rate by internet service"),
        (axes[1], "PaymentMethod", PAYMENT_ORDER, "Churn rate by payment method"),
    ]:
        table = _churn_rate_table(df, column, order)
        colors = [
            COLORS["churned"] if value == table["churn_rate_pct"].max() else COLORS["secondary"]
            for value in table["churn_rate_pct"]
        ]
        bars = ax.bar(
            [f"{row[0]}\n(n={row.customers:,})" for row in table.itertuples(index=False)],
            table["churn_rate_pct"],
            color=colors,
            width=0.6,
        )
        _annotate_bars(ax, bars, table["churn_rate_pct"])
        ax.set_title(title)
        ax.set_ylabel("Churn rate (%)")
        ax.set_ylim(0, max(table["churn_rate_pct"]) * 1.3)
        ax.grid(axis="x", visible=False)

    fig.suptitle("Fiber optic and electronic-check customers churn most", y=1.03)
    fig.tight_layout()
    save_figure(fig, "05_churn_by_internet_and_payment.png")


def _plot_services(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    table = _churn_rate_table(df, "NumServices")
    bars = axes[0].bar(
        table["NumServices"].astype(int).astype(str),
        table["churn_rate_pct"],
        color=sns.color_palette("crest", n_colors=len(table)),
        width=0.65,
    )
    _annotate_bars(axes[0], bars, table["churn_rate_pct"])
    axes[0].set_title("Churn rate by number of subscribed services")
    axes[0].set_xlabel("Number of services")
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].set_ylim(0, max(table["churn_rate_pct"]) * 1.3)
    axes[0].grid(axis="x", visible=False)

    table2 = _churn_rate_table(df, "SupportServices")
    bars = axes[1].bar(
        table2["SupportServices"].astype(int).astype(str),
        table2["churn_rate_pct"],
        color=sns.color_palette("crest", n_colors=max(len(table2), 1)),
        width=0.6,
    )
    _annotate_bars(axes[1], bars, table2["churn_rate_pct"])
    axes[1].set_title("Churn rate by support add-ons")
    axes[1].set_xlabel("Support services subscribed (0–4)")
    axes[1].set_ylabel("Churn rate (%)")
    axes[1].set_ylim(0, max(table2["churn_rate_pct"]) * 1.3)
    axes[1].grid(axis="x", visible=False)

    fig.suptitle("Bundling services is associated with stronger retention", y=1.03)
    fig.tight_layout()
    save_figure(fig, "06_services_vs_churn.png")


def _plot_correlation(df: pd.DataFrame) -> None:
    columns = [column for column in NUMERIC_FEATURES if column in df.columns]
    corr = df[columns + [TARGET]].corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10.5, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.75},
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title("Correlation matrix — numeric and engineered features")
    fig.tight_layout()
    save_figure(fig, "07_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, write_summary: bool = True) -> dict:
    """Generate all EDA figures and return the key findings as a dict."""
    setup_plot_style()
    df = df.copy()
    if "TenureGroup" not in df.columns:
        raise ValueError("Run engineer_features() before run_eda().")

    _plot_churn_distribution(df)
    _plot_contract(df)
    _plot_tenure(df)
    _plot_monthly_charges(df)
    _plot_service_and_payment(df)
    _plot_services(df)
    _plot_correlation(df)

    contract = _churn_rate_table(df, "Contract", CONTRACT_ORDER)
    internet = _churn_rate_table(df, "InternetService", INTERNET_ORDER)
    payment = _churn_rate_table(df, "PaymentMethod", PAYMENT_ORDER)
    tenure_group = _churn_rate_table(df, "TenureGroup", TENURE_ORDER)
    services = _churn_rate_table(df, "NumServices")

    corr = df[[c for c in NUMERIC_FEATURES if c in df.columns] + [TARGET]].corr(
        numeric_only=True
    )[TARGET].drop(TARGET).sort_values()
    top_positive = [
        {"feature": name, "correlation": float(value)} for name, value in corr.tail(3)[::-1].items()
    ]
    top_negative = [
        {"feature": name, "correlation": float(value)} for name, value in corr.head(3).items()
    ]

    churned = df[df[TARGET] == 1]
    retained = df[df[TARGET] == 0]
    insights = {
        "customers": int(len(df)),
        "churned_customers": int(len(churned)),
        "retained_customers": int(len(retained)),
        "churn_rate": float(df[TARGET].mean()),
        "avg_tenure_churned": float(churned["tenure"].mean()),
        "avg_tenure_retained": float(retained["tenure"].mean()),
        "median_tenure_churned": float(churned["tenure"].median()),
        "median_tenure_retained": float(retained["tenure"].median()),
        "avg_monthly_churned": float(churned["MonthlyCharges"].mean()),
        "avg_monthly_retained": float(retained["MonthlyCharges"].mean()),
        "contract": contract.to_dict(orient="records"),
        "internet": internet.to_dict(orient="records"),
        "payment": payment.to_dict(orient="records"),
        "tenure_group": tenure_group.to_dict(orient="records"),
        "services": services.to_dict(orient="records"),
        "top_positive_correlations": top_positive,
        "top_negative_correlations": top_negative,
    }

    if write_summary:
        _write_summary(insights)

    return insights


def _format_table(table: pd.DataFrame, column: str) -> str:
    header = f"| {column} | Customers | Churn rate |\n|---|---:|---:|\n"
    rows = [
        f"| {row[0]} | {int(row.customers):,} | {row.churn_rate_pct:.1f}% |"
        for row in table.itertuples(index=False)
    ]
    return header + "\n".join(rows)


def _write_summary(insights: dict) -> Path:
    path = REPORTS_DIR / "eda_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    positive = ", ".join(
        f"`{item['feature']}` ({item['correlation']:+.2f})"
        for item in insights["top_positive_correlations"]
    )
    negative = ", ".join(
        f"`{item['feature']}` ({item['correlation']:+.2f})"
        for item in insights["top_negative_correlations"]
    )

    content = f"""# Exploratory Data Analysis — Key Findings

## Dataset at a glance

- **{insights['customers']:,} customers** and **20 predictive features** (after removing the customer ID).
- **{insights['churned_customers']:,} customers churned ({pct(insights['churn_rate'])} churn rate)**.
- The target is imbalanced, so accuracy alone is misleading; recall, F1 and PR-AUC are the primary metrics.

## What separates churners from retained customers?

| Signal | Churned | Retained |
|---|---:|---:|
| Average tenure (months) | {insights['avg_tenure_churned']:.1f} | {insights['avg_tenure_retained']:.1f} |
| Median tenure (months) | {insights['median_tenure_churned']:.0f} | {insights['median_tenure_retained']:.0f} |
| Average monthly charges | ${insights['avg_monthly_churned']:.2f} | ${insights['avg_monthly_retained']:.2f} |

- Strongest **positive** correlations with churn: {positive}.
- Strongest **negative** correlations with churn: {negative}.

## Contract type

{_format_table(pd.DataFrame(insights['contract']), 'Contract')}

Month-to-month customers churn at a far higher rate than customers on one- or two-year contracts. Contract commitment is the single strongest categorical retention lever in the dataset.

## Internet service

{_format_table(pd.DataFrame(insights['internet']), 'Internet service')}

Fiber-optic customers churn more than DSL customers despite paying more, suggesting a perceived value or service-quality gap.

## Payment method

{_format_table(pd.DataFrame(insights['payment']), 'Payment method')}

Electronic-check customers churn the most; automatic payment methods are associated with materially lower churn.

## Tenure

{_format_table(pd.DataFrame(insights['tenure_group']), 'Tenure group')}

Churn risk is concentrated in the first year. Customers who survive the first 12 months become progressively more loyal.

## Services

{_format_table(pd.DataFrame(insights['services']), 'Services subscribed')}

Customers with more services and more support add-ons churn less, supporting bundle-based retention offers.

## Figures

All figures are saved in `reports/figures/`:

1. `01_churn_distribution.png` — target imbalance
2. `02_churn_by_contract.png` — contract-level churn
3. `03_tenure_analysis.png` — tenure distributions and buckets
4. `04_monthly_charges_by_churn.png` — billing differences
5. `05_churn_by_internet_and_payment.png` — service/payment segments
6. `06_services_vs_churn.png` — service bundling effect
7. `07_correlation_heatmap.png` — numeric feature relationships
"""
    path.write_text(content, encoding="utf-8")
    return path
