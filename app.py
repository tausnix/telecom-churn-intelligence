"""Interactive Streamlit demo for telecom churn prediction."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import (
    CLEANED_DATA_FILE,
    FIGURES_DIR,
    METRICS_FILE,
    REPORTS_DIR,
    RISK_BANDS,
)
from src.predict import load_assets, score_customer
from src.utils import load_json

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
    .stApp { background: #F8FAFC; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
    .hero {
        background: linear-gradient(120deg, #0F766E 0%, #0E7490 60%, #155E75 100%);
        border-radius: 20px;
        padding: 34px 38px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 16px 40px rgba(15, 118, 110, 0.22);
    }
    .hero h1 { color: white; font-size: 2.15rem; margin: 0 0 8px 0; letter-spacing: -0.5px; }
    .hero p { color: #CCFBF1; margin: 0; font-size: 1.02rem; }
    .hero .badges { margin-top: 14px; font-size: 0.86rem; color: #A7F3D0; letter-spacing: 0.2px; }
    .risk-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-left: 7px solid #0F766E;
        border-radius: 16px;
        padding: 24px 28px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
    }
    .risk-prob { font-size: 3rem; font-weight: 800; line-height: 1; margin: 6px 0 8px 0; }
    .risk-label { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 1.4px; color: #64748B; }
    .risk-badge {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        color: white; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.4px;
    }
    .driver {
        background: white; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 12px 15px; margin-bottom: 9px; font-size: 0.93rem;
        border-left: 4px solid #E76F51;
    }
    .action {
        background: #F0FDFA; border: 1px solid #99F6E4; border-radius: 12px;
        padding: 12px 15px; margin-bottom: 9px; font-size: 0.93rem;
    }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #E2E8F0; border-radius: 14px;
        padding: 14px 16px; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetricLabel"] { color: #64748B; }
    .section-title { font-weight: 750; font-size: 1.18rem; color: #0F172A; margin: 6px 0 12px 0; }
    footer { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_assets():
    return load_assets()


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    return load_json(METRICS_FILE)


@st.cache_data(show_spinner=False)
def get_comparison() -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / "model_comparison.csv")


@st.cache_data(show_spinner=False)
def get_data_preview(rows: int = 250) -> pd.DataFrame:
    return pd.read_csv(CLEANED_DATA_FILE).head(rows)


@st.cache_data(show_spinner=False)
def get_cleaned_data() -> pd.DataFrame:
    return pd.read_csv(CLEANED_DATA_FILE)


def risk_color(probability: float) -> tuple[str, str]:
    for low, high, label, color in RISK_BANDS:
        if low <= probability < high:
            return label, color
    return "High", RISK_BANDS[-1][3]


def risk_drivers(record: dict) -> list[str]:
    drivers = []
    if record["Contract"] == "Month-to-month":
        drivers.append("Month-to-month contract — no long-term commitment.")
    if record["tenure"] <= 12:
        drivers.append(f"Short tenure ({record['tenure']} months) — risk peaks in year one.")
    if record["InternetService"] == "Fiber optic":
        drivers.append("Fiber-optic service — the highest-churn internet segment.")
    if record["PaymentMethod"] == "Electronic check":
        drivers.append("Electronic-check payment — the highest-churn payment method.")
    if record.get("TechSupport") == "No":
        drivers.append("No tech-support add-on.")
    if record.get("OnlineSecurity") == "No":
        drivers.append("No online-security add-on.")
    if record["MonthlyCharges"] >= 80:
        drivers.append(f"High monthly bill (${record['MonthlyCharges']:.0f}).")
    if record["PaperlessBilling"] == "Yes":
        drivers.append("Paperless billing is associated with elevated churn.")
    service_count = sum(
        record[column] == "Yes"
        for column in [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
    )
    if service_count <= 2:
        drivers.append(f"Only {service_count} subscribed service(s) — low switching cost.")
    return drivers[:6]


def recommended_actions(record: dict) -> list[str]:
    actions = []
    if record["Contract"] == "Month-to-month":
        actions.append("Offer a one-year contract incentive or a short loyalty commitment.")
    if record["tenure"] <= 12:
        actions.append("Enrol the customer in the 90-day onboarding and check-in programme.")
    if record["InternetService"] == "Fiber optic":
        actions.append("Schedule a fiber service-quality and value review.")
    if record["PaymentMethod"] == "Electronic check":
        actions.append("Incentivise a switch to automatic payment.")
    if record.get("TechSupport") == "No":
        actions.append("Bundle tech support at a retention price.")
    if record["MonthlyCharges"] >= 80:
        actions.append("Review plan fit and offer a right-sized bundle.")
    if not actions:
        actions.append("Maintain engagement with loyalty rewards and periodic check-ins.")
    return actions[:4]


def render_hero(metrics: dict) -> None:
    tuned = metrics["best_model_test_metrics_tuned_threshold"]
    dataset = metrics["dataset"]
    st.markdown(
        f"""
        <div class="hero">
            <h1>Telecom Customer Churn Intelligence</h1>
            <p>Predict churn risk, understand the drivers, and prioritise retention outreach.</p>
            <div class="badges">
                {dataset['n_rows']:,} customers &nbsp;·&nbsp; 3 tuned models &nbsp;·&nbsp;
                best {metrics['best_model']} &nbsp;·&nbsp; ROC-AUC {tuned['roc_auc']:.3f} &nbsp;·&nbsp;
                recall {tuned['recall']:.1%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------
try:
    metadata, pipelines, thresholds = get_assets()
    metrics = get_metrics()
except FileNotFoundError as error:
    st.error(
        "Model artifacts were not found. Run `python run_pipeline.py` first, "
        f"then restart the app.\n\nDetails: {error}"
    )
    st.stop()

model_entries = metadata["models"]
model_names = [entry["name"] for entry in model_entries]
best_name = metadata["best_model"]

with st.sidebar:
    st.markdown("## 📞 Churn Intelligence")
    st.caption("Portfolio project · telecom retention analytics")
    st.divider()
    st.markdown("**Dataset**")
    st.write(f"**{metrics['dataset']['n_rows']:,}** customers")
    st.write(f"**{metrics['dataset']['churn_rate']:.1%}** churn rate")
    st.write(f"**{metrics['dataset']['n_features']}** model features")
    st.divider()
    st.markdown("**Best model**")
    st.write(f"**{best_name}**")
    st.write(f"Operating threshold **{metrics['operating_threshold']:.2f}**")
    st.write(f"ROC-AUC **{metrics['best_model_test_metrics_tuned_threshold']['roc_auc']:.3f}**")
    st.divider()
    st.markdown("**Stack**")
    st.caption("Python · pandas · scikit-learn · matplotlib · Streamlit")
    st.caption("IBM Telco Customer Churn dataset")

render_hero(metrics)
tabs = st.tabs(["🔮 Score a customer", "📈 Model performance", "🔍 Data insights"])

# ---------------------------------------------------------------------------
# Tab 1 — scoring
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="section-title">Customer profile</div>', unsafe_allow_html=True)
    st.caption(
        "Enter the customer's current profile. The model returns a churn probability "
        "and the key risk drivers."
    )

    with st.form("churn_form"):
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown("**Customer profile**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_label = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

        with col_right:
            st.markdown("**Contract & billing**")
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            monthly_charges = st.number_input(
                "Monthly charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0,
            )
            estimate_total = st.checkbox(
                "Auto-calculate total",
                value=True,
                help="Total charges = tenure × monthly charges.",
            )
            total_charges = st.number_input(
                "Total charges ($)",
                min_value=0.0,
                max_value=15000.0,
                value=float(round(monthly_charges * tenure, 2)),
                step=10.0,
                disabled=estimate_total,
            )
            if estimate_total:
                total_charges = float(round(monthly_charges * tenure, 2))

        st.markdown("**Services**")
        serv_phone, serv_internet, serv_addons = st.columns([1, 1, 1.6])
        with serv_phone:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines", ["No", "Yes"], disabled=(phone == "No")
            )
        with serv_internet:
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            no_internet = internet == "No"
        with serv_addons:
            addons = st.multiselect(
                "Add-on services",
                [
                    "Online security",
                    "Online backup",
                    "Device protection",
                    "Tech support",
                    "Streaming TV",
                    "Streaming movies",
                ],
                default=[],
                disabled=no_internet,
                help="Select every add-on the customer currently subscribes to.",
            )

        st.markdown("")
        model_name = st.selectbox(
            "Scoring model",
            model_names,
            index=model_names.index(best_name),
            help="All three models were tuned on the same training split.",
        )
        submitted = st.form_submit_button(
            "Predict churn risk", width="stretch", type="primary"
        )

    if submitted:
        selected_addons = set(addons)
        online_security = "Online security" in selected_addons
        online_backup = "Online backup" in selected_addons
        device_protection = "Device protection" in selected_addons
        tech_support = "Tech support" in selected_addons
        streaming_tv = "Streaming TV" in selected_addons
        streaming_movies = "Streaming movies" in selected_addons

        if internet == "No":
            online_security = online_backup = device_protection = False
            tech_support = streaming_tv = streaming_movies = False

        record = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_label == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": "No phone service" if phone == "No" else multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": "Yes" if online_security else ("No internet service" if no_internet else "No"),
            "OnlineBackup": "Yes" if online_backup else ("No internet service" if no_internet else "No"),
            "DeviceProtection": "Yes" if device_protection else ("No internet service" if no_internet else "No"),
            "TechSupport": "Yes" if tech_support else ("No internet service" if no_internet else "No"),
            "StreamingTV": "Yes" if streaming_tv else ("No internet service" if no_internet else "No"),
            "StreamingMovies": "Yes" if streaming_movies else ("No internet service" if no_internet else "No"),
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }

        with st.spinner("Scoring customer ..."):
            result = score_customer(
                record,
                pipelines[model_name],
                metadata,
                thresholds[model_name],
            )

        probability = result["churn_probability"]
        band, color = risk_color(probability)
        decision = (
            "Flagged for retention outreach"
            if result["prediction"] == 1
            else "Below the outreach threshold"
        )
        st.markdown("")
        st.markdown(
            f"""
            <div class="risk-card" style="border-left-color:{color};">
                <div class="risk-label">Predicted churn probability</div>
                <div class="risk-prob" style="color:{color};">{probability:.1%}</div>
                <span class="risk-badge" style="background:{color};">{band} risk</span>
                &nbsp;&nbsp;<strong>{decision}</strong>
                <div style="color:#64748B;font-size:0.88rem;margin-top:8px;">
                    {model_name} · decision threshold {result['threshold']:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-title">Key risk signals</div>', unsafe_allow_html=True)
            drivers = risk_drivers(record)
            if drivers:
                for driver in drivers:
                    st.markdown(f'<div class="driver">{driver}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="driver">No major risk flags in this profile.</div>',
                    unsafe_allow_html=True,
                )
        with right:
            st.markdown('<div class="section-title">Suggested retention actions</div>', unsafe_allow_html=True)
            for action in recommended_actions(record):
                st.markdown(f'<div class="action">✓ &nbsp;{action}</div>', unsafe_allow_html=True)

    st.divider()
    st.caption(
        "The score is a decision-support signal, not a certainty. Use it to rank "
        "customers and combine it with account context and retention capacity."
    )

# ---------------------------------------------------------------------------
# Tab 2 — model performance
# ---------------------------------------------------------------------------
with tabs[1]:
    tuned = metrics["best_model_test_metrics_tuned_threshold"]
    default = metrics["best_model_test_metrics_default_threshold"]

    st.markdown('<div class="section-title">Model performance on the held-out test set</div>', unsafe_allow_html=True)
    st.caption(
        f"{metrics['best_model']} at the recall-targeted threshold "
        f"{metrics['operating_threshold']:.2f}. The test set was not used for tuning or "
        "threshold selection."
    )
    values = [
        ("Accuracy", f"{tuned['accuracy']:.1%}"),
        ("Recall", f"{tuned['recall']:.1%}"),
        ("Precision", f"{tuned['precision']:.1%}"),
        ("F1", f"{tuned['f1']:.1%}"),
        ("ROC-AUC", f"{tuned['roc_auc']:.3f}"),
        ("PR-AUC", f"{tuned['pr_auc']:.3f}"),
    ]
    for row_start in (0, 3):
        metric_cols = st.columns(3)
        for column, (label, value) in zip(metric_cols, values[row_start : row_start + 3]):
            column.metric(label, value)

    st.markdown("")
    st.markdown("**Model comparison @ 0.50 threshold**")
    comparison = get_comparison()
    visible = comparison[
        [
            "Model",
            "CV ROC AUC",
            "Test Accuracy",
            "Test Recall",
            "Test F1",
            "Test ROC AUC",
        ]
    ].copy()
    st.dataframe(
        visible.style.format(
            {
                "CV ROC AUC": "{:.3f}",
                "Test Accuracy": "{:.1%}",
                "Test Recall": "{:.1%}",
                "Test F1": "{:.1%}",
                "Test ROC AUC": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Precision, PR-AUC and the full comparison are included in the downloadable CSV."
    )
    st.download_button(
        "Download comparison CSV",
        data=comparison.to_csv(index=False),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    st.markdown("")
    st.markdown("**Operating points**")
    st.write(
        f"Default threshold 0.50: **{default['accuracy']:.1%}** accuracy, "
        f"**{default['recall']:.1%}** recall."
    )
    st.write(
        f"Recall-targeted threshold {metrics['operating_threshold']:.2f}: "
        f"**{tuned['accuracy']:.1%}** accuracy, **{tuned['recall']:.1%}** recall."
    )
    st.info(
        "Churn is imbalanced (26.5% positive). Accuracy alone is misleading, so "
        "the operating threshold is chosen to catch at least 80% of churners "
        "in cross-validation.",
        icon="ℹ️",
    )

    curve_col_1, curve_col_2 = st.columns(2)
    with curve_col_1:
        st.image(str(FIGURES_DIR / "10_roc_curves.png"), width="stretch")
    with curve_col_2:
        st.image(str(FIGURES_DIR / "11_precision_recall_curves.png"), width="stretch")

    st.image(str(FIGURES_DIR / "09_confusion_matrices.png"), width="stretch")
    st.image(str(FIGURES_DIR / "12_threshold_tuning.png"), width="stretch")

    st.markdown('<div class="section-title">What drives churn?</div>', unsafe_allow_html=True)
    st.image(str(FIGURES_DIR / "13_feature_importance_rf.png"), width="stretch")
    st.image(str(FIGURES_DIR / "14_logistic_coefficients.png"), width="stretch")

# ---------------------------------------------------------------------------
# Tab 3 — data insights
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="section-title">Exploratory data analysis</div>', unsafe_allow_html=True)
    dataset = metrics["dataset"]
    overview_cols = st.columns(4)
    overview_cols[0].metric("Customers", f"{dataset['n_rows']:,}")
    overview_cols[1].metric("Churn rate", f"{dataset['churn_rate']:.1%}")
    overview_cols[2].metric("Churned", f"{dataset['churned_customers']:,}")
    overview_cols[3].metric("Model features", f"{dataset['n_features']}")

    st.caption(
        "The dataset combines demographics, subscribed services, contract terms and "
        "billing. Churn is concentrated among new, month-to-month, fiber-optic and "
        "electronic-check customers."
    )

    for figure_name in (
        "01_churn_distribution.png",
        "02_churn_by_contract.png",
        "03_tenure_analysis.png",
        "04_monthly_charges_by_churn.png",
        "05_churn_by_internet_and_payment.png",
        "06_services_vs_churn.png",
        "07_correlation_heatmap.png",
    ):
        st.image(str(FIGURES_DIR / figure_name), width="stretch")

    st.markdown("")
    st.markdown("**Cleaned data preview**")
    st.dataframe(get_data_preview(), width="stretch", hide_index=True)
    st.download_button(
        "Download cleaned dataset",
        data=get_cleaned_data().to_csv(index=False),
        file_name="telco_churn_clean.csv",
        mime="text/csv",
    )
