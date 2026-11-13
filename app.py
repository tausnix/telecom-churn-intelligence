"""Telecom Churn Intelligence — interactive retention console.

    streamlit run app.py

Layout follows the same visual language as the PromoFinder app: a dark shell,
a control panel in the sidebar, and a split view (scored customers on the left,
the selected customer's case file on the right).
"""

from __future__ import annotations

import json
from datetime import datetime

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
    page_title="Churn Intelligence",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- theme
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp { font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; }
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] *:not([class*="material"]),
.stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricDelta"], .stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stExpander"] summary,
.stApp button, .stApp input, .stApp textarea, .stApp [data-baseweb="select"] *, .stApp [role="tab"],
.stApp [data-testid="stDataFrame"] { font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; }
.stApp { background: radial-gradient(1100px 520px at 18% -12%, #1a2030 0%, #0e1117 48%) fixed; }
section[data-testid="stSidebar"] { background: #0a0d13; border-right: 1px solid #1d2330; }
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px; }

.brand { font-size: 2.0rem; font-weight: 800; letter-spacing: -.045em; line-height: 1.1; }
.brand .dot { color: #FF4B4B; }
.brand .slash { color: #5b6472; font-weight: 500; margin: 0 .5rem; }
.brand .sub { color: #f3f4f6; font-weight: 700; letter-spacing: -.025em; }
.stats { color: #9ca3af; font-size: .8rem; margin: .3rem 0 1.0rem; }

.side-title { font-size: 1.05rem; font-weight: 800; letter-spacing: -.02em; margin-bottom: .35rem; }
.side-copy { color: #9ca3af; font-size: .8rem; line-height: 1.45; margin-bottom: .9rem; }
.side-h { color: #e5e7eb; font-size: .74rem; font-weight: 800; letter-spacing: .1em;
          text-transform: uppercase; margin: 1.05rem 0 .3rem; }

.section-label { color: #6b7280; font-size: .66rem; letter-spacing: .15em; text-transform: uppercase;
                 font-weight: 800; margin: .1rem 0 .55rem; }
.kicker { color: #FF4B4B; font-size: .7rem; letter-spacing: .16em; font-weight: 800;
          text-transform: uppercase; }
.title { font-size: 1.5rem; font-weight: 800; letter-spacing: -.035em; margin: .1rem 0 .45rem; }
.badge { display: inline-block; border: 1px solid #FF4B4B; color: #ff9a9a; border-radius: 999px;
         padding: .1rem .55rem; font-size: .68rem; font-weight: 700; }
.badge.gray { border-color: #374151; color: #9ca3af; }

.label { color: #6b7280; font-size: .64rem; letter-spacing: .15em; text-transform: uppercase;
         font-weight: 800; margin-top: .95rem; }
.ts { color: #9ca3af; font-size: .74rem; margin-bottom: .55rem; }
.body { color: #cbd5e1; font-size: .85rem; line-height: 1.5; }

.prob { font-size: 2.9rem; font-weight: 800; letter-spacing: -.05em; line-height: 1; margin: .15rem 0 .1rem; }
.prob-cap { color: #9ca3af; font-size: .74rem; margin-bottom: .8rem; }

.item { border: 1px solid #1f2531; border-left: 3px solid #FF4B4B; border-radius: .45rem;
        background: #0d121b; padding: .5rem .65rem; margin-bottom: .4rem;
        color: #cbd5e1; font-size: .82rem; line-height: 1.4; }
.item.act { border-left-color: #14b8a6; }

.meter { position: relative; height: 8px; background: #1f2531; border-radius: 999px; margin: .5rem 0 .15rem; }
.meter-fill { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 999px; }
.meter-mark { position: absolute; top: -4px; bottom: -4px; width: 2px; background: #f9fafb; }

.step { display: flex; gap: .7rem; margin-bottom: 1rem; }
.step-num { flex: 0 0 1.35rem; width: 1.35rem; height: 1.35rem; border: 1px solid #FF4B4B; color: #FF4B4B;
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            font-size: .7rem; font-weight: 800; }
.step-title { font-weight: 700; color: #f3f4f6; font-size: .88rem; margin-bottom: .12rem; }
.step-body { color: #9ca3af; font-size: .79rem; line-height: 1.45; }

div[data-testid="stButton"] > button[kind="primary"],
button[data-testid="stBaseButton-primary"] { background: #FF4B4B; border: 0; color: #fff; font-weight: 700;
    border-radius: .55rem; }
div[data-testid="stButton"] > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover { background: #ff6464; border: 0; color: #fff; }
[data-testid="stDataFrame"] { border: 1px solid #1f2531; border-radius: .7rem; overflow: hidden; }
[data-testid="stImage"] img { border: 1px solid #1f2531; border-radius: .6rem; }
div[data-testid="stExpander"] { border: 1px solid #1f2531; border-radius: .7rem; background: #0d121b; }
[data-testid="stMetricValue"] { font-size: 1.35rem; }
[data-testid="stMetricLabel"] { color: #9ca3af; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

BRAND = (
    '<div class="brand"><span>churn<span class="dot">.</span></span>'
    '<span class="slash">/</span><span class="sub">Intelligence</span></div>'
)

# Dark-theme risk palette (thresholds still come from src/config.py).
BAND_COLORS = {"Low": "#2DD4BF", "Moderate": "#F5B841", "High": "#FF4B4B"}

PRESETS: dict[str, dict] = {
    "Fiber optic · new customer": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "addons": [],
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0,
    },
    "DSL · mid-tenure": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 24, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "addons": ["Online security", "Tech support"],
        "Contract": "One year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 62.0,
    },
    "Two-year · loyal": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 58, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL",
        "addons": ["Online security", "Online backup", "Device protection",
                   "Tech support", "Streaming TV", "Streaming movies"],
        "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 88.0,
    },
    "Voice only · no internet": {
        "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "tenure": 14, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "No", "addons": [],
        "Contract": "Month-to-month", "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 25.0,
    },
}
ADDONS = [
    "Online security", "Online backup", "Device protection",
    "Tech support", "Streaming TV", "Streaming movies",
]
ADDON_LABELS = {
    "Online security": "Security",
    "Online backup": "Backup",
    "Device protection": "Protection",
    "Tech support": "Support",
    "Streaming TV": "TV",
    "Streaming movies": "Movies",
}
ADDON_FIELDS = {
    "Online security": "OnlineSecurity",
    "Online backup": "OnlineBackup",
    "Device protection": "DeviceProtection",
    "Tech support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming movies": "StreamingMovies",
}


# --------------------------------------------------------------------------- data helpers
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


def band_of(probability: float) -> str:
    for low, high, label, _ in RISK_BANDS:
        if low <= probability < high:
            return label
    return "High"


def step(n: int, title: str, body: str) -> None:
    st.markdown(
        f'<div class="step"><span class="step-num">{n}</span><div>'
        f'<div class="step-title">{title}</div><div class="step-body">{body}</div></div></div>',
        unsafe_allow_html=True,
    )


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
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
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


# --------------------------------------------------------------------------- load artefacts
try:
    metadata, pipelines, thresholds = get_assets()
    metrics = get_metrics()
except FileNotFoundError as error:
    st.markdown(BRAND, unsafe_allow_html=True)
    st.error(
        "Model artifacts were not found. Run `python run_pipeline.py` first, "
        f"then restart the app.\n\nDetails: {error}"
    )
    st.stop()

model_names = [entry["name"] for entry in metadata["models"]]
best_name = metadata["best_model"]
tuned = metrics["best_model_test_metrics_tuned_threshold"]
default_point = metrics["best_model_test_metrics_default_threshold"]
dataset = metrics["dataset"]
history: list[dict] = st.session_state.setdefault("history", [])

# --------------------------------------------------------------------------- sidebar controls
with st.sidebar:
    st.markdown('<div class="side-title">Score a customer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-copy">Pick a preset or set the profile by hand. The model returns '
        'a churn probability, the risk band and the drivers behind the score.</div>',
        unsafe_allow_html=True,
    )

    preset_name = st.selectbox("Preset", list(PRESETS), label_visibility="collapsed")
    preset = PRESETS[preset_name]
    key = list(PRESETS).index(preset_name)

    st.markdown('<div class="side-h">Profile</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    gender = c1.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(preset["gender"]), key=f"g{key}")
    senior = c2.selectbox("Senior", ["No", "Yes"], index=preset["SeniorCitizen"], key=f"s{key}")
    c3, c4 = st.columns(2)
    partner = c3.selectbox("Partner", ["No", "Yes"], index=["No", "Yes"].index(preset["Partner"]), key=f"p{key}")
    dependents = c4.selectbox("Dependents", ["No", "Yes"], index=["No", "Yes"].index(preset["Dependents"]), key=f"d{key}")
    tenure = st.slider("Tenure (months)", 0, 72, preset["tenure"], key=f"t{key}")

    st.markdown('<div class="side-h">Contract &amp; billing</div>', unsafe_allow_html=True)
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"],
        index=["Month-to-month", "One year", "Two year"].index(preset["Contract"]), key=f"c{key}",
    )
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=["Electronic check", "Mailed check", "Bank transfer (automatic)",
               "Credit card (automatic)"].index(preset["PaymentMethod"]),
        key=f"pm{key}",
    )
    c5, c6 = st.columns(2)
    paperless = c5.selectbox("Paperless", ["Yes", "No"], index=["Yes", "No"].index(preset["PaperlessBilling"]), key=f"pb{key}")
    monthly = c6.number_input("Monthly ($)", 0.0, 200.0, float(preset["MonthlyCharges"]), 1.0, key=f"m{key}")
    auto_total = st.checkbox("Total = tenure × monthly", value=True, key=f"at{key}")
    total = float(round(monthly * tenure, 2)) if auto_total else st.number_input(
        "Total charges ($)", 0.0, 15000.0, float(round(monthly * tenure, 2)), 10.0, key=f"tc{key}"
    )

    st.markdown('<div class="side-h">Services</div>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    phone = c7.selectbox("Phone", ["Yes", "No"], index=["Yes", "No"].index(preset["PhoneService"]), key=f"ph{key}")
    lines = c8.selectbox(
        "Lines", ["No", "Yes"], index=["No", "Yes"].index(preset["MultipleLines"]),
        disabled=(phone == "No"), key=f"ml{key}",
    )
    internet = st.selectbox(
        "Internet", ["DSL", "Fiber optic", "No"],
        index=["DSL", "Fiber optic", "No"].index(preset["InternetService"]), key=f"i{key}",
    )
    st.markdown('<div class="side-h" style="margin-top:.8rem">Add-ons</div>', unsafe_allow_html=True)
    no_net = internet == "No"
    addon_flags: dict[str, bool] = {}
    a1, a2 = st.columns(2)
    for i, label in enumerate(ADDONS):
        column = a1 if i % 2 == 0 else a2
        addon_flags[label] = column.checkbox(
            ADDON_LABELS[label], value=label in preset["addons"],
            disabled=no_net, key=f"a{key}{i}",
        )
    addons = [label for label, enabled in addon_flags.items() if enabled]

    st.markdown('<div class="side-h">Decision</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Threshold", 0.05, 0.95, float(round(metrics["operating_threshold"], 2)), 0.005,
        key="threshold",
        help="Tuned on out-of-fold predictions to catch at least 80% of churners.",
    )
    st.caption(f"Tuned operating point {metrics['operating_threshold']:.3f} · 80% recall target")

    predict = st.button("Predict churn risk", type="primary", width="stretch")

# --------------------------------------------------------------------------- header
head_l, head_r = st.columns([2.0, 1.35], vertical_alignment="center")
with head_l:
    st.markdown(BRAND, unsafe_allow_html=True)
with head_r:
    model_name = st.selectbox(
        "Model", model_names, index=model_names.index(best_name), label_visibility="collapsed",
    )

# --------------------------------------------------------------------------- score on demand
if predict:
    selected = set(addons)
    no_net = internet == "No"
    record = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": "No phone service" if phone == "No" else lines,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }
    for label, column in ADDON_FIELDS.items():
        record[column] = (
            "Yes" if label in selected else ("No internet service" if no_net else "No")
        )

    result = score_customer(record, pipelines[model_name], metadata, threshold)
    history.append(
        {
            "record": record,
            "model": model_name,
            "probability": result["churn_probability"],
            "threshold": result["threshold"],
            "prediction": result["prediction"],
            "band": band_of(result["churn_probability"]),
            "scored_at": datetime.now().strftime("%H:%M:%S"),
        }
    )
    st.session_state["selected_idx"] = len(history) - 1

st.markdown(
    f'<div class="stats">{dataset["n_rows"]:,} customers · {dataset["churn_rate"]:.1%} churn rate · '
    f'3 tuned models · {model_name} scores now · ROC-AUC {tuned["roc_auc"]:.3f} · '
    f'threshold tuned for ≥80% recall</div>',
    unsafe_allow_html=True,
)

with st.expander("How to use this app", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        step(1, "Score a customer on the left",
             "Pick a preset or set the profile, contract and services by hand. "
             "<b>Predict churn risk</b> adds the customer to the session log.")
        step(3, "Check the models",
             "The <b>Model performance</b> tab compares logistic regression, SVM and random "
             "forest at the same threshold, with ROC/PR curves, confusion matrices and the "
             "features that move the score.")
    with c2:
        step(2, "Read the case file on the right",
             "Probability, risk band, where it sits against the decision threshold, the signals "
             "that pushed it up and the retention actions worth trying first.")
        step(4, "Inspect the data",
             "The <b>Data insights</b> tab has the EDA gallery, the cleaned-data preview and "
             "downloads for both the dataset and the session log.")

tabs = st.tabs(["Score a customer", "Model performance", "Data insights"])

# --------------------------------------------------------------------------- tab 1
with tabs[0]:
    left, right = st.columns([1.08, 1], gap="large")

    with left:
        label_col, filter_col = st.columns([1.7, 1], vertical_alignment="center")
        with label_col:
            st.markdown('<div class="section-label">Scored customers</div>', unsafe_allow_html=True)
        with filter_col:
            flagged_only = st.toggle("Flagged", value=False, help="Show only customers above the threshold.")

        shown_idx = [
            i for i, item in enumerate(history)
            if (item["prediction"] == 1 if flagged_only else True)
        ]
        ordered = [(i, history[i]) for i in reversed(shown_idx)]

        if not ordered:
            st.info("No customers scored yet. Use the panel on the left to score the first one.")
        else:
            rows = [
                {
                    "#": str(i + 1),
                    "Profile": f"{item['record']['Contract']} · {item['record']['tenure']} mo",
                    "P(churn)": f"{item['probability']:.1%}",
                    "Band": item["band"],
                }
                for i, item in ordered
            ]
            event = st.dataframe(
                pd.DataFrame(rows), hide_index=True, width="stretch",
                height=min(520, 42 + 35 * len(rows)),
                on_select="rerun", selection_mode="single-row", key="history_table",
                column_config={
                    "#": st.column_config.TextColumn(width="small"),
                    "Profile": st.column_config.TextColumn(width="medium"),
                    "P(churn)": st.column_config.TextColumn(width="small"),
                    "Band": st.column_config.TextColumn(width="small"),
                },
            )
            picked = getattr(getattr(event, "selection", None), "rows", []) or []
            if picked and picked[0] < len(ordered):
                st.session_state["selected_idx"] = ordered[picked[0]][0]
            st.caption("Tick a row to open that customer on the right.")

            status = pd.DataFrame(
                [
                    {
                        "scored_at": item["scored_at"],
                        "contract": item["record"]["Contract"],
                        "tenure": item["record"]["tenure"],
                        "internet": item["record"]["InternetService"],
                        "payment": item["record"]["PaymentMethod"],
                        "monthly_charges": item["record"]["MonthlyCharges"],
                        "total_charges": item["record"]["TotalCharges"],
                        "churn_probability": round(item["probability"], 4),
                        "threshold": round(item["threshold"], 3),
                        "band": item["band"],
                        "flagged": item["prediction"],
                        "model": item["model"],
                    }
                    for item in history
                ]
            )
            st.download_button(
                f"Download session log ({len(history)})",
                status.to_csv(index=False), "churn_session_log.csv", "text/csv", width="stretch",
            )

    with right:
        if not history:
            st.markdown('<div class="section-label">Case file</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="body">Score a customer and this panel fills with the probability, '
                'the risk band, the signals that drove the score and the actions to prioritise.</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="label">What you get</div>', unsafe_allow_html=True)
            for line in (
                "Churn probability and risk band",
                "Position against the decision threshold",
                "The signals that pushed the score up",
                "Retention actions worth trying first",
            ):
                st.markdown(f'<div class="item act">{line}</div>', unsafe_allow_html=True)
        else:
            idx = st.session_state.get("selected_idx", len(history) - 1)
            idx = min(max(idx, 0), len(history) - 1)
            item = history[idx]
            record = item["record"]
            probability = item["probability"]
            band = item["band"]
            color = BAND_COLORS[band]
            decision = "Flagged for retention outreach" if item["prediction"] else "Below the outreach threshold"

            st.markdown(f'<div class="kicker">Customer {idx + 1:02d}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="title">{record["Contract"]} · {record["tenure"]} months</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<span class="badge" style="border-color:{color};color:{color}">{band} risk</span> '
                f'<span class="badge {"gray" if not item["prediction"] else ""}">{decision}</span> '
                f'<span class="badge gray">{item["model"]} · {item["threshold"]:.3f}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="body" style="margin-top:.55rem">{record["InternetService"]} internet · '
                f'${record["MonthlyCharges"]:.0f}/mo · ${record["TotalCharges"]:,.0f} lifetime · '
                f'{record["PaymentMethod"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="label">Predicted churn probability</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prob" style="color:{color}">{probability:.1%}</div>', unsafe_allow_html=True)
            delta = probability - item["threshold"]
            delta_text = (
                f"{delta:.1%} above the threshold" if delta >= 0
                else f"{abs(delta):.1%} below the threshold"
            )
            st.markdown(
                f'<div class="prob-cap">{delta_text} · decision threshold {item["threshold"]:.3f} · '
                f'scored {item["scored_at"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="meter"><div class="meter-fill" style="width:{probability * 100:.2f}%;'
                f'background:{color}"></div>'
                f'<div class="meter-mark" style="left:calc({item["threshold"] * 100:.2f}% - 1px)"></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="ts" style="margin-top:.35rem">Bar = probability · white line = threshold</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tenure", f"{record['tenure']} mo")
            m2.metric("Monthly", f"${record['MonthlyCharges']:.0f}")
            m3.metric("Lifetime", f"${record['TotalCharges']:,.0f}")
            m4.metric(
                "Add-ons",
                sum(record[c] == "Yes" for c in ADDON_FIELDS.values()),
            )

            st.markdown(
                '<div class="label">'
                + ("Minor signals" if band == "Low" else "Why it scored here")
                + "</div>",
                unsafe_allow_html=True,
            )
            drivers = risk_drivers(record)
            if band == "Low":
                drivers = drivers[:2]
            if drivers:
                for driver in drivers:
                    st.markdown(f'<div class="item">{driver}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="item">No major risk flags in this profile.</div>', unsafe_allow_html=True)

            st.markdown('<div class="label">Suggested retention actions</div>', unsafe_allow_html=True)
            for action in recommended_actions(record):
                st.markdown(f'<div class="item act">{action}</div>', unsafe_allow_html=True)

            with st.expander("Full customer record"):
                st.dataframe(
                    pd.DataFrame([{"field": k, "value": str(v)} for k, v in record.items()]),
                    hide_index=True, width="stretch",
                )
                st.download_button(
                    "Download this profile (JSON)",
                    json.dumps(record, indent=2), f"customer_{idx + 1:02d}.json", "application/json",
                    width="stretch",
                )

    st.caption(
        "The score is a decision-support signal, not a certainty. Use it to rank customers "
        "and combine it with account context and retention capacity."
    )

# --------------------------------------------------------------------------- tab 2
with tabs[1]:
    st.markdown('<div class="section-label">Held-out test set</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="body">{metrics["best_model"]} at the recall-targeted threshold '
        f'{metrics["operating_threshold"]:.2f}. The test set was not used for tuning or '
        f'threshold selection.</div>',
        unsafe_allow_html=True,
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
        cols = st.columns(3)
        for column, (label, value) in zip(cols, values[row_start : row_start + 3]):
            column.metric(label, value)

    st.markdown('<div class="label">Model comparison at a 0.50 threshold</div>', unsafe_allow_html=True)
    comparison = get_comparison()
    visible = comparison[
        ["Model", "CV ROC AUC", "Test Accuracy", "Test Recall", "Test F1", "Test ROC AUC"]
    ]
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
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download comparison CSV", comparison.to_csv(index=False),
        "model_comparison.csv", "text/csv", width="stretch",
    )
    d2.download_button(
        "Download metrics JSON",
        json.dumps(metrics, indent=2),
        file_name="metrics.json",
        mime="application/json",
        width="stretch",
    )

    st.markdown('<div class="label">Operating points</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="body">Default 0.50 → <b>{default_point["accuracy"]:.1%}</b> accuracy, '
        f'<b>{default_point["recall"]:.1%}</b> recall. Recall-targeted '
        f'{metrics["operating_threshold"]:.2f} → <b>{tuned["accuracy"]:.1%}</b> accuracy, '
        f'<b>{tuned["recall"]:.1%}</b> recall. Churn is imbalanced (26.5% positive), so accuracy '
        f'alone is misleading; the threshold is set to catch at least 80% of churners.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="label">Curves and confusion matrices</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    g1.image(str(FIGURES_DIR / "10_roc_curves.png"), width="stretch")
    g2.image(str(FIGURES_DIR / "11_precision_recall_curves.png"), width="stretch")
    g3, g4 = st.columns(2)
    g3.image(str(FIGURES_DIR / "09_confusion_matrices.png"), width="stretch")
    g4.image(str(FIGURES_DIR / "12_threshold_tuning.png"), width="stretch")

    st.markdown('<div class="label">What drives churn</div>', unsafe_allow_html=True)
    g5, g6 = st.columns(2)
    g5.image(str(FIGURES_DIR / "13_feature_importance_rf.png"), width="stretch")
    g6.image(str(FIGURES_DIR / "14_logistic_coefficients.png"), width="stretch")

# --------------------------------------------------------------------------- tab 3
with tabs[2]:
    st.markdown('<div class="section-label">Dataset overview</div>', unsafe_allow_html=True)
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Customers", f"{dataset['n_rows']:,}")
    o2.metric("Churn rate", f"{dataset['churn_rate']:.1%}")
    o3.metric("Churned", f"{dataset['churned_customers']:,}")
    o4.metric("Model features", f"{dataset['n_features']}")

    st.markdown(
        '<div class="body">Demographics, subscribed services, contract terms and billing. '
        'Churn is concentrated among new, month-to-month, fiber-optic and electronic-check '
        'customers.</div>',
        unsafe_allow_html=True,
    )

    gallery = [
        "01_churn_distribution.png",
        "02_churn_by_contract.png",
        "03_tenure_analysis.png",
        "04_monthly_charges_by_churn.png",
        "05_churn_by_internet_and_payment.png",
        "06_services_vs_churn.png",
    ]
    for i in range(0, len(gallery), 2):
        cols = st.columns(2)
        for column, figure_name in zip(cols, gallery[i : i + 2]):
            column.image(str(FIGURES_DIR / figure_name), width="stretch")
    mid_left, mid_center, mid_right = st.columns([1, 2, 1])
    mid_center.image(str(FIGURES_DIR / "07_correlation_heatmap.png"), width="stretch")

    st.markdown('<div class="label">Cleaned data preview</div>', unsafe_allow_html=True)
    st.dataframe(get_data_preview(), width="stretch", hide_index=True)
    st.download_button(
        "Download cleaned dataset",
        get_cleaned_data().to_csv(index=False),
        "telco_churn_clean.csv",
        "text/csv",
        width="stretch",
    )
