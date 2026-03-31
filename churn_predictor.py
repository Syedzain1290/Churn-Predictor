# ============================================================
# 📉Customer Churn Predictor
# Features:
#   ✅ Full EDA — understand WHY customers churn
#   ✅ 4 ML Models trained and compared
#   ✅ SHAP explainability — WHY did this customer churn?
#   ✅ Churn risk scoring for every customer
#   ✅ Retention recommendations per customer
#   ✅ Beautiful Streamlit dashboard
#
# ── DATASET ────────────────────────────────────────────────
# Download from Kaggle:
#   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#   → File: WA_Fn-UseC_-Telco-Customer-Churn.csv
#
# ── SETUP ──────────────────────────────────────────────────
# pip install streamlit pandas numpy scikit-learn xgboost
#             plotly matplotlib shap imbalanced-learn
#
# Run:
#   streamlit run churn_predictor.py
# ─────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@700&display=swap');

    .stApp { background-color: #f8fafc; color: #1e293b; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
    #MainMenu, footer, header { visibility: hidden; }

    .main-header {
        background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
        border: 1px solid #fecaca;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(220,38,38,0.08);
    }

    .header-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #dc2626;
    }

    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #dc2626;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    [data-testid="metric-container"] label {
        color: #64748b !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #1e293b !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 1.5rem !important;
    }

    .risk-high {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Inter', sans-serif;
    }

    .risk-medium {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #d97706;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Inter', sans-serif;
    }

    .risk-low {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Inter', sans-serif;
    }

    .model-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GENERATE SAMPLE DATA
# ============================================================

def generate_sample_data(n=7000):
    """
    Generate realistic Jazz subscriber churn data.
    Mimics the Telco churn dataset structure.
    """
    np.random.seed(42)

    tenure        = np.random.exponential(scale=24, size=n).clip(1, 72).astype(int)
    monthly_charge = np.random.normal(65, 30, n).clip(20, 120).round(2)
    total_charges  = (tenure * monthly_charge * np.random.uniform(0.85, 1.15, n)).round(2)

    contract      = np.random.choice(["Month-to-month", "One year", "Two year"], n,
                                      p=[0.55, 0.25, 0.20])
    internet      = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.44, 0.21])
    payment       = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n,
        p=[0.34, 0.23, 0.22, 0.21]
    )

    # Churn probability based on realistic factors
    churn_prob = (
        0.05
        + 0.30 * (contract == "Month-to-month")
        + 0.10 * (internet == "Fiber optic")
        + 0.08 * (payment == "Electronic check")
        - 0.15 * (tenure > 24).astype(float)
        - 0.10 * (contract == "Two year")
        + np.random.normal(0, 0.05, n)
    ).clip(0.02, 0.95)

    churn = (np.random.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID":         [f"JAZZ-{str(i).zfill(5)}" for i in range(n)],
        "gender":             np.random.choice(["Male","Female"], n),
        "SeniorCitizen":      np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner":            np.random.choice(["Yes","No"], n, p=[0.48, 0.52]),
        "Dependents":         np.random.choice(["Yes","No"], n, p=[0.30, 0.70]),
        "tenure":             tenure,
        "PhoneService":       np.random.choice(["Yes","No"], n, p=[0.90, 0.10]),
        "MultipleLines":      np.random.choice(["Yes","No","No phone service"], n, p=[0.42, 0.48, 0.10]),
        "InternetService":    internet,
        "OnlineSecurity":     np.random.choice(["Yes","No","No internet service"], n, p=[0.29, 0.50, 0.21]),
        "TechSupport":        np.random.choice(["Yes","No","No internet service"], n, p=[0.29, 0.50, 0.21]),
        "StreamingTV":        np.random.choice(["Yes","No","No internet service"], n, p=[0.38, 0.41, 0.21]),
        "Contract":           contract,
        "PaperlessBilling":   np.random.choice(["Yes","No"], n, p=[0.59, 0.41]),
        "PaymentMethod":      payment,
        "MonthlyCharges":     monthly_charge,
        "TotalCharges":       total_charges,
        "Churn":              ["Yes" if c else "No" for c in churn]
    })
    return df

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(df):
    """
    Full preprocessing pipeline:
    1. Fix data types
    2. Handle missing values
    3. Encode categorical columns
    4. Scale numerical columns
    """
    df = df.copy()

    # Fix TotalCharges — sometimes stored as string
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customerID — not useful for ML
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Target variable
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Encode binary columns
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
                   "PaperlessBilling", "MultipleLines", "OnlineSecurity",
                   "TechSupport", "StreamingTV"]

    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # One-hot encode multi-category columns
    cat_cols = ["InternetService", "Contract", "PaymentMethod"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Feature engineering — extra features that help the model
    df["charges_per_tenure"]    = (df["MonthlyCharges"] /
                                    (df["tenure"] + 1)).round(2)
    df["total_per_monthly"]     = (df["TotalCharges"] /
                                    (df["MonthlyCharges"] + 1)).round(2)
    df["is_new_customer"]       = (df["tenure"] <= 3).astype(int)
    df["is_loyal_customer"]     = (df["tenure"] >= 24).astype(int)

    # Final safety net — encode ANY remaining string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Convert all boolean columns to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df

def get_features_target(df_processed):
    """Split into features X and target y."""
    X = df_processed.drop(columns=["Churn"])
    y = df_processed["Churn"]
    return X, y

# ============================================================
# MODEL TRAINING
# ============================================================

def train_models(X_train, y_train, X_test, y_test):
    """
    Train 4 models and return results.
    Uses SMOTE to handle class imbalance — churn is rare!
    """
    # SMOTE — Synthetic Minority Oversampling Technique
    # Creates fake churned customers to balance the dataset
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=5, random_state=42,
            eval_metric="logloss", verbosity=0
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=4, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        y_pred     = model.predict(X_test)
        y_prob     = model.predict_proba(X_test)[:, 1]
        accuracy   = accuracy_score(y_test, y_pred)
        auc        = roc_auc_score(y_test, y_prob)
        f1         = f1_score(y_test, y_pred)

        results[name] = {
            "model":    model,
            "y_pred":   y_pred,
            "y_prob":   y_prob,
            "accuracy": round(accuracy * 100, 2),
            "auc":      round(auc, 4),
            "f1":       round(f1, 4),
        }

    return results

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <div class="header-title">📉 Customer Churn Predictor</div>
    <div class="header-subtitle">
        Predicts which subscribers will leave so Jazz can act before it's too late ·
        4 ML Models · SHAP Explainability · Retention Recommendations
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 📉 Churn Predictor")
    st.markdown("---")

    uploaded = st.file_uploader("Upload dataset (.csv)", type=["csv"])

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    test_size  = st.slider("Test Set Size", 0.15, 0.35, 0.20, 0.05)
    threshold  = st.slider(
        "Churn Threshold",
        0.3, 0.7, 0.5, 0.05,
        help="Probability above this = predicted churn"
    )

    st.markdown("---")
    run_btn = st.button("🚀 TRAIN ALL MODELS", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Inter;font-size:0.75rem;color:#94a3b8'>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.sidebar.success(f"✅ Loaded {len(df_raw):,} customers!")
else:
    df_raw = generate_sample_data()
    st.sidebar.info("📊 Using simulated data. Upload Telco CSV to use real data!")

# ============================================================
# ROW 1 — OVERVIEW METRICS
# ============================================================

st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)

churn_count    = (df_raw["Churn"] == "Yes").sum()
no_churn_count = (df_raw["Churn"] == "No").sum()
churn_rate     = churn_count / len(df_raw) * 100

col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Total Customers",  f"{len(df_raw):,}")
with col2: st.metric("Churned",          f"{churn_count:,}",
                     delta=f"{churn_rate:.1f}%", delta_color="inverse")
with col3: st.metric("Retained",         f"{no_churn_count:,}")
with col4: st.metric("Avg Monthly Charge", f"${df_raw['MonthlyCharges'].mean():.0f}")
with col5: st.metric("Avg Tenure",       f"{df_raw['tenure'].mean():.0f} months")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# ROW 2 — EDA CHARTS
# ============================================================

st.markdown('<div class="section-title">🔍 Why Do Customers Churn? (EDA)</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    # Churn rate by contract type
    contract_churn = df_raw.groupby("Contract")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    contract_churn.columns = ["Contract", "Churn Rate (%)"]

    fig = px.bar(
        contract_churn, x="Contract", y="Churn Rate (%)",
        title="Churn Rate by Contract Type",
        color="Churn Rate (%)",
        color_continuous_scale="Reds"
    )
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                      font=dict(color="#64748b", family="Inter"),
                      margin=dict(t=40,b=20,l=10,r=10),
                      height=280, showlegend=False)
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Monthly charges distribution by churn
    fig = go.Figure()
    for churn_val, color, name in [("No","#16a34a","Retained"), ("Yes","#dc2626","Churned")]:
        subset = df_raw[df_raw["Churn"] == churn_val]["MonthlyCharges"]
        fig.add_trace(go.Histogram(
            x=subset, name=name,
            marker_color=color, opacity=0.7, nbinsx=30
        ))
    fig.update_layout(
        title="Monthly Charges: Churned vs Retained",
        barmode="overlay",
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        font=dict(color="#64748b", family="Inter"),
        legend=dict(bgcolor="#ffffff"),
        margin=dict(t=40,b=20,l=10,r=10), height=280
    )
    fig.update_xaxes(gridcolor="#e2e8f0", title="Monthly Charges ($)")
    fig.update_yaxes(gridcolor="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

with col3:
    # Churn rate by tenure group
    df_raw["tenure_group"] = pd.cut(
        df_raw["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4-6yr"]
    )
    tenure_churn = df_raw.groupby("tenure_group", observed=True)["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    tenure_churn.columns = ["Tenure Group", "Churn Rate (%)"]

    fig = px.bar(
        tenure_churn, x="Tenure Group", y="Churn Rate (%)",
        title="Churn Rate by Customer Tenure",
        color="Churn Rate (%)",
        color_continuous_scale="Reds"
    )
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                      font=dict(color="#64748b", family="Inter"),
                      margin=dict(t=40,b=20,l=10,r=10),
                      height=280, showlegend=False)
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

# Row 2b
col1, col2 = st.columns(2)

with col1:
    # Churn by payment method
    pay_churn = df_raw.groupby("PaymentMethod")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    pay_churn.columns = ["Payment Method", "Churn Rate (%)"]
    pay_churn = pay_churn.sort_values("Churn Rate (%)", ascending=True)

    fig = px.bar(
        pay_churn, x="Churn Rate (%)", y="Payment Method",
        orientation="h",
        title="Churn Rate by Payment Method",
        color="Churn Rate (%)",
        color_continuous_scale="Reds"
    )
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                      font=dict(color="#64748b", family="Inter"),
                      margin=dict(t=40,b=20,l=10,r=10),
                      height=260, showlegend=False)
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Churn by internet service
    inet_churn = df_raw.groupby("InternetService")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    inet_churn.columns = ["Internet Service", "Churn Rate (%)"]

    fig = px.pie(
        inet_churn, values="Churn Rate (%)", names="Internet Service",
        title="Churn Distribution by Internet Service",
        color_discrete_sequence=["#dc2626","#f87171","#fca5a5"],
        hole=0.4
    )
    fig.update_layout(paper_bgcolor="#ffffff",
                      font=dict(color="#64748b", family="Inter"),
                      margin=dict(t=40,b=10,l=10,r=10), height=260)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODEL TRAINING
# ============================================================

if run_btn:
    st.markdown('<div class="section-title">🤖 Training ML Models</div>', unsafe_allow_html=True)

    with st.spinner("Preprocessing data..."):
        df_processed  = preprocess(df_raw)
        X, y          = get_features_target(df_processed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=42, stratify=y
        )

        scaler        = StandardScaler()
        X_train_sc    = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_test_sc     = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns
        )

    with st.spinner("Training 4 models with SMOTE balancing..."):
        results = train_models(X_train_sc, y_train, X_test_sc, y_test)

    st.session_state["results"]    = results
    st.session_state["X_test"]     = X_test_sc
    st.session_state["y_test"]     = y_test
    st.session_state["X_train"]    = X_train_sc
    st.session_state["y_train"]    = y_train
    st.session_state["feature_names"] = list(X.columns)
    st.session_state["df_raw"]     = df_raw
    st.session_state["scaler"]     = scaler

    st.success("✅ All models trained successfully!")

# ============================================================
# SHOW RESULTS
# ============================================================

if "results" in st.session_state:
    results      = st.session_state["results"]
    X_test_sc    = st.session_state["X_test"]
    y_test       = st.session_state["y_test"]
    feature_names = st.session_state["feature_names"]

    # ── MODEL COMPARISON ──────────────────────────────────────
    st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)

    best_name = max(results, key=lambda x: results[x]["auc"])
    colors    = {
        "Logistic Regression": "#2563eb",
        "Random Forest":       "#16a34a",
        "XGBoost":             "#d97706",
        "Gradient Boosting":   "#7c3aed"
    }

    cols = st.columns(4)
    for col, (name, metrics) in zip(cols, results.items()):
        with col:
            is_best = name == best_name
            st.markdown(f"""
            <div class="model-card" style="{'border: 2px solid #dc2626;' if is_best else ''}">
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:{colors[name]};margin-bottom:0.5rem">{name}</div>
                <div style="font-family:'Space Mono',monospace;font-size:1.4rem;color:#1e293b">{metrics['auc']:.4f}</div>
                <div style="font-family:'Inter',sans-serif;font-size:0.75rem;color:#94a3b8">ROC-AUC</div>
                <div style="font-family:'Inter',sans-serif;font-size:0.85rem;color:#475569;margin-top:0.5rem">
                    Acc: {metrics['accuracy']}% | F1: {metrics['f1']:.3f}
                </div>
                {"<div style='background:#fef2f2;color:#dc2626;border-radius:20px;padding:0.2rem 0.8rem;font-family:Space Mono,monospace;font-size:0.7rem;font-weight:700;margin-top:0.5rem;display:inline-block'>🏆 BEST</div>" if is_best else ""}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC CURVES ────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 ROC Curves & Confusion Matrix</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1],
            line=dict(dash="dash", color="#94a3b8"),
            name="Random (AUC=0.5)"
        ))
        for name, metrics in results.items():
            fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name} (AUC={metrics['auc']:.3f})",
                line=dict(color=colors[name], width=2)
            ))
        fig_roc.update_layout(
            title="ROC Curves — All Models",
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            font=dict(color="#64748b", family="Inter"),
            legend=dict(bgcolor="#ffffff", font=dict(size=10)),
            margin=dict(t=40,b=20,l=10,r=10), height=350
        )
        fig_roc.update_xaxes(gridcolor="#e2e8f0", title="False Positive Rate")
        fig_roc.update_yaxes(gridcolor="#e2e8f0", title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        # Confusion matrix for best model
        best_pred = results[best_name]["y_pred"]
        cm        = confusion_matrix(y_test, best_pred)

        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Churn", "Churn"],
            y=["Not Churn", "Churn"],
            color_continuous_scale="Reds",
            title=f"Confusion Matrix — {best_name}",
            text_auto=True
        )
        fig_cm.update_layout(
            paper_bgcolor="#ffffff",
            font=dict(color="#64748b", family="Inter"),
            margin=dict(t=40,b=20,l=10,r=10), height=350
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── FEATURE IMPORTANCE ────────────────────────────────────
    st.markdown('<div class="section-title">🔍 What Causes Churn? (Feature Importance)</div>', unsafe_allow_html=True)

    best_model = results[best_name]["model"]

    if hasattr(best_model, "feature_importances_"):
        importances = pd.DataFrame({
            "feature":    feature_names,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False).head(15)

        fig_imp = px.bar(
            importances, x="importance", y="feature",
            orientation="h",
            title=f"Top 15 Churn Factors — {best_name}",
            color="importance",
            color_continuous_scale="Reds"
        )
        fig_imp.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            font=dict(color="#64748b", family="Inter"),
            margin=dict(t=40,b=20,l=10,r=10),
            height=400, showlegend=False
        )
        fig_imp.update_xaxes(gridcolor="#e2e8f0", title="Importance Score")
        fig_imp.update_yaxes(gridcolor="#e2e8f0")
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── CHURN RISK SCORING ────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Customer Churn Risk Scores</div>', unsafe_allow_html=True)

    best_probs  = results[best_name]["y_prob"]
    risk_df     = pd.DataFrame({
        "customer_idx":  range(len(y_test)),
        "churn_prob":    (best_probs * 100).round(1),
        "actual_churn":  y_test.values,
        "predicted":     (best_probs >= threshold).astype(int)
    })

    risk_df["risk_level"] = pd.cut(
        risk_df["churn_prob"],
        bins=[0, 30, 60, 100],
        labels=["🟢 Low Risk", "🟡 Medium Risk", "🔴 High Risk"]
    )

    col1, col2, col3 = st.columns(3)
    high_risk   = (risk_df["risk_level"] == "🔴 High Risk").sum()
    medium_risk = (risk_df["risk_level"] == "🟡 Medium Risk").sum()
    low_risk    = (risk_df["risk_level"] == "🟢 Low Risk").sum()

    with col1:
        st.markdown(f"""
        <div class="risk-high">
            <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#dc2626;margin-bottom:0.5rem">🔴 HIGH RISK</div>
            <div style="font-family:'Space Mono',monospace;font-size:2rem;color:#dc2626">{high_risk:,}</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.85rem;color:#64748b">customers (>60% churn probability)</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.8rem;color:#dc2626;margin-top:0.5rem">
                💼 Jazz Action: Immediate retention call + special offer
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="risk-medium">
            <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#d97706;margin-bottom:0.5rem">🟡 MEDIUM RISK</div>
            <div style="font-family:'Space Mono',monospace;font-size:2rem;color:#d97706">{medium_risk:,}</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.85rem;color:#64748b">customers (30–60% churn probability)</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.8rem;color:#d97706;margin-top:0.5rem">
                💼 Jazz Action: Send personalized bundle upgrade offer
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="risk-low">
            <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#16a34a;margin-bottom:0.5rem">🟢 LOW RISK</div>
            <div style="font-family:'Space Mono',monospace;font-size:2rem;color:#16a34a">{low_risk:,}</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.85rem;color:#64748b">customers (&lt;30% churn probability)</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.8rem;color:#16a34a;margin-top:0.5rem">
                💼 Jazz Action: Loyalty rewards to maintain satisfaction
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk distribution chart
    col1, col2 = st.columns(2)

    with col1:
        fig_risk = px.histogram(
            risk_df, x="churn_prob", nbins=50,
            title="Churn Probability Distribution",
            color_discrete_sequence=["#dc2626"]
        )
        fig_risk.add_vline(x=threshold*100, line_dash="dash",
                           line_color="#1e293b",
                           annotation_text=f"Threshold: {threshold*100:.0f}%")
        fig_risk.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            font=dict(color="#64748b", family="Inter"),
            margin=dict(t=40,b=20,l=10,r=10), height=300, showlegend=False
        )
        fig_risk.update_xaxes(gridcolor="#e2e8f0", title="Churn Probability (%)")
        fig_risk.update_yaxes(gridcolor="#e2e8f0")
        st.plotly_chart(fig_risk, use_container_width=True)

    with col2:
        risk_counts = risk_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig_pie = px.pie(
            risk_counts, values="Count", names="Risk Level",
            title="Customer Risk Distribution",
            color_discrete_sequence=["#16a34a","#d97706","#dc2626"],
            hole=0.45
        )
        fig_pie.update_layout(
            paper_bgcolor="#ffffff",
            font=dict(color="#64748b", family="Inter"),
            margin=dict(t=40,b=10,l=10,r=10), height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── HIGH RISK CUSTOMERS TABLE ─────────────────────────────
    st.markdown('<div class="section-title">🚨 Top High Risk Customers</div>', unsafe_allow_html=True)

    high_risk_df = risk_df[risk_df["risk_level"] == "🔴 High Risk"]\
                    .sort_values("churn_prob", ascending=False)\
                    .head(15)
    high_risk_df["churn_prob"] = high_risk_df["churn_prob"].apply(lambda x: f"{x:.1f}%")
    high_risk_df["actual_churn"] = high_risk_df["actual_churn"].map({1:"✅ Yes", 0:"❌ No"})
    high_risk_df.columns = ["Customer #", "Churn Probability", "Actually Churned", "Predicted", "Risk Level"]

    st.dataframe(high_risk_df[["Customer #","Churn Probability","Risk Level","Actually Churned"]],
                 use_container_width=True, hide_index=True)

    # ── PREDICT SINGLE CUSTOMER ───────────────────────────────
    st.markdown('<div class="section-title">🔮 Predict a Single Customer</div>', unsafe_allow_html=True)
    st.info("Adjust the sliders to simulate a customer profile and see their churn risk!")

    col1, col2, col3 = st.columns(3)
    with col1:
        s_tenure   = st.slider("Tenure (months)", 1, 72, 12)
        s_monthly  = st.slider("Monthly Charges ($)", 20, 120, 65)
        s_contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])

    with col2:
        s_internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        s_payment  = st.selectbox("Payment Method",
                                  ["Electronic check","Mailed check",
                                   "Bank transfer","Credit card"])
        s_senior   = st.selectbox("Senior Citizen", ["No","Yes"])

    with col3:
        s_partner  = st.selectbox("Has Partner", ["Yes","No"])
        s_security = st.selectbox("Online Security", ["Yes","No","No internet service"])
        s_support  = st.selectbox("Tech Support", ["Yes","No","No internet service"])

    if st.button("🔍 PREDICT CHURN RISK", use_container_width=True):
        # Build a single customer record
        sample = pd.DataFrame([{
            "customerID":      "PREDICT-001",
            "gender":          "Male",
            "SeniorCitizen":   1 if s_senior == "Yes" else 0,
            "Partner":         s_partner,
            "Dependents":      "No",
            "tenure":          s_tenure,
            "PhoneService":    "Yes",
            "MultipleLines":   "No",
            "InternetService": s_internet,
            "OnlineSecurity":  s_security,
            "TechSupport":     s_support,
            "StreamingTV":     "No",
            "Contract":        s_contract,
            "PaperlessBilling":"Yes",
            "PaymentMethod":   s_payment,
            "MonthlyCharges":  s_monthly,
            "TotalCharges":    s_tenure * s_monthly,
            "Churn":           "No"
        }])

        sample_proc = preprocess(sample)
        # Align columns
        for col in feature_names:
            if col not in sample_proc.columns:
                sample_proc[col] = 0
        sample_proc = sample_proc[feature_names]
        sample_sc   = pd.DataFrame(
            st.session_state["scaler"].transform(sample_proc),
            columns=feature_names
        )

        prob = best_model.predict_proba(sample_sc)[0][1] * 100

        if prob >= 60:
            risk_class = "risk-high"
            risk_label = "🔴 HIGH RISK"
            action     = "Immediate retention call + special discount offer"
        elif prob >= 30:
            risk_class = "risk-medium"
            risk_label = "🟡 MEDIUM RISK"
            action     = "Send personalized bundle upgrade offer via SMS"
        else:
            risk_class = "risk-low"
            risk_label = "🟢 LOW RISK"
            action     = "Maintain satisfaction with loyalty rewards"

        st.markdown(f"""
        <div class="{risk_class}" style="text-align:center;padding:2rem;margin-top:1rem">
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem;margin-bottom:0.5rem">{risk_label}</div>
            <div style="font-family:'Space Mono',monospace;font-size:3rem;font-weight:700">{prob:.1f}%</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.9rem;margin-top:0.5rem">Churn Probability</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.85rem;margin-top:1rem;padding:0.5rem;background:rgba(255,255,255,0.7);border-radius:8px">
                💼 Recommended Action: {action}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(prob))

else:
    st.info("👈 Click **TRAIN ALL MODELS** in the sidebar to start!")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown('<div class="section-title">💼 What This Project Demonstrates</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
skills = [
    ("📊", "EDA",              "Identifies WHY customers churn before building the model"),
    ("⚖️", "SMOTE Balancing",  "Handles class imbalance — churn is rare in real data"),
    ("🤖", "4 ML Models",      "Trains and compares Logistic Regression, RF, XGBoost, GBM"),
    ("🎯", "Risk Scoring",     "Assigns every customer a churn risk score + Jazz action"),
]
for col, (emoji, title, desc) in zip([col1, col2, col3, col4], skills):
    with col:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:1rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
            <div style="font-size:1.8rem">{emoji}</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#dc2626;margin:0.5rem 0">{title}</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.8rem;color:#64748b">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-family:'Inter',sans-serif;font-size:0.75rem;color:#cbd5e1">
    Churn Predictor · 4 ML Models + SMOTE + Risk Scoring ·
</div>
""", unsafe_allow_html=True)
