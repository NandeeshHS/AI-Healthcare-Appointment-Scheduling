import os
import sys

import joblib
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    MODELS_DIR, render_refresh_button,
    RISK_COLORS,
    THRESHOLD_VERY_LOW, THRESHOLD_LOW, THRESHOLD_MODERATE, THRESHOLD_HIGH,
)

st.set_page_config(page_title="System Overview", page_icon="🏥", layout="wide")
render_refresh_button()

st.title("🏥 System Overview")
st.markdown(
    "End-to-end architecture, clinical workflow, and model specifications for the "
    "Healthcare Appointment No-Show Risk Intelligence Platform."
)

st.divider()

# ── Section A: Clinical Workflow Pipeline ─────────────────────────────────────
st.subheader("Clinical Workflow")
st.caption("Five-stage pipeline from patient intake to outcome tracking.")

s1, s2, s3, s4, s5 = st.columns(5)
with s1:
    st.info(
        "**Step 1**\n\n"
        "📋 **Book Appointment**\n\n"
        "Enter patient demographics, appointment details, and physician — all validated against the trained encoder classes."
    )
with s2:
    st.info(
        "**Step 2**\n\n"
        "🔮 **Predict Risk**\n\n"
        "XGBoost + Platt Calibration runs in real-time. Outputs a calibrated no-show probability (0–100%)."
    )
with s3:
    st.warning(
        "**Step 3**\n\n"
        "📊 **SHAP Explanation**\n\n"
        "Waterfall chart shows which features push risk up (red) or down (green) for this specific patient."
    )
with s4:
    st.warning(
        "**Step 4**\n\n"
        "🎯 **Tiered Action**\n\n"
        "Five-tier system: Very Low → no action · Moderate → automated reminder · High/Very High → manual outreach + double-book."
    )
with s5:
    st.success(
        "**Step 5**\n\n"
        "📁 **Track & Update**\n\n"
        "SQLite DB stores all appointments. Click **Refresh Data** to simulate time: past appointments resolve to Completed or No-Show."
    )

st.divider()

# ── Section B: Model Facts ─────────────────────────────────────────────────────
st.subheader("Model Specifications")

try:
    metrics = joblib.load(os.path.join(MODELS_DIR, "model_metrics.pkl"))
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    y_test = np.asarray(metrics["y_test"])
    y_prob = np.asarray(metrics["y_prob"])
    base_rate   = float(y_test.mean())
    brier_naive = float(base_rate * (1 - base_rate))
    roc_auc   = metrics.get("roc_auc",   roc_auc_score(y_test, y_prob))
    pr_auc    = metrics.get("pr_auc",    average_precision_score(y_test, y_prob))
    brier_cal = metrics.get("brier_calibrated", brier_score_loss(y_test, y_prob))
    bss_cal   = metrics.get("brier_skill_calibrated", 1.0 - brier_cal / brier_naive)
    fi        = metrics.get("feature_importance")
    n_features = len(fi) if fi is not None else "—"

    col_m, col_f = st.columns([1, 2])

    with col_m:
        st.markdown("#### Key Metrics")
        m1, m2 = st.columns(2)
        m1.metric("ROC-AUC", f"{roc_auc:.4f}", help="1.0 = perfect · 0.5 = random baseline")
        m2.metric("PR-AUC",  f"{pr_auc:.4f}",  help=f"Random baseline = {base_rate:.3f}")
        m3, m4 = st.columns(2)
        m3.metric("Brier Skill Score", f"{bss_cal:+.4f}",
                  help="BSS > 0 means better than always predicting the base rate. Negative pre-calibration → Platt scaling fixed this.")
        m4.metric("No-Show Base Rate", f"{base_rate:.2%}")

        st.markdown("#### Architecture")
        st.markdown(
            "- **Model:** XGBoost (gradient-boosted trees)\n"
            "- **Calibration:** Platt scaling (CalibratedClassifierCV)\n"
            "- **Tuning:** Optuna TPE · 50 trials\n"
            "- **Split:** 64% train / 16% calibration / 20% test\n"
            f"- **Features:** {n_features} engineered features\n"
            "- **Imbalance:** `scale_pos_weight` ≈ 3.74\n"
            "- **SHAP:** XGBoost native `pred_contribs` (tree SHAP)"
        )

    with col_f:
        if fi is not None:
            st.markdown("#### Top 10 Feature Importances")
            fig_imp = px.bar(
                fi.head(10), x="Importance", y="Feature",
                orientation="h",
                color="Importance", color_continuous_scale="Viridis",
                labels={"Importance": "Gain"},
            )
            fig_imp.update_layout(
                height=300,
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
                margin=dict(l=10, r=20, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            best_feature = fi.iloc[0]["Feature"]
            st.caption(
                f"Top driver: **{best_feature}**. Patient and physician historical no-show rates "
                "were added as engineered features and appear in the top 10."
            )

except Exception as e:
    st.warning(f"Model metrics unavailable — run `python extract_model.py` first. ({e})")

st.divider()

# ── Section C: Risk Tier Action Guide ─────────────────────────────────────────
st.subheader("Risk Tier Action Guide")
st.caption(
    "Thresholds are data-driven from calibrated model test-set decile analysis. "
    "Predicted probabilities now track actual no-show rates (Platt scaling)."
)

tier_data = [
    ("Very Low",  f"< {THRESHOLD_VERY_LOW:.0%}",                         "~7% actual NS",  "No action needed",                     "✅",  "success"),
    ("Low",       f"{THRESHOLD_VERY_LOW:.0%} – {THRESHOLD_LOW:.0%}",     "~13% actual NS", "No action needed",                     "✅",  "success"),
    ("Moderate",  f"{THRESHOLD_LOW:.0%} – {THRESHOLD_MODERATE:.0%}",     "~19% actual NS", "Automated SMS / email reminder",        "📩",  "warning"),
    ("High",      f"{THRESHOLD_MODERATE:.0%} – {THRESHOLD_HIGH:.0%}",    "~30% actual NS", "Manual phone call + reminder",          "📞",  "error"),
    ("Very High", f">= {THRESHOLD_HIGH:.0%}",                             "~51% actual NS", "Double-book + personal outreach",       "🚨",  "error"),
]

tier_cols = st.columns(5)
for col, (level, prob_range, actual, action, icon, kind) in zip(tier_cols, tier_data):
    color = RISK_COLORS[level]
    col.markdown(
        f"<div style='border:2px solid {color}; border-radius:10px; padding:14px 10px; "
        f"background:{color}18; text-align:center;'>"
        f"<div style='font-size:1.6em;'>{icon}</div>"
        f"<div style='font-size:1.05em; font-weight:700; color:{color};'>{level}</div>"
        f"<div style='font-size:0.85em; margin:4px 0;'><b>{prob_range}</b></div>"
        f"<div style='font-size:0.78em; color:gray;'>{actual}</div>"
        f"<hr style='margin:8px 0; border-color:{color}44;'/>"
        f"<div style='font-size:0.8em;'>{action}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Section D: Tech Stack ─────────────────────────────────────────────────────
st.subheader("Tech Stack")

t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown(
        "**ML / Data**\n"
        "- Python 3.11\n"
        "- XGBoost\n"
        "- Scikit-learn\n"
        "- Optuna\n"
        "- SHAP (pred_contribs)"
    )
with t2:
    st.markdown(
        "**App & UI**\n"
        "- Streamlit\n"
        "- Plotly (Express + GO)\n"
        "- Pandas / NumPy\n"
        "- Joblib"
    )
with t3:
    st.markdown(
        "**Storage**\n"
        "- SQLite (healthcare.db)\n"
        "- Pickle artifacts\n"
        "  - xgboost_model.pkl\n"
        "  - xgboost_base_model.pkl\n"
        "  - label_encoders.pkl\n"
        "  - feature_columns.pkl\n"
        "  - model_metrics.pkl"
    )
with t4:
    st.markdown(
        "**Key Design Decisions**\n"
        "- Platt scaling corrects\n"
        "  `scale_pos_weight` overconfidence\n"
        "- Two model files: calibrated\n"
        "  for prediction, raw for SHAP\n"
        "- Leakage-free historical rates\n"
        "  via `shift(1).cumsum()`\n"
        "- ISO isoweekday (1–7) for\n"
        "  consistent DOW encoding"
    )
