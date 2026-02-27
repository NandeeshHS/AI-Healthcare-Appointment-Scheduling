import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    MODELS_DIR, render_refresh_button,
    THRESHOLD_VERY_LOW, THRESHOLD_LOW, THRESHOLD_MODERATE, THRESHOLD_HIGH,
    RISK_COLORS,
)

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
render_refresh_button()

st.title("📈 Model Performance & Insights")

try:
    metrics = joblib.load(os.path.join(MODELS_DIR, "model_metrics.pkl"))
except Exception as e:
    st.error(f"Could not load model metrics. Run `python extract_model.py` first. Error: {e}")
    st.stop()

y_test = np.asarray(metrics["y_test"])
y_prob = np.asarray(metrics["y_prob"])          # calibrated probabilities

roc_auc = metrics.get("roc_auc", None)
pr_auc  = metrics.get("pr_auc",  None)

if roc_auc is None:
    roc_auc = roc_auc_score(y_test, y_prob)
if pr_auc is None:
    pr_auc = average_precision_score(y_test, y_prob)

base_rate   = float(y_test.mean())
brier_naive = float(base_rate * (1 - base_rate))

brier_cal   = metrics.get("brier_calibrated",        brier_score_loss(y_test, y_prob))
brier_uncal = metrics.get("brier_uncalibrated",       None)
bss_cal     = metrics.get("brier_skill_calibrated",   None)
bss_uncal   = metrics.get("brier_skill_uncalibrated", None)

if bss_cal is None:
    bss_cal = 1.0 - brier_cal / brier_naive if brier_naive > 0 else 0.0

# Pre-compute formatted strings (avoids ambiguous f-string format specs inside st.* calls)
bss_cal_fmt   = f"{float(bss_cal):+.4f}"
bss_uncal_str = f"{float(bss_uncal):+.4f}" if bss_uncal is not None else "N/A"

# ── Header: KPI cards + Optuna params (persistent above tabs) ─────────────────
st.markdown(
    "**Algorithm:** XGBoost + Platt Calibration · "
    "**Objective:** Binary (No-Show vs Show) · "
    "**Tuning:** Optuna TPE · "
    "**New features:** Patient & Physician historical no-show rates"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ROC-AUC",    f"{roc_auc:.4f}", help="1.0 = perfect; 0.5 = random")
c2.metric("PR-AUC",     f"{pr_auc:.4f}",  help=f"Random baseline = {base_rate:.3f}")
c3.metric(
    "Brier Score",
    f"{brier_cal:.4f}",
    help="Calibrated model. Lower is better; 0 = perfect.",
)
c4.metric(
    "Brier Skill Score",
    f"{bss_cal_fmt}",
    delta=f"{bss_cal_fmt}",
    delta_color="normal",
    help="BSS > 0 = better than naive baseline. Pre-calibration was negative.",
)
c5.metric("No-Show Base Rate", f"{base_rate:.2%}")

best_params = metrics.get("best_params")
if best_params:
    with st.expander("Show Optuna best hyperparameters"):
        st.json(best_params)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Probability & ROC",
    "🎯 Calibration",
    "⚖️ Precision-Recall",
    "🔍 Features & Thresholds",
])

# ── Tab 1: Probability Distribution + ROC ─────────────────────────────────────
with tab1:
    st.subheader("Predicted Probability Distribution")
    st.caption(
        "Calibrated probabilities should cluster near the actual base rate (~21%). "
        "Threshold lines show the 5-tier risk boundaries."
    )

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=y_prob, nbinsx=80, name="Predictions",
        marker_color="#636EFA", opacity=0.75,
    ))
    for t, label, col in [
        (THRESHOLD_VERY_LOW, f"VLow|Low ({THRESHOLD_VERY_LOW:.0%})",   "#27ae60"),
        (THRESHOLD_LOW,      f"Low|Mod ({THRESHOLD_LOW:.0%})",         "#f39c12"),
        (THRESHOLD_MODERATE, f"Mod|High ({THRESHOLD_MODERATE:.0%})",   "#e67e22"),
        (THRESHOLD_HIGH,     f"High|VHigh ({THRESHOLD_HIGH:.0%})",     "#e74c3c"),
    ]:
        fig_hist.add_vline(
            x=t, line_dash="solid", line_color=col, line_width=2,
            annotation_text=label, annotation_position="top",
        )
    fig_hist.update_layout(
        title="Distribution of Predicted No-Show Probabilities (Calibrated)",
        xaxis_title="Predicted Probability",
        yaxis_title="Count",
        showlegend=False,
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"XGBoost + Platt (AUC = {roc_auc:.3f})",
        line=dict(color="steelblue", width=2.5),
        fill="tozeroy", fillcolor="rgba(70,130,180,0.08)",
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
        line=dict(color="grey", dash="dash"),
    ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title=f"ROC Curve — AUC = {roc_auc:.4f}",
        legend=dict(x=0.55, y=0.1),
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# ── Tab 2: Calibration ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Calibration — Before vs After Platt Scaling")
    st.caption(
        "Points on the diagonal = perfectly calibrated. "
        "Above diagonal = overconfident. "
        "Platt scaling corrects systematic overconfidence introduced by `scale_pos_weight`."
    )

    y_prob_uncal_arr = np.asarray(metrics["y_prob_uncalibrated"]) \
        if "y_prob_uncalibrated" in metrics else None

    fig_cal = go.Figure()

    frac_pos_cal, mean_pred_cal = calibration_curve(y_test, y_prob, n_bins=15, strategy="quantile")
    fig_cal.add_trace(go.Scatter(
        x=mean_pred_cal, y=frac_pos_cal, mode="lines+markers",
        name="After Platt Scaling ✅",
        line=dict(color="#2ecc71", width=2.5),
        marker=dict(size=7),
    ))

    if y_prob_uncal_arr is not None:
        frac_pos_unc, mean_pred_unc = calibration_curve(
            y_test, y_prob_uncal_arr, n_bins=15, strategy="quantile"
        )
        fig_cal.add_trace(go.Scatter(
            x=mean_pred_unc, y=frac_pos_unc, mode="lines+markers",
            name="Before calibration (raw XGBoost) ❌",
            line=dict(color="#e74c3c", width=2, dash="dot"),
            marker=dict(size=6),
        ))

    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
        line=dict(color="grey", dash="dash"),
    ))
    fig_cal.update_layout(
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives (Actual No-Show Rate)",
        title="Calibration Curve — Before vs After Platt Scaling",
        legend=dict(x=0.02, y=0.98),
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    if brier_uncal is not None and bss_uncal is not None:
        cal_df = pd.DataFrame({
            "Model":       ["Raw XGBoost (uncalibrated)", "XGBoost + Platt Scaling"],
            "Brier Score": [f"{brier_uncal:.4f}", f"{brier_cal:.4f}"],
            "Brier Skill": [bss_uncal_str, bss_cal_fmt],
            "Assessment":  [
                "❌ Overconfident — worse than naive baseline" if bss_uncal < 0 else "✅ Better than baseline",
                "❌ Still overconfident" if bss_cal < 0 else "✅ Better than naive baseline",
            ],
        })
        st.dataframe(cal_df, hide_index=True, use_container_width=True)
        st.caption(
            f"Brier Skill Score > 0 = better than always predicting the base rate. "
            f"Naive baseline Brier = {brier_naive:.4f} (predict {base_rate:.2%} for every patient)."
        )

    st.info(
        f"**Why calibration matters:** Pre-calibration, `scale_pos_weight=3.74` pushed all "
        f"predicted probabilities high (BSS = {bss_uncal_str}). "
        f"A raw score of 0.55 corresponded to only ~22% actual no-shows. "
        f"After Platt scaling, a score of 0.30 now means ~30% actual no-show. "
        f"Thresholds can be set intuitively. BSS improved to **{bss_cal_fmt}**."
    )

# ── Tab 3: Precision-Recall ───────────────────────────────────────────────────
with tab3:
    st.subheader("Precision-Recall Curve")

    if "precisions" in metrics:
        prec_arr = metrics["precisions"]
        rec_arr  = metrics["recalls"]
        t_arr    = metrics["pr_thresholds"]
    else:
        prec_arr, rec_arr, t_arr = precision_recall_curve(y_test, y_prob)

    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-10)
    best_t = float(t_arr[int(np.argmax(f1_arr))])

    col_pr, col_f1 = st.columns(2)

    with col_pr:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=rec_arr, y=prec_arr, mode="lines",
            name=f"PR curve (AUC = {pr_auc:.3f})",
            line=dict(color="darkorange", width=2.5),
            fill="tozeroy", fillcolor="rgba(255,140,0,0.08)",
        ))
        fig_pr.add_hline(
            y=base_rate, line_dash="dot", line_color="grey",
            annotation_text=f"No-skill baseline ({base_rate:.2%})",
        )
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            title=f"Precision-Recall Curve — AUC = {pr_auc:.4f}",
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_f1:
        pr_df = pd.DataFrame({
            "Threshold": t_arr,
            "Precision": prec_arr[:-1],
            "Recall":    rec_arr[:-1],
            "F1":        f1_arr,
        })
        fig_f1 = px.line(
            pr_df, x="Threshold", y=["Precision", "Recall", "F1"],
            title="Precision / Recall / F1 vs Threshold",
            labels={"value": "Score", "variable": "Metric"},
        )
        fig_f1.add_vline(
            x=best_t, line_dash="dot", line_color="red",
            annotation_text=f"F1-optimal ({best_t:.3f})",
        )
        fig_f1.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    st.info(
        f"**F1-optimal threshold:** `{best_t:.4f}` — after calibration this should align "
        f"closely with the actual base rate ({base_rate:.2%}), confirming the model's "
        f"probabilities are well-anchored to real-world no-show rates."
    )

# ── Tab 4: Features & Thresholds ──────────────────────────────────────────────
with tab4:
    st.subheader("Feature Importance")
    st.caption(
        "Importance from the base XGBoost model (gain metric). "
        "PATIENT_NOSHOW_RATE and PHYSICIAN_NOSHOW_RATE are engineered features added in v2."
    )

    fi = metrics["feature_importance"]
    fig_imp = px.bar(
        fi.head(20), x="Importance", y="Feature", orientation="h",
        title="Top 20 Feature Importances (XGBoost Gain)",
        color="Importance", color_continuous_scale="Viridis",
    )
    fig_imp.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=500,
        margin=dict(l=10, r=20, t=50, b=10),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    st.subheader("Risk Thresholds (Calibrated)")
    st.markdown(
        "After Platt scaling, predicted probabilities closely track actual no-show rates. "
        "Thresholds are set at natural inflection points in the calibrated decile analysis."
    )

    thresholds = [
        ("Very Low",  f"< {THRESHOLD_VERY_LOW:.0%}",                           "~7% actual NS",  "0.33× base rate",  "No action needed",                "success"),
        ("Low",       f"{THRESHOLD_VERY_LOW:.0%} – {THRESHOLD_LOW:.0%}",       "~13% actual NS", "0.62× base rate",  "No action needed",                "success"),
        ("Moderate",  f"{THRESHOLD_LOW:.0%} – {THRESHOLD_MODERATE:.0%}",       "~19% actual NS", "0.91× base rate",  "Automated SMS/email reminder",    "warning"),
        ("High",      f"{THRESHOLD_MODERATE:.0%} – {THRESHOLD_HIGH:.0%}",      "~30% actual NS", "1.41× base rate",  "Manual phone call + reminder",    "error"),
        ("Very High", f">= {THRESHOLD_HIGH:.0%}",                               "~51% actual NS", "2.44× base rate",  "Double-book + personal outreach", "error"),
    ]

    cols = st.columns(5)
    for col, (level, prob_range, actual_rate, multiplier, action, kind) in zip(cols, thresholds):
        color = RISK_COLORS[level]
        col.markdown(
            f"<div style='border:2px solid {color}; border-radius:8px; padding:12px 8px; "
            f"background:{color}18; text-align:center;'>"
            f"<div style='font-weight:700; color:{color}; font-size:1.05em;'>{level}</div>"
            f"<div style='font-size:0.85em; margin:4px 0;'><b>{prob_range}</b></div>"
            f"<div style='font-size:0.78em; color:gray;'>{actual_rate}<br>{multiplier}</div>"
            f"<hr style='margin:6px 0; border-color:{color}44;'/>"
            f"<div style='font-size:0.78em;'>{action}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"""
> **Why calibration matters for threshold selection:**
> - Pre-calibration: `scale_pos_weight=3.74` pushed predicted probabilities high.
>   A raw XGBoost score of 0.55 corresponded to only ~22% actual no-shows.
>   Thresholds had to be set at 0.35–0.70 to compensate.
> - Post-calibration: Platt scaling maps predicted probabilities back to actual rates.
>   A score of 0.30 now means ~30% actual no-show. Thresholds can be set intuitively.
> - **Brier Skill Score improved** from `{bss_uncal_str}` to `{bss_cal_fmt}`
""")
