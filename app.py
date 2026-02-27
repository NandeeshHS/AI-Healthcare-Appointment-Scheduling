import sys
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_all_appointments, render_refresh_button, RISK_COLORS, RISK_ORDER

st.set_page_config(
    page_title="Healthcare Risk Intelligence",
    page_icon="🏥",
    layout="wide",
)
render_refresh_button()

st.title("🏥 Healthcare Appointment Risk Intelligence")
st.markdown(
    "ML-powered no-show prediction · XGBoost + Platt Calibration · "
    "ROC-AUC 0.712 · Brier Skill Score +0.11"
)

st.divider()


@st.cache_data(ttl=60)
def load_dashboard_data():
    df = get_all_appointments()
    if df.empty:
        return df
    df["appointment_date"] = pd.to_datetime(df["appointment_date"], format="mixed")
    return df


df = load_dashboard_data()

if df.empty:
    st.info(
        "No appointments in the database yet. "
        "Go to **Book Appointment** to add your first record, "
        "then return here to see the live dashboard."
    )
    st.markdown(
        "**Tech Stack:** Python · XGBoost · Scikit-learn · Platt Calibration · "
        "Streamlit · SQLite · Plotly · Optuna"
    )
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────
total       = len(df)
high_risk   = df["risk_level"].isin(["High", "Very High"]).sum()
avg_risk    = df["predicted_risk_prob"].mean()
no_show_cnt = (df["status"] == "No-Show").sum() if "status" in df.columns else 0
completed   = (df["status"] == "Completed").sum() if "status" in df.columns else 0
ns_rate     = no_show_cnt / (no_show_cnt + completed) if (no_show_cnt + completed) > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Appointments", f"{total:,}")
c2.metric(
    "High / Very High Risk",
    f"{high_risk:,}",
    delta=f"{high_risk/total:.1%} of total",
    delta_color="inverse",
)
c3.metric("Avg Risk Probability", f"{avg_risk:.2%}")
c4.metric(
    "Observed No-Show Rate",
    f"{ns_rate:.2%}" if (no_show_cnt + completed) > 0 else "—",
    help="Among resolved appointments (Completed + No-Show)",
)

st.divider()

# ── Main content: donut + high-risk alert table ───────────────────────────────
col_left, col_right = st.columns([6, 4])

with col_left:
    st.subheader("Risk Distribution")
    risk_counts = (
        df["risk_level"]
        .value_counts()
        .reindex(RISK_ORDER)
        .dropna()
        .reset_index()
    )
    risk_counts.columns = ["Risk Level", "Count"]
    fig_donut = px.pie(
        risk_counts,
        names="Risk Level",
        values="Count",
        hole=0.52,
        color="Risk Level",
        color_discrete_map=RISK_COLORS,
        category_orders={"Risk Level": RISK_ORDER},
    )
    fig_donut.update_traces(
        textinfo="percent+label",
        textposition="outside",
        pull=[0.04 if r in ("High", "Very High") else 0 for r in risk_counts["Risk Level"]],
    )
    fig_donut.update_layout(
        showlegend=True,
        legend=dict(orientation="v", x=1.0, y=0.5),
        margin=dict(l=20, r=20, t=10, b=10),
        height=320,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_right:
    st.subheader("🚨 Upcoming High Risk")
    today = datetime.now().date()
    upcoming = df[
        df["risk_level"].isin(["High", "Very High"]) &
        (df.get("status", pd.Series(["Scheduled"] * len(df))) == "Scheduled") &
        (df["appointment_date"].dt.date >= today)
    ].sort_values("appointment_date").head(6)

    if upcoming.empty:
        st.success("No upcoming High/Very High risk appointments. ✅")
    else:
        cols_show = [c for c in ["patient_id", "appointment_date", "risk_level", "predicted_risk_prob"] if c in upcoming.columns]
        st.dataframe(
            upcoming[cols_show],
            column_config={
                "predicted_risk_prob": st.column_config.ProgressColumn(
                    "Risk %", format="%.2f", min_value=0, max_value=1
                ),
                "risk_level": st.column_config.TextColumn("Risk"),
                "appointment_date": st.column_config.DatetimeColumn(
                    "Appt Date", format="MMM D, YYYY h:mm a"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )
        st.caption(f"Showing up to 6 upcoming High/Very High risk appointments.")

st.divider()

# ── Status breakdown ──────────────────────────────────────────────────────────
if "status" in df.columns:
    st.subheader("Appointment Status Overview")
    status_risk = (
        df.groupby(["status", "risk_level"])
        .size()
        .reset_index(name="count")
    )
    fig_status = px.bar(
        status_risk,
        x="status", y="count", color="risk_level",
        color_discrete_map=RISK_COLORS,
        category_orders={"risk_level": RISK_ORDER},
        title="Appointments by Status and Risk Level",
        labels={"count": "Appointments", "status": "Status", "risk_level": "Risk Level"},
        barmode="stack",
    )
    fig_status.update_layout(
        legend_title="Risk Level",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_status, use_container_width=True)

st.info("Use the sidebar to navigate between pages. Click **🔄 Refresh Data** to simulate appointment outcomes.")
