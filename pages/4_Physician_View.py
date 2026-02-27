import sys
import os

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_all_appointments, render_refresh_button, RISK_COLORS, RISK_ORDER

st.set_page_config(page_title="Physician View", page_icon="👨‍⚕️", layout="wide")
render_refresh_button()

st.title("👨‍⚕️ Physician Risk View")
st.markdown("Identify which physicians have disproportionately high-risk patient panels.")

df = get_all_appointments()

if df.empty:
    st.info("No appointment data available.")
    st.stop()

# ── Sidebar: physician drilldown ──────────────────────────────────────────────
st.sidebar.header("Drilldown")
all_physicians = sorted(df["attending_physician"].dropna().unique().tolist())
selected_physician = st.sidebar.selectbox(
    "Select Physician (optional)", ["All"] + all_physicians
)

# ── Aggregate physician stats ─────────────────────────────────────────────────
physician_stats = (
    df.groupby("attending_physician")
    .agg(
        total_patients=("id", "count"),
        avg_risk_score=("predicted_risk_prob", "mean"),
        high_risk_count=("risk_level", lambda x: x.isin(["High", "Very High"]).sum()),
    )
    .reset_index()
)
physician_stats.columns = [
    "Physician", "Total Patients", "Avg Risk Score", "High/VH Risk Count"
]
physician_stats["High Risk Rate %"] = (
    physician_stats["High/VH Risk Count"] / physician_stats["Total Patients"] * 100
).round(1)
physician_stats = physician_stats.sort_values("High Risk Rate %", ascending=False).reset_index(drop=True)

# ── Section 1: Scatter quadrant — Risk Rate vs Volume ─────────────────────────
st.subheader("Risk Rate vs. Patient Volume")
st.caption(
    "Each bubble = one physician. Size = total patients. "
    "Dashed lines = median values. Top-right quadrant = highest operational risk."
)

med_volume   = physician_stats["Total Patients"].median()
med_risk_rate = physician_stats["High Risk Rate %"].median()

flagged = physician_stats[
    (physician_stats["Total Patients"]  >= med_volume) &
    (physician_stats["High Risk Rate %"] >= med_risk_rate)
]

fig_scatter = px.scatter(
    physician_stats,
    x="Total Patients",
    y="High Risk Rate %",
    size="Total Patients",
    color="High Risk Rate %",
    color_continuous_scale="RdYlGn_r",
    hover_name="Physician",
    hover_data={
        "Total Patients": True,
        "High Risk Rate %": ":.1f",
        "Avg Risk Score": ":.3f",
        "High/VH Risk Count": True,
    },
    title="Physician Risk Rate vs. Patient Volume",
    labels={"High Risk Rate %": "High Risk Rate (%)"},
    size_max=40,
)

# Quadrant lines
fig_scatter.add_vline(
    x=med_volume, line_dash="dash", line_color="grey", line_width=1.5,
    annotation_text=f"Median volume ({med_volume:.0f})", annotation_position="top right",
)
fig_scatter.add_hline(
    y=med_risk_rate, line_dash="dash", line_color="grey", line_width=1.5,
    annotation_text=f"Median risk rate ({med_risk_rate:.1f}%)", annotation_position="right",
)

# Quadrant labels
x_max = physician_stats["Total Patients"].max() * 1.05
y_max = physician_stats["High Risk Rate %"].max() * 1.05
for txt, x_ref, y_ref, color in [
    ("Low Volume\nLow Risk ✅",   med_volume * 0.3, med_risk_rate * 0.4,  "#27ae60"),
    ("High Volume\nLow Risk ✅",  x_max * 0.75,     med_risk_rate * 0.4,  "#2ecc71"),
    ("Low Volume\nHigh Risk ⚠️",  med_volume * 0.3, y_max * 0.85,         "#e67e22"),
    ("High Volume\nHigh Risk 🔴", x_max * 0.75,     y_max * 0.85,         "#e74c3c"),
]:
    fig_scatter.add_annotation(
        x=x_ref, y=y_ref, text=txt, showarrow=False,
        font=dict(size=10, color=color), opacity=0.7,
    )

fig_scatter.update_layout(
    height=460,
    coloraxis_showscale=True,
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Section 2: Flagged physicians callout ─────────────────────────────────────
if not flagged.empty:
    st.warning(
        f"⚠️ **{len(flagged)} physician(s)** are in the high-volume & high-risk quadrant "
        f"(above-median on both dimensions). Consider targeted interventions."
    )
    flagged_display = flagged[["Physician", "Total Patients", "High Risk Rate %", "Avg Risk Score", "High/VH Risk Count"]].copy()
    st.dataframe(
        flagged_display,
        column_config={
            "High Risk Rate %": st.column_config.ProgressColumn(
                "High Risk Rate", format="%.1f%%", min_value=0, max_value=100
            ),
            "Avg Risk Score": st.column_config.NumberColumn(
                "Avg Risk Score", format="%.3f"
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# ── Section 3: Full physician ranking table ───────────────────────────────────
st.subheader("All Physicians — Risk Ranking")
st.caption("Sorted by High Risk Rate % (descending). Physicians with more high-risk patients per total panel are prioritised.")

st.dataframe(
    physician_stats,
    column_config={
        "High Risk Rate %": st.column_config.ProgressColumn(
            "High Risk Rate", format="%.1f%%", min_value=0, max_value=100
        ),
        "Avg Risk Score": st.column_config.NumberColumn(
            "Avg Risk Score", format="%.3f"
        ),
        "Total Patients": st.column_config.NumberColumn("Total Patients"),
        "High/VH Risk Count": st.column_config.NumberColumn("High/VH Count"),
    },
    hide_index=True,
    use_container_width=True,
)

st.divider()

# ── Section 4: Single physician drilldown ─────────────────────────────────────
if selected_physician != "All":
    st.subheader(f"Drilldown: {selected_physician}")
    df_phys = df[df["attending_physician"] == selected_physician].copy()
    df_phys["appointment_date"] = pd.to_datetime(df_phys["appointment_date"], format="mixed")

    p1, p2, p3 = st.columns(3)
    p1.metric("Total Appointments", len(df_phys))
    p2.metric(
        "High/Very High Risk",
        df_phys["risk_level"].isin(["High", "Very High"]).sum(),
    )
    p3.metric("Avg Risk Score", f"{df_phys['predicted_risk_prob'].mean():.3f}")

    # Risk distribution for this physician
    risk_counts = (
        df_phys["risk_level"]
        .value_counts()
        .reindex(RISK_ORDER)
        .dropna()
        .reset_index()
    )
    risk_counts.columns = ["Risk Level", "Count"]
    fig_phys_pie = px.pie(
        risk_counts, names="Risk Level", values="Count",
        hole=0.45, color="Risk Level",
        color_discrete_map=RISK_COLORS,
        title=f"Risk Distribution — {selected_physician}",
    )
    fig_phys_pie.update_traces(textinfo="percent+label")
    fig_phys_pie.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_phys_pie, use_container_width=True)

    cols_show = ["patient_id", "appointment_date", "risk_level", "predicted_risk_prob", "status"]
    cols_show = [c for c in cols_show if c in df_phys.columns]
    st.dataframe(
        df_phys[cols_show].sort_values("appointment_date", ascending=False),
        column_config={
            "predicted_risk_prob": st.column_config.ProgressColumn(
                "Risk Prob", format="%.2f", min_value=0, max_value=1
            ),
        },
        hide_index=True,
        use_container_width=True,
    )
