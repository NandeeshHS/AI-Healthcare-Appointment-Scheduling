import sys
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_all_appointments, render_refresh_button, RISK_COLORS, RISK_ORDER

st.set_page_config(page_title="Overall View", page_icon="📊", layout="wide")
render_refresh_button()

st.title("📊 Overall View — Risk Dashboard")

df = get_all_appointments()

if df.empty:
    st.info("No appointments found. Go to 'Book Appointment' to add data.")
    st.stop()

# ── Date parsing & derived columns ──────────────────────────────────────────
df["appointment_date"] = pd.to_datetime(df["appointment_date"], format="mixed")
df["booking_date"]     = pd.to_datetime(df["booking_date"],     format="mixed")
df["date_only"]        = df["appointment_date"].dt.date
df["lead_time_days"]   = (df["appointment_date"] - df["booking_date"]).dt.days.clip(lower=0)

# ── Date range filter ────────────────────────────────────────────────────────
min_date = df["appointment_date"].min().date()
max_date = df["appointment_date"].max().date()

col_f1, col_f2 = st.columns([2, 1])
with col_f1:
    date_range = st.date_input("Filter by Date Range", value=(min_date, max_date))

if len(date_range) == 2:
    s_date, e_date = date_range
    df = df[
        (df["appointment_date"].dt.date >= s_date) &
        (df["appointment_date"].dt.date <= e_date)
    ]

if df.empty:
    st.warning("No data for the selected date range.")
    st.stop()

# ── KPI cards ────────────────────────────────────────────────────────────────
total_appts = len(df)
high_risk   = df["risk_level"].isin(["High", "Very High"]).sum()
avg_risk    = df["predicted_risk_prob"].mean()
completed   = (df["status"] == "Completed").sum() if "status" in df.columns else 0
no_show     = (df["status"] == "No-Show").sum()   if "status" in df.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Appointments",    f"{total_appts:,}")
c2.metric("High / Very High Risk", f"{high_risk:,}",   delta_color="inverse")
c3.metric("Avg Risk Probability",  f"{avg_risk:.2%}")
c4.metric("Completed",             f"{completed:,}")
c5.metric("No-Show",               f"{no_show:,}")

st.divider()

# ── 7-day Rolling Risk Trend ──────────────────────────────────────────────────
st.subheader("📈 7-Day Rolling Risk Trend")
st.caption("Bar = daily appointment volume (right axis). Line = 7-day rolling average risk probability (left axis).")

df_trend = (
    df.groupby("date_only")
    .agg(
        appts=("predicted_risk_prob", "count"),
        avg_risk=("predicted_risk_prob", "mean"),
    )
    .reset_index()
    .sort_values("date_only")
)
df_trend["date_only"]     = pd.to_datetime(df_trend["date_only"])
df_trend["rolling_risk"]  = df_trend["avg_risk"].rolling(7, min_periods=1).mean()

fig_trend = go.Figure()
fig_trend.add_trace(go.Bar(
    x=df_trend["date_only"], y=df_trend["appts"],
    name="Daily Appointments", yaxis="y2",
    marker_color="rgba(100, 149, 237, 0.3)", showlegend=True,
))
fig_trend.add_trace(go.Scatter(
    x=df_trend["date_only"], y=df_trend["avg_risk"],
    mode="markers", name="Daily Avg Risk",
    marker=dict(color="#aaaaaa", size=4), opacity=0.5,
))
fig_trend.add_trace(go.Scatter(
    x=df_trend["date_only"], y=df_trend["rolling_risk"],
    mode="lines", name="7-Day Rolling Avg",
    line=dict(color="#e74c3c", width=2.5),
))
# Base rate reference line
fig_trend.add_hline(
    y=0.211, line_dash="dot", line_color="grey", line_width=1,
    annotation_text="Base rate 21.1%", annotation_position="right",
)
fig_trend.update_layout(
    yaxis=dict(title="Risk Probability", tickformat=".0%", range=[0, 0.8]),
    yaxis2=dict(title="Appointments", overlaying="y", side="right", showgrid=False),
    xaxis=dict(title=None),
    legend=dict(orientation="h", y=1.08),
    height=340,
    margin=dict(l=10, r=60, t=30, b=10),
    hovermode="x unified",
)
st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ── Calendar heatmap ──────────────────────────────────────────────────────────
st.subheader("📅 Appointment Calendar View")

col_cal1, col_cal2 = st.columns([1, 2])
with col_cal1:
    cal_metric = st.selectbox(
        "Metric",
        ["Number of Appointments", "Average Risk Score", "Average Lead Time", "Active Physicians Count"],
    )
with col_cal2:
    with st.expander("Filter Calendar Data"):
        sel_gender  = st.multiselect("Gender",         df["sex"].unique(),                  default=list(df["sex"].unique()))
        sel_plan    = st.multiselect("Plan Type",       df["primary_plan_type"].unique(),    default=list(df["primary_plan_type"].unique()))
        sel_service = st.multiselect("Medical Service", df["medical_service_code"].unique(), default=list(df["medical_service_code"].unique()))

df_cal = df[
    df["sex"].isin(sel_gender) &
    df["primary_plan_type"].isin(sel_plan) &
    df["medical_service_code"].isin(sel_service)
].copy()

df_cal["week_year"]   = df_cal["appointment_date"].dt.isocalendar().week.astype(int)
df_cal["day_of_week"] = df_cal["appointment_date"].dt.dayofweek
df_cal["year"]        = df_cal["appointment_date"].dt.year

if cal_metric == "Number of Appointments":
    cal_data = df_cal.groupby(["year", "week_year", "day_of_week", "date_only"]).size().reset_index(name="value")
    color_scale, title = "Greens", "Daily Appointment Volume"
elif cal_metric == "Average Risk Score":
    cal_data = df_cal.groupby(["year", "week_year", "day_of_week", "date_only"])["predicted_risk_prob"].mean().reset_index(name="value")
    color_scale, title = "RdYlGn_r", "Daily Average Risk Score"
elif cal_metric == "Average Lead Time":
    cal_data = df_cal.groupby(["year", "week_year", "day_of_week", "date_only"])["lead_time_days"].mean().reset_index(name="value")
    color_scale, title = "Blues", "Daily Average Lead Time (Days)"
else:
    cal_data = df_cal.groupby(["year", "week_year", "day_of_week", "date_only"])["attending_physician"].nunique().reset_index(name="value")
    color_scale, title = "Purples", "Daily Active Physicians"

if cal_data.empty:
    st.warning("No data for selected filters.")
else:
    cal_data["year_week_sort"] = (
        cal_data["year"].astype(str) + "-W" +
        cal_data["week_year"].astype(str).str.zfill(2)
    )

    def _week_label(row):
        d = pd.Timestamp(row["date_only"])
        start = d - pd.Timedelta(days=d.dayofweek)
        end   = start + pd.Timedelta(days=6)
        return f"{start.strftime('%b %d')} – {end.strftime('%b %d')}"

    cal_data["week_label"] = cal_data.apply(_week_label, axis=1)
    week_map = (
        cal_data[["year_week_sort", "week_label"]]
        .drop_duplicates()
        .set_index("year_week_sort")["week_label"]
        .to_dict()
    )

    pivot_data = cal_data.pivot_table(
        index="day_of_week", columns="year_week_sort", values="value", aggfunc="mean"
    )
    pivot_data = pivot_data.reindex(range(7)).sort_index(axis=1)

    x_vals   = list(pivot_data.columns)
    x_labels = [week_map.get(x, x) for x in x_vals]
    y_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    hover_text = []
    for day_idx in range(7):
        row_text = []
        for week_col in x_vals:
            val = pivot_data.loc[day_idx, week_col] if day_idx in pivot_data.index else float("nan")
            if pd.isna(val):
                row_text.append("No Data")
            else:
                match = cal_data[(cal_data["day_of_week"] == day_idx) & (cal_data["year_week_sort"] == week_col)]
                d_str = pd.Timestamp(match.iloc[0]["date_only"]).strftime("%Y-%m-%d") if not match.empty else "?"
                row_text.append(
                    f"Date: {d_str}<br>{'Risk: ' + f'{val:.2%}' if cal_metric == 'Average Risk Score' else 'Value: ' + f'{val:.1f}'}"
                )
        hover_text.append(row_text)

    fig_cal = go.Figure(data=go.Heatmap(
        z=pivot_data.values, x=x_vals, y=y_labels,
        colorscale=color_scale, xgap=4, ygap=4,
        text=hover_text, hoverinfo="text", showscale=True,
    ))
    fig_cal.update_layout(
        title=title, height=300,
        yaxis=dict(autorange="reversed", title=None,
                   tickmode="array", tickvals=list(range(7)), ticktext=y_labels),
        xaxis=dict(title=None, showgrid=False, tickangle=-45,
                   tickmode="array", tickvals=x_vals, ticktext=x_labels),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=80),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

st.divider()

# ── Risk vs. Lead Time ────────────────────────────────────────────────────────
st.subheader("Risk vs. Lead Time Analysis")
fig_bubble = px.scatter(
    df, x="lead_time_days", y="predicted_risk_prob",
    size="age", color="risk_level",
    hover_data=["patient_id", "medical_service_code"],
    title="Risk Probability vs. Lead Time (bubble size = Age)",
    labels={"lead_time_days": "Lead Time (Days)", "predicted_risk_prob": "Risk Probability"},
    color_discrete_map=RISK_COLORS,
    category_orders={"risk_level": RISK_ORDER},
    size_max=20,
)
fig_bubble.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig_bubble, use_container_width=True)

# ── Risk distribution (donut) & gender box ────────────────────────────────────
col_c1, col_c2 = st.columns(2)

with col_c1:
    st.subheader("Risk Level Distribution")
    present_levels = [r for r in RISK_ORDER if r in df["risk_level"].values]
    fig_donut = px.pie(
        df, names="risk_level",
        title="Appointments by Risk Level",
        category_orders={"risk_level": present_levels},
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        hole=0.5,
    )
    fig_donut.update_traces(textinfo="percent+label", textposition="outside")
    fig_donut.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_donut, use_container_width=True)

with col_c2:
    st.subheader("Risk by Gender")
    fig_box = px.box(
        df, x="sex", y="predicted_risk_prob", color="sex",
        title="Risk Probability Distribution by Gender",
        labels={"predicted_risk_prob": "Risk Probability", "sex": "Gender"},
    )
    fig_box.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

# ── Recent appointments table ─────────────────────────────────────────────────
st.subheader("Recent Appointments")
cols_show = ["patient_id", "booking_date", "appointment_date", "risk_level",
             "predicted_risk_prob", "status"]
cols_show = [c for c in cols_show if c in df.columns]
st.dataframe(
    df[cols_show].sort_values("appointment_date", ascending=False).head(10),
    column_config={
        "predicted_risk_prob": st.column_config.ProgressColumn(
            "Risk Prob", format="%.2f", min_value=0, max_value=1
        ),
    },
    hide_index=True,
    use_container_width=True,
)
