import sys
import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_all_appointments, render_refresh_button, RISK_ORDER

st.set_page_config(page_title="My Appointments", page_icon="📋", layout="wide")
render_refresh_button()

st.title("📋 My Appointments")
st.markdown("View, filter, and export all scheduled appointments.")

try:
    df = get_all_appointments()
except Exception as e:
    st.error(f"Error loading appointments: {e}")
    st.stop()

if df.empty:
    st.info("No appointments found in the database.")
    st.stop()

# ── Date parse ───────────────────────────────────────────────────────────────
df["appointment_date"] = pd.to_datetime(df["appointment_date"], format="mixed")

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")

search_id = st.sidebar.text_input("Search by Patient ID", placeholder="e.g. P12345")

selected_risks  = st.sidebar.multiselect("Risk Level",  options=RISK_ORDER, default=RISK_ORDER)
physicians      = sorted(df["attending_physician"].dropna().unique().tolist())
selected_phys   = st.sidebar.multiselect("Physician",   options=physicians,  default=physicians)
status_opts     = sorted(df["status"].dropna().unique().tolist()) if "status" in df.columns else []
selected_status = st.sidebar.multiselect("Status",      options=status_opts, default=status_opts)

min_date   = df["appointment_date"].min().date()
max_date   = df["appointment_date"].max().date()
date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date))

# ── Apply filters ─────────────────────────────────────────────────────────────
fdf = df[
    df["risk_level"].isin(selected_risks) &
    df["attending_physician"].isin(selected_phys)
].copy()

if status_opts and selected_status:
    fdf = fdf[fdf["status"].isin(selected_status)]

if len(date_range) == 2:
    fdf = fdf[
        (fdf["appointment_date"].dt.date >= date_range[0]) &
        (fdf["appointment_date"].dt.date <= date_range[1])
    ]

if search_id.strip():
    fdf = fdf[fdf["patient_id"].astype(str).str.contains(search_id.strip(), case=False, na=False)]

# ── Urgent attention banner ───────────────────────────────────────────────────
cutoff = (datetime.now() + timedelta(hours=48)).date()
upcoming_high = fdf[
    fdf["risk_level"].isin(["High", "Very High"]) &
    (fdf.get("status", pd.Series(["Scheduled"] * len(fdf), index=fdf.index)) == "Scheduled") &
    (fdf["appointment_date"].dt.date <= cutoff)
]
if not upcoming_high.empty:
    st.error(
        f"🚨 **{len(upcoming_high)} High / Very High risk appointment(s)** scheduled within the "
        f"next 48 hours. Review immediately and initiate outreach."
    )

# ── KPI row + download button ─────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 3])
c1.metric("Total", f"{len(fdf):,}")
c2.metric("High / Very High Risk", int(fdf["risk_level"].isin(["High", "Very High"]).sum()))

avg_risk = fdf["predicted_risk_prob"].mean()
c3.metric("Avg Risk Probability", f"{avg_risk:.2%}" if pd.notna(avg_risk) else "—")

no_show = int((fdf["status"] == "No-Show").sum()) if "status" in fdf.columns else 0
c4.metric("No-Show", f"{no_show:,}")

with c5:
    st.download_button(
        "⬇ Download CSV",
        fdf.to_csv(index=False).encode("utf-8"),
        "appointments.csv",
        "text/csv",
        key="download-csv",
        use_container_width=True,
    )

st.divider()

if fdf.empty:
    st.info("No appointments match the selected filters.")
    st.stop()

# ── Table ─────────────────────────────────────────────────────────────────────
st.subheader(f"Appointments ({len(fdf):,} records)")

# Sort: soonest Scheduled high-risk first, then by appointment date
fdf_sorted = fdf.sort_values(
    ["appointment_date"],
    ascending=[True],
)

st.dataframe(
    fdf_sorted,
    column_config={
        "predicted_risk_prob": st.column_config.ProgressColumn(
            "Risk Probability",
            help="Probability of No-Show / Cancellation",
            format="%.2f",
            min_value=0,
            max_value=1,
        ),
        "risk_level":       st.column_config.TextColumn("Risk Level"),
        "status":           st.column_config.TextColumn("Status"),
        "appointment_date": st.column_config.DatetimeColumn(
            "Appointment Date", format="MMM D, YYYY h:mm a"
        ),
        "booking_date": st.column_config.DatetimeColumn(
            "Booking Date", format="MMM D, YYYY"
        ),
    },
    hide_index=True,
    use_container_width=True,
)
