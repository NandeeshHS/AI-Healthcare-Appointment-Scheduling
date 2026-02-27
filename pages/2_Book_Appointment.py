import sys
import os
from datetime import datetime, time

import streamlit as st
import plotly.graph_objects as go
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_artifacts, predict_risk, save_appointment,
    get_shap_values, get_patient_appointment_count,
    get_patient_noshow_rate, get_physician_noshow_rate,
    render_refresh_button,
    THRESHOLD_VERY_LOW, THRESHOLD_LOW, THRESHOLD_MODERATE, THRESHOLD_HIGH,
    RISK_COLORS,
)

st.set_page_config(page_title="Book Appointment", page_icon="📅", layout="wide")
render_refresh_button()

st.title("📅 Book a New Appointment")
st.markdown("Enter patient details below to predict the risk of No-Show / Cancellation.")

# ── Load model artifacts for dropdown options ─────────────────────────────────
try:
    model, encoders, feature_cols = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# ── Helper: safe default index ────────────────────────────────────────────────
def _idx(classes, value, fallback=0):
    try:
        return list(classes).index(value)
    except ValueError:
        return fallback

# ── Booking form ──────────────────────────────────────────────────────────────
with st.form("appointment_form"):
    st.markdown("### 1. Patient Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        patient_id = st.text_input("Patient ID", value="P12345")
        sex        = st.selectbox("Gender", encoders["SEX"].classes_,
                                  index=_idx(encoders["SEX"].classes_, "F"))
    with c2:
        age      = st.number_input("Age", min_value=0, max_value=120, value=45)
        language = st.selectbox("Language", encoders["LANGUAGE"].classes_,
                                index=_idx(encoders["LANGUAGE"].classes_, "English"))
    with c3:
        zipcode          = st.selectbox("Zipcode (Area)", encoders["ZIPCODE3"].classes_)
        primary_plan_type= st.selectbox("Primary Plan Type", encoders["PRIMARY_PLAN_TYPE"].classes_,
                                        index=_idx(encoders["PRIMARY_PLAN_TYPE"].classes_, "Medicare"))

    st.markdown("---")
    st.markdown("### 2. Appointment Details")
    c4, c5, c6 = st.columns(3)
    with c4:
        booking_date     = st.date_input("Booking Date",     value=datetime.now().date())
        appointment_date = st.date_input("Appointment Date", value=datetime.now().date())
    with c5:
        appointment_time  = st.time_input("Appointment Time", value=time(9, 0))
        appt_type_code    = st.selectbox("Appointment Type", encoders["APPT_TYPE_CODE"].classes_)
    with c6:
        medical_service_code = st.selectbox("Medical Service", encoders["MEDICAL_SERVICE_CODE"].classes_)
        facility_code        = st.selectbox("Facility",         encoders["FACILITY_CODE"].classes_)

    st.markdown("---")
    st.markdown("### 3. Medical Context")
    c7, c8 = st.columns(2)
    with c7:
        nurse_unit_code  = st.selectbox("Nurse Unit",    encoders["NURSE_UNIT_CODE"].classes_)
        service_line_used= st.selectbox("Service Line",  encoders["SERVICE_LINE_USED"].classes_)
    with c8:
        attending_physician = st.selectbox("Attending Physician", encoders["ATTENDING_PHYSICIAN"].classes_)
        referring_physician = st.selectbox("Referring Physician", encoders["REFERRING_PHYSICIAN"].classes_)

    st.markdown("---")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        predict_btn = st.form_submit_button(
            "🔮 Predict Risk", use_container_width=True, type="primary"
        )

    st.markdown(
        "<div style='text-align:center;font-size:0.8em;color:gray;'>"
        f"<b>Risk Thresholds:</b> "
        f"🟢 Very Low (&lt;{THRESHOLD_VERY_LOW:.0%}) | "
        f"🟢 Low ({THRESHOLD_VERY_LOW:.0%}–{THRESHOLD_LOW:.0%}) | "
        f"🟠 Moderate ({THRESHOLD_LOW:.0%}–{THRESHOLD_MODERATE:.0%}) | "
        f"🔴 High ({THRESHOLD_MODERATE:.0%}–{THRESHOLD_HIGH:.0%}) | "
        f"🔴 Very High (&gt;{THRESHOLD_HIGH:.0%})"
        "</div>",
        unsafe_allow_html=True,
    )

# ── On predict ────────────────────────────────────────────────────────────────
if predict_btn:
    full_appt_dt    = datetime.combine(appointment_date, appointment_time)
    full_booking_dt = datetime.combine(booking_date, time(0, 0))

    data = {
        "patient_id":           patient_id,
        "sex":                  sex,
        "age":                  int(age),
        "language":             language,
        "zipcode":              zipcode,
        "primary_plan_type":    primary_plan_type,
        "booking_date":         full_booking_dt,
        "appointment_date":     full_appt_dt,
        "appt_type_code":       appt_type_code,
        "medical_service_code": medical_service_code,
        "facility_code":        facility_code,
        "nurse_unit_code":      nurse_unit_code,
        "service_line_used":    service_line_used,
        "attending_physician":  attending_physician,
        "referring_physician":  referring_physician,
    }

    with st.status("Analysing risk profile…", expanded=True) as status:
        try:
            status.write("Calculating risk probability…")
            prob, risk_level = predict_risk(data)

            status.write("Generating SHAP explanation…")
            shap_exp = get_shap_values(data)   # fast with TreeExplainer

            st.session_state["prediction_result"] = {
                "prob":       prob,
                "risk_level": risk_level,
                "data":       data,
                "shap_exp":   shap_exp,        # cache so we don't recompute
            }
            status.update(label="Analysis complete!", state="complete", expanded=False)

        except Exception as e:
            status.update(label="Analysis failed", state="error")
            st.error(f"Prediction error: {e}")

# ── Display results ───────────────────────────────────────────────────────────
if "prediction_result" in st.session_state:
    res        = st.session_state["prediction_result"]
    prob       = res["prob"]
    risk_level = res["risk_level"]
    data       = res["data"]
    shap_exp   = res["shap_exp"]

    st.divider()
    st.subheader("Prediction Results")

    # ── Gauge + Risk Badge + Action ───────────────────────────────────────────
    gauge_color = RISK_COLORS[risk_level]
    _, gauge_col, _ = st.columns([1, 2, 1])
    with gauge_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 42, "color": gauge_color}},
            title={"text": "No-Show Probability", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickwidth": 1},
                "bar":  {"color": gauge_color, "thickness": 0.25},
                "steps": [
                    {"range": [0,  10], "color": "#d5f5e3"},
                    {"range": [10, 15], "color": "#a9dfbf"},
                    {"range": [15, 25], "color": "#fdebd0"},
                    {"range": [25, 40], "color": "#fad7a0"},
                    {"range": [40, 100],"color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": gauge_color, "width": 4},
                    "thickness": 0.85,
                    "value": prob * 100,
                },
            },
        ))
        fig_gauge.update_layout(
            height=260,
            margin=dict(l=30, r=30, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    cr1, cr2, cr3 = st.columns(3)
    cr1.metric("Risk Probability", f"{prob:.2%}")
    cr2.markdown(
        f"<div style='text-align:center; padding:12px 8px; border-radius:8px; "
        f"background-color:{gauge_color}22; border:2px solid {gauge_color};'>"
        f"<span style='font-size:1.1em; color:{gauge_color}; font-weight:700;'>"
        f"Risk Level</span><br>"
        f"<span style='font-size:1.6em; font-weight:800; color:{gauge_color};'>"
        f"{risk_level}</span></div>",
        unsafe_allow_html=True,
    )
    with cr3:
        if risk_level in ("High", "Very High"):
            st.error("⚠️ **Action Required**\n\nManual outreach + consider double-booking")
        elif risk_level == "Moderate":
            st.warning("📩 **Action Suggested**\n\nSend automated SMS / email reminder")
        else:
            st.success("✅ **No Action Needed**\n\nStandard appointment — no extra intervention")

    # ── SHAP waterfall ────────────────────────────────────────────────────────
    st.subheader("Risk Factors (SHAP Explanation)")

    if shap_exp is not None:
        values        = shap_exp.values
        feature_names = shap_exp.feature_names

        # Sort all features by absolute impact (smallest → largest for waterfall)
        order = np.argsort(np.abs(values))

        sorted_values = values[order]
        sorted_names  = np.array(feature_names)[order]

        # ── Human-readable feature values for labels ──────────────────────────
        def _display_val(feature: str, d: dict) -> str:
            fmap = {
                "age":                  lambda: d["age"],
                "language":             lambda: d["language"],
                "sex":                  lambda: d["sex"],
                "zipcode":              lambda: d["zipcode"],
                "primary_plan_type":    lambda: d["primary_plan_type"],
                "appt_type_code":       lambda: d["appt_type_code"],
                "medical_service_code": lambda: d["medical_service_code"],
                "facility_code":        lambda: d["facility_code"],
                "nurse_unit_code":      lambda: d["nurse_unit_code"],
                "service_line_used":    lambda: d["service_line_used"],
                "attending_physician":  lambda: d["attending_physician"],
                "referring_physician":  lambda: d["referring_physician"],
            }
            key = feature.lower()
            if key in fmap:
                return str(fmap[key]())
            if feature == "LEAD_TIME_DAYS":
                return str((d["appointment_date"] - d["booking_date"]).days)
            if feature == "APPT_DOW":
                days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                # isoweekday: 1=Mon … 7=Sun; index with -1
                return days[d["appointment_date"].isoweekday() - 1]
            if feature == "APPT_HOUR":
                return str(d["appointment_date"].hour)
            if feature == "IS_WEEKEND":
                return "Yes" if d["appointment_date"].isoweekday() in (6, 7) else "No"
            if feature == "APPT_COUNT_BY_PATIENT":
                return str(get_patient_appointment_count(d["patient_id"]))
            if feature == "PATIENT_NOSHOW_RATE":
                return f"{get_patient_noshow_rate(d['patient_id']):.1%}"
            if feature == "PHYSICIAN_NOSHOW_RATE":
                return f"{get_physician_noshow_rate(d['attending_physician']):.1%}"
            if feature == "SEASON_BUCKET":
                m = d["appointment_date"].month
                return ("Winter" if m in (12,1,2) else "Spring" if m in (3,4,5)
                        else "Summer" if m in (6,7,8) else "Fall")
            return ""

        # Top-5 text summary (highest absolute impact first)
        st.markdown("#### Top 5 Influential Features")
        for idx in order[-1:-6:-1]:
            v    = values[idx]
            name = feature_names[idx]
            dval = _display_val(name, data)
            impact_txt = "Increases Risk" if v > 0 else "Decreases Risk"
            col  = "red" if v > 0 else "green"
            st.markdown(f"- **{name}** ({dval}): :{col}[{impact_txt}] ({v:+.3f})")

        # Waterfall chart
        bar_names = [f"{n} ({_display_val(n, data)})" for n in sorted_names]

        fig = go.Figure(go.Waterfall(
            name="Impact", orientation="h",
            measure=["relative"] * len(sorted_values),
            x=sorted_values,
            y=bar_names,
            text=[f"{v:+.3f}" for v in sorted_values],
            textposition="outside",
            connector={"line": {"color": "rgb(63,63,63)"}},
            decreasing={"marker": {"color": "#2ecc71"}},
            increasing={"marker": {"color": "#ff4b4b"}},
        ))
        fig.update_layout(
            title="All Feature Contributions (Red = Higher Risk, Green = Lower Risk)",
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=max(400, len(sorted_values) * 28),
            margin=dict(l=10, r=80, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("SHAP explanation not available for this prediction.")

    st.divider()

    # ── Confirm & book ────────────────────────────────────────────────────────
    _, mid2, _ = st.columns([1, 2, 1])
    with mid2:
        if st.button("📝 Confirm & Book Appointment", type="primary", use_container_width=True):
            try:
                save_appointment(data, prob, risk_level)
                st.success("Appointment booked and saved successfully!")
                del st.session_state["prediction_result"]
                st.rerun()
            except Exception as e:
                st.error(f"Error saving appointment: {e}")
