"""
Healthcare Appointment No-Show — Core ML Utilities
====================================================
Fixes applied vs original:
  1. IS_WEEKEND: corrected to [6, 7] (ISO isoweekday scale, 6=Sat, 7=Sun).
     Both training data (CSV DOW 1-7) and inference now use isoweekday().
  2. APPT_DOW: inference uses isoweekday() (1=Mon..7=Sun) to match
     the 1-7 scale stored in the raw CSV training data.
  3. TIME_OF_DAY_BUCKET removed: collinear with APPT_HOUR (continuous).
  4. SHAP: uses XGBoost native pred_contribs on the raw base model.
     xgboost_base_model.pkl is loaded separately because the calibrated
     wrapper (CalibratedClassifierCV) does not expose a booster attribute.
  5. PATIENT_NOSHOW_RATE: live lookup from the appointments DB — each
     patient's actual historical no-show rate at inference time.
  6. PHYSICIAN_NOSHOW_RATE: live lookup from the appointments DB — each
     physician's actual historical no-show rate (defaults to base rate
     for physicians with no prior records).
  7. Risk thresholds: data-driven from calibrated model held-out test set.
     After Platt scaling, predicted probabilities closely track actual rates
     (Brier Skill Score = +0.1125, up from -0.2647 pre-calibration).

     Base rate = 21.1%
     Very Low  < 0.10  actual NS ~7%    (0.33x base)  No action
     Low       < 0.15  actual NS ~13%   (0.62x base)  No action
     Moderate  < 0.25  actual NS ~19%   (0.91x base)  Automated reminder
     High      < 0.40  actual NS ~30%   (1.41x base)  Manual outreach
     Very High >= 0.40 actual NS ~51%   (2.44x base)  Double-book + outreach
"""

import os
import sqlite3
from datetime import datetime

import joblib
import pandas as pd

DB_NAME = "healthcare.db"
MODELS_DIR = "models"

# ---------------------------------------------------------------------------
# Risk threshold boundaries — data-driven from calibrated model test set.
# Platt scaling corrected systematic overconfidence from scale_pos_weight.
# Probabilities now track actual rates: predicted ~30% → actual ~30% NS.
# Boundaries set at natural inflection points in the decile analysis.
# ---------------------------------------------------------------------------
THRESHOLD_VERY_LOW = 0.10   # < 10%  actual NS ~7%    (0.33x base)  No action
THRESHOLD_LOW      = 0.15   # 10-15% actual NS ~13%   (0.62x base)  No action
THRESHOLD_MODERATE = 0.25   # 15-25% actual NS ~19%   (0.91x base)  Automated reminder
THRESHOLD_HIGH     = 0.40   # 25-40% actual NS ~30%   (1.41x base)  Manual outreach
# >= 0.40                   actual NS ~51%  (2.44x base)  Very High: Double-book

# ---------------------------------------------------------------------------
# Design system — consistent risk colors and ordering across all pages
# ---------------------------------------------------------------------------
RISK_COLORS = {
    "Very Low":  "#27ae60",
    "Low":       "#2ecc71",
    "Moderate":  "#f39c12",
    "High":      "#e67e22",
    "Very High": "#e74c3c",
}
RISK_ORDER = ["Very Low", "Low", "Moderate", "High", "Very High"]


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def load_artifacts():
    model        = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    encoders     = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    return model, encoders, feature_cols


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def get_patient_appointment_count(patient_id: str) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM appointments WHERE patient_id = ?", (patient_id,)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_patient_noshow_rate(patient_id: str) -> float:
    """Return historical no-show rate for a patient (0.0 for first visit)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN status = 'No-Show' THEN 1 ELSE 0 END) AS no_shows
        FROM appointments
        WHERE patient_id = ?
        """,
        (patient_id,),
    )
    row = cursor.fetchone()
    conn.close()
    total    = row["total"]
    no_shows = row["no_shows"] or 0
    return float(no_shows / total) if total > 0 else 0.0


def get_physician_noshow_rate(attending_physician: str) -> float:
    """
    Return historical no-show rate for a physician.
    Falls back to the population base rate (0.211) when the physician has
    no prior records — a sensible Bayesian prior.
    """
    POPULATION_BASE_RATE = 0.211
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN status = 'No-Show' THEN 1 ELSE 0 END) AS no_shows
        FROM appointments
        WHERE attending_physician = ?
        """,
        (attending_physician,),
    )
    row = cursor.fetchone()
    conn.close()
    total    = row["total"]
    no_shows = row["no_shows"] or 0
    return float(no_shows / total) if total > 0 else POPULATION_BASE_RATE


# ---------------------------------------------------------------------------
# Feature engineering (must mirror extract_model.py definitions exactly)
# ---------------------------------------------------------------------------
def preprocess_input(data: dict, encoders: dict, feature_cols: list) -> pd.DataFrame:
    """Convert a raw booking dict into a model-ready single-row DataFrame."""

    appt_date    = pd.to_datetime(data["appointment_date"], format="mixed")
    booking_date = pd.to_datetime(data["booking_date"],     format="mixed")

    lead_time_days = max(0, (appt_date - booking_date).days)

    # Use isoweekday() → 1=Mon … 7=Sun, matching the 1-7 ISO scale in the
    # training CSV.  dayofweek() returns 0-6 and would be off-by-one.
    appt_dow  = appt_date.isoweekday()   # 1=Mon, 2=Tue … 6=Sat, 7=Sun
    appt_hour = appt_date.hour

    # weekend = [6, 7] (Saturday, Sunday in ISO 1-7 scale)
    is_weekend = 1 if appt_dow in [6, 7] else 0

    patient_count         = get_patient_appointment_count(data["patient_id"])
    patient_noshow_rate   = get_patient_noshow_rate(data["patient_id"])
    physician_noshow_rate = get_physician_noshow_rate(data["attending_physician"])

    # Season bucket
    month = appt_date.month
    if   month in (12, 1, 2): season = "Winter"
    elif month in (3,  4, 5): season = "Spring"
    elif month in (6,  7, 8): season = "Summer"
    else:                      season = "Fall"

    input_dict = {
        "AGE":                    [data["age"]],
        "LEAD_TIME_DAYS":         [lead_time_days],
        "APPT_DOW":               [appt_dow],
        "APPT_HOUR":              [appt_hour],
        "IS_WEEKEND":             [is_weekend],
        "APPT_COUNT_BY_PATIENT":  [patient_count],
        "PATIENT_NOSHOW_RATE":    [patient_noshow_rate],
        "PHYSICIAN_NOSHOW_RATE":  [physician_noshow_rate],
        "SEX":                    [data["sex"]],
        "LANGUAGE":               [data["language"]],
        "ZIPCODE3":               [data["zipcode"]],
        "APPT_TYPE_CODE":         [data["appt_type_code"]],
        "MEDICAL_SERVICE_CODE":   [data["medical_service_code"]],
        "FACILITY_CODE":          [data["facility_code"]],
        "NURSE_UNIT_CODE":        [data["nurse_unit_code"]],
        "SERVICE_LINE_USED":      [data["service_line_used"]],
        "ATTENDING_PHYSICIAN":    [data["attending_physician"]],
        "REFERRING_PHYSICIAN":    [data["referring_physician"]],
        "PRIMARY_PLAN_TYPE":      [data["primary_plan_type"]],
        "SEASON_BUCKET":          [season],
    }

    df = pd.DataFrame(input_dict)

    # Label-encode categoricals (with fallback for unseen labels)
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    # Align to training feature order; pad any missing column with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]


# ---------------------------------------------------------------------------
# Risk prediction
# ---------------------------------------------------------------------------
def predict_risk(data: dict) -> tuple[float, str]:
    """Return (probability_of_noshow, risk_level_label)."""
    model, encoders, feature_cols = load_artifacts()
    df_processed = preprocess_input(data, encoders, feature_cols)

    prob = float(model.predict_proba(df_processed)[0][1])

    if prob < THRESHOLD_VERY_LOW:
        risk_level = "Very Low"
    elif prob < THRESHOLD_LOW:
        risk_level = "Low"
    elif prob < THRESHOLD_MODERATE:
        risk_level = "Moderate"
    elif prob < THRESHOLD_HIGH:
        risk_level = "High"
    else:
        risk_level = "Very High"

    return prob, risk_level


# ---------------------------------------------------------------------------
# SHAP explanation — lightweight result wrapper
# ---------------------------------------------------------------------------
class _SHAPResult:
    """
    Minimal container matching the .values / .feature_names API that
    the Book Appointment page expects from a shap.Explanation object.
    """
    __slots__ = ("values", "feature_names")

    def __init__(self, values, feature_names):
        self.values        = values
        self.feature_names = feature_names


def get_shap_values(data: dict):
    """
    Return a _SHAPResult for the single appointment row, or None.

    Uses XGBoost's built-in pred_contribs (exact tree SHAP, log-odds space).
    We load xgboost_base_model.pkl (the raw XGBoost before Platt calibration)
    because CalibratedClassifierCV does not expose a get_booster() method.
    The calibration layer only rescales the final probability — it does not
    change which features drive the model's decision, so SHAP values from
    the base model remain fully interpretable.
    """
    import xgboost as xgb

    _, encoders, feature_cols = load_artifacts()
    df_processed = preprocess_input(data, encoders, feature_cols)

    # Load the raw base model for SHAP (calibrated wrapper has no booster)
    base_model_path = os.path.join(MODELS_DIR, "xgboost_base_model.pkl")
    if os.path.exists(base_model_path):
        shap_model = joblib.load(base_model_path)
    else:
        # Graceful fallback: try the main model (works if it's uncalibrated)
        shap_model, _, _ = load_artifacts()

    try:
        booster  = shap_model.get_booster()
        dm       = xgb.DMatrix(df_processed, feature_names=list(feature_cols))
        # pred_contribs: shape (n_samples, n_features + 1); last col = bias
        contribs  = booster.predict(dm, pred_contribs=True)
        shap_vals = contribs[0, :-1]   # drop bias, keep per-feature values
        return _SHAPResult(values=shap_vals, feature_names=list(feature_cols))

    except Exception as e:
        print(f"SHAP calculation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Database write
# ---------------------------------------------------------------------------
def save_appointment(data: dict, prob: float, risk_level: str) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO appointments (
            patient_id, appointment_date, booking_date, sex, age, language, zipcode,
            appt_type_code, medical_service_code, facility_code, nurse_unit_code,
            service_line_used, attending_physician, referring_physician,
            primary_plan_type, predicted_risk_prob, risk_level, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data["patient_id"],
            str(data["appointment_date"]),
            str(data["booking_date"]),
            data["sex"], data["age"], data["language"], data["zipcode"],
            data["appt_type_code"], data["medical_service_code"],
            data["facility_code"], data["nurse_unit_code"],
            data["service_line_used"], data["attending_physician"],
            data["referring_physician"], data["primary_plan_type"],
            float(prob), risk_level,
            data.get("status", "Scheduled"),
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Database read
# ---------------------------------------------------------------------------
def get_all_appointments() -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM appointments", conn)
    conn.close()
    if "predicted_risk_prob" in df.columns:
        df["predicted_risk_prob"] = pd.to_numeric(df["predicted_risk_prob"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Appointment status simulation (end-of-day refresh)
# ---------------------------------------------------------------------------
def update_appointment_status() -> int:
    """
    Mark past 'Scheduled' appointments as Completed (90%) or No-Show (10%).
    Returns the count of records updated.
    """
    import random

    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now()

    cursor.execute(
        "SELECT id, appointment_date FROM appointments WHERE status = 'Scheduled'"
    )
    appointments = cursor.fetchall()
    updated = 0

    for appt in appointments:
        try:
            appt_date = pd.to_datetime(appt["appointment_date"], format="mixed")
            if appt_date < now:
                new_status = "No-Show" if random.random() < 0.10 else "Completed"
                cursor.execute(
                    "UPDATE appointments SET status = ? WHERE id = ?",
                    (new_status, appt["id"]),
                )
                updated += 1
        except Exception as e:
            print(f"Error processing appointment {appt['id']}: {e}")

    conn.commit()
    conn.close()
    return updated


# ---------------------------------------------------------------------------
# Streamlit sidebar helper
# ---------------------------------------------------------------------------
def render_refresh_button():
    import streamlit as st

    if st.sidebar.button("🔄 Refresh Data (Simulate Day)"):
        count = update_appointment_status()
        if count > 0:
            st.sidebar.success(f"Updated {count} records!")
        else:
            st.sidebar.info("Data is up to date.")
        st.rerun()
