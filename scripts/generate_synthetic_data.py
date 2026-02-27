"""
generate_synthetic_data.py
==========================
Clears the appointments DB and populates it with realistic synthetic data
spanning the last 6 months, using the trained XGBoost + Platt model for
risk predictions.

Usage:
    python generate_synthetic_data.py

Approach:
  - 700 appointments distributed across the last ~6 months (plus ~2 weeks future)
  - ~200 recurring patient IDs so APPT_COUNT_BY_PATIENT / PATIENT_NOSHOW_RATE
    features carry meaningful signal
  - Past appointments resolved with risk-aligned no-show rates:
      Very Low  ~7%  | Low ~13% | Moderate ~19% | High ~30% | Very High ~51%
  - Future appointments (up to +14 days) stay as "Scheduled"
  - Weekly appointment volume follows a weekday-heavier distribution
    (more Mon-Fri, fewer Sat, none Sun)
"""

import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np

# Compute project root from script location so it works from any CWD
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
DB_PATH    = os.path.join(_PROJECT_ROOT, "healthcare.db")
N_APPTS    = 700           # total synthetic appointments
N_PATIENTS = 200           # unique patient pool (repeat visits occur naturally)
FUTURE_DAYS = 14           # appointments up to this many days ahead are "Scheduled"
START_DAYS_BACK = 183      # ~6 months ago

# No-show rates per risk level (from calibrated model decile analysis)
NS_RATES = {
    "Very Low":  0.07,
    "Low":       0.13,
    "Moderate":  0.19,
    "High":      0.30,
    "Very High": 0.51,
}

# ── Load encoder classes ──────────────────────────────────────────────────────
print("Loading model artifacts …")
encoders     = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
model        = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))

def encoder_classes(col):
    return list(encoders[col].classes_)

# ── Patient pool (stable IDs to allow repeat-visit features) ──────────────────
patient_pool = [f"P{random.randint(10000, 99999):05d}" for _ in range(N_PATIENTS)]
# Deduplicate (rare collision)
patient_pool = list(dict.fromkeys(patient_pool))

# ── Helper: predict risk with utils ──────────────────────────────────────────
from utils import predict_risk, save_appointment

# ── DB setup ─────────────────────────────────────────────────────────────────
def reset_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Deleted {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE appointments (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id          TEXT,
            appointment_date    TEXT,
            booking_date        TEXT,
            sex                 TEXT,
            age                 INTEGER,
            language            TEXT,
            zipcode             TEXT,
            appt_type_code      TEXT,
            medical_service_code TEXT,
            facility_code       TEXT,
            nurse_unit_code     TEXT,
            service_line_used   TEXT,
            attending_physician TEXT,
            referring_physician TEXT,
            primary_plan_type   TEXT,
            predicted_risk_prob REAL,
            risk_level          TEXT,
            status              TEXT DEFAULT 'Scheduled',
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("Fresh database initialised.")

# ── Date distribution: weekday-heavy, uniformly spread over range ─────────────
def random_appointment_date(now: datetime) -> datetime:
    """Return a random appointment datetime within [now - START_DAYS_BACK, now + FUTURE_DAYS].
    Skips Sundays; Saturday has 20% weight vs weekdays."""
    while True:
        delta = random.randint(-START_DAYS_BACK, FUTURE_DAYS)
        d = now + timedelta(days=delta)
        dow = d.isoweekday()   # 1=Mon … 7=Sun
        if dow == 7:           # skip Sunday
            continue
        if dow == 6 and random.random() > 0.20:  # only 20% of Saturdays
            continue
        hour  = random.choices(
            range(8, 18),
            weights=[2, 3, 4, 5, 5, 4, 3, 3, 2, 1],  # peak mid-morning
            k=1
        )[0]
        minute = random.choice([0, 15, 30, 45])
        return d.replace(hour=hour, minute=minute, second=0, microsecond=0)

# ── Main generation loop ──────────────────────────────────────────────────────
def generate():
    now = datetime.now()
    reset_db()

    print(f"Generating {N_APPTS} appointments …")
    errors = 0

    for i in range(N_APPTS):
        appt_dt    = random_appointment_date(now)
        lead_days  = random.randint(1, 21)
        book_dt    = appt_dt - timedelta(days=lead_days)
        if book_dt < (now - timedelta(days=START_DAYS_BACK + 30)):
            book_dt = appt_dt - timedelta(days=random.randint(1, 7))

        # Weighted language: ~80% English
        lang_classes = encoder_classes("LANGUAGE")
        if "English" in lang_classes:
            other_langs = [l for l in lang_classes if l != "English"]
            lang = random.choices(
                ["English"] + other_langs,
                weights=[80] + [20 / max(len(other_langs), 1)] * len(other_langs),
                k=1
            )[0]
        else:
            lang = random.choice(lang_classes)

        data = {
            "patient_id":           random.choice(patient_pool),
            "sex":                  random.choice(encoder_classes("SEX")),
            "age":                  random.randint(18, 85),
            "language":             lang,
            "zipcode":              random.choice(encoder_classes("ZIPCODE3")),
            "primary_plan_type":    random.choice(encoder_classes("PRIMARY_PLAN_TYPE")),
            "booking_date":         book_dt,
            "appointment_date":     appt_dt,
            "appt_type_code":       random.choice(encoder_classes("APPT_TYPE_CODE")),
            "medical_service_code": random.choice(encoder_classes("MEDICAL_SERVICE_CODE")),
            "facility_code":        random.choice(encoder_classes("FACILITY_CODE")),
            "nurse_unit_code":      random.choice(encoder_classes("NURSE_UNIT_CODE")),
            "service_line_used":    random.choice(encoder_classes("SERVICE_LINE_USED")),
            "attending_physician":  random.choice(encoder_classes("ATTENDING_PHYSICIAN")),
            "referring_physician":  random.choice(encoder_classes("REFERRING_PHYSICIAN")),
        }

        try:
            prob, risk_level = predict_risk(data)

            # Assign status: past appts resolved per calibrated no-show rates
            is_past = appt_dt < now
            if is_past:
                ns_rate = NS_RATES[risk_level]
                status  = "No-Show" if random.random() < ns_rate else "Completed"
            else:
                status = "Scheduled"

            data["status"] = status
            save_appointment(data, prob, risk_level)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{N_APPTS} done")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [!] Error on record {i + 1}: {e}")

    print(f"\nDone. {N_APPTS - errors} records written, {errors} errors.")
    _print_summary()

def _print_summary():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT status, COUNT(*) FROM appointments GROUP BY status").fetchall()
    conn.close()
    print("\nStatus breakdown:")
    for status, cnt in rows:
        print(f"  {status:<12} {cnt:>5}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    generate()
