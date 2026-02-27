"""
startup.py
==========
Auto-seeds healthcare.db on first run (e.g. Streamlit Community Cloud
where the gitignored database file does not exist yet).

Called from app.py:
    if not os.path.exists(os.path.join(_APP_DIR, "healthcare.db")):
        from startup import seed_database
        seed_database()
"""

import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np

# Project root = directory containing this file
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
DB_PATH    = os.path.join(_PROJECT_ROOT, "healthcare.db")

N_APPTS        = 700
N_PATIENTS     = 200
FUTURE_DAYS    = 14
START_DAYS_BACK = 183   # ~6 months

# Calibrated no-show rates per risk tier (from held-out test set analysis)
NS_RATES = {
    "Very Low":  0.07,
    "Low":       0.13,
    "Moderate":  0.19,
    "High":      0.30,
    "Very High": 0.51,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id           TEXT,
            appointment_date     TEXT,
            booking_date         TEXT,
            sex                  TEXT,
            age                  INTEGER,
            language             TEXT,
            zipcode              TEXT,
            appt_type_code       TEXT,
            medical_service_code TEXT,
            facility_code        TEXT,
            nurse_unit_code      TEXT,
            service_line_used    TEXT,
            attending_physician  TEXT,
            referring_physician  TEXT,
            primary_plan_type    TEXT,
            predicted_risk_prob  REAL,
            risk_level           TEXT,
            status               TEXT DEFAULT 'Scheduled',
            created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def _random_appt_date(now: datetime) -> datetime:
    """Weekday-heavy random appointment datetime within the configured range."""
    while True:
        delta = random.randint(-START_DAYS_BACK, FUTURE_DAYS)
        d     = now + timedelta(days=delta)
        dow   = d.isoweekday()   # 1=Mon … 7=Sun
        if dow == 7:             # skip Sundays
            continue
        if dow == 6 and random.random() > 0.20:  # 20% Saturday acceptance
            continue
        hour   = random.choices(
            range(8, 18),
            weights=[2, 3, 4, 5, 5, 4, 3, 3, 2, 1],  # peak mid-morning
            k=1,
        )[0]
        minute = random.choice([0, 15, 30, 45])
        return d.replace(hour=hour, minute=minute, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def seed_database() -> None:
    """
    Create healthcare.db and populate it with 700 synthetic appointments.
    Safe to call multiple times — exits early if the DB already exists.
    """
    if os.path.exists(DB_PATH):
        return   # already initialised; nothing to do

    print("[startup] healthcare.db not found — seeding synthetic data …")
    random.seed(42)
    np.random.seed(42)

    # Create schema
    conn = sqlite3.connect(DB_PATH)
    _create_schema(conn)
    conn.close()

    # Load model artifacts
    try:
        encoders     = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
        feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))  # noqa: F841
    except FileNotFoundError as exc:
        print(f"[startup] Could not load model artifacts: {exc}")
        print("[startup] Database schema created but left empty.")
        return

    def enc(col):
        return list(encoders[col].classes_)

    # Import runtime helpers from utils (relative path works since project root is on sys.path)
    from utils import predict_risk, save_appointment

    # Build a stable patient pool for realistic repeat-visit features
    patient_pool = list(dict.fromkeys(
        f"P{random.randint(10000, 99999):05d}" for _ in range(N_PATIENTS + 20)
    ))[:N_PATIENTS]

    # Language distribution: ~80% English
    lang_classes  = enc("LANGUAGE")
    other_langs   = [l for l in lang_classes if l != "English"]
    lang_weights  = ([80] + [20 / max(len(other_langs), 1)] * len(other_langs)
                     if "English" in lang_classes else None)

    now    = datetime.now()
    errors = 0

    for i in range(N_APPTS):
        appt_dt   = _random_appt_date(now)
        lead_days = random.randint(1, 21)
        book_dt   = appt_dt - timedelta(days=lead_days)
        if book_dt < (now - timedelta(days=START_DAYS_BACK + 30)):
            book_dt = appt_dt - timedelta(days=random.randint(1, 7))

        lang = (
            random.choices(["English"] + other_langs, weights=lang_weights, k=1)[0]
            if lang_weights
            else random.choice(lang_classes)
        )

        data = {
            "patient_id":           random.choice(patient_pool),
            "sex":                  random.choice(enc("SEX")),
            "age":                  random.randint(18, 85),
            "language":             lang,
            "zipcode":              random.choice(enc("ZIPCODE3")),
            "primary_plan_type":    random.choice(enc("PRIMARY_PLAN_TYPE")),
            "booking_date":         book_dt,
            "appointment_date":     appt_dt,
            "appt_type_code":       random.choice(enc("APPT_TYPE_CODE")),
            "medical_service_code": random.choice(enc("MEDICAL_SERVICE_CODE")),
            "facility_code":        random.choice(enc("FACILITY_CODE")),
            "nurse_unit_code":      random.choice(enc("NURSE_UNIT_CODE")),
            "service_line_used":    random.choice(enc("SERVICE_LINE_USED")),
            "attending_physician":  random.choice(enc("ATTENDING_PHYSICIAN")),
            "referring_physician":  random.choice(enc("REFERRING_PHYSICIAN")),
        }

        try:
            prob, risk_level = predict_risk(data)
            is_past = appt_dt < now
            status  = (
                ("No-Show" if random.random() < NS_RATES[risk_level] else "Completed")
                if is_past else "Scheduled"
            )
            data["status"] = status
            save_appointment(data, prob, risk_level)
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"[startup]   Warning — record {i + 1}: {exc}")

    print(f"[startup] Done: {N_APPTS - errors} records written, {errors} errors.")
