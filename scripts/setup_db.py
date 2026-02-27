import sqlite3
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(_PROJECT_ROOT, 'healthcare.db')

def setup_database():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"Removed existing database {DB_NAME}")
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create appointments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        appointment_date TEXT,
        booking_date TEXT,
        sex TEXT,
        age INTEGER,
        language TEXT,
        zipcode TEXT,
        appt_type_code TEXT,
        medical_service_code TEXT,
        facility_code TEXT,
        nurse_unit_code TEXT,
        service_line_used TEXT,
        attending_physician TEXT,
        referring_physician TEXT,
        primary_plan_type TEXT,
        predicted_risk_prob REAL,
        risk_level TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database {DB_NAME} created successfully with 'appointments' table.")

if __name__ == "__main__":
    setup_database()
