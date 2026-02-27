# Healthcare Appointment No-Show Risk Intelligence

> **ML-powered no-show prediction for outpatient clinics — XGBoost + Platt Calibration · ROC-AUC 0.712 · Brier Skill Score +0.112**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-healthcare-appointment-scheduling.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1-orange)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A full-stack clinical decision-support dashboard built with Python, Streamlit, and XGBoost. It predicts the probability that a booked appointment will result in a no-show or cancellation, explains the prediction with SHAP, and surfaces actionable risk tiers for clinical staff — all backed by a live SQLite database.

---

## Screenshots

| Home Dashboard | Book Appointment |
|---|---|
| ![Home](assets/home_dashboard.png) | ![Book](assets/book_appointment.png) |

| Overall Trends | Physician Panel |
|---|---|
| ![Trends](assets/overall_view.png) | ![Physician](assets/physician_view.png) |

| Model Performance | My Appointments |
|---|---|
| ![Model](assets/model_performance.png) | ![Appointments](assets/my_appointments.png) |

---

## Features

| Capability | Detail |
|---|---|
| **Risk Prediction** | Calibrated XGBoost model outputs a 0–100% no-show probability per appointment |
| **SHAP Explainability** | Waterfall chart shows which features push risk up or down for each patient |
| **5-Tier Action System** | Very Low / Low / Moderate / High / Very High — each tier maps to a clinical workflow |
| **Live Dashboard** | 6 Streamlit pages covering individual bookings, fleet-wide trends, and physician panels |
| **Calibrated Probabilities** | Platt scaling corrects `scale_pos_weight` overconfidence — predicted 30% ≈ actual 30% NS |
| **Engineered Features** | Patient & physician historical no-show rates (leakage-free via `shift(1).cumsum()`) |
| **Synthetic Data Mode** | Populate the DB with 700 realistic historical appointments in one command |

---

## Dashboard Pages

### 🏥 Home — Risk Intelligence Overview
Live KPI cards (total appointments, high-risk count, avg risk, observed no-show rate), risk distribution donut chart, upcoming high-risk scheduled appointments, and stacked status bar chart.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Total Appts   High/VH Risk   Avg Risk Prob   Observed NS Rate      │
│    700            183           21.4%             28.2%             │
│                                                                     │
│  [Risk Distribution Donut]     [Upcoming High Risk Table]           │
│   Very Low 31%                  Patient   Date      Risk   Prob     │
│   Low      24%                  P84231   Mar 2     High   0.38      │
│   Moderate 22%                  P10923   Mar 3   V.High   0.52      │
│   High     14%                  ...                                  │
│   V.High    9%                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 📅 Book Appointment — Predict & Book
Enter patient demographics and appointment details → model predicts in real-time → risk gauge + SHAP waterfall explanation → confirm and save to DB.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Patient Info │ Appointment Details │ Medical Context               │
│                                                                     │
│              [ 🔮 Predict Risk ]                                    │
│                                                                     │
│         ┌──────── Gauge ────────┐                                   │
│         │      38.2%  🔴        │   Risk Level: HIGH               │
│         │   No-Show Prob        │   ⚠️ Action Required             │
│         └──────────────────────┘   Manual outreach + double-book   │
│                                                                     │
│  Top 5 Risk Factors (SHAP):                                         │
│  • PATIENT_NOSHOW_RATE (45%): red[Increases Risk] (+0.31)          │
│  • LEAD_TIME_DAYS (18): red[Increases Risk] (+0.18)                │
│  • APPT_TYPE_CODE (NP): green[Decreases Risk] (-0.09)              │
└─────────────────────────────────────────────────────────────────────┘
```

### 📊 Overall View — Fleet-Wide Trends
Date-range filter → KPI row → 7-day rolling risk trend (dual-axis: volume bars + risk line) → interactive calendar heatmap (4 metrics) → risk vs. lead-time bubble chart → risk distribution donut + gender box plots → recent appointments table.

### 👨‍⚕️ Physician View — Risk Panel Analysis
Scatter quadrant (risk rate vs. patient volume, bubble = panel size, quadrant lines at median) → flagged physicians callout → full ranked table with progress bars → individual physician drilldown (KPIs + risk donut + appointment history).

### 📈 Model Performance — 4-Tab Deep Dive

| Tab | Contents |
|---|---|
| Probability & ROC | Calibrated probability histogram with threshold overlays · ROC curve (AUC = 0.712) |
| Calibration | Before/after Platt scaling curves · Brier score comparison table |
| Precision-Recall | PR curve (AUC = 0.438) · F1 vs threshold chart |
| Features & Thresholds | Top-20 feature importance (XGBoost gain) · 5-tier risk card grid |

### 📋 My Appointments — Search & Filter
Patient ID search · risk/physician/status/date filters · 48-hour urgent-attention banner for High/VH risk · sortable table · CSV export.

---

## ML Pipeline

```
Raw CSVs (data/)
     │
     ▼
Feature Engineering
  • Temporal: LEAD_TIME_DAYS, APPT_DOW (ISO 1-7), APPT_HOUR, IS_WEEKEND, SEASON_BUCKET
  • Historical: PATIENT_NOSHOW_RATE, PHYSICIAN_NOSHOW_RATE (shift+cumsum, no leakage)
  • Categorical: LabelEncoded (SEX, LANGUAGE, ZIPCODE3, APPT_TYPE_CODE, …)
     │
     ▼
3-Way Split: 64% train / 16% calibration / 20% test
     │
     ├─► Optuna TPE (50 trials, StratifiedKFold) → best XGBoost hyperparams
     │
     ├─► Base XGBoost fit on train set
     │         └─► saved as xgboost_base_model.pkl  (used for SHAP)
     │
     └─► CalibratedClassifierCV (Platt scaling) fit on calibration set
               └─► saved as xgboost_model.pkl       (used for predictions)
```

### Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | **0.7120** |
| PR-AUC | 0.4376 |
| Brier Score (calibrated) | 0.1477 |
| Brier Skill Score | **+0.1125** (was −0.2647 pre-calibration) |
| No-Show Base Rate | 21.1% |

### Risk Tiers

| Tier | Probability | Actual NS Rate | Action |
|---|---|---|---|
| Very Low | < 10% | ~7% | No action |
| Low | 10–15% | ~13% | No action |
| Moderate | 15–25% | ~19% | Automated SMS/email |
| High | 25–40% | ~30% | Manual phone call |
| Very High | ≥ 40% | ~51% | Double-book + outreach |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML Model** | XGBoost 3.1 + Scikit-learn CalibratedClassifierCV |
| **Hyperparameter Tuning** | Optuna TPE (50 trials, StratifiedKFold CV) |
| **Explainability** | SHAP (`pred_contribs` — native tree SHAP) |
| **Frontend** | Streamlit 1.51 |
| **Visualisation** | Plotly Express + Graph Objects |
| **Database** | SQLite (via Python `sqlite3`) |
| **Data Processing** | Pandas 2.3 · NumPy 2.2 |
| **Artifact Storage** | Joblib pickle (`.pkl`) |

---

## Project Structure

```
AI-Healthcare-Appointment-Scheduling/
│
├── app.py                          # Streamlit landing page (Home dashboard)
├── utils.py                        # Core ML utilities, DB helpers, design system
├── requirements.txt
├── README.md
├── .gitignore
│
├── pages/                          # Streamlit multi-page app
│   ├── 1_How_to_Use.py             # System Overview (pipeline, metrics, tiers)
│   ├── 2_Book_Appointment.py       # Predict + SHAP + book
│   ├── 3_Overall_View.py           # Fleet-wide trend dashboard
│   ├── 4_Physician_View.py         # Physician panel risk analysis
│   ├── 5_Model_Performance.py      # Model evaluation (4 tabs)
│   └── 6_My_Appointments.py        # Search, filter, export
│
├── models/                         # Trained model artifacts
│   ├── xgboost_model.pkl           # Platt-calibrated model (predictions)
│   ├── xgboost_base_model.pkl      # Raw XGBoost (SHAP explanations)
│   ├── label_encoders.pkl          # Fitted LabelEncoders per categorical
│   ├── feature_columns.pkl         # Ordered feature list for inference
│   ├── model_metrics.pkl           # Evaluation metrics + test arrays
│   └── background_data.pkl         # SHAP background dataset
│
├── data/                           # Raw training data (CSV parts a–d)
│   ├── Data_part_a.csv
│   ├── Data_part_b.csv
│   ├── Data_part_c.csv
│   └── Data_part_d.csv
│
├── scripts/                        # Utility scripts (run from project root)
│   ├── extract_model.py            # Train model + export all artifacts
│   ├── generate_synthetic_data.py  # Populate DB with 700 synthetic appointments
│   └── setup_db.py                 # Initialise empty appointments table
│
└── notebooks/                      # Analysis & exploration
    ├── data_exploration_and_thresholds.ipynb
    └── DSO_568_Healthcare_Analytics_Final_Project_Python.ipynb
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/NandeeshHS/AI-Healthcare-Appointment-Scheduling.git
cd AI-Healthcare-Appointment-Scheduling
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the database with synthetic appointments

```bash
python scripts/generate_synthetic_data.py
```

This clears any existing DB and populates it with 700 synthetic appointments spanning the last 6 months, with realistic status resolution based on predicted risk.

### 5. Run the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deploy to Streamlit Community Cloud (free)

> The app auto-seeds 700 synthetic appointments on first launch — no manual DB setup required.

1. **Fork or push** this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repository → set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit installs dependencies from `requirements.txt` automatically.
5. Once live, copy the public URL and update the badge at the top of this README:

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_APP_URL)
```

> **Note:** The SQLite database (`healthcare.db`) is gitignored. On every fresh deploy or
> container restart, `startup.py` recreates it automatically with reproducible synthetic data
> (seeded with `random.seed(42)`).

---

## Retraining the Model

If you update the training data or want to retune hyperparameters:

```bash
python scripts/extract_model.py
```

This runs a 50-trial Optuna search, fits Platt calibration, evaluates on a held-out test set, and saves all artifacts to `models/`. After retraining, update the risk thresholds in `utils.py` if needed and re-run `generate_synthetic_data.py`.

---

## Key Design Decisions

**Why Platt scaling?**
XGBoost with `scale_pos_weight=3.74` (for 21% minority class) produces systematically overconfident scores — a raw score of 0.55 corresponded to only ~22% actual no-shows (Brier Skill Score: −0.265). Platt scaling on a held-out 16% calibration set corrects this: predicted probabilities now track actual rates within ±3% per decile (BSS: +0.112).

**Why two model files?**
`xgboost_model.pkl` is the Platt-calibrated `CalibratedClassifierCV` wrapper used for predictions. It doesn't expose a `.get_booster()` method, so `xgboost_base_model.pkl` (the raw XGBoost) is loaded separately for SHAP `pred_contribs`. Calibration only rescales probabilities — it doesn't change which features drive the model.

**Leakage-free historical features**
`PATIENT_NOSHOW_RATE` and `PHYSICIAN_NOSHOW_RATE` are computed with `shift(1).cumsum()` so each row's rate is based on history *before* that appointment. At inference time, rates are live-queried from the SQLite DB.

**ISO isoweekday (1–7) for day-of-week**
The training CSV stores DOW on a 1-7 ISO scale (Monday=1). Inference uses `isoweekday()` (not `dayofweek()` which is 0-6) to maintain consistent encoding.

---

## License

MIT — see [LICENSE](LICENSE) for details.
