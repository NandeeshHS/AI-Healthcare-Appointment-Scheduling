# Healthcare Appointment No-Show Risk Intelligence

> **ML-powered no-show prediction for outpatient clinics — XGBoost + Platt Calibration · ROC-AUC 0.712 · Brier Skill Score +0.112**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-healthcare-appointment-scheduling-etmkmsyzrrcyduuqwfrhji.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1-orange)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A full-stack clinical decision-support dashboard built with Python, Streamlit, and XGBoost. It predicts the probability that a booked appointment will result in a no-show or cancellation, explains the prediction with SHAP, and surfaces actionable risk tiers for clinical staff — all backed by a live SQLite database.

**[Live Demo](https://ai-healthcare-appointment-scheduling-etmkmsyzrrcyduuqwfrhji.streamlit.app/)**

---

## Table of Contents

- [Screenshots](#screenshots)
- [Features](#features)
- [Dashboard Pages](#dashboard-pages)
- [ML Pipeline](#ml-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Deploy to Streamlit Cloud](#deploy-to-streamlit-community-cloud-free)
- [Retraining the Model](#retraining-the-model)
- [Key Design Decisions](#key-design-decisions)

---

## Screenshots

### Home — Risk Intelligence Dashboard
![Home Dashboard](assets/home_dashboard.png)
*Live KPI cards, risk distribution donut chart, upcoming high-risk appointment table, and stacked status overview — refreshed from the live SQLite database.*

### System Overview
![System Overview](assets/system_overview.png)
*End-to-end clinical workflow pipeline, live model metrics, top-10 feature importance chart, risk tier action guide cards, and full tech stack breakdown.*

### Book Appointment — Predict & SHAP Explain
![Book Appointment](assets/book_appointment.png)
*Three-section form (patient demographics, appointment details, medical context) feeds the real-time XGBoost + Platt Calibration model. Returns a risk gauge, probability, and SHAP waterfall explanation.*

### Overall View — Fleet-Wide Trends
![Overall View](assets/overall_view.png)
*Date-range filter, 5 KPI cards, 7-day rolling risk trend (dual-axis: volume bars + risk line), and interactive appointment calendar heatmap by day of week.*

### Physician Risk View
![Physician View](assets/physician_view.png)
*Scatter quadrant (risk rate vs. patient volume) with median dividers, flagged high-volume/high-risk physician callout, and a ranked table with progress bars.*

### Model Performance & Insights
![Model Performance](assets/model_performance.png)
*Header metrics (ROC-AUC 0.712, PR-AUC 0.438, Brier Score 0.148, BSS +0.112) with four deep-dive tabs: Probability & ROC · Calibration · Precision-Recall · Features & Thresholds.*

### My Appointments — Search, Filter & Export
![My Appointments](assets/my_appointments.png)
*48-hour urgent-attention banner for imminent High/Very High risk appointments, sidebar filters, KPI row with inline CSV export, and full sortable appointment table.*

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
| **Auto-Seeded Demo Data** | On first run (e.g. Streamlit Cloud), 700 realistic appointments generated automatically |

---

## Dashboard Pages

### Home — Risk Intelligence Overview
Live KPI cards (total appointments, high/very-high risk count, avg risk probability, observed no-show rate), a color-coded risk distribution donut chart, an upcoming high-risk appointment table sorted by date, and a stacked appointment status bar chart.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Total Appts   High/VH Risk   Avg Risk Prob   Observed NS Rate      │
│    700            369           28.5%             27.8%             │
│                                                                     │
│  [Risk Distribution Donut]     [Upcoming High Risk Table]           │
│   Very Low  5%                  Patient   Date        Risk   Prob   │
│   Low       13%                 P24371   Feb 28     V.High   0.54   │
│   Moderate  25%                 P59735   Mar 2       High   0.37    │
│   High      31%                 P93320   Mar 2     V.High   0.57    │
│   V.High    22%                 ...                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### System Overview (How to Use)
Five-step clinical pipeline diagram, model metrics cards (ROC-AUC, PR-AUC, Brier Skill Score, base rate), top-10 feature importance bar chart, color-coded risk tier action cards with recommended interventions, and tech stack reference.

### Book Appointment — Predict & Book
Enter patient demographics and appointment details → model predicts in real-time → risk gauge + SHAP waterfall explanation → confirm and save to DB.

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Patient Info   2. Appointment Details   3. Medical Context       │
│                                                                     │
│              [ Predict Risk ]                                       │
│                                                                     │
│         ┌──────── Gauge ────────┐                                   │
│         │      38.2%  HIGH      │   Risk Level: HIGH               │
│         │   No-Show Prob        │   Manual outreach recommended     │
│         └──────────────────────┘                                   │
│                                                                     │
│  Top 5 Risk Factors (SHAP):                                         │
│  • PATIENT_NOSHOW_RATE (+0.31)  Increases Risk                      │
│  • LEAD_TIME_DAYS (+0.18)       Increases Risk                      │
│  • APPT_TYPE_CODE (-0.09)       Decreases Risk                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Overall View — Fleet-Wide Trends
Date-range filter → 5 KPI cards → 7-day rolling risk trend (dual-axis: volume bars + risk line) → interactive calendar heatmap (4 metrics: count / avg risk / no-show count / no-show rate) → recent appointments table.

### Physician View — Risk Panel Analysis
Scatter quadrant (risk rate vs. patient volume, bubble = panel size, quadrant lines at median values) → flagged physicians callout (above-median volume AND risk rate) → full ranked table with `ProgressColumn` for risk rate → individual physician drilldown via sidebar selector.

### Model Performance — 4-Tab Deep Dive

| Tab | Contents |
|---|---|
| Probability & ROC | Calibrated probability histogram with threshold overlays · ROC curve (AUC = 0.712) |
| Calibration | Before/after Platt scaling curves · Brier score comparison table |
| Precision-Recall | PR curve (AUC = 0.438) · F1 vs threshold chart |
| Features & Thresholds | Top-20 feature importance (XGBoost gain) · 5-tier risk card grid |

### My Appointments — Search & Filter
Patient ID search · risk/physician/status/date sidebar filters · 48-hour urgent-attention banner for High/VH risk · sortable full-width table · inline CSV export.

---

## ML Pipeline

```
Raw CSVs (data/)
     │
     ▼
Feature Engineering
  • Temporal: LEAD_TIME_DAYS, APPT_DOW (ISO 1-7), APPT_HOUR, IS_WEEKEND, SEASON_BUCKET
  • Historical: PATIENT_NOSHOW_RATE, PHYSICIAN_NOSHOW_RATE (shift+cumsum, no leakage)
  • Categorical: LabelEncoded (SEX, LANGUAGE, ZIPCODE3, APPT_TYPE_CODE, ...)
     │
     ▼
3-Way Split: 64% train / 16% calibration / 20% test
     │
     ├── Optuna TPE (50 trials, StratifiedKFold) → best XGBoost hyperparams
     │
     ├── Base XGBoost fit on train set
     │         └── saved as xgboost_base_model.pkl  (used for SHAP)
     │
     └── CalibratedClassifierCV (Platt scaling) fit on calibration set
               └── saved as xgboost_model.pkl       (used for predictions)
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
| Moderate | 15–25% | ~19% | Automated SMS / email reminder |
| High | 25–40% | ~30% | Manual phone call |
| Very High | ≥ 40% | ~51% | Double-book + personal outreach |

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
├── startup.py                      # Auto-seeds DB on first run (cloud deploy)
├── utils.py                        # Core ML utilities, DB helpers, design system
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml                 # Healthcare-blue dark theme
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
├── assets/                         # Dashboard screenshots (README images)
│   ├── home_dashboard.png
│   ├── system_overview.png
│   ├── book_appointment.png
│   ├── overall_view.png
│   ├── physician_view.png
│   ├── model_performance.png
│   └── my_appointments.png
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

> **Auto-seeded:** `startup.py` generates 700 synthetic appointments on first launch — no manual DB setup required.

1. **Push** this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repository → set **Main file path** to `app.py`.
4. Click **Deploy**. Dependencies from `requirements.txt` install automatically.
5. Your app goes live at a public URL within a few minutes.

> **Note:** `healthcare.db` is gitignored (runtime state). On every fresh deploy or container restart, `startup.py` recreates it automatically with reproducible synthetic data (`random.seed(42)`).

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
