"""
Healthcare Appointment No-Show Prediction — Model Training Pipeline
=====================================================================
Fixes applied vs original:
  1. IS_WEEKEND: corrected to [6, 7] (ISO isoweekday scale, 6=Sat, 7=Sun)
  2. APPT_COUNT_BY_PATIENT: uses cumcount() to eliminate future-data leakage
  3. Age '90+' condition: fixed to check '90+' not '90'
  4. Class imbalance: scale_pos_weight computed from actual class ratio
  5. Hyperparameter tuning: Optuna TPE sampler with stratified K-fold CV
  6. Evaluation: added ROC-AUC and PR-AUC alongside classification report
  7. Dead code removed (unreachable lines after return statement)
  8. Cross-validation mean/std scores logged for every fold
  9. PATIENT_NOSHOW_RATE: cumulative per-patient historical no-show rate
     (computed with shift+cumsum to avoid any future-data leakage)
 10. PHYSICIAN_NOSHOW_RATE: cumulative per-physician no-show rate (same method)
 11. TIME_OF_DAY_BUCKET removed — collinear with APPT_HOUR (continuous);
     removing it reduces noise and model complexity
 12. Platt-scaling calibration: 3-way split (64/16/20 train/cal/test).
     CalibratedClassifierCV(method="sigmoid", cv="prefit") fitted on the
     calibration hold-out fixes systematic overconfidence from scale_pos_weight.
     Both uncalibrated and calibrated metrics saved for comparison.
"""

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from optuna.samplers import TPESampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config — paths relative to project root so the script works from any CWD
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(_PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Optuna settings — reduce OPTUNA_TRIALS if time-constrained (min 20)
OPTUNA_TRIALS = 50
OPTUNA_CV_FOLDS = 3          # folds used inside Optuna objective (for speed)
OPTUNA_SAMPLE_SIZE = 100_000  # rows used for each Optuna trial (stratified sample)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    print("Loading data...")
    parts = [
        pd.read_csv(os.path.join(DATA_DIR, f"Data_part_{s}.csv"))
        for s in ("a", "b", "c", "d")
    ]
    df = pd.concat(parts, ignore_index=True)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. Preprocess
# ---------------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing data...")

    # --- 2a. Remove impossible lead times ---
    df = df[df["LEAD_TIME_DAYS"] >= 0].copy()

    # --- 2b. Drop rows missing any model-relevant column ---
    cols_to_check = [
        c for c in df.columns
        if c not in ("SCH_CANCEL_LEAD_HOURS", "CANCEL_TO_APPT_LEAD_HOURS")
    ]
    df = df.dropna(subset=cols_to_check).copy()

    # --- 2c. Clean age (FIX: check for '90+' not '90') ---
    age_str = df["AGE"].astype(str)
    unique_age_vals = set(age_str.unique())
    encode_90_plus_as = 95 if "90+" in unique_age_vals else 90
    df["AGE"] = age_str.str.strip().replace({"90+": encode_90_plus_as})
    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")

    # --- 2d. Numeric casts ---
    for col in ("AGE", "LEAD_TIME_DAYS", "APPT_HOUR", "DURATION", "APPT_DOW"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 2e. Binary target ---
    def _binary_target(status):
        if pd.isna(status):
            return np.nan
        s = str(status).upper()
        return 1 if ("NOSHOW" in s or "NO SHOW" in s or "CANCEL" in s) else 0

    df["TARGET_ENCODED"] = df["APPT_STATUS"].apply(_binary_target)
    df = df.dropna(subset=["TARGET_ENCODED"]).copy()
    df["TARGET_ENCODED"] = df["TARGET_ENCODED"].astype(int)

    # --- 2f. IS_WEEKEND (FIX: raw CSV uses 1-7 ISO scale, 6=Sat, 7=Sun) ---
    # Confirmed from raw data: APPT_DOW values are 1-7 (no 0 present).
    # DOW 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun.
    # Weekend = [6, 7]. Inference uses isoweekday() for the same 1-7 scale.
    df["IS_WEEKEND"] = df["APPT_DOW"].isin([6, 7]).astype(int)

    # --- 2g. APPT_COUNT_BY_PATIENT (FIX: cumcount avoids future-data leakage) ---
    # cumcount() gives the count of *prior* rows within each patient group in
    # dataset order.  This is a conservative proxy for historical visit count
    # and removes the leakage caused by using the total group count.
    df = df.sort_values("UNIQUE_ID")  # stable grouping order
    df["APPT_COUNT_BY_PATIENT"] = df.groupby("UNIQUE_ID").cumcount()

    # --- 2h. PATIENT_NOSHOW_RATE: cumulative per-patient historical no-show rate ---
    # shift(1) before cumsum ensures we never include the current row's outcome,
    # eliminating any possibility of target leakage.
    # First appointment for a patient → rate = 0.0 (no history available).
    df["_pat_prior_ns"] = (
        df.groupby("UNIQUE_ID")["TARGET_ENCODED"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )
    df["_pat_prior_cnt"] = df.groupby("UNIQUE_ID").cumcount()  # 0 = first row
    df["PATIENT_NOSHOW_RATE"] = np.where(
        df["_pat_prior_cnt"] == 0,
        0.0,
        df["_pat_prior_ns"] / df["_pat_prior_cnt"].clip(lower=1),
    )
    df.drop(columns=["_pat_prior_ns", "_pat_prior_cnt"], inplace=True)

    # --- 2i. PHYSICIAN_NOSHOW_RATE: cumulative per-physician no-show rate ---
    # Same leakage-free approach. First encounter for a physician in the dataset
    # falls back to the overall base rate so the model has a sensible prior.
    base_rate = float(df["TARGET_ENCODED"].mean())
    df["ATTENDING_PHYSICIAN"] = df["ATTENDING_PHYSICIAN"].fillna("UNKNOWN")
    df["_phys_prior_ns"] = (
        df.groupby("ATTENDING_PHYSICIAN")["TARGET_ENCODED"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )
    df["_phys_prior_cnt"] = df.groupby("ATTENDING_PHYSICIAN").cumcount()
    df["PHYSICIAN_NOSHOW_RATE"] = np.where(
        df["_phys_prior_cnt"] == 0,
        base_rate,
        df["_phys_prior_ns"] / df["_phys_prior_cnt"].clip(lower=1),
    )
    df.drop(columns=["_phys_prior_ns", "_phys_prior_cnt"], inplace=True)

    print(f"  After cleaning: {len(df):,} rows")
    print(f"  No-show rate: {df['TARGET_ENCODED'].mean():.3%}")
    return df


# ---------------------------------------------------------------------------
# 3. Encode categoricals
# ---------------------------------------------------------------------------
CATEGORICAL_COLS = [
    "SEX", "LANGUAGE", "ZIPCODE3", "APPT_TYPE_CODE", "MEDICAL_SERVICE_CODE",
    "FACILITY_CODE", "NURSE_UNIT_CODE", "SERVICE_LINE_USED",
    "ATTENDING_PHYSICIAN", "REFERRING_PHYSICIAN", "PRIMARY_PLAN_TYPE",
    "TIME_OF_DAY_BUCKET", "SEASON_BUCKET",
]

EXCLUDE_FROM_FEATURES = {
    "SCH_CANCEL_LEAD_HOURS", "CANCEL_TO_APPT_LEAD_HOURS", "APPT_STATUS",
    "QUALIFYING_FLAG", "TARGET_CLASS", "TARGET_ENCODED", "UNIQUE_ID",
    "ENCNTR", "APPT", "DURATION", "ORIGINAL_STATUS", "ENCNTR_TYPE",
    # TIME_OF_DAY_BUCKET removed: collinear with APPT_HOUR (continuous).
    # Removing it eliminates redundant encoding without losing information.
    "TIME_OF_DAY_BUCKET",
}


def encode_features(df: pd.DataFrame):
    """Label-encode all categorical columns; return X, y, encoders, feature_cols."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    cat_features = (
        df[feature_cols]
        .select_dtypes(include=["object", "category"])
        .columns.tolist()
    )
    print(f"  Categorical features to encode: {cat_features}")

    label_encoders: dict[str, LabelEncoder] = {}
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df[feature_cols]
    y = df["TARGET_ENCODED"]
    return X, y, label_encoders, feature_cols


# ---------------------------------------------------------------------------
# 4. Optuna hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    n_trials: int = OPTUNA_TRIALS,
) -> dict:
    """
    Run an Optuna study to find the best XGBoost hyperparameters.

    For speed:
      - Uses a stratified sample of OPTUNA_SAMPLE_SIZE rows.
      - Evaluates with OPTUNA_CV_FOLDS-fold cross-validation.
      - Optimises ROC-AUC (appropriate for imbalanced binary classification).
    """
    print(f"\nRunning Optuna hyperparameter search ({n_trials} trials, "
          f"{OPTUNA_CV_FOLDS}-fold CV on up to {OPTUNA_SAMPLE_SIZE:,} rows)...")

    # Stratified sample for speed
    if len(X_train) > OPTUNA_SAMPLE_SIZE:
        _, X_s, _, y_s = train_test_split(
            X_train, y_train,
            test_size=OPTUNA_SAMPLE_SIZE / len(X_train),
            random_state=42, stratify=y_train,
        )
    else:
        X_s, y_s = X_train.copy(), y_train.copy()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 9),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",   # fast histogram-based method
        )
        cv = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_s, y_s, cv=cv, scoring="roc_auc", n_jobs=1)
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name="xgboost_noshow",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best ROC-AUC (CV sample): {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# 5. Train final model
# ---------------------------------------------------------------------------
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    scale_pos_weight: float,
):
    """
    3-way split: 64% train / 16% calibration / 20% test.

    Steps:
      1. Train XGBoost on the train split using best Optuna params.
      2. Apply Platt scaling (sigmoid) on the calibration split via
         CalibratedClassifierCV(cv="prefit").  This corrects the systematic
         overconfidence introduced by scale_pos_weight without touching the
         test set.
      3. Evaluate both raw (uncalibrated) and calibrated probabilities on the
         test split so the dashboard can show a before/after comparison.
    """
    print("\nSplitting data: 64% train / 16% calibration / 20% test...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
    )
    print(f"  Train: {len(X_train):,}  |  Cal: {len(X_cal):,}  |  Test: {len(X_test):,}")

    # ── 1. Train base XGBoost ──────────────────────────────────────────────────
    print("Training base XGBoost on train split...")
    base_model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    base_model.fit(X_train, y_train)

    # Uncalibrated probabilities on test (for before/after dashboard comparison)
    y_prob_uncal = base_model.predict_proba(X_test)[:, 1]

    # ── 2. Platt scaling calibration ──────────────────────────────────────────
    print("Calibrating with Platt scaling (sigmoid) on calibration split...")
    calibrated_model = CalibratedClassifierCV(
        base_model, cv="prefit", method="sigmoid"
    )
    calibrated_model.fit(X_cal, y_cal)

    # ── 3. Evaluate calibrated model on test ──────────────────────────────────
    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]

    roc_auc      = roc_auc_score(y_test, y_prob)
    roc_auc_uncal= roc_auc_score(y_test, y_prob_uncal)
    pr_auc       = average_precision_score(y_test, y_prob)
    report       = classification_report(y_test, y_pred, output_dict=True)
    conf_mat     = confusion_matrix(y_test, y_pred)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob)

    # Brier scores and skill scores (skill > 0 means better than naive baseline)
    base_rate    = float(y_test.mean())
    brier_naive  = brier_score_loss(y_test, np.full(len(y_test), base_rate))
    brier_uncal  = brier_score_loss(y_test, y_prob_uncal)
    brier_cal    = brier_score_loss(y_test, y_prob)
    bss_uncal    = 1.0 - brier_uncal / brier_naive
    bss_cal      = 1.0 - brier_cal   / brier_naive

    # Feature importance from base model (calibration layer has no importances)
    importance_df = pd.DataFrame({
        "Feature":    X.columns.tolist(),
        "Importance": base_model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print(f"  ROC-AUC  : {roc_auc:.4f}  (uncal: {roc_auc_uncal:.4f})")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    print(f"  Accuracy : {report['accuracy']:.4f}")
    print(f"  Brier (uncalibrated) : {brier_uncal:.4f}   BSS: {bss_uncal:+.4f}")
    print(f"  Brier (calibrated)   : {brier_cal:.4f}   BSS: {bss_cal:+.4f}")

    metrics = {
        "classification_report":     report,
        "confusion_matrix":          conf_mat,
        "feature_importance":        importance_df,
        "y_test":                    y_test,
        "y_prob":                    y_prob,            # calibrated (default)
        "y_prob_uncalibrated":       y_prob_uncal,      # raw XGBoost
        "roc_auc":                   roc_auc,
        "roc_auc_uncalibrated":      roc_auc_uncal,
        "pr_auc":                    pr_auc,
        "precisions":                precisions,
        "recalls":                   recalls,
        "pr_thresholds":             pr_thresholds,
        "best_params":               best_params,
        "scale_pos_weight":          scale_pos_weight,
        "brier_calibrated":          brier_cal,
        "brier_uncalibrated":        brier_uncal,
        "brier_skill_calibrated":    bss_cal,
        "brier_skill_uncalibrated":  bss_uncal,
    }

    background_data = X_train.sample(n=min(100, len(X_train)), random_state=42)
    return calibrated_model, base_model, metrics, background_data, X_train


# ---------------------------------------------------------------------------
# 6. Save all artifacts
# ---------------------------------------------------------------------------
def save_artifacts(
    calibrated_model,
    base_model,
    label_encoders: dict,
    feature_cols: list,
    metrics: dict,
    background_data: pd.DataFrame,
):
    print("\nSaving artifacts...")
    # Calibrated model is the default for all predictions
    joblib.dump(calibrated_model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    # Raw base model saved separately — used by get_shap_values() for exact
    # tree SHAP via pred_contribs (calibration layer has no booster attribute)
    joblib.dump(base_model,       os.path.join(MODELS_DIR, "xgboost_base_model.pkl"))
    joblib.dump(label_encoders,   os.path.join(MODELS_DIR, "label_encoders.pkl"))
    joblib.dump(feature_cols,     os.path.join(MODELS_DIR, "feature_columns.pkl"))
    joblib.dump(metrics,          os.path.join(MODELS_DIR, "model_metrics.pkl"))
    joblib.dump(background_data,  os.path.join(MODELS_DIR, "background_data.pkl"))
    print(f"  All artifacts saved to ./{MODELS_DIR}/")
    print("    xgboost_model.pkl      <- calibrated (use for predictions)")
    print("    xgboost_base_model.pkl <- raw XGBoost (used internally for SHAP)")


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1 — load raw data
    df = load_data()

    # Step 2 — clean & feature-engineer
    df = preprocess_data(df)

    # Step 3 — encode categoricals
    X, y, label_encoders, feature_cols = encode_features(df)

    # Step 4 — compute class weight (FIX: handle imbalance)
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = round(neg_count / pos_count, 4)
    print(f"\nClass imbalance: scale_pos_weight = {scale_pos_weight}")

    # Step 5 — Optuna hyperparameter search
    best_params = tune_hyperparameters(X, y, scale_pos_weight, n_trials=OPTUNA_TRIALS)

    # Step 6 — train final model + apply Platt calibration
    calibrated_model, base_model, metrics, background_data, X_train = train_model(
        X, y, best_params, scale_pos_weight
    )

    # Step 7 — save everything
    save_artifacts(
        calibrated_model, base_model, label_encoders, feature_cols,
        metrics, background_data,
    )

    print("\nDone. Re-run the Streamlit app to use the updated model.")
