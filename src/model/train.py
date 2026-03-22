"""
train.py — NIDS ML Training Pipeline
=====================================
Loads CICIDS2017 data, cleans it, handles class imbalance,
trains Random Forest (baseline) and XGBoost (main model),
evaluates both, and saves the best model + scaler to disk.

Run:
    python src/model/train.py
"""

import os
import sys
import time
import joblib
import logging
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]   # nids-ml/
RAW_DIR      = ROOT / "data" / "raw"
PROCESSED_DIR= ROOT / "data" / "processed"
MODEL_PATH   = ROOT / "model.pkl"
SCALER_PATH  = ROOT / "scaler.pkl"
ENCODER_PATH = ROOT / "label_encoder.pkl"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP A — Load Data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(raw_dir: Path) -> pd.DataFrame:
    """
    Load all CSV files from data/raw/ into one DataFrame.
    CICIDS2017 comes as multiple CSVs (one per day of capture).
    """
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        log.error(f"No CSV files found in {raw_dir}")
        log.error("Download CICIDS2017 from https://cicresearch.ca and place CSVs in data/raw/")
        sys.exit(1)

    log.info(f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")

    frames = []
    # for f in csv_files:
    #     log.info(f"  Loading {f.name} ...")
    #     # df = pd.read_csv(f, encoding="utf-8", low_memory=False)
    #     df = df.sample(n=50000, random_state=42)
    #     # Strip leading/trailing spaces from column names (CICIDS2017 quirk)
    #     df.columns = df.columns.str.strip()
    #     frames.append(df)

    # combined = pd.concat(frames, ignore_index=True)
    # combined = combined.sample(n=50000, random_state=42)

    for f in csv_files:
        log.info(f"  Loading {f.name} ...")
        df = pd.read_csv(f, encoding="utf-8", low_memory=False, nrows=50000)
        df.columns = df.columns.str.strip()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    
    log.info(f"Total rows loaded: {len(combined):,}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# STEP B — Clean Data
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove NaN, Infinity values, and duplicate rows.
    These make up roughly 1-2% of CICIDS2017 and cause model training errors.
    """
    original_len = len(df)

    # Replace Infinity with NaN so we can drop them in one step
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop any row that has at least one NaN value
    df.dropna(inplace=True)

    # Drop exact duplicate rows
    df.drop_duplicates(inplace=True)

    dropped = original_len - len(df)
    log.info(f"Cleaned data: removed {dropped:,} rows ({dropped/original_len*100:.1f}%)")
    log.info(f"Rows remaining: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP C — Separate Features and Labels
# ─────────────────────────────────────────────────────────────────────────────

def split_features_labels(df: pd.DataFrame):
    """
    Separate the feature columns (X) from the label column (y).
    CICIDS2017 label column is named ' Label' (note the space — we stripped it).
    """
    label_col = "Attack Type"

    if label_col not in df.columns:
        # Try to auto-detect the label column
        candidates = [c for c in df.columns if "label" in c.lower()]
        if not candidates:
            log.error("Could not find a 'Label' column. Check your CSV files.")
            sys.exit(1)
        label_col = candidates[0]
        log.warning(f"Using '{label_col}' as the label column.")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Drop non-numeric columns if any slipped through
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log.warning(f"Dropping non-numeric columns: {non_numeric}")
        X.drop(columns=non_numeric, inplace=True)

    log.info(f"Features: {X.shape[1]} columns  |  Label: '{label_col}'")
    log.info(f"Class distribution:\n{y.value_counts().to_string()}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# STEP D — Encode Labels
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(y: pd.Series):
    """
    Convert string class names to integers.
    e.g. 'BENIGN' → 0, 'DDoS' → 1, 'PortScan' → 2, ...
    Saves the encoder so we can reverse-lookup at prediction time.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    log.info("Label encoding map:")
    for i, cls in enumerate(le.classes_):
        log.info(f"  {i:2d} → {cls}")

    joblib.dump(le, ENCODER_PATH)
    log.info(f"Label encoder saved to {ENCODER_PATH}")
    return y_encoded, le


# ─────────────────────────────────────────────────────────────────────────────
# STEP E — Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────

def make_split(X, y, test_size=0.2, random_state=42):
    """
    Stratified split: keeps class proportions the same in train and test.
    80% train, 20% test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,           # preserve class ratios
        random_state=random_state
    )
    log.info(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# STEP F — Handle Class Imbalance with SMOTE
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X_train, y_train, random_state=42):
    """
    SMOTE (Synthetic Minority Over-sampling TEchnique):
    Creates synthetic samples for minority attack classes so the model
    does not just learn to predict BENIGN all the time.

    Applied only on training data — NEVER on test data.
    """
    log.info("Applying SMOTE to balance training classes ...")
    log.info(f"Before SMOTE — training samples: {len(X_train):,}")

    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    log.info(f"After  SMOTE — training samples: {len(X_resampled):,}")
    return X_resampled, y_resampled


# ─────────────────────────────────────────────────────────────────────────────
# STEP G — Scale Features
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(X_train, X_test):
    """
    StandardScaler: transforms each feature to have mean=0, std=1.
    Fit ONLY on training data. Apply (transform) to both train and test.
    Saves the scaler so we can use it at prediction time.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
    X_test_scaled  = scaler.transform(X_test)         # transform only

    joblib.dump(scaler, SCALER_PATH)
    log.info(f"Scaler saved to {SCALER_PATH}")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# STEP H — Train Models
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    """
    Random Forest — baseline model.
    Simple, fast to train, gives us a benchmark to beat with XGBoost.
    """
    log.info("Training Random Forest (baseline) ...")
    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=100,    # 100 trees
        max_depth=20,        # limit depth to avoid overfitting
        n_jobs=-1,           # use all CPU cores
        random_state=42,
        class_weight="balanced"  # helps with imbalance
    )
    rf.fit(X_train, y_train)

    log.info(f"Random Forest trained in {time.time()-t0:.1f}s")
    return rf


def train_xgboost(X_train, y_train, num_classes: int):
    """
    XGBoost — main model.
    Gradient boosted trees: more accurate than Random Forest on tabular data.
    """
    log.info("Training XGBoost (main model) ...")
    t0 = time.time()

    xgb = XGBClassifier(
        n_estimators=300,        # number of boosting rounds
        max_depth=6,             # tree depth
        learning_rate=0.1,       # step size shrinkage
        subsample=0.8,           # fraction of rows per tree
        colsample_bytree=0.8,    # fraction of features per tree
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss",  # multi-class log loss
        verbosity=0,             # suppress XGBoost logs
        # num_class=num_classes    # number of attack classes
    )
    xgb.fit(X_train, y_train)

    log.info(f"XGBoost trained in {time.time()-t0:.1f}s")
    return xgb


# ─────────────────────────────────────────────────────────────────────────────
# STEP I — Evaluate Models
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str, label_encoder):
    """
    Compute accuracy, macro F1, per-class F1, and confusion matrix.
    Print a full classification report with class names.
    """
    log.info(f"\n{'='*55}")
    log.info(f"  Results: {model_name}")
    log.info(f"{'='*55}")

    y_pred = model.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_wt  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    log.info(f"  Accuracy       : {acc*100:.2f}%")
    log.info(f"  Macro F1       : {f1_mac*100:.2f}%")
    log.info(f"  Weighted F1    : {f1_wt*100:.2f}%")

    # Full per-class report
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    log.info(f"\nClassification Report:\n{report}")

    return {"accuracy": acc, "macro_f1": f1_mac, "weighted_f1": f1_wt}


# ─────────────────────────────────────────────────────────────────────────────
# STEP J — Cross Validation
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(model, X, y, model_name: str, n_splits=5):
    """
    5-fold stratified cross-validation.
    More reliable than a single train/test split.
    Shows how the model performs across different subsets of data.
    """
    log.info(f"\nRunning {n_splits}-fold cross-validation for {model_name} ...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X, y,
        cv=skf,
        scoring="f1_macro",
        n_jobs=-1
    )

    log.info(f"  CV F1 per fold : {[f'{s:.4f}' for s in scores]}")
    log.info(f"  Mean CV F1     : {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# STEP K — Save Best Model
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, metrics: dict, model_name: str):
    """
    Save the trained model as model.pkl.
    Also save a small metadata dict alongside it.
    """
    joblib.dump(model, MODEL_PATH)
    log.info(f"\nModel saved → {MODEL_PATH}")
    log.info(f"  Model type : {model_name}")
    log.info(f"  Macro F1   : {metrics['macro_f1']*100:.2f}%")
    log.info(f"  Accuracy   : {metrics['accuracy']*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 55)
    log.info("  NIDS — ML Training Pipeline")
    log.info("=" * 55)

    # A. Load
    df = load_data(RAW_DIR)

    # B. Clean
    df = clean_data(df)

    # C. Split features / labels
    X, y = split_features_labels(df)

    # D. Encode labels
    y_encoded, label_encoder = encode_labels(y)

    # E. Train/test split
    X_train, X_test, y_train, y_test = make_split(X.values, y_encoded)

    # F. SMOTE (on training data only)
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # G. Scale features
    X_train_sc, X_test_sc, scaler = scale_features(X_train_bal, X_test)

    num_classes = len(label_encoder.classes_)

    # H. Train both models
    rf  = train_random_forest(X_train_sc, y_train_bal)
    xgb = train_xgboost(X_train_sc, y_train_bal, num_classes)

    # I. Evaluate both on test set
    rf_metrics  = evaluate_model(rf,  X_test_sc, y_test, "Random Forest", label_encoder)
    xgb_metrics = evaluate_model(xgb, X_test_sc, y_test, "XGBoost",       label_encoder)

    # J. Cross-validate XGBoost (uses full scaled data — just first 50k rows for speed)
    sample_size = min(50_000, len(X_train_sc))
    cross_validate_model(
        xgb,
        X_train_sc[:sample_size],
        y_train_bal[:sample_size],
        "XGBoost"
    )

    # K. Save the best model (XGBoost wins if its F1 > RF's F1)
    if xgb_metrics["macro_f1"] >= rf_metrics["macro_f1"]:
        log.info("\nXGBoost is the better model — saving as model.pkl")
        save_model(xgb, xgb_metrics, "XGBoost")
    else:
        log.info("\nRandom Forest is the better model — saving as model.pkl")
        save_model(rf, rf_metrics, "Random Forest")

    log.info("\nTraining complete!")
    log.info(f"  model.pkl      → {MODEL_PATH}")
    log.info(f"  scaler.pkl     → {SCALER_PATH}")
    log.info(f"  label_encoder  → {ENCODER_PATH}")
    log.info("\nNext step: run predict.py to test inference on a sample flow.")


if __name__ == "__main__":
    main()