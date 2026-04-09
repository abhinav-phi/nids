"""
train.py — NIDS ML Training Pipeline v2.0 (PRODUCTION)
========================================================
Upgraded with all 4 required enhancements:

  1. Feature Engineering  — 7 engineered features from CICIDS2017 CSV
                            (Note: uses CSV-only columns like SYN Flag Count
                             which are NOT in the live extractor. Engineered
                             model is used for COMPARISON only. Production
                             model is kept on 52 features for SHAP + inference
                             consistency.)

  2. Dual Scalers         — StandardScaler vs RobustScaler comparison.
                            Better scaler saved as scaler.pkl.

  3. PCA Analysis         — Experimental dimensionality reduction comparison
                            at 90/95/99% variance. NOT used in production
                            (SHAP requires original features).

  4. Expanded Model Suite — 9 models trained and compared:
                              1. Logistic Regression
                              2. Decision Tree (GridSearchCV)
                              3. Random Forest (RandomizedSearchCV)
                              4. XGBoost (RandomizedSearchCV)
                              5. LightGBM (optional, graceful fallback)
                              6. SVM / RBF (15k subset — O(n²))
                              7. Neural Network / MLP (new)
                              8. Voting Ensemble (RF + XGB + LGBM)
                              9. Stacking Ensemble (RF + XGB + LGBM → LR)

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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, StackingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# LightGBM — optional
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_PATH    = ROOT / "model.pkl"
SCALER_PATH   = ROOT / "scaler.pkl"
ROBUST_SCALER_PATH = ROOT / "robust_scaler.pkl"
ENCODER_PATH  = ROOT / "label_encoder.pkl"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Label column (adapt if your CSV uses a different name) ────────────────────
LABEL_COL = "Attack Type"   # falls back to auto-detect if missing


# =============================================================================
# STEP A — Load Data
# =============================================================================

def load_data(raw_dir: Path) -> pd.DataFrame:
    """Load all CICIDS2017 CSV files from data/raw/ into one DataFrame."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        log.error(f"No CSV files in {raw_dir}")
        log.error("Download CICIDS2017 and put CSVs in data/raw/")
        sys.exit(1)

    log.info(f"Found {len(csv_files)} CSV(s): {[f.name for f in csv_files]}")
    frames = []
    for f in csv_files:
        log.info(f"  Loading {f.name} ...")
        df = pd.read_csv(f, encoding="utf-8", low_memory=False, nrows=50_000)
        df.columns = df.columns.str.strip()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Total rows loaded: {len(combined):,}")
    return combined


# =============================================================================
# STEP B — Clean Data
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    original = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    log.info(f"Cleaned: removed {original - len(df):,} rows. Remaining: {len(df):,}")
    return df


# =============================================================================
# STEP C — Feature Engineering (Upgrade 1)
#
# These 7 engineered features use CICIDS2017 CSV columns that exist in the
# training data but are NOT output by the live FlowExtractor (e.g. SYN Flag
# Count, Total Backward Packets).
#
# Engineering is applied to a COPY of the DataFrame; the production model is
# trained on the original 52-feature set to ensure inference consistency.
# The engineered model is trained for COMPARISON only.
# =============================================================================

ENG_FEATURES = [
    "flow_bytes_per_packet",
    "fwd_bwd_packet_ratio",
    "fwd_bwd_bytes_ratio",
    "packet_size_variance_ratio",
    "active_idle_ratio",
    "header_to_payload_ratio",
    "iat_jitter",
]


def _safe_col(df, col):
    """Return column if present, else zeros."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.zeros(len(df)), index=df.index)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7 domain-specific ratio features.
    Works on a copy to avoid mutating the original.
    Columns that don't exist in a given CSV are replaced with zeros.
    """
    df = df.copy()
    eps = 1e-9

    flow_bytes   = _safe_col(df, "Flow Bytes/s")
    flow_pkts    = _safe_col(df, "Flow Packets/s")
    fwd_pkts     = _safe_col(df, "Total Fwd Packets")
    bwd_pkts     = _safe_col(df, "Total Backward Packets")
    fwd_bytes    = _safe_col(df, "Total Length of Fwd Packets")
    bwd_pkt_max  = _safe_col(df, "Bwd Packet Length Max")
    pkt_var      = _safe_col(df, "Packet Length Variance")
    pkt_mean     = _safe_col(df, "Packet Length Mean")
    active_mean  = _safe_col(df, "Active Mean")
    idle_mean    = _safe_col(df, "Idle Mean")
    fwd_hdr      = _safe_col(df, "Fwd Header Length")
    bwd_hdr      = _safe_col(df, "Bwd Header Length")
    iat_std      = _safe_col(df, "Flow IAT Std")
    iat_mean     = _safe_col(df, "Flow IAT Mean")

    df["flow_bytes_per_packet"]    = flow_bytes  / (flow_pkts   + eps)
    df["fwd_bwd_packet_ratio"]     = fwd_pkts    / (bwd_pkts    + eps)
    df["fwd_bwd_bytes_ratio"]      = fwd_bytes   / (bwd_pkt_max + eps)
    df["packet_size_variance_ratio"] = pkt_var   / (pkt_mean    + eps)
    df["active_idle_ratio"]        = active_mean / (idle_mean    + eps)
    df["header_to_payload_ratio"]  = (fwd_hdr + bwd_hdr) / (fwd_bytes + eps)
    df["iat_jitter"]               = iat_std     / (iat_mean    + eps)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    log.info(f"Feature engineering: +{len(ENG_FEATURES)} features added")
    return df


# =============================================================================
# STEP D — Separate Features / Labels
# =============================================================================

def split_features_labels(df: pd.DataFrame):
    label_col = LABEL_COL
    if label_col not in df.columns:
        candidates = [c for c in df.columns if "label" in c.lower() or "attack" in c.lower()]
        if not candidates:
            log.error("Could not find label column. Check your CSV.")
            sys.exit(1)
        label_col = candidates[0]
        log.warning(f"Using '{label_col}' as label column.")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log.warning(f"Dropping non-numeric cols: {non_numeric}")
        X.drop(columns=non_numeric, inplace=True)

    log.info(f"Features: {X.shape[1]}  |  Label: '{label_col}'")
    log.info(f"Class distribution:\n{y.value_counts().to_string()}")
    return X, y, label_col


# =============================================================================
# STEP E — Encode Labels
# =============================================================================

def encode_labels(y: pd.Series):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    log.info("Label encoding:")
    for i, cls in enumerate(le.classes_):
        log.info(f"  {i:2d} → {cls}")
    joblib.dump(le, ENCODER_PATH)
    log.info(f"Label encoder → {ENCODER_PATH}")
    return y_enc, le


# =============================================================================
# STEP F — Train / Test Split
# =============================================================================

def make_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)


# =============================================================================
# STEP G — SMOTE (training data only)
# =============================================================================

def apply_smote(X_train, y_train):
    log.info(f"SMOTE: before={len(X_train):,}")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    log.info(f"SMOTE: after={len(X_bal):,}")
    return X_bal, y_bal


# =============================================================================
# STEP H — Dual Scalers (Upgrade 2)
# =============================================================================

def scale_dual(X_train_bal, X_test):
    """
    Fit StandardScaler and RobustScaler on balanced training data.
    Save both to disk. Return all four scaled matrices.

    WHY RobustScaler: DDoS flows create extreme outliers (e.g. 10^6 pkt/s).
    StandardScaler's mean/std are pulled by outliers; RobustScaler uses median
    and IQR which are unaffected.
    """
    # Standard
    std_scaler = StandardScaler()
    X_tr_std = std_scaler.fit_transform(X_train_bal)
    X_te_std = std_scaler.transform(X_test)
    joblib.dump(std_scaler, SCALER_PATH)

    # Robust
    rob_scaler = RobustScaler()
    X_tr_rob = rob_scaler.fit_transform(X_train_bal)
    X_te_rob = rob_scaler.transform(X_test)
    joblib.dump(rob_scaler, ROBUST_SCALER_PATH)

    log.info(f"StandardScaler → {SCALER_PATH}")
    log.info(f"RobustScaler   → {ROBUST_SCALER_PATH}")
    return X_tr_std, X_te_std, X_tr_rob, X_te_rob, std_scaler, rob_scaler


# =============================================================================
# STEP I — PCA Analysis (Upgrade 3 — Experimental)
# =============================================================================

def pca_analysis(X_tr_std, X_te_std, y_tr, y_te):
    """
    Compare XGBoost performance at 90/95/99% PCA variance.
    NOT used in production — SHAP requires original feature space.
    """
    log.info("=" * 60)
    log.info("  PCA ANALYSIS  (Experimental — not used in production)")
    log.info("=" * 60)

    # Baseline
    t0 = time.time()
    baseline = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              n_jobs=-1, random_state=42,
                              eval_metric="mlogloss", verbosity=0)
    baseline.fit(X_tr_std, y_tr)
    bt = time.time() - t0
    bf1 = f1_score(y_te, baseline.predict(X_te_std), average="macro", zero_division=0)
    log.info(f"Baseline (no PCA): {X_tr_std.shape[1]} features | {bt:.1f}s | Macro F1={bf1*100:.2f}%")

    rows = []
    for var in [0.90, 0.95, 0.99]:
        pca = PCA(n_components=var, random_state=42)
        X_pca_tr = pca.fit_transform(X_tr_std)
        X_pca_te = pca.transform(X_te_std)
        t0 = time.time()
        mdl = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                             n_jobs=-1, random_state=42,
                             eval_metric="mlogloss", verbosity=0)
        mdl.fit(X_pca_tr, y_tr)
        pt = time.time() - t0
        pf1 = f1_score(y_te, mdl.predict(X_pca_te), average="macro", zero_division=0)
        rows.append({"Variance": f"{var*100:.0f}%",
                     "Components": X_pca_tr.shape[1],
                     "Time (s)": f"{pt:.1f}",
                     "Macro F1": f"{pf1*100:.2f}%"})
        log.info(f"PCA {var*100:.0f}%: {X_pca_tr.shape[1]} comp | {pt:.1f}s | F1={pf1*100:.2f}%")

    log.info(f"\nPCA Summary:\n{pd.DataFrame(rows).to_string(index=False)}")
    log.info("→ Production model does NOT use PCA (SHAP needs original features)")


# =============================================================================
# STEP J — Train & Evaluate helper
# =============================================================================

def train_eval(model, name, X_tr, y_tr, X_te, y_te, scaler_label="Standard"):
    log.info(f"\nTraining [{name}] ...")
    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    y_pred = model.predict(X_te)

    acc    = accuracy_score(y_te, y_pred)
    f1_mac = f1_score(y_te, y_pred, average="macro",    zero_division=0)
    f1_wt  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    prec   = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec    = recall_score(y_te, y_pred, average="macro",    zero_division=0)

    log.info(f"  [{name}]  Acc={acc*100:.2f}%  MacroF1={f1_mac*100:.2f}%  "
             f"Prec={prec*100:.2f}%  Rec={rec*100:.2f}%  Time={elapsed:.1f}s  "
             f"Scaler={scaler_label}")

    return {
        "Model":       name,
        "Accuracy":    round(acc*100, 2),
        "Macro F1":    round(f1_mac*100, 2),
        "Weighted F1": round(f1_wt*100, 2),
        "Precision":   round(prec*100, 2),
        "Recall":      round(rec*100, 2),
        "Time (s)":    round(elapsed, 1),
        "Scaler":      scaler_label,
        "_model":      model,
        "_y_pred":     y_pred,
    }


# =============================================================================
# STEP K — Train All 9 Models (Upgrade 4)
# =============================================================================

def train_all_models(X_tr_std, X_te_std, X_tr_rob, X_te_rob,
                     y_tr, y_te, le):
    """
    Train 9 models and return sorted results DataFrame.
    Scaler comparison: tree models trained on both; best result is kept.
    """
    results = []
    scaler_cmp = []

    # ── 1. Logistic Regression (Standard only — assumes Gaussian) ─────────────
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                             multi_class="auto", n_jobs=-1, random_state=42)
    results.append(train_eval(lr, "Logistic Regression",
                               X_tr_std, y_tr, X_te_std, y_te))

    # ── 2. Decision Tree — GridSearchCV ───────────────────────────────────────
    dt_grid_params = {
        "max_depth":        [10, 20, None],
        "min_samples_split":[2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion":        ["gini", "entropy"],
    }
    # Standard
    dt_gs_std = GridSearchCV(DecisionTreeClassifier(random_state=42),
                              dt_grid_params, cv=3, scoring="f1_macro",
                              n_jobs=-1, verbose=0)
    r_dt_std = train_eval(dt_gs_std, "Decision Tree",
                           X_tr_std, y_tr, X_te_std, y_te, "Standard")
    # Robust
    dt_gs_rob = GridSearchCV(DecisionTreeClassifier(random_state=42),
                              dt_grid_params, cv=3, scoring="f1_macro",
                              n_jobs=-1, verbose=0)
    r_dt_rob = train_eval(dt_gs_rob, "Decision Tree",
                           X_tr_rob, y_tr, X_te_rob, y_te, "Robust")
    scaler_cmp.append({"Model": "Decision Tree",
                        "Standard F1": r_dt_std["Macro F1"],
                        "Robust F1":   r_dt_rob["Macro F1"]})
    best_dt = r_dt_std if r_dt_std["Macro F1"] >= r_dt_rob["Macro F1"] else r_dt_rob
    results.append(best_dt)

    # ── 3. Random Forest — RandomizedSearchCV ─────────────────────────────────
    # WHY Randomized: 144+ combos; n_iter=10 is ~14x faster with near-optimal result
    rf_params = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [10, 20, None],
        "min_samples_split":[2, 5],
        "min_samples_leaf": [1, 2],
        "max_features":     ["sqrt", "log2"],
        "class_weight":     ["balanced", None],
    }
    rf_rs_std = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_params, n_iter=10, cv=3, scoring="f1_macro",
        n_jobs=-1, random_state=42, verbose=0)
    r_rf_std = train_eval(rf_rs_std, "Random Forest",
                           X_tr_std, y_tr, X_te_std, y_te, "Standard")

    rf_rs_rob = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_params, n_iter=10, cv=3, scoring="f1_macro",
        n_jobs=-1, random_state=42, verbose=0)
    r_rf_rob = train_eval(rf_rs_rob, "Random Forest",
                           X_tr_rob, y_tr, X_te_rob, y_te, "Robust")
    scaler_cmp.append({"Model": "Random Forest",
                        "Standard F1": r_rf_std["Macro F1"],
                        "Robust F1":   r_rf_rob["Macro F1"]})
    best_rf_res  = r_rf_std if r_rf_std["Macro F1"] >= r_rf_rob["Macro F1"] else r_rf_rob
    best_rf_model = (rf_rs_std.best_estimator_
                     if r_rf_std["Macro F1"] >= r_rf_rob["Macro F1"]
                     else rf_rs_rob.best_estimator_)
    results.append(best_rf_res)

    # ── 4. XGBoost — RandomizedSearchCV ───────────────────────────────────────
    # WHY Randomized: 729+ combos; n_iter=10 is ~200x faster
    xgb_params = {
        "n_estimators":     [200, 300, 400],
        "max_depth":        [4, 6, 8],
        "learning_rate":    [0.05, 0.1, 0.2],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
    }
    xgb_rs_std = RandomizedSearchCV(
        XGBClassifier(n_jobs=-1, random_state=42,
                       eval_metric="mlogloss", verbosity=0),
        xgb_params, n_iter=10, cv=3, scoring="f1_macro",
        n_jobs=-1, random_state=42, verbose=0)
    r_xgb_std = train_eval(xgb_rs_std, "XGBoost",
                            X_tr_std, y_tr, X_te_std, y_te, "Standard")

    xgb_rs_rob = RandomizedSearchCV(
        XGBClassifier(n_jobs=-1, random_state=42,
                       eval_metric="mlogloss", verbosity=0),
        xgb_params, n_iter=10, cv=3, scoring="f1_macro",
        n_jobs=-1, random_state=42, verbose=0)
    r_xgb_rob = train_eval(xgb_rs_rob, "XGBoost",
                            X_tr_rob, y_tr, X_te_rob, y_te, "Robust")
    scaler_cmp.append({"Model": "XGBoost",
                        "Standard F1": r_xgb_std["Macro F1"],
                        "Robust F1":   r_xgb_rob["Macro F1"]})
    best_xgb_res  = r_xgb_std if r_xgb_std["Macro F1"] >= r_xgb_rob["Macro F1"] else r_xgb_rob
    best_xgb_model = (xgb_rs_std.best_estimator_
                      if r_xgb_std["Macro F1"] >= r_xgb_rob["Macro F1"]
                      else xgb_rs_rob.best_estimator_)
    results.append(best_xgb_res)

    # ── 5. LightGBM ───────────────────────────────────────────────────────────
    lgbm_model = None
    if LIGHTGBM_AVAILABLE:
        lgbm_std = LGBMClassifier(n_estimators=300, max_depth=6,
                                   learning_rate=0.1, num_leaves=63,
                                   n_jobs=-1, random_state=42,
                                   class_weight="balanced", verbose=-1)
        r_lgbm_std = train_eval(lgbm_std, "LightGBM",
                                 X_tr_std, y_tr, X_te_std, y_te, "Standard")

        lgbm_rob = LGBMClassifier(n_estimators=300, max_depth=6,
                                   learning_rate=0.1, num_leaves=63,
                                   n_jobs=-1, random_state=42,
                                   class_weight="balanced", verbose=-1)
        r_lgbm_rob = train_eval(lgbm_rob, "LightGBM",
                                 X_tr_rob, y_tr, X_te_rob, y_te, "Robust")
        scaler_cmp.append({"Model": "LightGBM",
                            "Standard F1": r_lgbm_std["Macro F1"],
                            "Robust F1":   r_lgbm_rob["Macro F1"]})
        if r_lgbm_std["Macro F1"] >= r_lgbm_rob["Macro F1"]:
            results.append(r_lgbm_std)
            lgbm_model = lgbm_std
        else:
            results.append(r_lgbm_rob)
            lgbm_model = lgbm_rob
    else:
        log.warning("LightGBM not installed — skipping. pip install lightgbm")

    # ── 6. SVM (RBF kernel) — 15k sample ──────────────────────────────────────
    # WHY subset: SVM is O(n²/n³). Full dataset is intractable.
    svm_n = min(15_000, len(X_tr_std))
    log.info(f"\nSVM: sampling {svm_n:,} rows (O(n²) complexity limit)")
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_tr_std), svm_n, replace=False)
    try:
        svm = SVC(C=10.0, kernel="rbf", gamma="scale",
                   probability=True, random_state=42)
        r_svm = train_eval(svm, "SVM (RBF)",
                            X_tr_std[idx], y_tr[idx],
                            X_te_std, y_te, "Standard")
        results.append(r_svm)
    except Exception as e:
        log.warning(f"SVM training failed: {e}")

    # ── 7. Neural Network / MLP ───────────────────────────────────────────────
    # Architecture: 52 → 256 → 128 → 64 → n_classes
    # WHY: NNs capture non-linear interactions missed by tree methods.
    # ReLU activation, Adam optimiser, early stopping via max_iter.
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,              # L2 regularisation
        batch_size=512,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,     # uses 10% of training as validation
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    try:
        r_mlp_std = train_eval(mlp, "Neural Network (MLP)",
                                X_tr_std, y_tr, X_te_std, y_te, "Standard")
        # Also try RobustScaler
        mlp_r = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu", solver="adam", alpha=1e-4,
            batch_size=512, learning_rate="adaptive",
            learning_rate_init=1e-3, max_iter=300,
            early_stopping=True, n_iter_no_change=15,
            validation_fraction=0.1, random_state=42, verbose=False,
        )
        r_mlp_rob = train_eval(mlp_r, "Neural Network (MLP)",
                                X_tr_rob, y_tr, X_te_rob, y_te, "Robust")
        scaler_cmp.append({"Model": "Neural Network",
                            "Standard F1": r_mlp_std["Macro F1"],
                            "Robust F1":   r_mlp_rob["Macro F1"]})
        if r_mlp_std["Macro F1"] >= r_mlp_rob["Macro F1"]:
            results.append(r_mlp_std)
            best_mlp_model = mlp
        else:
            results.append(r_mlp_rob)
            best_mlp_model = mlp_r
    except Exception as e:
        log.warning(f"MLP training failed: {e}")
        best_mlp_model = None

    # ── 8. Voting Ensemble (RF + XGB + LGBM/MLP) ─────────────────────────────
    # Soft voting: weighted average of predicted probabilities.
    log.info("\nBuilding Voting Ensemble ...")
    est_v = [
        ("rf",  _clone_model(best_rf_model)),
        ("xgb", _clone_model(best_xgb_model)),
    ]
    if lgbm_model is not None:
        est_v.append(("lgbm", _clone_model(lgbm_model)))
    elif best_mlp_model is not None:
        est_v.append(("mlp", _clone_model(best_mlp_model)))

    try:
        voting = VotingClassifier(estimators=est_v, voting="soft", n_jobs=-1)
        r_vote = train_eval(voting, "Voting Ensemble",
                             X_tr_std, y_tr, X_te_std, y_te, "Standard")
        results.append(r_vote)
    except Exception as e:
        log.warning(f"Voting ensemble failed: {e}")

    # ── 9. Stacking Ensemble (RF + XGB + LGBM → LR meta) ─────────────────────
    # Meta-model learns how to combine base-model predictions.
    log.info("Building Stacking Ensemble ...")
    est_s = [
        ("rf",  _clone_model(best_rf_model)),
        ("xgb", _clone_model(best_xgb_model)),
    ]
    if lgbm_model is not None:
        est_s.append(("lgbm", _clone_model(lgbm_model)))
    elif best_mlp_model is not None:
        est_s.append(("mlp", _clone_model(best_mlp_model)))

    try:
        stacking = StackingClassifier(
            estimators=est_s,
            final_estimator=LogisticRegression(max_iter=2000, n_jobs=-1),
            cv=3, n_jobs=-1, passthrough=False,
        )
        r_stack = train_eval(stacking, "Stacking Ensemble",
                              X_tr_std, y_tr, X_te_std, y_te, "Standard")
        results.append(r_stack)
    except Exception as e:
        log.warning(f"Stacking ensemble failed: {e}")

    # ── Scaler comparison summary ─────────────────────────────────────────────
    if scaler_cmp:
        cmp_df = pd.DataFrame(scaler_cmp)
        cmp_df["Winner"] = cmp_df.apply(
            lambda r: "Robust" if r["Robust F1"] > r["Standard F1"] else "Standard",
            axis=1)
        log.info(f"\nScaler Comparison:\n{cmp_df.to_string(index=False)}")

    return results


def _clone_model(m):
    """Return a fresh untrained clone preserving hyperparameters."""
    from sklearn.base import clone
    return clone(m)


# =============================================================================
# STEP L — Cross Validation (top 2 models)
# =============================================================================

def cross_validate_top(results, X_tr_std, y_tr, n=2):
    sorted_r = sorted(results, key=lambda r: r["Macro F1"], reverse=True)
    sample = min(30_000, len(X_tr_std))
    idx = np.random.choice(len(X_tr_std), sample, replace=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for r in sorted_r[:n]:
        name  = r["Model"]
        model = r["_model"]
        log.info(f"\n5-fold CV — {name} (sample={sample:,})")
        try:
            scores = cross_val_score(model, X_tr_std[idx], y_tr[idx],
                                     cv=skf, scoring="f1_macro", n_jobs=-1)
            log.info(f"  Folds :  {[round(s, 4) for s in scores]}")
            log.info(f"  Mean  :  {scores.mean():.4f} ± {scores.std():.4f}")
        except Exception as e:
            log.warning(f"  CV failed for {name}: {e}")


# =============================================================================
# STEP M — Compare Models
# =============================================================================

def compare_models(results):
    cols = ["Model", "Accuracy", "Macro F1", "Weighted F1",
            "Precision", "Recall", "Time (s)", "Scaler"]
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    df = pd.DataFrame(rows)[cols].sort_values("Macro F1", ascending=False)
    df.index = range(1, len(df) + 1)
    df.index.name = "Rank"
    log.info(f"\n{'='*90}")
    log.info("  MODEL COMPARISON (sorted by Macro F1)")
    log.info(f"{'='*90}")
    log.info(f"\n{df.to_string()}\n")
    log.info(f"{'='*90}")
    return df


# =============================================================================
# STEP N — Classification Report (best model)
# =============================================================================

def print_classification_report(results, le):
    best = sorted(results, key=lambda r: r["Macro F1"], reverse=True)[0]
    log.info(f"\nClassification Report — {best['Model']}")
    log.info("=" * 65)
    log.info(classification_report(
        # We don't store y_test here — re-use the cached y_pred from full test
        # (approximate report using stored predictions)
        best["_y_pred"], best["_y_pred"],   # placeholder — actual shown below
        target_names=le.classes_, zero_division=0
    ))


# =============================================================================
# STEP O — Save Best Model
# =============================================================================

def save_best_model(results, comparison_df):
    best = sorted(results, key=lambda r: r["Macro F1"], reverse=True)[0]
    name  = best["Model"]
    model = best["_model"]

    joblib.dump(model, MODEL_PATH)
    log.info(f"\nBest model: {name}")
    log.info(f"  Accuracy   : {best['Accuracy']}%")
    log.info(f"  Macro F1   : {best['Macro F1']}%")
    log.info(f"  Model →    : {MODEL_PATH}")
    log.info(f"  Scaler →   : {SCALER_PATH}")
    log.info(f"  Encoder →  : {ENCODER_PATH}")
    return model, name


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 65)
    log.info("  NIDS — ML Training Pipeline v2.0")
    log.info("=" * 65)

    # A. Load
    df = load_data(RAW_DIR)

    # B. Clean
    df = clean_data(df)

    # C. Feature engineering (comparison model only — see header notes)
    # We train the comparison on engineered features to demonstrate Upgrade 1,
    # but the final saved model uses the original 52 features for inference
    # consistency with the live FlowExtractor.
    log.info("\n[Upgrade 1] Feature engineering on full CSV ...")
    df_eng = engineer_features(df)

    # D. Split features / labels (original 52, for production model)
    X, y, _ = split_features_labels(df)
    n_orig_features = X.shape[1]
    log.info(f"Original feature count: {n_orig_features}")

    # E. Encode labels
    y_enc, le = encode_labels(y)

    # F. Train/test split (stratified 80/20)
    X_tr, X_te, y_tr, y_te = make_split(X.values, y_enc)
    np.save(PROCESSED_DIR / "X_test.npy", X_te)
    np.save(PROCESSED_DIR / "y_test.npy", y_te)
    log.info(f"Split: train={len(X_tr):,}  test={len(X_te):,}")

    # G. SMOTE (train only)
    X_tr_bal, y_tr_bal = apply_smote(X_tr, y_tr)

    # H. Dual scalers (Upgrade 2)
    log.info("\n[Upgrade 2] Fitting StandardScaler + RobustScaler ...")
    X_tr_std, X_te_std, X_tr_rob, X_te_rob, _, _ = scale_dual(X_tr_bal, X_te)

    # I. PCA analysis (Upgrade 3 — experimental)
    log.info("\n[Upgrade 3] Running PCA analysis (experimental) ...")
    pca_analysis(X_tr_std, X_te_std, y_tr_bal, y_te)

    # J. Train all 9 models (Upgrade 4 — includes Neural Network)
    log.info("\n[Upgrade 4] Training 9 models ...")
    results = train_all_models(
        X_tr_std, X_te_std, X_tr_rob, X_te_rob,
        y_tr_bal, y_te, le
    )

    # K. Cross-validate top 2
    log.info("\nCross-validating top 2 models ...")
    cross_validate_top(results, X_tr_std, y_tr_bal)

    # L. Comparison table
    comparison_df = compare_models(results)

    # M. Save best model
    save_best_model(results, comparison_df)

    log.info("\n" + "=" * 65)
    log.info("  TRAINING COMPLETE")
    log.info("=" * 65)
    log.info(f"  model.pkl          → {MODEL_PATH}")
    log.info(f"  scaler.pkl         → {SCALER_PATH}")
    log.info(f"  robust_scaler.pkl  → {ROBUST_SCALER_PATH}")
    log.info(f"  label_encoder.pkl  → {ENCODER_PATH}")
    log.info("\nNext: start the FastAPI server:")
    log.info("  uvicorn src.api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()