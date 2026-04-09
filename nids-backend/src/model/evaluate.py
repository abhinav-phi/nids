"""
evaluate.py — NIDS Model Evaluation Utilities
===============================================
Standalone functions for computing and printing metrics.
Called by train.py but can also be run independently
to re-evaluate a saved model on new data.
Run:
    python src/model/evaluate.py
"""
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
log = logging.getLogger(__name__)
ROOT         = Path(__file__).resolve().parents[2]
MODEL_PATH   = ROOT / "model.pkl"
SCALER_PATH  = ROOT / "scaler.pkl"
ENCODER_PATH = ROOT / "label_encoder.pkl"
def compute_metrics(y_true, y_pred, label_names=None) -> dict:
    """
    Compute a full set of evaluation metrics and return as a dictionary.
    Parameters
    ----------
    y_true      : array of true integer labels
    y_pred      : array of predicted integer labels
    label_names : list of class name strings (optional, for readability)
    Returns
    -------
    dict with keys: accuracy, macro_f1, weighted_f1, macro_precision,
                    macro_recall, per_class_f1
    """
    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class = {}
    for i, score in enumerate(f1_per_class):
        name = label_names[i] if label_names and i < len(label_names) else str(i)
        per_class[name] = round(score, 4)
    return {
        "accuracy":         round(acc,  4),
        "macro_f1":         round(f1m,  4),
        "weighted_f1":      round(f1w,  4),
        "macro_precision":  round(prec, 4),
        "macro_recall":     round(rec,  4),
        "per_class_f1":     per_class,
    }
def print_metrics(metrics: dict):
    """Pretty-print the metrics dictionary."""
    print("\n" + "="*50)
    print("  MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"  Accuracy          : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1          : {metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted F1       : {metrics['weighted_f1']*100:.2f}%")
    print(f"  Macro Precision   : {metrics['macro_precision']*100:.2f}%")
    print(f"  Macro Recall      : {metrics['macro_recall']*100:.2f}%")
    print("\n  Per-class F1:")
    for cls, score in metrics["per_class_f1"].items():
        bar = "█" * int(score * 20)
        print(f"    {cls:<30} {score:.4f}  {bar}")
    print("="*50)
def false_positive_rate(y_true, y_pred, benign_label: int = 0) -> float:
    """
    Compute the False Positive Rate for the BENIGN class.
    FPR = FP / (FP + TN)
    In NIDS, FPR matters because a high FPR means normal traffic
    is wrongly flagged as attacks — causing alert fatigue.
    """
    tn = np.sum((y_true == benign_label) & (y_pred == benign_label))
    fp = np.sum((y_true == benign_label) & (y_pred != benign_label))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return round(fpr, 4)
def plot_confusion_matrix(y_true, y_pred, label_names, output_path=None):
    """
    Plot and save a confusion matrix heatmap.
    Saves to data/processed/confusion_matrix.png
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(14, 10))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title("NIDS — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = output_path or (ROOT / "data" / "processed" / "confusion_matrix.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\nConfusion matrix saved → {save_path}")
def plot_feature_importance(model, feature_names, top_n=20, output_path=None):
    """
    Plot top N most important features from XGBoost or Random Forest.
    Helps understand which network features matter most for detection.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("This model does not have feature_importances_")
        return
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores   = importances[indices]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), top_scores[::-1], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold")
    plt.tight_layout()
    save_path = output_path or (ROOT / "data" / "processed" / "feature_importance.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Feature importance plot saved → {save_path}")
if __name__ == "__main__":
    import sys
    if not MODEL_PATH.exists():
        print(f"model.pkl not found at {MODEL_PATH}")
        print("Run train.py first.")
        sys.exit(1)
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    test_X_path = ROOT / "data" / "processed" / "X_test.npy"
    test_y_path = ROOT / "data" / "processed" / "y_test.npy"
    if not test_X_path.exists():
        print("No saved test data found. Run train.py first (it saves X_test.npy).")
        sys.exit(1)
    X_test = np.load(test_X_path)
    y_test = np.load(test_y_path)
    y_pred  = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, list(encoder.classes_))
    print_metrics(metrics)
    fpr = false_positive_rate(y_test, y_pred, benign_label=0)
    print(f"\n  False Positive Rate : {fpr*100:.2f}%")
    plot_confusion_matrix(y_test, y_pred, list(encoder.classes_))
