"""
predict.py — NIDS Inference Wrapper
=====================================
Loads the saved model, scaler, and label encoder once at import time.
Exposes a single predict() function used by the FastAPI backend.
"""

import joblib
import logging
import numpy as np

from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parents[2]
MODEL_PATH   = ROOT / "model.pkl"
SCALER_PATH  = ROOT / "scaler.pkl"
ENCODER_PATH = ROOT / "label_encoder.pkl"

# ── Severity mapping ──────────────────────────────────────────────────────────
SEVERITY_RULES = {
    "DDoS":        ("CRITICAL", 0.90),
    "DoS":         ("CRITICAL", 0.90),
    "Bots":        ("HIGH",     0.80),
    "Brute Force": ("HIGH",     0.80),
    "Port Scanning":("MEDIUM",  0.70),
    "Web Attacks": ("MEDIUM",   0.70),
}

def get_severity(prediction: str, confidence: float) -> str:
    if "normal" in prediction.lower():
        return "NONE"
    for attack_keyword, (level, threshold) in SEVERITY_RULES.items():
        if attack_keyword.lower() in prediction.lower() and confidence >= threshold:
            return level
    if confidence >= 0.50:
        return "LOW"
    return "NONE"


# ── Load artifacts ────────────────────────────────────────────────────────────
def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model.pkl not found at {MODEL_PATH}. Run src/model/train.py first."
        )
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    log.info(f"Model loaded. Classes: {list(encoder.classes_)}")
    return model, scaler, encoder


try:
    _model, _scaler, _encoder = _load_artifacts()
    _model_loaded = True
except FileNotFoundError as e:
    log.warning(str(e))
    _model_loaded = False


# ── Main predict function ─────────────────────────────────────────────────────
def predict(features: dict, feature_names: Optional[list] = None) -> dict:
    """
    Run inference on a single network flow.
    Returns: prediction, confidence, severity, shap_top5
    """
    if not _model_loaded:
        raise RuntimeError("Model not loaded. Run train.py first.")

    values = np.array(list(features.values()), dtype=np.float64).reshape(1, -1)
    values_scaled = _scaler.transform(values)

    pred_index    = int(_model.predict(values_scaled)[0])
    probabilities = _model.predict_proba(values_scaled)[0]
    confidence    = float(probabilities[pred_index])
    prediction    = _encoder.inverse_transform([pred_index])[0]
    severity      = get_severity(prediction, confidence)

    # SHAP explanation
    shap_top5 = []
    try:
        import shap
        explainer  = shap.TreeExplainer(_model)
        shap_values = explainer.shap_values(values_scaled)
        names = feature_names or list(features.keys())
        if isinstance(shap_values, list):
            sv = shap_values[pred_index][0]
        else:
            sv = shap_values[0]
        pairs = sorted(zip(names, sv.tolist()), key=lambda x: abs(x[1]), reverse=True)[:5]
        shap_top5 = [{"feature": n, "impact": round(float(v), 4)} for n, v in pairs]
    except Exception as e:
        log.warning(f"SHAP failed: {e}")

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "severity":   severity,
        "shap_top5":  shap_top5,
    }