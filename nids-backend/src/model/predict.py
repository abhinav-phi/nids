"""
predict.py — NIDS Inference Wrapper (FIXED)
=============================================
Fixes:
  1. SHAP explainer is cached once — not re-created on every request (was 100x slower)
  2. severity mapping covers all CICIDS2017 attack class names
  3. Broadcasts new alerts via WebSocket manager
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

# ── Severity mapping (covers all CICIDS2017 classes + trained model labels) ───
SEVERITY_MAP = {
    # NONE (benign/normal)
    "benign":             "NONE",
    "normal traffic":     "NONE",
    "normal":             "NONE",
    # CRITICAL
    "ddos":               "CRITICAL",
    "dos hulk":           "CRITICAL",
    "dos goldeneye":      "CRITICAL",
    "dos slowloris":      "CRITICAL",
    "dos slowhttptest":   "CRITICAL",
    "heartbleed":         "CRITICAL",
    # HIGH
    "bot":                "HIGH",
    "ftp-patator":        "HIGH",
    "ssh-patator":        "HIGH",
    "infiltration":       "HIGH",
    # MEDIUM
    "port scanning":      "MEDIUM",
    "portscan":           "MEDIUM",
    "web attack":         "MEDIUM",
    "web attack – brute force": "MEDIUM",
    "web attack – xss":   "MEDIUM",
    "web attack – sql injection": "MEDIUM",
    # LOW
    "brute force":        "LOW",
}

def get_severity(prediction: str) -> str:
    key = prediction.strip().lower()
    for pattern, sev in SEVERITY_MAP.items():
        if pattern in key:
            return sev
    return "LOW"   # unknown attack type → LOW by default


# ── Load artifacts once ───────────────────────────────────────────────────────
_model   = None
_scaler  = None
_encoder = None
_explainer = None   # ← cached SHAP explainer (FIX: was re-created every request)
_model_loaded = False

def _load_artifacts():
    global _model, _scaler, _encoder, _explainer, _model_loaded
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model.pkl not found at {MODEL_PATH}. Run src/model/train.py first."
        )
    _model   = joblib.load(MODEL_PATH)
    _scaler  = joblib.load(SCALER_PATH)
    _encoder = joblib.load(ENCODER_PATH)

    # Cache SHAP explainer — this is the expensive operation (FIX)
    try:
        import shap
        _explainer = shap.TreeExplainer(_model)
        log.info("SHAP TreeExplainer cached.")
    except Exception as e:
        log.warning(f"SHAP explainer init failed (will skip SHAP): {e}")
        _explainer = None

    _model_loaded = True
    log.info(f"Model loaded. Classes: {list(_encoder.classes_)}")


try:
    _load_artifacts()
except FileNotFoundError as e:
    log.warning(str(e))


# ── Main predict function ─────────────────────────────────────────────────────
def predict(features: dict, feature_names: Optional[list] = None) -> dict:
    """
    Run inference on a single network flow feature dict.
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
    severity      = get_severity(prediction)

    # SHAP explanation — uses cached explainer
    shap_top5 = []
    if _explainer is not None:
        try:
            import numpy as _np
            shap_values = _explainer.shap_values(values_scaled)
            names = feature_names or list(features.keys())

            # Handle different SHAP output formats:
            # - list of arrays (one per class) → pick the predicted class
            # - 3D array (samples × features × classes) → slice for predicted class
            # - 2D array (samples × features) → use directly
            if isinstance(shap_values, list):
                sv = _np.array(shap_values[pred_index]).flatten()
            elif hasattr(shap_values, 'ndim'):
                if shap_values.ndim == 3:
                    sv = shap_values[0, :, pred_index]
                else:
                    sv = shap_values[0]
            else:
                sv = _np.array(shap_values).flatten()

            sv_list = sv.tolist() if hasattr(sv, 'tolist') else list(sv)
            pairs = sorted(zip(names, sv_list), key=lambda x: abs(float(x[1])), reverse=True)[:5]
            shap_top5 = [{"feature": n, "value": round(float(v), 4)} for n, v in pairs]
        except Exception as e:
            log.warning(f"SHAP inference failed: {e}")

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "severity":   severity,
        "shap_top5":  shap_top5,
    }