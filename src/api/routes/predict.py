"""
routes/predict.py — POST /api/predict
=======================================
Receives a feature dictionary from the sniffer (or any client),
runs ML inference, saves the alert to the database,
and returns the prediction with SHAP explanation.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.models import Alert
from src.api.schemas import PredictRequest, PredictResponse, SHAPItem

log = logging.getLogger(__name__)

router = APIRouter()

# We import the predict function lazily (inside the route) so the API starts
# even if model.pkl doesn't exist yet — it will return a 503 instead of crashing.
_predict_fn = None

def _get_predict():
    """Load the predict function once and cache it."""
    global _predict_fn
    if _predict_fn is None:
        try:
            from src.model.predict import predict
            _predict_fn = predict
        except Exception as e:
            log.warning(f"Could not load model: {e}")
    return _predict_fn


@router.post("/predict", response_model=PredictResponse)
def predict_flow(
    request: PredictRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Accept a network flow's feature vector, run the ML model,
    save the result to the database, and return the prediction.

    The sniffer sends requests here automatically.
    You can also test it manually via the Swagger UI at /docs.
    """
    predict_fn = _get_predict()
    if predict_fn is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run src/model/train.py first."
        )

    # ── Extract metadata fields (start with underscore) ───────────────────────
    raw = request.model_dump(mode="python")
    source_ip      = str(raw.get("_source_ip", "")) or "unknown"
    destination_ip = str(raw.get("_destination_ip", "")) or "unknown"
    src_port       = int(raw.get("_src_port", 0) or 0)
    dst_port       = int(raw.get("_dst_port", 0) or 0)

    # ── Build feature dict (no metadata) ─────────────────────────────────────
    features = request.to_feature_dict()

        # Keep only the 52 features the model was trained on
    EXPECTED = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets',
        'Total Length of Fwd Packets', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
        'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
        'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
        'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
        'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
        'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size',
        'Subflow Fwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
        'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Max',
        'Active Min', 'Idle Mean', 'Idle Max', 'Idle Min'
    ]
    features = {k: float(features.get(k, 0.0)) for k in EXPECTED}

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        result = predict_fn(features)
    except Exception as e:
        log.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    prediction = result["prediction"]
    confidence = result["confidence"]
    severity   = result["severity"]
    shap_top5  = result.get("shap_top5", [])

    # ── Save to database ──────────────────────────────────────────────────────
    alert = Alert(
        timestamp      = datetime.utcnow(),
        source_ip      = source_ip,
        destination_ip = destination_ip,
        src_port       = src_port,
        dst_port       = dst_port,
        prediction     = prediction,
        confidence     = confidence,
        severity       = severity,
        shap_json      = json.dumps(shap_top5),
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)

    # ── Log alerts to console ─────────────────────────────────────────────────
    if prediction != "BENIGN":
        log.warning(
            f"[{severity}] {source_ip} → {destination_ip}  "
            f"{prediction}  ({confidence*100:.1f}%)"
        )

    return PredictResponse(
        alert_id   = alert.id,
        prediction = prediction,
        confidence = confidence,
        severity   = severity,
        source_ip  = source_ip,
        shap_top5  = [SHAPItem(**s) for s in shap_top5],
        timestamp  = alert.timestamp.isoformat(),
    )