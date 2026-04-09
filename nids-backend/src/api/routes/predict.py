"""
routes/predict.py — POST /api/predict (PRODUCTION)
====================================================
Accepts feature vectors with CICIDS2017 column names directly from
the FlowExtractor/sniffer pipeline or from manual API calls.

After running inference:
  1. Saves result to database
  2. Broadcasts attack alerts via WebSocket to all connected dashboards
"""

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.models import Alert
from src.api.schemas import PredictResponse, SHAPItem

log = logging.getLogger(__name__)
router = APIRouter()

_predict_fn = None

def _get_predict():
    global _predict_fn
    if _predict_fn is None:
        try:
            from src.model.predict import predict
            _predict_fn = predict
        except Exception as e:
            log.warning(f"Could not load model: {e}")
    return _predict_fn


# ── The EXACT 52 features the trained model expects ───────────────────────────
# These MUST match the output of src/features/extractor.py (FlowExtractor).
# Any feature sent by the sniffer that is not in this list is ignored.
# Any feature in this list not sent by the sniffer defaults to 0.0.
EXPECTED_FEATURES = [
    'Destination Port',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Length of Fwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Bwd Packet Length Max',
    'Bwd Packet Length Min',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Bwd IAT Total',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd Header Length',
    'Bwd Header Length',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'Packet Length Variance',
    'FIN Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'Average Packet Size',
    'Subflow Fwd Bytes',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'act_data_pkt_fwd',
    'min_seg_size_forward',
    'Active Mean',
    'Active Max',
    'Active Min',
    'Idle Mean',
    'Idle Max',
    'Idle Min',
]  # 52 features — matches FlowExtractor.CICIDS_FEATURES exactly


@router.post("/predict", response_model=PredictResponse)
async def predict_flow(
    request_obj: Request,
    db: Session = Depends(get_db),
):
    """
    Accept a feature vector and run ML inference.

    The request body is a flat JSON dict. It can contain:
    - CICIDS2017 feature names (e.g. 'Flow Duration', 'Flow Bytes/s')
    - Metadata keys prefixed with '_' (e.g. '_source_ip', '_dst_port')

    Metadata keys are stripped before inference; only the 52 model features
    are passed to the predict function.
    """
    predict_fn = _get_predict()
    if predict_fn is None:
        raise HTTPException(status_code=503, detail="ML model not loaded.")

    # Parse raw JSON body (not via Pydantic schema — allows any keys)
    try:
        raw = await request_obj.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Extract metadata (underscore-prefixed keys)
    source_ip      = str(raw.get("_source_ip", "")) or "unknown"
    destination_ip = str(raw.get("_destination_ip", "")) or "unknown"
    src_port       = int(raw.get("_src_port", 0) or 0)
    dst_port       = int(raw.get("_dst_port", 0) or 0)

    # Build feature dict using ONLY the expected CICIDS2017 feature names
    # If a feature is present in the incoming dict → use it
    # If not → default to 0.0 (this handles partial feature vectors)
    features = {}
    missing_count = 0
    for feat_name in EXPECTED_FEATURES:
        if feat_name in raw:
            try:
                features[feat_name] = float(raw[feat_name])
            except (ValueError, TypeError):
                features[feat_name] = 0.0
        else:
            features[feat_name] = 0.0
            missing_count += 1

    if missing_count > 0:
        log.debug(f"Feature vector has {missing_count}/52 missing features (defaulted to 0)")

    if missing_count == 52:
        log.warning("ALL 52 features are missing — this looks like an invalid request")

    # Run inference
    try:
        result = predict_fn(features, feature_names=EXPECTED_FEATURES)
    except Exception as e:
        log.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    prediction = result["prediction"]
    confidence = result["confidence"]
    severity   = result["severity"]
    shap_top5  = result.get("shap_top5", [])

    # Save to database
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

    # ── Broadcast to WebSocket clients ────────────────────────────────────
    if prediction != "BENIGN":
        try:
            ws_manager = request_obj.app.state.ws_manager
            await ws_manager.broadcast({
                "id":          alert.id,
                "timestamp":   alert.timestamp.isoformat(),
                "src_ip":      source_ip,
                "source_ip":   source_ip,
                "attack_type": prediction,
                "prediction":  prediction,
                "severity":    severity,
                "confidence":  confidence,
                "shap_top5":   shap_top5,
            })
        except Exception as e:
            log.warning(f"WS broadcast failed: {e}")

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