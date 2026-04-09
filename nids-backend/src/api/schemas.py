"""
schemas.py — Pydantic Schemas (PRODUCTION)
============================================
Defines response schemas for the API.
Note: The predict route now accepts raw JSON directly (not via PredictRequest)
to support CICIDS2017 feature names that contain special characters like '/'.
PredictRequest is kept for documentation/reference but not used in the route.
"""
from typing import Optional, List, Dict
from pydantic import BaseModel
class PredictRequestDoc(BaseModel):
    """
    Documentation schema — shows the expected feature names.
    Not used as a route parameter (raw JSON is parsed directly).
    """
    model_config = {"extra": "allow"}
class SHAPItem(BaseModel):
    feature: str
    value:   float
class PredictResponse(BaseModel):
    alert_id:   int
    prediction: str
    confidence: float
    severity:   str
    source_ip:  Optional[str] = None
    shap_top5:  List[SHAPItem] = []
    timestamp:  str
class AlertResponse(BaseModel):
    id:             int
    timestamp:      Optional[str]
    source_ip:      Optional[str]
    destination_ip: Optional[str]
    src_port:       Optional[int]
    dst_port:       Optional[int]
    prediction:     str
    confidence:     float
    severity:       str
    shap_json:      Optional[str]
class StatsResponse(BaseModel):
    total_flows:         int
    total_attacks:       int
    benign_count:        int
    attacks_by_type:     Dict[str, int]
    attacks_by_severity: Dict[str, int]
    uptime_seconds:      float
class HealthResponse(BaseModel):
    status: str
    db:     str
    model:  str
