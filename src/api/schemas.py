"""
schemas.py — Pydantic Request & Response Schemas
==================================================
Pydantic validates all incoming and outgoing JSON automatically.
FastAPI uses these schemas to generate the Swagger UI docs at /docs.
"""

import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ── Request: incoming feature vector from sniffer or test client ──────────────

class PredictRequest(BaseModel):
    """
    Feature dictionary sent to POST /api/predict.
    All values are floats (network flow features from FlowExtractor).
    Extra fields with underscore prefix are metadata (source_ip, etc.)
    and are passed through but not fed to the ML model.
    """

    # Core flow features (must match FlowExtractor output)
    flow_duration:            float = 0.0
    total_fwd_packets:        float = 0.0
    total_bwd_packets:        float = 0.0
    total_fwd_bytes:          float = 0.0
    total_bwd_bytes:          float = 0.0
    packet_length_mean:       float = 0.0
    packet_length_max:        float = 0.0
    packet_length_min:        float = 0.0
    packet_length_std:        float = 0.0
    fwd_packet_length_mean:   float = 0.0
    fwd_packet_length_max:    float = 0.0
    fwd_packet_length_min:    float = 0.0
    fwd_packet_length_std:    float = 0.0
    bwd_packet_length_mean:   float = 0.0
    bwd_packet_length_max:    float = 0.0
    bwd_packet_length_min:    float = 0.0
    bwd_packet_length_std:    float = 0.0
    flow_iat_mean:            float = 0.0
    flow_iat_max:             float = 0.0
    flow_iat_min:             float = 0.0
    flow_iat_std:             float = 0.0
    packets_per_second:       float = 0.0
    bytes_per_second:         float = 0.0
    syn_flag_count:           float = 0.0
    fin_flag_count:           float = 0.0
    rst_flag_count:           float = 0.0
    ack_flag_count:           float = 0.0
    psh_flag_count:           float = 0.0
    urg_flag_count:           float = 0.0

    # Optional metadata (added by sniffer, used for DB logging only)
    _source_ip:      Optional[float] = None
    _destination_ip: Optional[float] = None
    _src_port:       Optional[float] = None
    _dst_port:       Optional[float] = None

    model_config = {"extra": "allow"}   # allow extra fields (metadata, unknown features)

    def to_feature_dict(self) -> dict:
        """
        Return only the ML features (no underscore-prefix metadata fields).
        This is the dict passed to the ML model.
        """
        data = self.model_dump()
        return {k: v for k, v in data.items() if not k.startswith("_")}


# ── SHAP explanation item ─────────────────────────────────────────────────────

class SHAPItem(BaseModel):
    feature: str
    impact:  float


# ── Response: prediction result ───────────────────────────────────────────────

class PredictResponse(BaseModel):
    """Returned by POST /api/predict."""
    alert_id:   int
    prediction: str
    confidence: float
    severity:   str
    source_ip:  Optional[str] = None
    shap_top5:  List[SHAPItem] = []
    timestamp:  str


# ── Alert row (for GET /api/alerts list) ─────────────────────────────────────

class AlertResponse(BaseModel):
    """One alert entry returned by GET /api/alerts."""
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


# ── Stats response ────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    """Returned by GET /api/stats."""
    total_flows:       int
    total_attacks:     int
    benign_count:      int
    attacks_by_type:   Dict[str, int]
    attacks_by_severity: Dict[str, int]
    uptime_seconds:    float


# ── Health check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    db:     str
    model:  str