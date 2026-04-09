"""
routes/alerts.py — GET /api/alerts
=====================================
Returns recent alerts from the database with optional filtering.
"""
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from src.api.database import get_db
from src.api.models import Alert
from src.api.schemas import AlertResponse
log = logging.getLogger(__name__)
router = APIRouter()
@router.get("/alerts", response_model=List[AlertResponse])
def get_alerts(
    db:       Session = Depends(get_db),
    limit:    int     = Query(default=50, ge=1, le=500,
                              description="Max number of alerts to return"),
    offset:   int     = Query(default=0,  ge=0,
                              description="Number of rows to skip (for pagination)"),
    type:     Optional[str] = Query(default=None,
                              description="Filter by attack type, e.g. 'DDoS'"),
    severity: Optional[str] = Query(default=None,
                              description="Filter by severity: LOW/MEDIUM/HIGH/CRITICAL"),
    exclude_benign: bool     = Query(default=True,
                              description="If true, only return attack alerts"),
):
    """
    Return recent alerts, newest first.
    Examples:
        GET /api/alerts                          → last 50 alerts
        GET /api/alerts?type=DDoS                → only DDoS alerts
        GET /api/alerts?severity=CRITICAL        → only critical alerts
        GET /api/alerts?limit=10&offset=10       → page 2
        GET /api/alerts?exclude_benign=false     → include BENIGN flows too
    """
    query = db.query(Alert).order_by(desc(Alert.timestamp))
    if exclude_benign:
        query = query.filter(Alert.prediction != "BENIGN")
    if type:
        query = query.filter(Alert.prediction.ilike(f"%{type}%"))
    if severity:
        query = query.filter(Alert.severity == severity.upper())
    alerts = query.offset(offset).limit(limit).all()
    return [
        AlertResponse(
            id             = a.id,
            timestamp      = a.timestamp.isoformat() if a.timestamp else None,
            source_ip      = a.source_ip,
            destination_ip = a.destination_ip,
            src_port       = a.src_port,
            dst_port       = a.dst_port,
            prediction     = a.prediction,
            confidence     = round(a.confidence, 4) if a.confidence else 0.0,
            severity       = a.severity,
            shap_json      = a.shap_json,
        )
        for a in alerts
    ]
