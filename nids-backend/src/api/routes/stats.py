"""
routes/stats.py — GET /api/stats
===================================
Returns overall system statistics aggregated from the alerts table.
Also exposes GET /api/ip-leaderboard for the top attacker IPs.
"""

import time
import logging
from typing import List
from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from src.api.database import get_db
from src.api.models import Alert
from src.api.schemas import StatsResponse

log = logging.getLogger(__name__)
router = APIRouter()

# Record when the API process started
_start_time = time.time()


@router.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """
    Return overall system stats:
    - total flows analyzed (all predictions including BENIGN)
    - total attacks detected
    - attack counts broken down by type
    - attack counts broken down by severity
    - system uptime in seconds
    """
    # Total rows = total flows processed
    total_flows = db.query(func.count(Alert.id)).scalar() or 0

    # Rows where prediction is not BENIGN = actual attacks
    total_attacks = (
        db.query(func.count(Alert.id))
        .filter(Alert.prediction != "BENIGN")
        .scalar() or 0
    )

    benign_count = total_flows - total_attacks

    # Count per attack type  e.g. {"DDoS": 42, "PortScan": 15, ...}
    type_rows = (
        db.query(Alert.prediction, func.count(Alert.id))
        .filter(Alert.prediction != "BENIGN")
        .group_by(Alert.prediction)
        .all()
    )
    attacks_by_type = {row[0]: row[1] for row in type_rows}

    # Count per severity level
    sev_rows = (
        db.query(Alert.severity, func.count(Alert.id))
        .filter(Alert.prediction != "BENIGN")
        .group_by(Alert.severity)
        .all()
    )
    attacks_by_severity = {row[0]: row[1] for row in sev_rows}

    return StatsResponse(
        total_flows         = total_flows,
        total_attacks       = total_attacks,
        benign_count        = benign_count,
        attacks_by_type     = attacks_by_type,
        attacks_by_severity = attacks_by_severity,
        uptime_seconds      = round(time.time() - _start_time, 1),
    )


@router.get("/ip-leaderboard")
def ip_leaderboard(
    db:    Session = Depends(get_db),
    limit: int     = 10,
):
    """
    Return the top N source IPs by attack count.
    Used by the dashboard's 'Top Attackers' table.
    """
    rows = (
        db.query(
            Alert.source_ip,
            func.count(Alert.id).label("attack_count"),
            func.max(Alert.timestamp).label("last_seen"),
        )
        .filter(Alert.prediction != "BENIGN")
        .group_by(Alert.source_ip)
        .order_by(desc("attack_count"))
        .limit(limit)
        .all()
    )

    return [
        {
            "rank":         i + 1,
            "source_ip":    row.source_ip,
            "attack_count": row.attack_count,
            "last_seen":    row.last_seen.isoformat() if row.last_seen else None,
        }
        for i, row in enumerate(rows)
    ]