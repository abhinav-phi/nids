# %% [markdown]
# ## 🗄️ Step 3 — Persistence + validation helpers
#
# Ports of:
# - `src/api/database.py` (SQLite + WAL pragmas) and `src/api/models.py` (`Alert`)
# - `src/api/routes/predict.py` validation (`_validate_features`, `_coerce_port`,
#   `_validate_metadata_ip`, `_sanitize_shap`) — no silent coercion, ≤8 missing
#   features tolerated, non-negative finite values only, real IPv4/IPv6 metadata.

# %%
import math
import time
import ipaddress
import threading
import asyncio
import json as _json
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, event, Column, Integer, String, Float, DateTime, Text,
    desc, func, text as sql_text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ── database.py ──────────────────────────────────────────────────────────
DB_URL = f"sqlite:////content/nids_colab.db"
_engine = create_engine(DB_URL, connect_args={"check_same_thread": False}, pool_pre_ping=True)

@event.listens_for(_engine, "connect")
def _sqlite_pragmas(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
Base = declarative_base()

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def iso_utc(dt) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

# ── models.py ────────────────────────────────────────────────────────────
class Alert(Base):
    __tablename__ = "alerts"
    id             = Column(Integer, primary_key=True, index=True)
    timestamp      = Column(DateTime, default=utcnow, index=True)
    source_ip      = Column(String(45), index=True)
    destination_ip = Column(String(45))
    src_port       = Column(Integer, nullable=True)
    dst_port       = Column(Integer, nullable=True)
    prediction     = Column(String(50), index=True)
    confidence     = Column(Float)
    severity       = Column(String(20), index=True)
    shap_json      = Column(Text, nullable=True)

Base.metadata.create_all(bind=_engine)
print(f"SQLite ready → {DB_URL}")

# ── routes/predict.py validation ─────────────────────────────────────────
EXPECTED_FEATURES      = list(CICIDS_FEATURES)
MAX_INVALID_FEATURES   = 8
MAX_ABS_FEATURE_VALUE  = 1e15
MAX_METADATA_IP_LENGTH = 45

def _validate_features(raw: dict):
    """Classify each expected feature as valid / missing / invalid.
    Missing → 0.0 (tolerated up to 8); non-numeric, non-finite, negative or
    >1e15 values are invalid and reject the request (no silent coercion)."""
    features, missing, invalid = {}, 0, []
    for feat_name in EXPECTED_FEATURES:
        if feat_name not in raw:
            features[feat_name] = 0.0
            missing += 1
            continue
        try:
            value = float(raw[feat_name])
        except (ValueError, TypeError):
            invalid.append(feat_name)
            features[feat_name] = 0.0
            continue
        if (not math.isfinite(value) or value < 0 or abs(value) > MAX_ABS_FEATURE_VALUE):
            invalid.append(feat_name)
            features[feat_name] = 0.0
            continue
        features[feat_name] = value
    return features, missing, invalid

def _coerce_port(value) -> int:
    try:
        as_float = float(value)
    except (ValueError, TypeError):
        return 0
    if not math.isfinite(as_float):
        return 0
    port = int(as_float)
    return port if 0 <= port <= 65535 else 0

def _validate_metadata_ip(value, field: str) -> str:
    """Metadata IPs must be real IPv4/IPv6 — blocks log forging and
    data-borne prompt injection (same rule as production)."""
    if value is None:
        return "unknown"
    text_value = str(value).strip()
    if not text_value or text_value.lower() == "unknown":
        return "unknown"
    if len(text_value) > MAX_METADATA_IP_LENGTH:
        raise HTTPException(422, detail={"message": f"{field} exceeds max IP length.", "field": field})
    if any(not ch.isprintable() for ch in text_value):
        raise HTTPException(422, detail={"message": f"{field} contains control characters.", "field": field})
    try:
        ipaddress.ip_address(text_value)
    except ValueError:
        raise HTTPException(422, detail={"message": f"{field} must be a valid IPv4/IPv6 address.", "field": field})
    return text_value

def _sanitize_shap(shap_top5: list) -> list:
    """Coerce non-finite SHAP values to 0.0 → browser-safe JSON."""
    safe = []
    for item in shap_top5 or []:
        try:
            value = float(item.get("value", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        safe.append({"feature": str(item.get("feature", "")), "value": round(value, 4)})
    return safe

print("Validation helpers ready ✔")

# %% [markdown]
# ## 🌐 Step 4 — The FastAPI app (port of `src/api/main.py` + routes)
#
# Endpoints (same contract as the production backend):
#
# | Method | Endpoint | Purpose |
# |---|---|---|
# | GET | `/health` | liveness: db, model, uptime |
# | POST | `/api/predict` | classify one flow (52 CICIDS features + `_source_ip` etc.) |
# | GET | `/api/alerts` | paginated alert history (filters: `type`, `severity`, `exclude_benign`) |
# | GET | `/api/stats` | totals, attacks by type / severity, uptime |
# | GET | `/api/ip-leaderboard` | top attacking source IPs |
# | WS | `/ws/live` | live attack broadcast (last-50 history on connect, ping every 10 s) |
#
# > The sniffer endpoints (`/api/sniffer/*`) are **not** portable to Colab — a VM
# > has no raw-packet access to your LAN. The offline replay (Step 6) feeds the
# > same pipeline instead.

# %%
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool
from typing import List, Optional

_START_TIME = time.time()
RATE_LIMIT_PER_MINUTE = 300
MAX_BODY_BYTES = 1_000_000
_rate_hits: dict = defaultdict(list)
OPEN_PATHS = {"/", "/health", "/docs", "/openapi.json"}


class ConnectionManager:
    """Bounded WebSocket fan-out with per-client send timeout
    (ports main.py::ConnectionManager)."""

    def __init__(self, max_clients: int = 20, send_timeout_s: float = 5.0):
        self.active: List[WebSocket] = []
        self.max_clients = max_clients
        self.send_timeout_s = send_timeout_s

    def can_accept(self) -> bool:
        return len(self.active) < self.max_clients

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def _safe_send(self, ws: WebSocket, data: str) -> bool:
        try:
            await asyncio.wait_for(ws.send_text(data), timeout=self.send_timeout_s)
            return True
        except Exception:
            self.disconnect(ws)
            return False

    async def broadcast(self, message: dict):
        if not self.active:
            return
        data = _json.dumps(message)
        await asyncio.gather(*(self._safe_send(ws, data) for ws in list(self.active)))


ws_manager = ConnectionManager()
app = FastAPI(title="NIDS — Network Intrusion Detection API (Colab)", version="2.0.0-colab")


@app.middleware("http")
async def security_middleware(request, call_next):
    path = request.url.path
    if path not in OPEN_PATHS:
        content_length = request.headers.get("content-length", "")
        if content_length.isdigit() and int(content_length) > MAX_BODY_BYTES:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=413, content={"detail": "Request body too large."})
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = [t for t in _rate_hits[client_ip] if t > now - 60]
        window.append(now)
        _rate_hits[client_ip] = window
        if len(window) > RATE_LIMIT_PER_MINUTE:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})
    return await call_next(request)


@app.get("/health")
def health_check():
    db_ok = False
    try:
        db = SessionLocal()
        try:
            db.execute(sql_text("SELECT 1"))
            db_ok = True
        finally:
            db.close()
    except Exception:
        pass
    return {
        "status": "ok",
        "db": "ok" if db_ok else "error",
        "model": "ok" if _model_loaded else "not loaded",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "ws_clients": len(ws_manager.active),
    }


@app.post("/api/predict")
async def predict_flow_route(request: Request):
    try:
        raw = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object.")

    source_ip      = _validate_metadata_ip(raw.get("_source_ip"), "_source_ip")
    destination_ip = _validate_metadata_ip(raw.get("_destination_ip"), "_destination_ip")
    src_port       = _coerce_port(raw.get("_src_port", 0))
    dst_port       = _coerce_port(raw.get("_dst_port", 0))

    features, missing_count, invalid_names = _validate_features(raw)
    if missing_count == len(EXPECTED_FEATURES):
        raise HTTPException(status_code=400,
            detail="Invalid request: none of the 52 CICIDS2017 features were provided.")
    if invalid_names:
        raise HTTPException(status_code=422, detail={
            "message": "Feature values must be finite, non-negative numbers within sane bounds.",
            "invalid_features": invalid_names[:20],
        })
    if missing_count > MAX_INVALID_FEATURES:
        raise HTTPException(status_code=422, detail={
            "message": f"{missing_count} of {len(EXPECTED_FEATURES)} features are missing "
                       f"(max tolerated: {MAX_INVALID_FEATURES}).",
            "missing": missing_count,
        })

    try:
        result = await run_in_threadpool(predict_flow, features, EXPECTED_FEATURES)
    except Exception:
        raise HTTPException(status_code=500, detail="Inference failed. Check server logs.")

    prediction, confidence = result["prediction"], result["confidence"]
    severity, shap_top5    = result["severity"], _sanitize_shap(result.get("shap_top5", []))

    def _persist():
        db = SessionLocal()
        try:
            alert = Alert(
                timestamp=utcnow(), source_ip=source_ip, destination_ip=destination_ip,
                src_port=src_port, dst_port=dst_port, prediction=prediction,
                confidence=confidence, severity=severity,
                shap_json=_json.dumps(shap_top5, allow_nan=False),
            )
            db.add(alert)
            db.commit()
            db.refresh(alert)
            return alert
        finally:
            db.close()

    alert = await run_in_threadpool(_persist)

    if not is_benign(prediction):
        await ws_manager.broadcast({
            "id": alert.id,
            "timestamp": iso_utc(alert.timestamp),
            "src_ip": source_ip,
            "source_ip": source_ip,
            "attack_type": prediction,
            "prediction": prediction,
            "severity": severity,
            "confidence": confidence,
            "shap_top5": shap_top5,
            "missing_features": missing_count,
        })

    return {
        "alert_id": alert.id,
        "prediction": prediction,
        "confidence": confidence,
        "severity": severity,
        "source_ip": source_ip,
        "shap_top5": shap_top5,
        "timestamp": iso_utc(alert.timestamp) or "",
        "missing_features": missing_count,
    }


@app.get("/api/alerts")
def get_alerts(
    limit: int = 50, offset: int = 0,
    type: Optional[str] = None, severity: Optional[str] = None,
    exclude_benign: bool = True,
):
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    query = SessionLocal().query(Alert).order_by(desc(Alert.timestamp), desc(Alert.id))
    db = query.session
    try:
        if exclude_benign:
            query = query.filter(Alert.prediction.notin_(BENIGN_LABELS))
        if type:
            term = type.strip()[:100].replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            query = query.filter(Alert.prediction.ilike(f"%{term}%", escape="\\"))
        if severity:
            query = query.filter(Alert.severity == severity.upper())
        rows = query.offset(offset).limit(limit).all()
        return [{
            "id": a.id, "timestamp": iso_utc(a.timestamp),
            "source_ip": a.source_ip, "destination_ip": a.destination_ip,
            "src_port": a.src_port, "dst_port": a.dst_port,
            "prediction": a.prediction,
            "confidence": round(a.confidence, 4) if a.confidence else 0.0,
            "severity": a.severity, "shap_json": a.shap_json,
        } for a in rows]
    finally:
        db.close()


@app.get("/api/stats")
def get_stats():
    db = SessionLocal()
    try:
        is_attack = Alert.prediction.notin_(BENIGN_LABELS)
        total_flows   = db.query(func.count(Alert.id)).scalar() or 0
        total_attacks = db.query(func.count(Alert.id)).filter(is_attack).scalar() or 0
        by_type = dict(db.query(Alert.prediction, func.count(Alert.id))
                       .filter(is_attack).group_by(Alert.prediction).all())
        by_sev  = dict(db.query(Alert.severity, func.count(Alert.id))
                       .filter(is_attack).group_by(Alert.severity).all())
        return {
            "total_flows": total_flows,
            "total_attacks": total_attacks,
            "benign_count": total_flows - total_attacks,
            "attacks_by_type": by_type,
            "attacks_by_severity": by_sev,
            "uptime_seconds": round(time.time() - _START_TIME, 1),
        }
    finally:
        db.close()


@app.get("/api/ip-leaderboard")
def ip_leaderboard(limit: int = 10):
    limit = max(1, min(int(limit), 100))
    db = SessionLocal()
    try:
        rows = (db.query(Alert.source_ip,
                         func.count(Alert.id).label("attack_count"),
                         func.max(Alert.timestamp).label("last_seen"))
                .filter(Alert.prediction.notin_(BENIGN_LABELS))
                .group_by(Alert.source_ip)
                .order_by(desc("attack_count"), desc("last_seen"))
                .limit(limit).all())
        return [{
            "rank": i + 1, "source_ip": r.source_ip,
            "attack_count": r.attack_count, "last_seen": iso_utc(r.last_seen),
        } for i, r in enumerate(rows)]
    finally:
        db.close()


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Live attack stream: last-50 history on connect, then real-time
    broadcasts, ping every 10 s, client cap enforced."""
    if not ws_manager.can_accept():
        await websocket.accept()
        await websocket.close(code=1013, reason="Too many connected clients")
        return
    await ws_manager.connect(websocket)
    try:
        db = SessionLocal()
        try:
            recent = (db.query(Alert)
                      .filter(Alert.prediction.notin_(BENIGN_LABELS))
                      .order_by(desc(Alert.timestamp), desc(Alert.id))
                      .limit(50).all())
        finally:
            db.close()
        if recent:
            history = [{
                "id": a.id, "timestamp": iso_utc(a.timestamp) or "",
                "src_ip": a.source_ip or "unknown",
                "attack_type": a.prediction, "severity": a.severity,
                "confidence": round(a.confidence or 0, 4),
                "shap_top5": _json.loads(a.shap_json) if a.shap_json else [],
            } for a in reversed(recent)]
            await websocket.send_text(_json.dumps(history))
        while True:
            await asyncio.sleep(10)
            await websocket.send_text(_json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


@app.get("/")
def root():
    return {
        "message": "NIDS API v2.0 (Colab) — Real-time Network Intrusion Detection",
        "docs": f"{API_BASE}/docs",
        "health": f"{API_BASE}/health",
        "ws": f"ws://<public-url>/ws/live",
    }


print("FastAPI app ready ✔")
