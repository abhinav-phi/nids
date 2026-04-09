"""
main.py — FastAPI Application Entry Point (PRODUCTION)
========================================================
Features:
  1. ML model loading on startup
  2. WebSocket /ws/live for real-time alert streaming
  3. Integrated network sniffer control via API endpoints
  4. CORS for frontend dev servers
  5. Health check with sniffer status

Run:
    uvicorn src.api.main:app --reload --port 8000

With auto-start sniffer:
    set NIDS_CAPTURE=1
    uvicorn src.api.main:app --port 8000
"""

import os
import time
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.database import engine, Base, SessionLocal
from src.api.routes import predict, alerts, stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_startup_time = time.time()


# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        log.info(f"[WS] Client connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        log.info(f"[WS] Client disconnected. Total: {len(self.active)}")

    async def broadcast(self, message: dict):
        """Send a message to all connected WebSocket clients."""
        data = json.dumps(message)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.active:
                self.active.remove(ws)


ws_manager = ConnectionManager()

# ── Sniffer instance (global, controlled via API) ─────────────────────────────

_sniffer = None


def _get_sniffer():
    """Lazy-init the sniffer instance."""
    global _sniffer
    if _sniffer is None:
        try:
            from src.capture.sniffer import NetworkSniffer
            _sniffer = NetworkSniffer(interface="auto")
            log.info("Sniffer instance created.")
        except Exception as e:
            log.warning(f"Could not create sniffer: {e}")
    return _sniffer


# ── Startup / shutdown lifecycle ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting NIDS API ...")
    Base.metadata.create_all(bind=engine)
    log.info("Database tables ready.")

    try:
        from src.model.predict import predict as _p
        log.info("ML model loaded successfully.")
    except FileNotFoundError:
        log.warning("model.pkl not found. Run src/model/train.py first.")

    # Attach manager to app state so routes can broadcast alerts
    app.state.ws_manager = ws_manager

    # Auto-start sniffer if NIDS_CAPTURE env var is set
    if os.environ.get("NIDS_CAPTURE", "").strip() in ("1", "true", "yes"):
        sniffer = _get_sniffer()
        if sniffer:
            sniffer.start()
            log.info("Sniffer auto-started (NIDS_CAPTURE=1)")

    yield

    # Shutdown: stop sniffer if running
    if _sniffer and _sniffer.is_running():
        _sniffer.stop()
    log.info("Shutting down NIDS API.")


# ── App instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="NIDS — Network Intrusion Detection API",
    description="ML-powered network intrusion detection with real-time packet capture and SHAP explainability.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api", tags=["Prediction"])
app.include_router(alerts.router,  prefix="/api", tags=["Alerts"])
app.include_router(stats.router,   prefix="/api", tags=["Stats"])


# ── WebSocket /ws/live ────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Live alert stream. On connect, sends the last 50 alerts as an initial
    batch, then pushes new alerts in real time as they come in.
    """
    await ws_manager.connect(websocket)

    # Send recent alert history on connection
    try:
        db = SessionLocal()
        from src.api.models import Alert
        from sqlalchemy import desc
        recent = (
            db.query(Alert)
            .filter(Alert.prediction != "BENIGN")
            .order_by(desc(Alert.timestamp))
            .limit(50)
            .all()
        )
        db.close()

        if recent:
            history = []
            for a in reversed(recent):
                history.append({
                    "id":          a.id,
                    "timestamp":   a.timestamp.isoformat() if a.timestamp else "",
                    "src_ip":      a.source_ip or "unknown",
                    "source_ip":   a.source_ip or "unknown",
                    "attack_type": a.prediction,
                    "prediction":  a.prediction,
                    "severity":    a.severity,
                    "confidence":  round(a.confidence or 0, 4),
                    "shap_top5":   json.loads(a.shap_json) if a.shap_json else [],
                })
            await websocket.send_text(json.dumps(history))

    except Exception as e:
        log.warning(f"[WS] Could not send history: {e}")

    # Keep connection alive; ping every 10s
    try:
        while True:
            await asyncio.sleep(10)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# ── Sniffer control endpoints ─────────────────────────────────────────────────

@app.post("/api/sniffer/start", tags=["Sniffer"])
def start_sniffer(interface: Optional[str] = None):
    """
    Start the packet capture sniffer.
    Optionally specify a network interface (default: auto-detect).
    """
    global _sniffer
    sniffer = _get_sniffer()
    if sniffer is None:
        return {"status": "error", "message": "Scapy not available. Install scapy and Npcap."}

    if sniffer.is_running():
        return {"status": "already_running", **sniffer.get_stats()}

    if interface:
        # Re-create sniffer with the specified interface
        from src.capture.sniffer import NetworkSniffer
        _sniffer = NetworkSniffer(interface=interface)
        sniffer = _sniffer

    sniffer.start()
    return {"status": "started", **sniffer.get_stats()}


@app.post("/api/sniffer/stop", tags=["Sniffer"])
def stop_sniffer():
    """Stop the packet capture sniffer."""
    sniffer = _get_sniffer()
    if sniffer is None or not sniffer.is_running():
        return {"status": "not_running"}

    sniffer.stop()
    return {"status": "stopped", **sniffer.get_stats()}


@app.get("/api/sniffer/stats", tags=["Sniffer"])
def sniffer_stats():
    """Get current sniffer statistics."""
    sniffer = _get_sniffer()
    if sniffer is None:
        return {"status": "unavailable", "message": "Scapy not installed"}
    return {"status": "ok", **sniffer.get_stats()}


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    db_status = "ok"
    try:
        db = SessionLocal()
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db.close()
    except Exception as e:
        db_status = f"error: {str(e)}"

    model_status = "ok"
    try:
        from src.model.predict import _model_loaded
        if not _model_loaded:
            model_status = "not loaded — run train.py"
    except Exception:
        model_status = "not loaded"

    sniffer_status = "not initialized"
    if _sniffer:
        sniffer_status = "running" if _sniffer.is_running() else "stopped"

    return {
        "status":         "ok",
        "db":             db_status,
        "model":          model_status,
        "sniffer":        sniffer_status,
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "ws_clients":     len(ws_manager.active),
    }


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "NIDS API v2.0 — Real-time Network Intrusion Detection",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/health",
        "ws":      "ws://localhost:8000/ws/live",
        "sniffer": {
            "start": "POST /api/sniffer/start",
            "stop":  "POST /api/sniffer/stop",
            "stats": "GET  /api/sniffer/stats",
        }
    }