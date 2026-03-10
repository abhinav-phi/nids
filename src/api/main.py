"""
main.py — FastAPI Application Entry Point
==========================================
Creates the app, registers all routes, sets up the database,
and handles startup/shutdown events.

Run:
    uvicorn src.api.main:app --reload --port 8000

Then open:
    http://localhost:8000/docs      ← interactive Swagger UI
    http://localhost:8000/health    ← quick health check
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
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


# ── Startup / shutdown lifecycle ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup and once at shutdown.
    Creates all database tables if they don't already exist.
    """
    log.info("Starting NIDS API ...")

    # Create tables (safe to call multiple times — skips existing tables)
    Base.metadata.create_all(bind=engine)
    log.info("Database tables ready.")

    # Warm up: try loading the model now so first request is fast
    try:
        from src.model.predict import predict as _p
        log.info("ML model loaded successfully.")
    except FileNotFoundError:
        log.warning(
            "model.pkl not found. Run src/model/train.py first. "
            "POST /api/predict will return 503 until the model is trained."
        )

    yield   # app runs here

    log.info("Shutting down NIDS API.")


# ── App instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "NIDS — Network Intrusion Detection API",
    description = (
        "ML-powered API that classifies network flows as BENIGN or one of "
        "13 attack types (DDoS, PortScan, Brute Force, Botnet, etc.).\n\n"
        "**Dataset:** CICIDS2017 | **Model:** XGBoost | **Explainability:** SHAP"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── CORS — allow the React dashboard (localhost:3000) to call this API ────────

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Register route modules ────────────────────────────────────────────────────

app.include_router(predict.router, prefix="/api", tags=["Prediction"])
app.include_router(alerts.router,  prefix="/api", tags=["Alerts"])
app.include_router(stats.router,   prefix="/api", tags=["Stats"])


# ── Health check endpoint ─────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    """
    Quick check to verify the API, database, and ML model are all working.
    Used by Docker healthchecks and monitoring tools.
    """
    # Check DB connection
    db_status = "ok"
    try:
        db = SessionLocal()
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db.close()
    except Exception as e:
        db_status = f"error: {str(e)}"

    # Check model is loaded
    model_status = "ok"
    try:
        from src.model.predict import _model_loaded
        if not _model_loaded:
            model_status = "not loaded — run train.py"
    except Exception:
        model_status = "not loaded"

    return {
        "status":          "ok",
        "db":              db_status,
        "model":           model_status,
        "uptime_seconds":  round(time.time() - _startup_time, 1),
    }


# ── Root redirect ─────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "NIDS API is running.",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/health",
    }