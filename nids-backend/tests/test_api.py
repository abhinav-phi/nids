"""
test_api.py — API endpoint tests
Run with:  pytest tests/test_api.py -v

Uses FastAPI's TestClient so no real server needs to be running.
Uses SQLite in-memory DB so no PostgreSQL is needed.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Use SQLite for tests — no PostgreSQL needed
os.environ["DATABASE_URL"] = "sqlite:///./test_nids.db"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.database import Base, get_db
from src.api.main import app

# ── In-memory SQLite test database ───────────────────────────────────────────
TEST_DB_URL = "sqlite:///./test_nids.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSession  = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)

def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# ── Sample feature payload ────────────────────────────────────────────────────
SAMPLE_FEATURES = {
    "flow_duration":          1234567.0,
    "total_fwd_packets":      8.0,
    "total_bwd_packets":      6.0,
    "total_fwd_bytes":        3456.0,
    "total_bwd_bytes":        1200.0,
    "packet_length_mean":     300.0,
    "packet_length_max":      1460.0,
    "packet_length_min":      52.0,
    "packet_length_std":      220.0,
    "fwd_packet_length_mean": 432.0,
    "fwd_packet_length_max":  1460.0,
    "fwd_packet_length_min":  52.0,
    "fwd_packet_length_std":  280.0,
    "bwd_packet_length_mean": 200.0,
    "bwd_packet_length_max":  800.0,
    "bwd_packet_length_min":  52.0,
    "bwd_packet_length_std":  150.0,
    "flow_iat_mean":          50000.0,
    "flow_iat_max":           200000.0,
    "flow_iat_min":           1000.0,
    "flow_iat_std":           30000.0,
    "packets_per_second":     11.4,
    "bytes_per_second":       3800.0,
    "syn_flag_count":         1.0,
    "fin_flag_count":         1.0,
    "rst_flag_count":         0.0,
    "ack_flag_count":         6.0,
    "psh_flag_count":         2.0,
    "urg_flag_count":         0.0,
    "_source_ip":             0.0,
    "_destination_ip":        0.0,
    "_src_port":              12345.0,
    "_dst_port":              80.0,
}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "db"     in data
    assert "model"  in data


def test_health_db_connected():
    response = client.get("/health")
    data = response.json()
    # DB should be 'ok' since we're using SQLite
    assert data["db"] == "ok"


def test_get_alerts_returns_list():
    response = client.get("/api/alerts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_alerts_filter_by_type():
    response = client.get("/api/alerts?type=DDoS")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_alerts_filter_by_severity():
    response = client.get("/api/alerts?severity=HIGH")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_alerts_pagination():
    response = client.get("/api/alerts?limit=5&offset=0")
    assert response.status_code == 200
    assert len(response.json()) <= 5


def test_get_stats_returns_required_keys():
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    required = [
        "total_flows", "total_attacks", "benign_count",
        "attacks_by_type", "attacks_by_severity", "uptime_seconds"
    ]
    for key in required:
        assert key in data, f"Missing key: {key}"


def test_get_stats_counts_are_non_negative():
    response = client.get("/api/stats")
    data = response.json()
    assert data["total_flows"]   >= 0
    assert data["total_attacks"] >= 0
    assert data["benign_count"]  >= 0


def test_ip_leaderboard_returns_list():
    response = client.get("/api/ip-leaderboard")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_predict_returns_503_without_model():
    """
    If no model.pkl exists, /api/predict should return 503 Not Found,
    not a 500 crash.
    """
    response = client.post("/api/predict", json=SAMPLE_FEATURES)
    # Either 200 (model loaded) or 503 (model not trained yet) — not 500
    assert response.status_code in (200, 503)


def test_predict_response_has_required_keys():
    """Only run this if model.pkl exists."""
    response = client.post("/api/predict", json=SAMPLE_FEATURES)
    if response.status_code == 503:
        pytest.skip("model.pkl not found — run train.py first")
    assert response.status_code == 200
    data = response.json()
    for key in ["alert_id", "prediction", "confidence", "severity", "timestamp"]:
        assert key in data, f"Missing key: {key}"


def test_predict_confidence_is_valid_float():
    response = client.post("/api/predict", json=SAMPLE_FEATURES)
    if response.status_code == 503:
        pytest.skip("model.pkl not found")
    data = response.json()
    conf = data["confidence"]
    assert 0.0 <= conf <= 1.0


def test_predict_severity_is_valid():
    response = client.post("/api/predict", json=SAMPLE_FEATURES)
    if response.status_code == 503:
        pytest.skip("model.pkl not found")
    data = response.json()
    assert data["severity"] in ("NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_root_redirects():
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()


# ── Cleanup ───────────────────────────────────────────────────────────────────
def teardown_module(module):
    """Remove SQLite test file after tests complete."""
    import os
    try:
        os.remove("test_nids.db")
    except FileNotFoundError:
        pass