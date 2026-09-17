"""
smoke_test.py — Local end-to-end verification of 03_Inference_API_Colab.ipynb.

Executes the notebook's embedded code (inference stack, FlowExtractor, synthetic
flows, FastAPI app, uvicorn server) in a controlled namespace with fake model
artifacts, then hits the live API: health, predictions, and all validation
guards. Run after editing the _build sources:

    python smoke_test.py
"""
import json
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB3 = HERE.parent / "03_Inference_API_Colab.ipynb"

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── 1. Fake artifacts with the production contract ───────────────────────
CLASSES = ["Bots", "Brute Force", "DDoS", "DoS", "Normal Traffic",
           "Port Scanning", "Web Attacks"]
rng = np.random.RandomState(0)
X_fake = rng.rand(700, 52).astype(np.float64)
y_fake = rng.randint(0, 7, 700)

encoder = LabelEncoder().fit(CLASSES)
scaler = StandardScaler().fit(X_fake)
model = DecisionTreeClassifier(max_depth=6, random_state=0).fit(
    scaler.transform(X_fake), y_fake)

ART = Path(tempfile.mkdtemp(prefix="nids_smoke_artifacts_"))
joblib.dump(model, ART / "model.pkl")
joblib.dump(scaler, ART / "scaler.pkl")
joblib.dump(encoder, ART / "label_encoder.pkl")
(ART / "manifest.json").write_text(json.dumps({"model_type": "DecisionTreeClassifier",
                                               "feature_count": 52}))
# will populate feature_names.json + X_test.npy + y_test.npy after inference cell loads CICIDS_FEATURES
print(f"fake artifacts → {ART}")

# ── 2. Extract notebook cells ─────────────────────────────────────────────
nb = json.loads(NB3.read_text(encoding="utf-8"))
code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

def cell(marker):
    hits = [s for s in code_cells if marker in s]
    assert hits, f"cell with marker {marker!r} not found"
    return hits[0]

ns = {"__name__": "smoke"}

# Set matplotlib to Agg backend BEFORE any cell runs — prevents plt.show()
# from hanging in headless / sandbox environments.
import matplotlib as _mpl
_mpl.use("Agg")

# discovery cell is Colab-specific — provide the globals it would have set
ns["ART_DIR"] = ART
ns["DATA_PATH"] = None
ns["API_PORT"] = 8077
ns["API_BASE"] = "http://127.0.0.1:8077"
ns["DRIVE_ART_DIR"] = "nids_artifacts"
ns["DRIVE_DATA_DIR"] = "nids_data"
ns["DATA_FILENAME"] = "cicids2017_cleaned.csv"
ns["Path"] = Path
ns["json"] = json
ns["shutil"] = __import__("shutil")
ns["zipfile"] = __import__("zipfile")
ns["display"] = print   # IPython builtin in real notebooks

def run(src, label):
    src = src.replace("sqlite:////content/nids_colab.db",
                      f"sqlite:///{(ART / 'nids_smoke.db').as_posix()}")
    try:
        exec(compile(src, label, "exec"), ns)
        print(f"✔ executed: {label}")
    except Exception:
        print(f"✖ FAILED: {label}")
        traceback.print_exc()
        sys.exit(1)

run(cell("CICIDS_FEATURES = ["), "inference stack")

# populate artifacts the SHAP deep-dive cell expects
from pathlib import Path as _P
(ART / "feature_names.json").write_text(json.dumps(list(ns["CICIDS_FEATURES"])))
np.save(ART / "X_test.npy", X_fake.astype(np.float32))
np.save(ART / "y_test.npy", y_fake)
print("extra artifacts written: feature_names.json, X_test.npy, y_test.npy")

run(cell("class FlowExtractor"), "flow extractor")
run(cell("def _packet("), "synthetic flows")
run(cell("import matplotlib.pyplot as plt"), "SHAP deep dive")
run(cell("EXPECTED_FEATURES      = list(CICIDS_FEATURES)"), "validation helpers")
run(cell("class ConnectionManager"), "FastAPI app")
run(cell("uvicorn.Config"), "uvicorn server thread")

# ── 3. Hit the live API ───────────────────────────────────────────────────
import requests as rq
BASE = ns["API_BASE"]
failures = []

def check(name, cond, detail=""):
    status = "✔" if cond else "✖"
    print(f"  {status} {name} {detail}")
    if not cond:
        failures.append(name)

health = rq.get(f"{BASE}/health", timeout=5).json()
check("GET /health", health.get("status") == "ok" and health.get("model") == "ok", str(health))

f_norm = ns["f_norm"]; f_ddos = ns["f_ddos"]; f_bf = ns["f_bf"]; f_extra = ns.get("f_ddos")

def post(payload):
    return rq.post(f"{BASE}/api/predict", json=payload, timeout=20)

r = post({**f_norm, "_source_ip": "192.168.1.50", "_destination_ip": "93.184.216.34"})
j = r.json()
check("POST predict (normal flow)", r.status_code == 200
      and {"prediction", "confidence", "severity", "shap_top5", "alert_id"} <= set(j), f"HTTP {r.status_code}")
r = post({**f_ddos, "_source_ip": "10.0.0.66"})
check("POST predict (DDoS flow)", r.status_code == 200 and "prediction" in r.json(), f"HTTP {r.status_code}")

r = post({k: v for k, v in list(f_norm.items())[:40]})
check("422 when >8 features missing", r.status_code == 422, f"HTTP {r.status_code}")
r = post({"_source_ip": "1.2.3.4"})
check("400 when no features at all", r.status_code == 400, f"HTTP {r.status_code}")
bad = dict(f_norm); bad["Flow Duration"] = -5.0
r = post(bad)
check("422 on negative feature", r.status_code == 422, f"HTTP {r.status_code}")
bad = {**f_norm, "_source_ip": "not-an-ip"}
r = post(bad)
check("422 on invalid metadata IP", r.status_code == 422, f"HTTP {r.status_code}")
bad = dict(f_norm); bad["Flow Bytes/s"] = float("nan")
# requests' json= refuses NaN client-side — send the raw JSON `NaN` token instead,
# mirroring what a hostile client would do (server must still reject it).
r = rq.post(f"{BASE}/api/predict", data=json.dumps(bad),
            headers={"Content-Type": "application/json"}, timeout=20)
check("422/400 on NaN feature value", r.status_code in (400, 422), f"HTTP {r.status_code}")

alerts = rq.get(f"{BASE}/api/alerts", params={"limit": 10, "exclude_benign": "false"}, timeout=5).json()
check("GET /api/alerts returns persisted rows",
      isinstance(alerts, list) and len(alerts) == 2 and all("prediction" in a for a in alerts),
      f"{len(alerts)} rows")
stats = rq.get(f"{BASE}/api/stats", timeout=5).json()
check("GET /api/stats counts flows", stats.get("total_flows", 0) == 2, str(stats.get("total_flows")))
board = rq.get(f"{BASE}/api/ip-leaderboard", timeout=5).json()
check("GET /api/ip-leaderboard", isinstance(board, list) and len(board) >= 1,
      f"top={board[0]['source_ip'] if board else '—'}")

# severity mapping spot-checks (src/model/predict.py contract)
get_severity, is_benign = ns["get_severity"], ns["is_benign"]
check("severity: DDoS→CRITICAL", get_severity("DDoS") == "CRITICAL")
check("severity: Normal Traffic→NONE", get_severity("Normal Traffic") == "NONE")
check("severity: Brute Force→LOW", get_severity("Brute Force") == "LOW")
check("severity: Bot→HIGH", get_severity("Bots") == "HIGH")
check("is_benign covers both spellings", is_benign("Normal Traffic") and not is_benign("DDoS"))

print()
if failures:
    print(f"SMOKE TEST FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("SMOKE TEST PASSED — inference, extraction, API, persistence and "
      "validation guards all work.")
