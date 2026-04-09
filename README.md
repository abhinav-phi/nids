<div align="center">

# 🛡️ The Sentinel — Network Intrusion Detection System

**A production-grade, ML-powered NIDS with real-time packet capture, SHAP explainability, and a live threat intelligence dashboard.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript)](https://typescriptlang.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikitlearn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-blue)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📌 What is This?

**The Sentinel** is a complete, end-to-end Network Intrusion Detection System that:

- **Captures live network packets** using Scapy and assembles them into bidirectional flows
- **Extracts 52 CICIDS2017-compatible features** per flow (IATs, packet lengths, TCP flags, etc.)
- **Classifies flows** using an ML ensemble (9 models trained, best saved automatically)
- **Explains every prediction** using SHAP values — top 5 most influential features per alert
- **Streams alerts in real time** to a React dashboard via WebSocket
- **Simulates attack traffic** for demo and testing without a real adversary

Built for hackathons, research, and production prototyping. No black box — every prediction is explainable.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         LIVE NETWORK                             │
│                   (Wi-Fi / Ethernet traffic)                     │
└──────────────────────┬───────────────────────────────────────────┘
                       │  Raw packets
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    NetworkSniffer  (Scapy)                       │
│  • Captures IP/TCP/UDP/ICMP packets on auto-detected interface   │
│  • Groups packets into flows via 5-tuple key                     │
│    (src_ip, dst_ip, src_port, dst_port, protocol)                │
│  • Closes flows on TCP FIN/RST or after 30s timeout              │
└──────────────────────┬───────────────────────────────────────────┘
                       │  Flow packet dicts
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                 FlowExtractor  (CICIDS2017)                      │
│  • Computes 52 exact CICIDS2017 features per flow                │
│  • Statistical: packet length mean/std/min/max                   │
│  • Temporal: IATs, flow duration, active/idle periods            │
│  • Protocol: TCP flags (FIN/PSH/ACK), window sizes               │
│  • Rates: Flow Bytes/s, Flow Packets/s, Fwd/Bwd Packets/s        │
└──────────────────────┬───────────────────────────────────────────┘
                       │  Feature dict (52 keys)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                FastAPI Backend  (port 8000)                      │
│                                                                  │
│  POST /api/predict ──► StandardScaler ──► ML Model              │
│                                     ├──► SHAP TreeExplainer      │
│                                     └──► SQLite / PostgreSQL DB  │
│                                                                  │
│  GET  /api/stats          — Total flows, attack counts by type   │
│  GET  /api/alerts         — Paginated alert history              │
│  GET  /api/ip-leaderboard — Top attacker IPs                     │
│  WS   /ws/live            — Real-time alert stream               │
│  POST /api/sniffer/start  — Start packet capture                 │
│  POST /api/sniffer/stop   — Stop packet capture                  │
│  GET  /health             — System health (DB, model, sniffer)   │
└──────────────────────┬───────────────────────────────────────────┘
                       │  WebSocket / REST
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              React Dashboard  (port 5173)                        │
│                                                                  │
│  • KPI Cards — total flows, attacks, detection rate, uptime      │
│  • Live Traffic Chart — alerts per minute over time              │
│  • Attack Pie Chart — distribution by attack type                │
│  • Alert Feed — scrollable list of real-time alerts              │
│  • IP Leaderboard — top source IPs by attack count               │
│  • Attack Timeline — temporal heat map of attack events          │
│  • SHAP Explainer — per-alert feature importance bars            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🧠 ML Pipeline

### Dataset
The model is trained on the **CICIDS2017** dataset — a widely used benchmark containing normal traffic and 14 attack categories including DDoS, DoS, PortScan, Brute Force, Bot, and Web Attacks.

### Training Pipeline (`src/model/train.py`)

| Step | Description |
|------|-------------|
| **A. Load** | Reads all CSVs from `data/raw/` (up to 50k rows each) |
| **B. Clean** | Removes NaN, Inf, and duplicate rows |
| **C. Feature Engineering** | Adds 7 domain-specific ratio features (flow bytes/packet, fwd/bwd ratios, IAT jitter, etc.) |
| **D. Split** | Separates features and label column (`Attack Type`) |
| **E. Encode** | `LabelEncoder` → numeric class indices, saved as `label_encoder.pkl` |
| **F. Stratified Split** | 80/20 train/test with `stratify=y` |
| **G. SMOTE** | Oversamples minority classes on training data only |
| **H. Dual Scalers** | Fits `StandardScaler` **and** `RobustScaler`; best scaler saved as `scaler.pkl` |
| **I. PCA Analysis** | Experimental comparison at 90/95/99% variance (not used in production) |
| **J. Train 9 Models** | See table below |
| **K. Cross-Validation** | 5-fold stratified CV on top 2 models |
| **L. Compare** | Sorted leaderboard by Macro F1 |
| **M. Save Best** | Best model saved as `model.pkl` |

### Model Suite (9 Classifiers)

| # | Model | Search Strategy | Notes |
|---|-------|----------------|-------|
| 1 | Logistic Regression | Fixed params | Baseline |
| 2 | Decision Tree | GridSearchCV | `max_depth`, `criterion` |
| 3 | Random Forest | RandomizedSearchCV | `n_estimators`, `max_features` |
| 4 | XGBoost | RandomizedSearchCV | `learning_rate`, `subsample` |
| 5 | LightGBM | Fixed params | Optional (graceful fallback) |
| 6 | SVM (RBF kernel) | Fixed params | 15k sample (O(n²) cap) |
| 7 | **Neural Network (MLP)** | Fixed params | `256→128→64` ReLU, Adam, early stopping |
| 8 | Voting Ensemble | Soft voting | RF + XGB + LGBM/MLP |
| 9 | Stacking Ensemble | LR meta-learner | RF + XGB + LGBM/MLP → LR |

Both **StandardScaler** and **RobustScaler** are compared for tree models. RobustScaler handles DDoS-induced outliers (e.g. 10⁶ pkt/s) better because it uses median and IQR instead of mean/std.

### Neural Network Architecture
```
Input (52 features)
       │
   Dense(256, relu)
       │
   Dense(128, relu)
       │
    Dense(64, relu)
       │
   Dense(n_classes, softmax)

Optimizer: Adam (lr=1e-3, adaptive)
L2 Reg:    α = 1e-4
Batch:     512
Early Stop: 15 no-improve epochs on 10% validation split
```

### SHAP Explainability
A `TreeExplainer` is cached once at server startup. For every non-benign prediction, the top 5 features by absolute SHAP value are returned alongside the prediction. The React `SHAPExplainer` component visualizes these as a horizontal bar chart per alert.

---

## 🌐 Feature Engineering

The `FlowExtractor` (`src/features/extractor.py`) converts raw Scapy packet dicts into the exact **52-feature CICIDS2017 vector** expected by the model.

### Feature Categories

| Category | Features |
|----------|---------|
| **Basic** | Destination Port, Flow Duration, Total Fwd/Bwd Packets |
| **Packet Length** | Fwd/Bwd Packet Length (Max, Min, Mean, Std), Min/Max/Mean/Std/Variance overall |
| **Flow Rates** | Flow Bytes/s, Flow Packets/s, Fwd Packets/s, Bwd Packets/s |
| **IAT (Inter-Arrival Time)** | Flow IAT (Mean/Std/Max/Min), Fwd IAT (Total/Mean/Std/Max/Min), Bwd IAT (same) |
| **Headers** | Fwd Header Length, Bwd Header Length |
| **TCP Flags** | FIN Flag Count, PSH Flag Count, ACK Flag Count |
| **Window / Segment** | Init_Win_bytes_forward, Init_Win_bytes_backward, min_seg_size_forward |
| **Subflow** | Subflow Fwd Bytes, act_data_pkt_fwd, Average Packet Size |
| **Active / Idle** | Active Mean/Max/Min, Idle Mean/Max/Min |

IATs are computed in **microseconds** (matching CICIDS2017 scale). Active/Idle periods are classified by a 5-second inter-packet gap threshold.

---

## 🔄 Real-Time Data Flow

```
1. NetworkSniffer captures packet on interface 'eth0' / 'Wi-Fi'
         ↓
2. Packet parsed → 12-field dict (src_ip, dst_ip, ports, protocol,
   size, payload_len, header_len, time, tcp_flags, window_size, ttl)
         ↓
3. Packet added to flow bucket (keyed by 5-tuple)
   Flow closes on: TCP FIN/RST | 30s timeout | 500-packet cap
         ↓
4. FlowExtractor.extract_from_dicts() → {52 CICIDS features}
         ↓
5. POST http://localhost:8000/api/predict (non-blocking thread)
         ↓
6. Backend:
   a. Strips metadata (_source_ip, _dst_port, ...)
   b. Scales 52 features with StandardScaler
   c. model.predict() → class index
   d. model.predict_proba() → confidence score
   e. LabelEncoder.inverse_transform() → human-readable label
   f. SHAP TreeExplainer → top-5 feature importances
   g. Severity mapping (CRITICAL/HIGH/MEDIUM/LOW/NONE)
   h. Saves Alert to database
   i. If not BENIGN → broadcast via WebSocket
         ↓
7. Dashboard WebSocket client receives alert JSON
   → Updates KPICards, AlertFeed, TrafficChart, PieChart in real-time
```

---

## 🖥️ Frontend Dashboard

Built with **React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui + Recharts**.

| Component | Description |
|-----------|-------------|
| `StatusBar` | WS connection indicator, live clock, system status |
| `KPICards` | Total flows, attacks detected, detection rate, uptime |
| `TrafficChart` | Recharts `LineChart` — attacks per minute, last 30 points |
| `AttackPieChart` | Recharts `PieChart` — attack type distribution |
| `AlertFeed` | Real-time scrollable alert list with severity color coding |
| `IPLeaderboard` | Top 10 most aggressive source IPs |
| `AttackTimeline` | Temporal bar chart of attack events over time |
| `SHAPExplainer` | Per-alert SHAP feature importance bar chart |
| `Sidebar` | Navigation rail with system overview |

### WebSocket Hook (`useWebSocket.ts`)
- Connects to `ws://localhost:8000/ws/live`
- On connect: receives batch of last 50 alerts for history seeding
- Auto-reconnects on disconnect (3s delay)
- Normalizes both new (`value`) and legacy (`impact`) SHAP field names

---

## 🔌 REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Submit a 52-feature flow dict for classification |
| `GET` | `/api/alerts` | Paginated alert history |
| `GET` | `/api/stats` | Total flows, attacks by type/severity, uptime |
| `GET` | `/api/ip-leaderboard` | Top N attacker IPs |
| `POST` | `/api/sniffer/start` | Start live packet capture |
| `POST` | `/api/sniffer/stop` | Stop live packet capture |
| `GET` | `/api/sniffer/stats` | Sniffer stats (packets, flows, alerts) |
| `GET` | `/health` | System health (DB, model, sniffer, WS clients) |
| `WS` | `/ws/live` | Real-time alert stream (WebSocket) |

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Backend & ML |
| Node.js | 18+ | Frontend |
| [Npcap](https://npcap.com) | Latest | Packet capture on Windows |
| Git | Any | Clone repo |

> **Windows note:** Install Npcap with "WinPcap API-compatible mode" enabled. Run the backend as Administrator for packet capture.

---

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nids.git
cd nids
```

---

### 2. Backend Setup

```bash
cd nids-backend

# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

**Optional — PostgreSQL:**
Create a `.env` file if you want to use PostgreSQL instead of the default SQLite:
```env
DATABASE_URL=postgresql://nids:password@localhost:5432/nids_db
```

---

### 3. Train the ML Model

Download the CICIDS2017 dataset CSVs and place them in `nids-backend/data/raw/`.

> The dataset can be obtained from [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html). Place CSV files directly in `data/raw/`.

```bash
# From inside nids-backend/
python src/model/train.py
```

This will:
1. Load and clean the CSV data
2. Engineer 7 additional ratio features
3. Balance classes with SMOTE
4. Train and compare 9 ML models (takes ~10–30 minutes depending on hardware)
5. Save the best model, scaler, and label encoder:
   - `nids-backend/model.pkl`
   - `nids-backend/scaler.pkl`
   - `nids-backend/label_encoder.pkl`

> **Shortcut:** If you have pre-trained artifacts, place them in `nids-backend/` and skip this step.

---

### 4. Start the Backend Server

```bash
# From inside nids-backend/
# Standard mode (no auto-start of packet capture):
uvicorn src.api.main:app --reload --port 8000

# With automatic packet capture on startup:
set NIDS_CAPTURE=1   # Windows
uvicorn src.api.main:app --port 8000
```

Verify it's running:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "db": "ok",
  "model": "ok",
  "sniffer": "stopped",
  "uptime_seconds": 3.2,
  "ws_clients": 0
}
```

---

### 5. Start the Frontend

```bash
cd nids-frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

### 6. Start Packet Capture (Live Mode)

**Option A — Via the API:**
```bash
curl -X POST http://localhost:8000/api/sniffer/start
```

**Option B — Environment variable** (set `NIDS_CAPTURE=1` before starting the server, see Step 4).

**Option C — Standalone sniffer:**
```bash
# From inside nids-backend/
python src/capture/sniffer.py --interface auto
```

---

## 🎭 Attack Simulation (Demo Mode)

No real adversary? Use the built-in simulators to generate attack traffic for demonstration.

```bash
cd nids-backend

# Simulate a DDoS UDP flood (300 packets, ~1000 pps)
python src/simulation/sim_ddos.py --target 127.0.0.1 --count 300

# Simulate a port scan
python src/simulation/sim_portscan.py --target 127.0.0.1

# Simulate a brute force attack
python src/simulation/sim_bruteforce.py --target 127.0.0.1

# Simulate a mixed attack scenario
python src/simulation/sim_mixed.py
```

The sniffer will pick up the generated packets, extract features, and send them to the prediction API — alerts will appear on the dashboard in real time.

---

## 🧪 Running Tests

```bash
cd nids-backend

# Feature extraction sanity check
python test_pipeline.py

# API integration tests (requires server running)
python test_api.py

# ML prediction test
python check.py

# Run the full pytest suite
pytest tests/
```

---

## 📁 Project Structure

```
nids/
├── nids-backend/
│   ├── src/
│   │   ├── api/
│   │   │   ├── main.py          # FastAPI app, WebSocket, sniffer control
│   │   │   ├── database.py      # SQLAlchemy engine + session factory
│   │   │   ├── models.py        # Alert ORM model
│   │   │   ├── schemas.py       # Pydantic request/response schemas
│   │   │   └── routes/
│   │   │       ├── predict.py   # POST /api/predict — ML inference endpoint
│   │   │       ├── alerts.py    # GET /api/alerts — alert history
│   │   │       └── stats.py     # GET /api/stats, /api/ip-leaderboard
│   │   ├── capture/
│   │   │   └── sniffer.py       # Live packet capture, flow assembly
│   │   ├── features/
│   │   │   └── extractor.py     # 52-feature CICIDS2017 extractor
│   │   ├── model/
│   │   │   ├── train.py         # Full ML training pipeline (9 models)
│   │   │   ├── predict.py       # Inference wrapper + SHAP
│   │   │   └── evaluate.py      # Model evaluation utilities
│   │   └── simulation/
│   │       ├── sim_ddos.py      # UDP DDoS flood simulator
│   │       ├── sim_portscan.py  # Port scan simulator
│   │       ├── sim_bruteforce.py# Brute force simulator
│   │       └── sim_mixed.py     # Mixed attack scenario
│   ├── notebooks/
│   │   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   │   ├── 02_training.ipynb    # Training walkthrough
│   │   └── 02_training_with_nn.ipynb # Neural Network training
│   ├── model.pkl                # ← Trained model (generated by train.py)
│   ├── scaler.pkl               # ← StandardScaler (generated by train.py)
│   ├── label_encoder.pkl        # ← LabelEncoder (generated by train.py)
│   └── requirements.txt
│
└── nids-frontend/
    └── src/
        ├── pages/
        │   └── Index.tsx        # Main dashboard page layout
        ├── components/
        │   ├── StatusBar.tsx    # Connection & system status bar
        │   ├── KPICards.tsx     # Key performance indicator cards
        │   ├── TrafficChart.tsx # Live traffic line chart
        │   ├── AttackPieChart.tsx # Attack type pie chart
        │   ├── AlertFeed.tsx    # Real-time alert list
        │   ├── IPLeaderboard.tsx# Top attacker IP table
        │   ├── AttackTimeline.tsx # Temporal attack timeline
        │   ├── SHAPExplainer.tsx # SHAP feature importance bars
        │   └── Sidebar.tsx      # Navigation sidebar
        └── hooks/
            └── useWebSocket.ts  # WebSocket connection + alert normalization
```

---

## 🛠️ Tech Stack

### Backend
| Technology | Version | Role |
|------------|---------|------|
| **Python** | 3.10+ | Core language |
| **FastAPI** | 0.110 | REST API + WebSocket server |
| **Uvicorn** | 0.29 | ASGI web server |
| **SQLAlchemy** | 2.0 | ORM + database layer |
| **scikit-learn** | 1.4 | ML models, scalers, SMOTE |
| **XGBoost** | 2.0 | Gradient-boosted tree classifier |
| **LightGBM** | 4.x | Optional fast gradient boosting |
| **SHAP** | 0.45 | Model explainability |
| **Scapy** | 2.5 | Live packet capture |
| **imbalanced-learn** | 0.12 | SMOTE class balancing |
| **pandas / numpy** | 2.2 / 1.26 | Data processing |
| **SQLite / PostgreSQL** | — | Alert persistence |

### Frontend
| Technology | Version | Role |
|------------|---------|------|
| **React** | 18.3 | UI framework |
| **TypeScript** | 5.8 | Type-safe JavaScript |
| **Vite** | 7 | Build tool + dev server |
| **Tailwind CSS** | 3.4 | Utility-first styling |
| **shadcn/ui** | — | Accessible component primitives |
| **Recharts** | 2.15 | Charts (line, pie, bar) |
| **TanStack Query** | 5 | Server state management |
| **React Router** | 6 | Client-side routing |
| **Lucide React** | — | Icon library |

---

## ⚙️ Configuration

### Backend Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./nids.db` | Database connection string |
| `NIDS_CAPTURE` | `0` | Set to `1` to auto-start sniffer on startup |

### Key Constants (`sniffer.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `FLOW_TIMEOUT_SECONDS` | `30` | Close inactive flows after N seconds |
| `MAX_PACKETS_PER_FLOW` | `500` | Safety cap per flow before early processing |
| `ACTIVE_TIMEOUT` | `5.0s` | IAT threshold for active/idle classification |

---

## 🔍 Severity Classification

| Level | Attack Types |
|-------|-------------|
| 🔴 **CRITICAL** | DDoS, DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS SlowHTTPTest, Heartbleed |
| 🟠 **HIGH** | Bot, FTP-Patator, SSH-Patator, Infiltration |
| 🟡 **MEDIUM** | PortScan, Web Attack (Brute Force, XSS, SQL Injection) |
| 🟢 **LOW** | Brute Force (generic), unknown attack types |
| ⚪ **NONE** | BENIGN (normal traffic) |

---

## 📊 Model Performance (Typical on CICIDS2017)

> Exact metrics will vary depending on the CSV files and sample sizes used. Run `python src/model/train.py` to reproduce.

| Model | Accuracy | Macro F1 | Notes |
|-------|---------|---------|-------|
| XGBoost | ~99%+ | ~97%+ | Usually best single model |
| Random Forest | ~99% | ~96% | Very close to XGBoost |
| Voting Ensemble | ~99%+ | ~97%+ | RF + XGB + LGBM |
| Neural Network | ~98% | ~93% | 256→128→64 MLP |
| Decision Tree | ~98% | ~90% | After GridSearchCV |
| Logistic Regression | ~95% | ~75% | Baseline |
| SVM (RBF) | ~97% | ~85% | 15k subset only |

---

## 🧩 Hackathon Demo Flow

1. **Start backend** — `uvicorn src.api.main:app --reload --port 8000`
2. **Start frontend** — `npm run dev` in `nids-frontend/`
3. **Open dashboard** — [http://localhost:5173](http://localhost:5173)
4. **Start sniffer** — `curl -X POST http://localhost:8000/api/sniffer/start`
5. **Trigger attacks** — run any simulator from `src/simulation/`
6. **Watch the dashboard** — alerts flood in, KPIs update, SHAP bars explain each detection

---

## 🛡️ Security Notes

- The sniffer requires **administrator / root privileges** to capture raw packets
- On Windows, **Npcap** must be installed (download from [https://npcap.com](https://npcap.com))
- CORS is configured for localhost dev servers; update `allow_origins` for production
- The simulation scripts send real packets on your network — use `127.0.0.1` as target in demos

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ for network security research and hackathon competition.

**The Sentinel** — _See every packet. Understand every threat._

</div>